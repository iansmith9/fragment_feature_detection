"""NMF fitting and hyperparameter tuning for scan windows.

A scan window is a dense matrix ``m`` of shape ``(n_scans, n_mz_bins)``: rows are
consecutive MS2 scans of one isolation window (GPF), columns are discretized fragment
m/z bins, values are (max-scaled) intensities. NMF approximates ``m ≈ W @ H`` with

- ``W`` of shape ``(n_scans, n_components)``: one elution profile per component, and
- ``H`` of shape ``(n_components, n_mz_bins)``: one fragment spectrum per component
  (called "weights" elsewhere in the package, e.g. ``ms2_component_weight_*``).

This module contains:

- ``fit_nmf_matrix_custom_init``: the single NMF fit used by the pipeline and tuning.
- Cross-validation splitters (``MzBinMaskingSplitter`` hides random cells,
  ``MzBinSampleSplitter`` hides whole scans) and ``NMFMaskWrapper``, an sklearn-style
  estimator that fits NMF on many windows per split and records error and
  shape metrics (sparsity, orthogonality, coverage).
- Two searches over the regularization (``alpha_W``, ``alpha_H``, ``l1_ratio``):
  ``RandomizedSearchReconstructionCV`` (sklearn ``ParameterSampler``) and
  ``OptunaSearchReconstructionCV`` (multi-objective Optuna, workers sharing a journal
  file), plus ``pick_parameters_harmonic_mean`` to choose one setting from the
  multi-objective results.
- ``tune_hyperparameters_*``: entry points used by the CLI, which also redirect
  logging to a tuning-specific log file for the duration of the search.
"""

import base64
import logging
import numbers
import pickle
import shutil
import tempfile
import time
import warnings
from collections import defaultdict
from datetime import datetime
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import Manager, Queue
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
from sklearn.base import BaseEstimator, clone
from sklearn.decomposition import NMF
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.utils._param_validation import Interval
from sklearn.utils.parallel import Parallel, delayed

from fragment_feature_detection.config import Config, format_logger
from fragment_feature_detection.utils import calculate_nmf_summary

logger = logging.getLogger(__name__)

# Per-fit metric lists recorded on NMFMaskWrapper during fit(). _fit_and_score copies
# them into each fold's test/train score dict (with the leading underscore stripped and
# the sign flipped), so they become columns like "mean_test_weight_orthogonality" in
# cv_results and can be used as tuning objectives.
_FIT_AND_SCORE = [
    "_test_reconstruction_errors",
    "_weight_orthogonality",
    "_sample_orthogonality",
    "_nonzero_component_fraction",
    "_neglog_ratio_train_test_reconstruction_error",
    "_mean_weight_sparsity",
    "_mean_sample_sparsity",
    "_fraction_window_component",
]


def _set_queue_handler(logger: logging.Logger, log_queue: Queue) -> None:
    """Point a logger at log_queue, replacing QueueHandlers left by earlier runs.

    joblib reuses worker processes, so adding a handler on every call duplicates log
    lines and can leave handlers pointing at queues from finished runs.
    """
    for handler in [h for h in logger.handlers if isinstance(h, QueueHandler)]:
        logger.removeHandler(handler)
    logger.addHandler(QueueHandler(log_queue))


def _journal_storage(journal_path: str) -> optuna.storages.JournalStorage:
    """Optuna storage backed by a journal file, usable from several processes.

    A journal file is safe for several processes on one machine (sqlite gets "database
    is locked" under load), and the open-based lock works on Windows, where the default
    symlink lock needs extra privileges.
    """
    return optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            journal_path,
            lock_obj=optuna.storages.journal.JournalFileOpenLock(journal_path),
        )
    )


class OptimizationParameters:
    """Target values for the tuning objectives, and the scoring against them.

    Several tuning objectives are not "bigger is better" but "close to a target is
    better" (e.g. the fraction of components that are non-zero should be about
    components_in_window / n_components). This class holds those targets and turns a
    metric into a score that is maximal (0) when the metric equals its target.

    Objectives not listed in ``_scores`` (``test_reconstruction_errors``,
    ``neglog_ratio_train_test_reconstruction_error``) are passed through unchanged.

    Attributes:
        _components_in_window (float): Target number of components per window
        _components (float): Total number of components
        _scan_width (float): Width of scan window
        _component_sigma (float): Component sigma value
        _error (str): Error metric type ('l1' or 'l2')
        _scores (dict): Target scores for different optimization metrics
    """

    _components_in_window = 8.0
    _components = 20.0
    _scan_width = 150.0
    _component_sigma = 3.0
    _error = "l1"

    def __init__(self, error: Literal["l1", "l2"] = "l1"):
        """Create targets from the class defaults.

        Args:
            error (Literal["l1", "l2"]): Distance used to compare a metric with its
                target: absolute ("l1") or squared ("l2") difference.
        """
        self._error = error
        self.set_params()

    def set_params(self) -> None:
        """Sets target scores for different optimization metrics.

        Updates the internal _scores dictionary with target values for various
        optimization metrics based on the current parameter settings.

        Targets are stored with the sign used by the scores, which are the negated
        metrics (see _fit_and_score), so ``score`` computes ``-|target - metric|``:

        - weight/sample orthogonality: 0, i.e. no deviation from an identity Gram
          matrix (components share no fragments / don't co-elute).
        - nonzero_component_fraction: components_in_window / n_components, the
          expected share of the n_components that end up non-zero.
        - mean weight/sample sparsity: Hoyer sparsity of 1 (fully sparse). The
          summary already negates sparsity, so the score holds +sparsity and the
          stored target is -1.
        - fraction_window_component: fraction of scans covered by at least one
          component if components_in_window peaks each span about 4 sigma scans
          (±2 sigma, ~95% of a Gaussian) without overlapping.
        """
        self._scores = {
            "weight_orthogonality": 0.0,
            "sample_orthogonality": 0.0,
            "nonzero_component_fraction": self._components_in_window / self._components,
            "mean_weight_sparsity": -1.0,
            "mean_sample_sparsity": -1.0,
            "fraction_window_component": (
                self._components_in_window * 4 * self._component_sigma
            )
            / self._scan_width,
            # 'fraction_window_component': 0.0,
        }

    def score(
        self, param: str, value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Calculates optimization score for a parameter value.

        Args:
            param (str): Name of parameter to score
            value (Union[float, np.ndarray]): Parameter value(s) to score

        Returns:
            Union[float, np.ndarray]: Calculated score based on error metric
        """
        # value is the negated metric, so value + target == target - metric. The score
        # is minus its distance from zero: 0 at the target, more negative further away.
        if param in self._scores.keys():
            if self._error == "l1":
                return -1.0 * np.abs(value + self._scores[param])
            elif self._error == "l2":
                return -1.0 * (value + self._scores[param]) ** 2
        return value

    @classmethod
    def from_config(cls, config: Config = Config()) -> "OptimizationParameters":
        """Creates OptimizationParameters instance from config object.

        Args:
            config (Config): Configuration object containing optimization parameters

        Returns:
            OptimizationParameters: New instance initialized with config values
        """
        obj = cls()

        obj._components_in_window = config.tuning.components_in_window
        obj._components = config.tuning.n_components
        obj._scan_width = config.scan_filter.scan_width
        # component_sigma is scale parameter of normal distribution of component in
        # units of scans, not in RT.
        obj._component_sigma = config.tuning.component_sigma

        obj.set_params()

        return obj


class MzBinMaskingSplitter:
    """Custom splitter that randomly masks bins in scan windows for cross-validation.

    This splitter creates train/test splits by masking random mass-to-charge (m/z) bins
    in scan windows, allowing evaluation of NMF reconstruction performance.

    Args:
        n_splits (int): Number of train/test splits to generate
        mask_fraction (float): Fraction of bins to mask in each split
        random_state (int): Random seed for reproducibility
        mask_signal (bool): Whether to only mask bins containing signal
        balance_mask_signal (bool): Whether to balance masked signal vs non-signal bins

    Attributes:
        _n_splits (int): Number of splits
        _mask_fraction (float): Fraction to mask
        _random_state (int): Random seed
        _balance_mask_signal (bool): Balance masking flag
        _mask_signal (bool): Signal masking flag
    """

    def __init__(
        self,
        n_splits: int = 5,
        mask_fraction: float = 0.2,
        random_state: int = 42,
        mask_signal: bool = False,
        balance_mask_signal: bool = True,
    ):
        self._n_splits = n_splits
        self._mask_fraction = mask_fraction
        self._random_state = random_state
        self._balance_mask_signal = balance_mask_signal
        self._mask_signal = mask_signal

    def split(self, X: List[np.ndarray], y=None):
        """Generate train-test masks by randomly hiding cells of each scan window.

        Args:
            X (List[np.ndarray]): Scan window matrices, each (n_scans, n_mz_bins).
            y: Ignored, present for sklearn compatibility.

        Yields:
            Tuple[List[np.ndarray], List[np.ndarray]]: For each split, the train
                masks and the test masks, one boolean array per window with the
                window's shape. True marks a cell belonging to that set; the two masks
                are complements.
        """
        rng = np.random.default_rng(seed=self._random_state)
        for _ in range(self._n_splits):
            unmasked = []
            masked = []
            for m in X:
                if not self._mask_signal:
                    # Hide a uniform random fraction of all cells. Most cells are zero,
                    # so most test cells would be empty.
                    sub_mask = rng.random(m.shape) < self._mask_fraction
                else:
                    # Hide mask_fraction of the non-zero (signal) cells, so the test set
                    # actually measures how well signal is reconstructed.
                    flat_mask = np.zeros_like(m.flatten(), dtype=bool)
                    non_zero_idxes = np.where((m > 0).flatten())[0]
                    mask_non_zero_idxes = (
                        rng.random(non_zero_idxes.shape) < self._mask_fraction
                    )
                    flat_mask[non_zero_idxes[mask_non_zero_idxes]] = True
                    if self._balance_mask_signal:
                        # Also hide zero cells, at the overall rate masked so far
                        # (masked signal cells / all cells), so the test set includes
                        # some empty cells and over-predicting noise is penalized.
                        zero_idxes = np.where((m == 0).flatten())[0]
                        mask_zero_idxes = rng.random(zero_idxes.shape) < (
                            flat_mask.sum() / flat_mask.size
                        )
                        flat_mask[zero_idxes[mask_zero_idxes]] = True
                    sub_mask = flat_mask.reshape(m.shape)
                unmasked.append(~sub_mask)
                masked.append(sub_mask)
            yield unmasked, masked

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits (sklearn splitter interface)."""
        return self._n_splits

    @staticmethod
    def mask_train_matrix(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Masks values in training matrix.

        Held-out cells are set to 0 (in place) rather than treated as missing, so NMF
        is fit to zeros there (see SUGGESTIONS.md C1).

        Args:
            m (np.ndarray): Input matrix to mask
            mask (np.ndarray): Boolean mask array

        Returns:
            np.ndarray: Masked matrix with zeros in masked positions
        """
        m[mask] = 0.0
        return m

    @staticmethod
    def reconstruction_error(
        m: np.ndarray,
        m_reconstructed: np.ndarray,
        mask: np.ndarray,
        *args: Any,
    ) -> float:
        """Calculates reconstruction error on masked values.

        Used for both the test cells (mask = test mask) and the train cells
        (mask = train mask).

        Args:
            m (np.ndarray): Original matrix
            m_reconstructed (np.ndarray): Reconstructed matrix (W @ H)
            mask (np.ndarray): Boolean mask array
            *args: Additional arguments (unused; the model, for API parity with
                MzBinSampleSplitter)

        Returns:
            float: Mean squared error between original and reconstructed masked values
        """
        return np.nanmean((m[mask] - m_reconstructed[mask]) ** 2)


class MzBinSampleSplitter:
    """Custom splitter that samples scans in scan windows for cross-validation.

    This splitter creates train/test splits by randomly sampling entire scans
    from scan windows, allowing evaluation of NMF reconstruction performance.

    NMF is fit on the training scans only. Held-out scans are then projected onto the
    learned spectra (H fixed, a new W solved for them) and the error of that
    projection is the test error. This is a proper held-out evaluation, unlike
    zero-filling cells in MzBinMaskingSplitter.

    Args:
        n_splits (int): Number of train/test splits to generate
        mask_fraction (float): Fraction of scans to mask in each split
        random_state (int): Random seed for reproducibility
        **kwargs: Additional keyword arguments

    Attributes:
        _n_splits (int): Number of splits
        _mask_fraction (float): Fraction to mask
        _random_state (int): Random seed
    """

    def __init__(
        self,
        n_splits: int = 5,
        mask_fraction: float = 0.2,
        random_state: int = 42,
        **kwargs: Dict[str, Any],
    ):
        self._n_splits = n_splits
        self._mask_fraction = mask_fraction
        self._random_state = random_state

    def split(self, X: List[np.ndarray], y=None):
        """Generate train-test masks by randomly holding out whole scans.

        Args:
            X (List[np.ndarray]): Scan window matrices, each (n_scans, n_mz_bins).
            y: Ignored, present for sklearn compatibility.

        Yields:
            Tuple[List[np.ndarray], List[np.ndarray]]: For each split, the train and
                test row masks, one 1-D boolean array of length n_scans per window.
        """
        rng = np.random.default_rng(seed=self._random_state)
        for _ in range(self._n_splits):
            unmasked = []
            masked = []
            for m in X:
                # One draw per scan (row): about mask_fraction of scans are held out.
                sub_mask = rng.random(m.shape[0]) < self._mask_fraction
                unmasked.append(~sub_mask)
                masked.append(sub_mask)
            yield unmasked, masked

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits (sklearn splitter interface)."""
        return self._n_splits

    @staticmethod
    def mask_train_matrix(m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Drop the held-out scans from a window.

        Args:
            m (np.ndarray): Scan window matrix (n_scans, n_mz_bins).
            mask (np.ndarray): Boolean row mask, True for held-out (test) scans.

        Returns:
            np.ndarray: The training rows only, (n_train_scans, n_mz_bins).
        """
        return m[~mask]

    @staticmethod
    def reconstruction_error(
        m: np.ndarray,
        m_reconstructed: np.ndarray,
        mask: np.ndarray,
        model: NMF,
    ) -> float:
        """Mean squared error of the selected scans, projected onto the fitted spectra.

        The selected rows are projected with ``model.transform`` (solve for W with H
        fixed) and reconstructed as ``W_new @ H``. ``m_reconstructed`` is ignored,
        because it only covers the training rows.

        Args:
            m (np.ndarray): Full scan window matrix (n_scans, n_mz_bins).
            m_reconstructed (np.ndarray): Unused.
            mask (np.ndarray): Boolean row mask selecting the scans to evaluate.
            model (NMF): NMF fitted on the training scans.

        Returns:
            float: Mean squared error over the selected scans.
        """
        # A model with no non-zero spectra (or no signal in the selected scans) can't
        # be projected meaningfully; score it as reconstructing all zeros.
        if model.components_.sum() == 0.0 or m[mask].sum() == 0.0:
            return np.nanmean((m[mask] - np.zeros_like(m[mask])) ** 2)
        mt = model.transform(m[mask])
        m_reconstructed = mt @ model.components_

        return np.nanmean((m[mask] - m_reconstructed) ** 2)


class NMFMaskWrapper(BaseEstimator):
    """Wrapper for NMF that handles masked cross-validation.

    This class wraps NMF models to handle masked cross-validation splits,
    calculating various performance metrics across splits.

    ``X`` is a list of scan window matrices, not a single sample matrix. One ``fit``
    call evaluates one hyperparameter setting: for every split and every window it fits
    an NMF on the training part and records the metrics. The NMF hyperparameters live
    in ``_nmf_kwargs`` and are exposed through get_params/set_params, so sklearn's
    ``clone`` and the searches can change ``alpha_W``, ``alpha_H`` and ``l1_ratio``.

    Each metric list has one entry per (split, window) fit, in split-major order:
    index = split_number * len(X) + window_number.

    Args:
        splitter_type (str): Type of splitter to use ('Sample' or 'Mask')
        n_splits (int): Number of cross-validation splits
        mask_fraction (float): Fraction of data to mask
        mask_signal (bool): Whether to only mask signal-containing bins
        balance_mask_signal (bool): Whether to balance masked signal bins
        **nmf_kwargs: Additional keyword arguments for NMF

    Attributes:
        splitter_type (str): Type of splitter
        mask_fraction (float): Masking fraction
        n_splits (int): Number of splits
        mask_signal (bool): Signal masking flag
        balance_mask_signal (bool): Balance masking flag
        _nmf_kwargs (dict): NMF parameters
        _models (list): Fitted NMF models
        _reconstruction_errors (list): Reconstruction error scores
        _test_reconstruction_errors (list): Test set reconstruction errors
        _train_reconstruction_errors (list): Training set reconstruction errors
        _neglog_ratio_train_test_reconstruction_error (list): Log ratios of train/test errors
        _nonzero_component_fraction (list): Fraction of nonzero components
        _fraction_window_component (list): Component fractions per window
        _weight_orthogonality (list): Weight matrix orthogonality scores
        _sample_orthogonality (list): Sample matrix orthogonality scores
        _mean_weight_sparsity (list): Mean weight matrix sparsity
        _mean_sample_sparsity (list): Mean sample matrix sparsity
    """

    def __init__(
        self,
        splitter_type: Literal["Sample", "Mask"] = "Mask",
        n_splits: int = 5,
        mask_fraction: float = 0.2,
        mask_signal: bool = True,
        balance_mask_signal: bool = True,
        **nmf_kwargs: Dict[str, Any],
    ):
        # Public attributes are the estimator's own parameters (sklearn get_params reads
        # them from the signature); everything else is forwarded to the NMF fit.
        self.splitter_type = splitter_type
        self._rng = np.random.default_rng(seed=nmf_kwargs.get("random_state", 42))
        self.mask_fraction = mask_fraction
        self.n_splits = n_splits
        self.mask_signal = mask_signal
        self.balance_mask_signal = balance_mask_signal
        self._nmf_kwargs = nmf_kwargs

    def fit_model(self, m: np.ndarray):
        """Fits NMF model to input matrix.

        Args:
            m (np.ndarray): Input matrix to fit

        Returns:
            Tuple[np.ndarray, np.ndarray, NMF]: W (elution profiles,
                n_scans x n_components), H (spectra, n_components x n_mz_bins) and
                the fitted model.
        """
        W, H, model = fit_nmf_matrix_custom_init(
            m,
            return_model=True,
            **self._nmf_kwargs,
        )
        return W, H, model

    def fit(self, X, y=None):
        """Fits the NMF model with masked cross-validation.

        Performs multiple fits with different random masks to evaluate reconstruction
        performance and calculate various metrics.

        Args:
            X: Input data matrices, a list of scan windows (n_scans, n_mz_bins)
            y: Ignored, present for sklearn compatibility

        Returns:
            self: Returns self for method chaining
        """
        # Reset the per-fit metric lists (one entry per split x window).
        self._models = []
        self._reconstruction_errors = []
        self._test_reconstruction_errors = []
        self._train_reconstruction_errors = []
        self._neglog_ratio_train_test_reconstruction_error = []
        self._nonzero_component_fraction = []
        self._fraction_window_component = []
        self._weight_orthogonality = []
        self._sample_orthogonality = []
        self._test_samples = []
        self._train_samples = []
        self._mean_weight_sparsity = []
        self._mean_sample_sparsity = []

        # Pick how data is held out: random cells ("Mask") or whole scans ("Sample").
        if self.splitter_type == "Mask":
            splitter = MzBinMaskingSplitter(
                n_splits=self.n_splits,
                mask_fraction=self.mask_fraction,
                mask_signal=self.mask_signal,
                balance_mask_signal=self.balance_mask_signal,
                random_state=self._nmf_kwargs.get("random_state", 42),
            )
        elif self.splitter_type == "Sample":
            splitter = MzBinSampleSplitter(
                n_splits=self.n_splits,
                mask_fraction=self.mask_fraction,
                random_state=self._nmf_kwargs.get("random_state", 42),
            )
        else:
            raise ValueError(
                f"Invalid splitter type: '{self.splitter_type}'. Must be either 'Mask' or 'Sample'."
            )

        for train_idxes, test_idxes in splitter.split(X):
            for m, train_mask, test_mask in zip(X, train_idxes, test_idxes):
                # Fit NMF on the window with the test part hidden (zeroed cells, or
                # dropped scans for the Sample splitter).
                m_train = m.copy()
                m_train = splitter.mask_train_matrix(m_train, test_mask)
                W, H, model = self.fit_model(m_train)

                # Shape metrics of the solution (sparsity, orthogonality, coverage),
                # see utils.calculate_nmf_summary.
                summary_results = calculate_nmf_summary(W, H)

                m_reconstructed = W @ H

                self._models.append(model)

                # Reconstruction error on the held-out (test) and the fitted (train)
                # parts of the window.
                mse = splitter.reconstruction_error(
                    m,
                    m_reconstructed,
                    test_mask,
                    model,
                )
                mse_train = splitter.reconstruction_error(
                    m,
                    m_reconstructed,
                    train_mask,
                    model,
                )

                # Record this fit's metrics. Note the naming in calculate_nmf_summary:
                # "weight_deviation_identity" is computed from H (spectra) and
                # "sample_deviation_identity" from W (elution profiles), while
                # "weight_sparsity" is the (negated) Hoyer sparsity of W's columns and
                # "sample_sparsity" that of H's rows.
                self._nonzero_component_fraction.append(
                    summary_results["nonzero_component_fraction"]
                )
                self._mean_weight_sparsity.append(summary_results["weight_sparsity"])
                self._mean_sample_sparsity.append(summary_results["sample_sparsity"])
                self._weight_orthogonality.append(
                    summary_results["weight_deviation_identity"]
                )
                self._sample_orthogonality.append(
                    summary_results["sample_deviation_identity"]
                )
                # -log2(train / test): 0 when test error equals train error, positive
                # when test error is larger (over-fitting). Infinite/NaN if mse is 0.
                self._neglog_ratio_train_test_reconstruction_error.append(
                    -1.0 * np.log2(mse_train / mse)
                )
                self._test_reconstruction_errors.append(mse)
                self._train_reconstruction_errors.append(mse_train)
                self._test_samples.append(test_mask.sum())
                self._train_samples.append(train_mask.sum())
                self._fraction_window_component.append(
                    summary_results["fraction_window_component"]
                )

        return self

    def score(self, X, y=None):
        """Calculates overall score as negative mean reconstruction error.

        Note: ``_reconstruction_errors`` is reset in ``fit`` but never filled, so this
        returns NaN. The searches don't call it; they score through _fit_and_score.

        Args:
            X: Input data (unused)
            y: Ignored, present for sklearn compatibility

        Returns:
            float: Negative mean reconstruction error score
        """
        return -1 * np.nanmean(self._reconstruction_errors)

    def get_params(self, deep: bool = True):
        """Gets parameters for this estimator.

        Override get_params to allow access to nmf hyperparameters of interest.
        Including them is what lets ``clone`` rebuild the wrapper with the same NMF
        settings (they come back in through ``**nmf_kwargs``).

        Args:
            deep (bool): If True, will return the parameters for this estimator and
                contained subobjects that are estimators

        Returns:
            dict: Parameter names mapped to their values
        """
        return {**super().get_params(deep=deep), **self._nmf_kwargs}

    def set_params(self, **params: Dict[str, Any]):
        """Sets the parameters of this estimator.

        Override set_params to update the parameters in _nmf_kwargs. The searches
        call this with a candidate's ``alpha_W``/``alpha_H``/``l1_ratio``; names that
        aren't NMF settings are set as attributes on the wrapper.

        Args:
            **params: Estimator parameters

        Returns:
            self: Estimator instance
        """
        for param, value in params.items():
            if param in self._nmf_kwargs.keys():
                self._nmf_kwargs[param] = value
            else:
                setattr(self, param, value)
        return self


class RandomizedSearchReconstructionCV(BaseSearchCV):
    """Cross-validation with randomized parameter search for NMF reconstruction.

    Performs randomized search over parameter spaces for NMF models, evaluating
    reconstruction performance through cross-validation.

    It subclasses sklearn's BaseSearchCV only to reuse its result formatting
    (``_format_results``, which builds ``cv_results`` with ``mean_test_<metric>`` and
    ``rank_test_<metric>`` columns). ``fit`` is reimplemented because X is a list of
    scan windows rather than one sample matrix, and the wrapped estimator does its own
    splitting.

    Args:
        estimator: NMF estimator to optimize
        param_distributions: Dictionary with parameters names (string) as keys and
            distributions or lists of parameters to try
        n_iter (int): Number of parameter settings sampled
        scoring: Strategy to evaluate predictions on the test set
        n_jobs (int): Number of jobs to run in parallel
        verbose (int): Verbosity level
        pre_dispatch (str): Controls the number of jobs that get dispatched
        random_state (int): Random seed for reproducibility
        error_score (float): Value to assign to the score if an error occurs
        return_train_score (bool): Whether to return training scores

    Attributes:
        param_distributions: Parameter distributions for search
        n_iter (int): Number of iterations
        random_state (int): Random seed
        best_index_ (int): Index of best model
        best_score_ (float): Score of best model
        best_params_ (dict): Parameters of best model
        cv_results (dict): Cross-validation results
        n_splits_ (int): Number of splits used
    """

    _required_parameters = ["estimator", "param_distributions"]

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates: Callable):
        """Evaluate n_iter parameter settings sampled from param_distributions.

        Args:
            evaluate_candidates (Callable): Function from ``fit`` that scores a list of
                candidate parameter dicts.
        """
        evaluate_candidates(
            ParameterSampler(
                self.param_distributions,
                self.n_iter,
                random_state=self.random_state,
            )
        )

    def fit(
        self, X: Union[List[Any], np.ndarray], y=None, **params: Dict[str, Any]
    ) -> "RandomizedSearchReconstructionCV":
        """Score every sampled parameter setting in parallel and pick the best.

        Each candidate is one joblib task that clones the estimator, sets the
        candidate's parameters and fits it on all windows (see _fit_and_score). Worker
        log records go through a multiprocessing queue to the root logger's handlers.

        Args:
            X (Union[List[Any], np.ndarray]): Scan window matrices to tune on.
            y: Ignored, present for sklearn compatibility.
            **params: Unused.

        Returns:
            RandomizedSearchReconstructionCV: self, with cv_results, best_index_,
                best_score_ and best_params_ set (best by mean test reconstruction
                score; the CLI instead picks by harmonic mean of the objectives).
        """
        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        n_splits = self.estimator.n_splits

        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def objective(
                estimator: BaseEstimator,
                X: Union[List[Any], np.ndarray],
                log_queue: Queue,
                iter_n: int,
                extra_scores: Optional[List[str]] = None,
                parameters: Dict[str, Any] = {},
                **kwargs: Dict[str, Any],
            ) -> List[Dict]:
                """Fit and score one candidate (runs in a joblib worker).

                Args:
                    estimator (BaseEstimator): Fresh clone of the NMFMaskWrapper.
                    X (Union[List[Any], np.ndarray]): Scan window matrices.
                    log_queue (Queue): Queue forwarding log records to the parent.
                    iter_n (int): Candidate number, used in the logger name.
                    extra_scores (Optional[List[str]]): Metric attributes to report.
                    parameters (Dict[str, Any]): Candidate NMF parameters.
                    **kwargs: Passed on to _fit_and_score.

                Returns:
                    List[Dict]: One result dict per split, see _fit_and_score.
                """
                iter_logger = logging.getLogger(f"iter_{iter_n}")
                _set_queue_handler(iter_logger, log_queue)
                format_logger(iter_logger, level=logging.INFO)
                iter_logger.info(f"Iter {iter_n} starting with params: {parameters}")

                return _fit_and_score(
                    estimator,
                    X,
                    parameters=parameters,
                    extra_scores=extra_scores,
                    **kwargs,
                )

            def evaluate_candidates(
                candidate_params: ParameterSampler,
                more_results: Optional[Dict[str, Any]] = None,
            ) -> Dict[str, Any]:
                """Score candidates in parallel and accumulate them into cv_results.

                Args:
                    candidate_params (ParameterSampler): Parameter dicts to try.
                    more_results (Optional[Dict[str, Any]]): Extra columns for
                        cv_results (sklearn interface).

                Returns:
                    Dict[str, Any]: cv_results for all candidates evaluated so far.

                Raises:
                    ValueError: If nothing was fit, or the number of results doesn't
                        match n_candidates x n_splits.
                """
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                with Manager() as manager:
                    # Create a queue for logging
                    log_queue = manager.Queue()
                    # Get the root logger and create a queue listener
                    logger = logging.getLogger()
                    listener = QueueListener(log_queue, *logger.handlers)
                    listener.start()

                    out = parallel(
                        delayed(objective)(
                            clone(base_estimator),
                            X,
                            log_queue,
                            i,
                            parameters=parameters,
                            extra_scores=_FIT_AND_SCORE,
                        )
                        for i, parameters in enumerate(candidate_params)
                    )

                    listener.stop()

                # Flatten [candidate][split] -> one list in candidate-major order, the
                # layout _format_results expects.
                out = [i for j in out for i in j]

                if len(out) < 1:
                    raise ValueError("No fits were performed.")
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "Returned fits not consistent with size of parameter space."
                    )

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results

                # sklearn turns the per-split dicts into cv_results arrays
                # (params, mean/std/rank per metric).
                results = self._format_results(
                    all_candidate_params,
                    n_splits,
                    all_out,
                    all_more_results,
                )

                return results

        self._run_search(evaluate_candidates)

        # Rank 1 = best mean test score (the negative test reconstruction error).
        self.best_index_ = results["rank_test_score"].argmin()
        self.best_score_ = results["mean_test_score"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.cv_results = results
        self.n_splits_ = n_splits

        return self


class OptunaSearchReconstructionCV(BaseSearchCV):
    """Cross-validation with Optuna-based parameter search for NMF reconstruction.

    Uses Optuna for hyperparameter optimization of NMF models, supporting
    multi-objective optimization and pruning of poor parameter combinations.

    Each trial suggests ``alpha_W``/``alpha_H``/``l1_ratio`` from ``param_sampling``,
    fits the wrapped estimator on all tuning windows (see _fit_and_score) and returns
    one score per entry of ``objective_params``. Optuna (NSGA-II) maximizes all of them
    at once and keeps the Pareto front (``best_trials_``). n_jobs joblib workers run
    trials against one study stored in a journal file. As in
    RandomizedSearchReconstructionCV, BaseSearchCV is used for its ``_format_results``.

    Args:
        estimator: NMF estimator to optimize
        param_sampling: Parameter space definition for Optuna
        objective_params: Parameters to optimize in objective function
        n_iter (int): Number of trials
        scoring: Strategy to evaluate predictions
        n_jobs (int): Number of parallel jobs
        verbose (int): Verbosity level
        pre_dispatch (str): Controls job dispatching
        random_state (int): Random seed
        error_score (float): Score to assign on error
        return_train_score (bool): Whether to return training scores
        prune_bad_parameter_spaces (bool): Whether to prune poor parameter combinations
        optimization_parameters (OptimizationParameters): Parameters for optimization

    Attributes:
        param_sampling: Parameter sampling strategy
        n_iter (int): Number of iterations
        random_state (int): Random seed
        objective_params: Objective parameters
        prune_bad_parameter_spaces (bool): Pruning flag
        optimization_parameters (OptimizationParameters): Optimization parameters
        best_index_ (int): Index of best model
        best_score_ (float): Score of best model
        best_params_ (dict): Parameters of best model
        best_trials_ (list): Best trials from optimization
        cv_results (dict): Cross-validation results
        n_splits_ (int): Number of splits used

    Class Attributes:
        _prune_at_zero (list): Parameters to prune when zero
        _required_parameters (list): Required initialization parameters
        _parameter_constraints (dict): Constraints on parameters
    """

    # Metrics that are exactly 0 for degenerate solutions (e.g. fewer than two non-zero
    # components, see calculate_nmf_summary). With prune_bad_parameter_spaces, a 0 is
    # replaced by -10 so those settings score badly instead of looking ideal.
    _prune_at_zero = [
        "_weight_orthogonality",
        "_sample_orthogonality",
        "_nonzero_component_fraction",
        "_mean_weight_sparsity",
        "_mean_sample_sparsity",
        "_fraction_window_component",
    ]

    _required_parameters = ["estimator", "param_distributions"]

    _parameter_constraints: dict = {
        **BaseSearchCV._parameter_constraints,
        "param_distributions": [dict, list],
        "n_iter": [Interval(numbers.Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        estimator,
        param_sampling,
        objective_params,
        *,
        n_iter=10,
        scoring=None,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        pre_dispatch: str = "2*n_jobs",
        random_state: Optional[int] = None,
        error_score: float = np.nan,
        return_train_score: bool = False,
        prune_bad_parameter_spaces: bool = False,
        optimization_parameters: OptimizationParameters = OptimizationParameters(),
    ):
        self.param_sampling = param_sampling
        self.n_iter = n_iter
        self.random_state = random_state
        self.objective_params = objective_params
        self.prune_bad_parameter_spaces = prune_bad_parameter_spaces
        self.optimization_parameters = optimization_parameters
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )

    def _run_search(self, evaluate_candidates: Callable):
        """Run the Optuna search over the parameter space definition.

        Args:
            evaluate_candidates (Callable): Function from ``fit``; receives the search
                space dict (name -> (low, high[, "log"]) or categorical choices).
        """
        evaluate_candidates(self.param_sampling)

    def fit(
        self, X: Union[List[Any], np.ndarray], y=None, **params: Dict[str, Any]
    ) -> "OptunaSearchReconstructionCV":
        """Run the multi-objective Optuna search on the tuning windows.

        Args:
            X (Union[List[Any], np.ndarray]): Scan window matrices to tune on.
            y: Ignored, present for sklearn compatibility.
            **params: Unused.

        Returns:
            OptunaSearchReconstructionCV: self, with cv_results (successful trials
                only), best_index_/best_score_/best_params_ (by mean test
                reconstruction score) and best_trials_ (the Pareto front).

        Raises:
            ValueError: If no trial, or fewer than half of n_iter trials, succeeded.
        """
        base_estimator = clone(self.estimator)

        n_splits = self.estimator.n_splits

        results = {}
        best_trials = []

        all_more_results = defaultdict(list)

        def evaluate_candidates(
            candidate_params: Dict[str, Tuple[Any]],
            more_results: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            """Run the Optuna study and convert its trials into cv_results.

            Args:
                candidate_params (Dict[str, Tuple[Any]]): Search space, name ->
                    (low, high[, "log"]) for floats or a tuple of strings for a
                    categorical.
                more_results (Optional[Dict[str, Any]]): Extra columns for
                    cv_results (sklearn interface).

            Returns:
                Dict[str, Any]: cv_results built from the successful trials.
            """

            def objective(trial: optuna.Trial) -> Union[np.ndarray, Tuple[np.ndarray]]:
                """Objective function for Optuna optimization.

                Evaluates a trial's parameter set by fitting models and calculating scores.

                The per-split results are pickled into the trial's user attribute
                "out", because the trials are later read back from storage in the
                parent process to build cv_results. A failed trial stores "error" and
                is raised as TrialPruned, so it doesn't stop the study.

                Args:
                    trial (optuna.Trial): Current optimization trial

                Returns:
                    Union[float, Tuple[float, ...]]: Score(s) for the trial

                Raises:
                    optuna.TrialPruned: If trial should be pruned
                """
                results = None
                params = {}
                try:
                    # Ask the sampler for a value of each parameter in the space.
                    for k, v in candidate_params.items():
                        # Suggest categorical variable.
                        if isinstance(v[0], str):
                            params[k] = trial.suggest_categorical(k, v)
                        # Suggest continuous variable.
                        elif isinstance(v[0], (float, int)):
                            params[k] = trial.suggest_float(
                                k,
                                *v[:2],
                                log=True if (len(v) > 2 and v[2] == "log") else False,
                            )

                    # Fit NMF on every tuning window for every split with these
                    # parameters (this is where nearly all tuning time goes).
                    results = _fit_and_score(
                        clone(base_estimator),
                        X,
                        parameters=params,
                        extra_scores=_FIT_AND_SCORE,
                    )

                    if any([r["fit_error"] for r in results]):
                        raise Exception("One or more fits failed in cv.")

                    trial.set_user_attr(
                        "out", base64.b64encode(pickle.dumps(results)).decode("utf-8")
                    )

                    # One value per objective: the mean over splits of that metric's
                    # test score, turned into "distance from target" where a target
                    # exists (OptimizationParameters.score). All are maximized.
                    try:
                        output_parameters = []
                        for k in self.objective_params:
                            parameter_value = np.nanmean(
                                [r["test_scores"][k] for r in results]
                            )
                            if (
                                self.prune_bad_parameter_spaces
                                and f"_{k}" in self._prune_at_zero
                                and parameter_value == 0.0
                            ):
                                parameter_value = -10.0
                            parameter_value = self.optimization_parameters.score(
                                k, parameter_value
                            )
                            output_parameters.append(parameter_value)
                        return tuple(output_parameters)
                    except Exception as e:
                        print("Only one objective defined.", e)

                    # Single-objective fallback. test_scores are dicts here, so this
                    # likely raises too and the trial ends up pruned below.
                    return np.nanmean([r["test_scores"] for r in results])

                except Exception as e:
                    # Mark the trial failed and keep the study going. "out" gets
                    # placeholders so every trial has the attribute.
                    if results is not None:
                        print([r["fit_error"] for r in results])
                    print(e)
                    trial.set_user_attr("error", str(e))
                    trial.set_user_attr(
                        "out",
                        base64.b64encode(
                            pickle.dumps([None] * int(base_estimator.n_splits))
                        ).decode("utf-8"),
                    )
                    raise optuna.TrialPruned()

            def optimize_optuna_study(
                study_name: str,
                journal_path: str,
                objective: Callable,
                n_trials: int,
                worker_n: int,
                log_queue: Queue,
            ) -> None:
                """Runs Optuna optimization study.

                Runs in a joblib worker. Every worker opens the same study through its
                own storage object on the shared journal file, so trials from all
                workers end up in one study and the samplers see each other's results.

                Args:
                    study_name (str): Name for the study
                    journal_path (str): Path of the journal file storing study results
                    objective (Callable): Objective function to optimize
                    n_trials (int): Number of trials to run
                    worker_n (int): Worker number, used for the logger name and seed.
                    log_queue (Queue): Queue forwarding log records to the parent.
                """
                working_logger = logging.getLogger(f"worker_{worker_n}")
                _set_queue_handler(working_logger, log_queue)
                format_logger(working_logger, level=logging.INFO)
                working_logger.info(f"Worker {worker_n} starting...")

                class OptunaLoggingCallback:
                    """Callback for logging Optuna optimization progress.

                    Logs trial results, parameters, and optimization progress.
                    """

                    def __init__(self, logger: logging.Logger, worker: int) -> None:
                        """
                        Args:
                            logger (logging.Logger): Instance of worker logger.
                            worker (int): Worker number.
                        """
                        self.logger = logger
                        self.worker = worker

                    def __call__(
                        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
                    ) -> None:
                        """Log the outcome of a finished trial.

                        Args:
                            study (optuna.study.Study): The running study.
                            trial (optuna.trial.FrozenTrial): The trial that finished.
                        """
                        if "error" in trial.user_attrs:
                            error_msg = trial.user_attrs.get("error", "Unknown error")
                            self.logger.error(
                                f"Failed trial: {trial.number} with params: {trial.params} on {self.worker}, {error_msg}."
                            )
                        else:
                            self.logger.info(
                                f"Completed trial: {trial.number} with params: {trial.params} on {self.worker}."
                            )

                # Seed each worker's sampler differently so workers don't propose the
                # same parameters. Parallel scheduling still makes runs not exactly
                # reproducible.
                sampler = optuna.samplers.NSGAIISampler(
                    seed=(
                        None
                        if self.random_state is None
                        else int(self.random_state) + worker_n
                    )
                )
                study = optuna.create_study(
                    study_name=study_name,
                    storage=_journal_storage(journal_path),
                    load_if_exists=True,
                    sampler=sampler,
                )
                callback = OptunaLoggingCallback(working_logger, worker_n)
                study.optimize(objective, n_trials=n_trials, callbacks=[callback])

            # Journal storage in a temp dir, not sqlite in /tmp/ (missing on Windows).
            temp_dir = tempfile.mkdtemp(prefix="ffd_optuna_")
            journal_path = str(Path(temp_dir) / "optuna_journal.log")
            storage = _journal_storage(journal_path)
            try:
                with Manager() as manager:
                    # Worker processes can't write to the parent's log handlers, so
                    # they send records over a managed queue, and a QueueListener in
                    # this process passes them to the root logger's handlers.
                    root_logger = logging.getLogger()
                    log_queue = manager.Queue()
                    listener = QueueListener(log_queue, *root_logger.handlers)
                    listener.start()

                    try:
                        # Create the shared study once in the parent: one "maximize"
                        # direction per objective (multi-objective, Pareto front).
                        study = optuna.create_study(
                            storage=storage,
                            directions=["maximize" for _ in self.objective_params],
                        )

                        # n_jobs workers, each running n_iter // n_jobs + 1 trials, so
                        # slightly more than n_iter trials in total.
                        with Parallel(
                            n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch
                        ) as parallel:
                            parallel(
                                delayed(optimize_optuna_study)(
                                    study.study_name,
                                    journal_path,
                                    objective,
                                    (self.n_iter // self.n_jobs) + 1,
                                    i,
                                    log_queue,
                                )
                                for i in range(self.n_jobs)
                            )
                    finally:
                        listener.stop()

                # Read everything needed from storage before removing the journal.
                study = optuna.load_study(study_name=study.study_name, storage=storage)
                trials = study.trials
                study_best_trials = [
                    {
                        "params": t.params,
                        "scores": dict(zip(self.objective_params, t.values)),
                    }
                    for t in study.best_trials
                ]
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            # Rebuild the per-split results of every successful trial from the pickled
            # "out" attribute: n_splits dicts per trial, in trial order.
            successful_trials = [
                t
                for t in trials
                if "out" in t.user_attrs and not t.user_attrs.get("error", None)
            ]
            all_out = [
                r
                for t in successful_trials
                for r in pickle.loads(base64.b64decode(t.user_attrs["out"]))
            ]

            # Failed trials are pruned and excluded. Only give up if too few succeeded.
            n_successful = len(successful_trials)
            if n_successful < 1:
                raise ValueError("No fits were performed.")
            elif n_successful < 0.5 * self.n_iter:
                raise ValueError(
                    f"Only {n_successful} of {len(trials)} trials succeeded, fewer "
                    f"than half of n_iter ({self.n_iter})."
                )
            elif n_successful < self.n_iter:
                logger.warning(
                    f"Only {n_successful} of {len(trials)} trials succeeded "
                    f"(n_iter = {self.n_iter}). Continuing with successful trials."
                )

            all_candidate_params = [t.params for t in successful_trials]

            if more_results:
                for key, value in more_results.items():
                    all_more_results[key].extend(value)

            nonlocal results

            # sklearn builds cv_results (params plus mean/std/rank of every metric,
            # e.g. "mean_test_weight_orthogonality") from the per-split dicts.
            results = self._format_results(
                all_candidate_params,
                n_splits,
                all_out,
                all_more_results,
            )

            nonlocal best_trials

            best_trials = study_best_trials

            return results

        self._run_search(evaluate_candidates)

        # "Best" here is only by test reconstruction score. The pipeline chooses its
        # parameters with pick_parameters_harmonic_mean over all objectives instead.
        self.best_index_ = results["rank_test_score"].argmin()
        self.best_score_ = results["mean_test_score"][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]
        self.best_trials_ = best_trials

        self.cv_results = results
        self.n_splits_ = n_splits

        return self


def _fit_and_score(
    estimator: BaseEstimator,
    X: Union[List[Any], np.ndarray],
    extra_scores: Optional[List[str]] = None,
    parameters: Dict[str, Any] = {},
    **kwargs: Dict[str, Any],
) -> List[Dict]:
    """Fits an estimator and calculates scores for cross-validation.

    Scores are negated metrics, so that larger is better for all of them: the test
    score is -MSE on held-out data, and each extra score is -1 x the recorded metric.

    Note: one entry is read per index ``i < n_splits`` from the estimator's metric
    lists, but NMFMaskWrapper.fit appends one entry per (split, window) in split-major
    order. So entry i is window i of the first split, not an aggregate of split i.

    Args:
        estimator (BaseEstimator): Scikit-learn estimator to fit
        X (Union[List[Any], np.ndarray]): Training data (list of scan windows)
        extra_scores (Optional[List[str]]): Additional scoring metrics to calculate
            (attribute names on the estimator, e.g. _FIT_AND_SCORE)
        parameters (Dict[str, Any]): Parameters to set on the estimator
        **kwargs (Dict[str, Any]): Additional keyword arguments

    Returns:
        List[Dict]: List of dictionaries containing for each fold:
            - test_scores: Test set scores (a dict {"score": ..., <metric>: ...} when
              extra_scores are given)
            - train_scores: Training set scores
            - n_test_samples: Number of test samples
            - estimator: Fitted estimator
            - fit_time: Time taken to fit
            - score_time: Time taken to score
            - fit_error: Any error during fitting

        On a fit failure only estimator/fit_time/score_time/fit_error are present,
        with fit_error set to the exception message.
    """
    start_time = time.time()
    fit_error = None
    score_time = 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.set_params(**parameters)
            estimator.fit(X)
    except Exception as e:
        fit_time = time.time() - start_time
        fit_error = str(e)
        # The fit may have failed before all models were appended (or before _models
        # was set), don't let that hide the original error.
        models = list(getattr(estimator, "_models", []))
        models += [None] * (estimator.n_splits - len(models))
        return [
            {
                "estimator": models[i],
                "fit_time": fit_time,
                "score_time": score_time,
                "fit_error": fit_error,
            }
            for i in range(estimator.n_splits)
        ]
    fit_time = time.time() - start_time

    # The estimator's metric lists hold one entry per (split, window), ordered split by
    # split. Average each metric over the windows of a split to get one value per fold.
    n_windows = len(estimator._models) // estimator.n_splits

    def split_mean(values: List[float], i: int) -> float:
        return float(np.nanmean(values[i * n_windows : (i + 1) * n_windows]))

    # Base result per fold: test/train scores are the negated reconstruction MSEs.
    results = [
        {
            "test_scores": -1.0
            * split_mean(estimator._test_reconstruction_errors, i),
            "train_scores": -1.0
            * split_mean(estimator._train_reconstruction_errors, i),
            "n_test_samples": int(
                np.sum(estimator._test_samples[i * n_windows : (i + 1) * n_windows])
            ),
            # First model of the fold only, to keep the pickled trial results small.
            "estimator": estimator._models[i * n_windows],
            "fit_time": fit_time,
            "score_time": score_time,
            "fit_error": fit_error,
        }
        for i in range(estimator.n_splits)
    ]

    # Turn the scalar scores into dicts with the extra metrics (negated, underscore
    # stripped). sklearn's _format_results then reports each key as its own metric.
    # The same values are used for test and train, since the metrics describe the fit.
    if extra_scores and all(hasattr(estimator, es) for es in extra_scores):
        for i, r in enumerate(results):
            r["test_scores"] = {
                "score": r["test_scores"],
            }
            r["test_scores"].update(
                {
                    es.lstrip("_"): -1.0 * split_mean(getattr(estimator, es), i)
                    for es in extra_scores
                }
            )
            r["train_scores"] = {
                "score": r["train_scores"],
            }
            r["train_scores"].update(
                {
                    es.lstrip("_"): -1.0 * split_mean(getattr(estimator, es), i)
                    for es in extra_scores
                }
            )

    return results


def fit_nmf_matrix_custom_init(
    m: np.ndarray,
    n_components: int = 20,
    alpha_W: float = 0.00001,
    alpha_H: float = 0.0375,
    l1_ratio: float = 0.75,
    max_iter: int = 500,
    solver: str = "cd",
    random_state: int = 42,
    init: Optional[str] = None,
    W_init: Optional[np.ndarray] = None,
    H_init: Optional[np.ndarray] = None,
    return_model: bool = False,
    **nmf_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Optional[NMF]]:
    """Fits NMF model with optional custom initialization.

    This is the one NMF call used by both the pipeline (one call per scan window) and
    tuning. scikit-learn minimizes

        0.5 * ||m - W H||_F^2
        + alpha_W * n_features * (l1_ratio * |W|_1 + 0.5 * (1 - l1_ratio) * ||W||_F^2)
        + alpha_H * n_samples * (l1_ratio * |H|_1 + 0.5 * (1 - l1_ratio) * ||H||_F^2)

    with n_samples = number of scans and n_features = number of m/z bins (penalties
    scaled by matrix size since scikit-learn 1.2). The L1 part pushes elution
    profiles (W) and spectra (H) toward sparsity, which helps separate co-eluting
    peptides.

    Args:
        m (np.ndarray): Input data matrix (n_scans, n_mz_bins), non-negative
        n_components (int): Number of components to extract
        alpha_W (float): L1/L2 regularization parameter for W matrix
        alpha_H (float): L1/L2 regularization parameter for H matrix
        l1_ratio (float): Ratio of L1 vs L2 regularization
        max_iter (int): Maximum number of iterations
        solver (str): NMF solver to use
        random_state (int): Random seed
        init (Optional[str]): Initialization method
        W_init (Optional[np.ndarray]): Initial W matrix
        H_init (Optional[np.ndarray]): Initial H matrix
        return_model (bool): Whether to return fitted model
        **nmf_kwargs (Dict[str, Any]): Additional keyword arguments for NMF

    Returns:
        Tuple[np.ndarray, np.ndarray, Optional[NMF]]:
            - W matrix (n_scans, n_components): elution profile of each component
            - H matrix (n_components, n_mz_bins): fragment spectrum of each component
              (sklearn's ``components_``)
            - Fitted NMF model (if return_model=True)

        If the fit fails, or the matrix is empty, W and H are all zeros and the
        model's reconstruction_err_ is set to 1e6.
    """
    # Initialization: custom when both W_init and H_init are given (warm start),
    # otherwise the requested init, defaulting to NNDSVD for the coordinate-descent
    # solver (deterministic; its zeros stay zero, which favors sparse solutions) and
    # NNDSVDa for others (zeros filled with the mean so multiplicative updates can
    # move them).
    init = (
        "custom"
        if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray)
        else (init if init else ("nndsvd" if solver == "cd" else "nndsvda"))
    )
    # Rank can't exceed the smaller matrix dimension (NNDSVD requires it); keep >= 1.
    n_components = min(n_components, max(min(m.shape), 1))
    model = NMF(
        n_components=n_components,
        alpha_W=alpha_W,
        alpha_H=alpha_H,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        solver=solver,
        init=init,
        random_state=random_state,
        **nmf_kwargs,
    )
    try:
        if min(m.shape) == 0:
            # Output empty basis and weight matrix.
            W = np.zeros((m.shape[0], n_components))
            model.components_ = np.zeros((n_components, m.shape[1]))
            model.reconstruction_err_ = 1e6
        else:
            if isinstance(W_init, np.ndarray) and isinstance(H_init, np.ndarray):
                W = model.fit_transform(m, W=W_init, H=H_init)
            else:
                try:
                    W = model.fit_transform(m)
                except ValueError:
                    if m.dtype == np.float64:
                        raise
                    # In float32, the randomized SVD behind NNDSVD can overflow to
                    # NaN on some small windows, which used to leave the window with
                    # no components. Retry once in float64 and return the input's
                    # dtype. Windows that fit in float32 are unaffected.
                    W = model.fit_transform(m.astype(np.float64)).astype(m.dtype)
                    model.components_ = model.components_.astype(m.dtype)
    except Exception as e:
        # This exception getting raised in scipy linalg somewhere during nmf initialization.
        # str(e) == "array must not contain infs or NaNs"

        # Output empty basis and weight matrix.
        W = np.zeros((m.shape[0], n_components))
        model.components_ = np.zeros((n_components, m.shape[1]))
        model.reconstruction_err_ = 1e6

        logger.exception(e)

    # H is sklearn's components_ (n_components x n_mz_bins).
    if return_model:
        return W, model.components_, model
    return W, model.components_


def pick_parameters_harmonic_mean(
    bcv: BaseSearchCV,
    parameter_names: List[str],
    constant: float = 0.001,
    optimization_parameters: OptimizationParameters = OptimizationParameters(),
) -> Dict[str, Any]:
    """Selects optimal parameters using harmonic mean of cross-validation scores.

    Reduces the multi-objective results to one number per candidate:

    1. take each objective's mean test score per candidate,
    2. convert target-based objectives to "distance from target" scores,
    3. min-max scale each objective across candidates to [0, 1] (1 = best),
    4. combine with a harmonic mean, which rewards candidates that are decent on every
       objective over ones that are excellent on some and poor on others.

    Because of the min-max scaling, the result depends on which candidates were tried.
    If all candidates tie on one objective, its scaling divides by zero (NaN).

    Args:
        bcv (BaseSearchCV): Fitted cross-validation object
        parameter_names (List[str]): Names of the objectives to combine (keys of the
            form ``mean_test_<name>`` in cv_results)
        constant (float): Small constant to avoid division by zero
        optimization_parameters (OptimizationParameters): Parameters for optimization

    Returns:
        np.ndarray: Harmonic-mean score per candidate, aligned with
            ``bcv.cv_results["params"]``. The caller takes the argmax.

    Raises:
        ValueError: If any parameter name is not found in cv_results
    """
    if not all([f"mean_test_{p}" in bcv.cv_results.keys() for p in parameter_names]):
        raise ValueError(
            "All parameter names must be a valid parameter name in cv_results."
        )

    parameters = {p: bcv.cv_results[f"mean_test_{p}"] for p in parameter_names}

    # reset params if the target value was different than reported here.
    parameters = {k: optimization_parameters.score(k, v) for k, v in parameters.items()}

    # min-max scale all parameters.
    parameters = {p: (v - v.min()) / (v - v.min()).max() for p, v in parameters.items()}

    # n_objectives / sum(1 / (scaled + constant)), per candidate (column).
    harmonic_mean = len(parameters) / (
        1.0
        / (
            np.vstack(
                list(parameters.values()),
            )
            + constant
        )
    ).sum(axis=0)

    return harmonic_mean


def tune_hyperparameters_randomizedsearchcv(
    ms: List[np.ndarray],
    config: Config = Config(),
    save_run: bool = True,
    n_jobs: int = 4,
) -> Dict[str, Any]:
    """Tune hyperparameters using RandomizedSearchCV.

    Samples config.tuning.n_iter settings from config.tuning.hyperparameter_grid,
    scores each with masked cross-validation on the tuning windows and returns the
    setting with the best harmonic mean over config.tuning.objective_params.

    While tuning runs, the root logger's file handler is swapped for a tuning-specific
    log file (``<main log>.tuning_<timestamp>.log``) and restored afterwards.

    Args:
        ms (List[np.ndarray]): List of matrices to tune on
        config (Config): Configuration object containing tuning parameters
        save_run (bool): Whether to save the tuning results (pickle of the search
            object next to the tuning log)
        n_jobs (int): Number of parallel workers

    Returns:
        Dict[str, Any]: Best parameters found by harmonic mean scoring
    """
    # Get root logger setup
    logger = logging.getLogger()
    original_logger_state = logger.disabled
    original_logger_level = logger.level

    log_suffix = f"tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    base_file_name = Path("randomized_tuning.log")

    # Create new file handler: replace the first root FileHandler (the run log) with
    # one writing to a tuning-specific file next to it, so the many per-trial lines
    # don't flood the main log.
    old_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_file_name = Path(handler.baseFilename)
            base_file_name = base_file_name.with_suffix(f".{log_suffix}.log")
            new_handler = logging.FileHandler(
                base_file_name,
                mode="w",
            )
            new_handler.setFormatter(handler.formatter)
            new_handler.setLevel(handler.level)
            logger.addHandler(new_handler)
            old_handler = handler
            logger.removeHandler(old_handler)
            break

    # Add a filehandler if one didn't already exist on root logger
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        new_handler = logging.FileHandler(
            base_file_name.with_suffix(f".{log_suffix}.log"),
            mode="w",
        )
        logger.addHandler(new_handler)

    # Format all handlers on logger.
    format_logger(logger)

    try:
        # Create parameters to optimize from config
        op = OptimizationParameters.from_config(config)

        # The estimator starts from the current [nmf] values; the search overrides
        # alpha_W/alpha_H/l1_ratio per candidate. Tuning uses tuning.n_components.
        rcv = RandomizedSearchReconstructionCV(
            NMFMaskWrapper(
                splitter_type=config.tuning.splitter_type,
                mask_fraction=config.tuning.test_fraction,
                n_components=config.tuning.n_components,
                n_splits=config.tuning.n_splits,
                l1_ratio=config.nmf.l1_ratio,
                alpha_W=config.nmf.alpha_W,
                alpha_H=config.nmf.alpha_H,
            ),
            config.tuning.hyperparameter_grid,
            n_iter=config.tuning.n_iter,
            random_state=config.random_seed,
            return_train_score=True,
            n_jobs=n_jobs,
        )

        logger.info("Starting hyperparameter optimization")
        logger.info(f"Number of iterations: {config.tuning.n_iter}")
        logger.info(f"Parameter space: {config.tuning.hyperparameter_grid}")

        rcv.fit([m.copy() for m in ms])

        logger.info("Optimization complete")
        logger.info(f"Best parameters: {rcv.best_params_}")
        logger.info(f"Best score: {rcv.best_score_}")

        # Calculate harmonic mean scores
        harmonic_scores = pick_parameters_harmonic_mean(
            rcv, config.tuning.objective_params, optimization_parameters=op
        )
        best_harmonic_idx = np.argmax(harmonic_scores)
        logger.info(
            f"Best parameters by harmonic mean: {rcv.cv_results['params'][best_harmonic_idx]}"
        )
        logger.info(f"Best harmonic mean score: {harmonic_scores[best_harmonic_idx]}")

        if save_run:
            pickle_fh = base_file_name.with_suffix(".pkl")
            with open(pickle_fh, "wb") as f:
                pickle.dump(rcv, f)

        best_params = rcv.cv_results["params"][best_harmonic_idx]
    finally:
        # Restore the original handlers even if tuning raised.
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                if handler is new_handler:
                    logger.removeHandler(handler)
                    handler.close()

        if old_handler is not None:
            logger.addHandler(old_handler)

        logger.disabled = original_logger_state
        logger.setLevel(original_logger_level)

    return best_params


def tune_hyperparameters_optunasearchcv(
    ms: List[np.ndarray],
    config: Config = Config(),
    save_run: bool = True,
    n_jobs: int = 4,
) -> Dict[str, Any]:
    """Tune hyperparameters using OptunaSearchCV.

    Performs hyperparameter optimization using Optuna with multiple objectives.
    Logs results and optionally saves the optimization run.

    Runs config.tuning.n_iter Optuna trials over
    config.tuning.optuna_hyperparameter_grid with config.tuning.objective_params as
    objectives, then picks one setting by the
    harmonic mean over those objectives. Logging is redirected to a tuning-specific
    file for the duration, as in tune_hyperparameters_randomizedsearchcv, and Optuna's
    own log messages are routed through the root logger.

    Args:
        ms (List[np.ndarray]): List of matrices to tune on
        config (Config): Configuration object containing tuning parameters
        save_run (bool): Whether to save the tuning results (pickle of the search
            object next to the tuning log)
        n_jobs (int): Number of parallel Optuna workers

    Returns:
        Dict[str, Any]: Best parameters found by harmonic mean scoring
    """
    # Get root logger setup.
    logger = logging.getLogger()
    original_logger_state = logger.disabled
    original_logger_level = logger.level

    log_suffix = f"tuning_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    base_file_name = Path("optuna_tuning.log")

    # Create new file handler: swap the run log's FileHandler for a tuning-specific
    # file next to it (restored in the finally block below).
    old_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            base_file_name = Path(handler.baseFilename)
            base_file_name = base_file_name.with_suffix(f".{log_suffix}.log")
            new_handler = logging.FileHandler(
                base_file_name,
                mode="w",
            )
            new_handler.setFormatter(handler.formatter)
            new_handler.setLevel(handler.level)
            logger.addHandler(new_handler)
            old_handler = handler
            logger.removeHandler(old_handler)
            break

    # Add a filehandler if one didn't already exist on root logger.
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        new_handler = logging.FileHandler(
            base_file_name.with_suffix(f".{log_suffix}.log"),
            mode="w",
        )
        logger.addHandler(new_handler)

    # Format all handlers on logger.
    format_logger(logger)

    try:
        # Send Optuna's messages through the root logger (and so to the tuning log)
        # instead of Optuna's own stderr handler.
        optuna.logging.disable_default_handler()
        optuna.logging.enable_propagation()

        # Create parameters to optimize from config.
        op = OptimizationParameters.from_config(config)

        # The estimator starts from the current [nmf] values; each trial overrides
        # alpha_W/alpha_H/l1_ratio. Tuning uses tuning.n_components.
        ocv = OptunaSearchReconstructionCV(
            NMFMaskWrapper(
                splitter_type=config.tuning.splitter_type,
                mask_fraction=config.tuning.test_fraction,
                n_components=config.tuning.n_components,
                n_splits=config.tuning.n_splits,
                l1_ratio=config.nmf.l1_ratio,
                alpha_W=config.nmf.alpha_W,
                alpha_H=config.nmf.alpha_H,
            ),
            config.tuning.optuna_hyperparameter_grid,
            objective_params=config.tuning.objective_params,
            n_iter=config.tuning.n_iter,
            random_state=config.random_seed,
            return_train_score=True,
            n_jobs=n_jobs,
            prune_bad_parameter_spaces=False,
            optimization_parameters=op,
        )

        logger.info("Starting hyperparameter optimization")
        logger.info(f"Number of iterations: {config.tuning.n_iter}")
        logger.info(f"Parameter space: {config.tuning.optuna_hyperparameter_grid}")

        ocv.fit([m.copy() for m in ms])

        logger.info("Optimization complete")

        for trial in ocv.best_trials_:
            logger.info(f"Trial parameters: {trial['params']}")
            logger.info(f"Trial scores: {trial['scores']}")

        # Calculate harmonic mean scores
        harmonic_scores = pick_parameters_harmonic_mean(
            ocv, config.tuning.objective_params, optimization_parameters=op
        )
        best_harmonic_idx = np.argmax(harmonic_scores)
        logger.info(
            f"Best parameters by harmonic mean: {ocv.cv_results['params'][best_harmonic_idx]}"
        )
        logger.info(f"Best harmonic mean score: {harmonic_scores[best_harmonic_idx]}")

        if save_run:
            pickle_fh = base_file_name.with_suffix(".pkl")
            with open(pickle_fh, "wb") as f:
                pickle.dump(ocv, f)

        best_params = ocv.cv_results["params"][best_harmonic_idx]
    finally:
        # Restore the original handlers even if tuning raised.
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                if handler is new_handler:
                    logger.removeHandler(handler)
                    handler.close()

        if old_handler is not None:
            logger.addHandler(old_handler)

        # Disable propagation on the optuna logger.
        optuna.logging.disable_propagation()

        logger.disabled = original_logger_state
        logger.setLevel(original_logger_level)

    return best_params
