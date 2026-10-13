"""ScanWindow: one time tile of one isolation window, the unit that NMF is fit on.

A GPFRun (all MS2 scans of one isolation target) is cut into overlapping tiles of
``scan_width`` scans. Each tile is a ScanWindow holding a scans x m/z-bins intensity
matrix plus everything learned about it: the filtered matrix, the NMF factors W/H,
Gaussian fits of each component's elution, which components survive deduplication,
attached MS1 features and the MS1-MS2 matching results.
"""

import copy
import logging
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np
import scipy.ndimage as ndi
import scipy.sparse as sps
from sklearn.decomposition import NMF

from ..storage import create_dataset
from .ms1feature import (
    MS1Feature,
    dump_ms1_feature_table,
    load_ms1_feature_table,
)

logger = logging.getLogger(__name__)


class ScanWindow:
    """A class representing a window of mass spectrometry scans for a specific GPF (Gas Phase Fraction) value.

    This class handles the processing and filtering of mass spectrometry data within a specific scan window,
    including operations like denoising, scaling, and filtering of m/z values.

    Lifecycle (the order the pipeline calls things in):

    1. Built by ``GPFRun.slice_scan_window`` from rows ``[start, start + scan_width)``
       of the GPF's sparse scans x bins matrix.
    2. ``filter_scans`` cleans the matrix (drops bins, scales, denoises).
    3. ``set_nmf_fit`` stores the NMF factors: ``_w`` (scans x components, elution
       profiles) and ``_h`` (components x kept bins, spectra).
    4. ``set_peak_fits`` stores a Gaussian (mean, sigma, error) per component and drops
       components that failed to fit or sit at the clipped edges.
    5. ``GPFRun.collapse_redundant_components`` calls ``mask_component`` on duplicates
       found in overlapping windows.
    6. ``set_ms1_features`` attaches MS1 features whose isotopes fall in the isolation
       window, interpolated onto this window's retention times.
    7. ``set_ms1_ms2_feature_matches`` stores the regression of each component's
       elution profile onto the MS1 feature profiles.
    8. ``dump_h5``/``from_h5`` save and restore all of the above (lazy mode reloads
       windows from HDF5 between stages).

    How the m/z arrays and masks relate:

    - ``_mz_bin_indices_unfilter``/``_mz_unfilter`` list every bin of the GPF (the
      columns of ``_m_unfilter``). ``_mz_masked`` is aligned with them and is True
      for bins removed by filtering.
    - ``_mz_bin_indices``/``_mz`` list the bins that survived filtering, i.e. the
      columns of ``_m`` and ``_h``. They equal the unfiltered arrays at
      ``~_mz_masked`` (bin indices are sorted, so ``np.searchsorted`` maps between
      them).
    - ``_m_max`` holds the per-bin maximum used for max-scaling and
      ``_m_max_bin_indices`` the bins it was computed for (bins still present at
      scaling time). ``reverse_transform_maxscale_scans`` looks up the current bins
      in it to turn H back into intensity units.

    How components are selected:

    - ``_non_zero_components``: NMF columns of W that aren't all zero.
    - ``_component_keep``: starts equal to ``_non_zero_components``. Set False for
      failed/edge Gaussian fits and for duplicates masked during collapse.
    - ``component_indices`` = where both are True. The ``w``, ``h``,
      ``component_names``, ``component_fit_parameters`` and ``component_maes``
      properties all return only these components. The underscore arrays keep all
      ``n_components`` columns/rows, indexed by the original component number.

    Attributes:
        _scan_number (np.ndarray): Array of scan numbers.
        _retention_time (np.ndarray): Array of retention times for each scan.
        _gpf (float): Gas Phase Fractionation value.
        _mz_masked (np.ndarray): Boolean array indicating filtered m/z windows.
        _mz_unfilter (np.ndarray): Unfiltered m/z values.
        _mz (np.ndarray): Filtered m/z values.
        _mz_bin_indices_unfilter (np.ndarray): Unfiltered bin indices.
        _mz_bin_indices (np.ndarray): Filtered bin indices.
        _m_unfilter (scipy.sparse.csr_matrix): Original sparse intensity matrix.
        _m (np.ndarray): Dense filtered intensity matrix.
        _m_max (np.ndarray): Maximum intensity in each m/z window.
        _denoise_scans (bool): Flag to enable scan denoising.
        _scale_scans (bool): Flag to enable scan scaling.
        _ms1_features (List[MS1Feature]): List of MS1 features associated with this window.
        _component_ms1_coef_matrix (np.ndarray): Matrix of coefficients relating MS1 to MS2 features.
        _component_ms1_global_exp_var (np.ndarray): Global explained variance for MS1-MS2 matches.
        _component_ms1_individual_exp_var (np.ndarray): Individual explained variance for each MS1-MS2 match.
        _is_ms1_features_fit (bool): Whether MS1 features have been fit to MS2 components.
    """

    # Processing switches for filter_scans. These are class-level defaults, not part of
    # Config, but they are saved per window in h5 and restored by from_h5.
    _filter_edge_nscans = 5
    _denoise_scans = True
    _scale_scans = True
    _percentile_filter_scans = True
    _percentile_filter = 10
    _log_scans = False
    _filter_edge_scans = True
    _is_filtered = False
    _downcast_h5 = True
    _downcast_scans = True
    _downcast_bins = True

    # NMF results (set_nmf_fit).
    _w = None
    _h = None
    _nmf_reconstruction_error = None
    _non_zero_components = None
    _component_keep = None
    _is_fit = False

    # Gaussian fits of each component's elution profile (set_peak_fits).
    _component_means = None
    _component_sigmas = None
    _component_maes = None
    _component_names = None
    _is_component_fit = False

    # MS1 features (set_ms1_features) and MS1-MS2 matches
    # (set_ms1_ms2_feature_matches).
    _ms1_features = None
    _component_ms1_coef_matrix = None
    _component_ms1_global_exp_var = None
    _component_ms1_individual_exp_var = None
    _is_ms1_features_fit = False

    def __init__(
        self,
        gpf: float,
        m: sps.csr_matrix,
        scan_number: np.ndarray,
        scan_index: np.ndarray,
        bin_indices: np.ndarray,
        retention_time: np.ndarray,
        mz: np.ndarray,
        gpf_low: Optional[float] = None,
        gpf_high: Optional[float] = None,
    ):
        """Initialize a ScanWindow instance.

        Nothing is filtered yet: the filtered copies (``_m``, ``_mz``,
        ``_mz_bin_indices``) start equal to the unfiltered inputs and ``_mz_masked``
        is all False.

        Args:
            gpf (float): Gas Phase Fractionation value (isolation window target m/z).
            m (scipy.sparse.csr_matrix): Sparse intensity matrix, scans x m/z bins.
            scan_number (np.ndarray): Scan number of each row.
            scan_index (np.ndarray): Index of each row within the GPF run (0-based).
            bin_indices (np.ndarray): Global m/z bin index of each column (sorted).
            retention_time (np.ndarray): Retention time of each row.
            mz (np.ndarray): m/z value (bin right edge) of each column.
            gpf_low (Optional[float]): Lower isolation bound, if known.
            gpf_high (Optional[float]): Upper isolation bound, if known.
        """
        self._scan_number = scan_number
        self._scan_index = scan_index
        self._retention_time = retention_time
        self._gpf = gpf
        self._gpf_low = gpf_low
        self._gpf_high = gpf_high
        self._mz_masked = np.zeros_like(bin_indices, dtype=bool)
        self._mz_unfilter = mz
        self._mz = mz.copy()
        self._mz_bin_indices_unfilter = bin_indices
        self._mz_bin_indices = bin_indices.copy()
        self._m_unfilter = m
        self._m = m.copy()
        self._m_max = None
        self._m_max_bin_indices = None

    def __copy__(self):
        """Create a deep copy of the ScanWindow instance.

        Returns:
            ScanWindow: A new instance with deep copies of all attributes.
        """
        cls = self.__class__
        new_instance = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(new_instance, k, copy.deepcopy(v))
        return new_instance

    @property
    def m(self) -> np.ndarray:
        """Get the intensity matrix as a dense numpy array.

        Returns:
            np.ndarray: Dense intensity matrix.
        """
        if not isinstance(self._m, np.ndarray):
            return self._m.toarray()
        return self._m

    @property
    def retention_time(self) -> np.ndarray:
        """Get the retention time array.

        Returns:
            np.ndarray: Array of retention times.
        """
        return self._retention_time

    @property
    def scan_number(self) -> np.ndarray:
        """Get the scan number array.

        Returns:
            np.ndarray: Array of scan numbers.
        """
        return self._scan_number

    @property
    def scan_index(self) -> np.ndarray:
        """Get the scan index array.

        Returns:
            np.ndarray: Array of scan indices.
        """
        return self._scan_index

    @property
    def mz(self) -> np.ndarray:
        """Get the m/z for the m/z bins.

        Returns:
            np.ndarray: Array of m/z bins.
        """
        return self._mz

    @property
    def mz_bin_indices(self) -> np.ndarray:
        """Get the m/z bin indices array.

        Returns:
            np.ndarray: Array of m/z bin indices.
        """
        return self._mz_bin_indices

    @property
    def w(self) -> np.ndarray:
        """Get the NMF basis matrix for kept components.

        Returns:
            np.ndarray: NMF basis matrix

        Raises:
            AttributeError: If NMF decomposition has not been performed
        """
        if not getattr(self, "_is_fit", False):
            raise AttributeError("ScanWindow not fit.")
        return self._w[:, self.component_indices]

    @property
    def h(self) -> np.ndarray:
        """Get the NMF coefficient matrix for kept components.

        Returns:
            np.ndarray: NMF coefficient matrix

        Raises:
            AttributeError: If NMF decomposition has not been performed
        """
        if not getattr(self, "_is_fit", False):
            raise AttributeError("ScanWindow not fit.")
        return self._h[self.component_indices, :]

    @property
    def component_indices(self) -> np.ndarray:
        """Get indices of non-zero components that are kept.

        Returns:
            np.ndarray: Array of component indices

        Raises:
            AttributeError: If NMF decomposition has not been performed
        """
        if not getattr(self, "_is_fit", False):
            raise AttributeError("ScanWindow not fit.")
        return np.where(self._non_zero_components & self._component_keep)[0]

    @property
    def component_maes(self) -> np.ndarray:
        """Get mean absolute errors of peak fits for kept components.

        Returns:
            np.ndarray: Array of mean absolute errors

        Raises:
            AttributeError: If peak fitting has not been performed
        """
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError("Need to fit gaussians.")
        return self._component_maes[self.component_indices]

    @property
    def component_fit_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the fitted peak parameters (means and sigmas) for kept components.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Array of peak means/centers
                - Array of peak standard deviations/widths

        Raises:
            AttributeError: If peak fitting has not been performed
        """
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError("Need to fit gaussians.")
        return (
            self._component_means[self.component_indices],
            self._component_sigmas[self.component_indices],
        )

    @property
    def component_names(self) -> np.ndarray:
        """Get the names of kept components.

        Returns:
            np.ndarray: Array of component names

        Raises:
            AttributeError: If peak fitting has not been performed
        """
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError("Need to fit gaussians.")
        return self._component_names[self.component_indices]

    @property
    def is_filtered(self) -> bool:
        """Check if scan data has been filtered.

        Returns:
            bool: True if filtering has been applied, False otherwise
        """
        return self._is_filtered

    @property
    def is_fit(self) -> bool:
        """Check if NMF decomposition has been performed.

        Returns:
            bool: True if NMF has been fit, False otherwise
        """
        return self._is_fit

    @property
    def is_component_fit(self) -> bool:
        """Check if peak fitting has been performed on components.

        Returns:
            bool: True if components have been fit, False otherwise
        """
        return self._is_component_fit

    @property
    def ms1_features(self) -> List["MS1Feature"]:
        """Get the list of associated MS1 features.

        Returns:
            List[MS1Feature]: List of MS1 features
        """
        return self._ms1_features

    @property
    def ms1_features_information(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get interpolated intensities and IDs for all MS1 features.

        Each column is one feature's hill interpolated onto this window's retention
        times (see ``MS1Feature.interpolate_onto_ms2``). This is the design matrix that
        MS1-MS2 matching regresses each component's elution profile onto.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - 2D array of interpolated intensities (scans × features)
                - Array of feature IDs

        Raises:
            AttributeError: If no MS1 features have been added
        """
        if self._ms1_features is None:
            raise AttributeError("No ms1 features added to this scanwindow.")

        # No features attached: return an empty matrix and no IDs. Note the transpose
        # gives shape (0, n_scans), not (n_scans, 0); matching still works because
        # every feature mask built from it ends up empty.
        if len(self._ms1_features) == 0:
            return (
                np.empty(
                    (
                        self.m.shape[0],
                        0,
                    )
                ).T,
                np.array([]),
            )

        # Stack the per-feature profiles as columns: scans x features.
        ms1_elution = np.vstack(
            [f._interpolated_intensity for f in self._ms1_features]
        ).T

        ms1_feature_name = np.array([f._id for f in self._ms1_features])

        return (ms1_elution, ms1_feature_name)

    @property
    def is_ms1_features_fit(self) -> bool:
        """Check if MS1 features have been matched to MS2 components.

        Returns:
            bool: True if MS1-MS2 matching has been performed, False otherwise
        """
        return self._is_ms1_features_fit

    def mask_component(self, component_index) -> None:
        """Mask/exclude a component from further analysis.

        Used by ``GPFRun.collapse_redundant_components`` to drop a duplicate of a
        component found in an overlapping window. The component stays in ``_w``/``_h``
        but disappears from ``component_indices`` and every property built on it.

        Args:
            component_index (int): Original component number (column of ``_w``), not a
                position within ``component_indices``.

        Raises:
            AttributeError: If peak fitting has not been performed
        """
        if not getattr(self, "_is_component_fit", False):
            raise AttributeError
        self._component_keep[component_index] = False

    def filter_scans(self) -> None:
        """Apply all configured filtering operations to scan data.

        This includes:
        - Removing zero m/z values
        - Clipping constant signals
        - Percentile filtering (if enabled)
        - Log transformation (if enabled)
        - Max scaling (if enabled)
        - Denoising (if enabled)
        - Edge scan filtering (if enabled)
        - Downcasting data types (if enabled)

        Every step that can empty a bin ends with ``filter_zero_mz``, so ``_m`` only
        ever has columns with some signal. ``_m`` keeps its storage type: a window
        sliced from a GPFRun stays a CSR matrix (denoising densifies temporarily and
        converts back), and the ``m`` property returns it dense for NMF.
        """
        # 1. Drop bins with no signal in this window.
        self.filter_zero_mz()
        # 2. Drop bins whose signal is roughly constant over the window (background,
        #    contaminants): they carry no elution information.
        self.filter_clip_constant()
        # 3. Zero out the weakest values (below a percentile of non-zero intensities).
        if self._percentile_filter_scans:
            self.filter_percentile_scans()
        # 4. Optional log transform (off by default).
        if self._log_scans:
            self.transform_log_scans()
        # 5. Scale each bin to max 1, so weak and strong fragments weigh alike in NMF.
        if self._scale_scans:
            self.transform_maxscale_scans()
        # 6. Morphological denoising along time to remove single-scan spikes.
        if self._denoise_scans:
            self.filter_denoise_scans()
        # 7. Zero the first/last few scans; components peaking there are cut off.
        if self._filter_edge_scans:
            self.filter_edge_scans()
        # 8. Downcast intensities.
        if self._downcast_scans:
            # Push through downcast to make results reproducible if ran through h5
            # intermediate or not. dump_h5 stores the matrix as float16, so rounding
            # through float16 here makes in-memory and reloaded runs see the same
            # values.
            if self._downcast_h5:
                self._m = self._m.astype(np.float16)
            self._m = self._m.astype(np.float32)
        if self._downcast_bins:
            # These are pretty safe options for most scenarios. Need to revisit if for whatever reason
            # int32 insufficient for bin_indices or float32 insufficient for mz.
            self._mz = self._mz.astype(np.float32)
            self._mz_unfilter = self._mz_unfilter.astype(np.float32)
            self._mz_bin_indices = self._mz_bin_indices.astype(np.int32)
            self._mz_bin_indices_unfilter = self._mz_bin_indices_unfilter.astype(
                np.int32
            )
        self._is_filtered = True

    def filter_mz_axis(self, mask: np.ndarray) -> None:
        """Filter the m/z axis using a boolean mask.

        Removes columns from the filtered arrays (``_m``, ``_mz``, ``_mz_bin_indices``)
        and records the removed bins in ``_mz_masked``, which is aligned with the
        unfiltered bins.

        Args:
            mask (np.ndarray): Boolean mask over the *current* filtered columns
                indicating which m/z values to keep.
        """
        # Map the removed bins back to their positions in the unfiltered bin list
        # (both lists are sorted) and flag them.
        self._mz_masked[
            np.searchsorted(
                self._mz_bin_indices_unfilter,
                self._mz_bin_indices[~mask],
            )
        ] = True
        self._mz = self._mz[mask]
        self._mz_bin_indices = self._mz_bin_indices[mask]
        self._m = self._m[:, mask]

    def filter_edge_scans(self) -> None:
        """Filter out edge scans by setting their intensities to zero.

        Uses _filter_edge_nscans to determine number of scans to filter. Overlapping
        windows cover these scans in their interior, so nothing is lost; it stops NMF
        from fitting peaks that are cut off at the window edge. ``set_peak_fits`` later
        drops components whose fitted peak reaches into this zone.
        """
        # Warnings silenced: assigning into a CSR matrix raises
        # SparseEfficiencyWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._m[: self._filter_edge_nscans, :] = 0.0
            self._m[-1 * self._filter_edge_nscans :, :] = 0.0
        self.filter_zero_mz()

    def filter_zero_mz(self) -> None:
        """Remove m/z values that have zero intensity across all scans."""
        mask = (self._m.sum(axis=0) > 0.0).flatten()
        # Column sums of a sparse matrix come back as a 1 x n np.matrix.
        if isinstance(mask, np.matrix):
            mask = np.array(mask).flatten()
        self.filter_mz_axis(mask)

    def transform_maxscale_scans(
        self,
        axis: int = 0,
        robust_scaling: bool = False,
        robust_scaling_percentile: float = 99.5,
    ) -> None:
        """Scale intensities by maximum value along specified axis.

        With the default ``axis=0`` each m/z bin is divided by its maximum over the
        window, so every fragment trace peaks at 1. Without this, NMF would mostly fit
        the few most intense fragments. The factors are kept in ``_m_max`` (with the
        bins they belong to in ``_m_max_bin_indices``) so exported spectra can be put
        back into intensity units (``reverse_transform_maxscale_scans``).

        Args:
            axis (int): Axis along which to find maximum (0=scans, 1=m/z)
            robust_scaling (bool): Use percentile instead of maximum if True
            robust_scaling_percentile (float): Percentile to use if robust_scaling is True
        """
        if not robust_scaling:
            m_m = self._m.max(axis=axis)
        else:
            m_m = np.percentile(
                self._m,
                robust_scaling_percentile,
                axis=axis,
                method="lower",
            )
        # Sparse max() returns a sparse 1 x n matrix; flatten it to a 1-D array.
        if not isinstance(m_m, np.ndarray):
            m_m = m_m.toarray().flatten()
        # Row-wise scaling needs a column vector to broadcast.
        if axis == 1:
            m_m = m_m[..., np.newaxis]
        # Remember the factors and the bins they were computed for, since later
        # filters can still remove bins.
        self._m_max = m_m
        self._m_max_bin_indices = self._mz_bin_indices.copy()
        # Sparse matrices divide via multiply() to stay sparse.
        if not isinstance(self._m, (np.ndarray, np.matrix)):
            self._m = self._m.multiply(1.0 / m_m)
        else:
            self._m = self._m / m_m

    def transform_log_scans(self) -> None:
        """Apply log1p transform to scan intensities."""
        self._m = np.log1p(self._m)

    def filter_percentile_scans(self) -> None:
        """Filter out low intensity values below specified percentile threshold.

        The threshold is the ``_percentile_filter`` percentile (default 10) of the
        non-zero intensities in the window. Values below it are set to zero, then
        bins left empty are dropped.
        """
        # Percentile over non-zero values only (a sparse matrix's .data already
        # holds just the stored non-zeros).
        if isinstance(self._m, np.ndarray):
            threshold_signal = np.percentile(
                self._m.flatten()[self._m.flatten() > 0],
                self._percentile_filter,
            )
        else:
            threshold_signal = np.percentile(self._m.data, self._percentile_filter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(self._m, np.ndarray):
                self._m[self._m < threshold_signal] = 0.0
            else:
                # Edit the stored values directly. Boolean-mask assignment on a sparse
                # matrix inserts entries one at a time and re-sorts the indices, which
                # made this the slowest step of building the run.
                self._m = self._m.tocsr()
                self._m.data[self._m.data < threshold_signal] = 0.0
                self._m.eliminate_zeros()
        self.filter_zero_mz()

    def filter_clip_constant(self, p: float = 0.3) -> None:
        """Remove m/z values with constant signal above threshold.

        Each bin is divided by its own maximum and averaged over the window's scans.
        A real eluting fragment is near zero most of the time, so its mean is low. A
        bin whose mean exceeds ``p`` is present at a large fraction of its maximum
        across the window (background ions, contaminants) and is removed.

        Args:
            p (float): Threshold for mean normalized intensity
        """
        # Keep bins with mean(normalized intensity) <= p.
        if isinstance(self._m, np.ndarray):
            mask = ~(((self._m / self._m.max(axis=0)).mean(axis=0)) > p)
        else:
            mask = np.array(
                ~(((self._m / self._m.max(axis=0).toarray()).mean(axis=0)) > p)
            ).flatten()
        self.filter_mz_axis(mask)

    def filter_denoise_scans(
        self,
        grey_erosion_size: int = 2,
        grey_closing_size: int = 2,
    ) -> np.ndarray:
        """Apply operations to denoise scan data.

        Works along the time axis only (structuring elements are ``(size, 1)``, so
        each m/z bin is treated separately):

        - grey closing (dilation then erosion) fills one-scan dropouts inside a peak;
        - grey erosion then removes features narrower than the element, such as
          single-scan noise spikes.

        Updates ``_m`` in place; the docstring's return value is not used (the method
        returns None).

        Args:
            grey_erosion_size (int): Size of structuring element for grey erosion
            grey_closing_size (int): Size of structuring element for grey closing

        Returns:
            np.ndarray: Denoised scan data
        """
        # scipy.ndimage needs a dense array; convert back afterwards if sparse.
        is_sparse = False
        if not isinstance(self._m, (np.ndarray, np.matrix)):
            is_sparse = True
            self._m = self._m.toarray()
        self._m = ndi.grey_erosion(
            ndi.grey_closing(
                self._m,
                size=(grey_closing_size, 1),
            ),
            size=(grey_erosion_size, 1),
        )
        self.filter_zero_mz()
        if is_sparse:
            self._m = sps.csr_matrix(self._m)

    def reverse_transform_maxscale_scans(self) -> np.ndarray:
        """Reverse the max scaling transformation previously applied to scans.

        Looks up the scaling factor of every current (filtered) bin. Multiplying a row
        of ``h`` by the result gives the component spectrum in intensity units.

        Returns:
            np.ndarray: Array of scale factors to reverse the transformation, aligned
                with ``mz_bin_indices``

        Raises:
            AttributeError: If max scale factors were not previously computed
        """
        if not isinstance(self._m_max, np.ndarray):
            raise AttributeError

        scale_indices = np.searchsorted(
            self._m_max_bin_indices,
            self._mz_bin_indices,
        )

        return self._m_max[scale_indices]

    def unfiltered_with_filter(self, scale_if_exists: bool = True) -> np.ndarray:
        """Get unfiltered data with filtering mask applied.

        Returns the original intensities (before percentile filtering, denoising and
        edge zeroing) for the bins that survived filtering, optionally max-scaled
        with the same factors as ``_m``. This needs ``_m_unfilter`` to still span all
        GPF bins, which isn't the case after reloading from h5 unless
        ``scan_window_save_m_unfilter`` was set.

        Args:
            scale_if_exists (bool): Whether to apply scaling if max scale factors exist

        Returns:
            np.ndarray: Filtered unfiltered data matrix
        """
        # After a reload without m_unfilter, _m_unfilter is a copy of the filtered
        # matrix and no longer lines up with _mz_masked.
        if self._m_unfilter.shape[1] != self._mz_masked.shape[0]:
            raise ValueError(
                "Unfiltered matrix not available (it isn't saved to h5 by default)."
            )
        m = self._m_unfilter[:, ~self._mz_masked]
        if not isinstance(m, np.ndarray):
            m = m.toarray()
        if scale_if_exists and self._m_max is not None:
            scale_indices = np.searchsorted(
                self._m_max_bin_indices,
                self._mz_bin_indices,
            )
            return m / self._m_max[scale_indices]
        return m

    def set_nmf_fit(self, w: np.ndarray, h: np.ndarray, model: NMF) -> None:
        """Set NMF decomposition results.

        A refit replaces any previous fit. Peak fits and MS1 matches belong to the
        previous components, so they are reset and need to be recomputed.

        Args:
            w (np.ndarray): NMF basis matrix, scans x components (elution profiles)
            h (np.ndarray): NMF coefficient matrix, components x filtered bins
                (component spectra in max-scaled units)
            model (NMF): Fitted NMF model object containing reconstruction error
        """
        # Results derived from the old components no longer apply.
        if self.is_fit:
            self._component_means = None
            self._component_sigmas = None
            self._component_maes = None
            self._component_names = None
            self._is_component_fit = False
            self._component_ms1_coef_matrix = None
            self._component_ms1_global_exp_var = None
            self._component_ms1_individual_exp_var = None
            self._is_ms1_features_fit = False

        self._w = w.copy()
        self._h = h.copy()
        self._nmf_reconstruction_error = model.reconstruction_err_
        # Regularization can switch components off entirely (all-zero W column);
        # only the non-zero ones are kept.
        self._non_zero_components = self._w.sum(axis=0) != 0.0
        self._component_keep = np.zeros(self._w.shape[1], dtype=bool)
        self._component_keep[self._non_zero_components] = True
        self._is_fit = True

    def set_peak_fits(
        self, m: np.ndarray, s: np.ndarray, maes: np.ndarray, keep: np.ndarray
    ) -> None:
        """Set peak fitting results for components.

        The inputs are aligned with the non-zero components (the order
        ``fit_scanwindow`` iterates in) and are scattered into arrays over all
        components. Components that failed to fit, or whose peak (mean +/- 2 sigma)
        reaches into the zeroed edge scans, are marked not kept. Also builds each
        component's name.

        Args:
            m (np.ndarray): Peak means/centers (retention time units)
            s (np.ndarray): Peak standard deviations/widths (retention time units)
            maes (np.ndarray): Mean absolute errors of peak fits
            keep (np.ndarray): Boolean mask of peaks to keep
        """
        # Scatter per-non-zero-component results into full-length arrays; zero-weight
        # components get mean/sigma/mae 0.
        self._component_keep[self._non_zero_components] = keep.copy()
        self._component_means = np.zeros(self._w.shape[1])
        self._component_means[self._non_zero_components] = m.copy()
        self._component_sigmas = np.zeros(self._w.shape[1])
        self._component_sigmas[self._non_zero_components] = s.copy()
        self._component_maes = np.zeros(self._w.shape[1])
        self._component_maes[self._non_zero_components] = maes.copy()

        # Filter fits that are within 2s of clipped edge. Those peaks are truncated in
        # this window; an overlapping window sees them whole.
        if self._filter_edge_scans:
            mask = (
                self.retention_time[self._filter_edge_nscans]
                > (self._component_means - 2 * self._component_sigmas)
            ) | (
                self.retention_time[-1 * self._filter_edge_nscans]
                < (self._component_means + 2 * self._component_sigmas)
            )
            self._component_keep[mask] = False
        # Name = "<gpf>_<component>-<scan index nearest the apex>_<mean>_<sigma>",
        # unique within a run. Built for every component, kept or not.
        self._component_names = np.array(
            [
                (
                    f"{self._gpf}_{i}-"
                    f"{self._scan_index[np.argmin(np.absolute(self._retention_time-self._component_means[i]))]}_"
                    f"{self._component_means[i]}_"
                    f"{self._component_sigmas[i]}"
                )
                for i in range(self._w.shape[1])
            ]
        )
        self._is_component_fit = True

    def query_component_info(
        self, value: str, by: Literal["name"] = "name"
    ) -> List[Dict[str, Any]]:
        """Query information about components.

        Not implemented yet: returns one empty dict per matching component.

        Args:
            value (str): Value to query for
            by (Literal["name"]): Field to query by, currently only supports "name"

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing component information
        """
        matching_components = []

        if by == "name":
            idxes = np.where(self.component_names == value)[0]
            for idx in idxes:
                matching_components.append(
                    {
                        # TODO implement stuff here
                    }
                )

        return matching_components

    def set_ms1_features(self, ms1_features: List[MS1Feature]) -> None:
        """Set MS1 features associated with this scan window.

        Replaces any previous list. Features are expected to be interpolated onto
        this window's retention times already (``add_ms1_features_scanwindow`` does
        this).

        Args:
            ms1_features (List[MS1Feature]): List of MS1 features to associate
        """
        self._ms1_features = ms1_features

    def set_ms1_ms2_feature_matches(
        self,
        coef_matrix: np.ndarray,
        global_variance_explained: np.ndarray,
        individual_variance_explained: np.ndarray,
    ) -> None:
        """Set results from matching MS1 features to MS2 components.

        Rows follow ``component_indices`` (kept components at matching time) and
        columns follow ``ms1_features``.

        Args:
            coef_matrix (np.ndarray): Matrix of coefficients relating MS1 to MS2 features
                (kept components x MS1 features)
            global_variance_explained (np.ndarray): Global explained variance for MS1-MS2 matches
                (one value per kept component)
            individual_variance_explained (np.ndarray): Individual explained variance for each match
                (kept components x MS1 features)
        """
        self._component_ms1_coef_matrix = coef_matrix.copy()
        self._component_ms1_global_exp_var = global_variance_explained.copy()
        self._component_ms1_individual_exp_var = individual_variance_explained.copy()
        self._is_ms1_features_fit = True

    def dump_h5(
        self,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "scan_window",
        save_m_unfilter: bool = True,
        mode: Literal["write", "update"] = "write",
        compact: bool = False,
        compression: Optional[str] = None,
    ) -> None:
        """Save the ScanWindow object to an HDF5 file or group.

        Layout of the group: scalar flags and bounds as attributes; index/m/z arrays,
        the filtered matrix (``m`` dense, or ``m_data``/``m_indices``/``m_indptr`` +
        ``m_shape`` attribute when sparse), NMF results, peak fits, MS1 features (one
        subgroup each under ``ms1_features``) and MS1-MS2 matches as datasets.

        In "update" mode (used by lazy MSRuns after every pipeline stage) the existing
        group is reused and each dataset is only rewritten when its contents changed,
        which avoids rewriting the large matrices on every stage.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group to store the object
            save_m_unfilter (bool): Whether to save unfiltered matrix data
            mode (Literal["write", "update"]): Whether to overwrite ("write") or update existing data
            compact (bool): Don't store the unfiltered m/z axis (``mz_unfilter``,
                ``mz_bin_indices_unfilter``, ``mz_masked``). It is the parent GPFRun's
                bin axis, so ``from_h5`` rebuilds it exactly from the GPFRun. The group
                is marked with the ``compact_axes`` attribute.
            compression (Optional[str]): h5py compression filter for the datasets
                (lossless), or None.
        """
        # Open (or create) the target group. "update" reuses an existing group.
        if isinstance(h5file, str):
            f = h5py.File(h5file, "r+" if mode == "update" else "w")
            if mode == "update" and dataset_name in f:
                grp = f[dataset_name]
            else:
                grp = f.create_group(dataset_name)
            close_file = True
        else:
            if mode == "update" and dataset_name in h5file:
                grp = h5file[dataset_name]
            else:
                grp = h5file.create_group(dataset_name)
            close_file = False

        def check_arrays_equal(a1: np.ndarray, a2: np.ndarray) -> bool:
            """Check if two arrays are equal, handling None values.

            Args:
                a1 (np.ndarray): First array
                a2 (np.ndarray): Second array

            Returns:
                bool: True if arrays are equal or both None, False otherwise
            """
            if a1 is None or a2 is None:
                return a1 is a2
            return np.array_equal(a1, a2)

        def save_dataset(
            name: str, data: Union[np.ndarray, sps.csr_matrix, None]
        ) -> None:
            """Save a dataset to the HDF5 group.

            Args:
                name (str): Name of the dataset
                data (Union[np.ndarray, sps.csr_matrix, None]): Data to save

            The function handles:
            - Sparse and dense arrays
            - None values (deletes existing dataset)
            - Updating existing datasets only if changed
            """
            if data is None:
                if name in grp:
                    del grp[name]
                return

            if isinstance(data, sps.csr_matrix):
                # For sparse matrices
                if mode == "update":
                    # Stored values are float16 (see below), so compare after the
                    # same cast. filter_scans already rounds _m through float16, so
                    # an unchanged matrix compares equal and isn't rewritten.
                    data_changed = (
                        f"{name}_data" not in grp
                        or not np.array_equal(
                            data.data, grp[f"{name}_data"][:].astype(np.float32)
                        )
                        or not np.array_equal(data.indices, grp[f"{name}_indices"][:])
                        or not np.array_equal(data.indptr, grp[f"{name}_indptr"][:])
                        or not np.array_equal(
                            data.shape, grp.attrs.get(f"{name}_shape", None)
                        )
                    )
                    if not data_changed:
                        return

                # Delete existing datasets if they exist
                for suffix in ["_data", "_indices", "_indptr"]:
                    if f"{name}{suffix}" in grp:
                        del grp[f"{name}{suffix}"]

                # Create new datasets. Values are stored as float16 to halve the file
                # size when _downcast_h5 is set.
                create_dataset(
                    grp,
                    f"{name}_data",
                    (
                        data.data
                        if not self._downcast_h5
                        else data.data.astype(np.float16)
                    ),
                    compression=compression,
                )
                create_dataset(
                    grp, f"{name}_indices", data.indices, compression=compression
                )
                create_dataset(
                    grp, f"{name}_indptr", data.indptr, compression=compression
                )
                grp.attrs[f"{name}_shape"] = data.shape
            else:
                # For regular arrays (dense arrays are stored at full precision)
                if mode == "update":
                    if name in grp and check_arrays_equal(data, grp[name][:]):
                        return
                if name in grp:
                    del grp[name]
                create_dataset(grp, name, data, compression=compression)

        # Save/update scalar attributes
        scalar_attrs = (
            "gpf",
            "gpf_low",
            "gpf_high",
            "is_filtered",
            "is_fit",
            "is_component_fit",
        )

        for sa in scalar_attrs:
            v = getattr(self, f"_{sa}", -1)
            # h5py can't store None, from_h5 maps -1 back to None.
            v = -1 if v is None else v
            if mode == "write" or sa not in grp.attrs or grp.attrs[sa] != v:
                grp.attrs[sa] = v

        # Save arrays
        array_datasets = (
            "scan_number",
            "scan_index",
            "retention_time",
            "mz_masked",
            "mz_unfilter",
            "mz",
            "mz_bin_indices_unfilter",
            "mz_bin_indices",
        )

        # The unfiltered axis of a window is its GPFRun's full bin axis, the same for
        # every window of the GPF. Compact mode stores only the dtypes needed to
        # rebuild it exactly (see from_h5); otherwise every window keeps a copy.
        gpf_axis_datasets = ("mz_masked", "mz_unfilter", "mz_bin_indices_unfilter")
        grp.attrs["compact_axes"] = compact
        if compact:
            grp.attrs["mz_unfilter_dtype"] = str(self._mz_unfilter.dtype)
            grp.attrs["mz_bin_indices_unfilter_dtype"] = str(
                self._mz_bin_indices_unfilter.dtype
            )

        for ds in array_datasets:
            if compact and ds in gpf_axis_datasets:
                save_dataset(ds, None)  # Removes a copy left by a non-compact write.
                continue
            if hasattr(self, f"_{ds}"):
                save_dataset(ds, getattr(self, f"_{ds}", None))

        # Save sparse intensity matrix, or if dense. m_is_sparse tells from_h5 which
        # layout to read back.
        if isinstance(self._m, (np.ndarray, np.matrix)):
            save_dataset("m", self._m)
            grp.attrs["m_is_sparse"] = False
        else:
            save_dataset("m", self._m)
            grp.attrs["m_is_sparse"] = True

        if save_m_unfilter:
            # Save unfiltered sparse matrix. Off by default (h5.scan_window_save_
            # m_unfilter) because it holds every GPF bin for every window.
            save_dataset("m_unfilter", self._m_unfilter)

        # Save NMF results if they exist. W/H keep all components; component_keep
        # carries the collapse masks.
        if self._is_fit and hasattr(self, "_w"):
            nmf_arrays = (
                "w",
                "h",
                "component_keep",
                "non_zero_components",
            )
            for na in nmf_arrays:
                if hasattr(self, f"_{na}"):
                    save_dataset(na, getattr(self, f"_{na}"))
            if hasattr(self, "_nmf_reconstruction_error"):
                grp.attrs["nmf_reconstruction_error"] = self._nmf_reconstruction_error

        # Save component fits if they exist
        if self._is_component_fit:
            component_arrays = (
                "component_means",
                "component_sigmas",
                "component_maes",
            )
            for ca in component_arrays:
                if hasattr(self, f"_{ca}"):
                    save_dataset(ca, getattr(self, f"_{ca}"))
            if hasattr(self, "_component_names") and self._component_names is not None:
                save_dataset("component_names", self._component_names.astype("S"))

        # Need to check and update MS1 features here.
        # Save MS1 features if they exist
        if self._ms1_features is not None:
            # If the group exists, delete group and re-add (features aren't compared,
            # so they are rewritten on every dump).
            for old in ("ms1_features", "ms1_feature_table"):
                if old in grp:
                    del grp[old]

            # Compact storage keeps all features in one table (few HDF5 objects);
            # it falls back to one group per feature if that can't be exact.
            stored_as_table = compact and dump_ms1_feature_table(
                grp,
                "ms1_feature_table",
                self._ms1_features,
                self._retention_time,
                compression=compression,
            )
            if not stored_as_table:
                ms1_grp = grp.create_group("ms1_features")
                for i, feature in enumerate(self._ms1_features):
                    feature.dump_h5(ms1_grp, f"feature_{i}")

        if self._is_ms1_features_fit:
            save_dataset("component_ms1_coef_matrix", self._component_ms1_coef_matrix)
            save_dataset(
                "component_ms1_global_exp_var", self._component_ms1_global_exp_var
            )
            save_dataset(
                "component_ms1_individual_exp_var",
                self._component_ms1_individual_exp_var,
            )
            grp.attrs["is_ms1_features_fit"] = self._is_ms1_features_fit

        # Save class-level configuration attributes, so a reloaded window remembers
        # how it was filtered.
        grp.attrs["filter_edge_nscans"] = self._filter_edge_nscans
        grp.attrs["denoise_scans"] = self._denoise_scans
        grp.attrs["scale_scans"] = self._scale_scans
        grp.attrs["percentile_filter_scans"] = self._percentile_filter_scans
        grp.attrs["percentile_filter"] = self._percentile_filter
        grp.attrs["log_scans"] = self._log_scans
        grp.attrs["filter_edge_scans"] = self._filter_edge_scans
        grp.attrs["downcast_h5"] = self._downcast_h5
        grp.attrs["downcast_scans"] = self._downcast_scans
        grp.attrs["downcast_bins"] = self._downcast_bins

        # Save m_max data if it exists (needed to reverse the max-scaling on export).
        if self._m_max is not None:
            save_dataset("m_max", self._m_max)
            save_dataset("m_max_bin_indices", self._m_max_bin_indices)

        # Save component names if they exist
        if self._component_names is not None:
            save_dataset("component_names", self._component_names.astype("S"))

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "scan_window",
        gpf_bin_indices: Optional[np.ndarray] = None,
        gpf_mz: Optional[np.ndarray] = None,
    ) -> "ScanWindow":
        """Load a ScanWindow object from an HDF5 file or group.

        Inverse of ``dump_h5``. If ``m_unfilter`` wasn't saved, ``_m_unfilter`` is set
        to a copy of the filtered matrix (so ``unfiltered_with_filter`` can't be used).

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group containing the object
            gpf_bin_indices (Optional[np.ndarray]): The parent GPFRun's bin axis.
                Required for windows written with ``compact=True``.
            gpf_mz (Optional[np.ndarray]): m/z of each bin in ``gpf_bin_indices``.
                Required for windows written with ``compact=True``.

        Returns:
            ScanWindow: Loaded object

        Raises:
            ValueError: If the window was written in compact mode and the parent
                GPFRun's axis isn't given.
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, "r")
            grp = f[dataset_name]
            close_file = True
        else:
            grp = h5file[dataset_name]
            close_file = False

        # Load sparse matrix (stored as float16, promoted back to float32)
        if grp.attrs["m_is_sparse"]:
            _m = sps.csr_matrix(
                (
                    grp["m_data"][:].astype(np.float32),
                    grp["m_indices"][:],
                    grp["m_indptr"][:],
                ),
                shape=grp.attrs["m_shape"],
            )
        else:
            _m = grp["m"][:].astype(np.float32)

        if "m_unfilter_data" in grp:
            # Load sparse matrix
            m = sps.csr_matrix(
                (
                    grp["m_unfilter_data"][:],
                    grp["m_unfilter_indices"][:],
                    grp["m_unfilter_indptr"][:],
                ),
                shape=grp.attrs["m_unfilter_shape"],
            )
        else:
            m = _m.copy()

        # Create instance. The constructor gets the filtered bins/m/z; the unfiltered
        # arrays and the mask are restored right after.
        obj = cls(
            gpf=grp.attrs["gpf"],
            m=m,
            scan_number=grp["scan_number"][:],
            scan_index=grp["scan_index"][:],
            bin_indices=grp["mz_bin_indices"][:],
            retention_time=grp["retention_time"][:],
            mz=grp["mz"][:],
            gpf_low=None if grp.attrs["gpf_low"] == -1 else grp.attrs["gpf_low"],
            gpf_high=None if grp.attrs["gpf_high"] == -1 else grp.attrs["gpf_high"],
        )

        # Restore the unfiltered axis and the mask of filtered-out bins.
        if grp.attrs.get("compact_axes", False):
            if gpf_bin_indices is None or gpf_mz is None:
                raise ValueError(
                    "Window was stored in compact mode. Load it through GPFRun.from_h5, "
                    "or pass gpf_bin_indices and gpf_mz."
                )
            # The unfiltered axis is the GPF's full axis, cast back to the dtypes the
            # window had when it was written.
            obj._mz_bin_indices_unfilter = np.asarray(gpf_bin_indices).astype(
                grp.attrs["mz_bin_indices_unfilter_dtype"]
            )
            obj._mz_unfilter = np.asarray(gpf_mz).astype(grp.attrs["mz_unfilter_dtype"])
            # filter_mz_axis marks exactly the bins that are no longer in the window.
            obj._mz_masked = ~np.isin(obj._mz_bin_indices_unfilter, obj._mz_bin_indices)
        else:
            obj._mz_masked = grp["mz_masked"][:]
            obj._mz_unfilter = grp["mz_unfilter"][:]
            obj._mz_bin_indices_unfilter = grp["mz_bin_indices_unfilter"][:]

        obj._m = _m
        obj._is_filtered = grp.attrs["is_filtered"]

        # Restore NMF results if they exist
        if grp.attrs["is_fit"]:
            obj._w = grp["w"][:]
            obj._h = grp["h"][:]
            obj._component_keep = grp["component_keep"][:]
            obj._non_zero_components = grp["non_zero_components"][:]
            obj._nmf_reconstruction_error = grp.attrs["nmf_reconstruction_error"]
            obj._is_fit = True

        # Restore component fits if they exist
        if grp.attrs["is_component_fit"]:
            obj._component_means = grp["component_means"][:]
            obj._component_sigmas = grp["component_sigmas"][:]
            obj._component_maes = grp["component_maes"][:]
            obj._is_component_fit = True

        # Load MS1 features if they exist (table layout from compact storage, or one
        # group per feature).
        if "ms1_feature_table" in grp:
            obj._ms1_features = load_ms1_feature_table(
                grp["ms1_feature_table"], obj._retention_time
            )
        elif "ms1_features" in grp:
            obj._ms1_features = []
            ms1_grp = grp["ms1_features"]
            for i in range(len(ms1_grp)):
                feature = MS1Feature.from_h5(ms1_grp, f"feature_{i}")
                obj._ms1_features.append(feature)

        if "is_ms1_features_fit" in grp.attrs and grp.attrs["is_ms1_features_fit"]:
            obj._component_ms1_coef_matrix = grp["component_ms1_coef_matrix"][:]
            obj._component_ms1_global_exp_var = grp["component_ms1_global_exp_var"][:]
            obj._component_ms1_individual_exp_var = grp[
                "component_ms1_individual_exp_var"
            ][:]
            obj._is_ms1_features_fit = grp.attrs["is_ms1_features_fit"]

        # Load class-level configuration attributes (as instance attributes)
        obj._filter_edge_nscans = grp.attrs["filter_edge_nscans"]
        obj._denoise_scans = grp.attrs["denoise_scans"]
        obj._scale_scans = grp.attrs["scale_scans"]
        obj._percentile_filter_scans = grp.attrs["percentile_filter_scans"]
        obj._percentile_filter = grp.attrs["percentile_filter"]
        obj._log_scans = grp.attrs["log_scans"]
        obj._filter_edge_scans = grp.attrs["filter_edge_scans"]
        obj._downcast_h5 = grp.attrs["downcast_h5"]
        obj._downcast_scans = grp.attrs["downcast_scans"]
        obj._downcast_bins = grp.attrs["downcast_bins"]

        # Load m_max data if it exists
        if "m_max" in grp:
            obj._m_max = grp["m_max"][:]
            obj._m_max_bin_indices = grp["m_max_bin_indices"][:]

        # Load component names if they exist
        if "component_names" in grp:
            obj._component_names = grp["component_names"][:].astype("U")

        if close_file:
            f.close()

        return obj
