"""Run configuration (`Config`), INI round-tripping, logging format and constants.

Every tunable setting of the pipeline lives on `Config`, grouped into sections that map
one-to-one onto INI sections (`[nmf]`, `[tuning]`, ...). Top-level settings (`lazy`,
`random_seed`, ...) go in the INI's `[general]` section.
"""

import ast
import configparser
import copy
import logging
from pathlib import Path
from typing import Any, Union


def format_logger(logger: logging.Logger, level: int = logging.INFO) -> None:
    """Configure basic formatting for a logger.

    Applies one `[time] {file:line} LEVEL - message` format to every handler on the
    logger and sets its level.

    Args:
        logger (logging.Logger): Logger instance to configure
        level (int): Logging level to set. Defaults to logging.INFO.
    """
    formatter = logging.Formatter(
        fmt="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    )
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    logger.setLevel(level)


class AttributeDict(dict):
    """Dictionary with attribute-style access.

    Used for the `Config` sections, so both `config.nmf.alpha_W` and
    `config.nmf["alpha_W"]` work, and `**config.nmf` can be passed straight on as
    keyword arguments.
    """

    def __getattr__(self, item: str) -> Any:
        """Return `self[item]`, raising AttributeError (not KeyError) when missing.

        Raising AttributeError keeps `hasattr`, `copy` and `pickle` working.

        Args:
            item (str): Key to look up.

        Returns:
            Any: The stored value.

        Raises:
            AttributeError: If the key doesn't exist.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{item}'")

    def __setattr__(self, key: Any, value: Any) -> None:
        """Store attribute assignments as dictionary items.

        Args:
            key (Any): Key to set.
            value (Any): Value to store.
        """
        self[key] = value

    def __repr__(self) -> str:
        """Return `AttributeDict(key=value, ...)`.

        Returns:
            str: String representation.
        """
        items = [f"{k}={v}" for k, v in self.items()]
        return f"{self.__class__.__name__}({', '.join(items)})"


def _parse_ini_value(value: str) -> Any:
    """Parse an INI value as a Python literal, falling back to the raw string.

    `ast.literal_eval` only accepts literals (numbers, strings, tuples, lists, dicts,
    booleans, None), so a config file can't execute code. Unquoted strings such as
    `bin_width = ppm` aren't valid literals and are returned unchanged.

    Args:
        value (str): Raw value from the INI file.

    Returns:
        Any: Parsed value, or the raw string.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


class Config:
    """All settings for a run.

    Defaults are class attributes. Each instance gets its own deep copy of every
    section in `__init__`, so changing one instance (loading an INI, writing tuned
    parameters) never changes the defaults seen by other instances.

    Sections:
        h5: What is stored in / loaded from the on-disk (lazy) MSRun.
        discretization: Fragment m/z binning (passed to `MzDiscretize`).
        tuning: Hyperparameter search for the NMF regularization.
        ms2_preprocessing: Options applied while converting mzML.
        scan_filter: Time tiling of each isolation window into scan windows.
        nmf: Final NMF settings (passed to `fit_nmf_matrix_custom_init`).
        fitpeaks: Gaussian fit of each component's elution profile.
        feature_matching: MS1 feature to MS2 component regression.
        feature_output: What the exported feature table contains.
    """

    # Seed for NMF initialization, window sampling and the tuning samplers.
    random_seed = 42
    # Show tqdm progress bars.
    tqdm_enabled = False
    # True: keep GPFRuns on disk in the run's HDF5 and reload one per stage (low RAM).
    # False: keep the whole run in memory.
    lazy = True
    # Store pivoted intensities as float32 instead of float64.
    downcast_intensities = True

    # scan_window_save_m_unfilter: also store each window's unfiltered matrix in HDF5.
    # scan_window_filter_during_load: filter windows as soon as a GPFRun is added.
    # compact_storage: smaller HDF5 for lazy runs (same data and results): windows don't
    #   repeat the GPF-wide m/z axis, the bin grid is stored once per file, and datasets
    #   are compressed with compact_compression ("gzip" or "lzf").
    # gpfs_per_file: 0 writes all GPFRuns into the run's HDF5. N > 0 writes them into
    #   part files of N isolation targets each, in "<stem>.gpf_runs/" next to it.
    h5 = AttributeDict(
        **{
            "scan_window_save_m_unfilter": False,
            "scan_window_filter_during_load": True,
            "compact_storage": False,
            "compact_compression": "gzip",
            "gpfs_per_file": 0,
        }
    )

    # bin_width: "ppm" (width proportional to m/z) or "da" (constant width).
    # tolerance: bin width, as a fraction for ppm (20e-6 = 20 ppm) or in Da.
    # steps: number of interleaved grids, each shifted by 1/steps of a bin.
    discretization = AttributeDict(
        **{
            "bin_width": "ppm",
            "mz_low": 200.0,
            "mz_high": 2000.0,
            "steps": 2,
            "tolerance": 20.0 / 1e6,
        }
    )

    # run_tune: tune at all. optuna: Optuna (True) or randomized grid search (False).
    # window_sampling_n: target number of scan windows to tune on, across all GPFs.
    # window_sampling_fraction: used instead when window_sampling_n is absent.
    # exclude_scan_window_edges: windows to skip at the start/end of each GPF.
    # *hyperparameter_grid: search spaces for Optuna and randomized search.
    # objective_params: metrics Optuna maximizes (multi-objective).
    # n_iter: number of trials. n_splits / test_fraction: cross-validation folds.
    # components_in_window / component_sigma: expected peptides per window and peak
    #   width, used as targets by OptimizationParameters.
    tuning = AttributeDict(
        **{
            "run_tune": True,
            "optuna": True,
            "window_sampling_n": 50,
            "window_sampling_fraction": 0.0125,
            "exclude_scan_window_edges": 10,
            "optuna_hyperparameter_grid": {
                "l1_ratio": (0.1, 0.9),
                "alpha_W": (0.0001, 0.1, "log"),
                "alpha_H": (0.0001, 0.1, "log"),
            },
            "hyperparameter_grid": {
                "alpha_W": [0.0001, 0.0005, 0.001, 0.005, 0.01],
                "alpha_H": [0.0001, 0.0005, 0.001, 0.005, 0.01],
                "l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            },
            "objective_params": [
                "test_reconstruction_errors",
                "neglog_ratio_train_test_reconstruction_error",
                "sample_orthogonality",
                "weight_orthogonality",
                "fraction_window_component",
            ],
            "n_iter": 500,
            "n_components": 20,
            "components_in_window": 8.0,
            "component_sigma": 2.0,  # In units of scans (95% of peak density falls within 8 scans.)
            "n_splits": 3,
            "test_fraction": 0.2,
            "splitter_type": "Mask",
        }
    )

    # include_gpf_bounds: store each scan's isolation lower/upper m/z.
    # filter_spectra: drop peaks at or below filter_spectra_percentile of each spectrum.
    ms2_preprocessing = AttributeDict(
        **{
            "include_gpf_bounds": True,
            "filter_spectra": False,
            "filter_spectra_percentile": 50,
        }
    )

    # scan_width: scans per window. scan_overlap: stride between window starts (in
    # scans), so consecutive windows share scan_width - scan_overlap scans.
    scan_filter = AttributeDict(
        **{
            "scan_overlap": 30,
            "scan_width": 150,
        }
    )

    # Passed as keyword arguments to sklearn.decomposition.NMF. alpha_W / alpha_H set
    # the regularization strength on W (elution profiles) and H (spectra); l1_ratio
    # mixes L1 (sparsity) and L2. Key case matters (alpha_W, not alpha_w).
    nmf = AttributeDict(
        **{
            "n_components": 20,
            "alpha_W": 0.00001,
            "alpha_H": 0.0375,
            "l1_ratio": 0.75,
            "max_iter": 500,
            "solver": "cd",
        }
    )

    # Bounds on the Gaussian mean / sigma, in retention time units (minutes).
    fitpeaks = AttributeDict(
        **{
            "mu_bounds": None,
            "sigma_bounds": [0.1, 10.0],
        }
    )

    # match_ms1: export MS1 matches (the CLI sets it from --ms1_features_file).
    # alpha: L1 penalty of the MS1 regression. c_cutoff: drop smaller coefficients.
    # extend_w_fraction: widen each component's 99% elution interval by this fraction
    #   of its width when selecting scans for the regression.
    feature_matching = AttributeDict(
        **{
            "match_ms1": True,
            "alpha": 0.1,
            "c_cutoff": 0.025,
            "extend_w_fraction": 0.5,
        }
    )

    feature_output = AttributeDict(
        **{
            "top_n_rescaled_peaks": 200,  # Number of peaks to report for weight/reweighted spectra.
            "matching_coefficient_threshold": 0.1,  # Minimum reporting threshold.
            "top_n_ms1_matches": 4,  # Number of matches to report before ties.
            # True: one row per (component, MS1 match), repeating the component's
            # spectra on each row. False: one row per component, MS1 matches stored
            # as arrays (smaller; dump_df_to_ms2 handles both).
            "explode_ms1_matches": True,
        }
    )

    def __init__(self) -> None:
        """Create a config with its own copy of every default section."""
        # Give each instance its own copy of the section dicts so that loading an INI
        # or writing tuned parameters doesn't change the defaults for other instances.
        for key, value in vars(self.__class__).items():
            if isinstance(value, AttributeDict):
                setattr(self, key, AttributeDict(copy.deepcopy(dict(value))))

    def __repr__(self):
        """Return string representation of Config showing all attributes"""
        items = []

        # Class defaults, overridden by anything set on this instance.
        attributes = dict(**vars(self.__class__))
        attributes.update(vars(self))

        for key, value in attributes.items():
            # Skip private names and methods, keep settings only.
            if (
                not key.startswith("_")
                and not isinstance(value, classmethod)
                and not callable(value)
            ):
                if isinstance(value, AttributeDict):
                    items.append(f"{key}={value}")
                else:
                    items.append(f"{key}={repr(value)}")
        attribute_sep = "\n  "
        return f"{self.__class__.__name__}(\n  {(', '+attribute_sep).join(items)}\n)"

    @classmethod
    def from_ini(cls, ini_path: Union[str, Path]) -> "Config":
        """Load configuration from an INI file.

        Starts from the defaults and overrides only the keys present in the file.
        Sections matching a Config section update that section; any other section
        (normally `[general]`) sets top-level attributes. Values are parsed with
        `_parse_ini_value`.

        Args:
            ini_path (Union[str, Path]): Path to the INI file

        Returns:
            Config: New config instance with values from INI file
        """
        config = configparser.ConfigParser()
        # Keep key case as written (configparser lower-cases keys by default).
        config.optionxform = str
        config.read(ini_path)

        instance = cls()

        for section in config.sections():
            section_value = getattr(instance, section, None)
            if isinstance(section_value, AttributeDict):
                # Match keys case-insensitively so that INIs written with lower-cased
                # keys (e.g. alpha_w) update the existing key (alpha_W).
                canonical_keys = {k.lower(): k for k in section_value}
                for key, value in config[section].items():
                    section_value[canonical_keys.get(key.lower(), key)] = (
                        _parse_ini_value(value)
                    )
            else:
                # Is general setting section:
                for key, value in config[section].items():
                    setattr(instance, key, _parse_ini_value(value))

        return instance

    def to_ini(self, ini_path: Union[Path, str]) -> None:
        """Save current configuration to an INI file.

        Section dicts become INI sections, other public non-callable attributes go in
        `[general]`. Values are written with `repr`, so `from_ini` reads them back as
        the same Python literals.

        Args:
            ini_path (Union[Path, str]): Path where to save the INI file
        """
        config = configparser.ConfigParser()
        # Keep key case (alpha_W) so the file loads back into the right keys.
        config.optionxform = str

        # Class defaults, overridden by anything set on this instance.
        attributes = dict(**vars(self.__class__))
        attributes.update(vars(self))

        config["general"] = {}
        for key, value in attributes.items():
            # Skip private names and methods, keep settings only.
            if (
                not key.startswith("_")
                and not isinstance(value, classmethod)
                and not callable(value)
            ):
                if isinstance(value, AttributeDict):
                    config[key] = {}
                    for k, v in value.items():
                        config[key][k] = repr(v)
                else:
                    config["general"][key] = repr(value)

        with open(ini_path, "w") as f:
            config.write(f)


class Constants:
    """Physical constants used for isotope m/z and MH+ calculations."""

    isotope_mu = 1.00336  # 13C - 12C mass difference.
    proton_mu = 1.0073  # Proton mass (Da).
