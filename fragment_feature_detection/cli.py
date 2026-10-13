"""Command line interface for fragment_feature_detection (the `ffd` command).

Each subcommand is split in two:

- a plain Python function (e.g. `process_mzml`) that does the work and can be called
  from scripts or notebooks, and
- a thin click wrapper (e.g. `_process_mzml`) that parses the command line, sets up a
  log file and calls the plain function.

Subcommands:

- `generate_template_ini`: write the default `Config` to `template.ini`.
- `convert_mzml_h5long`: mzML -> long-format HDF5 (one dataset per isolation target).
- `process_mzml`: the full pipeline (convert, tune, NMF, MS1 matching, export).
- `dump_df_to_ms2`: features Parquet -> `.ms2` file for database searching.
"""

import functools
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

import click
import pandas as pd

from fragment_feature_detection.config import Config, format_logger
from fragment_feature_detection.containers import (
    add_ms1_features_msrun,
    dump_features_to_df_msrun,
    fit_ms1_ms2_feature_matching_msrun,
    fit_nmf_matrix_msrun,
)
from fragment_feature_detection.containers.msrun import MSRun
from fragment_feature_detection.converter import MzMLParser
from fragment_feature_detection.decomposition import (
    tune_hyperparameters_optunasearchcv,
    tune_hyperparameters_randomizedsearchcv,
)
from fragment_feature_detection.utils import feature_df_to_ms2

# All modules log through the root logger. Send it to stdout here; a per-command file
# handler is added by configure_logger below.
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
format_logger(logger)


def arg_logger(f: Callable):
    """Decorator that logs function arguments and execution status.

    Logs a "Start" line with all positional and keyword arguments before the call and a
    "Finish" line after it returns, so every run's log records exactly how it was
    invoked.

    Args:
        f (Callable): Function to be decorated

    Returns:
        (Callable): Wrapped function that logs its arguments and execution status
    """

    @functools.wraps(f)
    def func(*args: Any, **kwargs: Any):
        logger.info("Start: %s -> args: %s, kwargs: %s" % (f.__name__, args, kwargs))
        res = f(*args, **kwargs)
        logger.info("Finish: %s" % f.__name__)
        return res

    return func


def configure_logger(fw: Optional[Callable] = None):
    """Decorator that configures logging for CLI commands.

    Sets up file-based logging in addition to console logging. Log files are created
    in the same directory as the input file (not `--output_dir`), named
    `<YYYYmmdd-HHMMSS>__<subcommand>__<input stem>.log`.

    Args:
        fw (Optional[Callable]): Optional wrapped function. Its name and docstring are
            copied onto the click command, so `ffd <command> --help` shows the plain
            function's docstring.

    Returns:
        (Callable): Decorator function that configures logging
    """

    def decorator(f: Callable):
        @functools.wraps(fw if fw else f)
        def func(*args: Any, **kwargs: Any):
            logger = logging.getLogger()
            context = click.get_current_context()
            subcommand = context.info_name
            # Only commands that take a FILE argument get a log file (all except
            # generate_template_ini).
            if "file" in context.params:
                path = Path(context.params["file"])
                # Add file handler that logs into the experiment directory.
                file_handler = logging.FileHandler(
                    "{path}/{:%Y%m%d-%H%M%S}__{cmd}__{file}.log".format(
                        datetime.now(),
                        cmd=subcommand,
                        path=path.parent,
                        file=path.stem,
                    ),
                    mode="w",
                )
                # Get default stream handler from root logger.
                stream_handler = [
                    h
                    for h in logging.getLogger().handlers
                    if isinstance(h, logging.StreamHandler)
                ]
                if len(stream_handler) > 0:
                    stream_handler = stream_handler[0]
                    file_handler.setFormatter(stream_handler.formatter)
                logger.addHandler(file_handler)
                format_logger(logger)
            return f(*args, **kwargs)

        return func

    return decorator


@click.group()
def main():
    """Fragment feature detection: deconvolve DIA MS2 data into component spectra."""
    pass


@arg_logger
def generate_template_ini(
    output_dir: Optional[Union[Path, str]] = None,
) -> None:
    """Generate a template configuration INI file.

    Writes the `Config()` defaults to `<output_dir>/template.ini`. Edit it and pass it
    to `process_mzml --config_ini_file`.
    \f
    Args:
        output_dir (Optional[Union[Path, str]]): Directory for template.ini. Defaults
            to the current directory.
    """
    if not output_dir:
        output_dir = Path()
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    template_ini_fh = output_dir / "template.ini"

    logger.info(f"Creating template ini at {template_ini_fh}...")

    config = Config()
    config.to_ini(template_ini_fh)


@main.command("generate_template_ini")
@click.option(
    "--output_dir", type=click.Path(), help="Output location for template INI file."
)
@configure_logger(fw=generate_template_ini)
def _generate_template_ini(*args: Any, **kwargs: Any) -> None:
    """Click wrapper for generate_template_ini (help text comes from that function)."""
    generate_template_ini(*args, **kwargs)


@arg_logger
def convert_mzml_h5long(
    file: Union[Path, str],
    output_dir: Optional[Union[Path, str]] = None,
    config_ini_file: Optional[Union[Path, str]] = None,
    config: Optional[Config] = None,
) -> Path:
    """Converts mass spectrometry data from mzML format to a more efficient
    HDF5 storage format for faster processing.

    Only MS2 spectra are kept. Peaks are stored in long format (one row per peak), with
    one `<isolation target m/z>/ms2_long` dataset per isolation window.
    \f
    Args:
        file (Union[Path, str]): Input mzML file (centroided).
        output_dir (Optional[Union[Path, str]]): Where to write `<stem>.h5`. Defaults
            to the mzML's directory.
        config_ini_file (Optional[Union[Path, str]]): INI used for the
            `[ms2_preprocessing]` options when `config` isn't given.
        config (Optional[Config]): Config to use directly (process_mzml passes its own).

    Returns:
        Path: Path of the written HDF5 file.
    """
    mzml_path = Path(file)
    if not output_dir:
        output_dir = Path(file).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # The converter reads [ms2_preprocessing] (gpf bounds, intensity filtering).
    if config is None:
        config = (
            Config.from_ini(config_ini_file) if config_ini_file is not None else Config()
        )

    out_fh = output_dir / mzml_path.with_suffix(".h5").name
    logger.info(f"Beginning conversion sample {mzml_path} to h5 long...")
    MzMLParser.to_ms2_h5(mzml_path, h5_fh=out_fh, config=config)

    return out_fh


@main.command("convert_mzml_h5long")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--output_dir", type=click.Path(), help="Output directory for processed files"
)
@click.option(
    "--config_ini_file",
    type=click.Path(exists=True),
    help="Path to configuration INI file",
)
@configure_logger(fw=convert_mzml_h5long)
def _convert_mzml_h5long(*args: Any, **kwargs: Any) -> None:
    """Click wrapper for convert_mzml_h5long (help text comes from that function)."""
    convert_mzml_h5long(*args, **kwargs)


@arg_logger
def process_mzml(
    file: Union[Path, str],
    ms1_features_file: Optional[Union[Path, str]] = None,
    config_ini_file: Optional[Union[Path, str]] = None,
    output_dir: Optional[Union[Path, str]] = None,
    n_jobs: int = 4,
) -> None:
    """Performs full analysis of mass spectrometry data including:
    - Converting mzML to HDF5 format
    - Optional hyperparameter tuning
    - NMF matrix fitting
    - Optional MS1 feature matching
    - Feature extraction and output generation

    Writes `<stem>.h5`, `<stem>.features.parq` and `<stem>.out.ini` (the config that
    was actually used, including tuned NMF parameters) to the output directory.
    \f
    Args:
        file (Union[Path, str]): Input mzML file (centroided, narrow-window DIA).
        ms1_features_file (Optional[Union[Path, str]]): Biosaur2 features TSV. If
            given, MS1 features are attached to scan windows and matched to MS2
            components; otherwise the export has no MS1 columns.
        config_ini_file (Optional[Union[Path, str]]): Config INI. Defaults to
            `Config()`.
        output_dir (Optional[Union[Path, str]]): Output directory. Defaults to the
            mzML's directory.
        n_jobs (int): Number of parallel workers for tuning, NMF, MS1 attachment and
            MS1 matching.
    """
    start_time = time.time()

    mzml_path = Path(file)
    if not output_dir:
        output_dir = mzml_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Read config_ini_file from arguments or create default config.
    if config_ini_file is not None:
        config = Config.from_ini(config_ini_file)
    else:
        config = Config()

    # Stage 1: convert mzML to long-format HDF5 (one dataset per isolation target).
    logger.info(f"Converting {mzml_path} to H5 format...")
    h5_long_path = convert_mzml_h5long(mzml_path, output_dir, config=config)

    # Stage 2: build the MSRun. Each isolation target becomes a GPFRun: its peaks are
    # binned onto the m/z grid, pivoted to a sparse scan x bin matrix and tiled into
    # overlapping ScanWindows. With config.lazy the GPFRuns are written into the same
    # HDF5 file (gpf_runs/gpf_<i>) and reloaded one at a time by later stages.
    logger.info("Creating MSRun object...")
    msrun = MSRun.from_h5_long(h5_long_path, config=config)

    # Stage 3 (optional): pick NMF regularization (alpha_W, alpha_H, l1_ratio) by
    # cross-validated search over a random sample of scan windows.
    if config.tuning.run_tune:
        # Get tuning windows.
        logger.info("Fetching tuning windows...")
        tuning_windows = msrun.get_tuning_windows(config=config)

        # Perform hyperparameter optimization
        logger.info("Starting Optuna parameter optimization...")
        if config.tuning.optuna:
            tune_function = tune_hyperparameters_optunasearchcv
        else:
            tune_function = tune_hyperparameters_randomizedsearchcv

        best_params = tune_function(
            tuning_windows,
            config=config,
            save_run=True,
            n_jobs=n_jobs,
        )

        # Update config with best parameters
        config.nmf.l1_ratio = best_params["l1_ratio"]
        config.nmf.alpha_H = best_params["alpha_H"]
        config.nmf.alpha_W = best_params["alpha_W"]
    else:
        logger.info(
            "Skipping hyperparameter tuning. This will likely over/under fit your data."
        )

    # Stage 4: NMF on every scan window, Gaussian fit of every component's elution
    # profile, then collapse duplicate components found in overlapping windows.
    logger.info("Fitting NMF for run...")
    fit_nmf_matrix_msrun(msrun, n_jobs=n_jobs, config=config)

    # Stage 5 (optional): attach Biosaur2 MS1 features to each window and regress each
    # MS2 component's elution profile on the MS1 feature profiles.
    if ms1_features_file is not None:
        # Set match_ms1 if you provide an ms1_features_file
        config.feature_matching.match_ms1 = True

        # Add MS1 features to the msrun object. The hill scan and intensity lists are
        # stored as strings in the TSV, pd.eval turns them back into lists.
        logger.info("Adding MS1 features...")
        ms1_features = pd.read_csv(
            ms1_features_file,
            sep="\t",
            converters={
                "mono_hills_scan_lists": pd.eval,
                "mono_hills_intensity_list": pd.eval,
            },
        )
        add_ms1_features_msrun(msrun, ms1_features, config=config, n_jobs=n_jobs)

        # Perform MS1-MS2 feature matching
        logger.info("Performing MS1-MS2 feature matching...")
        fit_ms1_ms2_feature_matching_msrun(msrun, n_jobs=n_jobs, config=config)
    else:
        # Without an ms1_features_file there is nothing to match, so export without MS1.
        config.feature_matching.match_ms1 = False

    # Stage 6: one row per kept component (exploded per MS1 match when matching ran).
    features_df = dump_features_to_df_msrun(msrun, config=config)
    features_df.to_parquet(output_dir / mzml_path.with_suffix(".features.parq").name)

    # Write ini file. It records the parameters actually used (e.g. tuned NMF values)
    # and can be passed back with --config_ini_file to reproduce the run.
    config.to_ini(output_dir / mzml_path.with_suffix(".out.ini").name)

    msrun.close()

    logger.info(f"Process mzml finished in {(time.time() - start_time) / 60} minutes.")


@main.command("process_mzml")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--ms1_features_file",
    type=click.Path(exists=True),
    help="Path to MS1 features file",
)
@click.option(
    "--config_ini_file",
    type=click.Path(exists=True),
    help="Path to configuration INI file",
)
@click.option(
    "--output_dir", type=click.Path(), help="Output directory for processed files"
)
@click.option("--n_jobs", type=int, default=4, help="Number of parallel jobs")
@configure_logger(fw=process_mzml)
def _process_mzml(*args: Any, **kwargs: Any) -> None:
    """Click wrapper for process_mzml (help text comes from that function)."""
    process_mzml(*args, **kwargs)


@arg_logger
def dump_df_to_ms2(
    file: Union[Path, str],
    wide_window: bool = False,
    output_type: Literal["raw", "intensity_transform_weight", "raw_weight"] = "raw",
    output_dir: Optional[Union[Path, str]] = None,
) -> None:
    """Converts processed feature data from parquet format to MS2 format
    for downstream analysis. output_type is type of output format to generate.
    Options are: ["raw": Original intensities, "intensity_transform_weight":
    weights rescaled to observed intensities, "raw_weight": raw weights from learned
    models].

    The output file is `<parquet stem>.<output_type>[.wide].ms2`.
    \f
    Args:
        file (Union[Path, str]): `<stem>.features.parq` from process_mzml.
        wide_window (bool): If True, ignore MS1 matches and use the isolation window
            center (charge 2) as the precursor. If False, write one spectrum per MS1
            match and skip components without a match.
        output_type (Literal["raw", "intensity_transform_weight", "raw_weight"]):
            Which peak list to write for each spectrum.
        output_dir (Optional[Union[Path, str]]): Output directory. Defaults to the
            Parquet's directory.
    """
    start_time = time.time()

    parquet_path = Path(file)
    if not output_dir:
        output_dir = parquet_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Recorded in the .ms2 header (H ExtractorOptions line).
    extractor_options = {
        "wide_window": wide_window,
        "output_type": output_type,
    }
    ms2_suffix = f".{output_type}{'.wide' if wide_window else ''}.ms2"
    logger.info(f"Starting to dump ms2 file with params: {extractor_options}...")
    feature_df_to_ms2(
        pd.read_parquet(parquet_path),
        output_dir / parquet_path.with_suffix(ms2_suffix).name,
        extractor_options=extractor_options,
    )
    logger.info(f"Finished dumping ms2 file in: {(time.time() - start_time)} seconds.")


@main.command("dump_df_to_ms2")
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--output_type",
    type=click.Choice(
        ["raw", "intensity_transform_weight", "raw_weight"], case_sensitive=False
    ),
    default="raw",
    help="Output format type",
)
@click.option(
    "--wide_window",
    is_flag=True,
    help="Ignore MS1 features, output GPF center as precursor mass",
)
@click.option(
    "--output_dir", type=click.Path(), help="Output directory for processed files"
)
@configure_logger(fw=dump_df_to_ms2)
def _dump_df_to_ms2(*args: Any, **kwargs: Any) -> None:
    """Click wrapper for dump_df_to_ms2 (help text comes from that function)."""
    dump_df_to_ms2(*args, **kwargs)
