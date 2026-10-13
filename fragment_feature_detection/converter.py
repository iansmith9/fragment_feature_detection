"""mzML reading and conversion to long-format arrays / HDF5.

The pipeline works on a "long" peak table: one row per centroided MS2 peak with the
columns

    0 MS2TargetMass   isolation window target m/z (identifies the isolation window)
    1 ScanNum         spectrum index + 1 (not the vendor scan number)
    2 RetentionTime   scan start time (minutes)
    3 m/z             peak m/z
    4 intensity       peak intensity
    5 lowerMass       isolation lower bound   (only if include_gpf_bounds)
    6 higherMass      isolation upper bound   (only if include_gpf_bounds)

`MzMLParser.to_ms2_h5` streams an mzML into an HDF5 file with one group per isolation
target (`/<MS2TargetMass>/ms2_long`), which `MSRun.from_h5_long` reads back.
"""

from typing import Any, Dict, Union, Literal, Optional, Type
from pathlib import Path
import logging
from collections import defaultdict

import pandas as pd
from pyteomics import mzml
import numpy as np
from tqdm import tqdm
import h5py

from fragment_feature_detection.config import Config

logger = logging.getLogger(__name__)


class MzMLParser(mzml.MzML):
    """pyteomics mzML reader that only iterates spectra of one MS level.

    pyteomics walks the XML with an XPath expression (`_default_iter_path`). The class
    replaces it with one that only matches `<spectrum>` elements carrying an
    `ms level` cvParam equal to 1 or 2, so MS1 (or MS2) spectra are skipped without
    being decoded. The index isn't used (`_use_index = False`), so the file is read
    sequentially from start to end.
    """

    # XPath: <spectrum> elements with a child cvParam "ms level" == 1 / == 2.
    _ms1_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="1"]]'
    )
    _ms2_iter_path = (
        '//spectrum[./*[local-name()="cvParam" and @name="ms level" and @value="2"]]'
    )
    _default_iter_path = '//spectrum[./*[local-name()="cvParam"]]'
    _use_index = False
    _iterative = False

    def __init__(self, *args: Any, keep_level: Literal[1, 2] = 2, **kwargs: Any):
        """Open an mzML file, iterating only spectra of MS level `keep_level`.

        Args:
            *args (Any): Passed to `pyteomics.mzml.MzML` (e.g. `source`).
            keep_level (Literal[1, 2]): MS level to iterate. Defaults to 2.
            **kwargs (Any): Passed to `pyteomics.mzml.MzML`.
        """
        if keep_level == 1:
            self._default_iter_path = self._ms1_iter_path
        elif keep_level == 2:
            self._default_iter_path = self._ms2_iter_path
        super().__init__(*args, **kwargs)

    @classmethod
    def to_ms2_h5(
        cls: Type["MzMLParser"],
        path: Union[Path, str],
        h5_fh: Optional[Union[Path, str]] = None,
        config: Config = Config(),
        chunk_size: int = 2500,
    ) -> None:
        """Stream the MS2 spectra of an mzML into a long-format HDF5 file.

        Peaks are grouped by isolation target. Each target gets a group named
        `str(MS2TargetMass)` holding an extendable, gzip-compressed float32 dataset
        `ms2_long` (columns described in the module docstring). Rows are buffered in
        memory and appended every `chunk_size` spectra to bound memory use.

        Spectra that fail to parse are logged and skipped. The file is opened with
        mode "w", so an existing HDF5 at `h5_fh` is overwritten.

        Args:
            path (Union[Path, str]): Input mzML.
            h5_fh (Optional[Union[Path, str]]): Output HDF5. Defaults to the mzML path
                with a `.h5` suffix.
            config (Config): Uses `[ms2_preprocessing]` (gpf bounds, intensity filter)
                and `tqdm_enabled`.
            chunk_size (int): Number of spectra buffered between HDF5 writes.
        """
        if not h5_fh:
            h5_fh = Path(path).with_suffix(".h5")

        obj = cls(source=str(path), keep_level=2)

        def write_chunks() -> None:
            """Append the buffered rows of every isolation target to its dataset.

            Creates the group and a resizable `(0, n_columns)` dataset the first time
            a target is seen, grows it by the number of buffered rows, writes them at
            the end, then clears the buffer.
            """
            nonlocal mz_arrays
            existing_groups = [m for m in f.keys() if isinstance(f[m], h5py.Group)]

            for m, a in mz_arrays.items():
                if len(a) == 0:
                    continue
                if m not in existing_groups:
                    n_columns = len(a[0])
                    group = f.create_group(m)
                    dataset = group.create_dataset(
                        "ms2_long",
                        shape=(0, n_columns),
                        maxshape=(None, n_columns),
                        compression="gzip",
                        dtype="float32",
                    )
                dataset = f[m]["ms2_long"]
                dataset.resize((dataset.shape[0] + len(a), dataset.shape[1]))
                dataset[-1 * len(a) :] = np.vstack(a)

            f.flush()
            del mz_arrays
            mz_arrays = defaultdict(list)

        with h5py.File(h5_fh, "w") as f:

            # Buffer: isolation target (as a string key) -> list of peak rows.
            mz_arrays = defaultdict(list)

            with tqdm(disable=(not config.tqdm_enabled)) as pbar:
                while True:
                    try:
                        scan = cls.parse_scan(next(obj), keep_level=2)
                        mz_array = scan["m/zArray"]
                        intensity_array = scan["IntensityArray"]
                        # Drop NaN / inf peaks.
                        mask = np.isfinite(mz_array) & np.isfinite(intensity_array)
                        mz_array = mz_array[mask]
                        intensity_array = intensity_array[mask]
                        # Optionally keep only peaks above a per-spectrum intensity
                        # percentile (removes low-level noise before binning).
                        if config.ms2_preprocessing.filter_spectra:
                            percentile_intensity = np.percentile(
                                intensity_array,
                                config.ms2_preprocessing.filter_spectra_percentile,
                            )
                            mz_array = mz_array[intensity_array > percentile_intensity]
                            intensity_array = intensity_array[
                                intensity_array > percentile_intensity
                            ]

                        # One output group per isolation window target m/z.
                        scan_key = str(scan["MS2TargetMass"])
                        base_scan_data = [
                            scan["MS2TargetMass"],
                            scan["ScanNum"],
                            scan["RetentionTime"],
                        ]

                        # One row per peak: scan-level columns repeated for each peak,
                        # plus the isolation bounds if requested.
                        if config.ms2_preprocessing.include_gpf_bounds:
                            extra_data = [scan["lowerMass"], scan["higherMass"]]
                            mz_arrays[scan_key].extend(
                                [
                                    base_scan_data + [mz, intensity] + extra_data
                                    for mz, intensity in zip(mz_array, intensity_array)
                                ]
                            )
                        else:
                            mz_arrays[scan_key].extend(
                                [
                                    base_scan_data + [mz, intensity]
                                    for mz, intensity in zip(mz_array, intensity_array)
                                ]
                            )

                        # Flush the buffer to disk every chunk_size spectra.
                        pbar.update(1)
                        if pbar.n % chunk_size == 0:
                            write_chunks()
                    except StopIteration:
                        # End of file: write whatever is still buffered.
                        write_chunks()
                        break
                    except Exception as e:
                        logger.exception(e)

    @classmethod
    def to_ms2_long(
        cls: Type["MzMLParser"], path: Union[Path, str], config: Config = Config()
    ) -> np.ndarray:
        """Read all MS2 peaks of an mzML into one in-memory long array.

        Same rows as `to_ms2_h5`, but for all isolation targets together and without
        HDF5. Useful for small files with `MSRun.from_long`.

        Args:
            path (Union[Path, str]): Input mzML.
            config (Config): Uses `[ms2_preprocessing]` and `tqdm_enabled`.

        Returns:
            np.ndarray: Long array, one row per peak (see module docstring).
        """
        obj = cls(source=str(path), keep_level=2)

        mz_arrays = []

        with tqdm(disable=(not config.tqdm_enabled)) as pbar:
            while True:
                try:
                    scan = cls.parse_scan(next(obj), keep_level=2)
                    mz_array = scan["m/zArray"]
                    intensity_array = scan["IntensityArray"]
                    # Drop NaN / inf peaks.
                    mask = np.isfinite(mz_array) & np.isfinite(intensity_array)
                    mz_array = mz_array[mask]
                    intensity_array = intensity_array[mask]
                    # Optional per-spectrum intensity percentile filter.
                    if config.ms2_preprocessing.filter_spectra:
                        percentile_intensity = np.percentile(
                            intensity_array,
                            config.ms2_preprocessing.filter_spectra_percentile,
                        )
                        mz_array = mz_array[intensity_array > percentile_intensity]
                        intensity_array = intensity_array[
                            intensity_array > percentile_intensity
                        ]

                    base_scan_data = [
                        scan["MS2TargetMass"],
                        scan["ScanNum"],
                        scan["RetentionTime"],
                    ]

                    # One row per peak, as in to_ms2_h5.
                    if config.ms2_preprocessing.include_gpf_bounds:
                        extra_data = [scan["lowerMass"], scan["higherMass"]]
                        mz_arrays.extend(
                            [
                                base_scan_data + [mz, intensity] + extra_data
                                for mz, intensity in zip(mz_array, intensity_array)
                            ]
                        )
                    else:
                        mz_arrays.extend(
                            [
                                base_scan_data + [mz, intensity]
                                for mz, intensity in zip(mz_array, intensity_array)
                            ]
                        )
                    pbar.update(1)
                except StopIteration:
                    break
                except Exception as e:
                    logger.exception(e)

        return np.vstack(mz_arrays)

    @classmethod
    def to_ms2_df(
        cls: Type["MzMLParser"], path: Union[Path, str], config: Config = Config()
    ) -> pd.DataFrame:
        """Read all MS2 spectra of an mzML into a DataFrame, one row per spectrum.

        Columns are the keys returned by `parse_scan`; `m/zArray` and
        `IntensityArray` hold the (finite) peak arrays. No intensity filtering.

        Args:
            path (Union[Path, str]): Input mzML.
            config (Config): Uses `tqdm_enabled`.

        Returns:
            pd.DataFrame: One row per MS2 spectrum.
        """
        obj = cls(source=str(path), keep_level=2)

        scans = []

        with tqdm(disable=(not config.tqdm_enabled)) as pbar:
            while True:
                try:
                    scan = cls.parse_scan(next(obj), keep_level=2)
                    mz_array = scan["m/zArray"]
                    intensity_array = scan["IntensityArray"]
                    # Drop NaN / inf peaks.
                    mask = np.isfinite(mz_array) & np.isfinite(intensity_array)
                    scan["m/zArray"] = mz_array[mask]
                    scan["IntensityArray"] = intensity_array[mask]
                    scans.append(scan)
                    pbar.update(1)
                except StopIteration:
                    break
                except Exception as e:
                    logger.exception(e)

        return pd.DataFrame.from_records(scans)

    @staticmethod
    def parse_scan(
        s: Dict[str, Any], keep_level: Literal[1, 2] = 1
    ) -> Union[Dict[str, Any], None]:
        """Extract the fields the pipeline needs from one pyteomics spectrum dict.

        Always extracted: `ScanNum` (spectrum index + 1), `RetentionTime` (scan start
        time), `m/zArray`, `IntensityArray`.

        MS2 only: `MS2TargetMass` (isolation window target m/z), `PrecursorDetected`
        (selected ion m/z), `lowerMass` / `higherMass` (selected ion m/z minus / plus
        the isolation window lower / upper offsets) and `IsolationSize`. For DIA the
        selected ion m/z is normally the window target.

        MS1 only: `lowerMass` / `higherMass` from the scan window limits.

        Args:
            s (Dict[str, Any]): Spectrum dict produced by pyteomics.
            keep_level (Literal[1, 2]): Expected MS level. Defaults to 1.

        Returns:
            Union[Dict[str, Any], None]: Parsed fields, or None if the spectrum's MS
            level isn't `keep_level`.
        """
        if s["ms level"] != keep_level:
            return

        scan = {}

        scan["ScanNum"] = s["index"] + 1
        scan["RetentionTime"] = s["scanList"]["scan"][0]["scan start time"]
        scan["m/zArray"] = s["m/z array"]
        scan["IntensityArray"] = s["intensity array"]

        if keep_level == 2:
            # The isolation window target identifies which DIA window this scan is.
            scan["MS2TargetMass"] = s["precursorList"]["precursor"][0][
                "isolationWindow"
            ]["isolation window target m/z"]
            scan["PrecursorDetected"] = s["precursorList"]["precursor"][0][
                "selectedIonList"
            ]["selectedIon"][0]["selected ion m/z"]

            # Isolation bounds, used later to decide which MS1 features (isotopes)
            # can have been co-isolated in this window.
            scan["lowerMass"] = (
                scan["PrecursorDetected"]
                - s["precursorList"]["precursor"][0]["isolationWindow"][
                    "isolation window lower offset"
                ]
            )
            scan["higherMass"] = (
                scan["PrecursorDetected"]
                + s["precursorList"]["precursor"][0]["isolationWindow"][
                    "isolation window upper offset"
                ]
            )

            scan["IsolationSize"] = scan["higherMass"] - scan["lowerMass"]

        elif keep_level == 1:
            # For MS1, record the acquired m/z range of the scan.
            scan["lowerMass"] = s["scanList"]["scan"][0]["scanWindowList"][
                "scanWindow"
            ][0]["scan window lower limit"]
            scan["higherMass"] = s["scanList"]["scan"][0]["scanWindowList"][
                "scanWindow"
            ][0]["scan window upper limit"]

        return scan
