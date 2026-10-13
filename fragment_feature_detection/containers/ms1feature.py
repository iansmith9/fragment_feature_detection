"""MS1 precursor features (from Biosaur2) and their elution on the MS2 time axis.

An MS1Feature is one isotope-pattern feature detected in the MS1 scans. The pipeline
attaches features to each ScanWindow whose isolation window contains one of the
feature's isotope peaks, then interpolates the feature's monoisotopic "hill" (its
intensity over time) onto that window's MS2 retention times. The interpolated profile is
what MS1-MS2 matching regresses each NMF component's elution profile against.
"""

from typing import (
    Union,
    Optional,
    Tuple,
    Type,
    Literal,
    List,
    Dict,
    Any,
)
from pathlib import Path
import logging

import numpy as np
import h5py
import pandas as pd

logger = logging.getLogger(__name__)


class MS1Feature:
    """One Biosaur2 MS1 feature and its elution profile.

    The raw hill is stored as ``_scan_index``/``_intensity`` with an approximate time
    axis ``_retention_time``. After ``interpolate_onto_ms2`` the profile resampled onto
    a scan window's MS2 retention times is in ``_interpolated_intensity``. That vector
    is one column of the matrix returned by ``ScanWindow.ms1_features_information``.

    Attributes:
        _id (int): Row index of the feature in the Biosaur2 table (used as its ID).
        _calibrated_mass (float): Neutral monoisotopic mass (Biosaur2 ``massCalib``).
        _retention_time_apex (float): RT of the feature apex.
        _intensity_apex (float): Intensity at the apex.
        _intensity_sum (float): Summed intensity of the feature.
        _charge (int): Charge state.
        _n_isotopes (int): Number of isotope peaks detected.
        _n_scans (int): Number of MS1 scans the feature spans.
        _mz (float): Monoisotopic m/z.
        _rt_start (float): First RT of the feature.
        _rt_end (float): Last RT of the feature.
        _scan_index (np.ndarray): MS1 scan indices of the monoisotopic hill.
        _intensity (np.ndarray): Hill intensities, one per entry of ``_scan_index``.
        _scan_apex (int): MS1 scan index of the apex.
        _retention_time (np.ndarray): RT for each hill point (see ``__init__``).
        _interpolated_retention_time (Optional[np.ndarray]): MS2 RTs the hill was
            interpolated onto.
        _interpolated_intensity (Optional[np.ndarray]): Hill intensity at those RTs.
    """

    # Set by interpolate_onto_ms2 (or restored from h5).
    _interpolated_retention_time = None
    _interpolated_intensity = None

    def __init__(
        self,
        df_index: int,
        calibrated_mass: float,
        retention_time_apex: float,
        intensity_apex: float,
        intensity_sum: float,
        charge: int,
        n_isotopes: int,
        n_scans: int,
        mz: float,
        rt_start: float,
        rt_end: float,
        scan_index: List[int],
        intensity: List[float],
        scan_apex: int,
    ) -> None:
        """Initialize an MS1Feature.

        Args:
            df_index (int): Row index in the Biosaur2 feature table, used as the ID.
            calibrated_mass (float): Neutral monoisotopic mass.
            retention_time_apex (float): RT of the apex.
            intensity_apex (float): Intensity at the apex.
            intensity_sum (float): Summed feature intensity.
            charge (int): Charge state.
            n_isotopes (int): Number of isotope peaks.
            n_scans (int): Number of MS1 scans spanned.
            mz (float): Monoisotopic m/z.
            rt_start (float): First RT of the feature.
            rt_end (float): Last RT of the feature.
            scan_index (List[int]): MS1 scan indices of the monoisotopic hill.
            intensity (List[float]): Hill intensities, aligned with scan_index.
            scan_apex (int): MS1 scan index of the apex.
        """
        self._id = df_index
        self._calibrated_mass = calibrated_mass
        self._retention_time_apex = retention_time_apex
        self._intensity_apex = intensity_apex
        self._intensity_sum = intensity_sum
        self._charge = charge
        self._n_isotopes = n_isotopes
        self._n_scans = n_scans
        self._mz = mz
        self._rt_start = rt_start
        self._rt_end = rt_end
        self._scan_index = np.array(scan_index)
        self._intensity = np.array(intensity)
        self._scan_apex = scan_apex
        # Biosaur2 doesn't report a time per hill point, so spread the points evenly
        # between rtStart and rtEnd. This assumes evenly spaced MS1 scans and no gaps
        # in the hill (see SUGGESTIONS.md A18).
        self._retention_time = np.linspace(
            rt_start,
            rt_end,
            len(scan_index),
            endpoint=True,
        )

    @classmethod
    def from_biosaur2_series(cls, series: pd.Series) -> "MS1Feature":
        """Create an MS1Feature from a Biosaur2 pandas Series.

        Maps Biosaur2 column names onto the constructor arguments. The hill columns
        (``mono_hills_scan_lists``/``mono_hills_intensity_list``) must already be parsed
        into lists (the CLI uses ``pd.eval`` converters when reading the TSV).

        Args:
            series (pd.Series): Series containing Biosaur2 feature data

        Returns:
            MS1Feature: New instance initialized with Biosaur2 data
        """
        return cls(
            df_index=series.name,
            calibrated_mass=series["massCalib"],
            retention_time_apex=series["rtApex"],
            intensity_apex=series["intensityApex"],
            intensity_sum=series["intensitySum"],
            charge=series["charge"],
            n_isotopes=series["nIsotopes"],
            n_scans=series["nScans"],
            mz=series["mz"],
            rt_start=series["rtStart"],
            rt_end=series["rtEnd"],
            scan_index=series["mono_hills_scan_lists"],
            intensity=series["mono_hills_intensity_list"],
            scan_apex=series["scanApex"],
        )

    @classmethod
    def from_biosaur2_series_with_ms2(
        cls, series: pd.Series, ms2_rts: np.ndarray
    ) -> "MS1Feature":
        """Create an MS1Feature and interpolate it onto a scan window's MS2 RTs.

        Args:
            series (pd.Series): Series containing Biosaur2 feature data.
            ms2_rts (np.ndarray): Retention times of the scan window's MS2 scans.

        Returns:
            MS1Feature: New instance with ``_interpolated_intensity`` set.
        """
        obj = cls.from_biosaur2_series(series)
        obj.interpolate_onto_ms2(ms2_rts)

        return obj

    def interpolate_onto_ms2(self, ms2_rts: np.ndarray) -> None:
        """Interpolate the hill intensities onto the MS2 retention times.

        The hill is padded with a zero one step before and after it, so the profile
        falls to zero outside the hill without overwriting its first and last points.
        MS2 times outside the padded hill get zero intensity.

        Args:
            ms2_rts (np.ndarray): Retention times of the scan window's MS2 scans.
        """
        rt = self._retention_time
        if rt.shape[0] == 0:
            # Empty hill: the feature contributes nothing to this window.
            intensities = np.zeros_like(ms2_rts, dtype=float)
        else:
            # Padding step: the hill's own point spacing, or, for a single-point hill
            # (or zero-width rt range), the median MS2 spacing.
            if rt.shape[0] > 1:
                step = (rt[-1] - rt[0]) / (rt.shape[0] - 1)
            else:
                step = 0.0
            if step <= 0.0:
                step = np.median(np.diff(ms2_rts)) if ms2_rts.shape[0] > 1 else 1e-3
            # Linear interpolation of [0, hill..., 0] at the MS2 times.
            intensities = np.interp(
                ms2_rts,
                np.concatenate([[rt[0] - step], rt, [rt[-1] + step]]),
                np.concatenate([[0.0], self._intensity.astype(float), [0.0]]),
                left=0.0,
                right=0.0,
            )

        self._interpolated_retention_time = ms2_rts
        self._interpolated_intensity = intensities

    def dump_h5(
        self,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "ms1_feature",
    ) -> None:
        """Save the MS1Feature object to an HDF5 file or group.

        Scalars are stored as group attributes and arrays as datasets. A string path
        opens the file in "w" mode (overwriting it); an h5py object gets a new subgroup.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group to store the object
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, "w")
            grp = f.create_group(dataset_name)
            close_file = True
        else:
            grp = h5file.create_group(dataset_name)
            close_file = False

        # Save scalar attributes
        grp.attrs["id"] = self._id
        grp.attrs["calibrated_mass"] = self._calibrated_mass
        grp.attrs["retention_time_apex"] = self._retention_time_apex
        grp.attrs["intensity_apex"] = self._intensity_apex
        grp.attrs["intensity_sum"] = self._intensity_sum
        grp.attrs["charge"] = self._charge
        grp.attrs["n_isotopes"] = self._n_isotopes
        grp.attrs["n_scans"] = self._n_scans
        grp.attrs["mz"] = self._mz
        grp.attrs["rt_start"] = self._rt_start
        grp.attrs["rt_end"] = self._rt_end
        grp.attrs["scan_apex"] = self._scan_apex

        # Save arrays
        grp.create_dataset("scan_index", data=self._scan_index)
        grp.create_dataset("intensity", data=self._intensity)
        grp.create_dataset("retention_time", data=self._retention_time)

        # Save interpolated data if it exists
        if self._interpolated_retention_time is not None:
            grp.create_dataset(
                "interpolated_retention_time", data=self._interpolated_retention_time
            )
            grp.create_dataset(
                "interpolated_intensity", data=self._interpolated_intensity
            )

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "ms1_feature",
    ) -> "MS1Feature":
        """Load an MS1Feature object from an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group containing the object

        Returns:
            MS1Feature: Loaded object
        """
        if isinstance(h5file, str):
            f = h5py.File(h5file, "r")
            grp = f[dataset_name]
            close_file = True
        else:
            grp = h5file[dataset_name]
            close_file = False

        # Create instance
        obj = cls(
            df_index=grp.attrs["id"],
            calibrated_mass=grp.attrs["calibrated_mass"],
            retention_time_apex=grp.attrs["retention_time_apex"],
            intensity_apex=grp.attrs["intensity_apex"],
            intensity_sum=grp.attrs["intensity_sum"],
            charge=grp.attrs["charge"],
            n_isotopes=grp.attrs["n_isotopes"],
            n_scans=grp.attrs["n_scans"],
            mz=grp.attrs["mz"],
            rt_start=grp.attrs["rt_start"],
            rt_end=grp.attrs["rt_end"],
            scan_index=grp["scan_index"][:],
            intensity=grp["intensity"][:],
            scan_apex=grp.attrs["scan_apex"],
        )

        # Load interpolated data if it exists
        if "interpolated_retention_time" in grp:
            obj._interpolated_retention_time = grp["interpolated_retention_time"][:]
            obj._interpolated_intensity = grp["interpolated_intensity"][:]

        if close_file:
            f.close()

        return obj


# Scalar MS1Feature fields stored as table columns: (dataset name, attribute name).
_TABLE_SCALARS = (
    ("id", "_id"),
    ("calibrated_mass", "_calibrated_mass"),
    ("retention_time_apex", "_retention_time_apex"),
    ("intensity_apex", "_intensity_apex"),
    ("intensity_sum", "_intensity_sum"),
    ("charge", "_charge"),
    ("n_isotopes", "_n_isotopes"),
    ("n_scans", "_n_scans"),
    ("mz", "_mz"),
    ("rt_start", "_rt_start"),
    ("rt_end", "_rt_end"),
    ("scan_apex", "_scan_apex"),
)
# Per-feature arrays, stored concatenated with offsets: (dataset name, attribute name).
_TABLE_ARRAYS = (("scan_index", "_scan_index"), ("intensity", "_intensity"))


def dump_ms1_feature_table(
    group: h5py.Group,
    name: str,
    features: List[MS1Feature],
    ms2_rts: np.ndarray,
    compression: Optional[str] = None,
) -> bool:
    """Store a scan window's MS1 features as one table instead of one group each.

    Used by compact storage. The standard layout writes 3-5 datasets per feature per
    window, so a run with many MS1 features creates a very large number of tiny HDF5
    objects. Here each field is one dataset for all features (per-feature arrays are
    concatenated, with offsets). The interpolated profiles aren't stored: they are
    recomputed on load from the same inputs, which gives identical values.

    Nothing is written, and False is returned, when the features can't be rebuilt
    exactly from a table (not interpolated onto ``ms2_rts``, mixed array dtypes, or
    non-numeric fields). The caller then uses the standard layout.

    Args:
        group (h5py.Group): The scan window's group.
        name (str): Name of the table group to create.
        features (List[MS1Feature]): The window's MS1 features.
        ms2_rts (np.ndarray): The window's retention times.
        compression (Optional[str]): h5py compression filter, or None.

    Returns:
        bool: True if the table was written.
    """
    from ..storage import create_dataset

    if any(
        f._interpolated_retention_time is None
        or not np.array_equal(f._interpolated_retention_time, ms2_rts)
        for f in features
    ):
        return False

    scalars = {
        key: np.array([getattr(f, attr) for f in features])
        for key, attr in _TABLE_SCALARS
    }
    if any(v.dtype == object for v in scalars.values()):
        return False

    arrays = {}
    for key, attr in _TABLE_ARRAYS:
        values = [np.asarray(getattr(f, attr)) for f in features]
        dtypes = {v.dtype for v in values}
        if len(dtypes) > 1 or any(v.ndim != 1 for v in values):
            return False
        dtype = dtypes.pop() if dtypes else np.float64
        arrays[key] = (
            np.concatenate(values) if values else np.empty(0, dtype=dtype),
            np.cumsum([0] + [v.size for v in values]),
        )

    grp = group.create_group(name)
    grp.attrs["n_features"] = len(features)
    for key, values in scalars.items():
        create_dataset(grp, key, values, compression=compression)
    for key, (values, offsets) in arrays.items():
        create_dataset(grp, key, values, compression=compression)
        create_dataset(grp, f"{key}_offsets", offsets, compression=compression)
    return True


def load_ms1_feature_table(grp: h5py.Group, ms2_rts: np.ndarray) -> List[MS1Feature]:
    """Rebuild the MS1 features written by ``dump_ms1_feature_table``.

    Args:
        grp (h5py.Group): The table group.
        ms2_rts (np.ndarray): The scan window's retention times, used to recompute
            each feature's interpolated profile.

    Returns:
        List[MS1Feature]: The features, in their original order.
    """
    n_features = int(grp.attrs["n_features"])
    scalars = {key: grp[key][()] for key, _ in _TABLE_SCALARS}
    arrays = {
        key: (grp[key][()], grp[f"{key}_offsets"][()]) for key, _ in _TABLE_ARRAYS
    }

    def ragged(key: str, i: int) -> np.ndarray:
        values, offsets = arrays[key]
        return values[offsets[i] : offsets[i + 1]]

    features = []
    for i in range(n_features):
        feature = MS1Feature(
            df_index=scalars["id"][i],
            calibrated_mass=scalars["calibrated_mass"][i],
            retention_time_apex=scalars["retention_time_apex"][i],
            intensity_apex=scalars["intensity_apex"][i],
            intensity_sum=scalars["intensity_sum"][i],
            charge=scalars["charge"][i],
            n_isotopes=scalars["n_isotopes"][i],
            n_scans=scalars["n_scans"][i],
            mz=scalars["mz"][i],
            rt_start=scalars["rt_start"][i],
            rt_end=scalars["rt_end"][i],
            scan_index=ragged("scan_index", i),
            intensity=ragged("intensity", i),
            scan_apex=scalars["scan_apex"][i],
        )
        feature.interpolate_onto_ms2(ms2_rts.copy())
        features.append(feature)
    return features
