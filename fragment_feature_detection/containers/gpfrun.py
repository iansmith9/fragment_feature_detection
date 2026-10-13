import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import h5py
import numpy as np
import scipy.sparse as sps
from scipy.spatial import distance

from ..config import Config
from ..discretization import MzDiscretize
from ..fitpeaks import approximate_overlap_curves
from ..storage import StorageOptions, create_dataset
from ..utils import pivot_unique_binned_mz_sparse
from .scanwindow import ScanWindow

logger = logging.getLogger(__name__)


class GPFRun:
    """All MS2 scans of one isolation window (GPF target) over the whole run.

    The containers form a three-level hierarchy:

    - ``MSRun``: the whole injection, one ``GPFRun`` per isolation target.
    - ``GPFRun`` (this class): one isolation target. Holds the pivoted scan x m/z-bin
      intensity matrix for every MS2 scan of that target, and cuts it into
      overlapping time tiles.
    - ``ScanWindow``: one time tile (default 150 consecutive scans). NMF, Gaussian
      peak fits and MS1 matching all happen per tile.

    Tiles start every ``window_scan_overlap`` scans (the stride, default 30) and are
    ``window_scan_width`` scans wide (default 150), so consecutive tiles share 120
    scans and each scan sits in up to 5 tiles. The same peptide is therefore usually
    found in several tiles; ``collapse_redundant_components`` masks the duplicates.

    In lazy mode the parent ``MSRun`` keeps this object in its HDF5 file under
    ``gpf_runs/gpf_<index>``, and writes scan windows back to it whenever they change
    (see ``set_scan_windows`` and ``persist_scan_windows``).

    Attributes:
        _msrun (Optional[MSRun]): Parent run, set by ``MSRun.add_gpf``/``get_gpf``.
            Used to find the HDF5 file in lazy mode.
        _gpf (float): Isolation window target m/z.
        _m_long (np.ndarray): Deduplicated raw peaks with columns
            [scan_number, retention_time, mz, intensity]. Used at export to report
            the observed spectrum at a component's apex scan.
        _m (scipy.sparse.csr_matrix): Scan x m/z-bin intensity matrix (rows are
            scans in acquisition order, columns follow ``_bin_indices``).
        _scan_number (np.ndarray): Scan number of each row of ``_m``.
        _bin_indices (np.ndarray): Global m/z bin index of each column of ``_m``.
        _retention_time (np.ndarray): Retention time of each row of ``_m``.
        _discretize (MzDiscretize): The m/z binning used, to turn bin indices back
            into m/z values.
        _window_scan_overlap (int): Stride between tile starts, in scans (despite
            the name, this is not the overlap).
        _window_scan_width (int): Width of each tile in scans.
        _scan_index (np.ndarray): Row position 0..n-1 of each scan in this GPF.
        _gpf_low (Optional[float]): Lower isolation bound, if stored.
        _gpf_high (Optional[float]): Upper isolation bound, if stored.
        _index (Optional[int]): Position of this GPF in the parent MSRun.
        _scan_windows (List[ScanWindow]): The time tiles.
    """

    _msrun = None
    _scan_windows = None

    def __init__(
        self,
        gpf: float,
        m_long: np.ndarray,
        m: sps.csr_matrix,
        scan_number: np.ndarray,
        bin_indices: np.ndarray,
        retention_time: np.ndarray,
        discretize: MzDiscretize,
        window_scan_overlap: int = 30,
        window_scan_width: int = 150,
        gpf_low: Optional[float] = None,
        gpf_high: Optional[float] = None,
        index: Optional[int] = None,
    ):
        """Store the pivoted data for one isolation target and build its tiles.

        Args:
            gpf (float): Isolation window target m/z.
            m_long (np.ndarray): Deduplicated raw peaks
                [scan_number, retention_time, mz, intensity].
            m (sps.csr_matrix): Scan x m/z-bin intensity matrix.
            scan_number (np.ndarray): Scan number of each row of ``m``.
            bin_indices (np.ndarray): Global m/z bin index of each column of ``m``.
            retention_time (np.ndarray): Retention time of each row of ``m``.
            discretize (MzDiscretize): Binning used to build ``m``.
            window_scan_overlap (int): Stride between tile starts, in scans.
            window_scan_width (int): Tile width, in scans.
            gpf_low (Optional[float]): Lower isolation bound.
            gpf_high (Optional[float]): Upper isolation bound.
            index (Optional[int]): Position of this GPF in the parent MSRun.
        """
        self._gpf = gpf
        self._m_long = m_long
        self._m = m
        self._scan_number = scan_number
        self._bin_indices = bin_indices
        self._retention_time = retention_time
        self._discretize = discretize
        self._window_scan_overlap = window_scan_overlap
        self._window_scan_width = window_scan_width
        self._scan_index = np.arange(0, scan_number.shape[0])
        self._gpf_low = gpf_low
        self._gpf_high = gpf_high
        self._index = index

        # Cut the run into tiles immediately (unfiltered, unfit ScanWindows).
        self.build_scan_windows()

    def slice_scan_window(self, start: int, end: int) -> ScanWindow:
        """Build a ScanWindow from rows ``start:end`` of the intensity matrix.

        Every tile gets all m/z bins of the GPF; empty and noisy bins are removed
        later by ``ScanWindow.filter_scans``.

        Args:
            start (int): First row (scan index) of the tile.
            end (int): One past the last row of the tile.

        Returns:
            ScanWindow: New, unfiltered tile.
        """
        return ScanWindow(
            self._gpf,
            self._m[start:end].copy(),
            self._scan_number[start:end].copy(),
            self._scan_index[start:end].copy(),
            self._bin_indices.copy(),
            self._retention_time[start:end].copy(),
            self._discretize.indices_to_mz(self._bin_indices),
            gpf_low=self._gpf_low,
            gpf_high=self._gpf_high,
        )

    @property
    def scan_windows(self) -> List[ScanWindow]:
        """List[ScanWindow]: The time tiles of this GPF, in order of start scan."""
        return self._scan_windows

    @scan_windows.setter
    def scan_windows(self, modified_scan_windows: List[ScanWindow]) -> None:
        """Replace the scan windows and, in lazy mode, write them to h5."""
        self.set_scan_windows(modified_scan_windows, persist=True)

    def set_scan_windows(
        self, modified_scan_windows: List[ScanWindow], persist: bool = True
    ) -> None:
        """Replace the scan windows.

        Parallel stages (NMF, MS1 attachment, matching) send each tile to a worker
        process and get modified copies back; this puts the copies in place.

        Writing to h5 is a separate step so a caller can keep changing the windows
        in memory first. ``fit_nmf_matrix_gpfrun`` relies on this: it assigns the
        fitted windows with ``persist=False``, collapses duplicates, then persists
        once. Persisting before the collapse would save the duplicates, and in lazy
        mode the next ``MSRun.get_gpf`` reload would bring them back.

        Args:
            modified_scan_windows (List[ScanWindow]): Scan windows to assign.
            persist (bool): If True and the parent MSRun is lazy, write the windows
                to h5. Pass False to modify the windows further before calling
                persist_scan_windows.

        Raises:
            ValueError: If an item isn't a ScanWindow, or the number of windows
                differs from the current number.
        """
        if not all(isinstance(sw, ScanWindow) for sw in modified_scan_windows):
            raise ValueError("Modified scan windows not correct object.")
        if not len(modified_scan_windows) == len(self.scan_windows):
            raise ValueError("List of modified scan windows incorrect len.")
        self._scan_windows = modified_scan_windows
        if persist:
            self.persist_scan_windows()

    def persist_scan_windows(self) -> None:
        """Write all scan windows to the parent MSRun's h5 file if it is lazy.

        Each window is written in "update" mode into
        ``gpf_runs/gpf_<index>/scan_windows/window_<i>``; datasets that haven't
        changed are left alone. Does nothing in in-memory mode, where the GPFRun
        object itself is kept by the MSRun.
        """
        if self._msrun is not None and self._msrun._lazy:
            gpf_group = self._msrun.gpf_group(self._index)
            for i, scan_window in enumerate(self._scan_windows):
                self.dump_scanwindow_to_h5(
                    i,
                    scan_window,
                    gpf_group,
                    mode="update",
                    storage=self._msrun._storage,
                )

    @property
    def m_long(self) -> np.ndarray:
        """np.ndarray: Raw peaks [scan_number, retention_time, mz, intensity]."""
        return self._m_long

    def update_scan_window_at_index(
        self, scan_window_index: int, scan_window: ScanWindow
    ) -> None:
        """Replace one scan window and, in lazy mode, write it to h5.

        Args:
            scan_window_index (int): Position of the window to replace.
            scan_window (ScanWindow): New window.
        """
        self._scan_windows[scan_window_index] = scan_window
        if self._msrun is not None and self._msrun._lazy:
            self.dump_scanwindow_to_h5(
                scan_window_index,
                scan_window,
                self._msrun.gpf_group(self._index),
                mode="update",
                storage=self._msrun._storage,
            )

    def build_scan_windows(
        self, assign: bool = True, indices: Optional[np.ndarray] = None
    ) -> Optional[List[ScanWindow]]:
        """Cut the run into overlapping time tiles.

        By default tiles start at 0, stride, 2*stride, ... and one more tile is
        added at ``n_scans - width`` so the last scans are covered. A GPF with fewer
        scans than one tile gets no tiles.

        Args:
            assign (bool): If True, store the tiles on ``_scan_windows``. If False,
                return them instead (used to sample tuning windows).
            indices (Optional[np.ndarray]): Start rows of the tiles. Defaults to
                the regular layout described above.

        Returns:
            Optional[List[ScanWindow]]: The tiles when ``assign`` is False.
        """
        if not isinstance(indices, np.ndarray):
            last_start = self._scan_number.shape[0] - self._window_scan_width
            if last_start < 0:
                logger.warning(
                    f"GPF {self._gpf} has {self._scan_number.shape[0]} scans, fewer "
                    f"than the scan window width {self._window_scan_width}. No scan "
                    "windows built."
                )
            # Regular tile starts, every `stride` scans.
            indices = np.arange(0, max(last_start, 0), self._window_scan_overlap)
            # Add a final window so the last scans are covered by at least one window.
            if last_start >= 0 and (indices.size == 0 or indices[-1] != last_start):
                indices = np.append(indices, last_start)
        if assign:
            self._scan_windows = [
                self.slice_scan_window(i, i + self._window_scan_width) for i in indices
            ]
        else:
            return [
                self.slice_scan_window(i, i + self._window_scan_width) for i in indices
            ]

    def sample_scan_windows(
        self,
        window_sampling_fraction: float = 0.05,
        exclude_scan_window_edges: int = 3,
        rng: np.random.Generator = np.random.default_rng(seed=Config.random_seed),
    ) -> List[ScanWindow]:
        """Randomly sample tiles from this GPF for hyperparameter tuning.

        Candidate tiles use the regular stride, skipping the first and last
        ``exclude_scan_window_edges`` tile positions (the start and end of the
        gradient are usually empty or dominated by contaminants). Each candidate is
        kept independently with probability ``window_sampling_fraction``.

        Args:
            window_sampling_fraction (float): Probability of keeping each candidate.
            exclude_scan_window_edges (int): Number of tile positions to skip at
                each end of the run.
            rng (np.random.Generator): Random generator. Note: the default is
                created once at import, so pass one for reproducible sampling.

        Returns:
            List[ScanWindow]: Newly built (unfiltered) tiles; not stored on self.
        """
        indices = np.arange(
            exclude_scan_window_edges * self._window_scan_overlap,
            self._scan_number.shape[0]
            - exclude_scan_window_edges * self._window_scan_overlap,
            self._window_scan_overlap,
        )
        sample = np.where((rng.random(indices.shape[0]) < window_sampling_fraction))[0]

        return self.build_scan_windows(assign=False, indices=indices[sample])

    def get_overlapping_scanwindow_indices(
        self,
        start: float,
        end: float,
        units: Literal["retention_time", "scan_index", "scan_number"] = "scan_index",
        ignore_indexed: bool = True,
    ) -> List[int]:
        """Find tiles that overlap the interval (start, end).

        Args:
            start (float): Interval start, in ``units``.
            end (float): Interval end, in ``units``.
            units (Literal["retention_time", "scan_index", "scan_number"]): Which
                ScanWindow axis to compare against.
            ignore_indexed (bool): If True, skip any tile that starts exactly at
                ``start``. When ``start`` is a tile's own first scan, this
                excludes that tile from its own overlap list.

        Returns:
            List[int]: Positions in ``_scan_windows`` of the overlapping tiles.

        Raises:
            AttributeError: If ``units`` isn't a ScanWindow attribute.
        """
        if not hasattr(self._scan_windows[0], units):
            raise AttributeError(f"Units not an attribute on ScanWindow: {units}")
        if ignore_indexed:
            return [
                i
                for i, sw in enumerate(self._scan_windows)
                if getattr(sw, units)[0] < end
                and getattr(sw, units)[-1] > start
                and getattr(sw, units)[0] != start
            ]
        return [
            i
            for i, sw in enumerate(self._scan_windows)
            if getattr(sw, units)[0] < end and getattr(sw, units)[-1] > start
        ]

    def get_overlapping_components(
        self,
        start: float,
        end: float,
        units: Literal["retention_time", "scan_index", "scan_number"] = "scan_index",
    ) -> Tuple[np.ndarray]:
        """Collect the elution profiles of all kept components in a time range.

        Puts the W column (elution profile) of every kept component from every tile
        that overlaps (start, end) onto one shared scan axis, e.g. to plot
        components from neighboring tiles together.

        Args:
            start (float): Range start, in ``units``.
            end (float): Range end, in ``units``.
            units (Literal["retention_time", "scan_index", "scan_number"]): Axis
                for ``start`` and ``end``.

        Returns:
            Tuple[np.ndarray]:
                - m: (n_scans, n_components) elution profiles, zero outside the
                  component's own tile.
                - component_names: Name of each column.
                - all_scan_indices: Scan index of each row.
                - all_retention_times: Retention time of each row.
        """

        indices = self.get_overlapping_scanwindow_indices(
            start, end, units=units, ignore_indexed=False
        )

        # Flatten components across the overlapping tiles, remembering which scans
        # each component's profile covers.
        scan_windows = [self._scan_windows[i] for i in indices]
        component_names = np.concatenate([sw.component_names for sw in scan_windows])
        all_retention_times = np.unique(
            [r for sw in scan_windows for r in sw.retention_time]
        )
        w_matrices = [w for sw in scan_windows for w in sw.w.T]
        scan_indices = [
            sw.scan_index for sw in scan_windows for _ in sw.component_names
        ]
        all_scan_indices = np.unique(
            [idx for sw in scan_windows for idx in sw.scan_index]
        )

        # Place each profile at its rows on the shared scan axis.
        m = np.zeros((all_scan_indices.shape[0], component_names.shape[0]))

        for i in range(component_names.shape[0]):
            m[scan_indices[i] - all_scan_indices.min(), i] = w_matrices[i]

        return m, component_names, all_scan_indices, all_retention_times

    def collapse_redundant_components(
        self,
        overlap_threshold: float = 0.5,
        similarity_threshold: float = 0.75,
    ) -> None:
        """Mask components found again in overlapping tiles.

        Each scan sits in up to width/stride tiles, so one peptide is usually
        extracted several times. Two components from overlapping tiles are treated
        as the same species when both hold:

        1. Their fitted Gaussian elution profiles overlap: the fraction of curve 1's
           area under min(curve 1, curve 2) is above ``overlap_threshold``.
        2. Their spectra (H rows, placed on the union of both tiles' m/z bins) have
           cosine similarity above ``similarity_threshold``.

        Of a matching pair, the component whose Gaussian fit has the lower mean
        absolute error is kept, and the other is masked with
        ``ScanWindow.mask_component``. Masking only changes the in-memory
        ``_component_keep`` flags; the caller must persist the windows in lazy mode.

        Requires every tile to be NMF-fit and peak-fit.

        Args:
            overlap_threshold (float): Minimum elution overlap fraction (0-1).
            similarity_threshold (float): Minimum spectral cosine similarity (0-1).
        """
        for sw in self._scan_windows:

            # Other tiles sharing scans with this one (excluding itself).
            ol_scan_indices = self.get_overlapping_scanwindow_indices(
                sw.scan_index[0],
                sw.scan_index[-1],
                units="scan_index",
                ignore_indexed=True,
            )
            # Index the full per-component arrays by component index, so values stay
            # aligned as components are masked during the loop.
            sw_fit_mean, sw_fit_sigma = sw._component_means, sw._component_sigmas

            for oli in ol_scan_indices:
                ol_sw = self._scan_windows[oli]

                # The two tiles kept different m/z bins after filtering. Build a
                # shared bin axis and the positions of each tile's bins on it, so
                # their spectra can be compared element by element.
                union_mz_bin_indices = np.unique(
                    np.concatenate([sw.mz_bin_indices, ol_sw.mz_bin_indices])
                )
                sw_mz_bin_indices = np.searchsorted(
                    union_mz_bin_indices, sw.mz_bin_indices
                )
                sw_component_weights = np.zeros_like(union_mz_bin_indices, dtype=float)
                ol_mz_bin_indices = np.searchsorted(
                    union_mz_bin_indices, ol_sw.mz_bin_indices
                )
                ol_component_weights = np.zeros_like(union_mz_bin_indices, dtype=float)

                ol_fit_mean = ol_sw._component_means
                ol_fit_sigma = ol_sw._component_sigmas
                ol_component_indices = ol_sw.component_indices

                for ni in sw.component_indices:
                    # Component may have been masked while comparing to an earlier
                    # window.
                    if not sw._component_keep[ni]:
                        continue

                    # This component's spectrum on the shared bin axis.
                    sw_component_weights[:] = 0.0
                    sw_component_weights[sw_mz_bin_indices] = sw._h[ni]

                    for nj in ol_component_indices:
                        if not ol_sw._component_keep[nj]:
                            continue

                        # Gaussians this far apart can't pass the overlap threshold.
                        if abs(sw_fit_mean[ni] - ol_fit_mean[nj]) > 3 * (
                            sw_fit_sigma[ni] + ol_fit_sigma[nj]
                        ):
                            continue

                        # Integrate over +/- 6 sigma (bounds are in retention time).
                        max_sigma = max(sw_fit_sigma[ni], ol_fit_sigma[nj])
                        overlap_density = approximate_overlap_curves(
                            sw_fit_mean[ni],
                            sw_fit_sigma[ni],
                            ol_fit_mean[nj],
                            ol_fit_sigma[nj],
                            bounds=(
                                min(sw_fit_mean[ni], ol_fit_mean[nj]) - 6 * max_sigma,
                                max(sw_fit_mean[ni], ol_fit_mean[nj]) + 6 * max_sigma,
                            ),
                        )
                        if overlap_density <= overlap_threshold:
                            continue

                        # Elution matches; now compare spectra by cosine distance
                        # (1 - cosine similarity).
                        ol_component_weights[:] = 0.0
                        ol_component_weights[ol_mz_bin_indices] = ol_sw._h[nj]

                        component_distance = distance.cosine(
                            sw_component_weights, ol_component_weights
                        )
                        if (overlap_density > overlap_threshold) and (
                            component_distance < 1 - similarity_threshold
                        ):
                            # Same species: keep the better Gaussian fit (lower
                            # MAE). If this tile's component loses, stop comparing
                            # it against the other tile's components.
                            if sw._component_maes[ni] < ol_sw._component_maes[nj]:
                                ol_sw.mask_component(nj)
                            else:
                                sw.mask_component(ni)
                                break

    def query_component_by_name(
        self,
        component: str,
    ) -> List[Dict[str, Any]]:
        """Look up a component by name in this GPF's tiles.

        Component names have the form
        ``<gpf>_<component index>-<apex scan index>_<mean RT>_<sigma>``.

        Args:
            component (str): Component name.

        Returns:
            List[Dict[str, Any]]: Matches from the first tile containing the name,
            or an empty list. ``ScanWindow.query_component_info`` is still a stub
            and returns empty dicts.
        """
        for sw in self.scan_windows:
            if sw.is_component_fit and (sw.component_names == component).any():
                return sw.query_component_info(
                    component,
                    by="name",
                )
        return []

    def dump_h5(
        self,
        h5_fh: Union[str, h5py.File, h5py.Group],
        config: Config = Config(),
        storage: Optional[StorageOptions] = None,
    ) -> None:
        """Save the GPFRun object to an HDF5 file.

        Layout of the group: attributes (gpf, isolation bounds, tile width/stride,
        index; ``None`` is stored as -1), the scan and bin axes, the CSR intensity
        matrix as ``m_data``/``m_indices``/``m_indptr`` plus the ``m_shape``
        attribute, ``m_long``, the ``discretize`` group, and one
        ``scan_windows/window_<i>`` group per tile.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object
            config (Config): Run configuration; ``config.h5`` controls whether each
                tile's unfiltered matrix is saved too. Used when ``storage`` is None.
            storage (Optional[StorageOptions]): How to write the data (compact layout,
                compression, unfiltered matrices). Defaults to the options from
                ``config``.
        """
        if storage is None:
            storage = StorageOptions.from_config(config)
        compression = storage.compression
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "w")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Save main attributes (h5py can't store None, so -1 stands in for it).
        f.attrs["gpf"] = self._gpf
        f.attrs["gpf_low"] = self._gpf_low if self._gpf_low is not None else -1
        f.attrs["gpf_high"] = self._gpf_high if self._gpf_high is not None else -1
        f.attrs["window_scan_overlap"] = self._window_scan_overlap
        f.attrs["window_scan_width"] = self._window_scan_width
        f.attrs["index"] = self._index if self._index is not None else -1

        # Save arrays
        create_dataset(f, "scan_number", self._scan_number, compression)
        create_dataset(f, "scan_index", self._scan_index, compression)
        create_dataset(f, "bin_indices", self._bin_indices, compression)
        create_dataset(f, "retention_time", self._retention_time, compression)

        # Save sparse matrix as its three CSR component arrays.
        create_dataset(f, "m_data", self._m.data, compression)
        create_dataset(f, "m_indices", self._m.indices, compression)
        create_dataset(f, "m_indptr", self._m.indptr, compression)
        f.attrs["m_shape"] = self._m.shape

        # Save m_long array
        create_dataset(f, "m_long", self._m_long, compression)

        # Save discretize object
        discretize_grp = f.create_group("discretize")
        self._discretize.dump_h5(
            discretize_grp, shared=storage.compact, compression=compression
        )

        # Save scan windows
        if self._scan_windows is not None:
            windows_grp = f.create_group("scan_windows")
            for i, window in enumerate(self._scan_windows):
                window.dump_h5(
                    windows_grp,
                    f"window_{i}",
                    save_m_unfilter=storage.save_m_unfilter,
                    compact=storage.compact,
                    compression=compression,
                )

        if close_file:
            f.close()

    def dump_scanwindow_to_h5(
        self,
        scanwindow_index: int,
        scan_window: ScanWindow,
        h5_fh: Union[str, h5py.File, h5py.Group],
        mode: Literal["write", "update"] = "update",
        storage: Optional[StorageOptions] = None,
    ) -> None:
        """
        Update/append a new scanwindow run to an existing HDF5 file.

        Writes to ``<h5_fh>/scan_windows/window_<scanwindow_index>``. The
        unfiltered matrix is never saved here, to keep the file small.

        Args:
            scanwindow_index (int): Index for the scan window
            scan_window (ScanWindow): Scan window to save
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to the HDF5 file, or the
                GPF's group (``gpf_runs/gpf_<i>``).
            mode (Literal["write", "update"]): "update" rewrites only datasets that
                changed; "write" creates the group from scratch.
            storage (Optional[StorageOptions]): Layout and compression. Defaults to
                the standard layout without compression.
        """
        if storage is None:
            storage = StorageOptions()
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "a")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        scan_window.dump_h5(
            f["scan_windows"],
            f"window_{scanwindow_index}",
            mode=mode,
            save_m_unfilter=False,
            compact=storage.compact,
            compression=storage.compression,
        )

        if close_file:
            f.close()

    @classmethod
    def from_h5(cls, h5_fh: Union[str, h5py.File, h5py.Group]) -> "GPFRun":
        """Load a GPFRun object from an HDF5 file.

        The constructor rebuilds fresh tiles, which are then replaced by the saved
        ``scan_windows`` (with their filtering, fits and matches) if present.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object

        Returns:
            GPFRun: Loaded object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "r")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Load discretize object
        discretize = MzDiscretize.from_h5(f["discretize"])

        # Load sparse matrix
        m = sps.csr_matrix(
            (f["m_data"][:], f["m_indices"][:], f["m_indptr"][:]),
            shape=f.attrs["m_shape"],
        )

        # Load m_long array
        m_long = f["m_long"][:]

        # Create instance (-1 attributes map back to None).
        obj = cls(
            gpf=f.attrs["gpf"],
            m_long=m_long,
            m=m,
            scan_number=f["scan_number"][:],
            bin_indices=f["bin_indices"][:],
            retention_time=f["retention_time"][:],
            discretize=discretize,
            window_scan_overlap=f.attrs["window_scan_overlap"],
            window_scan_width=f.attrs["window_scan_width"],
            gpf_low=None if f.attrs["gpf_low"] == -1 else f.attrs["gpf_low"],
            gpf_high=None if f.attrs["gpf_high"] == -1 else f.attrs["gpf_high"],
            index=None if f.attrs["index"] == -1 else f.attrs["index"],
        )

        # Load scan windows if they exist, replacing the freshly built tiles.
        if "scan_windows" in f:
            obj._scan_windows = []
            windows_grp = f["scan_windows"]
            # Compact windows don't store the GPF-wide axis; pass it in so they can
            # rebuild it (ignored for windows in the standard layout).
            gpf_bin_indices = obj._bin_indices
            gpf_mz = discretize.indices_to_mz(gpf_bin_indices)
            for i in range(len(windows_grp)):
                window = ScanWindow.from_h5(
                    windows_grp,
                    f"window_{i}",
                    gpf_bin_indices=gpf_bin_indices,
                    gpf_mz=gpf_mz,
                )
                obj._scan_windows.append(window)

        if close_file:
            f.close()

        return obj

    @classmethod
    def from_undiscretized_long(
        cls: Type["GPFRun"],
        gpf: float,
        m: np.ndarray,
        discretize: MzDiscretize,
        index: Optional[int] = None,
        config: Config = Config(),
    ) -> "GPFRun":
        """Build a GPFRun from raw long-format peaks.

        Selects the rows of one isolation target, bins their m/z, and passes the
        result to ``from_discretized_long``.

        Args:
            gpf (float): Isolation window target m/z to select.
            m (np.ndarray): Raw peaks, one row per peak, with columns
                [gpf, scan_number, retention_time, mz, intensity] and, when
                ``include_gpf_bounds`` is set, [isolation_low, isolation_high].
            discretize (MzDiscretize): m/z binning.
            index (Optional[int]): Position of this GPF in the MSRun.
            config (Config): Run configuration.

        Returns:
            GPFRun: New object with its tiles built.
        """
        sub_m = m[np.isclose(m[:, 0], gpf)]
        # Appends [bin_index, bin_right_edge]. Each peak appears once per grid
        # (steps), so with 2 grids every peak is in 2 rows.
        discretized_m = discretize.discretize_mz_array(sub_m)

        return cls.from_discretized_long(
            gpf,
            discretized_m,
            discretize,
            config=config,
            index=index,
        )

    @classmethod
    def from_discretized_long(
        cls: Type["GPFRun"],
        gpf: float,
        m: np.ndarray,
        discretize: MzDiscretize,
        index: Optional[int] = None,
        config: Config = Config(),
    ) -> "GPFRun":
        """Build a GPFRun from binned long-format peaks.

        Args:
            gpf (float): Isolation window target m/z to select.
            m (np.ndarray): Binned peaks with columns [gpf, scan_number,
                retention_time, mz, intensity, (isolation_low, isolation_high),
                bin_index, bin_right_edge]. The bounds columns are present only
                when ``config.ms2_preprocessing.include_gpf_bounds`` is True.
            discretize (MzDiscretize): The binning that produced ``bin_index``.
            index (Optional[int]): Position of this GPF in the MSRun.
            config (Config): Run configuration (tile width/stride, downcasting).

        Returns:
            GPFRun: New object with its tiles built.
        """
        sub_m = m[np.isclose(m[:, 0], gpf)]
        # Column holding the bin index: right after the raw columns, which are 2
        # wider when the isolation bounds are stored.
        mz_bin_index = 5
        gpf_low = None
        gpf_high = None
        if config.ms2_preprocessing.include_gpf_bounds:
            mz_bin_index += 2
            gpf_low = sub_m[:, 5][0]
            gpf_high = sub_m[:, 6][0]
        # Pivot to a scan x bin CSR matrix. Peaks landing in the same bin of the
        # same scan are summed.
        sub_m_p, scans, bin_indices = pivot_unique_binned_mz_sparse(
            sub_m, mz_bin_index=mz_bin_index
        )

        # Deduplicate the discretized values so we can pull ms2 data from specific
        # scans.
        # Columns [scan_number, retention_time, mz, intensity]; this undoes the
        # one-row-per-grid duplication from discretization.
        sub_m_dedup = np.unique(sub_m[:, 1:5], axis=0)

        if config.downcast_intensities:
            sub_m_p = sub_m_p.astype(np.float32)
            sub_m_dedup = sub_m_dedup.astype(np.float32)

        # One retention time per unique scan, in the same (sorted) order as `scans`.
        retention_times = np.unique(sub_m[:, [1, 2]], axis=0)[:, 1]

        return cls(
            gpf,
            sub_m_dedup,
            sub_m_p,
            scans,
            bin_indices,
            retention_times,
            discretize,
            gpf_low=gpf_low,
            gpf_high=gpf_high,
            window_scan_overlap=config.scan_filter.scan_overlap,
            window_scan_width=config.scan_filter.scan_width,
            index=index,
        )
