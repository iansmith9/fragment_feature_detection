import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import h5py
import numpy as np
from tqdm import tqdm

from ..config import Config
from ..discretization import MzDiscretize
from ..storage import StorageOptions
from .gpfrun import GPFRun
from .scanwindow import ScanWindow

logger = logging.getLogger(__name__)


class MSRun:
    """A class representing a complete mass spectrometry run with multiple GPF windows.

    Top of the container hierarchy: an ``MSRun`` holds one ``GPFRun`` per isolation
    target (keyed 0..n-1 in order of increasing target m/z), and each ``GPFRun``
    holds its time tiles (``ScanWindow``).

    Two storage modes:

    - **In memory** (``lazy=False``): ``_gpf_runs`` maps each index to the GPFRun
      object, and all data stays in RAM.
    - **Lazy** (``lazy=True``): ``_gpf_runs`` maps each index to the HDF5 group name
      ``"gpf_runs/gpf_<i>"``. ``get_gpf`` rebuilds a fresh GPFRun from the file on
      every call, and GPFRun writes changed tiles back through
      ``persist_scan_windows``. Only one GPF needs to be in RAM at a time, but each
      pipeline stage re-reads it, and changes made to a loaded GPFRun that aren't
      persisted are lost on the next ``get_gpf``.

    HDF5 layout (lazy mode, or after ``dump_h5``)::

        /                       attrs: isolation_low, isolation_high (-1 = None),
                                gpf_files (JSON index of part files, if split)
        /gpf_runs/gpf_<i>/      one GPFRun (see GPFRun.dump_h5)
        /<target m/z>/ms2_long  converted mzML data, when the run shares the file
                                with the converter output
        /shared_arrays/         arrays stored once per file (compact storage)

    With ``config.h5.gpfs_per_file = N > 0``, lazy runs write each GPFRun into a part
    file ``<stem>.gpf_runs/part_<k>.h5`` (``k = gpf_index // N``) next to the main
    file, with the same ``/gpf_runs/gpf_<i>`` layout inside. The main file's
    ``gpf_files`` attribute maps each GPF index to its part file, and ``gpf_group``
    resolves it, so loading and every pipeline stage work the same either way.

    Attributes:
        _gpf_runs (Dict[int, Union[GPFRun, str]]): GPF index -> GPFRun (in memory)
            or HDF5 group name (lazy).
        _isolation_low (Optional[float]): Lower bound of isolation window.
        _isolation_high (Optional[float]): Upper bound of isolation window.
        _lazy (bool): Whether GPFRuns are kept in the HDF5 file.
        _h5_file (Optional[h5py.File]): Open handle (mode "a") on the backing file.
    """

    _gpf_runs = None
    _h5_file = None
    _part_handles = None

    # Attribute on the main file holding the GPF index -> part file mapping.
    GPF_FILES_ATTR = "gpf_files"

    def __init__(
        self,
        isolation_low: Optional[float] = None,
        isolation_high: Optional[float] = None,
        h5_fh: Optional[Union[str, Path]] = None,
        lazy: bool = False,
        config: Optional[Config] = None,
    ):
        """Create an empty run, optionally backed by an HDF5 file.

        Args:
            isolation_low (Optional[float]): Lower isolation bound for the run.
            isolation_high (Optional[float]): Upper isolation bound for the run.
            h5_fh (Optional[Union[str, Path]]): Backing HDF5 file, opened in append
                mode. Required when ``lazy`` is True.
            lazy (bool): Keep GPFRuns in the HDF5 file instead of in memory.
            config (Optional[Config]): Run configuration. Only ``config.h5`` is used,
                to decide how GPFRuns are written (compact storage, part files).
                Defaults to ``Config()``.

        Raises:
            ValueError: If ``lazy`` is True and no ``h5_fh`` is given.
        """
        if config is None:
            config = Config()

        self._isolation_low = isolation_low
        self._isolation_high = isolation_high
        self._lazy = lazy
        self._gpf_runs = {}

        # How GPFRuns are written to HDF5. Layout only: it never changes results.
        self._storage = StorageOptions.from_config(config)
        self._gpfs_per_file = int(config.h5.get("gpfs_per_file", 0) or 0)
        # Split runs: GPF index -> part file path (relative to the main file's
        # folder), and the open handles on those part files.
        self._gpf_files: Dict[int, str] = {}
        self._part_handles: Dict[str, h5py.File] = {}
        self._h5_path: Optional[Path] = None

        if lazy and h5_fh is None:
            raise ValueError("If lazy, must provide h5_fh.")
        if h5_fh:
            self._h5_path = Path(h5_fh)
            self._h5_file = h5py.File(h5_fh, "a", **self._file_kwargs())
            # Instantiate all of the appropriate groups for writing to h5.
            self.setup_h5(self._h5_file)
            self._gpf_files = self.read_gpf_files(self._h5_file)

    def __del__(self) -> None:
        """Close h5 files if open handles exist on instance."""
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Close the backing HDF5 file(s), flushing any pending writes."""
        for handle in (self._part_handles or {}).values():
            if handle:
                handle.close()
        if self._h5_file:
            self._h5_file.close()

    def _file_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for opening the run's HDF5 files for writing.

        Compact storage writes new objects in the HDF5 1.10 format, whose object
        headers and chunk indexes are smaller. Such files need HDF5 1.10 or newer
        (any recent h5py) to read. The default storage keeps the library default.
        """
        return {"libver": ("v110", "latest")} if self._storage.compact else {}

    @classmethod
    def read_gpf_files(cls, h5_file: h5py.File) -> Dict[int, str]:
        """Read the GPF index -> part file mapping stored in a main HDF5 file.

        Args:
            h5_file (h5py.File): The run's main HDF5 file.

        Returns:
            Dict[int, str]: Part file path (relative to the main file's folder) for
            each GPF stored in a part file. Empty if the run isn't split.
        """
        if cls.GPF_FILES_ATTR not in h5_file.attrs:
            return {}
        index = json.loads(h5_file.attrs[cls.GPF_FILES_ATTR])
        return {int(k): v for k, v in index.items()}

    def _write_gpf_files(self) -> None:
        """Store the GPF index -> part file mapping on the main file."""
        self._h5_file.attrs[self.GPF_FILES_ATTR] = json.dumps(
            {str(k): v for k, v in sorted(self._gpf_files.items())}
        )

    def _part_file(self, relative_path: str) -> h5py.File:
        """Open (or reuse) a part file, creating it and its folder if needed.

        Args:
            relative_path (str): Part file path relative to the main file's folder.

        Returns:
            h5py.File: Handle opened in append mode, with a ``gpf_runs`` group.
        """
        handle = self._part_handles.get(relative_path)
        if handle is None or not handle.id.valid:
            path = self._h5_path.parent / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = h5py.File(path, "a", **self._file_kwargs())
            if "gpf_runs" not in handle:
                handle.create_group("gpf_runs")
            self._part_handles[relative_path] = handle
        return handle

    def gpf_group(self, gpf_index: int) -> h5py.Group:
        """Return the HDF5 group of a lazily stored GPF.

        Looks in the GPF's part file when the run is split, otherwise in the main
        file. Every place that reads or writes a stored GPFRun goes through here.

        Args:
            gpf_index (int): Index of the GPF.

        Returns:
            h5py.Group: The ``gpf_runs/gpf_<gpf_index>`` group.
        """
        name = f"gpf_{gpf_index}"
        if gpf_index in self._gpf_files:
            return self._part_file(self._gpf_files[gpf_index])["gpf_runs"][name]
        return self._h5_file["gpf_runs"][name]

    def get_gpf(self, gpf_index: int) -> GPFRun:
        """Get or retrieve GPFrun from disk.

        In lazy mode this reads the GPFRun from HDF5 every time, so the returned
        object is a fresh copy; changes only survive if they are persisted (for
        example through the ``GPFRun.scan_windows`` setter).

        Args:
            gpf_index (int): Index of GPF run to retrieve

        Returns:
            GPFRun: The requested GPF run

        Raises:
            KeyError: If GPF index doesn't exist
        """
        if gpf_index not in self._gpf_runs:
            raise KeyError(f"GPF index {gpf_index} not present in MSRun.")

        # Lazy: the stored value is the HDF5 group name, load it.
        if isinstance(self._gpf_runs[gpf_index], str):
            gpf = GPFRun.from_h5(self.gpf_group(gpf_index))
            # Link back so the GPFRun can write its windows to this file.
            gpf._msrun = self
            return gpf

        return self._gpf_runs[gpf_index]

    def add_gpf(
        self,
        gpf_index: int,
        gpf: Union[GPFRun, None],
        remove_if_exists: bool = False,
        config: Config = Config(),
    ) -> None:
        """Register a GPFRun with this run.

        Optionally filters every tile first
        (``config.h5.scan_window_filter_during_load``), so later stages and the HDF5 copy hold the smaller, filtered matrices. In
        lazy mode the GPFRun is written to ``gpf_runs/gpf_<gpf_index>`` and only the
        group name is kept.

        Args:
            gpf_index (int): Index for the GPF in this run.
            gpf (Union[GPFRun, None]): The GPFRun. ``None`` is allowed in lazy mode
                when the group already exists in the file (used by ``from_h5``).
            remove_if_exists (bool): In lazy mode, replace an existing group with the
                same index. If False, an existing group is kept as is.
            config (Config): Run configuration.
        """
        # Maintain a relationship between msrun-gpf objects.
        if gpf is not None:
            gpf._msrun = self

        # Filter all tiles now (drop empty/constant bins, scale, denoise, ...).
        if config.h5.scan_window_filter_during_load and gpf is not None:
            for sw in gpf.scan_windows:
                # Windows reloaded by from_h5 are already filtered. Filtering again
                # would max-scale and percentile-filter them a second time.
                if not sw.is_filtered:
                    sw.filter_scans()
        if not self._lazy:
            self._gpf_runs[gpf_index] = gpf
        else:
            group_name = f"gpf_runs/gpf_{gpf_index}"
            # gpf is None when registering a GPF that is already stored (from_h5);
            # its location comes from the file's part-file index.
            if gpf is not None:
                # Where to write: a part file when splitting, else the main file.
                if self._gpfs_per_file > 0:
                    part_number = gpf_index // self._gpfs_per_file
                    relative_path = (
                        f"{self._h5_path.stem}.gpf_runs/part_{part_number:04d}.h5"
                    )
                    target = self._part_file(relative_path)
                else:
                    relative_path = None
                    target = self._h5_file

                if remove_if_exists:
                    # Remove a copy stored elsewhere by an earlier run that used a
                    # different gpfs_per_file, so only one copy of the GPF remains.
                    old_path = self._gpf_files.get(gpf_index)
                    if old_path is not None and old_path != relative_path:
                        old_file = self._part_file(old_path)
                        if group_name in old_file:
                            del old_file[group_name]
                    if relative_path is not None and group_name in self._h5_file:
                        del self._h5_file[group_name]
                    if group_name in target:
                        del target[group_name]

                if group_name not in target:
                    self.dump_gpfrun_to_h5(gpf_index, gpf, target)

                if relative_path is None:
                    self._gpf_files.pop(gpf_index, None)
                else:
                    self._gpf_files[gpf_index] = relative_path
                self._write_gpf_files()
            self._gpf_runs[gpf_index] = group_name

    def get_tuning_windows(self, config: Config = Config()) -> List[np.ndarray]:
        """Sample scan windows from all GPFs to use for hyperparameter tuning.

        Tiles are sampled at random from every GPF, avoiding the first and last
        ``config.tuning.exclude_scan_window_edges`` tile positions. With
        ``window_sampling_n`` set, the per-tile sampling probability is chosen so
        that about that many tiles are drawn in total; otherwise
        ``window_sampling_fraction`` is used directly.

        Args:
            config (Config): Run configuration (``tuning`` and ``scan_filter``).

        Returns:
            List[np.ndarray]: Filtered, dense (scans x bins) intensity matrices.
            Only full-width tiles with at least one m/z bin are returned. Empty if
            no tiles are left after excluding the edges.
        """
        rng = np.random.default_rng(seed=config.random_seed)

        windows = []

        if "window_sampling_n" in config.tuning:
            # Count the candidate windows (after excluding edges) across all gpfs and
            # calculate a window sampling fraction that roughly gets you to the
            # required number of windows.
            n_candidates = 0
            for gpf_index, gpf_run in self._gpf_runs.items():
                if isinstance(gpf_run, str):
                    # Lazy, count windows in h5 without loading the GPFRun.
                    n_windows = len(self.gpf_group(gpf_index)["scan_windows"])
                else:
                    n_windows = len(gpf_run.scan_windows)
                n_candidates += max(
                    n_windows - config.tuning.exclude_scan_window_edges * 2, 0
                )
            if n_candidates == 0:
                logger.warning(
                    "No candidate tuning windows after excluding "
                    f"{config.tuning.exclude_scan_window_edges} windows at each edge. "
                    "Lower tuning.exclude_scan_window_edges for short runs."
                )
                return windows
            window_sampling_fraction = min(
                config.tuning.window_sampling_n / n_candidates, 1.0
            )
        else:
            window_sampling_fraction = config.tuning.window_sampling_fraction

        # Draw tiles from every GPF with the same generator, so the overall sample
        # is reproducible for a given random_seed.
        for gpf_index in tqdm(self._gpf_runs, disable=(not config.tqdm_enabled)):
            gpf = self.get_gpf(gpf_index)
            scan_windows = gpf.sample_scan_windows(
                window_sampling_fraction=window_sampling_fraction,
                exclude_scan_window_edges=config.tuning.exclude_scan_window_edges,
                rng=rng,
            )

            # Sampled tiles are freshly sliced, so they still need filtering.
            if not all(sw.is_filtered for sw in scan_windows):
                for sw in scan_windows:
                    sw.filter_scans()

            # Keep only full-width tiles with data; NMF tuning compares like sizes.
            windows.extend(
                [
                    sw.m.copy()
                    for sw in scan_windows
                    if sw.m.shape[1] > 0
                    and sw.m.shape[0] == config.scan_filter.scan_width
                ]
            )

        return windows

    def query_components_by_name(
        self,
        components: List[str],
    ) -> List[Dict[str, Any]]:
        """Look up components by name across all GPFs.

        In lazy mode this scans the HDF5 component names without loading whole
        GPFRuns, and loads only the tiles that contain a match.

        Args:
            components (List[str]): Component names to find.

        Returns:
            List[Dict[str, Any]]: One entry per match. Note:
            ``ScanWindow.query_component_info`` is still a stub that returns empty
            dicts.
        """
        matching_components = []

        if not self._lazy:
            for gpf_index in self._gpf_runs:
                gpf = self.get_gpf(gpf_index)
                for c in components:
                    matching_components.extend(gpf.query_component_by_name(c))
        else:
            for gpf_index in self._gpf_runs:
                gpf_group = self.gpf_group(gpf_index)

                if "scan_windows" not in gpf_group:
                    continue

                # GPF axis, needed to load windows stored in the compact layout.
                gpf_bin_indices = gpf_group["bin_indices"][:]
                gpf_mz = MzDiscretize.from_h5(gpf_group["discretize"]).indices_to_mz(
                    gpf_bin_indices
                )

                for sw_index in gpf_group["scan_windows"]:
                    sw_group = gpf_group["scan_windows"][sw_index]

                    if "component_names" not in sw_group:
                        continue

                    component_names = sw_group["component_names"][:].astype("U")

                    for c in components:
                        idx = np.where(c == component_names)[0]

                        if len(idx) > 0:
                            sw = ScanWindow.from_h5(
                                gpf_group["scan_windows"],
                                sw_index,
                                gpf_bin_indices=gpf_bin_indices,
                                gpf_mz=gpf_mz,
                            )
                            matching_components.extend(
                                sw.query_component_info(c, by="name")
                            )

        return matching_components

    @classmethod
    def from_h5_long(
        cls: Type["MSRun"],
        h5_fh: Union[Path, str],
        gpf_masses: Optional[List[str]] = None,
        discretize: Optional[MzDiscretize] = None,
        msrun_h5_fh: Optional[Union[Path, str]] = None,
        config: Config = Config(),
    ) -> "MSRun":
        """Build an MSRun from the converter's HDF5 output.

        ``MzMLParser.to_ms2_h5`` writes one top-level group per isolation target,
        named by its target m/z, each holding an ``ms2_long`` dataset with one row
        per peak: [gpf, scan_number, retention_time, mz, intensity] plus
        [isolation_low, isolation_high] when ``include_gpf_bounds`` is set. Each
        target becomes a GPFRun (binned, pivoted and tiled), added in order of
        increasing target m/z.

        Args:
            h5_fh (Union[Path, str]): Converter output file.
            gpf_masses (Optional[List[str]]): Only load these targets (group names).
            discretize (Optional[MzDiscretize]): m/z binning. Defaults to one built
                from ``config.discretization``.
            msrun_h5_fh (Optional[Union[Path, str]]): Backing file for the MSRun.
                Defaults to ``h5_fh``, so in lazy mode the GPFRuns are added to the
                converter's file under ``gpf_runs/``.
            config (Config): Run configuration.

        Returns:
            MSRun: The run, with every GPF added (and filtered if configured).
        """
        if discretize is None:
            discretize = MzDiscretize.from_config(config)

        # If msrun_h5_fh is not provided, use the same file as input
        if msrun_h5_fh is None:
            msrun_h5_fh = h5_fh

        msrun = cls(
            lazy=config.lazy,
            h5_fh=msrun_h5_fh,
            config=config,
        )

        # If using the same file, reuse the handle
        if Path(h5_fh) == Path(msrun_h5_fh) and msrun._h5_file is not None:
            f = msrun._h5_file
            close_file = False
        else:
            f = h5py.File(h5_fh, "r")
            close_file = True

        # Converter groups are the ones holding an `ms2_long` dataset (this skips
        # `gpf_runs`). Names are the target m/z, sorted numerically.
        unique_gpfs = [k for k in f.keys() if "ms2_long" in f[k].keys()]
        if gpf_masses is not None:
            unique_gpfs = [k for k in unique_gpfs if k in gpf_masses]

        unique_gpfs = sorted(unique_gpfs, key=lambda x: float(x))

        for i, gpf_mass in tqdm(
            enumerate(unique_gpfs),
            disable=(not config.tqdm_enabled),
            total=len(unique_gpfs),
        ):
            logger.info(f"Building GPFRun: {i}/{len(unique_gpfs)}...")
            sub_m = f[gpf_mass]["ms2_long"][:]

            gpf = GPFRun.from_undiscretized_long(
                float(gpf_mass),
                sub_m,
                discretize,
                config=config,
                index=i,
            )
            # Replace any GPFRun left in the file by an earlier run.
            msrun.add_gpf(i, gpf, remove_if_exists=True, config=config)

        if close_file:
            f.close()

        return msrun

    @classmethod
    def from_long(
        cls: Type["MSRun"],
        m: np.ndarray,
        discretize: Optional[MzDiscretize] = None,
        msrun_h5_fh: Optional[Union[Path, str]] = None,
        config: Config = Config(),
    ) -> "MSRun":
        """Build an MSRun from an in-memory long-format peak array.

        Args:
            m (np.ndarray): One row per peak, columns [gpf, scan_number,
                retention_time, mz, intensity] plus [isolation_low, isolation_high]
                when ``include_gpf_bounds`` is set.
            discretize (Optional[MzDiscretize]): m/z binning. Defaults to one built
                from ``config.discretization``.
            msrun_h5_fh (Optional[Union[Path, str]]): Backing HDF5 file. Required
                when ``config.lazy`` is True.
            config (Config): Run configuration.

        Returns:
            MSRun: The run, with one GPFRun per unique value of column 0.
        """

        if discretize is None:
            discretize = MzDiscretize.from_config(config)

        msrun = cls(lazy=config.lazy, h5_fh=msrun_h5_fh, config=config)

        unique_gpfs = np.unique(m[:, 0])

        for i, gpf_mass in tqdm(
            enumerate(unique_gpfs),
            disable=(not config.tqdm_enabled),
            total=unique_gpfs.size,
        ):

            sub_m = m[np.isclose(m[:, 0], gpf_mass)]

            gpf = GPFRun.from_undiscretized_long(
                gpf_mass,
                sub_m,
                discretize,
                config=config,
                index=i,
            )
            msrun.add_gpf(i, gpf, remove_if_exists=True, config=config)

        return msrun

    def dump_gpfrun_to_h5(
        self,
        gpf_index: int,
        gpf_run: GPFRun,
        h5_fh: Union[str, h5py.File],
    ) -> None:
        """
        Append a new GPF run to an existing HDF5 file.

        Args:
            gpf_index (int): Index for the GPF run
            gpf_run (GPFRun): GPF run to save
            h5_fh (Union[str, h5py.File]): Path to, or open handle on, a file that
                already has a ``gpf_runs`` group (see ``setup_h5``).

        Raises:
            ValueError: If GPF index already exists
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "a")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        if f"gpf_runs/gpf_{gpf_index}" in f:
            raise ValueError(f"GPF index {gpf_index} already exists in HDF5 file")

        gpf_run.dump_h5(
            f["gpf_runs"].create_group(f"gpf_{gpf_index}"), storage=self._storage
        )

        if close_file:
            f.close()

    def setup_h5(self, h5_file: h5py.File) -> None:
        """Setup h5 if not instantiated previously.

        Writes the isolation-bound attributes (-1 stands for None) and creates the
        ``gpf_runs`` group, leaving anything already present untouched.

        Args:
            h5_file (h5py.File): h5py object
        """
        # Save attributes if they don't exist yet.
        if "isolation_low" not in h5_file.attrs:
            h5_file.attrs["isolation_low"] = (
                self._isolation_low if self._isolation_low is not None else -1
            )
        if "isolation_high" not in h5_file.attrs:
            h5_file.attrs["isolation_high"] = (
                self._isolation_high if self._isolation_high is not None else -1
            )
        if "gpf_runs" not in h5_file:
            h5_file.create_group("gpf_runs")

    def dump_h5(self, h5_fh: Union[str, h5py.File]) -> None:
        """Save the MSRun object to an HDF5 file.

        Intended for in-memory runs: writes every GPFRun object under
        ``gpf_runs/gpf_<i>`` of this one file (``gpfs_per_file`` only applies to lazy
        runs), using the run's storage options. A path is opened in "w" mode,
        replacing the file.

        Args:
            h5_fh (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py object
        """
        if isinstance(h5_fh, str):
            f = h5py.File(h5_fh, "w")
            close_file = True
        else:
            f = h5_fh
            close_file = False

        # Setup the h5 file.
        self.setup_h5(f)
        gpf_runs_grp = f["gpf_runs"]

        for gpf_index, gpf_run in self._gpf_runs.items():
            gpf_run.dump_h5(
                gpf_runs_grp.create_group(f"gpf_{gpf_index}"), storage=self._storage
            )

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls, h5_fh: Union[str, Path, h5py.File, h5py.Group], config: Config = Config()
    ) -> "MSRun":
        """Load an MSRun object from an HDF5 file.

        With ``config.lazy`` True, only the group names are registered and GPFRuns
        load on demand through ``get_gpf``; a path must be given so the MSRun can
        keep its own handle. Otherwise every GPFRun is read into memory.

        Args:
            h5_fh (Union[str, Path, h5py.File, h5py.Group]): Path to HDF5 file or
                h5py object
            config (Config): Run configuration (``lazy``, ``tqdm_enabled``).

        Returns:
            MSRun: Loaded object

        Raises:
            ValueError: If ``config.lazy`` is True and an h5py object is passed.
        """
        if isinstance(h5_fh, (str, Path)):
            # Handle case when you want to lazy-load from_h5, use handle from object
            # itself.
            if config.lazy:
                f = None
                close_file = False
            # If lazy-loading not desired, can create handle.
            else:
                f = h5py.File(h5_fh, "r")
                close_file = True
        else:
            if config.lazy:
                raise ValueError(
                    "Can't lazy-load h5 and provide h5 file or group as h5_fh."
                )
            f = h5_fh
            close_file = False

        # Create instance
        # Only a lazy run keeps a (writable) handle. An in-memory reload reads everything
        # through f above; opening the same file again for writing would fail.
        obj = cls(
            lazy=config.lazy,
            h5_fh=h5_fh if config.lazy else None,
            config=config,
        )

        # For loading, either pass the handle from the new instance or the handle above.
        if obj._lazy:
            f = obj._h5_file

        obj._isolation_low = (
            None if f.attrs["isolation_low"] == -1 else f.attrs["isolation_low"]
        )
        obj._isolation_high = (
            None if f.attrs["isolation_high"] == -1 else f.attrs["isolation_high"]
        )

        # Register every saved GPF, in index order. GPFs can be in the main file or,
        # for split runs, in part files listed in the main file's index. In lazy mode
        # `None` is passed and add_gpf just records the GPF; get_gpf finds it.
        gpf_files = cls.read_gpf_files(f)
        gpf_runs_grp = f["gpf_runs"]
        gpf_indices = {int(name.split("_")[1]) for name in gpf_runs_grp} | set(
            gpf_files
        )
        base_dir = Path(f.file.filename).parent
        for gpf_index in tqdm(sorted(gpf_indices), disable=(not config.tqdm_enabled)):
            if obj._lazy:
                gpf_run = None
            elif gpf_index in gpf_files:
                with h5py.File(base_dir / gpf_files[gpf_index], "r") as part:
                    gpf_run = GPFRun.from_h5(part["gpf_runs"][f"gpf_{gpf_index}"])
            else:
                gpf_run = GPFRun.from_h5(gpf_runs_grp[f"gpf_{gpf_index}"])
            obj.add_gpf(gpf_index, gpf_run, config=config)

        if close_file:
            f.close()

        return obj
