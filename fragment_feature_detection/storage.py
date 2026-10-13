"""HDF5 storage options and helpers shared by the containers.

Lazy runs write every isolation target (GPFRun) and all of its scan windows to HDF5.
Two optional settings in ``[h5]`` change how that data is laid out on disk, without
changing what is stored or any result:

- ``compact_storage``: don't repeat data that can be rebuilt exactly, store the bin grid
  once per file, and compress datasets (lossless).
- ``gpfs_per_file``: write the GPFRuns into several smaller part files instead of one.

Reading never needs these settings: compact windows are marked in the file, compression
is transparent to h5py, and split runs keep an index in the main file.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import h5py
import numpy as np

# Group holding arrays shared by several GPFRuns in one file (e.g. the bin grid).
SHARED_GROUP = "shared_arrays"


@dataclass
class StorageOptions:
    """How containers write their data to HDF5.

    Attributes:
        compact (bool): Leave out per-window copies of the GPF-wide m/z axis (rebuilt
            from the parent GPFRun on load) and store the bin grid once per file.
        compression (Optional[str]): h5py compression filter for datasets (e.g.
            ``"gzip"`` or ``"lzf"``), or None for no compression.
        save_m_unfilter (bool): Also store each window's unfiltered matrix.
    """

    compact: bool = False
    compression: Optional[str] = None
    save_m_unfilter: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "StorageOptions":
        """Build the options from ``config.h5``.

        Compression is only used together with ``compact_storage``, so the default
        configuration writes files exactly as before.

        Args:
            config (Config): Run configuration.

        Returns:
            StorageOptions: Options for writing this run.
        """
        h5 = config.h5
        compact = bool(h5.get("compact_storage", False))
        return cls(
            compact=compact,
            compression=h5.get("compact_compression", "gzip") if compact else None,
            save_m_unfilter=bool(h5.get("scan_window_save_m_unfilter", False)),
        )


def dataset_kwargs(data: Any, compression: Optional[str]) -> Dict[str, Any]:
    """Keyword arguments for ``create_dataset`` with optional compression.

    Compression needs a chunked layout, which h5py can't use for scalars or empty
    arrays, so those are always stored uncompressed.

    Args:
        data (Any): Data that will be written.
        compression (Optional[str]): Compression filter, or None.

    Returns:
        Dict[str, Any]: Extra keyword arguments for ``create_dataset``.
    """
    if compression is None or np.ndim(data) == 0 or np.size(data) == 0:
        return {}
    # Shuffle reorders bytes before compression, which helps a lot for numeric arrays.
    return {"compression": compression, "shuffle": True}


def create_dataset(
    group: h5py.Group, name: str, data: Any, compression: Optional[str] = None
) -> h5py.Dataset:
    """Create a dataset, compressed when ``compression`` is set.

    Args:
        group (h5py.Group): Parent group.
        name (str): Dataset name.
        data (Any): Data to write.
        compression (Optional[str]): Compression filter, or None.

    Returns:
        h5py.Dataset: The new dataset.
    """
    return group.create_dataset(name, data=data, **dataset_kwargs(data, compression))


def link_shared_dataset(
    group: h5py.Group, name: str, data: np.ndarray, compression: Optional[str] = None
) -> None:
    """Store an array once per file and hard-link it into ``group`` as ``name``.

    If an identical array is already in the file's shared group, it is linked
    instead of written again. Readers see an ordinary dataset at ``group[name]``.

    Args:
        group (h5py.Group): Group that should contain the dataset.
        name (str): Dataset name within ``group``.
        data (np.ndarray): Array to store.
        compression (Optional[str]): Compression filter, or None.
    """
    shared = group.file.require_group(SHARED_GROUP)
    for key in shared:
        existing = shared[key]
        if (
            existing.shape == data.shape
            and existing.dtype == data.dtype
            and np.array_equal(existing[()], data)
        ):
            group[name] = existing
            return
    key = f"array_{len(shared)}"
    create_dataset(shared, key, data, compression=compression)
    group[name] = shared[key]
