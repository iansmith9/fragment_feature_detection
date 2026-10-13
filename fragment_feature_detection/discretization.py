"""Fragment m/z binning.

Centroided peaks of the same fragment drift by a few ppm from scan to scan, so peaks
are mapped onto a fixed grid of m/z bins before building scan x bin matrices. Two
details matter:

- **Right-edge convention.** `_bins` stores the right edge of each bin. A value `v`
  belongs to the first bin whose right edge is >= `v` (`np.searchsorted`, side
  "left"), so a value exactly on an edge goes to the lower bin.
- **Interleaved grids.** With `steps = 2`, a second grid is shifted by half a bin.
  Every peak is binned on every grid, so a fragment that sits on a bin edge in one
  grid falls near a bin center in the other. Bins of all grids share one global index
  space: `global index = step * n_bins + bin`.
"""

from typing import Literal, Tuple, Union, List, Dict, Any, Optional

import numpy as np
import h5py

from fragment_feature_detection.config import Config
from fragment_feature_detection.storage import link_shared_dataset


class MzDiscretize:
    """A class for discretizing mass-to-charge (m/z) values into bins.

    This class provides functionality to create and manage binned m/z values using either
    parts per million (ppm) or dalton (da) bin widths. It supports multiple overlapping
    bin sets.

    Args:
        bin_width (Literal["ppm", "da"]): The type of bin width to use. Defaults to "ppm".
        mz_low (float): The lower m/z boundary. Defaults to 200.0.
        mz_high (float): The upper m/z boundary. Defaults to 2000.0.
        steps (int): Number of overlapping bin sets. Defaults to 2.
        tolerance (float): The bin width in either ppm or da units. Defaults to 60.0/1e6.

    Attributes:
        _bin_width (str): Type of bin width ("ppm" or "da").
        _mz_low (float): Lower boundary of m/z range.
        _mz_high (float): Upper boundary of m/z range.
        _steps (int): Number of overlapping bin sets.
        _tolerance (float): Width of bins in specified units.
        _bins (np.ndarray): Array containing bin right edge values, shape (steps, n_bins).
    """

    def __init__(
        self,
        bin_width: Literal["ppm", "da"] = "ppm",
        mz_low: float = 200.0,
        mz_high: float = 2000.0,
        steps: int = 2,
        tolerance: float = 60.0 / 1e6,
        **kwargs: Dict[str, Any],
    ):
        self._bin_width = bin_width
        self._mz_low = mz_low
        self._mz_high = mz_high
        self._steps = steps
        self._tolerance = tolerance
        self._build_bins()

    @property
    def bin_center(self) -> np.ndarray:
        """Get the center values of all bins.

        Returns:
            np.ndarray: Array of bin center values.
        """
        return (self.bin_left_edge + self._bins) / 2

    @property
    def bin_left_edge(self) -> np.ndarray:
        """Get the left edge values of all bins.

        The left edge of a bin is the right edge of the previous bin. The first bin's
        left edge is one bin width below its right edge (a fixed width in Da, or
        right edge x tolerance in ppm).

        Returns:
            np.ndarray: Array of bin left edge values.
        """
        if self._bin_width == "da":
            first_left_edge = self._bins[:, 0] - self._tolerance
        else:
            first_left_edge = self._bins[:, 0] - self._bins[:, 0] * self._tolerance
        return np.hstack(
            [
                first_left_edge[..., np.newaxis],
                self._bins[:, :-1],
            ],
        )

    @property
    def bin_right_edge(self) -> np.ndarray:
        """Get the right edge values of all bins.

        Returns:
            np.ndarray: Array of bin right edge values.
        """
        return self._bins

    @property
    def bin_indices(self) -> np.ndarray:
        """Get the indices for all bins across all steps.

        Row `i` holds the global indices of grid `i`: `i * n_bins + [0..n_bins)`.

        Returns:
            np.ndarray: 2D array of bin indices, shape (steps, bins_per_step).
        """
        m = np.repeat(
            np.arange(self._bins.shape[1])[np.newaxis, ...], self._steps, axis=0
        )
        m += (np.arange(0, self._bins.shape[0]) * self._bins.shape[1])[..., np.newaxis]

        return m

    def indices_to_mz(
        self, indices: Union[np.ndarray, List[int]], center: bool = False
    ) -> np.ndarray:
        """Convert bin indices to m/z values.

        Works on global indices: flattening the (steps, n_bins) edge array row by row
        gives exactly the global index order.

        Args:
            indices (Union[np.ndarray, List[int]]): Bin indices to convert.
            center (bool): If True, return bin center values instead of right edges. Defaults to False.

        Returns:
            np.ndarray: Array of m/z values corresponding to the input indices.
        """
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices)

        indices = indices.astype(int)

        if center:
            m = self.bin_center
        else:
            m = self.bin_right_edge

        return m.flatten()[indices.flatten()]

    def _build_bins(self) -> None:
        """Build the bin edges based on the specified parameters.

        Creates bins using either constant dalton width or variable ppm width.
        Bins describe right edge of bin, ties go left by default.

        - "da": grid `i` starts at `mz_low + i * tolerance / steps` and advances by
          `tolerance`, i.e. each grid is shifted up by 1/steps of a bin.
        - "ppm": grid `i` starts at `mz_high` shifted down by `i/steps` of a bin, and
          each edge below is the previous one times (1 - tolerance), so bin width is
          proportional to m/z. Grids are cut to a common length, keeping the high
          end, and reversed to ascending order.

        In both cases, bin columns whose right edge is <= mz_low in any grid are
        dropped, so all grids have the same number of bins.
        """
        if self._bin_width == "da":
            # Build with an explicit count, np.arange lengths can differ by one between
            # steps due to floating point.
            n_bins = int(np.ceil((self._mz_high - self._mz_low) / self._tolerance))
            bins = np.array(
                [
                    self._mz_low
                    + (self._tolerance / self._steps * i)
                    + self._tolerance * np.arange(n_bins)
                    for i in range(self._steps)
                ]
            )
        elif self._bin_width == "ppm":
            bins = []
            for i in range(self._steps):
                # Top right edge of grid i, shifted down by i/steps of a bin.
                b = [
                    self._mz_high - (i * self._tolerance * self._mz_high / self._steps)
                ]
                # Walk down in constant-ppm steps until below mz_low.
                while b[-1] > self._mz_low:
                    b.append(b[-1] - (self._tolerance * b[-1]))
                bins.append(b)
            # Same number of bins per grid, ascending order.
            min_b = min([len(b) for b in bins])
            bins = np.array([b[:min_b][::-1] for b in bins])
        # Bins reflect right edge, filter bins < self._mz_low
        bins = bins[:, bins.min(axis=0) > self._mz_low]
        self._bins = bins

    def discretize(
        self, m: np.ndarray, center: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Discretize an array of m/z values into bins.

        Every value is binned on every grid. Values above the last right edge are
        clipped into the last bin and values below the first edge land in the first
        bin; `discretize_mz_array` removes those.

        Args:
            m (np.ndarray): Array of m/z values to discretize.
            center (bool): If True, return bin center values instead of right edges. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Array of global bin indices for each input m/z value, shape
                  (steps, n_values)
                - Array of corresponding bin edge or center values, same shape
        """
        # Per grid: index of the first right edge >= value.
        row_indices = np.empty((self._steps, m.shape[0]))
        for i in range(self._steps):
            indices = np.searchsorted(self._bins[i], m)
            indices[indices >= self._bins.shape[1]] = self._bins.shape[1] - 1
            # bin_indices[i] = indices + (i * self._bins.shape[1])
            row_indices[i] = indices
        # Look up the right edge (or center) of each assigned bin.
        if not center:
            bin = self._bins[
                np.repeat(np.arange(0, self._steps), m.shape[0]),
                row_indices.flatten().astype(int),
            ].reshape(self._steps, -1)

        else:
            bin = self.bin_center[
                np.repeat(np.arange(0, self._steps), m.shape[0]),
                row_indices.flatten().astype(int),
            ].reshape(self._steps, -1)

        # Offset grid i's local indices by i * n_bins to get global indices.
        bin_indices = (
            row_indices
            + (np.arange(0, self._bins.shape[0]) * self._bins.shape[1])[..., np.newaxis]
        )
        return bin_indices, bin

    def discretize_mz_array(
        self,
        m,
        sort_m: bool = True,
        mz_index: int = 3,
    ) -> np.ndarray:
        """Discretize an array containing m/z values and additional information.

        Each input row is emitted once per grid (only for grids whose range contains
        its m/z), with the global bin index and the bin's right edge appended. For the
        long MS2 array (7 columns) the output columns are therefore
        [gpf, scan, rt, mz, intensity, gpf_low, gpf_high, bin_index, bin_right_edge].

        Args:
            m (np.ndarray): 2D array where one column contains m/z values.
            sort_m (bool): If True, sort the array by m/z values. Defaults to True.
            mz_index (int): Column index containing m/z values. Defaults to 3.

        Returns:
            np.ndarray: Discretized array with additional columns for bin indices and edges.
                       Up to (n_rows * steps, n_columns + 2) rows, grid by grid.
        """
        if sort_m:
            m = m[np.argsort(m[:, mz_index])]

        # Keep only peaks inside the configured m/z range.
        m = m[(m[:, mz_index] > self._mz_low) & (m[:, mz_index] < self._mz_high)]

        bin_indices, bin_right_edge = self.discretize(m[:, mz_index])

        # discretize clips out-of-range values into the first/last bin, drop them
        # instead so every value lands in the bin whose edges contain it.
        bin_left_edge = self.bin_left_edge
        disc_m = []
        for i in range(self._steps):
            in_range = (m[:, mz_index] > bin_left_edge[i, 0]) & (
                m[:, mz_index] <= self._bins[i, -1]
            )
            disc_m.append(
                np.hstack(
                    [
                        m[in_range],
                        bin_indices[i][in_range][:, np.newaxis],
                        bin_right_edge[i][in_range][:, np.newaxis],
                    ]
                )
            )

        return np.concatenate(disc_m)

    @classmethod
    def from_config(cls, config: Config = Config()) -> "MzDiscretize":
        """Create a MzDiscretize instance from a configuration object.

        Args:
            config (Config): Configuration object containing discretization parameters.
                Defaults to Config.

        Returns:
            MzDiscretize: New instance initialized with configuration parameters.
        """
        return cls(**config.discretization)

    def dump_h5(
        self,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "mz_discretize",
        shared: bool = False,
        compression: Optional[str] = None,
    ) -> None:
        """Save the MzDiscretize object to an HDF5 file or group.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group to store the object. Defaults to 'mz_discretize'
        """

        # Handle string path vs h5py object
        if isinstance(h5file, str):
            f = h5py.File(h5file, "w")
            grp = f.create_group(dataset_name)
            close_file = True
        else:
            grp = h5file.create_group(dataset_name)
            close_file = False

        # Save attributes
        grp.attrs["bin_width"] = self._bin_width
        grp.attrs["mz_low"] = self._mz_low
        grp.attrs["mz_high"] = self._mz_high
        grp.attrs["steps"] = self._steps
        grp.attrs["tolerance"] = self._tolerance

        # Save bins array. With shared=True the grid is stored once per file and
        # hard-linked here, since every GPFRun of a run uses the same grid. Readers
        # see an ordinary "bins" dataset either way.
        if shared:
            link_shared_dataset(grp, "bins", self._bins, compression=compression)
        else:
            grp.create_dataset("bins", data=self._bins)

        if close_file:
            f.close()

    @classmethod
    def from_h5(
        cls,
        h5file: Union[str, h5py.File, h5py.Group],
        dataset_name: str = "mz_discretize",
    ) -> "MzDiscretize":
        """Load a MzDiscretize object from an HDF5 file or group.

        The stored bin edges replace the rebuilt ones, so indices stay valid even if
        the bin-building code changes.

        Args:
            h5file (Union[str, h5py.File, h5py.Group]): Path to HDF5 file or h5py File/Group object
            dataset_name (str): Name of the dataset/group containing the object. Defaults to 'mz_discretize'

        Returns:
            MzDiscretize: Loaded object
        """
        # Handle string path vs h5py object
        if isinstance(h5file, str):
            f = h5py.File(h5file, "r")
            grp = f[dataset_name]
            close_file = True
        else:
            grp = h5file[dataset_name]
            close_file = False

        # Create instance with stored parameters
        obj = cls(
            bin_width=grp.attrs["bin_width"],
            mz_low=grp.attrs["mz_low"],
            mz_high=grp.attrs["mz_high"],
            steps=grp.attrs["steps"],
            tolerance=grp.attrs["tolerance"],
        )

        obj._bins = grp["bins"][:]

        if close_file:
            f.close()

        return obj
