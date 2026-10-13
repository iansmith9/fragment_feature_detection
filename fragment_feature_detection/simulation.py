"""Toy simulator of co-eluting species and their fragments.

Builds synthetic data with known ground truth for testing the decomposition:

1. `assemble_species`: species with Gaussian elution (location, width, abundance) and
   a number of fragments. Elution is simulated by sampling one retention time per
   "ion", so abundance = number of samples.
2. `assemble_fragments`: random fragment m/z values and relative intensities.
3. `bin_scans`: histogram the sampled retention times into pseudo scans.
4. `add_noise_aggregated_matrix`: add random noise to a scan x fragment matrix.
"""

from typing import Tuple

import numpy as np
from tqdm import tqdm

from fragment_feature_detection.config import Config


def assemble_species(
    min_time: int = 0,
    max_time: int = 10000,
    n_species: int = 250,
    fragment_n_min: int = 4,
    fragment_n_max: int = 75,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create simulated species with gaussian elution times and varying intensities.

    For each species, draws an elution center (uniform in [min_time, max_time]),
    width (uniform in [1, 75]), abundance (integer in [5, 100000)) and fragment count,
    then samples `abundance` retention times from the elution Gaussian.

    Args:
        min_time (int): Minimum elution time. Defaults to 0.
        max_time (int): Maximum elution time. Defaults to 10000.
        n_species (int): Number of species to generate. Defaults to 250.
        fragment_n_min (int): Minimum number of fragments per species. Defaults to 4.
        fragment_n_max (int): Maximum number of fragments per species. Defaults to 75.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing:
            - species_ids: Array of species identifiers
            - species: Array of elution times for each species
            - species_info: Array of species parameters (loc, scale, intensity, fragment_n)
    """
    rng = np.random.default_rng(seed=seed)

    species_ids = np.array([], dtype=int)
    species = np.array([])
    species_info = np.empty(
        (
            0,
            4,
        )
    )

    for i in tqdm(range(n_species)):
        # Species parameters.
        loc = rng.uniform(min_time, max_time)
        scale = rng.uniform(1, 75)
        intensity = rng.integers(5, 100000)
        fragment_n = rng.integers(fragment_n_min, fragment_n_max)
        species_info = np.vstack([species_info, (loc, scale, intensity, fragment_n)])
        # One retention time sample per ion, labelled with the species id.
        species = np.concatenate([species, rng.normal(loc, scale, intensity)])
        species_ids = np.concatenate([species_ids, [i] * intensity])

    return species_ids, species, species_info


def assemble_fragments(
    species_info: np.ndarray,
    seed: int = 42,
    mz_min: int = 100,
    mz_max: int = 1500,
) -> np.ndarray:
    """Generate fragments for each species with random m/z values and intensities.

    Every species gets a row of `max(fragment_n)` slots. Slots beyond the species' own
    fragment count are NaN. Fragment intensity = species abundance x a random factor
    in [0.5, 2].

    Args:
        species_info (np.ndarray): Array of species parameters from assemble_species.
        seed (int): Random seed for reproducibility. Defaults to 42.
        mz_min (int): Minimum m/z value. Defaults to 100.
        mz_max (int): Maximum m/z value. Defaults to 1500.

    Returns:
        np.ndarray: 3D array of fragment information (species x max_fragments x [mz, intensity])
    """

    rng = np.random.default_rng(seed=seed)

    # Draw m/z and a relative intensity (2^U(-1, 1)) for every slot of every species.
    n_fragments = species_info[:, 3].max().astype(int)
    fragment_mz = rng.uniform(mz_min, mz_max, species_info.shape[0] * n_fragments)
    fragment_scale = 2 ** rng.uniform(-1, 1, species_info.shape[0] * n_fragments)

    # Each species has a row of n_fragments slots; keep the first n_i and blank the rest.
    ends = np.cumsum(np.repeat(n_fragments, species_info.shape[0]))
    starts = ends - n_fragments + species_info[:, 3].astype(int)

    for s, e in zip(starts, ends):
        fragment_mz[s:e] = np.nan
        fragment_scale[s:e] = np.nan

    # Stack to (species, slots, [m/z, intensity]).
    fragments = np.dstack(
        [
            fragment_mz.reshape(species_info.shape[0], -1),
            fragment_scale.reshape(species_info.shape[0], -1)
            * species_info[:, 2, np.newaxis],
        ],
    )

    return fragments


def bin_scans(
    species: np.ndarray,
    min_time: int = 0,
    max_time: int = 10000,
    n_bins: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create pseudo scans by binning species over elution time.

    Each sampled retention time is assigned to a time bin (a pseudo scan) with
    `np.digitize`, so counting samples per (scan, species) gives elution profiles.

    Args:
        species (np.ndarray): Array of species elution times.
        min_time (int): Minimum elution time. Defaults to 0.
        max_time (int): Maximum elution time. Defaults to 10000.
        n_bins (int): Number of time bins. Defaults to 5000.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            - bins: Array of bin edges
            - bin_ids: Array of bin assignments for each species
    """
    bins = np.linspace(min_time, max_time, n_bins)
    bin_ids = np.digitize(species, bins)

    return bins, bin_ids


def add_noise_aggregated_matrix(
    m: np.ndarray,
    noise_percent: float,
    seed: int = 42,
) -> np.ndarray:
    """Add random noise to an intensity matrix by redistributing a percentage of signal.

    Values of the matrix are shuffled, and about `noise_percent` of positions
    (chosen at random, with repeats) receive a shuffled value added on top of their
    own. The original signal is kept; noise is added, not moved.

    Args:
        m (np.ndarray): Input intensity matrix.
        noise_percent (float): Percentage of signal to redistribute as noise (0-1).
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        np.ndarray: Matrix with added noise.
    """

    rng = np.random.default_rng(seed=seed)

    m_flat = m.copy().reshape(-1)
    rng.shuffle(m_flat)
    m_noise = np.zeros(m_flat.shape)
    m_noise_locs = rng.integers(
        0, m_noise.shape[0], int(m_noise.shape[0] * noise_percent)
    )
    m_noise[m_noise_locs] = 1.0
    m_noise = m_noise * m_flat

    return m + m_noise.reshape(m.shape)
