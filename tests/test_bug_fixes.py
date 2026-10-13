"""Regression tests for the correctness bugs listed in SUGGESTIONS.md (section A/B)."""

import logging

import h5py
import numpy as np
import pytest

from fragment_feature_detection.config import Config
from fragment_feature_detection.containers import (
    dump_features_to_df_msrun,
    fit_nmf_matrix_msrun,
)
from fragment_feature_detection.containers.gpfrun import GPFRun
from fragment_feature_detection.containers.ms1feature import MS1Feature
from fragment_feature_detection.containers.msrun import MSRun
from fragment_feature_detection.decomposition import (
    NMFMaskWrapper,
    _fit_and_score,
    tune_hyperparameters_optunasearchcv,
)
from fragment_feature_detection.discretization import MzDiscretize
from fragment_feature_detection.fitpeaks import fit_gaussian_elution, gaussian_curve
from fragment_feature_detection.simulation import assemble_fragments


def make_long_array(
    n_scans: int = 420,
    n_species: int = 6,
    gpfs=(500.0, 502.0),
    seed: int = 0,
) -> np.ndarray:
    """Synthetic long-format MS2 data with gaussian-eluting species.

    Columns are [gpf, scan, rt, mz, intensity, gpf_low, gpf_high].
    """
    rng = np.random.default_rng(seed)
    rows = []
    scan = 0
    rt = np.linspace(0.0, 30.0, n_scans)
    species = []
    for gpf in gpfs:
        for _ in range(n_species):
            mu = rng.uniform(5.0, 25.0)
            sigma = rng.uniform(0.15, 0.4)
            fragments = rng.uniform(250.0, 1500.0, 15)
            weights = rng.uniform(0.2, 1.0, 15)
            species.append((gpf, mu, sigma, fragments, weights))
    for i in range(n_scans):
        for gpf in gpfs:
            scan += 1
            for g, mu, sigma, fragments, weights in species:
                if g != gpf:
                    continue
                elution = 1e5 * gaussian_curve(rt[i], mu, sigma)
                keep = elution * weights > 10.0
                for mz, w in zip(fragments[keep], weights[keep]):
                    rows.append([gpf, scan, rt[i], mz, elution * w, gpf - 1, gpf + 1])
            # Background noise.
            for mz in rng.uniform(250.0, 1500.0, 20):
                rows.append([gpf, scan, rt[i], mz, rng.uniform(1.0, 50.0), gpf - 1, gpf + 1])
    return np.array(rows)


# A1, A6, B3
def test_config_ini_round_trip_keeps_case(tmp_path):
    config = Config()
    config.nmf.alpha_W = 0.5
    config.to_ini(tmp_path / "out.ini")

    loaded = Config.from_ini(tmp_path / "out.ini")
    assert loaded.nmf.alpha_W == 0.5
    assert set(loaded.nmf) == set(Config().nmf)


def test_config_lowercase_keys_map_to_existing(tmp_path):
    (tmp_path / "old.ini").write_text(
        "[nmf]\nalpha_w = 0.25\nalpha_h = 0.125\n[discretization]\nbin_width = ppm\n"
    )
    loaded = Config.from_ini(tmp_path / "old.ini")
    assert loaded.nmf.alpha_W == 0.25
    assert loaded.nmf.alpha_H == 0.125
    assert "alpha_w" not in loaded.nmf
    # Unquoted strings fall back to the raw value instead of raising NameError.
    assert loaded.discretization.bin_width == "ppm"


def test_shipped_config_ini_loads():
    from pathlib import Path

    import fragment_feature_detection

    ini = Path(fragment_feature_detection.__file__).parent / "config.ini"
    loaded = Config.from_ini(ini)
    assert set(loaded.nmf) == set(Config().nmf)


def test_config_instances_dont_share_sections():
    a = Config()
    a.nmf.alpha_W = 123.0
    a.feature_matching.match_ms1 = False
    b = Config()
    assert b.nmf.alpha_W != 123.0
    assert b.feature_matching.match_ms1 is True


# A4, A16
def test_gaussian_fit_error_uses_unit_height_curve():
    t = np.linspace(0, 10, 201)  # Contains the apex at 5.0.
    for sigma in (0.1, 1.0):
        m = 100.0 + 1000.0 * gaussian_curve(t, 5.0, sigma)
        (mu, s), _, mae = fit_gaussian_elution(t, m, sigma_bounds=(0.01, 10.0))
        assert mu == pytest.approx(5.0, abs=1e-3)
        assert s == pytest.approx(sigma, rel=1e-3)
        assert mae < 1e-4


def test_gaussian_fit_raises_on_constant_profile():
    with pytest.raises(ValueError):
        fit_gaussian_elution(np.arange(10.0), np.zeros(10))


# A15
@pytest.mark.parametrize("bin_width,tolerance", [("ppm", 20e-6), ("da", 0.02)])
def test_discretized_values_fall_inside_their_bin(bin_width, tolerance):
    d = MzDiscretize(
        bin_width=bin_width, mz_low=200.0, mz_high=2000.0, steps=2, tolerance=tolerance
    )
    rng = np.random.default_rng(1)
    mz = np.sort(rng.uniform(200.0, 2000.0, 20000))
    m = np.column_stack([np.zeros_like(mz), np.zeros_like(mz), np.zeros_like(mz), mz])
    disc = d.discretize_mz_array(m)
    bin_indices = disc[:, 4].astype(int)
    left = d.bin_left_edge.flatten()[bin_indices]
    right = d.bin_right_edge.flatten()[bin_indices]
    assert np.all(disc[:, 3] > left)
    assert np.all(disc[:, 3] <= right)


# A17
def test_scan_windows_cover_last_scans():
    m = make_long_array(n_scans=400, gpfs=(500.0,))
    gpf = GPFRun.from_undiscretized_long(500.0, m, MzDiscretize(), config=Config())
    last_window = gpf.scan_windows[-1]
    assert last_window.scan_index[-1] == gpf._scan_index[-1]


# A18
def test_ms1_interpolation_keeps_hill_endpoints():
    feature = MS1Feature(
        df_index=0,
        calibrated_mass=1000.0,
        retention_time_apex=1.5,
        intensity_apex=10.0,
        intensity_sum=30.0,
        charge=2,
        n_isotopes=2,
        n_scans=3,
        mz=501.0,
        rt_start=1.0,
        rt_end=2.0,
        scan_index=[1, 2, 3],
        intensity=[5.0, 10.0, 5.0],
        scan_apex=2,
    )
    feature.interpolate_onto_ms2(np.array([0.0, 1.0, 1.5, 2.0, 3.0]))
    np.testing.assert_allclose(
        feature._interpolated_intensity, [0.0, 5.0, 10.0, 5.0, 0.0]
    )


# A21
def test_assemble_fragments_keeps_n_fragments_per_species():
    species_info = np.array([[0.0, 1.0, 100.0, 3.0], [0.0, 1.0, 100.0, 5.0]])
    fragments = assemble_fragments(species_info)
    assert (~np.isnan(fragments[0, :, 0])).sum() == 3
    assert (~np.isnan(fragments[1, :, 0])).sum() == 5


# A11, A12
def test_fit_and_score_failure_returns_original_error():
    estimator = NMFMaskWrapper(n_splits=2)
    assert estimator.splitter_type == "Mask"
    results = _fit_and_score(estimator, None, parameters={})
    assert len(results) == 2
    assert all(r["fit_error"] for r in results)


def _kept_components(msrun):
    return {
        gpf_index: [
            sw.component_names.tolist()
            for sw in msrun.get_gpf(gpf_index).scan_windows
        ]
        for gpf_index in msrun._gpf_runs
    }


# A2, A3, A9, A20
@pytest.mark.parametrize("include_gpf_bounds", [True, False])
def test_lazy_run_keeps_collapsed_components(tmp_path, include_gpf_bounds):
    m = make_long_array()
    if not include_gpf_bounds:
        m = m[:, :5]

    results = {}
    for lazy in (False, True):
        config = Config()
        config.lazy = lazy
        config.ms2_preprocessing.include_gpf_bounds = include_gpf_bounds
        config.feature_matching.match_ms1 = False
        msrun = MSRun.from_long(
            m.copy(), msrun_h5_fh=tmp_path / f"run_{lazy}.h5", config=config
        )
        fit_nmf_matrix_msrun(msrun, n_jobs=1, config=config)
        results[lazy] = _kept_components(msrun)
        df = dump_features_to_df_msrun(msrun, config=config)
        assert df["ms2_component"].is_unique
        msrun.close()

    assert results[True] == results[False]

    with h5py.File(tmp_path / "run_True.h5", "r") as f:
        assert "isolation_high" in f.attrs


# A5, B4, B9
def test_optuna_tuning_runs_and_restores_logging(tmp_path):
    m = make_long_array(n_scans=500)
    config = Config()
    config.lazy = False
    msrun = MSRun.from_long(m, config=config)
    config.tuning.window_sampling_n = 4
    config.tuning.exclude_scan_window_edges = 0
    windows = msrun.get_tuning_windows(config=config)
    assert len(windows) > 0

    config.tuning.n_iter = 4
    config.tuning.n_splits = 2

    root = logging.getLogger()
    handlers_before = list(root.handlers)
    file_handler = logging.FileHandler(tmp_path / "run.log")
    # Tuning logs next to the first FileHandler, put ours ahead of pytest's.
    root.handlers.insert(0, file_handler)
    try:
        best_params = tune_hyperparameters_optunasearchcv(
            windows, config=config, save_run=False, n_jobs=2
        )
        assert set(best_params) == {"l1_ratio", "alpha_W", "alpha_H"}
        assert file_handler in root.handlers
    finally:
        root.removeHandler(file_handler)
        file_handler.close()
    assert list(root.handlers) == handlers_before


# Tuning scores must average every window of a split, not read the first n_splits.
def test_fit_and_score_averages_all_windows():
    from fragment_feature_detection.decomposition import _FIT_AND_SCORE

    rng = np.random.default_rng(3)
    windows = [rng.random((40, 30)) for _ in range(5)]
    estimator = NMFMaskWrapper(n_splits=2, n_components=4, max_iter=50)
    results = _fit_and_score(estimator, windows, extra_scores=_FIT_AND_SCORE)
    errors = np.array(estimator._test_reconstruction_errors).reshape(2, 5)
    for i, r in enumerate(results):
        assert r["test_scores"]["score"] == pytest.approx(-errors[i].mean())
        assert r["n_test_samples"] == sum(estimator._test_samples[i * 5 : (i + 1) * 5])


# Reloading a saved in-memory run must not filter (max-scale) the windows again.
def test_reload_saved_run_does_not_refilter(tmp_path):
    config = Config()
    config.lazy = False
    msrun = MSRun.from_long(make_long_array(n_scans=300, gpfs=(500.0,)), config=config)
    before = [sw.m.copy() for sw in msrun.get_gpf(0).scan_windows]
    msrun.dump_h5(str(tmp_path / "saved.h5"))

    reloaded = MSRun.from_h5(tmp_path / "saved.h5", config=config)
    after = [sw.m for sw in reloaded.get_gpf(0).scan_windows]
    assert len(before) == len(after)
    for a, b in zip(before, after):
        np.testing.assert_allclose(a, b, rtol=1e-3)
    reloaded.close()


# --- Storage layouts (compact_storage / gpfs_per_file) must not change anything ---


def make_ms1_features(gpfs=(500.0, 502.0), n=12, seed=1, twins=False):
    """Synthetic Biosaur2-style MS1 feature table near the given isolation targets.

    With ``twins``, every feature gets a second one (charge 3) with the same elution,
    so components can match several MS1 features.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        gpf = gpfs[i % len(gpfs)]
        rt_start = rng.uniform(4.0, 24.0)
        n_scans = int(rng.integers(5, 30))
        intensities = list(1e4 * gaussian_curve(np.linspace(-2, 2, n_scans), 0, 1))
        rows.append(
            {
                "massCalib": 2 * gpf - 2 * 1.0073,
                "rtApex": rt_start + 0.5,
                "intensityApex": max(intensities),
                "intensitySum": sum(intensities),
                "charge": 2,
                "nIsotopes": 3,
                "nScans": n_scans,
                "mz": gpf + rng.uniform(-0.4, 0.4),
                "rtStart": rt_start,
                "rtEnd": rt_start + 1.0,
                "mono_hills_scan_lists": list(range(n_scans)),
                "mono_hills_intensity_list": intensities,
                "scanApex": n_scans // 2,
            }
        )
        if twins:
            twin = dict(rows[-1])
            twin["charge"] = 3
            twin["mz"] = rows[-1]["mz"] + 0.05
            rows.append(twin)
    return pd.DataFrame(rows)


def _run_layout(
    tmp_path,
    name,
    compact=False,
    gpfs_per_file=0,
    compression="gzip",
    explode_ms1_matches=True,
    twins=False,
):
    from fragment_feature_detection.containers import (
        add_ms1_features_msrun,
        fit_ms1_ms2_feature_matching_msrun,
    )

    gpfs = (500.0, 502.0, 504.0)
    config = Config()
    config.lazy = True
    config.h5.compact_storage = compact
    config.h5.compact_compression = compression
    config.h5.gpfs_per_file = gpfs_per_file
    config.feature_output.explode_ms1_matches = explode_ms1_matches
    h5_path = tmp_path / f"{name}.h5"
    msrun = MSRun.from_long(
        make_long_array(n_scans=330, gpfs=gpfs), msrun_h5_fh=h5_path, config=config
    )
    fit_nmf_matrix_msrun(msrun, n_jobs=1, config=config)
    add_ms1_features_msrun(
        msrun, make_ms1_features(gpfs, twins=twins), config=config, n_jobs=1
    )
    fit_ms1_ms2_feature_matching_msrun(msrun, n_jobs=1, config=config)
    df = dump_features_to_df_msrun(msrun, config=config)
    msrun.close()
    return df, h5_path


def _assert_same_value(a, b, where):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype, f"{where}: dtype {a.dtype} != {b.dtype}"
        assert a.shape == b.shape, f"{where}: shape {a.shape} != {b.shape}"
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f"), where
    elif hasattr(a, "toarray"):
        _assert_same_value(a.toarray(), b.toarray(), where)
    elif isinstance(a, list):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same_value(vars(x), vars(y), f"{where}[{i}]")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{where}: {a.keys() ^ b.keys()}"
        for k in a:
            _assert_same_value(a[k], b[k], f"{where}.{k}")
    else:
        assert a == b or (a != a and b != b), f"{where}: {a!r} != {b!r}"


def _assert_runs_identical(path_a, path_b, lazy):
    config = Config()
    config.lazy = lazy
    run_a = MSRun.from_h5(path_a, config=config)
    run_b = MSRun.from_h5(path_b, config=config)
    assert list(run_a._gpf_runs) == list(run_b._gpf_runs)
    for gpf_index in run_a._gpf_runs:
        windows_a = run_a.get_gpf(gpf_index).scan_windows
        windows_b = run_b.get_gpf(gpf_index).scan_windows
        assert len(windows_a) == len(windows_b)
        for k, (sw_a, sw_b) in enumerate(zip(windows_a, windows_b)):
            _assert_same_value(vars(sw_a), vars(sw_b), f"gpf {gpf_index} window {k}")
    run_a.close()
    run_b.close()


def _assert_frames_identical(a, b):
    assert list(a.columns) == list(b.columns)
    assert len(a) == len(b)
    for column in a.columns:
        for i, (x, y) in enumerate(zip(a[column], b[column])):
            _assert_same_value(x, y, f"{column}[{i}]")


@pytest.mark.parametrize(
    "layout",
    [
        {"compact": True},
        {"compact": True, "compression": "lzf"},
        {"gpfs_per_file": 2},
        {"compact": True, "gpfs_per_file": 1},
    ],
)
def test_storage_layouts_give_identical_results(tmp_path, layout):
    df_standard, h5_standard = _run_layout(tmp_path, "standard")
    df_layout, h5_layout = _run_layout(tmp_path, "layout", **layout)

    # Same exported results, value for value.
    assert len(df_standard) > 0 and df_standard["ms1_matching_coef"].notna().any()
    _assert_frames_identical(df_standard, df_layout)

    # Same objects when the saved runs are reloaded, lazily and into memory.
    _assert_runs_identical(h5_standard, h5_layout, lazy=True)
    _assert_runs_identical(h5_standard, h5_layout, lazy=False)

    if layout.get("gpfs_per_file"):
        parts = sorted((tmp_path / "layout.gpf_runs").glob("part_*.h5"))
        assert len(parts) == -(-3 // layout["gpfs_per_file"])
        with h5py.File(h5_layout, "r") as f:
            assert len(f["gpf_runs"]) == 0
            assert MSRun.read_gpf_files(f)


def test_rerun_with_different_split_keeps_one_copy(tmp_path):
    m = make_long_array(n_scans=300, gpfs=(500.0, 502.0))
    h5_path = tmp_path / "run.h5"
    for gpfs_per_file in (1, 0):
        config = Config()
        config.lazy = True
        config.h5.gpfs_per_file = gpfs_per_file
        MSRun.from_long(m.copy(), msrun_h5_fh=h5_path, config=config).close()
    with h5py.File(h5_path, "r") as f:
        assert sorted(f["gpf_runs"]) == ["gpf_0", "gpf_1"]
        assert MSRun.read_gpf_files(f) == {}
    for part in (tmp_path / "run.gpf_runs").glob("part_*.h5"):
        with h5py.File(part, "r") as f:
            assert len(f["gpf_runs"]) == 0


# NMF on float32 windows can hit NaN in the SVD initialization; it must retry in float64
# instead of returning an empty fit.
def test_nmf_retries_in_float64_when_float32_fails():
    from fragment_feature_detection.decomposition import fit_nmf_matrix_custom_init

    config = Config()
    config.lazy = False
    run = MSRun.from_long(
        make_long_array(n_scans=330, gpfs=(500.0, 502.0, 504.0)), config=config
    )
    m = run.get_gpf(2).scan_windows[1].m  # float32 window that failed before
    assert m.dtype == np.float32
    W, H, model = fit_nmf_matrix_custom_init(m, return_model=True, **config.nmf)
    assert model.reconstruction_err_ != 1e6
    assert W.dtype == np.float32 and H.dtype == np.float32
    assert W.any() and H.any()


# One row per component (MS1 matches as arrays) must hold exactly the same data.
def test_unexploded_ms1_matches_hold_same_data(tmp_path):
    import pandas as pd

    from fragment_feature_detection.utils import explode_ms1_matches, feature_df_to_ms2

    df_rows, _ = _run_layout(tmp_path, "rows", twins=True)
    df_lists, _ = _run_layout(
        tmp_path, "lists", explode_ms1_matches=False, twins=True
    )
    assert df_lists["ms2_component"].is_unique
    assert len(df_lists) < len(df_rows)

    # Round-trip through Parquet, as the CLI does, then expand.
    df_rows.to_parquet(tmp_path / "rows.parq")
    df_lists.to_parquet(tmp_path / "lists.parq")
    rows = pd.read_parquet(tmp_path / "rows.parq")
    lists = pd.read_parquet(tmp_path / "lists.parq")
    _assert_frames_identical(
        rows.astype({"ms1_charge": float, "ms1_id": float}),
        explode_ms1_matches(lists).astype({"ms1_charge": float, "ms1_id": float}),
    )

    # The .ms2 files are identical (apart from the creation date in the header).
    for wide_window in (False, True):
        for output_type in ("raw", "intensity_transform_weight", "raw_weight"):
            texts = []
            for name, table in (("rows", rows), ("lists", lists)):
                out = tmp_path / f"{name}_{output_type}_{wide_window}.ms2"
                feature_df_to_ms2(
                    table,
                    out,
                    {"output_type": output_type, "wide_window": wide_window},
                )
                texts.append(
                    [
                        line
                        for line in out.read_text().splitlines()
                        if not line.startswith("H	CreationDate")
                    ]
                )
            assert texts[0] == texts[1]
            assert len(texts[0]) > 10


# MS1 attachment used to crash when every feature had the same number of isotopes
# inside the window (DataFrame.apply expanded the lists into columns).
def test_ms1_attachment_with_equal_isotope_counts_and_empty_table():
    from fragment_feature_detection.containers import add_ms1_features_scanwindow

    config = Config()
    config.lazy = False
    run = MSRun.from_long(make_long_array(n_scans=300, gpfs=(500.0,)), config=config)
    sw = run.get_gpf(0).scan_windows[2]
    features = make_ms1_features((500.0,), n=6)
    features["mz"] = 500.0  # every feature: same isotope count inside the window
    features["rtStart"] = sw.retention_time[0]
    features["rtEnd"] = sw.retention_time[-1]
    add_ms1_features_scanwindow(sw, features, config=config)
    assert len(sw.ms1_features) == 6

    add_ms1_features_scanwindow(sw, features.iloc[:0], config=config)
    assert sw.ms1_features == []
