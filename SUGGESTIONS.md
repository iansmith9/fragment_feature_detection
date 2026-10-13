# Code review: bugs, improvements, speed, and GPU options

This started as a **read-only review** of the repository at commit `558ba09`. Every
finding came from reading the source.

> **Update (2026-10-13):** the correctness bugs in [section A](#a-correctness-bugs) and the
> bug-like items in [section B](#b-robustness-portability-and-engineering) have been worked
> through. The fixes are in the working tree and **not yet committed**. Each item now has a
> **Status** line, and [Fix status](#fix-status) summarizes what was done, what is partial,
> and how it was checked. Sections C–F (improvements, speed, GPU, new features) haven't
> been started.

- **Confidence** says how sure I am from the code alone:
  - **High**: the code path is clear and the failure follows directly.
  - **Medium**: very likely, but depends on data or library behavior I couldn't run.
  - **Low / question**: possibly intentional, worth a look.
- Line numbers refer to the files at `558ba09`, before the fixes.
- Code snippets are sketches of the fix, not tested patches. The applied fixes are in the
  working tree.

---

## Contents

- [Fix status](#fix-status)
- [Top 12 at a glance](#top-12-at-a-glance)
- [A. Correctness bugs](#a-correctness-bugs)
- [New findings (A23–A33)](#new-findings)
- [B. Robustness, portability and engineering](#b-robustness-portability-and-engineering)
- [C. Scientific / output-quality improvements](#c-scientific--output-quality-improvements)
- [D. Speed on CPU](#d-speed-on-cpu)
- [E. GPU acceleration](#e-gpu-acceleration)
- [F. Features for the peptidomics goals (clustering, re-search, delta masses, localization)](#f-features-for-the-peptidomics-goals)
- [G. Suggested order of work](#g-suggested-order-of-work)

---

## Fix status

| Status | Items |
|---|---|
| **Fixed** | A1, A2, A3, A4, A5, A7, A8, A9, A10, A11, A12, A13, A14, A15, A16, A17, A19, A20, A21, A23, A24, A25, A26, A27, A32, A33, B2, B3, B4, B5, B6, B9, B11 (opt-in), D11 |
| **Partly fixed** | A6, A18, B1, B7, B8, B10, D4 |
| **Open** | A22 (a design question), A28–A31, the rest of C, D, E, F |

A23–A33 and D11 were found later, while adding comments throughout the code,
measuring run times and testing the new storage settings (see [New findings](#new-findings)).

Two more bugs turned up while fixing these, and both are fixed:

- `MzDiscretize.bin_left_edge` used the ppm formula in `'da'` mode, so the first bin's left
  edge was wrong (see [A15](#a15)).
- A failed Gaussian fit (empty or constant profile) returned `None` values but left the
  component **kept**. It now raises, so the component is dropped (see [A16](#a16)).

### How the fixes were checked

- **Regression tests:** `tests/test_bug_fixes.py` (17 tests: 15 at first, plus 2 for A23
  and A24) covers A1–A5, A9, A11, A12,
  A15–A18, A20, A21, B3, B4 and B9. It includes:
  - lazy vs. non-lazy runs on synthetic data, which must give the same kept components
    with no duplicates;
  - a real 2-worker Optuna tuning run on Windows.

  All 15 pass on Python 3.9 with the pinned dependencies. At `558ba09`, 11 of the 15 fail,
  and the module can't even be imported without installing the package (B2).
- **End-to-end CLI run** on a synthetic mzML, written with `psims`: `process_mzml` with
  `lazy = True`, no MS1 file, tuning off and a separate `--output_dir`, then `dump_df_to_ms2`.
  It found 12 components for 12 simulated species with no duplicates, and wrote all
  outputs into the output directory.
- **Not checked:** real DIA data, and MS1 matching with a real Biosaur2 file.

To run the tests, install the dependencies and then run `python -m pytest tests` from the
repository root.

### Behavior changes to expect

These fixes change results, not just stability:

- **Fewer duplicates in lazy mode (A3).** The default `Config()` is lazy, so runs without
  an INI will now export far fewer components.
- **Different duplicates kept (A4).** The collapse now keeps the duplicate with the truly
  better Gaussian fit.
- **More tiles (A17).** Each isolation window gets one more tile at its end.
- **Slightly different MS1 matching (A13, A18, A25).** The isotope spacing, the hill
  interpolation and the matching region's last scan changed.
- **Different tuned parameters (A23).** Each trial is now scored on every sampled window,
  not on 3 of them.

---

## Top 12 at a glance

| # | Finding | Impact | Confidence | Status |
|---|---|---|---|---|
| [A1](#a1) | INI round-trip lowercases `alpha_W`/`alpha_H` → NMF `TypeError` whenever an INI is used | **Run crashes** | High | Fixed |
| [A2](#a2) | `match_ms1 = True` by default → export crashes when no MS1 file is given | **Run crashes** | High | Fixed |
| [A3](#a3) | In `lazy` mode, the redundant-component collapse is never saved | **Silent duplicates** (default `Config()` is lazy) | High | Fixed |
| [A4](#a4) | Gaussian-fit MAE compares against a density-normalized pdf | Wrong duplicate kept in collapse | High | Fixed |
| [A5](#a5) | Optuna result-count check rejects a whole study after > ~`n_jobs` failed trials | Tuning crashes after hours | High | Fixed |
| [B4](#b4) | Optuna temp DB hard-coded to `/tmp/`, plus SQLite write contention | Tuning fails on Windows; flaky elsewhere | High / Medium | Fixed |
| [A6](#a6) | `Config` sections are shared class-level dicts, and `from_ini`/tuning change them globally | Settings leak between runs | High | Partly fixed |
| [C1](#c1) | Masked CV sets held-out cells to 0 instead of treating them as missing | Biased tuning | High (by design) | Open |
| [C2](#c2) | Exported fragment m/z = bin right edge, peaks duplicated across 2 grids, zero-intensity padding | Worse search scores | High | Open |
| [D1](#d1) | Per-peak Python loops in the mzML converter | Slowest step on big files | High | Open |
| [D3](#d3) | MS1 matching uses SLSQP with numerical gradients | Needlessly slow | High | Open |
| [E1](#e1) | Batched NMF on GPU (PyTorch/CuPy) across all tiles of an isolation window | Large speed-up for NMF and tuning | Medium (needs benchmarking) | Open |

---

## A. Correctness bugs

<a name="a1"></a>
### A1. INI round-trip breaks the NMF call (`alpha_w` / `alpha_h`)

- **Status:** **Fixed.** `to_ini` keeps key case. `from_ini` matches keys case-insensitively, so older INIs with `alpha_w` still load correctly. The shipped `config.ini` is corrected, and its stale `tuning.n_jobs` key is removed.

- **Where:** `config.py:209` (`to_ini` uses `configparser.ConfigParser()` with the default
  lower-casing `optionxform`), `config.py:176` (`from_ini` sets `optionxform = str`,
  preserving case), `fragment_feature_detection/config.ini` (`alpha_w`, `alpha_h`),
  `containers/__init__.py:108` (`**config.nmf`), `decomposition.py:1149` (`NMF(**nmf_kwargs)`).
- **What happens:**
  1. `generate_template_ini` writes `alpha_w`/`alpha_h`, and the shipped `config.ini` has
     them too.
  2. `from_ini` then *adds* `alpha_w` next to the default `alpha_W` in `config.nmf`.
  3. `fit_nmf_matrix_custom_init(**config.nmf)` passes `alpha_w=` through `**nmf_kwargs`
     into `sklearn.decomposition.NMF`, which raises
     `TypeError: __init__() got an unexpected keyword argument 'alpha_w'`.

  The `NMF(...)` constructor is outside the `try`, so the whole run stops at the first
  tile. Worse, the *intended* values in the INI are never applied. Tuning isn't affected
  (its `NMFMaskWrapper` gets explicit arguments), so the crash comes **after** tuning has
  finished.
- **Fix:** set `config.optionxform = str` in `to_ini` too. In `from_ini`, match keys to
  existing keys case-insensitively. Regenerate `config.ini`.

  ```python
  # to_ini
  config = configparser.ConfigParser(); config.optionxform = str
  # from_ini, inside the section loop
  canon = {k.lower(): k for k in section_dict}
  section_dict[canon.get(key.lower(), key)] = typed_value
  ```

<a name="a2"></a>
### A2. Running without MS1 features crashes at export

- **Status:** **Fixed.** `process_mzml` sets `match_ms1 = False` when no MS1 file is given, and the export error now says how to export without MS1. Export still follows the flag rather than `is_ms1_features_fit`.

- **Where:** `config.py:134` (`match_ms1: True`), `cli.py:238–256` (only sets it when a
  file is given), `containers/__init__.py:456–457`.
- **What happens:** `ffd process_mzml sample.mzML` with no `--ms1_features_file` runs NMF
  for hours, then `dump_features_to_df_scanwindow` raises `AttributeError("MS1 features have
  not been matched…")`. Nothing is written.
- **Fix:** in `process_mzml`, add `else: config.feature_matching.match_ms1 = False`. Better,
  base the export on whether matching actually happened (`sw.is_ms1_features_fit`) instead
  of on a flag.

<a name="a3"></a>
### A3. `lazy = True` loses the redundant-component collapse

- **Status:** **Fixed.** New `GPFRun.set_scan_windows(..., persist=False)` and `GPFRun.persist_scan_windows()`. `fit_nmf_matrix_gpfrun` now collapses first, then writes once. The `scan_windows` setter still persists, as before.

- **Where:** `containers/__init__.py:122–123`, `gpfrun.py:94–109`, `msrun.py:76–79`.
- **What happens:**
  1. `gpfrun.scan_windows = fit_scan_windows` goes through the setter, which in lazy mode
     **immediately writes** every window to HDF5.
  2. *Then* `collapse_redundant_components()` masks duplicates **in memory only**.
  3. The next stage calls `msrun.get_gpf(i)`, which in lazy mode **reloads from HDF5**, so
     the masks are gone.

  `Config()` defaults to `lazy = True`, so a CLI run without an INI outputs every
  duplicate across overlapping tiles (up to 5× per peptide).
- **Fix:** collapse before assigning, or write again after collapsing:

  ```python
  gpfrun._scan_windows = fit_scan_windows       # bypass setter
  gpfrun.collapse_redundant_components()
  gpfrun.scan_windows = gpfrun._scan_windows    # persist once, after collapse
  ```

  The broader problem is that a setter with a side effect (writing to disk) is easy to
  trigger at the wrong time. An explicit `gpfrun.persist()` would be clearer.

<a name="a4"></a>
### A4. Gaussian-fit error is computed against the wrong curve

- **Status:** **Fixed.** The MAE is now computed against `gaussian_curve(t, *popt)`.

- **Where:** `fitpeaks.py:121`: `e = np.absolute(stats.norm(*popt).pdf(t) - m_scale).mean()`.
- **What happens:** the fit is to a **unit-height** Gaussian (`gaussian_curve`), but the
  error compares the data (peaking at about 1) with the **probability density**, which
  peaks at 1/(σ√2π). For σ = 0.1 min the pdf peaks at about 4, and for σ = 1 min at about
  0.4. So the MAE mostly reflects σ, not fit quality.

  `collapse_redundant_components` (`gpfrun.py:285`) keeps whichever duplicate has the lower
  MAE, so it systematically favors certain widths instead of better fits. The exported
  quality is also misleading.
- **Fix:** `e = np.abs(gaussian_curve(t, *popt) - m_scale).mean()`. You could also export
  R² of the fit as a quality score.

<a name="a5"></a>
### A5. Optuna study rejected if more than about `n_jobs` trials fail

- **Status:** **Fixed.** Tuning now raises only if fewer than half of `n_iter` trials succeeded. Otherwise it logs a warning and continues with the successful trials. Workers still run `n_iter // n_jobs + 1` trials each; `MaxTrialsCallback` wasn't adopted.

- **Where:** `decomposition.py:955` (each worker runs `n_iter // n_jobs + 1` trials),
  `decomposition.py:971–976`.
- **What happens:** the total number of trials is about `n_iter + n_jobs`. Failed trials are
  raised as `TrialPruned` and excluded. If the number of successful trials × `n_splits` is
  below `n_iter × n_splits`, the code raises `ValueError("Returned fits not consistent…")`
  and all tuning work is lost. With 4 workers, **5 failed trials out of 504 is enough**.
  Failures happen easily (e.g. `log2(mse_train/mse)` with `mse = 0`, or an NMF numerical
  error).
- **Fix:** turn the check into a warning with a minimum (e.g. ≥ 50% of `n_iter`). Use
  Optuna's own trial count (`study.optimize(n_trials=…)` with a shared study, stopping on
  `len(study.trials) >= n_iter` with a `MaxTrialsCallback`) instead of `n_iter // n_jobs + 1`
  per worker.

<a name="a6"></a>
### A6. `Config` state is shared and changed globally

- **Status:** **Partly fixed.** `Config.__init__` deep-copies every section, so instances no longer share dicts. The converter and `MSRun.add_gpf` now get the run's `config`. Still open: the `config: Config = Config()` defaults are still created at import. Each now has its own dicts, but changing a defaulted instance in place still persists. The `Optional[Config] = None` refactor hasn't been done.

- **Where:** `config.py:48–147` (sections are **class attributes**), `config.py:181–191`
  (`from_ini` changes them through `getattr(instance, section)`), `cli.py:226–228`,
  `cli.py:240` (tuning/matching write into `config.nmf`/`config.feature_matching`). Also every
  `config: Config = Config()` default argument (created once, at import).
- **What happens:**
  - All `Config()` instances share the same dicts. Loading one INI or tuning once changes
    the "defaults" for the rest of the Python process.
  - In a notebook, a second run with a different INI inherits keys from the first run
    (including the stray `alpha_w`).
  - Top-level keys (`lazy`, `tqdm_enabled`) are set on the instance and *don't* leak. The
    behavior is inconsistent.
  - This shared state also hides another gap: `MzMLParser.to_ms2_h5` is called in the CLI
    **without** `config` (`cli.py:153`). It only follows the INI's `[ms2_preprocessing]`
    because of the leak.
- **Fix:** deep-copy class defaults in `Config.__init__` (or turn `Config` into dataclasses
  with `default_factory`), use `config: Optional[Config] = None` in signatures, and pass
  `config` everywhere explicitly.

<a name="a7"></a>
### A7. Refitting a scan window crashes / discards the previous fit

- **Status:** **Fixed (replace).** A refit now replaces the previous fit and clears peak fits and MS1 matches, which must be recomputed. Append behavior wasn't implemented.

- **Where:** `scanwindow.py:576–597`.
- **What happens:** the `if self.is_fit:` branch uses `model._reconstruction_err_` (the
  attribute is `reconstruction_err_`), which raises `AttributeError`. Even if that worked,
  everything the branch appends is immediately overwritten by the unconditional code below
  it. `component_keep[w.sum(axis=0) != 0.0]` also has the wrong length. The pipeline doesn't
  refit today, but `fit_nmf_matrix_scanwindow` accepts `W_init/H_init` for exactly that.
- **Fix:** decide on "replace" or "append" behavior, implement one, and add an `else`.

<a name="a8"></a>
### A8. `to_ms2_long` / `to_ms2_df` keep only the **non-finite** peaks

- **Status:** **Fixed.**

- **Where:** `converter.py:145` and `converter.py:202`:
  `mask = ~np.isfinite(mz) | ~np.isfinite(intensity)`.
- **What happens:** the mask is inverted compared with `to_ms2_h5` (`converter.py:85`), so
  these helpers return (almost) empty arrays. The CLI doesn't use them, but anyone using
  the in-memory path (`MSRun.from_long`) will get no data.
- **Fix:** `mask = np.isfinite(mz) & np.isfinite(intensity)`.

<a name="a9"></a>
### A9. `include_gpf_bounds = False` breaks MS1 matching and lazy HDF5 writing

- **Status:** **Fixed.** `ScanWindow.dump_h5` also stores −1 for `None` attributes.

- **Where:** `containers/__init__.py:185–187`,
  `getattr(w, "_gpf_low", w._gpf - 1.0)`. The attribute *exists* and is `None`, so the
  default is never used, and `float >= None` raises `TypeError`. Also `scanwindow.py:807–810`
  writes `grp.attrs["gpf_low"] = None`, which h5py can't store (`TypeError`).
- **Fix:** `low = w._gpf_low if w._gpf_low is not None else w._gpf - 1.0`. Store −1 for None
  in `ScanWindow.dump_h5` (as `GPFRun.dump_h5` already does).

<a name="a10"></a>
### A10. `ScanWindow.unfiltered_with_filter` cannot work

- **Status:** **Fixed.** It raises a clear `ValueError` when the unfiltered matrix isn't available, which is the default after reloading from HDF5.

- **Where:** `scanwindow.py:559–565`. It indexes `self._mz_bin_indices_unfilter[:, …]`
  (a 1-D array) instead of `self._m_unfilter`, and `if self._m_max:` on an array raises
  "truth value is ambiguous".
- **Fix:** `m = self._m_unfilter[:, ~self._mz_masked].toarray()` and
  `if scale_if_exists and self._m_max is not None:`. Note that `_mz_masked` indexing also
  assumes sorted bin indices.

<a name="a11"></a>
### A11. `_fit_and_score` error path can raise its own error

- **Status:** **Fixed.**

- **Where:** `decomposition.py:1057–1065`. On a fit exception it builds results from
  `estimator._models[i]`. If the failure happened before `n_splits` models were appended,
  that raises `IndexError` (or `AttributeError` if `fit` failed before `_models` was set),
  which hides the original error.
- **Fix:** use `getattr(estimator, "_models", [])` and pad with `None`.

<a name="a12"></a>
### A12. `NMFMaskWrapper` default `splitter_type = "Masking"` is invalid

- **Status:** **Fixed.**

- **Where:** `decomposition.py:328` vs `386/394/402`. The constructor default is
  `"Masking"`, `fit` accepts only `"Mask"`/`"Sample"`, and the error message suggests
  `'Masking'`. Direct users of the class get a `ValueError`.
- **Fix:** default to `"Mask"` and fix the message.

<a name="a13"></a>
### A13. Isotope spacing constant is 1.008 (proton/H mass) instead of about 1.00336 (¹³C–¹²C)

- **Status:** **Fixed** (`1.00336`).

- **Where:** `config.py:233` (`Constants.isotope_mu = 1.008`), used in
  `containers/__init__.py:182–187`.
- **Impact:** small. For z = 1 and the 3rd isotope, the error is about 0.009 m/z. It only
  matters for isotopes near a window edge, but it is easy to fix. Peptide isotope spacing
  averages about 1.00286–1.00336 Da.
- **Fix:** `isotope_mu = 1.00336`.

<a name="a14"></a>
### A14. Unit mix-up in collapse integration bounds (minutes ± scans)

- **Status:** **Fixed.** The integration bounds are now ±6·max(σ). Pairs with |μ1−μ2| > 3(σ1+σ2) are skipped without integrating. The loop also indexes fit parameters by component index, and skips components already masked earlier in the same pass. The closed-form overlap ([D4](#d4)) is still open.

- **Where:** `gpfrun.py:266–271`. The means are in RT (minutes), but the bounds add ±
  `self._window_scan_overlap` (30, a scan count). With 10,000 grid points over ±30 min, the
  step is about 0.006 min. That works for σ ≥ 0.05 min, but it is accidental and very
  wasteful (see [D4](#d4)).
- **Fix:** use `±6·max(σ1, σ2)` bounds, or a closed-form overlap.

<a name="a15"></a>
### A15. Discretization edge cases

- **Status:** **Fixed.** Values outside a grid's range are dropped for that grid. `'da'` grids are built with an explicit bin count. Also fixed: `bin_left_edge` used the ppm formula in `'da'` mode.

- **Where:** `discretization.py:173`. Values above the last right edge of a shifted grid
  are clipped into the last bin. For grid 1 (shifted down by half a bin), m/z values between
  its top edge and `mz_high` land in the wrong bin. `discretization.py:126–140` (`'da'`
  mode): `np.empty((steps, ceil(range/tol)))` vs `np.arange(...)` length can differ by one
  due to floating point → broadcast error.
- **Fix:** drop out-of-range values per grid (or extend each grid by one bin). Build `'da'`
  grids with `np.linspace` and an explicit count.

<a name="a16"></a>
### A16. Profile normalization before the Gaussian fit

- **Status:** **Fixed** (divides by the range). Empty or constant profiles now raise, so the component is marked not kept. Previously it returned `None` values and stayed kept. Amplitude and baseline are still fixed parameters ([C6](#c6)).

- **Where:** `fitpeaks.py:87`: `(m - m.min()) / m.max()` should be `/ (m.max() - m.min())`.
  If `min > 0`, the peak is below 1, but the model's height is fixed at 1, which biases σ.
- **Fix:** divide by the range. Or fit amplitude (and optionally a baseline) as free
  parameters.

<a name="a17"></a>
### A17. Last scans of each isolation window may not be in any tile

- **Status:** **Fixed.** A final tile is added at `n − width`. Isolation windows with fewer scans than one tile still get no tiles, but now log a warning.

- **Where:** `gpfrun.py:134–138`: `np.arange(0, n - width, stride)`. For n = 1000 the last
  tile is 840–990, so scans 990–999 aren't covered (up to `stride − 1 + 5` edge scans).
  There is also no tile when `n ≤ width`.
- **Fix:** add a final tile at `n − width` when it isn't already there.

<a name="a18"></a>
### A18. `MS1Feature` interpolation assumptions

- **Status:** **Partly fixed.** Fixed: the attribute name; the hill is now padded with zeros outside it instead of overwriting its endpoints; hills of 0–2 points are handled. Still open: RT points still assume evenly spaced MS1 scans. That needs an MS1 scan → RT table stored at conversion.

- **Where:** `ms1feature.py:57–62, 103–110`.
  - RT points are `linspace(rtStart, rtEnd, len(scan_index))`, which assumes MS1 scans are
    evenly spaced and the hill has no gaps. Biosaur2 hills can skip scans.
  - Replacing the first and last intensities with 0 throws away real data. Hills of length
    ≤ 2 give mismatched lengths in `np.interp`, or a single-point hill.
  - The class attribute is `_interpolated_intensity_list` while the instance uses
    `_interpolated_intensity`, so an uninterpolated feature raises `AttributeError` in
    `ms1_features_information`.
- **Fix:** map `mono_hills_scan_lists` to real MS1 RTs (store an MS1 scan → RT table at
  conversion). Pad with zeros *outside* the hill instead of overwriting its endpoints. Guard
  against short hills.

<a name="a19"></a>
### A19. Tuning sample fraction can divide by zero / go negative

- **Status:** **Fixed.** The code counts candidate windows across all GPFs. In lazy mode it reads the count from HDF5 without loading each GPF. If there are no candidates it warns and returns an empty list. The return annotation is fixed.

- **Where:** `msrun.py:117–123`. `len(gpf.scan_windows) - 2*exclude_scan_window_edges` is
  ≤ 0 for short runs (with defaults, fewer than about 20 tiles, i.e. about 750 scans per
  isolation window). It also takes the size from the middle isolation window only.
- **Fix:** clip to ≥ 1, compute from all GPFs, and warn when the edge exclusion leaves no
  candidates. Also fix the return annotation (`-> None`, but it returns a list).

<a name="a20"></a>
### A20. Small HDF5 bookkeeping bugs

- **Status:** **Fixed** (all three). `from_h5_long` and `from_long` now replace existing `gpf_runs/gpf_i` groups.

- `msrun.py:322–329`: the check is `"isolation_low" not in h5_file` (that checks groups, not
  attrs), and `isolation_high` is set from `_isolation_low`.
- `msrun.py:212`: `h5_fh == msrun_h5_fh` compares `Path` with `str` in some call patterns.
- In lazy mode via the API, re-running `from_h5_long` on an HDF5 that already has
  `gpf_runs/gpf_i` **silently reuses the old group** (`add_gpf` with
  `remove_if_exists=False`). The CLI doesn't hit this because it reconverts with mode `"w"`.

<a name="a21"></a>
### A21. `simulation.assemble_fragments` blanks the wrong fragments

- **Status:** **Fixed.**

- **Where:** `simulation.py:79–85`. `starts = ends − n_fragments_i` then NaN-fills
  `[start:end]`. That blanks the species' `n_i` fragments and keeps `n_max − n_i` random
  ones, which is the opposite of the intent. This matters if you use the simulator to build
  tests (recommended, [B1](#b1)).
- **Fix:** NaN-fill `[row_start + n_i : row_end]`.

<a name="a22"></a>
### A22. Question: asymmetric scaling in MS1 matching

- **Status:** **Open.** This is a design question; nothing was changed.

- **Where:** `containers/__init__.py:332–338`. The MS2 target is scaled by its norm *inside*
  the matching region, and MS1 candidates by their norm over the *whole tile*. This may be
  deliberate (it penalizes features mostly outside the region; see DETAILS §11.2). But it
  also means a perfect but long MS1 feature cannot reach coefficient 1, and the fixed
  `c_cutoff`/`matching_coefficient_threshold` values behave differently for short and long
  features.
- **Suggestion:** decide explicitly. Either scale both inside the region and add a separate
  "fraction of MS1 signal inside the region" term, or document the current behavior.

---

<a name="new-findings"></a>
## New findings (found while commenting the code and measuring run times)

<a name="a23"></a>
### A23. Tuning scored only the first `n_splits` fits of each trial

- **Status:** **Fixed.** Each metric is now averaged over all windows within a split.
  Covered by `test_fit_and_score_averages_all_windows`.
- **Where:** `decomposition._fit_and_score` together with `NMFMaskWrapper.fit`.
- **What happened:**
  - `NMFMaskWrapper.fit` stores one result per (split, window), ordered split by split.
  - `_fit_and_score` read entries `0 … n_splits−1`, which are the first 3 windows of
    split 0.
  - With the defaults (3 splits, about 50 windows), each trial was judged on 3 of its 150
    fits. The tuned parameters rested on a handful of windows, and the other fits were
    wasted time.
- **Impact:** high. It decides the NMF settings for the whole run.

<a name="a24"></a>
### A24. Reloading a saved in-memory run failed

- **Status:** **Fixed.** Covered by `test_reload_saved_run_does_not_refilter`.
- **Where:** `MSRun.from_h5` with `lazy = False`, and `MSRun.add_gpf`.
- **What happened:**
  1. The file was opened read-only, and then the constructor opened it again for writing.
     HDF5 refuses that (`file is already open for read-only`).
  2. Even past that, `add_gpf` would have filtered the already-filtered windows a second
     time, scaling them twice.
- **Fix:** an in-memory reload no longer opens a writable handle, and `add_gpf` skips
  windows that are already filtered.

<a name="a25"></a>
### A25. MS1 matching left out the tile's last scan

- **Status:** **Fixed.**
- **Where:** `fit_ms1_ms2_feature_matching_scanwindow`.
- **What happened:** the slice end was capped at `n − 1` and then used as an exclusive
  end, so a matching region reaching the end of the tile lost the last scan.

<a name="a26"></a>
### A26. The Gaussian fit could start at the wrong end of its bounds

- **Status:** **Fixed.** Start values are now clipped to the nearest bound.
- **Where:** `fit_gaussian_elution`.
- **What happened:** a start value outside `mu_bounds`/`sigma_bounds` was replaced by the
  **upper** bound. Sharp peaks (moment σ below the 0.1-min floor) started at σ = 10 min.
  The optimizer usually recovered on clean test peaks, but it is a poor start for noisy
  profiles.

<a name="a27"></a>
### A27. `filter_spectra` could drop entire spectra

- **Status:** **Fixed.**
- **Where:** `MzMLParser.to_ms2_h5` and `MzMLParser.to_ms2_long`.
- **What happened:** the per-spectrum percentile was computed on the unmasked intensity
  array. One NaN made the threshold NaN, and every peak of that spectrum was dropped.

<a name="a32"></a>
### A32. NMF failed on some windows in float32 and left them empty

- **Status:** **Fixed.** Covered by `test_nmf_retries_in_float64_when_float32_fails`.
- **Where:** `fit_nmf_matrix_custom_init`.
- **What happened:**
  - Windows are float32. On some small windows, the randomized SVD behind NNDSVD
    initialization overflowed to NaN (`array must not contain infs or NaNs`).
  - The exception was caught, and the window got **no components**. That was 1 of ~30
    windows in a synthetic test.
  - The same matrix fits fine in float64.
- **Fix:** retry once in float64 when the float32 fit fails, and return float32. Windows that
  already fit are unaffected; windows that failed now produce components.

<a name="a33"></a>
### A33. Adding MS1 features crashed when a tile's features had equal isotope counts

- **Status:** **Fixed.** Covered by
  `test_ms1_attachment_with_equal_isotope_counts_and_empty_table`.
- **Where:** `add_ms1_features_scanwindow`.
- **What happened:** `DataFrame.apply` returning lists expands them into columns when every
  row's list has the same length. Assigning the result to one column then raised
  `ValueError: Cannot set a DataFrame with multiple columns...`. An empty feature table
  failed similarly. (`origin/dev` has a related fix for the empty case.)
- **Fix:** build the isotope lists with a plain list comprehension.

<a name="a28"></a>
### A28. Tuning ignores `[nmf] max_iter` and `solver`

- **Status:** **Open.**
- **Where:** `tune_hyperparameters_*`, which builds `NMFMaskWrapper` without these
  settings.
- **Impact:** the tuned values come from 500-iteration `'cd'` fits, even when the final run
  uses something else.
- **Fix:** pass `max_iter`, `solver` (and `tol`) from `config.nmf`.

<a name="a29"></a>
### A29. Fragile code paths in the searches

- **Status:** **Open.** None of these is on the default path.
  - **Randomized search** (`optuna = False`): a failed fit returns a result without
    `test_scores`, and the result-count check is still exact. One failed candidate breaks
    `_format_results`. This is [A5](#a5) on the other search path.
  - **Optuna objective:** the single-objective fallback calls `np.nanmean` on a list of
    dicts. It would raise, and the trial would be pruned.
  - **`NMFMaskWrapper.score`:** returns NaN, because `_reconstruction_errors` is never
    filled. Nothing calls it.
  - **Pickled models:** each Optuna trial pickles one NMF model per split into its stored
    result, so the journal and the saved `.pkl` grow with `n_iter`.

<a name="a30"></a>
### A30. HDF5 bookkeeping gaps

- **Status:** **Open.**
- **`ScanWindow.dump_h5` in update mode after a refit:** `is_ms1_features_fit` is only
  ever written as True. Stale MS1 matches stay in the file and are restored on load. Only
  matters if windows are refit ([A7](#a7)).
- **`ScanWindow.dump_h5` when `_m` changes type:** a dense `_m` is stored at full
  precision and a sparse one as float16. If the type changes between writes, the old
  datasets are left behind.
- **`MS1Feature.dump_h5`:** given a string path, it opens the file with `"w"` and
  overwrites it.
- **`ScanWindow.query_component_info`:** still a TODO that returns empty dicts. (The
  `KeyError` in `MSRun.query_components_by_name` was fixed with [B11](#b11).)
- *(Fixed with [B11](#b11): `MSRun.dump_gpfrun_to_h5` now passes the run's storage
  options, so `h5.scan_window_save_m_unfilter` is honored in lazy runs.)*
- **`GPFRun.from_h5`:** builds all tiles in `__init__` and then replaces them with the
  ones loaded from HDF5. That's extra work on every lazy reload.

<a name="a31"></a>
### A31. Smaller items

- **Status:** **Open.**
- `ScanWindow.ms1_features_information` with no features returns a `(0, n_scans)` matrix
  instead of `(n_scans, 0)`. Matching happens to work anyway.
- `ScanWindow.filter_denoise_scans` is annotated as returning an array but returns None.
- `GPFRun.sample_scan_windows` has a default `rng` created once at import and shared
  between calls. The pipeline always passes its own.
- `feature_df_to_ms2` without `--wide_window`, on a Parquet exported without MS1
  matching, raises `KeyError: 'ms1_matching_coef'` instead of a clear message.
- `calculate_simple_sparsity` returns the fraction of **non-zero** values, so higher means
  denser despite the name. `calculate_nmf_summary` computes `weight_deviation_identity`
  from H but `weight_sparsity` from W.
- `calculate_mz_from_masses` and `calculate_theoretical_precursor_monoisotopic_masses`
  aren't used anywhere, and treat their input as MH⁺ rather than neutral mass.
- Run logs and tuning logs go next to the input file rather than into `--output_dir`. With
  the Python API and no log file, tuning writes `optuna_tuning.*` to the current directory.
- Every tile is filtered serially in the main process (`MSRun.add_gpf`) before any
  parallel work starts.

---

## B. Robustness, portability and engineering

<a name="b1"></a>
### B1. No tests

- **Status:** **Started.** `tests/test_bug_fixes.py` has 15 regression tests (see [Fix status](#fix-status)). There's no CI or golden-file test yet.

There is no `tests/` directory and no CI. Suggested minimum:

1. **Unit tests**: `MzDiscretize` (every m/z lands in the bin whose edges contain it, on
   both grids), `pivot_unique_binned_mz_sparse`, `Config` INI round-trip (would catch
   [A1](#a1)), `fit_gaussian_elution` on a synthetic Gaussian (would catch [A4](#a4)),
   `least_squares_with_l1_bounds` against a known solution.
2. **End-to-end synthetic test**: use `simulation.py` (after fixing [A21](#a21)) to build 2–3
   isolation windows with known co-eluting "peptides". Write a tiny mzML with `psims` or
   pyteomics, run `process_mzml`, and check that every true spectrum is recovered (cosine >
   0.9) with no duplicates (would catch [A3](#a3)).
3. **Golden-file regression**: freeze a small real mzML subset (a few windows × 5 min) and
   compare the Parquet between versions.
4. GitHub Actions running pre-commit + pytest on Linux **and Windows**.

<a name="b2"></a>
### B2. Package version is read at import

- **Status:** **Fixed.** The version is resolved at call time, with `"dev"` as the fallback.

`utils.py:313`: `extractor_version: str = version("fragment_feature_detection")` is
evaluated at import. It raises `PackageNotFoundError` unless the package is installed.
**Fix:** resolve inside the function with a fallback, `try: version(...) except
PackageNotFoundError: "dev"`.

<a name="b3"></a>
### B3. `eval` on INI values

- **Status:** **Fixed** (`ast.literal_eval`, falling back to the raw string).

`config.py:188, 196`: `eval(value)` executes arbitrary code from a config file. It only
catches `ValueError`, so an unquoted string (`bin_width = ppm`) raises `NameError`.
**Fix:** `ast.literal_eval` with a `(ValueError, SyntaxError)` fallback to the raw string.

<a name="b4"></a>
### B4. Optuna temp DB: `/tmp/` path and SQLite contention

- **Status:** **Fixed** (option 2). Tuning uses `JournalStorage` with `JournalFileOpenLock` in a `tempfile.mkdtemp()` directory, which is removed afterwards. The open-based lock is used because the default symlink lock needs extra privileges on Windows. Each worker builds its own storage from the path. Tested with a 2-worker run on Windows.

- `decomposition.py:932–934`: `tempfile.NamedTemporaryFile(dir="/tmp/", suffix=".db")`.
  - On **Windows** `/tmp/` usually doesn't exist → `FileNotFoundError`.
  - Even if it does, a `NamedTemporaryFile` held open with delete-on-close can't be reopened
    by SQLite on Windows.
  - On Linux, several joblib processes writing to one SQLite file cause
    `database is locked` errors under load. Optuna's docs warn against SQLite for parallel
    optimization.
- **Fix options:**
  1. `tempfile.mkdtemp()` + a path inside it (cross-platform).
  2. `optuna.storages.JournalStorage(JournalFileBackend(path))`, which is designed for
     multi-process use on one machine.
  3. Run Optuna in one process and parallelize **inside** the objective: fit the
     `n_splits × windows` NMFs of one trial with joblib. That uses cores just as well,
     avoids shared storage entirely, and makes seeding simple ([B8](#b8)).

<a name="b5"></a>
### B5. Output paths ignore `--output_dir`

- **Status:** **Fixed.**

- `cli.py:259, 262`: Parquet and `.out.ini` use `Path(file).with_suffix(...)`, so they go
  next to the mzML.
- `cli.py:320`: `output_dir / parquet_path.with_suffix(...)`. When `parquet_path` is
  absolute, `Path.__truediv__` returns the absolute path and `output_dir` is ignored.
- **Fix:** `output_dir / Path(file).with_suffix(...).name` everywhere.

<a name="b6"></a>
### B6. CLI arguments not passed through

- **Status:** **Fixed** (all three).

- `cli.py:252`: `add_ms1_features_msrun(...)` doesn't get `n_jobs`, so it always uses 8.
- `cli.py:153`: `MzMLParser.to_ms2_h5(...)` doesn't get `config` (see [A6](#a6)).
- `convert_mzml_h5long` has no `--config_ini_file`.

<a name="b7"></a>
### B7. Dependency pins and deprecated APIs

- **Status:** **Partly fixed.** `np.NINF`/`np.Inf` and sparse `.A` were replaced. The pins, unused dependencies and private scikit-learn API use are unchanged.

- `numpy==1.24.4`, `scipy==1.10.1` and `pyopenms==2.7.0` limit you to Python ≤ 3.10/3.11.
- `pyopenms`, `mmh3` and `matplotlib` aren't imported. Dropping them (or making them
  optional extras) makes installation much easier.
- Before moving to NumPy 2 / newer SciPy, replace `np.NINF`/`np.Inf` (`fitpeaks.py:94`) and
  `.A` on sparse matrices (`utils.py:97`).
- `RandomizedSearchReconstructionCV`/`OptunaSearchReconstructionCV` subclass `BaseSearchCV`
  and call the private `_format_results`. That breaks easily across scikit-learn versions.
  A small self-contained results table (a pandas DataFrame) would be more robust.

<a name="b8"></a>
### B8. Tuning isn't reproducible

- **Status:** **Partly fixed.** Each worker's `NSGAIISampler` is seeded with `random_seed + worker`. Runs still aren't exactly reproducible, because of parallel trial ordering.

`optuna.create_study(...)` (`decomposition.py:942`) sets no sampler seed, and parallel
workers add ordering randomness. Fix: `sampler=optuna.samplers.NSGAIISampler(seed=config.random_seed)`
(or `TPESampler(seed=…, multivariate=True)`), together with option 3 in [B4](#b4).

<a name="b9"></a>
### B9. Logging

- **Status:** **Fixed.** QueueHandlers are replaced instead of stacked. The containers use the module logger. Tuning restores the root handlers in a `finally` block; it also no longer removes handlers from a list while iterating over it.

- `decomposition.py:601–603, 892–894` add a new `QueueHandler` to a named logger on every
  trial/worker. In reused worker processes, handlers pile up and lines get duplicated.
- `containers/__init__.py` uses `logging.info` (the root logger) where the other modules use
  `logger`.
- Tuning replaces the root file handler during the run. It is restored afterwards, but only
  if nothing raises. Use `try/finally`.

<a name="b10"></a>
### B10. Smaller code-health items

- **Status:** **Partly fixed.** The wrong annotations are fixed: `get_tuning_windows`, `OptunaSearchReconstructionCV.fit` and `tune_hyperparameters_randomizedsearchcv`. The other items are open.

- Many docstrings are empty (`""" """`), especially in `gpfrun.py`, `msrun.py` and
  `cli.py`.
- Wrong annotations: `get_tuning_windows -> None`, and `OptunaSearchReconstructionCV.fit ->
  "RandomizedSearchReconstructionCV"`.
- `query_component_info` is a TODO that returns empty dicts, yet `MSRun.query_components_by_name`
  depends on it.
- `ScanNum` is the spectrum index + 1 (`converter.py:224`), not the vendor scan number, so
  joining back to vendor tools needs care. Consider also storing the native ID.
- The `lowerMass/higherMass` bounds are centered on `selected ion m/z` rather than the
  isolation target (`converter.py:237–248`). For DIA they are usually the same, but the
  target is the documented reference.
- `pyproject.toml` description typo ("Decompisiton").
- Processing flags on `ScanWindow` (`_percentile_filter`, `_filter_edge_nscans`, denoise
  sizes, `p = 0.3`, collapse thresholds) aren't in `Config`, so runs can't be fully
  described by the `.out.ini`. Move them into `Config` sections.

<a name="b11"></a>
### B11. Output files are very large in lazy mode, and can't be split

- **Status:** **Implemented as opt-in settings** (the defaults write files exactly as
  before):
  - `[h5] compact_storage = True` (with `compact_compression = 'gzip'` or `'lzf'`);
  - `[h5] gpfs_per_file = N`;
  - `[feature_output] explode_ms1_matches = False`.

  Results are identical in every layout. Tests compare the exported table value for value,
  and every field of every reloaded window, against the default layout.
- **What the default layout stores** (synthetic run, 9,000 spectra, 36 MB mzML, MS1
  matching):
  - The `.h5` is **195 MB**, versus 2.9 MB with `lazy = False`.
  - **~69%** is the isolation target's full m/z axis, copied into every window as
    `mz_unfilter`, `mz_bin_indices_unfilter` and `mz_masked`.
  - The bin grid is stored again for every target.
  - The MS1 features add one small group of 3–5 datasets per feature per window. That's
    ~19,500 HDF5 objects here, and potentially millions on a real run with Biosaur2
    features, at ~1 KB of headers each.
  - Nothing is compressed except the converted peaks.
- **What `compact_storage` does:**
  - **Leaves out the per-window axis.** A window's unfiltered axis is its GPFRun's
    `bin_indices`. On load, `ScanWindow.from_h5` rebuilds `mz_unfilter` and
    `mz_bin_indices_unfilter` (cast back to the stored dtypes) and
    `mz_masked = ~isin(axis, kept bins)`, which is exactly what `filter_mz_axis`
    maintains. The window is marked with the `compact_axes` attribute.
  - **Stores each window's MS1 features as one table** (`ms1_feature_table`): one dataset
    per field, with the per-feature arrays concatenated and indexed by offsets. The
    interpolated profiles are recomputed on load from the same inputs. It falls back to
    the per-feature layout whenever a table couldn't be exact.
  - **Stores the bin grid once per file** under `/shared_arrays`, hard-linked into each
    `discretize/` group, so readers see an ordinary dataset.
  - **Compresses every dataset** (lossless; scalars and empty arrays excepted) and writes
    the HDF5 1.10 object format, which has smaller headers and chunk indexes.
- **What `gpfs_per_file` does:**
  - Lazy runs write the GPFRuns to `<stem>.gpf_runs/part_<k>.h5`, `N` targets per part.
  - The main file keeps a `gpf_files` index, and `MSRun.gpf_group` resolves every read and
    write through it, so the pipeline, `MSRun.from_h5` (lazy or in memory) and
    `query_components_by_name` work the same.
  - Rerunning with a different `gpfs_per_file` removes the old copy, so only one remains.
- **What `explode_ms1_matches = False` does:** one Parquet row per component, with the MS1
  match columns as arrays. A component matched to several MS1 features is stored once, not
  once per match. `dump_df_to_ms2` expands it and writes identical `.ms2` files
  (`utils.explode_ms1_matches`).
- **Measured** on the same synthetic run with `lazy = True`:

  | Settings | `.h5` total | Objects | Time |
  |---|---|---|---|
  | Default | 195.3 MB | 19,495 | 42 s |
  | `compact_storage` (gzip) | **21.7 MB (9.0×)** | 10,851 | 34 s |
  | `compact_storage` (lzf) | 23.9 MB | 10,851 | 30 s |
  | `compact_storage` + `gpfs_per_file = 2` | 2.9 MB + 3 × ~7 MB | | 32 s |

  The Parquet went from 0.138 to 0.114 MB with `explode_ms1_matches = False` (65 components,
  131 match rows). The saving grows with the number of matches per component and the size
  of the raw apex spectra.
- **Expected on real data:** a larger reduction than 9×. Both main contributors, the
  repeated axis (it grows with the m/z bins per target) and the per-feature MS1 objects
  (they grow with the Biosaur2 feature count), are much larger on real runs.
- **Not done (would lose an ability or change outputs):**
  - Dropping the converted peaks after the run is built: you couldn't rebuild from the
    `.h5`.
  - A separate MS1-match table file.
  - Merging the two grids' peaks or trimming zero padding ([C2](#c2)): this changes the
    spectra.

---

## C. Scientific / output-quality improvements

<a name="c1"></a>
### C1. Use missing-data (weighted) NMF for masked cross-validation

`MzBinMaskingSplitter.mask_train_matrix` sets held-out cells to **0**
(`decomposition.py:203`). NMF therefore fits zeros there, and the test error measures how
strongly the structure resists those fake zeros. This favors under-fitting and makes
`neglog_ratio_train_test…` hard to interpret. The textbook approach (Wold-style /
bi-cross-validation) is to **ignore** held-out cells:

$$\min_{W,H\ge0} \lVert M \odot (X - WH)\rVert_F^2 + \text{penalties}$$

scikit-learn doesn't support a mask, but weighted multiplicative updates take a few lines
and run well on a GPU ([E2](#e2)):

```python
# M = 1 for training cells, 0 for held-out; l1/l2 already scaled like sklearn
WH = W @ H
H *= (W.T @ (M * X)) / (W.T @ (M * WH) + l1_H + l2_H * H + eps)
WH = W @ H
W *= ((M * X) @ H.T) / ((M * WH) @ H.T + l1_W + l2_W * W + eps)
```

Keep the "Sample" splitter as a second, easy-to-understand check, since it already does
proper held-out projection.

<a name="c2"></a>
### C2. Exported fragment m/z and intensity

Three things lower search scores for the `*_weight` spectra (`containers/__init__.py:479–489`,
`gpfrun.py:84`):

1. **Bin right edge** used as the m/z (up to +1 bin, e.g. +20 ppm).
2. **Every fragment appears twice** (one per interleaved grid, about half a bin apart), which
   inflates peak counts and can double-count in scoring.
3. **Zero-intensity peaks** fill the top-200 list when a component has fewer non-zero bins.

Suggested export step:

- For each kept bin in the component, collect the **raw peaks** from `m_long` inside that
  bin across the component's apex ± 1σ scans, and report their **intensity-weighted mean
  m/z**. This recovers the instrument's mass accuracy (usually < 5 ppm). At least use bin
  centers (`indices_to_mz(center=True)`).
- **Merge** peaks from the two grids that are within one bin width, keeping the larger (or
  summing their weights).
- **Drop** zero or near-zero weights (e.g. < 1% of the component's max).

<a name="c3"></a>
### C3. Wide-window precursor: don't hard-code charge 2

`utils.py:369` writes `Z 2` for every wide-window spectrum. The `.ms2` format allows
**several `Z` lines per spectrum**, and Comet will search each. Write z = 1…4 (with the
corresponding MH⁺ for the window center), or infer the charge from fragment isotope spacing
in the component (bins at 1.00336/z). This is especially relevant for peptidomics, where
charge distributions are broad.

<a name="c4"></a>
### C4. Deduplicate across adjacent isolation windows

Collapse only compares tiles **within one isolation window**. With narrow windows, one
precursor's isotopes span 2–3 windows, so the same peptide appears as components in
neighboring windows with near-identical spectra and RT. Add a run-level pass that compares
components in windows *k* and *k ± 1..2* (same overlap/cosine test), or at least group by
`ms1_id`. Mark them as a group rather than deleting them, because the window holding the
monoisotopic peak often has the cleanest spectrum.

<a name="c5"></a>
### C5. Better "raw" spectra

`raw` exports the single scan nearest the apex (`containers/__init__.py:471–476`), which
is still fully chimeric. Two better variants to add:

- **Elution-weighted sum**: $\sum_t W_{t,k}\,X_{t,:}$ over the component's ±2σ scans, using
  raw peaks. This improves signal-to-noise and naturally down-weights co-eluting peptides
  with different profiles.
- **Correlation-filtered raw**: the apex scan's raw peaks, keeping only those whose bin has
  non-zero H for the component. That gives the instrument's exact m/z and intensities with
  NMF acting as a filter.

<a name="c6"></a>
### C6. Peak-width parameters and units

- `sigma_bounds = [0.1, 10.0]` is in **minutes**. The lower bound (6 s) is too high for
  modern nanoLC or microflow peptide peaks (σ ≈ 1.5–5 s). Consider expressing bounds in
  **scans** (convert with the median cycle time), to match `tuning.component_sigma`.
- Fit amplitude (and optionally a baseline or an exponentially modified Gaussian for tailing)
  instead of fixing the height at 1.
- Export fit R² and the number of scans above half height as quality columns.

<a name="c7"></a>
### C7. Export more per-component information

Add to the Parquet (or a sidecar `.npz`):

- the **elution profile** W column (for quantification and QC plots);
- the number of non-zero fragments;
- fit R²;
- the tile start/end;
- the component's share of the tile's total reconstruction;
- the full sparse H row on the global bin grid (for clustering, [F1](#f1)).

<a name="c8"></a>
### C8. Tuning objectives

- The two orthogonality metrics normalize the Gram matrix by row norms, not by
  $\sqrt{G_{ii}G_{jj}}$, so they aren't true cosine similarities. Normalizing W/H columns and
  rows first would make them interpretable (mean off-diagonal cosine).
- `fraction_window_component` depends on `component_sigma` in scans, which you must guess.
  You could estimate it from the data: median fitted σ from a quick first pass.
- The harmonic-mean pick min-max scales across trials, so the choice depends on which bad
  trials were sampled. If all trials tie on one objective, the result is NaN
  (`(v - v.min()).max() == 0`). Guard that, and consider choosing from the Pareto front only.

---

## D. Speed on CPU

Ordered by expected gain per unit of effort. None of this was benchmarked, so profile first:
`python -m cProfile -o prof.out -m fragment_feature_detection.cli process_mzml …`, then
`snakeviz prof.out`.

<a name="d1"></a>
### D1. Vectorize the mzML converter (likely the biggest single gain)

`converter.py:107–119` builds a **Python list per peak** (`base_scan_data + [mz,
intensity] + extra_data`), then `np.vstack` over millions of small lists. Replace it with
per-scan arrays:

```python
n = mz_array.size
block = np.empty((n, 7), dtype=np.float32)
block[:, 0] = scan["MS2TargetMass"]; block[:, 1] = scan["ScanNum"]; block[:, 2] = scan["RetentionTime"]
block[:, 3] = mz_array; block[:, 4] = intensity_array
block[:, 5] = scan["lowerMass"];     block[:, 6] = scan["higherMass"]
mz_arrays[scan_key].append(block)          # np.concatenate at write time
```

Expect a 10–50× speed-up of this stage. For the XML parsing itself:

- use `pyteomics.mzml.PreIndexedMzML` with its built-in multiprocessing `.map()`, or
- use **pyOpenMS** `MzMLFile().load()` / `OnDiscMSExperiment` (C++; already a dependency), or
- use `ThermoRawFileParser` → mzML with no compression, since zlib decoding costs time too.

<a name="d2"></a>
### D2. Index `m_long` by scan for export

`containers/__init__.py:476` runs `gpfrun.m_long[gpfrun.m_long[:, 0] == scan]`, a full
scan of the table **for every component**, so the cost is O(components × peaks). `m_long`
comes out of `np.unique`, so it is already sorted by scan. Use
`lo, hi = np.searchsorted(m_long[:, 0], [s, s], side=["left", "right"])`, or precompute the
start/end of each scan once per GPF.

<a name="d3"></a>
### D3. Faster MS1 matching solver

`fitpeaks.least_squares_with_l1_bounds` uses SLSQP with **finite-difference gradients**:
about *p* extra function evaluations per iteration for *p* candidate features. Because
$c \ge 0$, the L1 term is linear and the objective is smooth, so use L-BFGS-B with the
exact gradient:

```python
def f(c):
    r = X @ c - y
    return r @ r + alpha * c.sum(), 2 * X.T @ r + alpha
res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B",
               bounds=[(0, 1)] * X.shape[1])
```

This gives the same solution and is often 10–100× faster. You can also precompute
`XᵀX` and `Xᵀy` (the problems are tiny). Alternatively, use `sklearn.linear_model.Lasso(positive=True)`
with `alpha_lasso = alpha / (2 n)`, then clip at 1.

<a name="d4"></a>
### D4. Vectorize the redundant-component collapse

- **Status:** **Partly done.** The pre-filter and the ±6σ bounds went in with [A14](#a14). The closed-form overlap and the all-pairs sparse cosine are still open.

`gpfrun.py:221–289` is a nested Python loop over (tiles × overlapping tiles × components²).
Each step evaluates **two 10,000-point pdfs** (`fitpeaks.py:30`) and a dense cosine on the
union bin vector.

- **Gaussian overlap in closed form.** For the "fraction of curve 1 under min(p1, p2)", use
  the intersection points of the two Gaussians and `scipy.special.ndtr`. Or use the
  Bhattacharyya coefficient
  $BC = \sqrt{\tfrac{2\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2}}\exp\!\left(-\tfrac{(\mu_1-\mu_2)^2}{4(\sigma_1^2+\sigma_2^2)}\right)$,
  which is O(1) and vectorizable over all pairs.
- **Pre-filter** pairs with $|\mu_1-\mu_2| > 3(\sigma_1+\sigma_2)$.
- **Cosine for all pairs at once.** Put all kept H rows of the GPF on the global bin grid as a
  sparse matrix, normalize the rows, and compute `S = Hn @ Hn.T` restricted to candidate
  pairs.

This turns minutes of Python loops into a few sparse matrix products per GPF.

<a name="d5"></a>
### D5. Warm-start NMF from the previous tile

Consecutive tiles share 80% of their scans. Initialize tile *i + 1* from tile *i*: shift W
by 30 rows and pad; reuse H on the shared bins. `fit_nmf_matrix_custom_init` already
supports `W_init/H_init` (`init='custom'`). This usually cuts iterations several-fold. It
makes tiles depend on each other (processed in order, parallel across GPFs instead of across
tiles) and slightly changes results, so check against cold starts.

<a name="d6"></a>
### D6. Reduce inter-process overhead

- Each joblib task pickles a whole `ScanWindow`: filtered matrix, `_m_unfilter` (the full
  sparse slice), `_mz_unfilter` over **all** GPF bins, and MS1 features, both ways, for
  **each stage**.
  - Parallelize **per GPF** (one task per isolation window, running all its tiles), or
  - batch tiles (`batch_size` in `Parallel`), and don't keep `_m_unfilter` after filtering.
- Try `Parallel(prefer="threads")` for the NMF stage. scikit-learn's CD update is compiled
  Cython, and threads avoid pickling entirely. Limit BLAS threads per worker
  (`threadpoolctl`) to avoid oversubscription whichever backend you use.

<a name="d7"></a>
### D7. Fuse stages per GPF (one read, one write)

`process_mzml` runs *fit* → *attach MS1* → *match* → *dump* as four full passes over all
GPFs. In lazy mode each pass reloads each GPF from HDF5 (`GPFRun.from_h5` re-reads
`m_long`, the sparse matrix, and every tile) and writes again. A single
`process_gpf(gpf_index)` that does all four and returns a DataFrame would cut I/O about 4×.
It also makes [A3](#a3)-type ordering bugs impossible, and parallelizes cleanly across GPFs.

<a name="d8"></a>
### D8. Faster MS1 feature attachment

`add_ms1_features_scanwindow` filters the **whole** Biosaur2 table and runs a row-wise
`DataFrame.apply` for every tile. Instead:

- compute all isotope m/z values once per run (vectorized), and build an interval index on
  RT (sorted `rtStart` plus `searchsorted`) and on m/z;
- for each GPF, pre-select features whose isotopes hit the window bounds, then per tile only
  filter on RT;
- interpolate all attached features onto the tile RT grid at once.

<a name="d9"></a>
### D9. Make tuning cheaper

- Defaults are 500 trials × 3 splits × ~50 tiles, about 75k NMF fits. Try `n_iter` 100–150
  with TPE/NSGA-II. Multi-objective searches in 3 parameters usually settle quickly.
- Report per-split results with `trial.report` and use Optuna pruning (`MedianPruner`) to
  stop bad regions after the first split.
- Tune once per instrument/method and reuse (`run_tune = False`).
- Each worker is sent a copy of all tuning windows. With option 3 of [B4](#b4), they're sent
  once.

<a name="d10"></a>
### D10. Smaller items

- `ScanWindow.filter_clip_constant` on sparse input builds a dense `np.matrix` division.
  Doing the column-mean/column-max on `.data` directly avoids that.
- `filter_denoise_scans` converts to dense and back. Fine at 150 rows, but you can skip the
  CSR re-conversion when the next step (float casting, NMF) wants dense anyway.
- `GPFRun.slice_scan_window` recomputes `self._discretize.indices_to_mz(self._bin_indices)`
  for every tile. Compute it once per GPF.

<a name="d11"></a>
### D11. Sparse percentile filter (fixed)

- **Status:** **Fixed.**
- **Where:** `ScanWindow.filter_percentile_scans`.
- **What happened:** it ran `m[m < threshold] = 0` on a sparse matrix. SciPy handles that
  by inserting entries one at a time and re-sorting the matrix.
- **Measured:** 30 of the 40 s spent building the synthetic run (the largest single
  hotspot in the main process).
- **Fix:** edit `m.data` directly and call `eliminate_zeros()`. Results are identical, and
  building the run is now 4× faster (40 s → 10 s; whole run 94 s → 70 s). See
  [RUNNING.md § Performance](RUNNING.md#performance-bottlenecks-and-what-to-adjust) for
  the full stage timings.

---

## E. GPU acceleration

The workload is **thousands of small, independent NMFs** (150 × a few thousand, rank ≤ 20).
Launching one GPU kernel per small NMF wouldn't help much. **Batching** them does. All
tiles of an isolation window have the same number of rows (150), and the columns can be
padded.

<a name="e1"></a>
### E1. Batched NMF (PyTorch or CuPy)

- **Batching.** Stack all B tiles of a GPF (or of several GPFs) into a tensor
  `X: (B, 150, m_max)`, zero-padded in m/z. Zero columns stay zero under multiplicative or
  HALS updates, so padding doesn't change the solution.
- **Algorithm.** HALS / coordinate descent (matches the current `'cd'` solver) or
  multiplicative updates, implemented with batched matrix products (`torch.bmm`).
- **Same objective.** Use scikit-learn's objective *including its penalty scaling*
  (`l1_W = alpha_W·l1_ratio·m`, `l2_W = alpha_W·(1−l1_ratio)·m`, and likewise for H with
  n), so tuned `alpha_W/alpha_H/l1_ratio` values carry over.
- **Initialization.** Batched NNDSVD via `torch.linalg.svd` on the batch (or reuse
  CPU-computed inits).

```python
import torch
def batched_nmf_mu(X, k, a_W, a_H, rho, iters=500, eps=1e-10):
    B, n, m = X.shape
    l1W, l2W = a_W * rho * m, a_W * (1 - rho) * m      # sklearn scaling (W penalty ∝ n_features)
    l1H, l2H = a_H * rho * n, a_H * (1 - rho) * n      # (H penalty ∝ n_samples)
    W = torch.rand(B, n, k, device=X.device) * X.mean((1, 2), keepdim=True).sqrt()
    H = torch.rand(B, k, m, device=X.device) * X.mean((1, 2), keepdim=True).sqrt()
    for _ in range(iters):
        H *= (W.transpose(1, 2) @ X) / (W.transpose(1, 2) @ W @ H + l1H + l2H * H + eps)
        W *= (X @ H.transpose(1, 2)) / (W @ (H @ H.transpose(1, 2)) + l1W + l2W * W + eps)
    return W, H
```

(Multiplicative updates, shown for brevity. HALS converges faster and matches `'cd'`
better. Scale the sklearn Frobenius term's ½ consistently when you port the penalties.)

- **Expected benefit.** On one modern GPU, thousands of 150 × 3000 × 20 factorizations run
  in seconds instead of minutes to hours of CPU time. Tuning ([E2](#e2)) benefits the most,
  because every trial is another large batch.
- **Validation.** On a few hundred tiles, compare with sklearn:
  - reconstruction error within a few %;
  - number of surviving components;
  - matched spectra cosine > 0.95, pairing components with the Hungarian algorithm on cosine.
- **Integration.** Add `nmf.backend = 'sklearn' | 'torch'` to the config. Keep sklearn as
  the reference implementation. Use float32 (float16 accumulations are risky for the
  updates).

<a name="e2"></a>
### E2. Tuning on GPU, with the proper masked objective

Each Optuna trial is `n_splits × tiles` independent NMFs, which is one batch. With the
weighted updates from [C1](#c1), the mask is one more tensor in the batch. This both
**fixes the CV bias** and makes each trial take about one batched solve. You could even
evaluate several trials' hyperparameters in one batch (the penalties become per-batch
tensors).

<a name="e3"></a>
### E3. Other parts that map well to GPU (or vectorized CPU)

| Part | Approach | Worth it? |
|---|---|---|
| MS1 matching | Batched projected-gradient / FISTA NNLS with box [0, 1] over all components of a GPF (pad feature count) | Moderate. [D3](#d3) on CPU may be enough |
| Gaussian fits | Moment estimates + a few batched Gauss–Newton/LM steps, or a closed-form log-parabola fit around the apex, for all components at once | Moderate. Also removes `curve_fit` failures |
| Collapse | Closed-form overlap + sparse `Hn @ Hn.T` ([D4](#d4)) on GPU via `torch.sparse` / CuPy | Small once vectorized |
| **All-vs-all spectral similarity for clustering / delta-mass discovery** ([F2](#f2), [F3](#f3)) | Sparse matrix products of normalized spectra on a common grid, and FFT cross-correlation for shift searches | **High.** O(N²) pairs across runs is where a GPU clearly wins |
| mzML conversion, HDF5 I/O | — | No. CPU/I/O bound |

---

## F. Features for the peptidomics goals

These build on the component table to support clustering, deep dives for missed
identifications, re-searching, and PTM discovery through spectral similarity. DETAILS §12
explains the reasoning behind each.

<a name="f1"></a>
### F1. A component spectral store

Write every component's spectrum on the **global bin grid** as one sparse matrix
(`scipy.sparse.save_npz`), plus a metadata table (run, isolation target, μ, σ, MS1 matches,
quality). Also export an **MGF/MSP library** (MSFragger, Sage and spectral-library tools can
read MGF). This is the input for everything below.

<a name="f2"></a>
### F2. Clustering module

- Normalize the spectra (√intensity + unit length). Keep the top-N peaks after merging the
  two grids ([C2](#c2)).
- **kNN graph** by cosine (sparse matmul, or `pynndescent`/FAISS for big sets), then Leiden
  or HDBSCAN clustering.
- Within clusters: consensus spectrum, RT and precursor-mass statistics, and the ID
  propagation candidates.
- Across runs: cluster after RT alignment (a LOWESS fit on shared identified peptides) to
  find **recurring unidentified species**.

<a name="f3"></a>
### F3. Delta-mass discovery and site localization

For each unidentified component U and identified neighbor P (same cluster, or precursor
within ±500 Da):

1. **Modified cosine**: match U's peaks to P's at shift 0 or at Δ/z (z = 1, 2). Record
   matched-unshifted and matched-shifted counts.
2. **Δ histogram** over all high-scoring pairs, annotated against Unimod. For peptidomics,
   list C-terminal amidation, pyro-Glu, acetylation, oxidation, phospho/sulfo,
   deamidation, and ± single residues (N/C-terminal ladders are very common for endogenous
   peptides).
3. **Localization score**: for each residue position *i* of P's sequence, predict which b/y
   ions shift (DETAILS §12.4 table), count matches, and give a probability over positions
   (Ascore-like binomial, or the ratio of best to second-best position).
4. **FFT shift screen** (GPU-friendly): cross-correlate binned spectra over all shifts in
   ±500 Da to find Δ values with strong peak overlap, *without* needing the precursor mass.
   Useful for wide-window components.
5. Output a table (`U`, `P`, `Δ`, `Unimod candidates`, `best site`, `site score`, `modified
   cosine`, `n shifted`) and a **re-search parameter file** listing the discovered Δ values
   as variable mods on the suggested residues.

<a name="f4"></a>
### F4. Re-search loop

A command such as `ffd research --ids psms.tsv --features sample.features.parq`:

- selects unidentified, high-quality components (fit R², explained variance, fragment
  count);
- writes them to `.ms2`/MGF with multiple precursor candidates (MS1 matches plus
  wide-window z = 1–4);
- generates search parameters (open search, or mass-offset with the F3 Δ list);
- merges results back by `ms2_component`.

<a name="f5"></a>
### F5. Rescoring input

Export a Percolator/mokapot `.pin` join with component features: σ, fit R², MS1 coefficient,
global/individual explained variance, apex-RT difference between MS2 and MS1, number of
fragments, fraction of the component's intensity explained by matched ions, and the
isolation-window offset of the precursor. These features are specific to deconvolved DIA and
tend to separate true from false matches well.

<a name="f6"></a>
### F6. Quantification

With W exported ([C7](#c7)), provide per-component MS2 areas (sum of W × rescaled H), and
optionally a summed "best fragments" XIC. In crowded peptidomics MS1 regions, MS2-level
quantities from deconvolved components are often more selective than MS1 feature
intensities.

---

## G. Suggested order of work

1. ~~**Make runs succeed and be correct**~~ **Done** (see [Fix status](#fix-status)).
   Remaining: [A18](#a18) (MS1 scan → RT table), run the new tests on real data, add CI
   ([B1](#b1)), and decide on [A22](#a22).
   - **Done:** opt-in compact storage, split files and one-row-per-component Parquet
     ([B11](#b11)). Next: [D7](#d7) (one pass per isolation target).
2. **Config hygiene** (≈ 1 day): finish [A6](#a6) (`Optional[Config] = None` defaults),
   move hard-coded parameters into `Config`, fix the pins and drop unused dependencies
   ([B7](#b7)). [B3](#b3) is done.
3. **Output quality for searching** (≈ 2–3 days): [C2](#c2), [C3](#c3), [C4](#c4), [C5](#c5),
   [C7](#c7).
4. **CPU speed** (≈ 2–4 days): [D1](#d1), [D2](#d2), [D3](#d3), [D4](#d4), [D7](#d7). Profile
   before and after.
5. **Tuning quality and speed** (≈ 1 week): [C1](#c1) + [E1](#e1)/[E2](#e2) (GPU backend with
   masked objective), [B8](#b8), [D9](#d9).
6. **Discovery features** (ongoing): [F1](#f1) → [F2](#f2) → [F3](#f3) → [F4](#f4)/[F5](#f5).
