# Running `fragment_feature_detection`: step by step

This guide goes from a vendor raw file to searchable, deconvolved component spectra. It
also has a reference for every CLI option and configuration key. For *why* each step
works the way it does, see [DETAILS.md](DETAILS.md).

> This guide was written from the source code (commit `558ba09`) and updated after the
> bug fixes listed in [SUGGESTIONS.md § Fix status](SUGGESTIONS.md#fix-status). The
> pipeline has been run end to end on **synthetic** narrow-window data and through the
> regression tests in `tests/`. It has **not** yet been checked on real DIA data.

---

## Contents

1. [Known pitfalls (read before your first run)](#known-pitfalls-read-before-your-first-run)
2. [Step 0: Environment](#step-0-environment)
3. [Step 1: Raw → centroided mzML](#step-1-raw--centroided-mzml)
4. [Step 2: MS1 features with Biosaur2 (optional)](#step-2-ms1-features-with-biosaur2-optional)
5. [Step 3: Create and edit a config INI](#step-3-create-and-edit-a-config-ini)
6. [Step 4: Run the pipeline (`process_mzml`)](#step-4-run-the-pipeline-process_mzml)
7. [Step 5: Export spectra (`dump_df_to_ms2`)](#step-5-export-spectra-dump_df_to_ms2)
8. [Step 6: Search and downstream analysis](#step-6-search-and-downstream-analysis)
9. [Outputs](#outputs)
10. [CLI reference](#cli-reference)
11. [Configuration reference](#configuration-reference)
12. [Hard-coded parameters (not in the config)](#hard-coded-parameters-not-in-the-config)
13. [Using the Python API directly](#using-the-python-api-directly)
14. [Performance: bottlenecks and what to adjust](#performance-bottlenecks-and-what-to-adjust)
15. [Troubleshooting](#troubleshooting)

---

## Known pitfalls (read before your first run)

The crashes and silent errors listed here in earlier versions of this guide have been
fixed (see [Fixed since `558ba09`](#fixed-since-558ba09) below). What remains are
behaviors to be aware of rather than bugs that stop a run.

| # | Behavior | Why it matters | What to do |
|---|---|---|---|
| 1 | Exported fragment m/z values are **bin right edges**, each fragment appears **twice** (one per interleaved grid), and short spectra are padded with **zero-intensity** peaks | Search scores are lower than they could be ([SUGGESTIONS C2](SUGGESTIONS.md#c2)) | Set the fragment tolerance to ≥ 1.5× the bin width (see [Step 5](#step-5-export-spectra-dump_df_to_ms2)) |
| 2 | `--wide_window` spectra are always written with charge **2** | Other charge states are only found if the search engine tries them ([C3](SUGGESTIONS.md#c3)) | Let the engine try charges 1–4 |
| 3 | MS1 hill intensities are assumed to be **evenly spaced** between `rtStart` and `rtEnd` | Hills that skip MS1 scans are slightly distorted in time ([A18](SUGGESTIONS.md#a18)) | Usually minor. Check matching on a few known peptides |
| 4 | The default `Config()` (no INI) uses `lazy = True` | Every stage re-reads and re-writes each isolation target from HDF5, which is slower, and the default `.h5` layout is large | Use `lazy = False` if the run fits in RAM, or turn on `compact_storage` (see [Disk usage](#disk-usage)) |
| 5 | `ScanNum` / `ms2_apex_scan_number` is the 1-based **spectrum index** in the mzML, not the vendor scan number | Joins to vendor tools need care | Map through the mzML spectrum index |
| 6 | Tuning writes its own log and pickle next to the run's log file. From the Python API with no log file configured, they go to the **current working directory** as `optuna_tuning.tuning_<timestamp>.log` and `optuna_tuning.pkl` (or `randomized_tuning.*`) | Files appear in unexpected places | Add a `logging.FileHandler` before tuning, or pass `save_run=False` |
| 7 | Results change compared with runs made before the fixes | Lazy runs no longer contain duplicates, the collapse keeps a different duplicate, each isolation target gets one extra tile at the end, and MS1 isotope spacing changed ([Behavior changes](SUGGESTIONS.md#behavior-changes-to-expect)) | Don't mix old and new outputs in one comparison |

### Fixed since `558ba09`

If you have notes or scripts with workarounds for the old behavior, they are no longer
needed:

| Old symptom | Status |
|---|---|
| `TypeError: ... unexpected keyword argument 'alpha_w'` whenever an INI was used | Fixed. INI keys keep their case, and lower-case keys from older INIs are matched to the right setting ([A1](SUGGESTIONS.md#a1)) |
| `AttributeError: MS1 features have not been matched...` without `--ms1_features_file` | Fixed. `match_ms1` is set to `False` automatically when no MS1 file is given ([A2](SUGGESTIONS.md#a2)) |
| Duplicate components with `lazy = True` | Fixed ([A3](SUGGESTIONS.md#a3)) |
| Optuna tuning fails on Windows (`/tmp/`) | Fixed. Tuning uses a journal file in the system temp folder ([B4](SUGGESTIONS.md#b4)) |
| `ValueError: Returned fits not consistent...` after a few failed trials | Fixed. Tuning continues with a warning unless fewer than half of `n_iter` trials succeed ([A5](SUGGESTIONS.md#a5)) |
| `TypeError` with `include_gpf_bounds = False` | Fixed ([A9](SUGGESTIONS.md#a9)) |
| `PackageNotFoundError` when running from a source checkout | Fixed ([B2](SUGGESTIONS.md#b2)) |
| Outputs not written to `--output_dir` | Fixed. All outputs except the log go to `--output_dir` ([B5](SUGGESTIONS.md#b5)) |
| Settings leaking between runs in one Python session | Mostly fixed. Each `Config()` has its own copy of every section ([A6](SUGGESTIONS.md#a6)) |
| `ZeroDivisionError` / negative fraction in `get_tuning_windows` on short runs | Fixed. It now warns and returns no windows ([A19](SUGGESTIONS.md#a19)) |
| Last scans of each isolation target never analyzed | Fixed. A final tile is added at the end ([A17](SUGGESTIONS.md#a17)) |
| Duplicated log lines during tuning | Fixed ([B9](SUGGESTIONS.md#b9)) |
| Tuning judged each trial on only 3 windows | Fixed. Every sampled window is scored, so tuned values may differ from older runs ([A23](SUGGESTIONS.md#a23)) |
| `OSError: ... file is already open for read-only` when reloading a saved run with `lazy = False` | Fixed ([A24](SUGGESTIONS.md#a24)) |
| Building the run (`Creating MSRun object...`) is very slow | About 4× faster: the sparse percentile filter was the hotspot ([D11](SUGGESTIONS.md#d11)) |
| Some windows silently end up with no components (log: `array must not contain infs or NaNs`) | Fixed. NMF retries those windows in float64 ([A32](SUGGESTIONS.md#a32)) |
| `ValueError: Cannot set a DataFrame with multiple columns to the single column mz_isotopes` while adding MS1 features | Fixed ([A33](SUGGESTIONS.md#a33)) |

---

## Step 0: Environment

```bash
conda create -n ffd python=3.10 -y
conda activate ffd
cd fragment_feature_detection
pip install -e .
ffd --help
```

You should see four commands: `generate_template_ini`, `convert_mzml_h5long`,
`process_mzml`, `dump_df_to_ms2`.

Use Python 3.10. The NumPy/SciPy/pyOpenMS pins do not install on 3.12+.

---

## Step 1: Raw → centroided mzML

The converter keeps **MS2 scans only** and turns every peak into one row, so the spectra
**must be centroided**. MS1 scans are needed only by Biosaur2 (step 2).

**ProteoWizard msconvert** (Windows GUI or CLI):

```bash
msconvert sample.raw --mzML --64 --zlib \
    --filter "peakPicking vendor msLevel=1-" \
    -o ./mzml
```

**ThermoRawFileParser** (cross-platform):

```bash
ThermoRawFileParser -i sample.raw -o ./mzml -f 2      # -f 2 = indexed mzML, centroided by default
```

Check that the mzML has the following for every MS2 scan (msconvert writes them for
Thermo DIA):

- `isolation window target m/z`, `isolation window lower offset`, `isolation window upper offset`
- `selected ion m/z`
- `scan start time`

What the converter does with them:

- **`isolation window target m/z` is the grouping key.** Every distinct target value becomes
  one "GPF" time series. Staggered or overlapping window schemes are fine, because each
  distinct target becomes its own series.
- **`lowerMass`/`higherMass` are built around `selected ion m/z`**, not around the target:
  `selected ion m/z − lower offset` and `selected ion m/z + upper offset`. For DIA these
  are usually the same value.
- **`ScanNum` is the 1-based spectrum *index*** in the mzML, not the vendor scan number.

---

## Step 2: MS1 features with Biosaur2 (optional)

MS1 matching links each MS2 component to specific precursor(s): exact mass, charge,
apex RT and intensity. You need it if you want the precursor-specific `.ms2` output
(non-`--wide_window`).

```bash
pip install biosaur2
biosaur2 sample.mzML -o sample.features.tsv
```

The loader reads the file as tab-separated and needs these columns:

| Column | Used as |
|---|---|
| `massCalib` | Neutral monoisotopic mass (converted to MH⁺ in `.ms2`) |
| `mz`, `charge`, `nIsotopes` | Isotope m/z list for the window-membership test |
| `rtStart`, `rtEnd`, `rtApex` | Time extent and apex (minutes; must match the mzML RT units) |
| `intensityApex`, `intensitySum` | Reported in the output |
| `nScans`, `scanApex` | Stored |
| `mono_hills_scan_lists` | List of MS1 scan indices of the monoisotopic hill (parsed with `pd.eval`) |
| `mono_hills_intensity_list` | Matching intensities (parsed with `pd.eval`) |

Check that your Biosaur2 version writes the two `mono_hills_*` columns. If it doesn't,
matching cannot run. The feature ID in the output (`ms1_id`) is the **row number** of the
feature in this TSV.

How the file is used:

- A feature is attached to a scan window if its `[rtStart, rtEnd]` overlaps the window
  **and** at least one isotope peak `mz + i·1.00336/charge` (for `i < nIsotopes`) falls inside
  that window's isolation bounds. With narrow windows, one precursor can be attached to
  several adjacent isolation windows. That is intended.
- The hill intensities are assumed to be evenly spaced in time between `rtStart` and
  `rtEnd`. They are linearly interpolated onto the MS2 scan times of each scan window. The
  profile drops to zero one hill step before the first point and after the last point, and
  is zero outside the hill.

---

## Step 3: Create and edit a config INI

```bash
ffd generate_template_ini --output_dir .
# -> ./template.ini  (the Config() class defaults)
```

or copy `fragment_feature_detection/config.ini` (a tuned example that **differs** from the
class defaults; see the [reference table](#configuration-reference)).

No edits are required. Both files load as they are. (Older INIs with lower-case
`alpha_w`/`alpha_h` also load correctly now.)

**Recommended edits for a first run:**

```ini
[general]
lazy = False           ; faster: keeps the run in RAM instead of re-reading HDF5 per stage
tqdm_enabled = True    ; progress bars

[discretization]
tolerance = 2e-05      ; bin width as a fraction (2e-05 = 20 ppm); match to your MS2 mass accuracy

[fitpeaks]
sigma_bounds = [0.02, 10.0]   ; in RT units (minutes). The 0.1-min default lower bound is wide for sharp peaks

[tuning]
n_iter = 100           ; the default of 500 is expensive; start smaller
```

INI syntax rules (values are read as Python literals with `ast.literal_eval`):

- Strings can be quoted (`bin_width = 'ppm'`). Values that aren't valid Python literals,
  such as an unquoted `ppm`, are read as plain strings.
- Booleans are `True`/`False`. Missing values are `None`.
- Lists, tuples and dicts use Python syntax: `sigma_bounds = [0.1, 10.0]`.
- Keys that match an existing setting are matched **case-insensitively** (`alpha_w` sets
  `alpha_W`). New keys keep the case you write.
- Expressions such as `1/3` are not evaluated. Write `0.333`.

---

## Step 4: Run the pipeline (`process_mzml`)

```bash
ffd process_mzml sample.mzML \
    --ms1_features_file sample.features.tsv \
    --config_ini_file template.ini \
    --output_dir ./ffd_out \
    --n_jobs 8
```

What happens, in order (each item is logged):

1. **Config load.** `Config.from_ini(...)`, or `Config()` defaults if no INI is given.
2. **Convert** `sample.mzML` → `./ffd_out/sample.h5`. MS2 peaks are grouped by isolation
   target in HDF5 groups named by the target m/z. Each group holds a dataset `ms2_long` with
   columns `[target, scan, rt, mz, intensity, lowerMass, higherMass]` (float32, gzip).
   Non-finite values are dropped. With `filter_spectra = True`, peaks below the given
   per-scan percentile are dropped.
3. **Build the MSRun.** For each isolation target (sorted by m/z):
   - discretize fragment m/z onto `steps` interleaved ppm grids;
   - pivot to a sparse *scan × bin* matrix;
   - tile into `ScanWindow`s (`scan_width` scans, starting every `scan_overlap` scans, plus
     a final tile that ends on the last scan);
   - filter/denoise each window (`scan_window_filter_during_load = True`).

   With `lazy = True`, each GPFRun is written into `sample.h5` under `gpf_runs/gpf_<i>` and
   reloaded when needed.
4. **Tuning** (if `run_tune = True`).
   - About `window_sampling_n` scan windows are sampled across all isolation targets,
     skipping `exclude_scan_window_edges × scan_overlap` scans at both ends.
   - Optuna (or a randomized search) chooses `alpha_W`, `alpha_H` and `l1_ratio` by masked
     cross-validation, and the best values are written into `config.nmf`.
   - Failed trials are skipped with a warning. Tuning stops with an error only if fewer than
     half of `n_iter` trials succeed.
   - A separate tuning log and a pickle of the search object are written next to the run log.
5. **NMF.** Every scan window of every isolation target is factorized (in parallel across
   windows, `n_jobs`).
   - A Gaussian is fitted to each component's elution.
   - Components whose fit fails, or whose ±2σ extends into the 5 zeroed edge scans, are dropped.
   - Redundant components in overlapping windows are collapsed, and then (in lazy mode)
     the windows are written to HDF5.
6. **MS1 matching** (only if `--ms1_features_file` is given; this also forces
   `match_ms1 = True`. Without the file, `match_ms1` is set to `False`).
   - Features are attached to windows.
   - Each component's elution (its 99% Gaussian interval widened by `extend_w_fraction` on
     both sides) is regressed on the attached MS1 elution profiles, with coefficients limited
     to [0, 1] and an L1 penalty `alpha`.
7. **Export.** The component table is written to `./ffd_out/sample.features.parq`, and the
   final config (with tuned values) to `./ffd_out/sample.out.ini`. The `.out.ini` can be
   reused as-is with `--config_ini_file` (set `run_tune = False` to skip tuning).

To **skip tuning** (for example, reusing parameters tuned on a similar run), set
`run_tune = False` and put the tuned `alpha_W`, `alpha_H` and `l1_ratio` in `[nmf]`. The log
will warn: *"Skipping hyperparameter tuning. This will likely over/under fit your data."*

If you only want the conversion:

```bash
ffd convert_mzml_h5long sample.mzML --output_dir ./ffd_out [--config_ini_file template.ini]
```

The INI matters here only for `[ms2_preprocessing]` and `tqdm_enabled`.

---

## Step 5: Export spectra (`dump_df_to_ms2`)

```bash
ffd dump_df_to_ms2 sample.features.parq --output_type <raw|intensity_transform_weight|raw_weight> [--wide_window]
```

`--output_type` chooses the spectrum written for each component:

| `--output_type` | Peaks written | When to use |
|---|---|---|
| `raw` *(default)* | The **observed** centroided MS2 spectrum at the scan closest to the component's fitted apex (all peaks, not deconvolved) | Baseline; compare with the deconvolved versions; spectral-library style matching |
| `raw_weight` | The component's **NMF spectrum** (row of H): top `top_n_rescaled_peaks` bins by weight, in scaled units (each m/z channel was scaled to its maximum before NMF, so values are relative) | Clustering and similarity comparisons where relative presence matters more than absolute intensity |
| `intensity_transform_weight` | The NMF spectrum **multiplied back by each channel's maximum intensity**, which gives an approximate intensity-scaled spectrum. Top `top_n_rescaled_peaks` peaks | **Default choice for database searching** of deconvolved spectra |

`--wide_window` sets how the precursor is written:

| Mode | Rows written | Precursor line |
|---|---|---|
| default (MS1-matched) | One spectrum **per (component, MS1 match)** that has a coefficient. Rows with no match are dropped. Requires MS1 matching. | `S` line m/z = Biosaur2 `mz`. `Z` line = Biosaur2 charge, MH⁺ = `massCalib + 1.0073` |
| `--wide_window` | One spectrum **per component** | `S` m/z = isolation-window center. `Z` = charge **2**, MH⁺ = `2·center − 1.0073` |

The output is `sample.features.<output_type>[.wide].ms2` in `--output_dir` (default: next
to the Parquet file). Scan
numbers in the `.ms2` are simply 1…N in row order. Use `ms2_file_scan_number` order, or
re-join on row order, to map search results back to the Parquet rows.

Things to know about the exported peaks:

- **Peak m/z values are bin right edges**, not the measured m/z. With the default grids,
  each value is up to one bin width (the `tolerance`, e.g. 20 ppm) above the true m/z.
- Because there are two interleaved grids (`steps = 2`), each fragment typically shows up as
  **two nearby peaks** (about half a bin apart).
- If a component has fewer than 200 non-zero bins, the top-200 list is padded with
  **zero-intensity** peaks.

Set the search engine's fragment tolerance to cover this, for example ≥ 1.5× the bin width
in ppm. Or post-process the files (see [SUGGESTIONS.md](SUGGESTIONS.md#c2)).

---

## Step 6: Search and downstream analysis

- **Comet** reads `.ms2` directly. For MSFragger, Sage, and others, convert to MGF or mzML,
  e.g. `msconvert sample.features.intensity_transform_weight.ms2 --mgf`.
- A reasonable strategy for peptidomics:
  1. Search `intensity_transform_weight` (MS1-matched) with non-specific cleavage and a small
     set of variable mods.
  2. Search `--wide_window` spectra to rescue components that have no MS1 match. Allow
     charges 1–4 in the engine, because the file says charge 2.
  3. Take the unidentified components into an **open / mass-offset search**, or into the
     spectral-similarity comparisons described in [DETAILS.md § 12](DETAILS.md#12-using-the-components-downstream).
- Join search results back to `sample.features.parq` to get component RT, σ, MS1
  coefficient and explained variance. These make good features for rescoring (e.g.
  Percolator/mokapot).

---

## Outputs

### Files

| File | Written by | Location |
|---|---|---|
| `<stem>.h5` | `convert_mzml_h5long` / `process_mzml` | `--output_dir` (default: next to mzML) |
| `<stem>.features.parq` | `process_mzml` | `--output_dir` (default: next to mzML) |
| `<stem>.out.ini` | `process_mzml` | `--output_dir`. Can be reused directly as a config INI |
| `YYYYmmdd-HHMMSS__<command>__<stem>.log` | every command that takes a `file` | next to the input file |
| `<log stem>.tuning_<ts>.log`, `<log stem>.tuning_<ts>.pkl` | tuning | next to the run log |
| `<parq stem>.<type>[.wide].ms2` | `dump_df_to_ms2` | `--output_dir` (default: next to the Parquet) |

### HDF5 layout (`<stem>.h5`)

```
/<target m/z as str>/ms2_long           float32 [N, 7]  target, scan, rt, mz, intensity, low, high
/gpf_runs/gpf_<i>/                      (lazy=True, or after MSRun.dump_h5)
    attrs: gpf, gpf_low, gpf_high, window_scan_overlap, window_scan_width, index
    scan_number, scan_index, bin_indices, retention_time
    m_data, m_indices, m_indptr (+attr m_shape)      CSR scan × bin matrix
    m_long                                            [scan, rt, mz, intensity] (deduplicated)
    discretize/                                       bin grid
    scan_windows/window_<k>/                          filtered matrix, W, H, fits, MS1 features, matches
```

With `compact_storage = True`, each `window_<k>` leaves out `mz_unfilter`,
`mz_bin_indices_unfilter` and `mz_masked` (marked by the `compact_axes` attribute; they are
rebuilt from its GPF's `bin_indices` on load). The window's MS1 features are stored as one
`ms1_feature_table` group instead of one `ms1_features/feature_<n>` group each. The bin grid
is stored once in `/shared_arrays` and linked into every `discretize/` group.

With `gpfs_per_file = N`, the `/gpf_runs/gpf_<i>` groups are written to part files instead
of the main file:

```
<stem>.h5                       converted peaks, attrs incl. gpf_files (JSON: index -> part file)
<stem>.gpf_runs/part_0000.h5    /gpf_runs/gpf_0 … gpf_<N-1>
<stem>.gpf_runs/part_0001.h5    /gpf_runs/gpf_<N> … gpf_<2N-1>
…
```

Keep the `.h5` and its `.gpf_runs/` folder together. `MSRun.from_h5("<stem>.h5")` finds the
parts through `gpf_files`, so a split run reloads like a single file.

### Parquet columns (`<stem>.features.parq`)

One row per component. With MS1 matching, there is **one row per (component, kept MS1
match)**, and components with no kept match get one row with NaN MS1 fields. A component
matched to several MS1 features therefore appears on several rows, and its spectra are
repeated on each.

With `[feature_output] explode_ms1_matches = False`, the table has **one row per component**
and each `ms1_*` match column holds an array (one entry per kept match, empty if none). It
holds the same data without the repeats. `dump_df_to_ms2` reads both layouts and writes
identical `.ms2` files. In Python, `fragment_feature_detection.utils.explode_ms1_matches(df)`
turns it into the one-row-per-match layout.

| Column | Meaning |
|---|---|
| `ms2_component` | ID: `"{target}_{k}-{apex scan index}_{μ}_{σ}"` (k = component slot in the window) |
| `ms2_component_rt` | Fitted Gaussian mean μ (RT units, usually minutes) |
| `ms2_component_sigma` | Fitted Gaussian σ (RT units) |
| `ms2_gpf_center`, `ms2_gpf_low`, `ms2_gpf_high` | Isolation target and bounds |
| `ms2_apex_scan_index` | 0-based position of the apex scan within this isolation target's scan series |
| `ms2_apex_scan_number` | mzML spectrum index + 1 of the apex scan |
| `ms2_apex_scan_rt` | RT of that scan |
| `ms2_component_weight_mz`, `…_weight_intensity` | Top-N bins by raw NMF weight (H row) |
| `ms2_component_reweight_mz`, `…_reweight_intensity` | Top-N bins by weight × channel max (intensity-rescaled) |
| `ms2_mz_array`, `ms2_intensity_array` | Observed centroided spectrum at the apex scan |
| `ms1_matching_coef` | Regression coefficient (0–1) of this MS1 feature for the component |
| `ms1_global_explained_variance` | R² of all kept MS1 features jointly explaining the component elution |
| `ms1_individual_explained_variance` | R² of this MS1 feature alone (with its coefficient) |
| `ms1_mass`, `ms1_charge`, `ms1_mz` | Biosaur2 neutral mass, charge, m/z |
| `ms1_apex_rt`, `ms1_apex_intensity`, `ms1_sum_intensity` | Biosaur2 apex RT and intensities |
| `ms1_id` | Row index of the feature in the Biosaur2 TSV |

MS1 matches kept for a component: the top `top_n_ms1_matches` coefficients (ties included)
that are also ≥ `matching_coefficient_threshold`. Before that, the regression has already
zeroed any coefficient < `c_cutoff`.

---

## CLI reference

All commands log to stdout. Commands that take a `FILE` argument also write a log file
next to it.

### `ffd generate_template_ini`

| Option | Default | Description |
|---|---|---|
| `--output_dir PATH` | current dir | Where to write `template.ini` (the `Config()` defaults) |

### `ffd convert_mzml_h5long FILE`

| Argument/Option | Default | Description |
|---|---|---|
| `FILE` | *required* | Input mzML (must exist) |
| `--output_dir PATH` | next to FILE | Where to write `<stem>.h5` |
| `--config_ini_file PATH` | none (`Config()` defaults) | Config INI. The settings that matter here are `include_gpf_bounds`, `filter_spectra`, `filter_spectra_percentile` and `tqdm_enabled` |

### `ffd process_mzml FILE`

| Argument/Option | Default | Description |
|---|---|---|
| `FILE` | *required* | Input mzML |
| `--ms1_features_file PATH` | none | Biosaur2 feature TSV. If given, MS1 matching runs and `match_ms1` is set to `True`; if not, `match_ms1` is set to `False` |
| `--config_ini_file PATH` | none (`Config()` defaults) | Config INI |
| `--output_dir PATH` | next to FILE | Where to write the `.h5`, `.features.parq` and `.out.ini` |
| `--n_jobs INT` | 4 | Parallel workers for tuning, NMF, MS1 feature attachment and matching |

### `ffd dump_df_to_ms2 FILE`

| Argument/Option | Default | Description |
|---|---|---|
| `FILE` | *required* | `*.features.parq` |
| `--output_type [raw\|intensity_transform_weight\|raw_weight]` | `raw` | Spectrum flavor ([table above](#step-5-export-spectra-dump_df_to_ms2)) |
| `--wide_window` | off | Ignore MS1 matches. Write the isolation center as precursor, with charge 2 |
| `--output_dir PATH` | next to FILE | Where to write the `.ms2` file |

---

## Configuration reference

The **`Config()` default** column is what you get with no INI, or from
`generate_template_ini`. The **`config.ini` (shipped)** column is the example file in the
package. "—" means the key is absent from that file, so the class default applies.

### `[general]`

| Key | `Config()` default | `config.ini` | Description |
|---|---|---|---|
| `random_seed` | 42 | 42 | Seed for tuning-window sampling, CV masks, randomized search |
| `tqdm_enabled` | False | True | Progress bars |
| `lazy` | **True** | **False** | `True`: keep GPFRuns in the HDF5 and reload them at each stage (low RAM, slower). `False`: keep everything in memory (faster, needs RAM for the whole run) |
| `downcast_intensities` | True | True | Store pivoted matrices and `m_long` as float32 |

### `[h5]`

| Key | Default | Description |
|---|---|---|
| `scan_window_save_m_unfilter` | False | Also save each window's unfiltered matrix to HDF5 (large) |
| `scan_window_filter_during_load` | True | Filter/denoise every scan window when the GPFRun is added to the MSRun |
| `compact_storage` | False | Smaller HDF5 for lazy runs, with identical results: windows don't repeat their isolation target's m/z axis, MS1 features are stored as one table per window, the bin grid is stored once per file, and datasets are compressed. Files need HDF5 ≥ 1.10 to read (any recent h5py). See [Disk usage](#disk-usage) |
| `compact_compression` | `'gzip'` | Compression used by `compact_storage`: `'gzip'` (smaller) or `'lzf'` (a little faster) |
| `gpfs_per_file` | 0 | `0`: all isolation targets in `<stem>.h5`. `N > 0`: lazy runs write them to part files of `N` targets each in `<stem>.gpf_runs/`. Results and reloading are unchanged |

### `[discretization]`: fragment m/z binning

| Key | `Config()` | `config.ini` | Description |
|---|---|---|---|
| `bin_width` | `'ppm'` | `'ppm'` | `'ppm'` (bin width grows with m/z) or `'da'` (constant) |
| `mz_low`, `mz_high` | 200.0, 2000.0 | same | Fragment m/z range kept (fragments outside are dropped) |
| `steps` | 2 | 2 | Number of interleaved grids, offset by 1/`steps` of a bin. Each peak is counted once per grid |
| `tolerance` | **2e-05 (20 ppm)** | **6e-05 (60 ppm)** | Bin width: a fraction for ppm (2e-05 = 20 ppm) or Daltons for `'da'` |

### `[ms2_preprocessing]`: applied during mzML conversion

| Key | Default | Description |
|---|---|---|
| `include_gpf_bounds` | True | Store isolation lower/upper bounds. With `False`, MS1 attachment uses target ± 1 m/z as the bounds |
| `filter_spectra` | False | Drop peaks below a per-scan intensity percentile at conversion |
| `filter_spectra_percentile` | 50 | That percentile |

### `[scan_filter]`: time tiling

| Key | Default | Description |
|---|---|---|
| `scan_width` | 150 | Scans per scan window |
| `scan_overlap` | 30 | **Stride** between window starts, despite the name. With 150/30, consecutive windows share 120 scans and each scan is in up to 5 windows |

### `[tuning]`

| Key | `Config()` | `config.ini` | Description |
|---|---|---|---|
| `run_tune` | True | — | Run hyperparameter tuning in `process_mzml` |
| `optuna` | True | True | `True`: Optuna multi-objective search. `False`: randomized search over `hyperparameter_grid` |
| `window_sampling_n` | 50 | 50 | Target total number of scan windows used for tuning (across all isolation targets) |
| `window_sampling_fraction` | 0.0125 | 0.0125 | Only used if `window_sampling_n` is missing (it can't be removed through an INI) |
| `exclude_scan_window_edges` | 10 | 10 | Skip this many strides (×`scan_overlap` scans) at the start and end of each target when sampling |
| `optuna_hyperparameter_grid` | `{'l1_ratio': (0.1, 0.9), 'alpha_W': (1e-4, 0.1, 'log'), 'alpha_H': (1e-4, 0.1, 'log')}` | same | Optuna search space: `(low, high[, 'log'])` for floats, or a tuple of strings for a categorical |
| `hyperparameter_grid` | alpha_W, alpha_H ∈ {1e-4…1e-2}, l1_ratio ∈ {0.1…0.9} | same | Randomized-search space |
| `objective_params` | `['test_reconstruction_errors', 'neglog_ratio_train_test_reconstruction_error', 'sample_orthogonality', 'weight_orthogonality', 'fraction_window_component']` | same | Objectives. Also available: `nonzero_component_fraction`, `mean_weight_sparsity`, `mean_sample_sparsity` |
| `n_iter` | 500 | 500 | Number of trials (Optuna runs about `n_iter + n_jobs`). Failed trials are skipped; the run stops only if fewer than `n_iter / 2` succeed |
| `n_components` | 20 | 20 | NMF rank used *during tuning* |
| `components_in_window` | 8.0 | 8.0 | Expected number of real components per window. Sets the target for `fraction_window_component` (and `nonzero_component_fraction`) |
| `component_sigma` | **2.0** | **3.0** | Expected elution σ **in scans**. Sets the `fraction_window_component` target = `components_in_window × 4σ / scan_width` |
| `n_splits` | 3 | 3 | CV repeats per trial |
| `test_fraction` | 0.2 | 0.2 | Fraction of matrix entries (`'Mask'`) or scans (`'Sample'`) held out |
| `splitter_type` | `'Mask'` | `'Mask'` | `'Mask'`: hide random matrix cells (signal-balanced). `'Sample'`: hide whole scans |

### `[nmf]`: the final factorization (passed to `sklearn.decomposition.NMF`)

| Key | Default | Description |
|---|---|---|
| `n_components` | 20 | Maximum components per scan window. L1 regularization switches off unneeded ones |
| `alpha_W` | 1e-05 | Regularization strength on W (elution profiles) |
| `alpha_H` | 0.0375 | Regularization strength on H (spectra) |
| `l1_ratio` | 0.75 | Mix of L1 (sparsity) vs L2 (shrinkage). 1 = pure L1 |
| `max_iter` | 500 | Solver iterations |
| `solver` | `'cd'` | `'cd'` (coordinate descent, init NNDSVD) or `'mu'` (multiplicative update, init NNDSVDa) |
| *(any other)* | — | Passed through to `NMF`, e.g. `tol = 1e-4`, `init = 'nndsvdar'`, `beta_loss = 'kullback-leibler'` (needs `solver = 'mu'`) |

### `[fitpeaks]`: Gaussian fit to each component's elution

| Key | Default | Description |
|---|---|---|
| `mu_bounds` | None | Bounds on the apex RT (RT units) |
| `sigma_bounds` | [0.1, 10.0] | Bounds on σ **in RT units (minutes)**. For sharp nanoLC peaks (σ about 2–5 s), lower the minimum |
| `clip_width` | *(not set)* | Optional: ignore this many scans at each end of the profile when fitting |

### `[feature_matching]`: MS1 ↔ MS2

| Key | Default | Description |
|---|---|---|
| `match_ms1` | True | Export MS1 matches. `process_mzml` overrides it: `True` with `--ms1_features_file`, `False` without. Only matters when using the Python API |
| `alpha` | 0.1 | L1 penalty in the bounded regression. Higher = fewer MS1 features per component |
| `c_cutoff` | 0.025 | Coefficients ≤ this are set to 0 |
| `extend_w_fraction` | 0.5 | Widen the component's 99% interval by this fraction of its width on each side before regression |

### `[feature_output]`: export

| Key | Default | Description |
|---|---|---|
| `top_n_rescaled_peaks` | 200 | Peaks kept per component for the `*_weight_*` / `*_reweight_*` spectra |
| `matching_coefficient_threshold` | 0.1 | Minimum coefficient to report an MS1 match |
| `top_n_ms1_matches` | 4 | Keep this many best MS1 matches per component (ties included) |
| `explode_ms1_matches` | True | `True`: one row per (component, MS1 match). `False`: one row per component, with the MS1 matches as arrays, so spectra aren't repeated per match. `dump_df_to_ms2` handles both |

---

## Hard-coded parameters (not in the config)

You can change these from Python, e.g. `ScanWindow._percentile_filter = 5`, before building
the MSRun.

| Where | Parameter | Value | Effect |
|---|---|---|---|
| `ScanWindow` | `_percentile_filter_scans`, `_percentile_filter` | True, 10 | Zero intensities below the 10th percentile of non-zero values in the window |
| `ScanWindow.filter_clip_constant` | `p` | 0.3 | Drop m/z channels whose mean (max-normalized) intensity > 0.3, i.e. near-constant background ions |
| `ScanWindow` | `_scale_scans` | True | Divide each m/z channel by its max in the window |
| `ScanWindow` | `_log_scans` | False | Optional `log1p` transform |
| `ScanWindow` | `_denoise_scans` (sizes 2, 2) | True | Grey closing then grey erosion along time. Removes single-scan spikes and fills single-scan gaps |
| `ScanWindow` | `_filter_edge_scans`, `_filter_edge_nscans` | True, 5 | Zero the first/last 5 scans. Drop components within 2σ of them |
| `ScanWindow` | `_downcast_h5`, `_downcast_scans`, `_downcast_bins` | True | Round-trip intensities through float16 (for reproducibility with HDF5); float32/int32 elsewhere |
| `GPFRun.collapse_redundant_components` | `overlap_threshold`, `similarity_threshold` | 0.5, 0.75 | Merge components whose Gaussian overlap > 0.5 and spectral cosine similarity > 0.75. The one with the lower Gaussian-fit error is kept |
| `fit_ms1_ms2_feature_matching_scanwindow` | interval | 0.99 | Normal interval used as the matching region |
| `least_squares_with_l1_bounds` | bounds, maxiter | (0, 1), 100 | SLSQP settings |
| `MzBinMaskingSplitter` (via `NMFMaskWrapper`) | `mask_signal`, `balance_mask_signal` | True, True | Mask non-zero cells, plus an equal *fraction* of zero cells |
| `MzMLParser.to_ms2_h5` | `chunk_size` | 2500 | Scans buffered per HDF5 write |
| `Constants` | `isotope_mu`, `proton_mu` | 1.00336, 1.0073 | ¹³C–¹²C isotope spacing and proton mass |

---

## Using the Python API directly

This gives you control over each stage and lets you inspect intermediate results.

```python
# run_ffd.py
from pathlib import Path
import pandas as pd

from fragment_feature_detection.config import Config
from fragment_feature_detection.converter import MzMLParser
from fragment_feature_detection.containers.msrun import MSRun
from fragment_feature_detection.containers import (
    fit_nmf_matrix_msrun,
    add_ms1_features_msrun,
    fit_ms1_ms2_feature_matching_msrun,
    dump_features_to_df_msrun,
)
from fragment_feature_detection.decomposition import (
    tune_hyperparameters_optunasearchcv,
    tune_hyperparameters_randomizedsearchcv,
)
from fragment_feature_detection.utils import feature_df_to_ms2


def main():
    config = Config.from_ini("template.ini")   # or Config()
    config.lazy = False
    n_jobs = 8

    # 1. mzML -> HDF5 long
    MzMLParser.to_ms2_h5("sample.mzML", h5_fh="sample.h5", config=config)

    # 2. Build the run (discretize, pivot, tile, filter)
    msrun = MSRun.from_h5_long("sample.h5", config=config)

    # 3. Tune (optional)
    windows = msrun.get_tuning_windows(config=config)          # list of dense scan x bin arrays
    best = tune_hyperparameters_optunasearchcv(windows, config=config, n_jobs=n_jobs)
    config.nmf.update(best)                                     # alpha_W, alpha_H, l1_ratio

    # 4. NMF + Gaussian fits + collapse
    fit_nmf_matrix_msrun(msrun, n_jobs=n_jobs, config=config)

    # 5. MS1 matching (optional)
    ms1 = pd.read_csv(
        "sample.features.tsv", sep="\t",
        converters={"mono_hills_scan_lists": pd.eval, "mono_hills_intensity_list": pd.eval},
    )
    add_ms1_features_msrun(msrun, ms1, config=config, n_jobs=n_jobs)
    fit_ms1_ms2_feature_matching_msrun(msrun, n_jobs=n_jobs, config=config)
    # (if skipping MS1: config.feature_matching.match_ms1 = False)

    # 6. Export
    df = dump_features_to_df_msrun(msrun, config=config)
    df.to_parquet("sample.features.parq")
    feature_df_to_ms2(df, "sample.itw.ms2",
                      extractor_options={"output_type": "intensity_transform_weight",
                                         "wide_window": False})
    msrun.close()


if __name__ == "__main__":      # required on Windows/macOS (spawned joblib/multiprocessing workers)
    main()
```

### Inspecting one scan window

```python
gpf = msrun.get_gpf(0)                  # first isolation target (sorted by m/z)
sw = gpf.scan_windows[10]               # 11th time tile

sw.m                  # filtered, max-scaled matrix (scans x kept m/z bins)
sw.mz                 # m/z (bin right edge) of each column
sw.retention_time     # RT of each row
sw.w                  # elution profiles of kept components (scans x k)
sw.h                  # spectra of kept components (k x bins), scaled units
sw.component_fit_parameters   # (mu, sigma) arrays
sw.component_names
sw.reverse_transform_maxscale_scans()   # per-column max (multiply h by this for intensity units)
```

Quick plot of the decomposition (matplotlib is already a dependency):

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
ax[0].plot(sw.retention_time, sw.w); ax[0].set_xlabel("RT"); ax[0].set_title("Component elution (W)")
for k in range(sw.h.shape[0]):
    ax[1].vlines(sw.mz, 0, sw.h[k], alpha=0.6)
ax[1].set_xlabel("fragment m/z"); ax[1].set_title("Component spectra (H)")
plt.tight_layout(); plt.show()
```

### Reloading a saved run

```python
config = Config(); config.lazy = True
msrun = MSRun.from_h5("sample.h5", config=config)   # lazy: GPFRuns load on get_gpf()
```

---

## Performance: bottlenecks and what to adjust

### What was measured

**Run:** `process_mzml` on a synthetic narrow-window run, timed with each stage's start/finish log lines and `cProfile`.

**Data:** 6 isolation targets × 1,500 scans, i.e. 9,000 MS2 spectra and 300,000 peaks (a 36 MB mzML).

**Settings:**

- `n_jobs = 4`, on a Windows 11 laptop;
- tuning with `n_iter = 24` and `window_sampling_n = 16`;
- `lazy = True`.

A real run has 10–100× more targets, peaks and bins, so treat these numbers as *relative* costs.

| Stage | Time | Share | Notes |
|---|---|---|---|
| Convert mzML → HDF5 | 12–17 s | ~20% | XML parsing and per-peak Python lists |
| Build MSRun (discretize, pivot, tile, filter) | 10 s *(was 40 s)* | ~15% | The sparse percentile filter was the hotspot and has been fixed (see below) |
| Sample tuning windows | 2–5 s | ~4% | |
| **Tuning (24 trials × 3 splits × ~16 windows)** | **29–31 s** | **~45%** | Grows linearly with `n_iter` (default **500**, i.e. ~20× this) |
| NMF + Gaussian fits + collapse (~280 windows) | 5–7 s | ~9% | |
| Export to Parquet | 0.3–2.4 s | ~2% | Slower in lazy mode (reload from HDF5) |
| **Total** | **70 s** *(was 94 s)* | | |

**Speed fix applied while measuring:** the per-window percentile filter used a boolean-mask
assignment on a sparse matrix (`m[m < threshold] = 0`). SciPy handles that by inserting
entries and re-sorting the matrix, which took 30 of the 40 s spent building the run. It now
edits the stored values directly. The results are identical (same components) and the
stage is 4× faster.

### Where a real run will bottleneck

Most likely to least likely, with the settings you can change today. "Code fix" points to
the change in [SUGGESTIONS.md](SUGGESTIONS.md) that would remove the bottleneck.

1. **Tuning.** It makes `n_iter × n_splits × (number of tuning windows)` NMF fits. With the
   defaults that is 500 × 3 × ~50 ≈ 75,000 fits, more than the ~10,000 fits of the main NMF
   pass on a typical run.
   - **Biggest lever:** tune once per instrument method, then reuse the result. Copy
     `alpha_W`, `alpha_H` and `l1_ratio` from a previous `sample.out.ini` (or reuse that file
     directly) with `run_tune = False`.
   - `n_iter = 100–150` is usually enough for 3 parameters. `n_splits = 2`, or a smaller
     `window_sampling_n` (e.g. 30), cut the cost proportionally.
   - Code fixes: pruning after the first split ([D9](SUGGESTIONS.md#d9)), and batched GPU
     NMF ([E1](SUGGESTIONS.md#e1)/[E2](SUGGESTIONS.md#e2)).
2. **Conversion.** Pure Python, one list per peak, and the run time grows with the number
   of peaks. At the rate measured here (~20,000 peaks/s), a file with 100 million centroided
   peaks takes over an hour.
   - `process_mzml` always reconverts. When you rerun with different settings, use the
     Python API with `MSRun.from_h5_long` on the existing `.h5` (see
     [API](#using-the-python-api-directly)).
   - `filter_spectra = True` (with `filter_spectra_percentile`) drops the weakest peaks of
     each spectrum at conversion. That shrinks every later stage too, but can lose weak
     real fragments.
   - Code fix: vectorize the per-scan arrays ([D1](SUGGESTIONS.md#d1)), about 10–50× faster.
3. **Main NMF pass.** Cost is `windows × max_iter × n_components × bins`.
   - **Number of windows:** each scan sits in `scan_width / scan_overlap` windows (150/30 = 5).
     Keep `scan_overlap = 30`. It was chosen from real data, so the number of windows isn't
     something to trade away for speed.
   - **Bins (columns):** a wider `tolerance`, a narrower `mz_low`–`mz_high`, or `steps = 1`
     (halves the columns but loses the interleaved grid) all shrink every matrix.
   - **`n_components`:** if the export shows that windows rarely keep more than ~10
     components, lowering it from 20 speeds up each fit.
   - **`n_jobs`:** set it to the number of physical cores. joblib already limits BLAS
     threads inside workers. If you see heavy oversubscription, also set
     `OMP_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`).
   - Code fixes: warm starts from the previous tile ([D5](SUGGESTIONS.md#d5)), batching
     windows per isolation target ([D6](SUGGESTIONS.md#d6)), and GPU NMF
     ([E1](SUGGESTIONS.md#e1)).
4. **Lazy-mode I/O.** With `lazy = True`, every stage reloads each isolation target from
   HDF5 and writes all of its windows back. That's four full read/write passes, and the
   files are large (see [below](#disk-usage)).
   - Use `lazy = False` when the run fits in memory. The pivoted run is roughly 8 bytes ×
     (MS2 peaks × `steps`), plus all windows: a few GB for a typical run.
   - Code fix: process each isolation target through all stages in one pass
     ([D7](SUGGESTIONS.md#d7)).
5. **MS1 feature attachment and matching** (only with `--ms1_features_file`). Every tile
   filters the **whole** Biosaur2 table with a row-wise `DataFrame.apply`, and each
   regression uses SLSQP with numerical gradients.
   - Pre-filter the Biosaur2 TSV before the run, e.g. by charge (1–4 or 1–6), minimum
     `nIsotopes`, and minimum intensity. The work grows with the number of features.
   - Code fixes: [D8](SUGGESTIONS.md#d8) (interval index on RT and m/z) and
     [D3](SUGGESTIONS.md#d3) (L-BFGS-B with an exact gradient).
6. **Duplicate collapse and export.** These are smaller now. The collapse skips pairs whose
   elution peaks are far apart and integrates over ±6σ only. The export still scans the
   whole `m_long` table once per component ([D2](SUGGESTIONS.md#d2)); that becomes
   noticeable with very many components.

### Disk usage

On the synthetic test run (36 MB mzML, 6 isolation targets, MS1 matching with up to two
matches per component), with `lazy = True`:

| Settings | `.h5` size | HDF5 objects | Run time |
|---|---|---|---|
| Default layout | 195 MB | 19,500 | 42 s |
| `compact_storage = True` | **22 MB (9× smaller)** | 10,900 | 34 s |
| `compact_storage = True`, `compact_compression = 'lzf'` | 24 MB | 10,900 | 30 s |
| `compact_storage = True`, `gpfs_per_file = 2` | 2.9 MB main + 3 parts of ~7 MB | | 32 s |
| `lazy = False` (for comparison) | 2.9 MB (converted peaks only) | | |

Every setting gave identical results: the same Parquet, value for value, and the same
objects when the run is reloaded. Regression tests check this
(`test_storage_layouts_give_identical_results`).

Where the default layout's space goes, and what `compact_storage` does about it:

| Content (default layout) | Share | With `compact_storage` |
|---|---|---|
| Per-window copies of the isolation target's **full** m/z axis (`mz_unfilter`, `mz_bin_indices_unfilter`, `mz_masked`). Every window stores the same arrays again (~46 windows per target here, ~55 in a typical run) | **~69%** | Not stored. Rebuilt exactly from the target's axis on load |
| MS1 features: one group with 3–5 small datasets per feature, per window. Each HDF5 object costs ~1 KB of headers and indexes, so this is mostly overhead. On a real run with many Biosaur2 features it can reach millions of objects | grows with the number of MS1 features | One table per window (about a dozen datasets, however many features). Interpolated profiles are recomputed on load |
| The full bin grid, stored again for every isolation target | ~6% | Stored once per file |
| Everything else (sparse matrices, `m_long`, W, H, scaling factors), **uncompressed** | ~20% | Compressed (lossless) |

**On real data, expect a larger reduction than 9×.** The repeated axis grows with the
number of m/z bins an isolation target touches, which is far larger in real data than in
the synthetic test. The per-feature MS1 overhead grows with the number of Biosaur2
features. Both are exactly what compact storage removes.

**Splitting** (`gpfs_per_file`) doesn't change the total size. It keeps each file small, and
lets you move, back up or delete parts independently. Reloading a split run works exactly
like a single file.

Other ways to save space:

- Use `lazy = False` when memory allows. The `.h5` then holds only the converted peaks.
- Keep `scan_window_save_m_unfilter = False` (the default). Setting it to `True` also stores
  every window's unfiltered matrix.
- Set `explode_ms1_matches = False` to stop repeating each component's spectra on every
  MS1-match row of the Parquet (see [Parquet columns](#parquet-columns-stemfeaturesparq)).
- Delete the `.h5` (and `.gpf_runs/`) after the run if you don't need to reload it. The
  Parquet and `.out.ini` hold the results and settings.

---

## Troubleshooting

| Message / symptom | Likely cause → fix |
|---|---|
| `unexpected keyword argument 'alpha_w'` | You're running code from before the fixes → update |
| `MS1 features have not been matched to MS2 components` (Python API) | Set `config.feature_matching.match_ms1 = False`, or run MS1 attachment and matching first |
| `No candidate tuning windows after excluding ...` warning, then tuning fails | Too few scans per isolation target → lower `exclude_scan_window_edges`, or set `run_tune = False` |
| `Only N of M trials succeeded, fewer than half of n_iter` | Many NMF fits fail on the sampled windows → check the log for the trial errors; check the data and `n_components` |
| `GPF ... has N scans, fewer than the scan window width` warning | That isolation target is shorter than `scan_width` and is skipped → lower `scan_width` if those targets matter |
| Most components removed as "edge" components | σ too large relative to 150 scans, or `sigma_bounds` minimum too high → check the RT units and lower `sigma_bounds[0]` |
| Very few components per window | `alpha_H` too high → tune, or reduce `alpha_H` / `l1_ratio` |
| Many tiny, noisy components | `alpha_H` too low → tune, or increase |
| Search finds nothing in reweighted spectra | Fragment tolerance too tight for the bin-edge m/z offset → widen to ≥ 1.5× bin width |
| Empty `.ms2` with default (non-wide) mode | No MS1 matches passed thresholds → check RT units match between Biosaur2 and mzML; lower `matching_coefficient_threshold`; use `--wide_window` |
| A stage is much slower than expected | See [Performance](#performance-bottlenecks-and-what-to-adjust) |
