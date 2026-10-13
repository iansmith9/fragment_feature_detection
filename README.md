# Fragment Feature Detection (`ffd`)

`fragment_feature_detection` is a Python library and command-line tool that pulls clean,
per-analyte fragment spectra out of **narrow-window DIA** (data-independent acquisition)
mass spectrometry data. It uses **Non-negative Matrix Factorization (NMF)**. It was built
for **peptidomics**, where the search space is very large: no enzyme rules, endogenous
cleavages, and many PTMs. In that setting, chimeric DIA spectra are a big source of
missed identifications.

The idea: in every isolation window, fragment ions from the same precursor rise and fall
together over retention time. NMF factors each window's *scans × fragment-m/z* intensity
matrix into a small set of **components**. Each component has:

- an **elution profile** (how the component's intensity changes over time), and
- a **spectrum** (which fragment m/z values belong to it, and how much).

Next, each component is fitted with a Gaussian peak shape. Redundant components are
merged. Optionally, each component is linked to MS1 precursor features (from
[Biosaur2](https://github.com/abdrakhimov1/biosaur2)) by regressing MS1 elution shapes
onto the MS2 component elution. The output is a table of components plus `.ms2`
spectrum files that a search engine can read.

The components are useful beyond a first-pass search:

- **Clustering and deep dives.** Group unidentified component spectra by spectral
  similarity to find recurring unknowns.
- **Re-searching.** Search the cleaner, deconvolved spectra with wider or open-modification
  parameters.
- **Delta-mass / PTM discovery.** Compare unidentified components to confidently matched
  peptides using mass-shift-aware spectral similarity. This can suggest modifications
  (delta masses) and, from which fragment ions shift, possible site localization.

> **Documentation map**
>
> | File | What it covers |
> |---|---|
> | `README.md` (this file) | What the package is, installation, dependencies, and a quick start |
> | [`RUNNING.md`](RUNNING.md) | Step-by-step walkthrough of a full run, plus a reference for every CLI option and config key |
> | [`DETAILS.md`](DETAILS.md) | The theory (what happens at each step, why, and how NMF works in practice), plus a code walkthrough of every stage |
> | [`SUGGESTIONS.md`](SUGGESTIONS.md) | Code review: bugs (with fix status), speed-ups (including GPU), and feature ideas |

---

## What the pipeline does (one-paragraph version)

1. **Convert.** `mzML` → HDF5, storing MS2 peaks in long format, grouped by isolation-window
   target m/z.
2. **Discretize.** Fragment m/z values are binned onto two interleaved ppm grids.
3. **Pivot.** Each isolation window becomes a sparse *scan × m/z-bin* matrix.
4. **Tile.** Each matrix is cut into overlapping **scan windows** (default 150 scans, a new
   window every 30 scans, plus a final window ending on the last scan).
5. **Filter / denoise.** Remove constant background ions and low-intensity noise, scale each
   m/z channel to its maximum, remove one-scan spikes with morphological filtering, and zero
   the window edges.
6. **(Optional) tune.** Choose the NMF regularization (`alpha_W`, `alpha_H`, `l1_ratio`) by
   masked cross-validation with Optuna (multi-objective) or a randomized search.
7. **NMF.** Factor every scan window into up to `n_components` components.
8. **Peak fit.** Fit a Gaussian to each component's elution profile. Drop failed fits and
   components that sit on a window edge.
9. **Collapse.** In overlapping scan windows, merge components that elute at the same time
   and have similar spectra. The one with the better Gaussian fit is kept.
10. **(Optional) MS1 matching.** Interpolate Biosaur2 MS1 features onto the MS2 time axis
    and fit a non-negative, L1-regularized regression to decide which precursor(s) explain
    each component.
11. **Export.** Write a Parquet table of components and matches. Optionally write `.ms2`
    files with one of three spectrum flavors (raw apex scan, NMF weights, or NMF weights
    rescaled to observed intensity).

---

## Installation

The package is not on PyPI. Install it from the repository:

```bash
git clone <this repo>
cd fragment_feature_detection

# Recommended: an isolated environment with Python 3.10
conda create -n ffd python=3.10 -y
conda activate ffd

pip install -e .            # runtime install; adds the `ffd` command
pip install -e ".[dev]"     # + black, flake8, mypy, isort, pre-commit
```

Installing gives you the `ffd` command. The package can also be imported from a source
checkout without installing (for example when running the tests). The `.ms2` header then
reports the version as `dev`.

### Running the tests

```bash
pip install pytest
python -m pytest tests          # from the repository root
```

`tests/test_bug_fixes.py` holds regression tests for the bugs fixed after commit
`558ba09`. They build small synthetic runs, and include a short tuning run with two worker
processes. They take about 30 seconds.

### Python version

`pyproject.toml` declares `requires-python >= 3.8`, but the pinned versions narrow the
range that will actually install:

| Pin | Practical Python range |
|---|---|
| `numpy==1.24.4` | 3.8 – 3.11 |
| `scipy==1.10.1` | 3.8 – 3.11 |
| `pyopenms==2.7.0` | about 3.7 – 3.10 (wheels) |

**Use Python 3.10.** Python 3.12+ will not work with these pins. (The tests have also been
run on Python 3.9.) The code no longer uses the APIs removed in NumPy 2.x (`np.NINF`,
`np.Inf`, sparse `.A`), but newer versions haven't been tested, so keep the pins.

### Operating system

Linux, macOS and Windows all work. Optuna tuning keeps its trial history in a temporary
journal file in the system temp folder, which works with several worker processes on all
three. (Earlier versions used `/tmp/` and failed on Windows.) The tests, including a
two-worker tuning run, pass on Windows 11.

When you call the Python API with `n_jobs > 1` on Windows or macOS, put your script's
entry point under `if __name__ == "__main__":`. Worker processes are started fresh and
re-import your script.

---

## Dependencies

### Runtime (pinned in `pyproject.toml`)

| Package | Version | What it is used for here |
|---|---|---|
| `numpy` | 1.24.4 | All array math |
| `pandas` | 2.0.3 | MS1 feature tables, output DataFrames, Parquet I/O |
| `h5py` | 3.11.0 | Intermediate HDF5 storage (converted mzML, lazy on-disk MSRun) |
| `scipy` | 1.10.1 | Sparse matrices, morphological filters (`ndimage`), Gaussian fits (`curve_fit`), SLSQP regression, cosine distance |
| `scikit-learn` | 1.3.2 | `NMF` (coordinate-descent solver), `BaseSearchCV`, `ParameterSampler`, joblib `Parallel` |
| `optuna` | 4.2.0 | Multi-objective hyperparameter search |
| `pyteomics` | 4.7.5 | mzML parsing (`pyteomics.mzml.MzML`) |
| `lxml` | 5.3.0 | XML backend for pyteomics |
| `pyarrow` | 17.0.0 | Parquet engine for `features.parq` |
| `click` | 8.1.8 | CLI (`ffd`) |
| `tqdm` | 4.67.1 | Progress bars |
| `pyopenms` | 2.7.0 | *Listed but not imported anywhere in the current code* |
| `mmh3` | 5.0.1 | *Listed but not imported anywhere in the current code* |
| `matplotlib` | 3.7.5 | *Listed but not imported anywhere in the current code* |

### Development (`pip install -e ".[dev]"`)

`black==24.8.0`, `flake8==7.0.0` (+ `Flake8-pyproject`), `mypy==1.9.0`, `isort==5.13.2`,
`pre-commit==3.6.2`. The pre-commit hooks (`.pre-commit-config.yaml`) run
trailing-whitespace, end-of-file, YAML, and large-file checks, plus black, isort and
flake8 (line length 88).

### External tools you will likely need

| Tool | Why |
|---|---|
| **ProteoWizard `msconvert`** or **ThermoRawFileParser** | Convert vendor raw files to **centroided** mzML |
| **Biosaur2** (`pip install biosaur2`) | MS1 feature detection. Its feature TSV is the `--ms1_features_file` input |
| A search engine that reads `.ms2` (for example **Comet**), or a converter to MGF | To search the exported component spectra |

---

## Input data requirements

This method depends on the acquisition design. A generic DIA file with wide windows will
not give good results. What the code assumes:

- **Narrow-window DIA** (for example 1–4 m/z isolation windows), usually spread across several
  **gas-phase-fractionation (GPF)** injections that each cover part of the precursor range.
  Every distinct *isolation-window target m/z* is treated as its own time series. The code
  calls that series a "GPF" (see [DETAILS.md](DETAILS.md#terminology)).
- **Centroided MS2 spectra.** Every peak becomes one row in the long table. Profile-mode
  data will be enormous and give poor results.
- Enough cycles per isolation window to tile scan windows: at least a few hundred MS2 scans
  per window at the default 150-scan width. The tuning sampler also skips the first and last
  300 scans (10 × 30) of each window.
- Isolation-window metadata in the mzML (`isolation window target m/z`, lower/upper offsets,
  `selected ion m/z`). msconvert writes these.
- For MS1 matching: MS1 scans in the same file, processed with **Biosaur2**. The feature
  table must include `massCalib, rtApex, intensityApex, intensitySum, charge, nIsotopes,
  nScans, mz, rtStart, rtEnd, scanApex, mono_hills_scan_lists,
  mono_hills_intensity_list`.

---

## Quick start

```bash
# 1. Write a template config and edit it (see RUNNING.md for the two edits you must make)
ffd generate_template_ini --output_dir .

# 2. Detect MS1 features (optional, but needed for precursor-specific .ms2 output)
biosaur2 sample.mzML -o sample.features.tsv

# 3. Run the whole pipeline
ffd process_mzml sample.mzML \
    --ms1_features_file sample.features.tsv \
    --config_ini_file template.ini \
    --output_dir ./ffd_out \
    --n_jobs 8

# 4. Export spectra for searching
ffd dump_df_to_ms2 sample.features.parq --output_type intensity_transform_weight
ffd dump_df_to_ms2 sample.features.parq --output_type raw_weight --wide_window
```

Outputs (details in [RUNNING.md](RUNNING.md#outputs)):

| File | Location | Content |
|---|---|---|
| `sample.h5` | `--output_dir` | Converted MS2 data, plus the on-disk MSRun when `lazy = True` |
| `sample.gpf_runs/part_*.h5` | `--output_dir` | Only with `[h5] gpfs_per_file > 0`: the on-disk MSRun split into parts |
| `sample.features.parq` | `--output_dir` | One row per component (per MS1 match when matching) |
| `sample.out.ini` | `--output_dir` | Final config, including tuned NMF parameters. Reusable as-is |
| `YYYYmmdd-HHMMSS__process_mzml__sample.log` | next to the mzML | Run log |
| `…tuning_<timestamp>.log` / `.pkl` | next to the log | Tuning log and pickled search object |
| `sample.features.<type>[.wide].ms2` | `dump_df_to_ms2 --output_dir` (default: next to the Parquet) | Spectra for searching |

> **Before your first run,** skim [RUNNING.md § Known pitfalls](RUNNING.md#known-pitfalls-read-before-your-first-run).
> The crashes listed there in earlier versions (INI key names, runs without MS1 features,
> tuning on Windows) are fixed. What remains are output-format details to know when
> searching, plus a note that runs with `lazy = True` write a large `.h5` by default. Set
> `[h5] compact_storage = True` for a ~9× smaller file with identical results, and
> `gpfs_per_file` to split it into parts (see
> [RUNNING.md § Disk usage](RUNNING.md#disk-usage)).

---

## Package layout

```
fragment_feature_detection/
├── cli.py             # `ffd` click commands: generate_template_ini, convert_mzml_h5long,
│                      #   process_mzml, dump_df_to_ms2
├── config.py          # Config (defaults + INI round-trip), AttributeDict, Constants
├── config.ini         # Example config (differs from Config() defaults; see RUNNING.md)
├── converter.py       # MzMLParser: mzML -> HDF5 long table grouped by isolation target
├── discretization.py  # MzDiscretize: ppm/Da bins with interleaved grids
├── decomposition.py   # NMF wrapper, masked-CV splitters, Optuna/randomized search, tuning
├── fitpeaks.py        # Gaussian elution fit, Gaussian overlap, bounded L1 least squares
├── utils.py           # Pivoting to sparse matrices, NMF summary metrics, .ms2 writer
├── simulation.py      # Toy generators for synthetic species/fragments (not wired into pipeline)
└── containers/
    ├── __init__.py    # Pipeline drivers: fit / add MS1 / match / dump at window, GPF, run level
    ├── msrun.py       # MSRun: collection of GPFRuns (in memory or lazily in HDF5)
    ├── gpfrun.py      # GPFRun: one isolation window over time; tiles ScanWindows; collapses dups
    ├── scanwindow.py  # ScanWindow: one time tile; filtering, NMF results, peak fits, MS1 links
    └── ms1feature.py  # MS1Feature: Biosaur2 feature + interpolation onto MS2 time axis
```

Object hierarchy:

```
MSRun ──► GPFRun (one per isolation-window target m/z) ──► ScanWindow (150-scan tiles, stride 30)
                                                                 │
                                                                 ├── m        (scans × m/z bins, filtered)
                                                                 ├── W, H     (NMF elution × spectra)
                                                                 ├── μ, σ     (Gaussian fits per component)
                                                                 └── MS1Feature list + match coefficients
```

---

## Status and caveats

- Version `0.0.1`. Research code.
- Regression tests cover the bugs fixed after commit `558ba09` (see
  [Running the tests](#running-the-tests)). There is no CI yet.
- The pipeline has been run end to end on **synthetic** narrow-window data. It has not yet
  been checked on real DIA data in this repository's current state.
- See [SUGGESTIONS.md](SUGGESTIONS.md) for the fix status of every reviewed bug, and for
  remaining improvements.

## License

MIT. See [`LICENSE`](LICENSE).
