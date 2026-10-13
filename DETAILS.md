# How `fragment_feature_detection` works: theory and practice

This document explains **what each step of the pipeline does, why it is done that way,
and the math behind it**, with most of the space on Non-negative Matrix Factorization
(NMF). It is written for mass spectrometrists who are comfortable with DIA but may not
have used matrix factorization, and for data scientists who know NMF but not
proteomics. Skip whichever background sections you don't need.

Every statement about behavior refers to the code at commit `558ba09` **plus the bug fixes
of 2026-10-13** (listed in [SUGGESTIONS.md § Fix status](SUGGESTIONS.md#fix-status)). File
and function names are given so you can follow along in the source. Places where the code
seems to differ from its intent are flagged briefly here and covered in detail in
[SUGGESTIONS.md](SUGGESTIONS.md).

The document has two parts:

- **Part I (Sections 1–15)** explains the method: what each step does and why.
- **Part II (Sections 16–23)** is a code walkthrough. It follows `ffd process_mzml` call by
  call: which functions run, the shapes and columns of every array, the object state each
  step sets, and the exact numeric operations.

---

## Contents

0. [Terminology](#terminology)
1. [The problem: chimeric DIA spectra and huge search spaces](#1-the-problem-chimeric-dia-spectra-and-huge-search-spaces)
2. [The core idea in one picture](#2-the-core-idea-in-one-picture)
3. [Step 1: Reading the data](#3-step-1-reading-the-data)
4. [Step 2: Putting fragment m/z on a grid (discretization)](#4-step-2-putting-fragment-mz-on-a-grid-discretization)
5. [Step 3: Building matrices and tiling time (scan windows)](#5-step-3-building-matrices-and-tiling-time-scan-windows)
6. [Step 4: Cleaning each window (filtering and denoising)](#6-step-4-cleaning-each-window-filtering-and-denoising)
7. [Step 5: NMF, the heart of the method](#7-step-5-nmf-the-heart-of-the-method)
8. [Step 6: Choosing NMF settings automatically (tuning)](#8-step-6-choosing-nmf-settings-automatically-tuning)
9. [Step 7: Describing each component's elution (Gaussian fits)](#9-step-7-describing-each-components-elution-gaussian-fits)
10. [Step 8: Removing duplicates from overlapping windows](#10-step-8-removing-duplicates-from-overlapping-windows)
11. [Step 9: Linking components to precursors (MS1 matching)](#11-step-9-linking-components-to-precursors-ms1-matching)
12. [Using the components downstream](#12-using-the-components-downstream)
13. [Limitations and failure modes](#13-limitations-and-failure-modes)
14. [Glossary](#14-glossary)
15. [Further reading](#15-further-reading)

**Part II: Code walkthrough**

16. [Map of the code](#16-map-of-the-code)
17. [Data layouts and object state](#17-data-layouts-and-object-state)
18. [Walkthrough of `ffd process_mzml`, step by step](#18-walkthrough-of-ffd-process_mzml-step-by-step)
19. [Lazy vs in-memory: when data is read and written](#19-lazy-vs-in-memory-when-data-is-read-and-written)
20. [Parallelism: what runs where, and what gets pickled](#20-parallelism-what-runs-where-and-what-gets-pickled)
21. [Config keys and the code that reads them](#21-config-keys-and-the-code-that-reads-them)
22. [The Python API in the same order](#22-the-python-api-in-the-same-order)
23. [Code behaviors worth knowing](#23-code-behaviors-worth-knowing)

---

## Terminology

Some words in the code mean something slightly different from everyday lab usage.

| Term in code | What it means here |
|---|---|
| **GPF** / `GPFRun` | In the lab, *gas-phase fractionation* means several injections of one sample, each covering part of the precursor m/z range with narrow DIA windows. **In the code, a "GPF" is one isolation window**: all MS2 scans that share one `isolation window target m/z`, in RT order. One mzML with 100 narrow windows gives 100 `GPFRun`s. `ms2_gpf_center` in the output is that window's target m/z. |
| **Scan window** / `ScanWindow` | A tile in *time* (not m/z): 150 consecutive MS2 scans of one isolation window. |
| **Component** | One factor from NMF: an elution profile plus a fragment spectrum. Ideally, one component = one precursor (peptide). |
| **W** | The *elution* matrix (scans × components). Column k is how component k rises and falls over time. |
| **H** | The *spectrum* matrix (components × m/z bins). Row k is component k's fragment spectrum. |
| **Bin** | A small m/z interval. Every fragment peak is assigned to a bin. |
| `scan_overlap` | Despite the name, the **step** (stride) between the starts of consecutive scan windows. |

> The docstrings in `decomposition.py` sometimes call W the "components" and H the
> "activations". That is the usual machine-learning wording when rows are samples. In this
> pipeline rows are scans, so **W is elution and H is spectra**. `ScanWindow.w`
> (scans × k) and `ScanWindow.h` (k × bins) are used that way throughout.

---

## 1. The problem: chimeric DIA spectra and huge search spaces

### DIA in one paragraph

In **data-dependent acquisition (DDA)**, the instrument picks one precursor at a time and
fragments it, so each MS2 spectrum is (ideally) from one peptide. In
**data-independent acquisition (DIA)**, the instrument steps through fixed isolation windows
(say 400–402, 402–404, …) and fragments *everything* in each window, every cycle. DIA misses
fewer peptides and gives a regular time series for every window. The cost is that **each MS2
spectrum is a mixture** of fragments from every precursor co-isolated in that window at that
moment. These are "chimeric" spectra.

### Why this hurts peptidomics in particular

Tryptic proteomics has a small search space: peptides must end in K/R, sizes are
predictable, and few modifications are considered. Library-based DIA tools (Spectronaut,
DIA-NN, and others) work well because they only need to *confirm* expected peptides.

**Peptidomics** (endogenous peptides: neuropeptides, hormones, degradation products, MHC
ligands, and so on) is different:

- **No enzyme rule.** Any sub-sequence of any protein is a candidate. The search space is
  10–100× larger than tryptic.
- **Many PTMs**, some characteristic of bioactive peptides (C-terminal amidation, pyro-Glu,
  N-terminal acetylation, sulfation, disulfides, oxidation, phosphorylation).
- **Lower abundance, and many charge states.**

With a big search space, *any* extra noise in a spectrum causes random matches and pushes
true matches below the FDR threshold. A chimeric spectrum is the worst case: fragments from
peptide B look like noise to a search for peptide A, and vice versa. Both can be missed.

### The strategy

If we can **separate a chimeric DIA spectrum series into per-precursor pseudo-spectra**, we
can search them like DDA spectra, with DDA-style open or large searches. The information that
makes this possible is **co-elution**: all fragments of one peptide share its chromatographic
peak shape. They rise and fall together, and differently from fragments of a peptide that
elutes 10 seconds later.

**Narrow windows** help twice:
1. Fewer precursors per window, so each mixture is simpler.
2. The isolation window itself limits the precursor mass. With MS1 features, the exact
   precursor can often be identified (Section 11).

---

## 2. The core idea in one picture

Take one isolation window. Stack its MS2 spectra over time into a matrix: **rows = scans
(time), columns = fragment m/z bins, values = intensity**.

```
                 fragment m/z bins  →
             b1   b2   b3   b4   b5   b6
  scan 38  [ 0.1  0.0  0.2  0.0  0.1  0.0 ]
  scan 39  [ 0.5  0.0  0.9  0.1  0.4  0.1 ]
  scan 40  [ 1.0  0.1  1.0  0.3  0.9  0.2 ]   ← peptide A near apex
  scan 41  [ 0.6  0.4  0.7  0.8  0.6  0.6 ]
  scan 42  [ 0.2  0.9  0.3  1.0  0.3  0.9 ]   ← peptide B near apex
  scan 43  [ 0.0  0.6  0.1  0.6  0.1  0.6 ]
  scan 44  [ 0.0  0.2  0.0  0.2  0.0  0.2 ]
```

Bins b1, b3 and b5 go up and down together (peptide A). b2, b4 and b6 go together
(peptide B). NMF finds this automatically by writing the matrix as a product of two thinner
matrices:

```
      X (scans × bins)     ≈        W (scans × 2)        ×        H (2 × bins)

                                   A     B                     b1  b2  b3  b4  b5  b6
                        scan38  [ 0.1   0.0 ]            A  [ 1.0 0.0 1.0 0.0 0.9 0.0 ]
                        scan39  [ 0.5   0.1 ]            B  [ 0.0 1.0 0.0 1.0 0.0 0.9 ]
                        scan40  [ 1.0   0.2 ]
                        scan41  [ 0.6   0.7 ]
                        scan42  [ 0.2   1.0 ]
                        scan43  [ 0.0   0.6 ]
                        scan44  [ 0.0   0.2 ]
                        (elution profiles)             (pure spectra)
```

- **Column A of W** is peptide A's chromatographic peak.
- **Row A of H** is peptide A's fragment spectrum, *with B's fragments removed*.

The rest of the pipeline prepares the matrices so this works well (Sections 3–6), picks
the settings (Section 8), checks and cleans the results (Sections 9–10), and ties each
component to a precursor (Section 11).

---

## 3. Step 1: Reading the data

*Code: `converter.py` → `MzMLParser.to_ms2_h5`*

The mzML is read with pyteomics, **keeping MS2 scans only**. For each scan the parser
reads:

- `isolation window target m/z`, the key used to group scans into "GPFs" (isolation windows);
- the scan's index (+1) as `ScanNum`, and `scan start time` as RT;
- the isolation bounds (`selected ion m/z` ± offsets);
- the centroided peak list (m/z and intensity). Non-finite values are removed.

Each peak becomes a row `[target, scan, rt, mz, intensity, low, high]`. Rows are written to
HDF5 in chunks (every 2,500 scans), **one dataset per isolation target**. With
`filter_spectra = True`, peaks below a per-scan intensity percentile are dropped here.

**Why a long table in HDF5?** The mzML is XML, slow to parse and awkward to slice. The
long table can be read **one isolation window at a time**, which is exactly how the rest of
the pipeline works. It also lets the lazy mode keep memory low.

**Why centroided data?** Each row is treated as one fragment ion. Profile data would give
dozens of rows per ion, blur the bins, and make the files enormous.

---

## 4. Step 2: Putting fragment m/z on a grid (discretization)

*Code: `discretization.py` → `MzDiscretize`*

### Why bin at all?

A matrix needs fixed columns. The *same* fragment ion is measured at slightly different m/z
in each scan (e.g. 512.2701, 512.2694, 512.2708), so we need a rule that puts all of those
into the same column. That rule is binning.

### ppm bins

Mass error on Orbitrap/TOF instruments is roughly *proportional* to m/z, so the bins are
too. With `tolerance = 2e-05` (20 ppm), a bin near m/z 500 is about 0.010 wide and a bin
near m/z 1500 is about 0.030 wide. The grid is built from `mz_high` downward, each right edge
being `(1 − tolerance)` times the previous one:

$$ e_{j+1} = e_j\,(1 - \text{tol}) $$

The number of bins between 200 and 2000 is about $\ln(2000/200)/\text{tol}$, roughly
115,000 at 20 ppm or 38,000 at 60 ppm. Most are empty in any given window, and they are
stored sparsely.

### Two interleaved grids (`steps = 2`)

A fixed grid has one weakness: a fragment that falls right *on* a bin edge wobbles
between two bins from scan to scan. Its trace gets split into two noisy half-traces, which
is bad for co-elution analysis. The fix is a **second grid shifted by half a bin**. Every
peak is placed in *both* grids, so the matrix has one column block per grid. Whichever grid
has the fragment near a bin center gives a clean trace.

```
grid 0:  |-----|-----|-----|-----|
grid 1:     |-----|-----|-----|-----|
peak:          ^   (on an edge in grid 0, centered in grid 1)
```

The cost: **every fragment appears twice** in the matrix and in the exported spectra
(about half a bin apart). NMF handles this fine, because the two columns are highly
correlated and simply both load on the same component. Keep it in mind when reading
exported spectra.

### Which m/z is reported?

Bins are identified by their **right edge**, and the exported spectra use that edge. So
the reported m/z can be up to one bin width (the `tolerance`) *above* the true value. This
affects how you set fragment tolerance when searching (see SUGGESTIONS for a fix).

Peaks are only kept if they fall strictly inside a grid's range (between `mz_low` and
`mz_high`, and inside that grid's first and last bin). Because grid 1 is shifted, a peak
at the very top or bottom of the range can be in one grid but not the other.

---

## 5. Step 3: Building matrices and tiling time (scan windows)

*Code: `utils.pivot_unique_binned_mz_sparse`, `GPFRun.from_discretized_long`,
`GPFRun.build_scan_windows`*

For each isolation window, the binned long table is **pivoted** into a sparse matrix:

- rows = its MS2 scans in time order (one per DIA cycle);
- columns = bins that ever had a peak;
- values = summed intensity (if two peaks in one scan fall in the same bin, they add).

A copy of the original peaks (`m_long`: scan, rt, mz, intensity) is kept so the *raw* apex
spectrum can be exported later.

### Why cut the run into short time tiles?

A 60-minute run with a 2-second cycle has about 1,800 scans per isolation window.
Factoring the whole thing at once would be a bad idea:

1. **Locality.** A peptide elutes for maybe 10–60 seconds (5–30 scans). Within a short tile,
   only a handful of peptides are present, so a *small* number of components (low rank)
   describes the tile well. Over the whole run, hundreds of peptides would need hundreds of
   components, and NMF gets harder and less stable as the rank grows.
2. **Background changes over the gradient.** Chemical noise and solvent ions drift. Local
   tiles adapt to that.
3. **Parallelism and memory.** Thousands of small, independent problems are easy to spread
   across CPU cores (or a GPU).

Tiles are `scan_width = 150` scans long, with a new tile starting every `scan_overlap = 30`
scans. **Consecutive tiles share 120 scans**, and each scan belongs to up to 5 tiles:

```
scans:   0        30        60        90       120       150       180 ...
tile 0:  |==================================================|
tile 1:           |==================================================|
tile 2:                     |==================================================|
```

Why so much overlap? Each tile zeroes its first and last 5 scans, and components near a
tile's edge are thrown away (Section 9), because a peak cut off by the tile edge can't be
fitted reliably. Heavy overlap means **every peptide is fully inside at least one tile**.
The price is that the same peptide is usually found in several tiles. Section 10 removes
those duplicates.

> Note: tiles start at `0, 30, 60, …`, and one extra tile is added at `n − 150` when the
> stride doesn't land there, so the last scans are always covered. An isolation window with
> fewer than 150 scans gets no tiles at all (a warning is logged).

---

## 6. Step 4: Cleaning each window (filtering and denoising)

*Code: `ScanWindow.filter_scans` and the methods it calls*

Before NMF, every tile goes through a fixed sequence of transformations. Each has a
specific purpose.

| Order | Operation | What it does | Why |
|---|---|---|---|
| 1 | `filter_zero_mz` | Drop columns (bins) with no signal in this tile | Smaller matrix, faster NMF |
| 2 | `filter_clip_constant(p=0.3)` | Normalize each column to its max, then drop columns whose *mean* is > 0.3 | A real fragment is a **peak**: high for a few scans, low elsewhere, so its mean-to-max ratio is small. A background ion (solvent cluster, polysiloxane, persistent contaminant) is **flat**, with a mean close to its max. Such columns carry no co-elution information and would soak up components |
| 3 | `filter_percentile_scans` (10th pct) | Zero every value below the 10th percentile of non-zero intensities in the tile | Removes the lowest-level noise |
| 4 | *(optional, off)* `transform_log_scans` | `log(1+x)` | Would compress the dynamic range. Off by default |
| 5 | `transform_maxscale_scans` | Divide each column by its maximum in the tile, so every bin peaks at 1.0 | **Gives every fragment equal weight.** See below |
| 6 | `filter_denoise_scans` | Grey-scale morphological closing then erosion along the time axis, window 2 scans | Removes one-scan spikes and fills one-scan dropouts |
| 7 | `filter_edge_scans` | Zero the first and last 5 scans | Peaks touching tile edges are unreliable. Overlapping tiles cover them |
| 8 | downcast | Round-trip through float16, store as float32 | Memory, and identical results whether or not data went through HDF5 |

### Why max-scaling matters so much

NMF minimizes squared error, so without scaling **the brightest fragments dominate**.
A 10⁷-intensity y-ion would decide the fit, and a 10⁴-intensity diagnostic fragment would
be ignored. After max-scaling, every bin's trace has peak height 1. NMF then asks *"which
traces have the same **shape**?"* rather than *"how do I reproduce the big peaks?"*. That is
exactly the co-elution question.

The trade-off is that tiny, noisy bins are boosted to the same weight as strong ones. That
is why the percentile filter (step 3) and morphological denoising (step 6) come before and
after it. The scale factors are saved (`_m_max`), and multiplying a component's spectrum by
them turns it back into approximate intensities. That is the `intensity_transform_weight`
export.

### What morphological denoising does

Think of each bin's time trace as a skyline.

- **Closing** (dilate, then erode) fills narrow *valleys*. A single scan where a fragment
  briefly dropped out (common with low ion counts) gets filled in from its neighbors.
- **Erosion** with a 2-scan window then removes narrow *peaks*. A value that is high in only
  one scan, with its neighbors near zero, is replaced by the smaller neighbor, so it
  disappears.

A real chromatographic peak spans several scans and survives both, slightly trimmed. A
single-scan electronic or chemical spike does not survive. Because this only operates along
**time**, it never mixes different m/z bins.

```
before:  0  0  6  0  0  2  4  0  5  4  2  0      (spike at scan 2, dropout at scan 7)
after:   0  0  0  0  0  2  4  4  4  4  2  0      (illustrative)
```

---

## 7. Step 5: NMF, the heart of the method

*Code: `decomposition.fit_nmf_matrix_custom_init` (wraps `sklearn.decomposition.NMF`),
called by `containers.fit_nmf_matrix_scanwindow`*

### 7.1 Matrix factorization in plain words

A matrix $X$ with $n$ rows (scans) and $m$ columns (bins) holds $n \times m$ numbers. If the
data come from only $k$ underlying "sources" (peptides), each with a fixed spectrum and a
changing abundance over time, then

$$ X_{t,j} \;\approx\; \sum_{c=1}^{k} W_{t,c}\,H_{c,j} $$

In words: *the intensity of bin j at scan t is the sum over peptides c of "how much of
peptide c is eluting at t" times "how much of peptide c's signal lands in bin j".* This is
the physics of a chimeric spectrum (intensities add), written as a matrix product
$X \approx WH$.

Here $W$ is $n \times k$ and $H$ is $k \times m$. With $n = 150$, $m = 2{,}000$ and $k = 8$,
we describe 300,000 numbers with about 17,000. The factorization is forced to find the
*structure*, and random noise mostly can't be expressed with so few numbers.

### 7.2 Why "non-negative"?

Intensities and abundances can't be negative. NMF requires $W \ge 0$ and $H \ge 0$ (every
entry). This does more than keep things physical:

- **Parts-based results.** With only non-negative numbers, components can only *add*. They
  can't cancel each other. So each component has to look like a real piece of the data (a
  spectrum), not an abstract pattern with positive and negative lobes. PCA/SVD finds the
  best low-rank fit, but its components mix signs. A PCA "component" might be "peptide A
  minus half of peptide B", which is not a spectrum you could search.
- **Interpretability.** Row c of $H$ can be read directly as a spectrum, and column c of
  $W$ as a chromatogram.

This parts-based property is why Lee & Seung's 1999 paper made NMF popular. In their
example it learned *eyes, noses and mouths* from faces, where PCA learned ghostly
"eigenfaces".

### 7.3 What NMF is trying to minimize

scikit-learn's NMF (used here with the default Frobenius loss) solves

$$
\min_{W \ge 0,\; H \ge 0}\;
\underbrace{\tfrac12 \lVert X - WH \rVert_F^2}_{\text{fit the data}}
\;+\; \underbrace{\alpha_W\,\rho\, m\, \lVert W\rVert_1 + \tfrac12\alpha_W (1-\rho)\, m\, \lVert W\rVert_F^2}_{\text{penalty on elution profiles}}
\;+\; \underbrace{\alpha_H\,\rho\, n\, \lVert H\rVert_1 + \tfrac12\alpha_H (1-\rho)\, n\, \lVert H\rVert_F^2}_{\text{penalty on spectra}}
$$

where

- $\lVert X - WH\rVert_F^2$ is the sum of squared differences between data and model;
- $\lVert\cdot\rVert_1$ is the sum of absolute values (for non-negative matrices, just the
  sum of entries);
- $\rho$ is `l1_ratio`, which sets the L1 vs L2 mix;
- $\alpha_W$, $\alpha_H$ are `alpha_W`, `alpha_H`;
- $m$ = number of bins and $n$ = number of scans. **scikit-learn multiplies the W penalty by
  the number of columns and the H penalty by the number of rows**, so the penalties stay
  comparable to the data term whatever the matrix shape.

**Squared error assumes roughly Gaussian noise.** Ion counts are closer to Poisson (noise
grows with intensity). After max-scaling, the squared-error assumption is reasonable for
this purpose. scikit-learn also supports a Kullback–Leibler loss (`beta_loss =
'kullback-leibler'` with `solver = 'mu'`), which matches Poisson statistics. You can pass it
through `[nmf]` (see RUNNING), but it has not been tested here.

### 7.4 What the regularization does, and why this pipeline needs it

Without penalties, NMF with `n_components = 20` would use all 20 components. It would split
peptides into pieces and fit noise with the leftovers. The penalties fix that.

**L1 (sparsity), the `ρ` part.** An L1 penalty pushes small values to *exactly* zero. It's
cheaper for the model to drop a weak bin from a spectrum entirely than to keep it at a small
value.

- **On H (spectra)**, this gives *clean spectra*: each component keeps only the bins it
  really needs, and noise bins get dropped.
- **On whole components**: if the data don't need a component, L1 shrinks its entire row of
  H and column of W to zero. That is how the pipeline **picks the number of components
  automatically**. You ask for up to 20, and the penalty switches off the unneeded ones. The
  code then keeps only components with non-zero W (`ScanWindow._non_zero_components`).
- **On W (elution)**, this favors peaks that are exactly zero outside a short elution
  window, as real chromatographic peaks are.

**L2 (shrinkage), the `1−ρ` part.** This penalizes large values smoothly without forcing
zeros. It stabilizes the problem when two components are nearly identical. Mixing it with
L1 (an "elastic net") avoids L1's tendency to pick one of two correlated bins at random.
That matters here, because the duplicate interleaved-grid columns (Section 4) are exactly
such a pair.

**Why the defaults are so lopsided** (`alpha_W = 1e-5`, `alpha_H = 0.0375`): because of
scikit-learn's shape scaling and the matrix shape (≈150 rows × thousands of columns), the
*effective* penalty on H is several hundred times larger than on W. The design choice is
that **sparsity is enforced on the spectra**, where noise bins are the main problem, while
elution profiles are left free to follow the data. The tuning step (Section 8) searches
these values per dataset.

### 7.5 How the solution is found

There is no formula for the best $W, H$. The problem is *non-convex* (it has many local
minima), so it is solved iteratively.

1. **Start** from an initial guess.
2. **Hold H fixed and improve W**, then **hold W fixed and improve H**. Each half-step is
   an easy problem: a non-negative least-squares problem, which is convex.
3. Repeat until the improvement is tiny, or `max_iter` (500) is reached.

The default solver `'cd'` (**coordinate descent**, similar to HALS) updates one column of W
or one row of H at a time with a closed-form, non-negativity-clipped update. It converges
quickly and handles L1 penalties well, since the soft-threshold that creates exact zeros is
part of each update. `'mu'` (**multiplicative updates**, the original Lee & Seung algorithm)
multiplies each entry by a ratio that keeps it non-negative. It is simpler and needed for
non-Frobenius losses, but converges more slowly, and entries that reach zero stay zero.

**Initialization matters for a non-convex problem.** The code uses **NNDSVD** (for `'cd'`)
or **NNDSVDa** (for `'mu'`). These start from a singular value decomposition (the
"best-fitting" low-rank structure, like PCA) and keep the positive parts. The start is
*deterministic*, so results are reproducible, and already close to a good answer, so fewer
iterations are needed. It also tends to begin with zeros, which helps sparsity. The
functions also accept a custom `W_init`/`H_init` (not used by the pipeline yet; see
SUGGESTIONS for why overlapping tiles are a natural warm start).

If the matrix is empty, or the solver throws an error (e.g. NaN in the SVD), the code
returns all-zero W and H for that tile rather than crashing the run.

### 7.6 How many components? (`n_components`)

`n_components` (20) is an **upper limit**, not a target:

- It is capped at the smaller matrix dimension.
- It should be comfortably larger than the number of peptides you expect in a 150-scan
  tile of one narrow window. Tuning assumes about **8** (`components_in_window`).
- Too small, and co-eluting peptides are forced to share a component, giving a chimeric
  spectrum again. Too large (with too little L1), and peptides split into several partial
  components, or noise gets its own components. The L1 penalty on H is what makes
  "generous upper limit + automatic switch-off" work.

### 7.7 Is the answer unique? (Why sparsity and peak shapes help)

NMF has built-in ambiguities:

- **Scale.** $W_{:,c} \cdot s$ and $H_{c,:}/s$ give the same product. This is harmless: the
  pipeline only uses the *shape* of each spectrum and elution, and fits a normalized
  Gaussian to W. The regularization also fixes the scale in practice.
- **Order.** Components come in arbitrary order. The pipeline names them by apex time.
- **Mixing.** Sometimes you can blend two components and still have non-negative factors
  that fit equally well. Then the "pure" spectra are not recoverable from the data alone.

Mixing ambiguity goes away when the sources are **sufficiently distinct**. Mathematically,
NMF is (close to) unique when each component has some entries where it is the *only*
non-zero one ("separability", Donoho & Stodden 2003). In LC-MS/MS terms, **each peptide
needs some scans where it elutes alone, or some fragment bins that belong only to it.**
Different apex times and different fragment sets both provide that. Sparsity penalties push
the solution toward these unique answers.

The flip side: **two peptides with the same elution profile** (perfect co-elution) *and*
heavily shared fragments can't be separated by any method based on co-elution. They will
come out as one component (see Section 13).

### 7.8 NMF in practice in this pipeline

| Practical point | What happens here |
|---|---|
| Matrix orientation | Rows = 150 scans, columns = kept bins (hundreds to a few thousand). So W is 150 × k and H is k × bins |
| Scaling | Each column max-scaled to 1 before NMF. H is in scaled units, and `reverse_transform_maxscale_scans()` converts it back |
| Duplicate columns (two grids) | Both copies load on the same component, which is harmless. The exported spectra contain both |
| Reproducibility | Deterministic init + fixed `random_state = 42` + float16 round-trip = same result across reruns |
| Parallelism | One NMF per tile, spread across `n_jobs` processes with joblib |
| Zero components | Components with all-zero W are marked not-kept. They don't count and aren't exported |
| Failure handling | Solver exceptions → all-zero result for that tile, logged |

### 7.9 A worked mental example

Suppose one tile contains:

- **Peptide A** (2+), apex at scan 60, 12 fragments;
- **Peptide B** (3+), apex at scan 68, 20 fragments, **3 shared** with A (same bins);
- a background ion at m/z 445.12 that is present all the time;
- random single-scan noise spikes.

The pipeline goes like this:

1. **Constant filter** removes the 445.12 column (its mean/max is close to 1).
2. **Percentile filter + denoising** remove most spikes.
3. **Max-scaling** makes all 29 fragment traces peak at 1.
   - The 9 A-only traces peak at scan 60. The 17 B-only traces peak at 68.
   - The 3 shared traces are *sums* of both, so they show a broad double hump.
4. **NMF** with up to 20 components and the L1 penalty ends up with about 2 non-zero
   components.
   - Component 1: W has a peak at 60, and H has high weight on the 9 A-only bins, *partial*
     weight on the 3 shared bins, and zero on B-only bins.
   - Component 2 is the mirror for B.
   - The shared bins are split between the two in proportion to what best rebuilds the
     double hump. This is the key advantage over simple correlation grouping, which would
     have to give each bin to *one* peptide.
5. **Gaussian fit** of each W column gives μ ≈ RT(scan 60) and RT(scan 68), with their σ.
6. **Export**: component 1's H row, multiplied by the column maxima, becomes a clean
   12-fragment spectrum for A (the shared fragments at A's share of intensity). That is
   what gets searched.

---

## 8. Step 6: Choosing NMF settings automatically (tuning)

*Code: `decomposition.py`: `NMFMaskWrapper`, `MzBinMaskingSplitter`,
`MzBinSampleSplitter`, `OptunaSearchReconstructionCV`,
`RandomizedSearchReconstructionCV`, `pick_parameters_harmonic_mean`,
`tune_hyperparameters_*`; `MSRun.get_tuning_windows`*

### 8.1 Why tuning is needed, and why it's tricky

The penalties (`alpha_W`, `alpha_H`, `l1_ratio`) decide how many components survive and
how sparse the spectra are. The right values depend on the instrument, gradient length,
window width, sample complexity and noise level. There is no ground truth: we don't know
the "true" spectra. So the tuning has to judge settings by *internal* criteria.

### 8.2 Which data is used

`get_tuning_windows` samples about `window_sampling_n = 50` tiles **across all isolation
windows**. It skips the start and end of each isolation window (`exclude_scan_window_edges ×
scan_overlap` = 300 scans), where the gradient is usually empty. The sampling probability
is `window_sampling_n` divided by the number of candidate tiles summed over all isolation
windows. If the edge exclusion leaves no candidates (a very short run), it logs a warning
and returns no tiles. The tiles are filtered and scaled exactly as in the real run, so
tuning sees the same kind of matrices.

### 8.3 Cross-validation for an unsupervised method

In supervised learning you hold out labeled examples. Here we hold out **matrix entries**
(the "Mask" splitter, default), or **whole scans** (the "Sample" splitter):

- **Mask** (`MzBinMaskingSplitter` with `mask_signal = True, balance_mask_signal = True`):
  pick a random 20% (`test_fraction`) of the **non-zero** cells, plus a matching fraction of
  zero cells. Set them to zero in a training copy, fit NMF, then check how well $WH$
  predicts the **original** values of the hidden cells.
  - The idea: if the model has captured real co-elution structure, it can "fill in" a hidden
    fragment intensity from the other fragments of the same peptide at that scan and the
    same fragment at nearby scans.
  - A model that fits noise can't do that.
- **Sample** (`MzBinSampleSplitter`): hold out 20% of the scans (rows), fit on the rest,
  then project the held-out scans onto the learned spectra (`model.transform`) and measure
  the error. This tests whether the learned *spectra* generalize to unseen time points.

Each trial repeats this `n_splits = 3` times with different random masks, across all ~50
tiles.

> **Nuance (Mask splitter).** Hidden cells are set to **zero** in the training matrix, not
> treated as *missing*. So NMF is actually asked to reproduce zeros there, and the test error
> measures how far the structure *pulls the model away from* those fake zeros toward the
> truth. This still ranks settings sensibly, but it is biased toward models that don't fit
> small values closely. A missing-data-aware (weighted) NMF would be the textbook approach.
> See SUGGESTIONS.

> **Which fits are scored.** Each trial fits `n_splits × tiles` NMFs. Each metric is
> averaged over all tiles within a split, giving one value per split. Before a fix made on
> 2026-10-13, only the first `n_splits` fits (split 0, first few tiles) were scored. See
> [Section 18.5](#185-tuning-internals).

### 8.4 The objectives

Each trial is scored on several measures (`objective_params`). All are turned into
"higher is better" before optimization.

| Objective | What it measures | What we want |
|---|---|---|
| `test_reconstruction_errors` | Mean squared error on the held-out cells | **Low**: the model generalizes |
| `neglog_ratio_train_test_reconstruction_error` | $\log_2(\text{train MSE}/\text{test MSE})$ after sign flips | **Close to or above 0**: training error not much lower than test error, i.e. no overfitting |
| `weight_orthogonality` | How far the *spectra* (rows of H) are from being mutually orthogonal: the normalized Gram matrix $HH^\top$ compared with the identity | **Low**: components have distinct spectra (not the same peptide split in two) |
| `sample_orthogonality` | The same for the *elution profiles* (columns of W): $W^\top W$ vs identity | **Low**: components elute at different times (not duplicates) |
| `fraction_window_component` | Fraction of scans in the tile where at least one component is non-zero | **Close to a target** = `components_in_window × 4 × component_sigma / scan_width`. With 8 components, σ = 3 scans and 150 scans: 8·12/150 = 0.64. That is how much of the tile we *expect* to be covered by peptide peaks (±2σ each) |

Also available: `nonzero_component_fraction` (target = `components_in_window /
n_components`) and Hoyer-sparsity scores of W and H.

> The code's names are a bit crossed: `weight_orthogonality` is computed from **H**
> (spectra) and `sample_orthogonality` from **W** (elution). The table above describes what
> is actually computed.

Targets are scored as $-|\text{value} - \text{target}|$ (`OptimizationParameters.score`),
so the best score is 0, reached when the value hits the target.

### 8.5 Searching the settings space

- **Optuna** (`optuna = True`, default). A multi-objective study (`directions = maximize`
  for each objective) over
  - `l1_ratio ∈ [0.1, 0.9]`,
  - `alpha_W ∈ [1e-4, 0.1]` (log scale),
  - `alpha_H ∈ [1e-4, 0.1]` (log scale).

  Optuna's multi-objective sampler (NSGA-II) proposes new settings based on the **Pareto
  front**: the trials where no objective can be improved without making another worse.
  Workers run in parallel and share one study stored in a temporary **journal file**. The
  file is deleted at the end. Failed trials are dropped. The run only stops with an error
  if fewer than half of `n_iter` trials succeed.
- **Randomized search** (`optuna = False`). `n_iter` random picks from the discrete
  `hyperparameter_grid`, all evaluated in parallel.

### 8.6 Picking one winner: the harmonic mean

A multi-objective search gives a *set* of good trade-offs, but we need one setting.
`pick_parameters_harmonic_mean`:

1. takes each objective's mean over CV splits for every trial, and re-applies the target
   scoring;
2. **min-max scales** each objective across trials to [0, 1] (1 = best trial on that
   objective);
3. computes each trial's **harmonic mean** across objectives (plus a small constant
   0.001);
4. picks the highest.

**Why harmonic?** It is dominated by the *worst* score. A trial that is excellent on four
objectives but terrible on one gets a low harmonic mean, while a trial that is good on all
of them wins. That is the behavior you want for a balanced decomposition.

The chosen `alpha_W`, `alpha_H` and `l1_ratio` are written into `config.nmf`, used for the
real run, and saved in `<stem>.out.ini`.

**Cost:** about `n_iter × n_splits × tiles` NMF fits (500 × 3 × 50 ≈ 75,000 with defaults).
For many datasets you can tune once on a representative run and reuse the values with
`run_tune = False`.

---

## 9. Step 7: Describing each component's elution (Gaussian fits)

*Code: `fitpeaks.fit_gaussian_elution`, `containers.fit_scanwindow`,
`ScanWindow.set_peak_fits`*

Each kept component's elution profile (a column of W, over the tile's RTs) is:

1. rescaled to exactly [0, 1]: subtract its minimum, then divide by its range;
2. given starting guesses from its **moments**: μ₀ = intensity-weighted mean RT,
   σ₀ = intensity-weighted standard deviation;
3. fitted with a unit-height Gaussian $\exp(-(t-\mu)^2/2\sigma^2)$ using bounded nonlinear
   least squares (`scipy.optimize.curve_fit`), with `sigma_bounds` (default 0.1–10 RT
   units, usually minutes).

**Why a Gaussian?** Chromatographic peaks are close to Gaussian (usually with some tailing).
Two numbers, **apex μ** and **width σ**, give a compact, comparable description:

- **μ** is the component's retention time and picks the "apex scan" for the raw export.
- **μ ± a few σ** defines the time region for MS1 matching (Section 11).
- The fit **error (MAE)** is a quality score. It is used to decide which duplicate to keep
  (Section 10).

**Components are dropped when:**

- the fit fails, or the profile is empty or constant (that raises an error, which marks the
  component not kept), or
- the peak is **within 2σ of the zeroed edges** (`μ − 2σ` before the 6th scan or `μ + 2σ`
  after the 5th-from-last). Such peaks are probably cut off. An overlapping tile will have
  the whole peak.

> Two notes for interpretation. (1) The `sigma_bounds` are in **RT units**, while the tuning
> parameter `component_sigma` is in **scans**. With sharp peaks (σ ≈ 2–5 s ≈ 0.03–0.08 min),
> the default lower bound of 0.1 min forces σ up to 0.1. (2) The MAE is the mean absolute
> difference between the [0, 1]-scaled profile and the fitted unit-height curve, so it is
> on a 0–1 scale and comparable between components. The height is fixed at 1, not fitted,
> so a profile whose sampled maximum misses the true apex is fitted a little wide.

Components get a **name** encoding where they came from:
`"{isolation target}_{slot}-{apex scan index}_{μ}_{σ}"`.

---

## 10. Step 8: Removing duplicates from overlapping windows

*Code: `GPFRun.collapse_redundant_components`, `fitpeaks.approximate_overlap_curves`*

Because tiles overlap by 120 of 150 scans, a peptide eluting in the middle of that
overlap is found in up to 5 tiles. For every pair of overlapping tiles in the same
isolation window, and every pair of components (one from each tile):

1. **Do they elute at the same time?** Compute the **overlap of their fitted Gaussians**:
   the area under $\min(p_1, p_2)$ divided by the area under $p_1$. It is 1 for identical
   peaks and 0 for peaks far apart. The areas are summed numerically over ±6·max(σ₁, σ₂)
   around the two apexes. Pairs whose apexes are more than 3(σ₁ + σ₂) apart are skipped
   without integrating, because they can't reach the threshold. Skip the pair if ≤ 0.5.
2. **Do they have the same spectrum?** Put both H rows on the union of the two tiles' bins
   and compute **cosine similarity**. Treat them as duplicates if > 0.75.
3. If both hold, **keep the one with the lower Gaussian-fit error** and mask the other.

```
tile 3:   ····/\····          component 3.2  μ = 21.40 min, cos(H) with 4.1 = 0.93
tile 4:        ····/\····     component 4.1  μ = 21.41 min
                               overlap 0.97 > 0.5 and similarity 0.93 > 0.75 → keep the better fit
```

**Why both tests?** Same time with a different spectrum means two peptides that co-elute.
Keep both. Same spectrum at a different time means isomers, or the same peptide eluting
twice (e.g. an oxidation variant, or a carry-over tail). Keep both, because they may be
interesting.

This only compares tiles **within one isolation window**. With narrow windows, a
precursor's isotope envelope often spans *adjacent* isolation windows, so the same peptide
can appear as components in neighboring windows. Those are not collapsed here. Downstream,
group them by `ms1_id` or by spectrum similarity.

> The duplicate test uses the Gaussian-fit MAE from Section 9. Since the fixes, that MAE
> really measures fit quality, so the better-shaped duplicate is kept. In `lazy = True` mode
> the masks are written to HDF5 *after* the collapse, so lazy and in-memory runs export the
> same components.

---

## 11. Step 9: Linking components to precursors (MS1 matching)

*Code: `containers.add_ms1_features_scanwindow`,
`containers.fit_ms1_ms2_feature_matching_scanwindow`, `ms1feature.MS1Feature`,
`fitpeaks.least_squares_with_l1_bounds`*

A component tells us *which fragments* belong together and *when* they eluted, but only
that the precursor was somewhere in the isolation window. To search with a tight precursor
tolerance, we need the **exact precursor mass and charge**. That comes from MS1.

### 11.1 MS1 features

**Biosaur2** detects MS1 features: isotope envelopes traced over time ("hills"), each with
a calibrated monoisotopic mass, charge, m/z, RT extent and intensity profile. For every
tile, the pipeline attaches the features that:

- **overlap in time** with the tile (`rtStart < tile end` and `rtEnd > tile start`), and
- have **at least one isotope peak** `mz + i · 1.00336 / z` (i = 0 … nIsotopes−1)
  **inside the isolation window bounds**. 1.00336 Da is the ¹³C–¹²C mass difference.

The second rule matters for narrow windows. With a 2 m/z window, a 2+ peptide's M, M+1 and
M+2 isotopes (0.5 m/z apart) may fall in *different* windows. Each window that contains any
isotope really did fragment that peptide, so the peptide should be a candidate there.

Each feature's monoisotopic hill intensities are placed on an evenly spaced RT axis from
`rtStart` to `rtEnd`, then **linearly interpolated onto the tile's MS2 scan times**. A zero
is added one RT step before and after the hill, so the profile falls to zero outside the
hill without losing its first and last measured points. Outside that range it is zero. MS1
and MS2 profiles now share one time axis.

> The evenly spaced RT axis is an assumption. Biosaur2 hills can skip MS1 scans, and the
> real MS1 scan times aren't stored. Gapped hills are therefore slightly stretched or
> compressed in time (SUGGESTIONS A18, still open).

### 11.2 The regression

For each kept component *k* in a tile:

1. **Time region**: the 99% interval of its Gaussian, $[\mu - 2.58\sigma,\ \mu + 2.58\sigma]$,
   widened by `extend_w_fraction = 0.5` of its width on **each side**. The region is
   therefore about 2× the 99% interval, which includes the peak's shoulders.
2. **Target** $y$: the component's elution (W column) in that region, scaled to unit length.
3. **Candidates** $X$: one column per attached MS1 feature with any signal in the region,
   each scaled by the length of its *whole-tile* profile.
4. Solve

$$
\min_{0 \le c_j \le 1}\; \lVert X c - y\rVert_2^2 \;+\; \alpha \sum_j c_j
$$

with SLSQP, `alpha = 0.1`. Then set any $c_j \le$ `c_cutoff` (0.025) to zero.

**Why a regression and not a simple correlation?**

- Several precursors can be present in the window and elute close together (other
  peptides, the M+1 isotope of a lighter neighbor, other charge states). Their MS1 profiles
  are *correlated with each other*, so simple correlation would give several of them high
  scores. A **joint** regression shares the explanation between them and gives most weight
  to the one whose shape actually matches.
- **Non-negativity** ($c \ge 0$): a precursor can't contribute negative fragment
  intensity.
- **Upper bound 1**: with unit-scaled vectors, a perfect match has coefficient about 1.
- **L1 penalty α**: prefers **few** precursors (ideally one). A weak, partial match must
  justify its penalty to get a non-zero coefficient.

**Why the scaling?**

- Scaling the MS2 target to unit length makes the problem independent of how bright the
  component is.
- Scaling each MS1 candidate by its *whole-tile* length (not just the region) means a
  feature with most of its signal *outside* the component's elution region comes in with a
  small vector. It would need a large coefficient to matter, which the L1 penalty and the
  upper bound discourage. In effect, this penalizes features that only partly overlap in
  time.

### 11.3 What gets reported

For each component:

- `ms1_matching_coef`: $c_j$ for each kept feature (top `top_n_ms1_matches = 4` including
  ties, and ≥ `matching_coefficient_threshold = 0.1`).
- `ms1_global_explained_variance`: R² of $Xc$ vs $y$. How much of the component's elution
  shape the chosen precursors explain together. Close to 1 = convincing.
- `ms1_individual_explained_variance`: R² of each feature alone (with its coefficient).
- the feature's `mass`, `charge`, `mz`, apex RT and intensities, and `ms1_id`.

One component can match **several** precursors (it becomes several rows in the Parquet and
several spectra in the `.ms2`). That is deliberate. When it is ambiguous which precursor it
is, **let the search engine decide** by searching the same spectrum with each candidate
precursor. The same MS1 feature can also match components in several adjacent windows (the
isotope rule).

---

## 12. Using the components downstream

The Parquet table is a **component-level, deconvolved view of the whole DIA run**. That
supports several analyses beyond a single database search.

### 12.1 Database searching

- Search `intensity_transform_weight` spectra (MS1-matched) as the main pass. Use
  non-specific cleavage, peptidomics PTMs as variable, and a precursor tolerance matching
  Biosaur2's mass accuracy.
- Search `--wide_window` spectra for components without an MS1 match. The precursor is only
  known to about the window width, and the charge is written as 2, so let the engine try
  charges 1–4 and use a precursor tolerance of half the window width (in Da). This is like
  a "narrow open search".
- Set fragment tolerance ≥ 1.5 × the bin width (Section 4) to absorb the bin-edge m/z
  offset and the duplicate peaks from two grids.
- **FDR note.** Deconvolved pseudo-spectra have different score distributions than raw DIA
  or DDA spectra. Target–decoy is still valid if applied consistently. Rescoring with
  component-level features (below) usually helps a lot.

**Useful rescoring features from the Parquet:** `ms2_component_sigma` (peak width),
`ms1_matching_coef`, `ms1_global_explained_variance`, `ms1_individual_explained_variance`,
number of non-zero fragments in the component, RT difference between the MS2 apex and
`ms1_apex_rt`, and precursor intensity.

### 12.2 Clustering components

Each component's H row is a spectrum on a **fixed bin grid** (fixed for a given
discretization config). That makes comparisons cheap:

1. Build a sparse matrix *components × bins* from `ms2_component_weight_mz/intensity`
   (or re-bin with your own tolerance). Normalize rows (unit length, or √intensity for less
   dominance by top peaks).
2. **Cosine similarity** between components is then a sparse matrix product.
3. Cluster (connected components above a threshold, HDBSCAN, or Leiden on a kNN graph).
   Work within an isolation window or precursor-mass neighborhood, or globally for
   modification discovery (12.4).
4. Across runs, clusters become **consensus spectra**: recurring species with stable RT.

This lets you:

- find **recurring unidentified species** (big clusters with no IDs are the most
  interesting unknowns);
- **collapse duplicates** across adjacent isolation windows (isotope envelopes) and across
  replicate runs;
- propagate an identification to a whole cluster (with care).

### 12.3 Deep dives for missed identifications

For unidentified components with high quality (clean Gaussian, high explained variance,
many fragments):

- Check whether a **nearby identified component** (similar RT, similar spectrum) explains
  it. It may be a charge-state variant or isotope-window duplicate that failed the FDR cut
  alone.
- Re-search the component alone with a broader search space (more PTMs, semi/non-specific,
  other proteomes such as microbiome or variants).
- Look for **precursors outside the MS1 match list**. The wide-window mode tries all
  precursors in the isolation window.

### 12.4 Finding modifications with spectral similarity (delta masses and site localization)

This is where deconvolved components help the most. For an **identified** peptide P
(precursor mass $M_P$) and an **unidentified** component U (precursor mass $M_U$), let
$\Delta = M_U - M_P$.

If U is P carrying a modification of mass Δ at residue *i*, then U's fragments split into
two groups:

- fragments that **do not contain** residue i appear at the **same** m/z as in P;
- fragments that **do contain** residue i appear **shifted by Δ/z** (z = fragment charge).

For a peptide $A_1 A_2 \dots A_n$ modified at $A_i$:

| Ion series | Unshifted | Shifted by Δ |
|---|---|---|
| b-ions | $b_1 \dots b_{i-1}$ | $b_i \dots b_{n-1}$ |
| y-ions | $y_1 \dots y_{n-i}$ | $y_{n-i+1} \dots y_{n-1}$ |

Example: 6-residue peptide, modification on residue 3:

```
    A1  A2 [A3*] A4  A5  A6
b:  b1  b2 | b3* b4* b5*          b1–b2 unshifted, b3–b5 shifted
y:          y5* y4* | y3  y2  y1  y1–y3 unshifted, y4–y5 shifted
```

So:

1. **Similarity that allows a shift** finds candidate pairs. The "modified cosine" from
   metabolomics molecular networking matches each peak of U to a peak of P at either the
   same m/z or m/z + Δ/z. A high modified cosine with a large number of shifted matches
   strongly suggests "same peptide + Δ".
2. **Δ tells you what the modification may be.** Build a **histogram of Δ** over all
   high-similarity pairs. Peaks stand out at known masses. For peptidomics, look for:
   - −0.984 (C-terminal amidation)
   - −17.027 (pyro-Glu from Q) and −18.011 (from E)
   - +42.011 (acetylation)
   - +15.995 (oxidation)
   - +79.966 (phosphorylation) and +79.957 (sulfation), which are distinguishable at high
     mass accuracy
   - +14.016 (methylation)
   - +0.984 (deamidation)
   - amino-acid residue masses (extensions or truncations of the same peptide by one
     residue, very common for endogenous peptides)

   Unexpected peaks point to modifications you weren't searching for. Unimod lookup
   names them.
3. **Where the pattern switches from unshifted to shifted localizes the site.** In the
   table, the change between $b_{i-1}$/$b_i$ and $y_{n-i}$/$y_{n-i+1}$ brackets residue i.
   Counting matched shifted vs unshifted ions for each candidate position gives a
   localization score, the same idea as Ascore/PTMProphet, applied here to an
   *unidentified* spectrum matched to a *known* one.
4. **Confirm by re-searching** U with Δ as a variable modification on the suggested
   residue(s), or with a mass-offset search (e.g. MSFragger's offset mode) restricted to
   the discovered Δ values.

Why this works better on components than on raw DIA spectra: in a raw chimeric spectrum,
"shifted" matches can come from any co-isolated peptide by chance. A component's spectrum
has had the co-eluting peptides removed, so matched and shifted peaks are much more likely
to be real.

### 12.5 Quantification (not exported yet)

W holds each component's elution profile, and the Gaussian μ and σ are exported. With
the per-channel scale factors ($\max_j$), a component's total fragment signal is
approximately $\sum_t W_{t,k} \times \sum_j H_{k,j}\max_j$. You can also integrate a fitted
Gaussian over RT. That would give MS2-level quantification per precursor, which is
usually more selective than MS1 in crowded regions. The W columns themselves are not in
the Parquet (see SUGGESTIONS).

---

## 13. Limitations and failure modes

| Situation | What happens | Mitigation |
|---|---|---|
| **Perfect co-elution** of two precursors in the same window | NMF cannot separate them (no distinct elution information), so you get one chimeric component | Narrower windows; search the component with multiple precursors (MS1 matching does this) |
| **Heavily shared fragments** (isomers, sequence variants) | Shared bins are split, but spectra may still be mixed | Spectral similarity / localization (12.4) |
| **Very low-abundance peptides** | Few fragments survive the percentile and morphological filters, so the fit fails or the component is dropped | Longer peaks (slower gradient), lower `_percentile_filter`, lower `alpha_H` |
| **Very sharp peaks** (σ ≤ 2 scans) | Denoising and erosion shorten them, and the Gaussian σ is limited by `sigma_bounds[0]` | Faster cycle time or narrower tiles; lower `sigma_bounds[0]` |
| **Very broad peaks** (≥ 150 scans) | The peak never fits inside one tile, so it is dropped as an edge component | Increase `scan_width` |
| **Background with slow drift** | Can survive the constant filter (mean/max < 0.3) and get a component | Such components have huge σ or poor fits; filter on σ and fit error downstream |
| **Precursor isotopes spanning windows** | The same peptide appears in 2–3 adjacent windows | Deduplicate by `ms1_id` / spectrum similarity |
| **Bin-edge m/z** | Reported fragment m/z is biased high by up to one bin | Wider fragment tolerance, or re-centroid from raw peaks (SUGGESTIONS) |
| **Unknown charge (wide-window)** | Charge written as 2 | Let the search engine try multiple charges |

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Chimeric spectrum** | An MS2 spectrum that contains fragments of more than one precursor |
| **Co-elution** | Two signals rising and falling together over retention time |
| **Component** | One NMF factor: elution profile (W column) + spectrum (H row) |
| **Coordinate descent (CD)** | Optimization that updates one block of variables at a time, each by an exact small solve |
| **Cosine similarity** | $a\cdot b/(\lVert a\rVert\lVert b\rVert)$. 1 = same shape, 0 = nothing in common |
| **Elastic net** | Mix of L1 and L2 penalties |
| **Frobenius norm** | $\sqrt{\sum X_{ij}^2}$. Its square is the sum of squared entries |
| **Harmonic mean** | $k / \sum (1/x_i)$. Dominated by the smallest $x_i$ |
| **Hoyer sparsity** | A 0–1 measure of how concentrated a vector is (1 = one non-zero entry) |
| **Isolation window** | The m/z range the quadrupole passes for fragmentation |
| **L1 penalty** | Sum of absolute values. Drives small values to exactly zero |
| **L2 penalty** | Sum of squares. Shrinks values smoothly |
| **Max-scaling** | Dividing a vector by its maximum so it peaks at 1 |
| **Morphological closing/erosion** | Min/max filters that remove narrow valleys/peaks in a signal |
| **NNDSVD** | SVD-based deterministic initialization for NMF |
| **Pareto front** | The trials that are not beaten on every objective by any other trial |
| **Rank (k)** | Number of components in a factorization |
| **SLSQP** | A general-purpose constrained optimizer in SciPy |
| **Sparse matrix** | A matrix stored as only its non-zero entries |

---

## 15. Further reading

**NMF**
- Lee, D. D. & Seung, H. S. (1999). *Learning the parts of objects by non-negative matrix
  factorization.* Nature 401, 788–791. (The parts-based idea.)
- Lee, D. D. & Seung, H. S. (2001). *Algorithms for non-negative matrix factorization.*
  NIPS 13. (Multiplicative updates.)
- Cichocki, A. & Phan, A.-H. (2009). *Fast local algorithms for large scale nonnegative
  matrix and tensor factorizations.* IEICE Trans. (HALS / coordinate descent.)
- Boutsidis, C. & Gallopoulos, E. (2008). *SVD based initialization: a head start for
  nonnegative matrix factorization.* Pattern Recognition 41. (NNDSVD.)
- Hoyer, P. O. (2004). *Non-negative matrix factorization with sparseness constraints.*
  JMLR 5. (Hoyer sparsity.)
- Donoho, D. & Stodden, V. (2003). *When does non-negative matrix factorization give a
  correct decomposition into parts?* NIPS 16. (Uniqueness/separability.)
- Gillis, N. (2014). *The why and how of nonnegative matrix factorization.* arXiv:1401.5226.
  (Readable overview.)
- Owen, A. B. & Perry, P. O. (2009). *Bi-cross-validation of the SVD and the nonnegative
  matrix factorization.* Ann. Appl. Stat. 3. (Cross-validating matrix factorizations.)
- scikit-learn documentation, `sklearn.decomposition.NMF`. (Exact objective and
  regularization scaling used here.)

**DIA deconvolution and search**
- Tsou, C.-C. et al. (2015). *DIA-Umpire: comprehensive computational framework for
  data-independent acquisition proteomics.* Nat. Methods 12. (Pseudo-spectra from
  co-elution.)
- Peckner, R. et al. (2018). *Specter: linear deconvolution for targeted analysis of
  data-independent acquisition mass spectrometry proteomics.* Nat. Methods 15.
- Kong, A. T. et al. (2017). *MSFragger: ultrafast and comprehensive peptide identification
  in mass spectrometry-based proteomics.* Nat. Methods 14. (Open / mass-offset search.)

**Spectral similarity with mass shifts**
- Watrous, J. et al. (2012). *Mass spectral molecular networking of living microbial
  colonies.* PNAS 109. (Modified cosine / molecular networking.)

---

# Part II: Code walkthrough

Part I explained *why* each step exists. This part follows the code. It traces one
`ffd process_mzml` run call by call, in execution order. For each step it gives:

- the functions that run;
- the arrays going in and out, with their shapes and columns;
- the object fields that get set;
- the numeric operations, in order.

Functions are named as `module.function` or `Class.method`. Line numbers are left out on
purpose, because they change. Each step ends with a short **Cost** note. Measured timings,
and the settings that change them, are in
[RUNNING.md § Performance](RUNNING.md#performance-bottlenecks-and-what-to-adjust).

---

## 16. Map of the code

### 16.1 Modules

| Module | Main contents | Role |
|---|---|---|
| `cli.py` | `process_mzml`, `convert_mzml_h5long`, `dump_df_to_ms2`, `generate_template_ini`, logging decorators | Click entry points (`ffd …`), orchestration |
| `config.py` | `Config`, `AttributeDict`, `Constants`, `format_logger` | All settings, INI read/write, physical constants |
| `converter.py` | `MzMLParser` (subclass of `pyteomics.mzml.MzML`) | mzML → long-format HDF5 |
| `discretization.py` | `MzDiscretize` | ppm/Da bin grids, m/z → bin index |
| `utils.py` | `pivot_unique_binned_mz_sparse`, `calculate_nmf_summary`, `fraction_explained_variance`, `feature_df_to_ms2` | Pivoting, tuning metrics, R², `.ms2` writer |
| `fitpeaks.py` | `fit_gaussian_elution`, `gaussian_curve`, `approximate_overlap_curves`, `least_squares_with_l1_bounds` | Peak fitting, Gaussian overlap, bounded L1 regression |
| `decomposition.py` | `fit_nmf_matrix_custom_init`, `NMFMaskWrapper`, `MzBinMaskingSplitter`, `MzBinSampleSplitter`, `OptunaSearchReconstructionCV`, `RandomizedSearchReconstructionCV`, `_fit_and_score`, `OptimizationParameters`, `pick_parameters_harmonic_mean`, `tune_hyperparameters_*` | NMF wrapper and all tuning |
| `containers/msrun.py` | `MSRun` | The whole run: a dict of `GPFRun`s (or HDF5 paths to them) |
| `containers/gpfrun.py` | `GPFRun` | One isolation window over time: matrix, tiles, collapse |
| `containers/scanwindow.py` | `ScanWindow` | One tile: filtering, NMF results, fits, MS1 links, HDF5 I/O |
| `containers/ms1feature.py` | `MS1Feature` | One Biosaur2 feature and its interpolated profile |
| `containers/__init__.py` | `fit_nmf_matrix_*`, `fit_scanwindow`, `add_ms1_features_*`, `fit_ms1_ms2_feature_matching_*`, `dump_features_to_df_*` | Stage functions at three levels: per tile (`_scanwindow`), per isolation window (`_gpfrun`, parallel over tiles), and per run (`_msrun`, loops over isolation windows) |
| `simulation.py` | `assemble_species`, `assemble_fragments`, `bin_scans`, … | Synthetic data helpers (not used by the CLI) |

### 16.2 Call graph of `ffd process_mzml`

```
cli.process_mzml
├─ Config.from_ini(ini)  or  Config()
├─ cli.convert_mzml_h5long
│   └─ MzMLParser.to_ms2_h5 ─ parse_scan (per spectrum)          mzML → <stem>.h5:/<target>/ms2_long
├─ MSRun.from_h5_long                                             (opens <stem>.h5 in "a" mode)
│   └─ for each isolation target (sorted by m/z):
│       ├─ GPFRun.from_undiscretized_long
│       │   ├─ MzDiscretize.discretize_mz_array ─ discretize
│       │   └─ GPFRun.from_discretized_long
│       │       ├─ utils.pivot_unique_binned_mz_sparse
│       │       └─ GPFRun.__init__ ─ build_scan_windows ─ slice_scan_window → ScanWindow
│       └─ MSRun.add_gpf ─ ScanWindow.filter_scans (every tile, serial)
│                        └─ [lazy] MSRun.dump_gpfrun_to_h5 ─ GPFRun.dump_h5 ─ ScanWindow.dump_h5
├─ [tuning.run_tune]
│   ├─ MSRun.get_tuning_windows ─ GPFRun.sample_scan_windows ─ ScanWindow.filter_scans
│   └─ tune_hyperparameters_optunasearchcv   (or _randomizedsearchcv)
│       ├─ OptunaSearchReconstructionCV.fit
│       │   └─ joblib workers: optimize_optuna_study ─ study.optimize(objective)
│       │       └─ objective ─ _fit_and_score ─ NMFMaskWrapper.fit
│       │           ├─ MzBinMaskingSplitter.split  (or MzBinSampleSplitter)
│       │           ├─ fit_nmf_matrix_custom_init ─ sklearn NMF
│       │           └─ utils.calculate_nmf_summary
│       └─ pick_parameters_harmonic_mean            → config.nmf.{alpha_W, alpha_H, l1_ratio}
├─ containers.fit_nmf_matrix_msrun
│   └─ for each GPF: MSRun.get_gpf ─ fit_nmf_matrix_gpfrun
│       ├─ joblib per tile: [filter_scans] ─ fit_nmf_matrix_scanwindow ─ ScanWindow.set_nmf_fit
│       │                   └─ fit_scanwindow ─ fit_gaussian_elution ─ ScanWindow.set_peak_fits
│       ├─ GPFRun.set_scan_windows(persist=False)
│       ├─ GPFRun.collapse_redundant_components ─ approximate_overlap_curves, scipy cosine
│       └─ GPFRun.persist_scan_windows            ([lazy] write every tile)
├─ [--ms1_features_file]
│   ├─ pandas.read_csv(TSV)
│   ├─ add_ms1_features_msrun ─ per GPF ─ add_ms1_features_gpfrun
│   │   └─ joblib per tile: add_ms1_features_scanwindow ─ MS1Feature.from_biosaur2_series_with_ms2
│   └─ fit_ms1_ms2_feature_matching_msrun ─ per GPF ─ fit_ms1_ms2_feature_matching_gpfrun
│       └─ joblib per tile: fit_ms1_ms2_feature_matching_scanwindow
│           ├─ least_squares_with_l1_bounds ─ scipy SLSQP
│           ├─ utils.fraction_explained_variance
│           └─ ScanWindow.set_ms1_ms2_feature_matches
│   (no MS1 file → config.feature_matching.match_ms1 = False)
├─ dump_features_to_df_msrun ─ per GPF ─ dump_features_to_df_gpfrun ─ dump_features_to_df_scanwindow
├─ DataFrame.to_parquet(<output_dir>/<stem>.features.parq)
├─ Config.to_ini(<output_dir>/<stem>.out.ini)
└─ MSRun.close

cli.dump_df_to_ms2 ─ utils.feature_df_to_ms2      Parquet → <output_dir>/<stem>.features.<type>[.wide].ms2
```

---

## 17. Data layouts and object state

### 17.1 Arrays as they move through the pipeline

| Stage | Name in code | Type / shape | Columns |
|---|---|---|---|
| After conversion | `<stem>.h5:/<target>/ms2_long` | float32, (peaks, 7) | `[target, scan, rt, mz, intensity, low, high]`. Without `include_gpf_bounds`, there are 5 columns (no `low, high`) |
| After discretization | `disc_m` from `MzDiscretize.discretize_mz_array` | float64, (≈ 2 × peaks, 9) | the 7 columns above + `[bin_index, bin_right_edge]`. Rows are stacked grid 0 then grid 1, each sorted by m/z. `bin_index` is global: `row_in_grid + grid × n_bins_per_grid` |
| Pivoted | `GPFRun._m` | `scipy.sparse.csr_matrix`, (scans, unique bins), float32 | rows = sorted unique scan numbers. Columns = sorted unique global bin indices (`GPFRun._bin_indices`). Values = summed intensity |
| Raw peaks | `GPFRun._m_long` | float32, (unique peaks, 4) | `[scan, rt, mz, intensity]`: `np.unique` over the discretized rows' columns 1–4, which removes the second-grid copy. It is sorted by scan |
| Per tile | `ScanWindow._m` | csr, (150, kept bins) after filtering | max-scaled, denoised intensities. `ScanWindow.m` returns it dense |
| Tuning input | `windows` from `MSRun.get_tuning_windows` | list of dense float32 arrays, (150, kept bins) | copies of `ScanWindow.m` for sampled tiles |
| NMF output | `ScanWindow._w`, `_h` | (150, k) and (k, kept bins), float32 (scikit-learn keeps the input dtype) | elution profiles and spectra in scaled units |
| Export | `features_df` | pandas DataFrame | one row per (component × kept MS1 match). See [18.10](#1810-export-to-dataframe-and-parquet) |

### 17.2 `MSRun`

| Field | Meaning |
|---|---|
| `_gpf_runs` | `dict[int → GPFRun]` in memory, or `dict[int → "gpf_runs/gpf_<i>"]` (an HDF5 path) when lazy |
| `_lazy` | from `config.lazy` |
| `_h5_file` | open `h5py.File(…, "a")` on `<stem>.h5`, or `None` |
| `_isolation_low`, `_isolation_high` | unused by the pipeline. Stored as HDF5 attributes (−1 for `None`) |

`MSRun.get_gpf(i)` returns the in-memory object, or rebuilds a `GPFRun` from HDF5 with
`GPFRun.from_h5`. A rebuilt object is new each time: changes are only kept if they are
written back (Section 19).

### 17.3 `GPFRun`

| Field | Meaning |
|---|---|
| `_gpf` | isolation target m/z (float) |
| `_gpf_low`, `_gpf_high` | isolation bounds from the first row (`None` without `include_gpf_bounds`) |
| `_m`, `_m_long` | see 17.1 |
| `_scan_number` | sorted unique scan numbers (spectrum index + 1) |
| `_scan_index` | `0 … n_scans−1`, the row positions. Tiles and collapse use these |
| `_retention_time` | one RT per scan |
| `_bin_indices` | global bin index of every column of `_m` |
| `_discretize` | the `MzDiscretize` used (saved in HDF5 too) |
| `_window_scan_width`, `_window_scan_overlap` | `scan_filter.scan_width` (150) and `scan_filter.scan_overlap` (30, the stride) |
| `_scan_windows` | list of `ScanWindow` tiles in time order |
| `_msrun`, `_index` | back-reference and position, used to find its HDF5 group when lazy |

### 17.4 `ScanWindow`

Fields are set progressively. Each row of the table says which step sets the field.

| Field | Set by | Meaning |
|---|---|---|
| `_gpf`, `_gpf_low`, `_gpf_high` | `__init__` | copied from the `GPFRun` |
| `_scan_number`, `_scan_index`, `_retention_time` | `__init__` | the tile's 150 rows |
| `_m_unfilter` | `__init__` | the raw csr slice, (150, *all* GPF bins). Not saved to HDF5 by default, so after reloading it is just a copy of `_m` |
| `_mz_bin_indices_unfilter`, `_mz_unfilter` | `__init__` | all GPF bins and their right-edge m/z |
| `_m`, `_mz_bin_indices`, `_mz` | `__init__`, then shrunk by every filter | the working matrix and its column labels |
| `_mz_masked` | `filter_mz_axis` | boolean over the *unfiltered* bins: True = column removed |
| `_m_max`, `_m_max_bin_indices` | `transform_maxscale_scans` | the per-column maxima used for scaling, and the bins they belong to |
| `_is_filtered` | `filter_scans` | True after the whole filter chain |
| `_w`, `_h` | `set_nmf_fit` | (150, k) and (k, bins) for *all* k NMF slots |
| `_nmf_reconstruction_error` | `set_nmf_fit` | `model.reconstruction_err_` (Frobenius norm of X − WH) |
| `_non_zero_components` | `set_nmf_fit` | `W.sum(axis=0) != 0`, length k |
| `_component_keep` | `set_nmf_fit`, `set_peak_fits`, `mask_component` | length k. Starts equal to `_non_zero_components` and is cleared by failed fits, edge peaks and duplicates |
| `_component_means`, `_component_sigmas`, `_component_maes` | `set_peak_fits` | length k. Zero for slots that weren't fitted |
| `_component_names` | `set_peak_fits` | `"{gpf}_{slot}-{apex scan_index}_{μ}_{σ}"` for every slot |
| `_ms1_features` | `set_ms1_features` | list of `MS1Feature` |
| `_component_ms1_coef_matrix` | `set_ms1_ms2_feature_matches` | (kept components, features) regression coefficients |
| `_component_ms1_global_exp_var`, `_component_ms1_individual_exp_var` | `set_ms1_ms2_feature_matches` | (kept components,) and (kept components, features) |

The public properties `w`, `h`, `component_names`, `component_fit_parameters` and
`component_maes` return only the **kept** slots, indexed by
`component_indices = where(_non_zero_components & _component_keep)`.

> A kept component's position in `component_indices` changes whenever another slot is
> masked. Code that holds indices across a masking step must use the slot number `ni`, not
> the position `i`. `collapse_redundant_components` does this.

### 17.5 `<stem>.h5` layout

```
/<target m/z as string>/ms2_long                 written by MzMLParser.to_ms2_h5 (gzip, float32)
/gpf_runs/gpf_<i>/                               written when lazy (or by MSRun.dump_h5)
    attrs: gpf, gpf_low, gpf_high (-1 = None), window_scan_overlap, window_scan_width, index, m_shape
    scan_number, scan_index, bin_indices, retention_time
    m_data, m_indices, m_indptr                  GPFRun._m as CSR parts
    m_long                                       [scan, rt, mz, intensity]
    discretize/mz_discretize/                    attrs + bins
    scan_windows/window_<k>/
        attrs: gpf, gpf_low, gpf_high, is_filtered, is_fit, is_component_fit, m_is_sparse,
               nmf_reconstruction_error, filter/transform flags, is_ms1_features_fit
        scan_number, scan_index, retention_time, mz_masked, mz_unfilter, mz,
        mz_bin_indices_unfilter, mz_bin_indices
        m_data (float16), m_indices, m_indptr    the filtered tile matrix
        w, h, component_keep, non_zero_components
        component_means, component_sigmas, component_maes, component_names
        m_max, m_max_bin_indices
        ms1_features/feature_<j>/                attrs + scan_index, intensity, retention_time,
                                                 interpolated_retention_time, interpolated_intensity
        component_ms1_coef_matrix, component_ms1_global_exp_var, component_ms1_individual_exp_var
```

**Optional layouts** (`[h5]` settings; neither changes any result):

- `compact_storage = True`:
  - Each `window_<k>` leaves out `mz_unfilter`, `mz_bin_indices_unfilter` and `mz_masked`,
    and has the attribute `compact_axes = True` plus the two dtypes.
    `ScanWindow.from_h5(..., gpf_bin_indices, gpf_mz)` rebuilds them from the GPF's
    `bin_indices` (cast to the stored dtypes), with `mz_masked = ~isin(axis,
    mz_bin_indices)`. That's exactly what `filter_mz_axis` maintains.
  - MS1 features go into one `ms1_feature_table` group per window: one dataset per scalar
    field, plus `scan_index`/`intensity` concatenated with `*_offsets`. On load,
    `load_ms1_feature_table` rebuilds each `MS1Feature` and recomputes its interpolated
    profile on the window's retention times.
  - `discretize/mz_discretize/bins` is a hard link to one copy in `/shared_arrays`.
  - Datasets are compressed (`compact_compression`), and new objects use the HDF5 1.10
    format.
- `gpfs_per_file = N`: the `/gpf_runs/gpf_<i>` groups live in
  `<stem>.gpf_runs/part_<i // N>.h5`. The main file's `gpf_files` attribute (JSON) maps each
  index to its part file. `MSRun.gpf_group(i)` resolves every read and write.

---

## 18. Walkthrough of `ffd process_mzml`, step by step

### 18.1 Entry, configuration and logging

*`cli._process_mzml` → `cli.process_mzml`*

1. The `configure_logger` decorator adds a `FileHandler` to the root logger:
   `<input folder>/<YYYYmmdd-HHMMSS>__process_mzml__<stem>.log`. The log goes next to the
   **input** file, not `--output_dir`. `arg_logger` logs the arguments.
2. `output_dir` defaults to the mzML's folder and is created if needed.
3. `Config.from_ini(path)` or `Config()`:
   - `Config()` deep-copies every class-level section dict (`AttributeDict`) into the
     instance, so instances never share settings.
   - `from_ini` parses every value with `ast.literal_eval`. If that fails, the raw string
     is kept, so `bin_width = ppm` works without quotes.
   - Keys are matched case-insensitively to existing keys (`alpha_w` updates `alpha_W`).
     Unknown keys are added to the section as-is. Sections that aren't dict sections
     (e.g. `[general]`) set instance attributes.
   - Unknown keys in `[nmf]` are passed on to scikit-learn's `NMF`, so a typo there fails
     at the NMF step.

### 18.2 Conversion: mzML → long HDF5

*`cli.convert_mzml_h5long` → `MzMLParser.to_ms2_h5(path, h5_fh, config)`*

1. `MzMLParser(source, keep_level=2)` iterates spectra with an XPath that only matches
   `ms level = 2`. It is a non-indexed, non-iterative pyteomics reader.
2. For each spectrum, `parse_scan` reads:
   - `ScanNum = index + 1` (the position in the file, not the vendor scan number);
   - `RetentionTime` = `scan start time` (pyteomics returns the file's unit, usually
     minutes);
   - `MS2TargetMass` = `isolation window target m/z`;
   - `PrecursorDetected` = the first `selected ion m/z`;
   - `lowerMass = PrecursorDetected − lower offset` and
     `higherMass = PrecursorDetected + upper offset`.
3. Non-finite m/z or intensity values are dropped. If `filter_spectra` is set, peaks at or
   below that scan's `filter_spectra_percentile` intensity percentile are dropped. The
   percentile is computed on the finite values only.
4. Each peak becomes a Python list `[target, scan, rt, mz, intensity(, low, high)]`,
   appended to `mz_arrays[str(target)]`.
5. Every 2,500 spectra (`chunk_size`), and at the end, `write_chunks` runs. For each target
   it creates `/<target>/ms2_long` (gzip, float32, resizable) if needed, appends
   `np.vstack(rows)`, and flushes.
6. The HDF5 is opened with mode `"w"`, so **re-running overwrites `<stem>.h5`**, including
   any `gpf_runs` from an earlier lazy run.
7. A spectrum that fails to parse is logged and skipped. The loop continues.

**Cost:** pure-Python work per peak (list building, then `vstack`), plus XML parsing and
zlib decoding. This is often the slowest single stage on large files.
[More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.3 Building the MSRun: discretize, pivot, tile, pre-filter

*`MSRun.from_h5_long(h5, config=config)`*

1. `MzDiscretize.from_config(config)` builds the bin grids (`_build_bins`):
   - **ppm:** for each grid `i`, start at `mz_high − i·tol·mz_high/steps`, then
     repeatedly multiply by `(1 − tol)` until below `mz_low`. Every grid is truncated to the
     shortest grid's length (keeping the top bins), reversed into ascending order, and
     columns whose smallest edge ≤ `mz_low` are dropped. The result `_bins` has shape
     (steps, n_bins). Each value is a bin's **right edge**.
   - **Da:** `n = ceil((mz_high − mz_low)/tol)` edges per grid,
     `mz_low + i·tol/steps + tol·arange(n)`.
2. `MSRun(lazy=config.lazy, h5_fh=<stem>.h5)` opens the same file in `"a"` mode and makes
   sure `/gpf_runs` exists.
3. Isolation targets are the top-level groups that contain `ms2_long`, sorted numerically.
   For each one (index `i`):
   1. **Read** `sub_m = f[target]["ms2_long"][:]`.
   2. **Discretize** (`GPFRun.from_undiscretized_long` → `MzDiscretize.discretize_mz_array`):
      - sort rows by m/z;
      - keep `mz_low < mz < mz_high`;
      - for each grid, `np.searchsorted(bins[i], mz)` (side "left": a value equal to a
        right edge belongs to that bin);
      - keep only rows with `left_edge(first bin) < mz ≤ right_edge(last bin)` for that
        grid;
      - append `bin_index` and `bin_right_edge`, then stack the grids. The output has
        about twice as many rows as the input.
   3. **Pivot** (`GPFRun.from_discretized_long`): the bin column is 7 (5 without bounds).
      `utils.pivot_unique_binned_mz_sparse` runs `np.unique(…, return_inverse=True)` on
      scans and bins, builds a `coo_matrix((intensity, (scan_pos, bin_pos)))` (duplicate
      entries are **summed**), and converts it to csr.
   4. `m_long = np.unique(sub_m[:, 1:5], axis=0)`: one row per original peak (the grid
      copies collapse). It and the matrix are cast to float32 when
      `downcast_intensities = True`. `retention_time` is the RT of each unique scan.
   5. **Tile** (`GPFRun.__init__` → `build_scan_windows`):
      - window starts are `arange(0, n − width, stride)`, plus `n − width` when the stride
        doesn't land there;
      - with `n < width`, no windows are built and a warning is logged;
      - `slice_scan_window(s, s+width)` copies the csr rows, scan numbers, indices and RTs,
        and gives each tile **all** GPF bins and their right-edge m/z (`indices_to_mz`).
   6. **Register** (`MSRun.add_gpf(i, gpf, remove_if_exists=True, config=config)`):
      - set `gpf._msrun`;
      - if `h5.scan_window_filter_during_load` (default True), call `filter_scans()` on
        every tile. This runs **serially in the main process**;
      - in-memory mode stores the object;
      - lazy mode deletes any existing `gpf_runs/gpf_i` group, writes the GPFRun (matrix,
        `m_long`, discretizer, every tile) with `GPFRun.dump_h5`, and stores only the path.

**Cost:** discretization and pivoting are vectorized NumPy. Filtering every tile serially
is significant (Section 18.4). In lazy mode, every tile is also written once.
[More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.4 Filtering one tile (`ScanWindow.filter_scans`)

The matrix starts as csr (150 × all GPF bins). Each step below shrinks or rewrites
`_m`. **Every column removal** goes through `filter_mz_axis(mask)`, which:

- marks the removed bins in `_mz_masked`, found with `searchsorted` on the unfiltered bin
  list;
- subsets `_mz`, `_mz_bin_indices` and `_m` with the same mask.

| # | Method | Exact operation |
|---|---|---|
| 1 | `filter_zero_mz` | keep columns with `_m.sum(axis=0) > 0` |
| 2 | `filter_clip_constant(p=0.3)` | `r_j = mean_t(X_tj / max_t X_tj)`. Keep columns with `r_j ≤ 0.3`. For sparse input this division produces a dense matrix |
| 3 | `filter_percentile_scans` (if `_percentile_filter_scans`) | `thr = percentile(nonzero values, _percentile_filter = 10)`. Set `X[X < thr] = 0`, then `filter_zero_mz`. For sparse input this edits the stored values (`X.data`) and removes the zeros. A boolean-mask assignment on the sparse matrix was used before and made this the slowest step of building the run |
| 4 | `transform_log_scans` (if `_log_scans`, off) | `X = log1p(X)` |
| 5 | `transform_maxscale_scans` (if `_scale_scans`) | `m_j = max_t X_tj`. Store `_m_max = m`, `_m_max_bin_indices = current bins`. Then `X = X · (1/m)` per column |
| 6 | `filter_denoise_scans(2, 2)` (if `_denoise_scans`) | convert to dense; `X = grey_erosion(grey_closing(X, size=(2,1)), size=(2,1))`, along time only; `filter_zero_mz`; convert back to csr |
| 7 | `filter_edge_scans` (if `_filter_edge_scans`) | `X[:5] = 0; X[-5:] = 0` (`_filter_edge_nscans = 5`), then `filter_zero_mz` |
| 8 | downcast (if `_downcast_scans`) | `X = X.astype(float16).astype(float32)` (the float16 step is skipped if `_downcast_h5` is False) |
| 9 | downcast bins (if `_downcast_bins`) | `_mz`, `_mz_unfilter` → float32. Bin indices → int32 |

Notes:

- Scaling (step 5) comes **before** denoising and edge zeroing. After steps 6–7 a column's
  maximum can be below 1.
- `_m_max` keeps the maxima for the bins present at step 5.
  `reverse_transform_maxscale_scans()` looks up the current bins in it with `searchsorted`.
  It returns one scale factor per current column.
- All the flags and sizes (`_percentile_filter`, `p = 0.3`, the structuring sizes, edge
  scans) are **class attributes, not config keys**. They are saved as HDF5 attributes per
  tile, but they don't appear in `.out.ini`.

**Cost:** per tile, small. Over thousands of tiles in the main process, it adds up (it
runs once at load, and again for tuning samples).

### 18.5 Tuning internals

*Runs only if `tuning.run_tune` (default True).*

#### Sampling tiles: `MSRun.get_tuning_windows`

1. `rng = default_rng(config.random_seed)`.
2. With `window_sampling_n` in the config (the default):
   `candidates = Σ_gpf max(len(gpf tiles) − 2·exclude_scan_window_edges, 0)`, read from
   HDF5 without loading when lazy. Then
   `fraction = min(window_sampling_n / candidates, 1)`. With no candidates it warns and
   returns `[]`. Otherwise `fraction = window_sampling_fraction`.
3. For each GPF (loaded with `get_gpf`), `GPFRun.sample_scan_windows`:
   - candidate starts are
     `arange(excl·stride, n − excl·stride, stride)`;
   - each start is kept with probability `fraction` (independent Bernoulli draws, so the
     total is only *about* `window_sampling_n`);
   - **new** tiles are built for the kept starts (`build_scan_windows(assign=False)`) and
     filtered.
4. A tile is used if it has at least one column and exactly `scan_width` rows. Its dense
   `m` is copied into the output list.

#### The search: `tune_hyperparameters_optunasearchcv`

1. **Logging.** The first root `FileHandler` is swapped for a new one at
   `<that log>.tuning_<timestamp>.log`. Without one, it uses `optuna_tuning.tuning_<ts>.log`
   in the current folder. Optuna's own logging is routed through the root logger. A
   `finally` block restores the handlers even if tuning fails.
2. `op = OptimizationParameters.from_config(config)` sets the targets used in scoring
   (below).
3. The estimator is
   `NMFMaskWrapper(splitter_type, mask_fraction=test_fraction, n_components=tuning.n_components, n_splits, l1_ratio, alpha_W, alpha_H)`.
   `mask_signal` and `balance_mask_signal` stay at their defaults (True). There's no
   `max_iter`, `solver` or `random_state`, so the NMF uses `fit_nmf_matrix_custom_init`'s
   defaults (500, `'cd'`, 42), whatever `[nmf]` says.
4. `OptunaSearchReconstructionCV(estimator, optuna_hyperparameter_grid, objective_params, n_iter, random_state=random_seed, n_jobs, optimization_parameters=op).fit(windows)`.

#### Inside `OptunaSearchReconstructionCV.fit`

1. A temporary folder (`tempfile.mkdtemp(prefix="ffd_optuna_")`) holds
   `optuna_journal.log`. `_journal_storage(path)` wraps it in
   `JournalStorage(JournalFileBackend(path, lock_obj=JournalFileOpenLock(path)))`.
2. A `multiprocessing.Manager().Queue()` plus a `QueueListener` on the root handlers
   carries worker log lines back to the main log.
3. A study is created with `directions = ["maximize"] × len(objective_params)` (5 with
   the defaults).
4. `n_jobs` joblib workers each run `optimize_optuna_study`:
   - point a logger at the queue, replacing old queue handlers;
   - re-open the study from the journal path, with
     `NSGAIISampler(seed = random_seed + worker)`;
   - run `n_iter // n_jobs + 1` trials.

   In total that is about `n_iter + n_jobs` trials.
5. **One trial** (`objective(trial)`):
   1. Suggest each parameter: `suggest_float(name, low, high, log=(third item == "log"))`,
      or `suggest_categorical` for string tuples.
   2. `_fit_and_score(clone(base_estimator), X, parameters, extra_scores=_FIT_AND_SCORE)`.
      This calls `set_params` and then `fit(X)` (below), with warnings silenced.
   3. If any result has a `fit_error`, it raises. The `except` stores the error, and a
      placeholder `out` of `[None] × n_splits`, then raises `optuna.TrialPruned`.
   4. Otherwise the results are pickled and base64-encoded into the trial's user attribute
      `out`. That includes the fitted `NMF` objects, so the journal grows with every
      trial. The trial's value is a tuple, one number per objective:
      `score(k, nanmean over results of test_scores[k])`.
6. After the workers finish:
   - the study is reloaded from the journal;
   - trials with an `out` and no `error` are kept;
   - with **0** successes it raises; with fewer than `0.5 × n_iter` it raises; with fewer
     than `n_iter` it warns.

   The `out` lists are unpickled and passed with the trial parameters to scikit-learn's
   private `BaseSearchCV._format_results`. That produces `cv_results`, with
   `mean_test_<objective>`, `rank_test_<objective>`, `split<i>_test_<objective>` and so on.
   `best_trials_` records the Pareto front. The temporary folder is removed.

#### One CV evaluation: `NMFMaskWrapper.fit(X)`

```
splitter = MzBinMaskingSplitter(n_splits, mask_fraction, mask_signal, balance_mask_signal,
                                random_state = nmf_kwargs.get("random_state", 42))
for (train_masks, test_masks) in splitter.split(X):          # n_splits times
    for (m, train_mask, test_mask) in zip(X, …):              # every tuning tile
        m_train = splitter.mask_train_matrix(m.copy(), test_mask)
        W, H, model = fit_nmf_matrix_custom_init(m_train, **nmf_kwargs)
        mse       = splitter.reconstruction_error(m, W@H, test_mask,  model)
        mse_train = splitter.reconstruction_error(m, W@H, train_mask, model)
        append mse, mse_train, -log2(mse_train / mse) and calculate_nmf_summary(W, H) metrics
```

- **Mask splitter.** For each tile:
  - choose each non-zero cell with probability `mask_fraction`;
  - then (balanced) choose each zero cell with probability equal to the fraction of all
    cells already chosen;
  - `mask_train_matrix` sets the chosen cells to 0 in the training copy;
  - `reconstruction_error = nanmean((m[mask] − (WH)[mask])²)`.

  The rng is re-seeded with the same seed for every trial, so **all trials see the same
  masks**. That makes trials directly comparable.
- **Sample splitter.**
  - choose rows (scans) with probability `mask_fraction`, and fit on the other rows;
  - for the held-out rows: `W' = model.transform(m[mask])` and
    `error = nanmean((m[mask] − W'H)²)`. It is the mean of the squared values when the
    model or rows are all zero.

  "Train" error uses the same projection on the training rows.

#### The metrics: `utils.calculate_nmf_summary(W, H)`

With components restricted to those where both `W` column and `H` row sums are non-zero
(when ≥ 2 such components; otherwise all four structural metrics are 0):

| Key (stored as `_…` on the wrapper) | Formula |
|---|---|
| `weight_orthogonality` (from **H**) | $G = HH^\top$; $\tilde G_{ij} = G_{ij}/\lVert G_{i,:}\rVert_2$; value $= \lVert \tilde G - I\rVert_F$ |
| `sample_orthogonality` (from **W**) | same with $G = W^\top W$ |
| `mean_weight_sparsity` | $-\text{mean}_c\,\text{Hoyer}(W_{:,c})$, where $\text{Hoyer}(v) = (\sqrt{n} - \lVert v\rVert_1/\lVert v\rVert_2)/(\sqrt{n}-1)$ |
| `mean_sample_sparsity` | $-\text{mean}_c\,\text{Hoyer}(H_{c,:})$ |
| `nonzero_component_fraction` | fraction of W columns with sum > 0 (all components) |
| `fraction_window_component` | fraction of scans (W rows) with any non-zero component |
| `test_reconstruction_errors` | `mse` above |
| `neglog_ratio_train_test_reconstruction_error` | $-\log_2(\text{mse\_train}/\text{mse})$ |

#### Sign conventions and scoring

`_fit_and_score` turns every metric into a "test score" by **multiplying by −1**. So the
reported value $v$ is minus the raw metric. `OptimizationParameters.score(k, v)` then
applies:

- **for keys with a target** (`weight_orthogonality` and `sample_orthogonality`: target 0;
  `nonzero_component_fraction`: `components_in_window / n_components`; the two sparsities:
  target −1; `fraction_window_component`: `components_in_window · 4 · component_sigma /
  scan_width`):
  $\text{score} = -|v + \text{target}| = -|\text{target} - \text{raw}|$. The sparsity
  targets are written as −1 because the raw values are already negated once, so the
  effective target is Hoyer = 1.
- **for other keys,** `score = v`. So `test_reconstruction_errors` scores $-\text{mse}$,
  and `neglog_ratio…` scores $\log_2(\text{mse\_train}/\text{mse})$. That is ≤ 0 when the
  training error is lower, and highest when train and test errors match.

> **Which fits are scored.** `NMFMaskWrapper.fit` stores `n_splits × len(X)` results,
> split-major (all tiles of split 0, then split 1, …). `_fit_and_score` turns them into one
> result per split, using `nanmean` over that split's tiles for every metric (and the sum
> for `n_test_samples`). It keeps only the first model of each split in the result, so the
> pickled trial results stay small. The trial's objective values are the `nanmean` over
> splits. (Before 2026-10-13, the code read entries `0 … n_splits−1` directly: the first
> `n_splits` tiles of split 0. The test `test_fit_and_score_averages_all_windows` guards
> this.)

#### Picking the winner: `pick_parameters_harmonic_mean`

For each objective $p$, over all trials $t$:
$s_{p,t} = \text{score}(p, \text{mean\_test}_p[t])$, then min-max scaled
$\hat s_{p,t} = (s_{p,t} - \min_t s)/\max_t(s - \min_t s)$. Each trial gets
$HM_t = P / \sum_p 1/(\hat s_{p,t} + 0.001)$, and the trial with the highest $HM_t$ is
picked. If one objective has the same value for every trial, the division gives NaN. The
chosen `{l1_ratio, alpha_W, alpha_H}` overwrite `config.nmf`.

With `save_run`, the whole search object is pickled next to the tuning log (CLI: always).
That includes every trial's fitted models.

#### Randomized search (`optuna = False`)

`RandomizedSearchReconstructionCV` draws `n_iter` settings from `hyperparameter_grid` with
scikit-learn's `ParameterSampler`. With all-list grids, sampling is without replacement and
capped at the grid size: 5 × 5 × 9 = 225 < 500, so it runs all 225 with a warning. Each
setting is evaluated in one joblib task with `_fit_and_score`, and must return exactly
`n_splits` results. The same harmonic-mean pick follows.

**Cost:** every trial fits `n_splits × tiles` NMFs serially inside one worker, so the total
is about `(n_iter + n_jobs) × n_splits × tiles` fits. That is usually **the largest
single cost** of a run. [More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.6 The NMF stage

*`containers.fit_nmf_matrix_msrun(msrun, n_jobs, config)`*

For each GPF index in order: `gpf = msrun.get_gpf(i)` (a reload from HDF5 when lazy),
then `fit_nmf_matrix_gpfrun(gpf, n_jobs, config)`:

1. `joblib.Parallel(n_jobs, pre_dispatch="2*n_jobs")` maps `modify_fit_scanwindow` over
   the tiles. Each tile is pickled to a worker and back. In each worker:
   1. `filter_scans()` if the tile isn't filtered yet.
   2. `fit_nmf_matrix_scanwindow(sw, **config.nmf)` → `fit_nmf_matrix_custom_init(sw.m, …)`:
      - `init = "custom"` if `W_init`/`H_init` are given, else the `init` argument, else
        `"nndsvd"` for `'cd'` or `"nndsvda"` for other solvers;
      - `n_components = min(n_components, max(min(m.shape), 1))`;
      - `NMF(n_components, alpha_W, alpha_H, l1_ratio, max_iter, solver, init, random_state=42, **other [nmf] keys)`.
        scikit-learn minimizes
        $\tfrac12\lVert X-WH\rVert_F^2 + \alpha_W\rho\,m\lVert W\rVert_1 + \tfrac12\alpha_W(1-\rho)\,m\lVert W\rVert_F^2 + \alpha_H\rho\,n\lVert H\rVert_1 + \tfrac12\alpha_H(1-\rho)\,n\lVert H\rVert_F^2$
        with $n$ = 150 rows and $m$ = kept bins (Section 7.3);
      - `W = model.fit_transform(m)`, `H = model.components_`;
      - an empty matrix, or any exception, gives all-zero `W`/`H` and
        `reconstruction_err_ = 1e6`. Exceptions are logged.
   3. `ScanWindow.set_nmf_fit(W, H, model)`:
      - if the tile was already fitted, clear the peak fits, names and MS1 matches (a refit
        replaces everything);
      - store `_w`, `_h` and `_nmf_reconstruction_error`;
      - `_non_zero_components = W.sum(0) != 0`;
      - `_component_keep = _non_zero_components`.
   4. `fit_scanwindow(sw, **config.fitpeaks)`. For each kept slot, call
      `fit_gaussian_elution(rt, W[:, i], mu_bounds, sigma_bounds)`:
      1. `y = (w − min w)/(max w − min w)`. If the profile is empty or constant, it raises.
      2. Moment start values: `μ₀ = Σ t·y / Σ y`,
         `σ₀ = sqrt(Σ y (t−μ₀)² / Σ y)`. Each is clipped to its bounds
         (`mu_bounds`, `sigma_bounds`), so a sharp peak starts at the lower σ bound.
      3. `curve_fit(gaussian_curve, t, y, p0=[μ₀, σ₀], bounds)`, a bounded trust-region
         fit of $\exp(-(t-\mu)^2/2\sigma^2)$.
      4. `mae = mean |gaussian_curve(t, μ, σ) − y|`.

      Any exception logs it and records `μ = σ = mae = 0, keep = False`.
   5. `ScanWindow.set_peak_fits(mus, sigmas, maes, keep)`:
      - write the values into length-k arrays at the non-zero slots;
      - `_component_keep[non-zero] = keep`;
      - **edge filter:** `keep = False` where `rt[5] > μ − 2σ` or `rt[−5] < μ + 2σ`;
      - build `_component_names` for every slot from the scan index nearest μ.
2. `gpf.set_scan_windows(results, persist=False)` replaces the tile list (same length
   check), without writing.
3. `gpf.collapse_redundant_components()` (Section 18.7).
4. `gpf.persist_scan_windows()`. When lazy, each tile is written with
   `ScanWindow.dump_h5(mode="update")`. That rewrites a dataset only if it differs from
   what's stored, which requires reading the stored copy back to compare.

**Cost:** the NMF fits dominate: about `tiles × max_iter` coordinate-descent sweeps over
150 × bins matrices, parallel over tiles. Pickling every tile in and out of workers, and
the per-GPF lazy reload and rewrite, add overhead.
[More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.7 Collapsing duplicates (`GPFRun.collapse_redundant_components`)

```
for sw in tiles:                                    # time order
    others = tiles overlapping sw in scan_index, excluding sw itself
    μ, σ = sw._component_means, sw._component_sigmas         # indexed by slot
    for ol in others:
        union = sorted unique(sw.mz_bin_indices ∪ ol.mz_bin_indices)
        for ni in sw.component_indices:              # kept slots of sw, at this moment
            if not sw._component_keep[ni]: continue
            a = zeros(len(union)); a[pos of sw bins] = sw._h[ni]
            for nj in ol.component_indices:          # kept slots of ol, fixed for this ol
                if not ol._component_keep[nj]: continue
                if |μ_ni − μ_nj| > 3(σ_ni + σ_nj): continue
                s = max(σ_ni, σ_nj)
                x = linspace(min(μ) − 6s, max(μ) + 6s, 10000)
                overlap = Σ min(N(x; μ_ni, σ_ni), N(x; μ_nj, σ_nj)) / Σ N(x; μ_ni, σ_ni)
                if overlap ≤ 0.5: continue
                b = zeros(len(union)); b[pos of ol bins] = ol._h[nj]
                if cosine_distance(a, b) < 1 − 0.75:          # similarity > 0.75
                    if sw.mae[ni] < ol.mae[nj]: ol.mask_component(nj)
                    else: sw.mask_component(ni); break
```

- Thresholds `overlap_threshold = 0.5` and `similarity_threshold = 0.75` are method
  defaults, not config keys.
- Every overlapping pair of tiles is visited from both sides, so each pair of components
  can be compared twice. Masking is permanent, so the second visit only sees survivors.
- Cosines are on the **max-scaled** H rows. A tile's own scaling differs from its
  neighbor's, but the shared fragments of one peptide still line up.
- The whole loop is Python, in the main process, per GPF.

**Cost:** about tiles × 4 overlapping tiles × k² pairs, each with up to a 10,000-point
pdf evaluation and a dense cosine over the union of bins. The pre-filter removes most
pairs. [More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.8 Attaching MS1 features

*Only with `--ms1_features_file`.*

1. The CLI reads the Biosaur2 TSV with `pandas.read_csv(sep="\t")`. `pd.eval` turns the
   `mono_hills_scan_lists` and `mono_hills_intensity_list` strings into lists.
2. `add_ms1_features_msrun(msrun, df, config, n_jobs)` → per GPF →
   `add_ms1_features_gpfrun`. That is a joblib map over tiles, and **each task receives
   the whole feature table**. In each task, `add_ms1_features_scanwindow(w, df)`:
   1. `low, high = w._gpf_low, w._gpf_high`. If they are missing, use `w._gpf ± 1`.
   2. Rows with `rtStart < rt[-1]` and `rtEnd > rt[0]`, i.e. overlapping the tile in time.
   3. Row-wise `apply`: the isotope list
      `[mz + i·1.00336/charge for i in range(nIsotopes) if low ≤ … ≤ high]`. Keep rows with
      a non-empty list.
   4. For each row, `MS1Feature.from_biosaur2_series_with_ms2(row, w.retention_time)`:
      - stores the Biosaur2 fields (`massCalib`, `rtApex`, `intensityApex`,
        `intensitySum`, `charge`, `nIsotopes`, `nScans`, `mz`, `rtStart`, `rtEnd`, the
        hill lists, `scanApex`);
      - `_id` is the DataFrame index;
      - `_retention_time = linspace(rtStart, rtEnd, len(hill))`;
      - `interpolate_onto_ms2(rt)`: the step is `(rtEnd − rtStart)/(len − 1)`, or the
        median MS2 RT step for 1-point hills. It calls
        `np.interp(rt, [rt₀ − step, …hill RTs…, rt_end + step], [0, …intensities…, 0], left=0, right=0)`,
        with all zeros for an empty hill.
   5. `w.set_ms1_features(list)`.
3. `gpf.scan_windows = results` (the setter). When lazy, every tile is written, including
   one HDF5 group per attached feature.

**Cost:** pickling the full feature table into every tile task, plus a row-wise pandas
`apply` per tile. It grows with (tiles × features overlapping in time).
[More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.9 MS1–MS2 matching

*`fit_ms1_ms2_feature_matching_msrun` → per GPF → joblib per tile →
`fit_ms1_ms2_feature_matching_scanwindow(w, alpha, extend_w_fraction, c_cutoff)`*

Inputs: `μ, σ` of the kept components; the MS1 matrix `E` = (scans × features), stacked
from `_interpolated_intensity`; and the tile RTs `t`. For each kept component *i*:

1. `[lo, hi] = norm.interval(0.99, μ, σ)`, widened by `extend_w_fraction·(hi − lo)` on
   each side.
2. `a = max(searchsorted(t, lo, "left") − 1, 0)` and
   `b = min(searchsorted(t, hi, "right"), n)`. The rows `a:b` are used (`b` is exclusive,
   so a region reaching the end of the tile includes the last scan).
3. `y = W[a:b, i]`. If ‖y‖ ≈ 0, a warning is logged, the coefficients are 0, and it
   moves on. Otherwise `y /= ‖y‖`.
4. Candidates: the columns of `E[a:b]` with any signal there. Each is divided by its
   **whole-tile** norm `‖E[:, j]‖` (Section 11.2). With none, the coefficients are 0.
5. `least_squares_with_l1_bounds(X, y, alpha)`:
   - minimizes $\sum (Xc - y)^2 + \alpha \sum |c|$ with $0 \le c \le 1$;
   - starts from $c = 0$;
   - uses SLSQP with `maxiter = 100` and finite-difference gradients.
6. Zero any `c ≤ c_cutoff` (or ≈ 0). Store `c` in row *i* of the coefficient matrix, at
   the candidate columns.
7. `global R² = 1 − Σ(y − Xc)² / Σ(y − ȳ)²`, and the same per candidate with only that
   column. Both use `utils.fraction_explained_variance`.

Then `ScanWindow.set_ms1_ms2_feature_matches(coef, global, individual)` sets
`_is_ms1_features_fit`, and the setter persists the tiles when lazy.

**Cost:** one small SLSQP problem per kept component. With *p* candidates, each iteration
needs about *p* extra objective evaluations for the numerical gradient.
[More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.10 Export to DataFrame and Parquet

*`dump_features_to_df_msrun` → per GPF (`get_gpf`) → `dump_features_to_df_gpfrun` →
per tile → `dump_features_to_df_scanwindow(sw, gpf, config)`. Serial, main process.*

If `match_ms1` is True but the tile wasn't matched, it raises. The CLI sets `match_ms1 =
False` when there's no MS1 file. For each kept component *i*:

| Column | Computation |
|---|---|
| `ms2_component` | `component_names[i]` |
| `ms2_component_rt`, `ms2_component_sigma` | fitted μ, σ |
| `ms2_gpf_center`, `ms2_gpf_low`, `ms2_gpf_high` | tile's isolation target and bounds |
| `ms2_apex_scan_index`, `ms2_apex_scan_number`, `ms2_apex_scan_rt` | the tile scan with RT closest to μ |
| `ms2_component_weight_mz`, `ms2_component_weight_intensity` | the top `top_n_rescaled_peaks` (200) entries of `h[i]` (scaled units), by value. m/z = bin **right edge** (`sw.mz`). Zeros fill the list when fewer bins are non-zero |
| `ms2_component_reweight_mz`, `ms2_component_reweight_intensity` | the same with `h[i] × reverse_transform_maxscale_scans()` (approximate intensities) |
| `ms2_mz_array`, `ms2_intensity_array` | raw peaks of the apex scan: `m_long[m_long[:,0] == apex scan number]`, columns m/z and intensity. Both grid copies were already removed |

With MS1 matching, for each component:

- `thr` = the `top_n_ms1_matches`-th largest coefficient (0 if there are fewer features);
- matches are the features with `coef ≥ thr` **and** `coef ≥ matching_coefficient_threshold`.

The lists `ms1_matching_coef`, `ms1_individual_explained_variance`, `ms1_mass`
(`massCalib`), `ms1_charge`, `ms1_mz`, `ms1_apex_rt`, `ms1_apex_intensity`,
`ms1_sum_intensity` and `ms1_id` are added, plus the scalar
`ms1_global_explained_variance`. `DataFrame.explode` turns them into one row per match. A
component with no match keeps a single row with NaN MS1 fields.

All tiles and GPFs are concatenated. The CLI writes `<output_dir>/<stem>.features.parq`,
then `<output_dir>/<stem>.out.ini` (the config actually used, including tuned values and
`match_ms1`), and closes the HDF5.

**Cost:** for every component, the raw apex spectrum is found by scanning **all** of the
GPF's `m_long` (no index). That is O(components × peaks per GPF).
[More](RUNNING.md#performance-bottlenecks-and-what-to-adjust)

### 18.11 Writing `.ms2` files (`ffd dump_df_to_ms2`)

`utils.feature_df_to_ms2(df, path, {"wide_window", "output_type"})`:

1. The header has the creation date, `ExtractorVersion` (the installed package version, or
   `dev`) and the options.
2. Peak columns by `output_type`:
   - `raw` → `ms2_mz_array`/`ms2_intensity_array`;
   - `intensity_transform_weight` → `…reweight…`;
   - `raw_weight` → `…weight…`.
3. Precursors:
   - **MS1 mode** keeps rows with a non-NaN `ms1_matching_coef`, uses `ms1_mz` and
     `ms1_charge`, and writes `MH+ = ms1_mass + 1.0073`;
   - **Wide mode** keeps one row per component, uses `ms2_gpf_center` with **charge 2**,
     and writes `MH+ = 2·center − 2·1.0073 + 1.0073`.
4. Scans are numbered 1…N in the file order. Each is written as `S`, `Z` and the peak
   lines (`%.5f %.3f`).

---

## 19. Lazy vs in-memory: when data is read and written

| Stage | `lazy = False` | `lazy = True` |
|---|---|---|
| Build (`from_h5_long`) | GPFRuns kept in `MSRun._gpf_runs` | each GPFRun written to `/gpf_runs/gpf_i`; only the path is kept |
| Tuning sample | uses the objects | `get_gpf` reloads each GPF (full read: matrix, `m_long`, every tile) |
| NMF stage | objects replaced by worker results | reload GPF → fit → collapse → **write every tile once** (after the collapse) |
| MS1 attach | objects replaced | reload → attach → write every tile (plus feature groups) |
| MS1 match | objects replaced | reload → match → write every tile |
| Export | reads objects | reload every GPF once more |

Points to know:

- **Each `get_gpf` returns a fresh object.** Changes that aren't written back
  (`scan_windows` setter, `persist_scan_windows`, `update_scan_window_at_index`) are lost.
- `GPFRun.from_h5` calls `GPFRun.__init__`, which builds tiles from the matrix, and then
  replaces them with the tiles loaded from HDF5. The first set is thrown away.
- Tile matrices are stored as float16 (`_downcast_h5`). The float16 round-trip in
  `filter_scans` makes in-memory runs give the same numbers.
- `_m_unfilter` isn't stored by default. After a reload it equals the filtered matrix, so
  `unfiltered_with_filter()` raises a clear error. Set `h5.scan_window_save_m_unfilter`
  to store it. The MSRun passes it, with the other storage options, to every GPFRun it
  writes.
- In-memory runs still open `<stem>.h5` in append mode (it's the conversion file), but write
  nothing to `gpf_runs`.

---

## 20. Parallelism: what runs where, and what gets pickled

| Stage | Unit of parallel work | Pickled to each task | Runs in main process |
|---|---|---|---|
| Conversion | none | — | everything |
| Build / filter at load | none | — | discretize, pivot, tile, `filter_scans` for every tile |
| Tuning (Optuna) | one worker per `n_jobs`, each running ~`n_iter/n_jobs` trials | the `objective` closure, including **all tuning tiles** and the search object | journal reload, `_format_results`, harmonic pick |
| Tuning (randomized) | one task per parameter setting | estimator + all tuning tiles | formatting, pick |
| NMF + Gaussian fits | one task per tile (`pre_dispatch = 2·n_jobs`) | the `ScanWindow` (including `_m_unfilter`, all GPF bins) in and out | collapse, HDF5 writes |
| MS1 attach | one task per tile | the tile **and the whole Biosaur2 DataFrame** | HDF5 writes |
| MS1 match | one task per tile | the tile with its features | HDF5 writes |
| Export | none | — | everything |

joblib's default backend (loky) uses separate processes, so on Windows and macOS the
script that calls the API must use `if __name__ == "__main__":`. NumPy/BLAS inside each
worker may start its own threads. Limiting them (e.g. `OMP_NUM_THREADS=1`) avoids
oversubscription when `n_jobs` is close to the core count.

---

## 21. Config keys and the code that reads them

| Key | Read by | Effect |
|---|---|---|
| `[general] random_seed` | `MSRun.get_tuning_windows`, `tune_hyperparameters_*` (`random_state`, sampler seeds) | tuning tile sample and Optuna/randomized sampling. NMF uses its own fixed `random_state = 42` |
| `[general] tqdm_enabled` | every `*_msrun` stage, `from_h5_long`, converter | progress bars |
| `[general] lazy` | `MSRun.from_h5_long`, `from_long`, `from_h5` | Section 19 |
| `[general] downcast_intensities` | `GPFRun.from_discretized_long` | float32 matrix and `m_long` |
| `[h5] scan_window_filter_during_load` | `MSRun.add_gpf` | filter every tile at load |
| `[h5] scan_window_save_m_unfilter` | `StorageOptions.from_config` → `GPFRun.dump_h5` | store each tile's unfiltered matrix |
| `[h5] compact_storage`, `compact_compression` | `StorageOptions.from_config` → `GPFRun.dump_h5`, `ScanWindow.dump_h5`, `MzDiscretize.dump_h5`; `MSRun._file_kwargs` | compact layout (Section 17.5) |
| `[h5] gpfs_per_file` | `MSRun.__init__`, `MSRun.add_gpf`, `MSRun.gpf_group` | split the lazy run into part files (Section 17.5) |
| `[feature_output] explode_ms1_matches` | `dump_features_to_df_scanwindow`; `utils.explode_ms1_matches` | one row per match, or one row per component with MS1 matches as arrays |
| `[discretization] *` | `MzDiscretize.from_config` | bin grids (Section 4) |
| `[ms2_preprocessing] include_gpf_bounds` | `MzMLParser.to_ms2_h5`, `GPFRun.from_discretized_long` | stores and uses isolation bounds |
| `[ms2_preprocessing] filter_spectra(_percentile)` | `MzMLParser.to_ms2_h5` | per-scan intensity cut at conversion |
| `[scan_filter] scan_width, scan_overlap` | `GPFRun.from_discretized_long` → `GPFRun`; `OptimizationParameters.from_config`; `get_tuning_windows` (row check) | tile length and stride |
| `[tuning] run_tune, optuna` | `cli.process_mzml` | whether and how to tune |
| `[tuning] window_sampling_n / _fraction, exclude_scan_window_edges` | `MSRun.get_tuning_windows` | which tiles are sampled |
| `[tuning] optuna_hyperparameter_grid / hyperparameter_grid, n_iter, objective_params` | `tune_hyperparameters_*` | search space, budget, objectives |
| `[tuning] n_components, n_splits, test_fraction, splitter_type` | `tune_hyperparameters_*` → `NMFMaskWrapper` | CV design |
| `[tuning] components_in_window, component_sigma` (+ `n_components`, `scan_width`) | `OptimizationParameters.from_config` | objective targets |
| `[nmf] *` | `fit_nmf_matrix_gpfrun` → `fit_nmf_matrix_custom_init` → `sklearn NMF` | final factorization (extra keys are passed through). In tuning only `alpha_W`, `alpha_H` and `l1_ratio` are used, as start values |
| `[fitpeaks] mu_bounds, sigma_bounds` | `fit_scanwindow` → `fit_gaussian_elution` | Gaussian fit bounds |
| `[feature_matching] match_ms1` | `dump_features_to_df_scanwindow` (set by the CLI) | export MS1 columns |
| `[feature_matching] alpha, c_cutoff, extend_w_fraction` | `fit_ms1_ms2_feature_matching_scanwindow` | MS1 regression |
| `[feature_output] *` | `dump_features_to_df_scanwindow` | peaks per spectrum, MS1 match filtering |
| `Constants.isotope_mu` (1.00336), `proton_mu` (1.0073) | `add_ms1_features_scanwindow`; `feature_df_to_ms2` | not configurable from the INI |

Not configurable (class attributes or defaults): all `ScanWindow` filter settings (Section
18.4), the collapse thresholds (0.5 and 0.75), the NMF `random_state` (42), the SLSQP
`maxiter` (100), the conversion chunk size (2,500), and tuning's NMF `max_iter`/`solver`.

---

## 22. The Python API in the same order

The CLI is a thin layer. The same run, step by step:

```python
from fragment_feature_detection.config import Config
from fragment_feature_detection.converter import MzMLParser
from fragment_feature_detection.containers.msrun import MSRun
from fragment_feature_detection.containers import (
    fit_nmf_matrix_msrun, add_ms1_features_msrun,
    fit_ms1_ms2_feature_matching_msrun, dump_features_to_df_msrun,
)
from fragment_feature_detection.decomposition import tune_hyperparameters_optunasearchcv

if __name__ == "__main__":
    config = Config.from_ini("run.ini")
    MzMLParser.to_ms2_h5("sample.mzML", h5_fh="sample.h5", config=config)   # 18.2
    msrun = MSRun.from_h5_long("sample.h5", config=config)                  # 18.3–18.4
    windows = msrun.get_tuning_windows(config=config)                       # 18.5
    config.nmf.update(tune_hyperparameters_optunasearchcv(windows, config=config, n_jobs=8))
    fit_nmf_matrix_msrun(msrun, n_jobs=8, config=config)                    # 18.6–18.7
    # with MS1 features (18.8–18.9):
    #   add_ms1_features_msrun(msrun, ms1_df, config=config, n_jobs=8)
    #   fit_ms1_ms2_feature_matching_msrun(msrun, n_jobs=8, config=config)
    config.feature_matching.match_ms1 = False                               # if no MS1
    dump_features_to_df_msrun(msrun, config=config).to_parquet("sample.features.parq")  # 18.10
    msrun.close()
```

Inspecting one tile after the NMF stage:

```python
sw = msrun.get_gpf(0).scan_windows[10]
sw.w, sw.h                       # kept elution profiles (150 × k) and spectra (k × bins)
sw.component_fit_parameters      # (μ, σ) of kept components
sw.mz, sw.mz_bin_indices         # labels of sw.h columns (bin right edges, global bin ids)
sw.reverse_transform_maxscale_scans()   # factors to turn h back into intensities
```

---

## 23. Code behaviors worth knowing

These follow from the code as it stands. None of them is a crash, but each one affects
results or cost. Rows marked **(new)** were found while writing this walkthrough and are
also listed in [SUGGESTIONS.md § A23–A30](SUGGESTIONS.md#new-findings).

Fixed while this walkthrough was written (2026-10-13): tuning scored only the first
`n_splits` fits; the Gaussian fit started sharp peaks at the *upper* σ bound (it now clips
to the nearest bound); reloading a saved in-memory run (`MSRun.from_h5` with
`lazy = False`) failed to open the file and would have filtered the windows a second time;
MS1 matching left out the tile's last scan; and the sparse percentile filter was slow.

| Behavior | Where | Consequence |
|---|---|---|
| Tuning ignores `[nmf] max_iter`/`solver` **(new)** | `tune_hyperparameters_*` | tuned values come from 500-iteration `'cd'` fits even if the final run differs |
| The trial `out` attribute stores pickled NMF models (one per split) **(new)** | `OptunaSearchReconstructionCV` objective | the journal and the saved `.pkl` grow with every trial |
| Every tile is filtered serially at load **(new)** | `MSRun.add_gpf` | a single-core phase before any parallel work |
| `GPFRun.from_h5` builds tiles twice **(new)** | `GPFRun.__init__` + `from_h5` | extra work on every lazy reload |
| Exported weight spectra include both grid copies, right-edge m/z, and zero padding | `dump_features_to_df_scanwindow` | see SUGGESTIONS C2 |
| The MS1 hill RT axis is evenly spaced | `MS1Feature.__init__` | see SUGGESTIONS A18 |
| The MS2 target is normalized over the region, MS1 candidates over the whole tile | `fit_ms1_ms2_feature_matching_scanwindow` | see SUGGESTIONS A22 |
| `ScanWindow` filter settings and collapse thresholds aren't in `Config` | class attributes / method defaults | runs can't be fully described by `.out.ini` |
