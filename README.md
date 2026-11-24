# RAPID for Identifiability Audits in Machinery Wear Diagnostics

Karol Durczak, Jan Bocianowski, Maciej Zaborowicz (provisional author list)

## Functional comparison and performance benchmark

Run on Linux with the RAPID 0.1.0 candidate installed. For example, from its extracted source root, run `python -m pip install .`, then use the same interpreter to run:

    python /path/to/benchmark.py --out results.json

No input files or random seeds need to be supplied. `results.json` contains individual timings, numerical decisions, certificates, residuals, and environment metadata. The bundled results were obtained with Python 3.12.14 and NumPy 2.3.5 on an AMD EPYC 9V74.

### Functional comparison

The rank-only baseline reports the number of retained singular directions after nuisance residualization. The reference contrast test uses least-squares residualization and thin SVD to test membership in the retained row space. It uses the same thresholds as RAPID; it is a reference implementation of the same criterion, rather than an independent estimator or competing software package. RAPID additionally returns certificates checked by substitution into the original model.

| Plan | Adjusted rank | Reference contrast test | RAPID | Verified certificate |
|---|---:|---|---|---|
| A | 0 | Not estimable | Not estimable | h=(0.5,-0.5,0), g=0 |
| B | 1 | Not estimable | Not estimable | h=(0.5,-0.5,0), g=0 |
| C | 1 | Estimable | Estimable | w=(-1,1) |

The published StrucID application is a separate scope comparison, not a runtime competitor in this experiment. It uses output sensitivities of dynamic state-space models to analyse structural properties and presents singular values and null-space vectors. RAPID takes a static linear measurement design and a specified contrast. Source: Stigter and Joubert (2021), https://doi.org/10.1016/j.ifacol.2021.08.348; public documentation: https://github.com/jdstigter/StrucID. No feature-absence claims or performance comparisons with StrucID are made.

### Performance protocol

The synthetic base designs contain p−1 seeded Gaussian predictor columns, one zero column and q block indicators. The target selects the zero column. One new observation within the first block resolves that contrast. The driver verifies the original rank p−1, the original negative decision, and the positive decision with the shortcut when using the basis returned by the full original audit. In timed shortcut workers the same subspace is represented by its known coordinate basis, avoiding prior SVD allocations in the memory measurement.

Each method/size runs in its own process with one BLAS thread requested, two warm-ups and nine timed calls by default. The full call receives already assembled inputs with n+1 rows, returns the estimability certificate and residuals, and includes all internal SVD computations. The shortcut receives a prepared basis, block mean and new row and returns rank and distance only. Input generation, stacking, basis acquisition, and process startup are excluded from elapsed time. Linux VmHWM gives the lifetime process resident-memory high-water mark, including interpreter, arrays, setup and warm-up; it is not the incremental allocation of a function call. Reported time is the median of the timed calls. The benchmark uses one workload and environment; it does not establish speed on other platforms or for all design structures.
