# RAPID for Contrast Estimability Audits in Linear Measurement Designs

Karol Durczak, Jan Bocianowski, Maciej Zaborowicz.

RAPID 0.1.0 audits a specified contrast in a linear measurement design after
accounting for nuisance effects. It reports adjusted rank, singular values,
relative contrast distance and a constructive certificate with verification
residuals. A separate function screens a single candidate observation added
to an existing block. Software and the original examples are distributed under
the MIT license; see LICENSE. No raw sensor responses are redistributed.

## Installation

Python 3.10+ and NumPy 1.24+ are required. From this directory:

```sh
python -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python examples/run.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python examples/precision_example.py --out precision-rerun.json
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.
The distribution also includes a wheel and source archive in `dist/`.
For a local wheel installation:

```sh
python -m pip install dist/rapid_wear_audit-0.1.0-py3-none-any.whl
```

NumPy must already be available or downloadable by pip. Reproduce the tested
environment using `requirements-reproduce.txt`. The source archive includes
all examples, fixtures, test files and benchmark scripts.

## Public API

```python
from rapid_audit import audit, block_update
result = audit([[0, 0], [1, -1]], [[1], [1]], [1, -1])
print(result["estimable"], result["w"], result["res"])
```

`audit(X, Z, c, rtol=1e-12)` expects finite arrays with shapes `(n,p)`,
`(n,q)` and `(p,)`, with a nonzero contrast. Use an `(n,0)` nuisance array
when there are no nuisance terms. Responses are not required. The returned
fields are `rank`, `distance`, `basis`, `singular_values`, `rtol`, `estimable`
and `res`; a positive decision adds `w`, a negative decision adds `h` and `g`.

The estimability certificate satisfies `X.T @ w = c` and `Z.T @ w = 0`.
The non-estimability witness satisfies `X @ h + Z @ g = 0` and `c @ h = 1`.
Residuals must be inspected alongside the decision. The returned basis has
shape `(p,r)` for adjusted rank `r`.

`block_update(basis, mean, new, c, rtol=1e-12)` returns updated rank and contrast
distance. Inputs must be a valid orthonormal baseline basis, the predictor
mean of an existing block, one candidate predictor row and the same target.
It assumes nuisance block indicators only. It does not return a certificate
or a precision measure. Candidates are screened independently against the
supplied baseline. Run a full audit on each selected augmented design.

Predictors are not rescaled automatically. Rank uses the spectral norm of X
before nuisance projection. Compare physically identical contrasts and common
units; inspect nearby tolerances for borderline numerical results. The update
and full audit can differ near a cutoff because their scaling rules differ.

## Conditional precision

For a contrast classified as estimable, under independent, equal-variance
observation errors, the norm of `result["w"]` is the noise gain. Its square,
multiplied by the observation-noise variance, is the contrast-estimator variance.
No absolute precision is inferred without a justified noise variance. For a
general covariance matrix, the existing weights need not be variance-optimal.
No precision is assigned to a non-estimable target. These diagnostics do not
establish physical model validity or causality.

## Reproduction and provenance

`examples/run.py` reproduces Plans A/B/C (ranks 0/1/1; decisions no/no/yes)
and the extrusion reconstruction (ranks 0/1; non-estimable under both nuisance
specifications). Geometry and campaign counts in `examples/extrusion_design.json`
are derived from the published design in:
Durczak K, Witaszek K, Zaborowicz M. Applied Sciences 2026;16:10473.
https://doi.org/10.3390/app162110473
The fixture does not include raw thermal or electrical responses.

`examples/precision_example.py` uses the unchanged audit API to verify scaled
Plan C analytically, against an independent solution of the original constraints,
and using 200,000 Gaussian error pairs per scale. Recorded output is
`examples/precision_results.json`; use a different output filename when rerunning.
The three scales have equal rank and estimability, but different noise gains.

```sh
python benchmarks/benchmark.py --out benchmark-rerun.json
```

`benchmarks/results.json` preserves the original manuscript timing measurements;
they were not regenerated during submission packaging. See benchmarks/README.md.
The precision post-processing is not included in those timing measurements.
See VERIFICATION.md for the new packaging checks. Linux was tested; Windows and
macOS have not been verified. No code-coverage claim is made.

## Citation and support

CITATION.cff identifies version 0.1.0 and the authors. No repository URL or DOI
has been assigned by this local build. Add the actual public release URL when
publishing; do not invent an identifier. Questions: karol.durczak@up.poznan.pl.
