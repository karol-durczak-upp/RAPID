# Submission packaging verification

Verified on 2026-11-24 in a newly created Linux virtual environment.
The wheel was installed with no network access, using a separately downloaded
NumPy 2.3.5 wheel. The virtual environment did not inherit system packages.

- Package: rapid-wear-audit 0.1.0; Python import: rapid_audit.
- Four original unittest methods passed.
- Synthetic Plans A/B/C reproduced ranks 0/1/1 and decisions no/no/yes.
- Extrusion reconstruction reproduced ranks 0/1, non-estimable under both
  nuisance specifications; campaign residual 4.9960e-15 and common-intercept
  residual 2.8422e-14.
- Precision example passed analytical weights, independent original-constraint
  least squares, variance-factor checks and the 200,000-draw Monte Carlo check.
- The audit/block_update implementation is unchanged from the previously
  verified release candidate. Changes concern packaging and documentation.
- The original benchmarks/results.json is preserved; no new runtime or memory
  benchmark is claimed. Precision post-processing is not part of that benchmark.
- No cross-platform verification or code-coverage percentage is claimed.

The source archive and wheel include the MIT license. Software citation metadata
is provided in CITATION.cff. No public repository URL or DOI is fabricated.
