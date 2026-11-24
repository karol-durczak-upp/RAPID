"""Functional comparison and Linux benchmark for RAPID 0.1.0."""
import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter_ns

# Set before importing NumPy, including in each fresh worker.
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from rapid_audit import audit, block_update

TAU = 1e-12
SIZES = [(1000, 100, 4), (2500, 250, 8), (5000, 500, 8)]


def reference(X, Z, c):
    """Manual baseline: least-squares residualization, then thin SVD."""
    coef = np.linalg.lstsq(Z, X, rcond=TAU)[0]
    s, vt = np.linalg.svd(X - Z @ coef, full_matrices=False)[1:]
    keep = s > TAU * np.linalg.norm(X, 2)
    R = vt[keep]
    distance = np.linalg.norm(c - R.T @ (R @ c)) / np.linalg.norm(c)
    return int(sum(keep)), bool(distance <= TAU), float(distance)


def functional():
    c = np.array([1., -1., 0.])
    Z = np.ones((2, 1))
    rows = []
    for label, new, expected in zip("ABC", ([0, 0, 0], [0, 0, 1], [1, -1, 0]),
                                    ((0, False), (1, False), (1, True))):
        X = np.array([[0, 0, 0], new], dtype=float)
        rank, decision, distance = reference(X, Z, c)
        result = audit(X, Z, c)
        assert (rank, decision) == (result["rank"], result["estimable"]) == expected
        np.testing.assert_allclose(distance, result["distance"], atol=1e-12)
        if decision:
            w = result["w"]
            np.testing.assert_allclose(X.T @ w, c, atol=1e-12)
            np.testing.assert_allclose(Z.T @ w, 0, atol=1e-12)
            cert = {"w": w.tolist()}
        else:
            h, g = result["h"], result["g"]
            np.testing.assert_allclose(X @ h + Z @ g, 0, atol=1e-12)
            np.testing.assert_allclose(c @ h, 1, atol=1e-12)
            cert = {"h": h.tolist(), "g": g.tolist()}
        rows.append(dict(plan=label, rank=rank, svd_estimable=decision,
                         rapid_estimable=bool(result["estimable"]),
                         distance=float(result["distance"]), certificate=cert,
                         residuals=list(result["res"])))
    return rows


def fixture(n, p, q):
    # Dimensions n,p,q are those of the original design.
    rng = np.random.default_rng(20261112)
    X = np.column_stack((rng.standard_normal((n, p - 1)), np.zeros(n)))
    Z = np.eye(q)[np.arange(n) % q]
    c = np.eye(1, p, p - 1).ravel()
    new = rng.standard_normal(p)
    mean = X[::q].mean(axis=0)
    return X, Z, c, new, mean


def peak_mib():
    return next(int(line.split()[1]) / 1024 for line in
                Path("/proc/self/status").read_text().splitlines()
                if line.startswith("VmHWM:"))


def worker(method, n, p, q, repeat):
    X, Z, c, new, mean = fixture(n, p, q)
    if method == "update":
        # Known exact row space of this fixture; checked against the original audit
        # by the driver. Avoid including a full audit in peak RSS of the shortcut.
        basis = np.eye(p)[:, :-1].copy()
        fn = lambda: block_update(basis, mean, new, c)
    else:
        # Time audit on prepared inputs, excluding array generation and stacking.
        X = np.vstack((X, new))
        Z = np.vstack((Z, np.eye(q)[0]))
        fn = lambda: audit(X, Z, c)
    for _ in range(2):
        fn()
    samples = []
    for _ in range(repeat):
        start = perf_counter_ns()
        result = fn()
        samples.append((perf_counter_ns() - start) / 1e6)
    if method == "update":
        rank, distance = result
    else:
        rank, distance = result["rank"], result["distance"]
    assert rank == p and distance <= TAU
    return dict(method=method, n_base=n, p=p, q=q, rank=rank,
                distance=float(distance), samples_ms=samples,
                median_ms=statistics.median(samples),
                process_peak_mib=peak_mib())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", nargs=4, metavar=("METHOD", "N", "P", "Q"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if platform.system() != "Linux":
        parser.error("This benchmark reads Linux VmHWM.")
    if args.worker:
        method, n, p, q = args.worker
        print(json.dumps(worker(method, *map(int, (n, p, q)), args.repeat)))
        return
    cases = functional()
    results = []
    for n, p, q in SIZES:
        X, Z, c, new, mean = fixture(n, p, q)
        original = audit(X, Z, c)
        assert original["rank"] == p - 1 and not original["estimable"]
        actual_rank, actual_distance = block_update(original["basis"], mean, new, c)
        assert actual_rank == p and actual_distance <= TAU
        # Separate processes: VmHWM is a process high-water mark.
        for method in ("full", "update"):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", method,
                   str(n), str(p), str(q), "--repeat", str(args.repeat)]
            results.append(json.loads(subprocess.check_output(cmd, text=True)))
    from io import StringIO
    from contextlib import redirect_stdout
    info = StringIO()
    with redirect_stdout(info):
        np.show_config()
    cpu = next((line.split(":", 1)[1].strip() for line in
                Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), "unknown")
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  rapid=importlib.metadata.version("rapid-wear-audit"),
                  platform=platform.platform(), cpu=cpu, blas_threads_requested=1,
                  numpy_config=info.getvalue(), warmups=2, repeat=args.repeat,
                  functional=cases, benchmark=results)
    payload = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
