"""Run both manuscript examples against the installed package."""
import json
from pathlib import Path
import numpy as np
from rapid_audit import audit, block_update

def extrusion():
    fixture = json.loads(Path(__file__).with_name("extrusion_design.json").read_text())
    counts = fixture["counts"]
    Z = np.repeat(np.eye(2), counts, axis=0)
    X = Z @ np.array(fixture["diameters_mm"])
    c = np.array([1, -1, 0, 0, 0, 0, 0, 0])
    return [audit(X, Z, c), audit(X, np.ones((len(X), 1)), c)]

def synthetic():
    c = np.array([1., -1., 0.])
    x = np.zeros((1, 3))
    basis = audit(x, np.ones((1, 1)), c)["basis"]
    return [(audit(np.vstack([x, v]), np.ones((2, 1)), c),
             block_update(basis, x[0], v, c))
            for v in [[0, 0, 0], [0, 0, 1], [1, -1, 0]]]

if __name__ == "__main__":
    for name, result in zip(["campaign", "intercept", "A", "B", "C"],
                            extrusion() + [r for r, _ in synthetic()]):
        print(name, "rank", result["rank"], "estimable", result["estimable"],
              "distance", f'{result["distance"]:.6g}', "residuals", result["res"])
