"""
Figure: Ramanujan's phi(q)^3 coefficients = cubic XRD line multiplicities.

    python benchmarks/demo_theta_xrd.py   -> benchmarks/out/theta_xrd.png
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.theta_xrd.ramanujan_xrd import theta_cubic, cubic_pattern


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = 32
    coef = theta_cubic("P", M)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5))

    N = np.arange(1, M + 1)
    colors = ["#d1495b" if coef[n] == 0 else "#2a6fdb" for n in N]
    a1.bar(N, coef[1:], color=colors)
    for n in N:
        if coef[n] == 0:
            a1.annotate("no line", (n, 1), ha="center", fontsize=8, color="#d1495b", rotation=90)
    a1.set_title(r"Coefficients of Ramanujan's $\varphi(q)^3$ = number of (h,k,l) with $h^2+k^2+l^2=N$")
    a1.set_xlabel("N")
    a1.set_ylabel("multiplicity")

    lines = cubic_pattern(5.6402, "F", two_theta_max=90)
    for l in lines:
        a2.vlines(l.two_theta, 0, l.multiplicity, color="#2a6fdb", lw=2)
        a2.annotate(l.label, (l.two_theta, l.multiplicity), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    a2.set_title(r"NaCl (face-centred) predicted from $\varphi(q^4)^3 + 8q^3\psi(q^8)^3$, Cu K$\alpha_1$")
    a2.set_xlabel(r"2$\theta$ (deg)")
    a2.set_ylabel("multiplicity")
    fig.tight_layout()
    out = os.path.join(ROOT, "benchmarks", "out")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "theta_xrd.png")
    fig.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    main()
