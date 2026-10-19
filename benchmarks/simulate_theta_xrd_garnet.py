"""
Simulation: can the Ramanujan-theta lattice fit tell garnets apart from XRD?

Raman confuses the garnet solid-solution series (pyrope, almandine,
spessartine, grossular, andradite). Their cell edges differ, so a powder
XRD pattern should separate them. This script tests the CODE PATH before
we ask the team to download measured data:

  * "measured" patterns are generated independently of the theta series:
    brute-force (h,k,l) enumeration with the Ia-3d reflection conditions,
    Cu K-alpha1 + K-alpha2 doublet, random line intensities, random zero
    shift, noise, 10-80 deg 2-theta;
  * the cell edge is then recovered by fit_lattice() using Ramanujan-theta
    line positions only.

What this does NOT test: real instrument effects beyond those simulated,
preferred orientation, impurity phases. That needs measured RRUFF patterns.

Cell edges: Novak & Gibbs (1971), Am. Mineral. 56, 791.

    python benchmarks/simulate_theta_xrd_garnet.py
"""

import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.theta_xrd.fit import ia3d_allowed, fd3m_allowed, diamond_allowed, measured_peaks, fit_lattice

KA1, KA2 = 1.5406, 1.5444
GARNETS = {"pyrope": 11.459, "almandine": 11.531, "spessartine": 11.612,
           "grossular": 11.815, "andradite": 12.058}


def simulate(a, rule, rng, tt=np.arange(10, 80, 0.02), fwhm=0.10, noise=0.01, zero=0.0):
    """Independent forward model: explicit (h,k,l) loop, Bragg's law, K-alpha doublet."""
    y = np.zeros_like(tt)
    sig = fwhm / 2.3548
    shells = {}
    r = int(2 * a * np.sin(np.radians(tt.max() / 2)) / KA1) + 1
    for h in range(-r, r + 1):
        for k in range(-r, r + 1):
            for l in range(-r, r + 1):
                N = h * h + k * k + l * l
                if N and rule(h, k, l):
                    shells[N] = shells.get(N, 0) + 1
    for N, mult in shells.items():
        inten = mult * rng.lognormal(0, 1.0)
        for lam, frac in ((KA1, 1.0), (KA2, 0.5)):
            s = lam * np.sqrt(N) / (2 * a)
            if s >= 1:
                continue
            t = np.degrees(2 * np.arcsin(s)) + zero
            y += frac * inten * np.exp(-0.5 * ((tt - t) / sig) ** 2)
    y /= y.max()
    return tt, np.maximum(y + noise * rng.standard_normal(len(tt)), 0)


def main():
    rng = np.random.default_rng(42)
    trials = 20
    print("1) Garnet end-members: cell edge recovered by the Ramanujan-theta fit")
    print(f"   {'mineral':<12}{'a true (A)':>11}{'mean |err| (A)':>16}{'max |err| (A)':>15}"
          f"{'matched peaks':>15}{'classified':>12}")
    total_ok = 0
    for name, a_true in GARNETS.items():
        errs, matched, ok = [], [], 0
        for _ in range(trials):
            tt, y = simulate(a_true, ia3d_allowed, rng, zero=rng.uniform(-0.05, 0.05))
            pk, h = measured_peaks(tt, y)
            fit = fit_lattice(pk, "Ia-3d", (11.3, 12.2), weights=h)
            errs.append(abs(fit.a - a_true))
            matched.append(fit.matched / fit.n_measured)
            guess = min(GARNETS, key=lambda m: abs(GARNETS[m] - fit.a))
            ok += guess == name
        total_ok += ok
        print(f"   {name:<12}{a_true:>11.3f}{np.mean(errs):>16.4f}{np.max(errs):>15.4f}"
              f"{np.mean(matched):>15.0%}{ok:>9}/{trials}")
    print(f"   overall classification: {total_ok}/{trials * len(GARNETS)}")

    print("\n2) Solid solution pyrope-almandine (Vegard's law, linear in a)")
    a_py, a_al = GARNETS["pyrope"], GARNETS["almandine"]
    for x_true in (0.25, 0.50, 0.75):
        a_true = a_py + x_true * (a_al - a_py)
        est = []
        for _ in range(trials):
            tt, y = simulate(a_true, ia3d_allowed, rng, zero=rng.uniform(-0.05, 0.05))
            pk, h = measured_peaks(tt, y)
            fit = fit_lattice(pk, "Ia-3d", (11.3, 12.2), weights=h)
            est.append((fit.a - a_py) / (a_al - a_py))
        print(f"   almandine fraction true {x_true:.2f} -> estimated {np.mean(est):.3f} "
              f"(+/- {np.std(est):.3f})")

    print("\n3) Other cubic space groups")
    for name, a_true, sg, rule, rng_a in (("spinel MgAl2O4", 8.083, "Fd-3m", fd3m_allowed, (7.9, 8.3)),
                                          ("diamond", 3.567, "Fd-3m (diamond, 8a)", diamond_allowed, (3.4, 3.7))):
        errs = []
        for _ in range(trials):
            tt, y = simulate(a_true, rule, rng, tt=np.arange(10, 140, 0.02), zero=rng.uniform(-0.05, 0.05))
            pk, h = measured_peaks(tt, y)
            errs.append(abs(fit_lattice(pk, sg, rng_a, weights=h).a - a_true))
        print(f"   {name:<16} a = {a_true:.3f} A: mean |err| {np.mean(errs):.4f} A, max {np.max(errs):.4f} A")
    print("\nSimulated patterns only: this validates the fitting code, not the physics.")


if __name__ == "__main__":
    main()
