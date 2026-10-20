"""
Simulation check of the tetragonal / hexagonal Ramanujan-theta (a, c) refinement.
Patterns are generated independently (explicit h,k,l loop, Cu Ka1+Ka2, random
intensities, 30 % of lines randomly removed to mimic space-group absences,
zero shift, noise). The fit starts from a nominal cell offset by up to 2 %.

    python benchmarks/simulate_theta_xrd_uniaxial.py
"""
import os, sys
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.theta_xrd.fit import measured_peaks
from src.theta_xrd.fit_uniaxial import fit_uniaxial

KA1, KA2 = 1.5406, 1.5444
# approximate literature cells, used only to generate test patterns
CASES = [("zircon", "tetragonal", "I", 6.607, 5.982), ("rutile", "tetragonal", "P", 4.594, 2.959),
         ("quartz", "hexagonal", "P", 4.913, 5.405), ("calcite", "hexagonal", "R", 4.990, 17.06),
         ("corundum", "hexagonal", "R", 4.759, 12.99)]

def simulate(system, cent, a, c, rng, tt=np.arange(10, 90, 0.02)):
    y = np.zeros_like(tt); sig = 0.10 / 2.3548; z = rng.uniform(-0.05, 0.05)
    seen = {}
    r = 12
    for h in range(-r, r + 1):
        for k in range(-r, r + 1):
            for l in range(-20, 21):
                if h == k == l == 0: continue
                if cent == "I" and (h + k + l) % 2: continue
                if cent == "R" and (-h + k + l) % 3: continue
                A = h*h + k*k if system == "tetragonal" else h*h + h*k + k*k
                inv = A / a**2 + l*l / c**2 if system == "tetragonal" else 4*A / (3*a**2) + l*l / c**2
                key = round(inv, 9)
                seen[key] = seen.get(key, 0) + 1
    for inv, mult in seen.items():
        if rng.random() < 0.3: continue
        I = mult * rng.lognormal(0, 1)
        for lam, f in ((KA1, 1.0), (KA2, 0.5)):
            s = lam * np.sqrt(inv) / 2
            if s < 1:
                t = np.degrees(2 * np.arcsin(s)) + z
                y += f * I * np.exp(-0.5 * ((tt - t) / sig) ** 2)
    y /= y.max()
    return tt, np.maximum(y + 0.01 * rng.standard_normal(len(tt)), 0)

def main():
    rng = np.random.default_rng(7)
    print(f"{'mineral':<10}{'a err (A)':>11}{'c err (A)':>11}{'explained':>11}")
    for name, system, cent, a, c in CASES:
        ea, ec, ex = [], [], []
        for _ in range(10):
            tt, y = simulate(system, cent, a, c, rng)
            pk, h = measured_peaks(tt, y)
            a0, c0 = a * (1 + rng.uniform(-0.02, 0.02)), c * (1 + rng.uniform(-0.02, 0.02))
            f = fit_uniaxial(pk, system, cent, a0, c0, weights=h)
            ea.append(abs(f.a - a)); ec.append(abs(f.c - c)); ex.append(f.matched / f.n_measured)
        print(f"{name:<10}{np.mean(ea):>11.4f}{np.mean(ec):>11.4f}{np.mean(ex):>11.0%}   (max a {np.max(ea):.4f}, c {np.max(ec):.4f})")
    print("Simulation only: validates the code path, not the physics.")

if __name__ == "__main__":
    main()
