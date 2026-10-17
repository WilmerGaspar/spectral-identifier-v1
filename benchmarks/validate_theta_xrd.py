"""
Validation of Ramanujan-theta XRD module.

    python benchmarks/validate_theta_xrd.py

Checks
 1. Ramanujan theta identities == brute-force (h,k,l) counting (P, I, F cubic).
 2. Legendre: N = 4^a (8b+7) has zero coefficient in phi(q)^3.
 3. Theta (lattice) lines vs an independent structure-factor calculation
    for NaCl (rock salt) and Si (diamond): every allowed line must be predicted.
 4. Quartz (hexagonal axes): first lines vs reference 2-theta values.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.theta_xrd.ramanujan_xrd import (
    theta_cubic, brute_force_cubic, cubic_pattern, hexagonal_pattern,
    structure_factor_allowed_N, CU_KA1,
)


def check_identities(M=200):
    print("1) Ramanujan theta identities vs brute-force counting (N <= %d)" % M)
    for lat in "PIF":
        ok = np.array_equal(theta_cubic(lat, M), brute_force_cubic(lat, M))
        print(f"   {lat}: {'IDENTICAL' if ok else 'MISMATCH'}")
        assert ok


def check_legendre(M=200):
    th = theta_cubic("P", M)
    zeros = [N for N in range(1, M + 1) if th[N] == 0]
    def legendre(n):
        while n % 4 == 0 and n > 0:
            n //= 4
        return n % 8 == 7
    ok = zeros == [N for N in range(1, M + 1) if legendre(N)]
    print(f"\n2) Missing lines in primitive cubic: {zeros[:8]} ...")
    print(f"   equal to Legendre's 4^a(8b+7) set: {ok}")
    assert ok


def check_basis(name, a, lattice, basis):
    lines = cubic_pattern(a, lattice, two_theta_max=90)
    M = max(int(l.label[2:]) for l in lines)
    sf = structure_factor_allowed_N(basis, M)
    theta_N = {int(l.label[2:]) for l in lines}
    sf_N = {N for N in sf if N in theta_N or N <= M}
    missed = sf_N - theta_N
    extra = theta_N - sf_N
    print(f"\n3) {name} (a = {a} A, Cu Ka1), 2theta <= 90 deg")
    print(f"   lattice-theta lines: {len(theta_N)} | structure-factor lines: {len(sf_N)}")
    print(f"   real lines missed by theta: {sorted(missed)}  (must be empty)")
    print(f"   theta lines removed by atom basis: {sorted(extra)}")
    for l in lines[:6]:
        N = int(l.label[2:])
        tag = "" if N in sf_N else "  (absent: basis)"
        print(f"     {l.label:<6} 2theta = {l.two_theta:6.2f}  mult = {l.multiplicity}{tag}")
    assert not missed


def check_quartz():
    # Quartz cell (hexagonal axes); reference lines Cu Ka: (100) 20.86, (101) 26.64 deg
    lines = hexagonal_pattern(4.9134, 5.4052, two_theta_max=40)
    print("\n4) Quartz (a = 4.9134, c = 5.4052 A), first lattice lines:")
    for l in lines[:5]:
        print(f"     {l.label:<10} 2theta = {l.two_theta:6.2f}  mult = {l.multiplicity}")
    t = [l.two_theta for l in lines]
    for ref, hkl in [(20.86, "100"), (26.64, "101")]:
        best = min(t, key=lambda x: abs(x - ref))
        print(f"   reference ({hkl}) {ref:.2f}  -> predicted {best:.2f}  (diff {best - ref:+.2f})")
        assert abs(best - ref) < 0.05


if __name__ == "__main__":
    check_identities()
    check_legendre()
    nacl = [(0, 0, 0, 11), (.5, .5, 0, 11), (.5, 0, .5, 11), (0, .5, .5, 11),
            (.5, .5, .5, 17), (0, 0, .5, 17), (0, .5, 0, 17), (.5, 0, 0, 17)]
    check_basis("NaCl rock salt", 5.6402, "F", nacl)
    fcc = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    si = [(x, y, z, 14) for x, y, z in fcc] + [(x + .25, y + .25, z + .25, 14) for x, y, z in fcc]
    check_basis("Si diamond", 5.4309, "F", si)
    check_quartz()
    print("\nAll checks passed.")
