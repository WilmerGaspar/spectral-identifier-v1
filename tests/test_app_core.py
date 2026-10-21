"""The demo app must show the same numbers as the benchmark outputs
(benchmarks/output/validate_xrd_real*.txt, benchmarks/benchmark_real.py)."""

import numpy as np
import pytest

from app.core import load_xrd, fit_xrd, load_raman, identify_raman, literature

# rruff_id: (a fit, c fit or None, peaks explained, score, flags, garnet guess) from benchmarks/output
EXPECTED = {
    "R040076": (11.5442, None, 0.93, 0.92, [], "Almandine"),       # almandine, cubic
    "R050036": (11.8866, None, 1.00, 1.00, [], "Grossular"),       # grossular, cubic
    "R040031": (4.9146, 5.4060, 1.00, 1.00, [], None),             # quartz, hexagonal-P
    "R050035": (15.5661, 11.8045, 0.98, 0.93, [], None),           # vesuvianite (bug-fix case)
    "R060190": (5.1144, 13.5250, 0.11, 0.14, ["noisy", "unreliable"], None),  # hematite, flagged
}


@pytest.fixture(scope="module")
def xrd():
    return load_xrd()


@pytest.mark.parametrize("rid", EXPECTED)
def test_xrd_fit_matches_benchmark(xrd, rid):
    grid, P, index = xrd
    i = next(k for k, info in enumerate(index) if info["rruff_id"] == rid)
    r = fit_xrd(grid, P[i], index[i])
    a, c, explained, score, flags, garnet = EXPECTED[rid]
    assert r["error"] is None
    assert r["a"] == pytest.approx(a, abs=1e-4)
    if c is not None:
        assert r["c"] == pytest.approx(c, abs=1e-4)
    assert r["explained"] == pytest.approx(explained, abs=0.005)
    assert r["quality"]["score"] == pytest.approx(score, abs=0.005)
    assert r["quality"]["flags"] == flags
    assert r["garnet"] == garnet
    # the plotted lines and matched peaks must agree with the fit
    assert r["matched"].mean() == pytest.approx(r["explained"], abs=0.05)
    assert np.all((r["lines"] >= r["two_theta"][0]) & (r["lines"] <= r["two_theta"][-1]))


def test_raman_top1_matches_benchmark():
    grid, S, minerals, ids = load_raman()
    top = [identify_raman(S, minerals, ids, i) for i in range(len(S))]
    top1 = np.mean([t[0][0] == minerals[i] for i, t in enumerate(top)])
    top5 = np.mean([minerals[i] in [m for m, _, _ in t] for i, t in enumerate(top)])
    assert top1 == pytest.approx(0.933, abs=0.001)          # benchmark: cosine top-1 93.3 %
    assert top5 == 1.0                                       # benchmark: top-5 100 %
    assert all(ids[j] != ids[i] for i, t in enumerate(top) for _, _, j in t)  # query never in library


def test_literature_without_token_uses_saved_run():
    text, mode = literature(["Zircon"], "xrd", "")
    assert mode == "saved" and "Zircon" in text
