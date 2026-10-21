import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Spectral Identifier")


@app.cell
def _():
    import os
    import sys

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from core import (load_xrd, fit_xrd, load_raman, identify_raman, literature)
    return fit_xrd, go, identify_raman, literature, load_raman, load_xrd, mo, np, os


@app.cell
def _(mo):
    mo.md("""
    # Spectral Identifier
    Identify minerals from their spectra and fit their crystal cell with **Ramanujan theta functions**,
    on real measured data from the [RRUFF database](https://rruff.info). Every number below is computed live.
    """)
    return


@app.cell
def _(load_raman, load_xrd):
    xrd_grid, xrd_patterns, xrd_index = load_xrd()
    raman_grid, raman_spectra, raman_minerals, raman_ids = load_raman()
    return raman_grid, raman_ids, raman_minerals, raman_spectra, xrd_grid, xrd_index, xrd_patterns


@app.cell
def _(mo, xrd_index):
    _opts = {f"{info['mineral']} · {info['rruff_id']} · {info.get('space_group') or info['system'] + '-' + info['centering']}": i
             for i, info in sorted(enumerate(xrd_index), key=lambda t: (t[1]["mineral"], t[1]["rruff_id"]))}
    xrd_pick = mo.ui.dropdown(options=_opts, value=next(k for k in _opts if "R040076" in k),
                              label="Measured X-ray pattern", searchable=True)
    show_lines = mo.ui.switch(value=True, label="Show Ramanujan-theta lines")
    return show_lines, xrd_pick


@app.cell
def _(fit_xrd, xrd_grid, xrd_index, xrd_patterns, xrd_pick):
    xrd = fit_xrd(xrd_grid, xrd_patterns[xrd_pick.value], xrd_index[xrd_pick.value])
    return (xrd,)


@app.cell
def _(go, mo, np, show_lines, xrd, xrd_pick):
    def _xrd_figure(r, lines_on):
        y = r["intensity"] - r["intensity"].min()
        y = y / (y.max() or 1)
        fig = go.Figure()
        if lines_on and r["error"] is None:
            xs = np.repeat(r["lines"], 3)
            ys = np.tile([0, 1.08, np.nan], len(r["lines"]))
            xs = xs.astype(float)
            xs[2::3] = np.nan
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Ramanujan-theta lines",
                                     line=dict(color="rgba(13,148,136,0.45)", width=1)))
        fig.add_trace(go.Scatter(x=r["two_theta"], y=y, mode="lines", name="Measured pattern",
                                 line=dict(color="#334155", width=1.4)))
        pk = r["peaks"]
        if len(pk):
            ypk = np.interp(pk, r["two_theta"], y)
            m = r.get("matched", np.zeros(len(pk), bool))
            fig.add_trace(go.Scatter(x=pk[m], y=ypk[m] + 0.03, mode="markers", name="Peak explained",
                                     marker=dict(color="#16a34a", size=8, symbol="triangle-down")))
            fig.add_trace(go.Scatter(x=pk[~m], y=ypk[~m] + 0.03, mode="markers", name="Peak not explained",
                                     marker=dict(color="#dc2626", size=8, symbol="triangle-down")))
        fig.update_layout(height=430, margin=dict(l=50, r=20, t=30, b=50), template="plotly_white",
                          xaxis_title="2θ (degrees)", yaxis_title="Intensity (normalised)",
                          legend=dict(orientation="h", y=1.08, x=0), font=dict(size=14))
        fig.update_yaxes(range=[0, 1.12])
        return fig

    def _stats(r):
        if r["error"]:
            return mo.callout(mo.md(r["error"]), kind="warn")
        q = r["quality"]
        cards = [mo.stat(f"{r['a']:.4f} Å", label="Fitted a",
                         caption=f"RRUFF {r['a_ref']:.4f} Å · diff {r['a'] - r['a_ref']:+.4f}" if r["a_ref"] else "no RRUFF value")]
        if r["c"] is not None:
            cards.append(mo.stat(f"{r['c']:.4f} Å", label="Fitted c",
                                 caption=f"RRUFF {r['c_ref']:.4f} Å · diff {r['c'] - r['c_ref']:+.4f}"))
        cards.append(mo.stat(f"{r['explained']:.0%}", label="Peaks explained",
                             caption=f"{int(r['matched'].sum())} of {len(r['peaks'])} detected peaks"))
        cards.append(mo.stat(f"{q['score']:.2f}", label="Reliability score",
                             caption="0 = chance level, 1 = every peak matched"))
        if r["garnet"]:
            cards.append(mo.stat(r["garnet"], label="Closest garnet end-member",
                                 caption="real garnets are solid solutions"))
        if q["flags"]:
            verdict = mo.callout(mo.md(f"**Flags: {', '.join(q['flags'])}.** "
                                       "Treat this fit with caution; the numbers are shown unchanged."),
                                 kind="danger" if "unreliable" in q["flags"] else "warn")
        else:
            verdict = mo.callout(mo.md("**No reliability flags.**"), kind="success")
        return mo.vstack([mo.hstack(cards, justify="start", gap=1, wrap=True), verdict,
                          mo.md(f"<small>{r['system']} · λ = {r['lam']:.4f} Å · cell {r['protocol']} · "
                                f"zero shift {r['zero_shift']:+.3f}°</small>")])

    xrd_tab = mo.vstack([
        mo.hstack([xrd_pick, show_lines], justify="start", gap=2),
        mo.as_html(_xrd_figure(xrd, show_lines.value)),
        mo.md("<small>**Teal lines**: allowed reflections for the fitted cell (Ramanujan theta series; "
              "space group conditions applied in cubic minerals, only lattice centering in "
              "tetragonal/hexagonal). Intensities are not modelled, which is why many lines do not "
              "have a visible peak.</small>"),
        _stats(xrd),
    ])
    return (xrd_tab,)


@app.cell
def _(mo, raman_ids, raman_minerals):
    _opts = {f"{m} · {rid}": i for i, (m, rid) in
             sorted(enumerate(zip(raman_minerals, raman_ids)), key=lambda t: t[1])}
    raman_pick = mo.ui.dropdown(options=_opts, value=next(iter(_opts)), label="Measured Raman spectrum",
                                searchable=True)
    return (raman_pick,)


@app.cell
def _(go, identify_raman, mo, raman_grid, raman_ids, raman_minerals, raman_pick, raman_spectra):
    _i = raman_pick.value
    _top = identify_raman(raman_spectra, raman_minerals, raman_ids, _i)
    _truth = raman_minerals[_i]

    def _norm(s):
        s = s - s.min()
        return s / (s.max() or 1)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(x=raman_grid, y=_norm(raman_spectra[_i]) + 1.1, mode="lines",
                              name=f"Query: {raman_ids[_i]}", line=dict(color="#334155", width=1.4)))
    _best = _top[0][2]
    _fig.add_trace(go.Scatter(x=raman_grid, y=_norm(raman_spectra[_best]), mode="lines",
                              name=f"Best match: {raman_minerals[_best]} ({raman_ids[_best]})",
                              line=dict(color="#0d9488", width=1.4)))
    _fig.update_layout(height=430, margin=dict(l=50, r=20, t=30, b=50), template="plotly_white",
                       xaxis_title="Raman shift (cm⁻¹)", yaxis_title="Intensity (normalised, offset)",
                       legend=dict(orientation="h", y=1.08, x=0), font=dict(size=14))
    _fig.update_yaxes(showticklabels=False)

    _rows = [{"Rank": k + 1, "Mineral": m, "Similarity": f"{s:.3f}",
              "Library sample": raman_ids[j], "Correct": "✓" if m == _truth else ""}
             for k, (m, s, j) in enumerate(_top)]
    raman_tab = mo.vstack([
        raman_pick,
        mo.as_html(_fig),
        mo.md(f"**Top 5 minerals** (cosine similarity; the query sample {raman_ids[_i]} is never in the library)"),
        mo.ui.table(_rows, selection=None, pagination=False),
    ])
    return (raman_tab,)


@app.cell
def _(mo, os, xrd_index, xrd_pick):
    lit_phases = mo.ui.text(value=xrd_index[xrd_pick.value]["mineral"], label="Minerals (comma separated)",
                            full_width=True)
    lit_topic = mo.ui.radio(options={"Crystal structure (XRD)": "xrd", "Raman bands": "raman"},
                            value="Crystal structure (XRD)", label="Topic", inline=True)
    lit_go = mo.ui.run_button(label="Search the literature")
    has_token = bool(os.environ.get("VALENCY_TOKEN"))
    return has_token, lit_go, lit_phases, lit_topic


@app.cell
def _(has_token, lit_go, lit_phases, lit_topic, literature, mo, os):
    if lit_go.value:
        _phases = [p.strip() for p in lit_phases.value.split(",") if p.strip()]
        with mo.status.spinner(title="Searching millions of research papers with Valency MCP Bond…"):
            _text, _mode = literature(_phases, lit_topic.value, os.environ.get("VALENCY_TOKEN", ""))
    else:
        _text, _mode = literature([], "xrd", "")
    _source = (mo.callout(mo.md("**Live search** with Valency MCP Bond."), kind="success") if _mode == "live"
               else mo.callout(mo.md("Showing the **saved run** (`src/ai/output/literature_xrd.txt`)."
                                     + ("" if has_token else " Set `VALENCY_TOKEN` to search live.")), kind="info"))
    lit_tab = mo.vstack([
        mo.hstack([lit_phases, lit_topic, lit_go], justify="start", gap=2, align="end"),
        _source,
        mo.md(f"```text\n{_text}\n```"),
    ])
    return (lit_tab,)


@app.cell
def _(lit_tab, mo, raman_tab, xrd_tab):
    mo.ui.tabs({"X-ray · Ramanujan theta": xrd_tab, "Raman · identification": raman_tab,
                "Literature · Valency": lit_tab})
    return


if __name__ == "__main__":
    app.run()
