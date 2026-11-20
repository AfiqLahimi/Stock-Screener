"""Screener page: scan the universe chosen in the sidebar, then filter it with one-click presets or your own
combination of conditions (match ALL or ANY). Saved screens persist between visits (prefs.py).

Every condition reads values computed in core.scan_universe from completed candles. A stock whose value is N/A
(e.g. not enough history for EMA 200) never passes a condition that needs it.
"""
from functools import partial

import numpy as np
import pandas as pd
import streamlit as st

import ui
from core import SCAN_BASE, display_name, scan_universe  # noqa: F401

BUY, SELL = ["BUY", "STRONG BUY"], ["SELL", "STRONG SELL"]
SIGNALS = ["STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"]
POS = ["Any", "Above", "Below"]
CROSS = ["Any", "Bullish cross (recent)", "Bearish cross (recent)", "Fast above slow", "Fast below slow"]
MACD = ["Any", "Above signal", "Below signal", "Bullish cross (recent)", "Bearish cross (recent)", "Histogram rising"]
RET = ["Any", "> 0%", "> 5%", "> 10%", "> 20%", "< 0%", "< −10%"]
NEAR = ["Any", "Within 2%", "Within 3%", "Within 5%"]
BRK = ["Any", "20-bar breakout", "20-bar breakdown", "New 52-week high", "Within 5% of 52-week high",
       "Within 10% of 52-week low"]
ATR = ["Any", "Low (< 2%)", "Moderate (2–4%)", "High (> 4%)"]
VALUES = [0, 50_000, 100_000, 500_000, 1_000_000, 5_000_000, 20_000_000]

DEFAULTS = {
    "scr_c_signal": [], "scr_c_score": (-100, 100),
    "scr_c_e20": "Any", "scr_c_e50": "Any", "scr_c_e200": "Any", "scr_c_align": "Any",
    "scr_c_adx": (0, 100), "scr_c_di": "Any",
    "scr_c_x920": "Any", "scr_c_x2050": "Any", "scr_c_x50200": "Any", "scr_c_xn": 5,
    "scr_c_rsi": (0, 100), "scr_c_macd": "Any", "scr_c_ret": "Any", "scr_c_stoch": "Any",
    "scr_c_rvol": 0.0, "scr_c_value": 0,
    "scr_c_sup": "Any", "scr_c_res": "Any", "scr_c_rr": 0.0, "scr_c_brk": "Any", "scr_c_atr": "Any",
}
PRESETS = {
    "🟢 Bullish signals": {"scr_c_signal": BUY},
    "🔴 Bearish signals": {"scr_c_signal": SELL},
    "📈 Uptrend pullback": {"scr_c_e50": "Above", "scr_c_e200": "Above", "scr_c_rsi": (35, 55),
                           "scr_c_sup": "Within 3%"},
    "🚀 Breakout + volume": {"scr_c_brk": "20-bar breakout", "scr_c_rvol": 1.5},
    "⚡ Momentum leaders": {"scr_c_adx": (25, 100), "scr_c_di": "+DI above −DI", "scr_c_macd": "Above signal",
                           "scr_c_e50": "Above", "scr_c_ret": "> 10%"},
    "✨ Golden cross": {"scr_c_x50200": "Bullish cross (recent)", "scr_c_xn": 20},
    "💀 Death cross": {"scr_c_x50200": "Bearish cross (recent)", "scr_c_xn": 20},
    "🔁 Oversold bounce": {"scr_c_rsi": (0, 35), "scr_c_macd": "Bullish cross (recent)"},
    "🛟 Near support, R:R ≥ 2": {"scr_c_sup": "Within 3%", "scr_c_rr": 2.0},
    "🔥 Unusual volume": {"scr_c_rvol": 2.0},
    "🔻 Breakdown": {"scr_c_brk": "20-bar breakdown", "scr_c_rvol": 1.5},
    "😴 Sideways (ADX < 20)": {"scr_c_adx": (0, 20)},
}
VIEWS = {
    "Overview": ["Name", "Ticker", "Shariah", "Price", "Change %", "Signal", "Score", "RSI", "RVOL", "Trend (ADX)",
                 "R:R"],
    "Trend": ["Name", "Price", "Change %", "Score", "Above EMA20", "Above EMA50", "Above EMA200", "EMA9>20",
              "EMA20>50", "EMA50>200", "X 50/200", "ADX", "+DI>-DI", "Trend (ADX)"],
    "Momentum": ["Name", "Price", "Change %", "Score", "RSI", "Stoch %K", "MACD>signal", "X MACD", "X 9/20",
                 "Return 5 %", "Return 20 %", "Return 60 %"],
    "Volume & volatility": ["Name", "Price", "Change %", "Volume", "RVOL", "Avg value", "ATR %", "Breakout",
                            "Breakdown", "From 52w high %", "From 52w low %"],
    "Levels & plan": ["Name", "Price", "Signal", "Support", "Resistance", "To support %", "To resistance %",
                      "Buy zone", "Stop", "Target", "R:R"],
}


# ----------------------------------------------------------------------------
# State helpers (callbacks run before widgets are drawn, so they may set widget values)
# ----------------------------------------------------------------------------
def _reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = list(v) if isinstance(v, list) else v


def _apply_preset():
    name = st.session_state.get("scr_preset")
    _reset()
    for k, v in PRESETS.get(name, {}).items():
        st.session_state[k] = list(v) if isinstance(v, list) else v
    st.session_state["scr_mode"] = "Match ALL"


def _criteria():
    return {k: (list(st.session_state[k]) if isinstance(st.session_state.get(k), (list, tuple))
                else st.session_state.get(k, DEFAULTS[k])) for k in DEFAULTS}


def _save():
    name = (st.session_state.get("scr_save_name") or "").strip()
    if name:
        saved = dict(st.session_state.get("scr_saved", {}))
        saved[name] = {"criteria": _criteria(), "mode": st.session_state.get("scr_mode", "Match ALL")}
        st.session_state["scr_saved"] = saved
        st.session_state["scr_save_name"] = ""
        st.session_state["scr_flash"] = f"Saved screen “{name}”."


def _load_saved():
    name = st.session_state.get("scr_my")
    s = st.session_state.get("scr_saved", {}).get(name)
    if s:
        _reset()
        for k, v in s["criteria"].items():
            if k in DEFAULTS:
                st.session_state[k] = tuple(v) if isinstance(DEFAULTS[k], tuple) else v
        st.session_state["scr_mode"] = s.get("mode", "Match ALL")
        st.session_state["scr_preset"] = None


def _delete_saved():
    name = st.session_state.get("scr_my")
    saved = dict(st.session_state.get("scr_saved", {}))
    if name in saved:
        del saved[name]
        st.session_state["scr_saved"] = saved
        st.session_state["scr_my"] = None


def _open_row(key, tickers):
    sel = st.session_state.get(key)
    try:
        rows = sel["selection"]["rows"]
    except Exception:
        rows = []
    if rows and rows[0] < len(tickers):
        ui.goto("analysis", an_typed=tickers[rows[0]])


# ----------------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------------
def _is(col, val):
    return col.map(lambda v: v is not None and v == v and bool(v) == val)


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce")


def build_conditions(df, c):
    """List of (description, boolean mask) for every criterion that is not 'Any'."""
    out = []
    n = int(c["scr_c_xn"])
    if c["scr_c_signal"]:
        out.append(("Signal: " + ", ".join(c["scr_c_signal"]), df["_label"].isin(c["scr_c_signal"])))
    lo, hi = c["scr_c_score"]
    if (lo, hi) != (-100, 100):
        out.append((f"Score {lo:+d} to {hi:+d}", df["Score"].between(lo, hi)))
    for key, col, lab in (("scr_c_e20", "Above EMA20", "EMA 20"), ("scr_c_e50", "Above EMA50", "EMA 50"),
                          ("scr_c_e200", "Above EMA200", "EMA 200")):
        if c[key] != "Any":
            out.append((f"Price {c[key].lower()} {lab}", _is(df[col], c[key] == "Above")))
    if c["scr_c_align"] == "Bullish stack":
        out.append(("Bullish EMA stack (9 > 20 > 50 > 200)",
                    _is(df["EMA9>20"], True) & _is(df["EMA20>50"], True) & _is(df["EMA50>200"], True)))
    elif c["scr_c_align"] == "Bearish stack":
        out.append(("Bearish EMA stack (9 < 20 < 50 < 200)",
                    _is(df["EMA9>20"], False) & _is(df["EMA20>50"], False) & _is(df["EMA50>200"], False)))
    lo, hi = c["scr_c_adx"]
    if (lo, hi) != (0, 100):
        out.append((f"ADX {lo}–{hi}", _num(df, "ADX").between(lo, hi)))
    if c["scr_c_di"] != "Any":
        out.append((c["scr_c_di"], _is(df["+DI>-DI"], c["scr_c_di"].startswith("+DI"))))
    for key, base, lab in (("scr_c_x920", "X 9/20", "EMA 9/20"), ("scr_c_x2050", "X 20/50", "EMA 20/50"),
                           ("scr_c_x50200", "X 50/200", "EMA 50/200")):
        v = c[key]
        if v == "Any":
            continue
        dir_, age = df[f"{base} dir"], pd.to_numeric(df[f"{base} age"], errors="coerce")
        fast_above = {"X 9/20": "EMA9>20", "X 20/50": "EMA20>50", "X 50/200": "EMA50>200"}[base]
        if v.startswith("Bullish"):
            out.append((f"{lab} bullish cross in last {n} bars", (dir_ == 1) & (age <= n)))
        elif v.startswith("Bearish"):
            out.append((f"{lab} bearish cross in last {n} bars", (dir_ == -1) & (age <= n)))
        else:
            out.append((f"{lab}: {v.lower()}", _is(df[fast_above], v == "Fast above slow")))
    lo, hi = c["scr_c_rsi"]
    if (lo, hi) != (0, 100):
        out.append((f"RSI {lo}–{hi}", df["RSI"].between(lo, hi)))
    m = c["scr_c_macd"]
    if m == "Above signal":
        out.append(("MACD above signal", _is(df["MACD>signal"], True)))
    elif m == "Below signal":
        out.append(("MACD below signal", _is(df["MACD>signal"], False)))
    elif m.startswith("Bullish"):
        out.append((f"MACD bullish cross in last {n} bars",
                    (df["X MACD dir"] == 1) & (pd.to_numeric(df["X MACD age"], errors="coerce") <= n)))
    elif m.startswith("Bearish"):
        out.append((f"MACD bearish cross in last {n} bars",
                    (df["X MACD dir"] == -1) & (pd.to_numeric(df["X MACD age"], errors="coerce") <= n)))
    elif m == "Histogram rising":
        out.append(("MACD histogram rising", _is(df["MACD hist rising"], True)))
    r = c["scr_c_ret"]
    if r != "Any":
        thr = float(r.replace("> ", "").replace("< ", "").replace("%", "").replace("−", "-"))
        r20 = _num(df, "Return 20 %")
        out.append((f"20-day return {r}", r20 > thr if r.startswith(">") else r20 < thr))
    s = c["scr_c_stoch"]
    if s != "Any":
        k = _num(df, "Stoch %K")
        out.append((f"Stochastic {s.lower()}", k < 20 if s.startswith("Oversold") else k > 80))
    if c["scr_c_rvol"] > 0:
        out.append((f"RVOL ≥ {c['scr_c_rvol']:.2f}×", _num(df, "RVOL") >= c["scr_c_rvol"]))
    if c["scr_c_value"] > 0:
        out.append((f"Avg traded value ≥ {c['scr_c_value']:,.0f}", _num(df, "Avg value") >= c["scr_c_value"]))
    if c["scr_c_sup"] != "Any":
        pct = float(c["scr_c_sup"].split()[1].rstrip("%"))
        out.append((f"{c['scr_c_sup']} above support", df["To support %"].between(0, pct)))
    if c["scr_c_res"] != "Any":
        if c["scr_c_res"].startswith("Within"):
            pct = float(c["scr_c_res"].split()[1].rstrip("%"))
            out.append((f"{c['scr_c_res']} below resistance", df["To resistance %"].between(0, pct)))
    if c["scr_c_rr"] > 0:
        out.append((f"R:R ≥ {c['scr_c_rr']:.1f}", _num(df, "R:R") >= c["scr_c_rr"]))
    b = c["scr_c_brk"]
    if b == "20-bar breakout":
        out.append(("Closed above 20-bar high", _is(df["Breakout"], True)))
    elif b == "20-bar breakdown":
        out.append(("Closed below 20-bar low", _is(df["Breakdown"], True)))
    elif b == "New 52-week high":
        out.append(("New 52-week high", _is(df["New 52w high"], True)))
    elif b.startswith("Within 5%"):
        out.append(("Within 5% of 52-week high", _num(df, "From 52w high %") >= -5))
    elif b.startswith("Within 10%"):
        out.append(("Within 10% of 52-week low", _num(df, "From 52w low %") <= 10))
    a = c["scr_c_atr"]
    if a != "Any":
        atr = _num(df, "ATR %")
        out.append((f"Volatility {a}", atr < 2 if a.startswith("Low") else atr > 4 if a.startswith("High")
                    else atr.between(2, 4)))
    return [(lab, mask.fillna(False).astype(bool)) for lab, mask in out]


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------
def criteria_editor(mob):
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, list(v) if isinstance(v, list) else v)
    g1, g2, g3, g4 = (st.container(),) * 4 if mob else st.columns(4, gap="medium")
    with g1:
        ui.section("Signal & trend")
        st.multiselect("Signal", SIGNALS, key="scr_c_signal", placeholder="Any signal")
        st.slider("Score", -100, 100, key="scr_c_score", step=5)
        st.selectbox("Price vs EMA 20", POS, key="scr_c_e20")
        st.selectbox("Price vs EMA 50", POS, key="scr_c_e50")
        st.selectbox("Price vs EMA 200", POS, key="scr_c_e200")
        st.selectbox("EMA alignment", ["Any", "Bullish stack", "Bearish stack"], key="scr_c_align",
                     help="Bullish stack = EMA 9 > 20 > 50 > 200 (needs 200+ candles)")
    with g2:
        ui.section("Crossovers & strength")
        st.selectbox("EMA 9 / 20", CROSS, key="scr_c_x920")
        st.selectbox("EMA 20 / 50", CROSS, key="scr_c_x2050")
        st.selectbox("EMA 50 / 200 (golden / death cross)", CROSS, key="scr_c_x50200")
        st.slider("“Recent” cross = within last N bars", 1, 30, key="scr_c_xn")
        st.slider("ADX (trend strength)", 0, 100, key="scr_c_adx", step=5,
                  help="< 20 sideways, 20–25 weak trend, 25+ strong trend")
        st.selectbox("Direction (DMI)", ["Any", "+DI above −DI", "−DI above +DI"], key="scr_c_di")
    with g3:
        ui.section("Momentum")
        st.slider("RSI (14)", 0, 100, key="scr_c_rsi", step=5, help="< 30 oversold, > 70 overbought")
        st.selectbox("MACD (12, 26, 9)", MACD, key="scr_c_macd")
        st.selectbox("Stochastic %K", ["Any", "Oversold (< 20)", "Overbought (> 80)"], key="scr_c_stoch")
        st.selectbox("20-day return", RET, key="scr_c_ret")
        st.selectbox("Volatility (ATR % of price)", ATR, key="scr_c_atr")
    with g4:
        ui.section("Volume, levels & breakouts")
        st.slider("Min relative volume (RVOL)", 0.0, 5.0, key="scr_c_rvol", step=0.25,
                  help="Today's volume ÷ 20-day average. 1.5×+ = above normal, 2×+ = spike")
        st.select_slider("Min average daily traded value", VALUES, key="scr_c_value",
                         format_func=lambda v: "Any" if v == 0 else f"{v:,.0f}")
        st.selectbox("Distance above support", NEAR, key="scr_c_sup")
        st.selectbox("Distance below resistance", NEAR, key="scr_c_res")
        st.slider("Min reward : risk", 0.0, 5.0, key="scr_c_rr", step=0.5)
        st.selectbox("Breakout / 52-week", BRK, key="scr_c_brk")


def render(C):
    universe, interval, period, unit = C.universe, C.interval, C.period, C.unit
    shariah_set, mob = C.shariah_set, C.mobile
    if not universe:
        st.info("Pick a market or enter tickers in the sidebar.")
        return
    with st.spinner(f"Scanning {len(universe)} stocks…"):
        tbl = scan_universe(universe, interval, period, unit)
    got = set(tbl["Ticker"]) if not tbl.empty else set()
    missing = [display_name(t) for t in universe if t not in got]
    if tbl.empty:
        st.warning("No data came back for this universe. Try again in a minute or pick another market.")
        return
    if shariah_set:
        tbl.insert(2, "Shariah", ["✅" if t in shariah_set else ("–" if t.endswith(".KL") else "")
                                  for t in tbl["Ticker"]])
    for base in ("X 9/20", "X 20/50", "X 50/200", "X MACD"):
        tbl[base] = [("▲ " if d_ == 1 else "▼ " if d_ == -1 else "") + ("" if a_ is None or a_ != a_ else f"{int(a_)} bars ago")
                     if d_ else "none (60 bars)" for d_, a_ in zip(tbl[f"{base} dir"], tbl[f"{base} age"])]

    # ---- presets + criteria
    with ui.panel("Screens", f"{len(tbl)} stocks scanned · {C.tf_name}"):
        st.pills("Quick screens", list(PRESETS), key="scr_preset", on_change=_apply_preset,
                 label_visibility="collapsed")
        with st.expander("🧮 Custom criteria: combine any conditions", expanded=False):
            criteria_editor(mob)
            st.divider()
            a1, a2, a3, a4, a5 = st.columns([1.2, 0.8, 1.6, 0.8, 2.2]) if not mob else \
                (st.container(), st.container(), st.container(), st.container(), st.container())
            a1.segmented_control("Combine", ["Match ALL", "Match ANY"], default="Match ALL", key="scr_mode",
                                 label_visibility="collapsed")
            a2.button("↺ Reset", on_click=_reset, key="scr_reset", width="stretch")
            a3.text_input("Save as", placeholder="Name this screen…", key="scr_save_name",
                          label_visibility="collapsed")
            a4.button("💾 Save", on_click=_save, key="scr_saveb", width="stretch")
            saved = st.session_state.get("scr_saved", {})
            if saved:
                with a5:
                    m1, m2, m3 = st.columns([2, 1, 1])
                    m1.selectbox("My screens", list(saved), index=None, key="scr_my", placeholder="My screens…",
                                 label_visibility="collapsed")
                    m2.button("Load", on_click=_load_saved, key="scr_load", width="stretch")
                    m3.button("🗑", on_click=_delete_saved, key="scr_del", width="stretch")
            if st.session_state.get("scr_flash"):
                st.success(st.session_state.pop("scr_flash"))

    crit = _criteria() if "scr_c_signal" in st.session_state else dict(DEFAULTS)
    conds = build_conditions(tbl, crit)
    mode = st.session_state.get("scr_mode") or "Match ALL"
    if conds:
        masks = np.column_stack([m.values for _, m in conds])
        keep = masks.all(axis=1) if mode == "Match ALL" else masks.any(axis=1)
        view = tbl[keep]
        tbl_hits = masks.sum(axis=1)
        view = view.assign(Matched=tbl_hits[keep])
    else:
        view = tbl.assign(Matched=0)
    view = view.sort_values(["Matched", "Score"], ascending=False).reset_index(drop=True)

    # ---- summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", len(tbl))
    c2.metric("Match the screen", len(view))
    c3.metric("Buy signals", int(view["_label"].isin(BUY).sum()))
    c4.metric("Sell signals", int(view["_label"].isin(SELL).sum()))
    if conds:
        chips = "".join(f"<span class='chip'>{lab} · {int(m.sum())}</span>" for lab, m in conds)
        st.markdown(f"<div style='display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 2px'>"
                    f"<span class='muted' style='font-size:12px;align-self:center'>{mode}:</span>{chips}</div>",
                    unsafe_allow_html=True)

    # ---- results
    with ui.panel("Results", "click a row to open it in Analysis · click a column header to sort"):
        vsel = st.segmented_control("Columns", list(VIEWS), default="Overview", key="scr_cols",
                                    label_visibility="collapsed") or "Overview"
        cols = [c for c in VIEWS[vsel] if c in view.columns]
        if conds and mode == "Match ANY":
            cols = cols[:2] + ["Matched"] + cols[2:]
        if mob:
            cols = ["Name", "Price", "Change %", "Signal", "Score"]
        if view.empty:
            st.info("No stocks match all of these conditions. Loosen one (see the counts on each condition above) "
                    "or switch to Match ANY.")
        else:
            cc = st.column_config
            yes_no = {k: cc.CheckboxColumn(k, width="small") for k in
                      ("Above EMA20", "Above EMA50", "Above EMA200", "EMA9>20", "EMA20>50", "EMA50>200", "+DI>-DI",
                       "MACD>signal", "Breakout", "Breakdown")}
            key = "scr_tbl"
            st.dataframe(
                view, hide_index=True, width="stretch", column_order=cols, key=key,
                on_select=partial(_open_row, key, list(view["Ticker"])), selection_mode="single-row",
                height="content" if len(view) <= 20 else 760,
                column_config={
                    **yes_no,
                    "Name": cc.TextColumn(width="small"), "Price": cc.NumberColumn(format="%.3f"),
                    "Change %": cc.NumberColumn(format="%+.2f%%"),
                    "Score": cc.ProgressColumn(min_value=-100, max_value=100, format="%d", width="small"),
                    "RSI": cc.NumberColumn(format="%.0f", width="small"),
                    "ADX": cc.NumberColumn(format="%.0f", width="small"),
                    "RVOL": cc.NumberColumn(format="%.1f×", width="small"),
                    "Stoch %K": cc.NumberColumn(format="%.0f", width="small"),
                    "Volume": cc.NumberColumn(format="compact"), "Avg value": cc.NumberColumn(format="compact"),
                    "ATR %": cc.NumberColumn(format="%.2f%%"),
                    **{k: cc.NumberColumn(format="%+.1f%%") for k in
                       ("Return 5 %", "Return 20 %", "Return 60 %", "From 52w high %", "From 52w low %")},
                    "Support": cc.NumberColumn(format="%.3f"), "Resistance": cc.NumberColumn(format="%.3f"),
                    "To support %": cc.NumberColumn(format="%.1f%%"),
                    "To resistance %": cc.NumberColumn(format="%.1f%%"),
                    "Stop": cc.NumberColumn(format="%.3f"), "Target": cc.NumberColumn(format="%.3f"),
                    "R:R": cc.NumberColumn(format="%.1f", width="small"),
                    "Matched": cc.NumberColumn("Conditions met", format="%d", width="small"),
                })
        st.caption("Score −100 to +100 combines trend, RSI, MACD, distance to support/resistance, breakouts and volume. "
                   "R:R = (target − entry) ÷ (entry − stop). Crosses use EMAs with a full warm-up; stocks without "
                   "enough history show N/A and fail conditions that need them. Download the table with the ⬇ icon "
                   "at its top-right. These are research candidates, not buy or sell orders.")
    if missing:
        st.caption("No data for: " + ", ".join(missing[:60]) + (" …" if len(missing) > 60 else ""))


__all__ = ["render", "build_conditions", "PRESETS", "DEFAULTS"]
