"""Commodities page: ~20 futures from Yahoo in three groups, five change windows and distance from the 5-year
high, a chart, and which Bursa stocks a move typically helps or hurts, with the measured correlation of daily
returns over the last year as evidence. Crude palm oil (FCPO) isn't on Yahoo; soybean oil is shown as the
closest free proxy because the two oils substitute for each other."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import research as R
import ui
from core import P, display_name, get_many, load_one

COMMODITIES = {
    "Energy": [("BZ=F", "Brent crude", "US$/bbl"), ("CL=F", "WTI crude", "US$/bbl"), ("NG=F", "Natural gas", "US$/MMBtu"),
               ("RB=F", "Gasoline", "US$/gal"), ("HO=F", "Heating oil (diesel)", "US$/gal")],
    "Metals": [("GC=F", "Gold", "US$/oz"), ("SI=F", "Silver", "US$/oz"), ("HG=F", "Copper", "US$/lb"),
               ("ALI=F", "Aluminium", "US$/t"), ("PL=F", "Platinum", "US$/oz"), ("PA=F", "Palladium", "US$/oz")],
    "Agriculture": [("ZL=F", "Soybean oil (palm-oil proxy)", "US¢/lb"), ("ZS=F", "Soybeans", "US¢/bu"),
                    ("ZC=F", "Corn", "US¢/bu"), ("ZW=F", "Wheat", "US¢/bu"), ("ZM=F", "Soybean meal", "US$/t"),
                    ("SB=F", "Sugar", "US¢/lb"), ("KC=F", "Coffee", "US¢/lb"), ("CC=F", "Cocoa", "US$/t"),
                    ("CT=F", "Cotton", "US¢/lb")],
}
NAME = {t: n for g in COMMODITIES.values() for t, n, _ in g}
COMPANY = {'5199.KL': 'Hibiscus Petroleum', '5243.KL': 'Velesto Energy', '7277.KL': 'Dialog Group', '5183.KL': 'Petronas Chemicals', '5099.KL': 'Capital A (AirAsia)', '5347.KL': 'Tenaga Nasional', '5080.KL': 'Poh Kong', '8869.KL': 'Press Metal', '1961.KL': 'IOI Corporation', '2445.KL': 'Kuala Lumpur Kepong', '5285.KL': 'SD Guthrie', '5222.KL': 'FGV Holdings', '7084.KL': 'QL Resources', '4065.KL': 'PPB Group', '4707.KL': 'Nestlé Malaysia'}


def company(t):
    n = display_name(t)
    return COMPANY.get(t, n) if n == t else n
UNIT = {t: u for g in COMMODITIES.values() for t, _, u in g}

# Typical first-order effects of a price RISE on Bursa companies (+ = helps, − = hurts). Reference only: the
# measured correlation next to each name shows whether the link actually showed up in the last year.
IMPACT = {
    "BZ=F": {"why": "Oil producers and service firms earn more; airlines and fuel-heavy businesses pay more. Higher "
                    "oil also lifts Malaysia's government revenue.",
             "stocks": [("5199.KL", +1, "oil producer"), ("5243.KL", +1, "drilling rigs"), ("7277.KL", +1, "O&G services"),
                        ("5183.KL", +1, "chemical prices track oil"), ("5099.KL", -1, "jet fuel cost")]},
    "CL=F": {"why": "Same story as Brent (the US benchmark).",
             "stocks": [("5199.KL", +1, "oil producer"), ("5243.KL", +1, "drilling rigs"), ("5099.KL", -1, "jet fuel cost")]},
    "NG=F": {"why": "Gas is a major fuel for power plants.",
             "stocks": [("5347.KL", -1, "fuel cost (partly passed on via tariffs)")]},
    "GC=F": {"why": "A safe-haven asset: tends to rise when investors are nervous or real interest rates fall.",
             "stocks": [("5080.KL", +1, "gold jewellery inventory")]},
    "ALI=F": {"why": "Aluminium smelters sell at world prices while their power costs are largely fixed.",
              "stocks": [("8869.KL", +1, "aluminium smelter")]},
    "HG=F": {"why": "Copper is a barometer of global industrial demand (construction, power grids, EVs, data centres).",
             "stocks": []},
    "ZL=F": {"why": "Soybean oil and palm oil substitute for each other, so they usually move together. Higher "
                    "edible-oil prices lift planters' earnings.",
             "stocks": [("1961.KL", +1, "planter"), ("2445.KL", +1, "planter"), ("5285.KL", +1, "planter"),
                        ("5222.KL", +1, "planter")]},
    "ZC=F": {"why": "Corn and soybean meal are the main animal-feed ingredients.",
             "stocks": [("7084.KL", -1, "poultry feed cost")]},
    "ZM=F": {"why": "Soybean meal is the main protein in animal feed.",
             "stocks": [("7084.KL", -1, "poultry feed cost")]},
    "ZW=F": {"why": "Wheat is the raw material for flour.",
             "stocks": [("4065.KL", -1, "flour milling cost")]},
    "SB=F": {"why": "Sugar is an input for food and drink makers.", "stocks": [("4707.KL", -1, "ingredient cost")]},
    "KC=F": {"why": "Coffee beans are a key input for instant-coffee makers.", "stocks": [("4707.KL", -1, "ingredient cost")]},
    "CC=F": {"why": "Cocoa is a key input for chocolate and malted drinks.", "stocks": [("4707.KL", -1, "ingredient cost")]},
}


def _chg(s, n):
    s = s.dropna()
    return float((s.iloc[-1] / s.iloc[-1 - n] - 1) * 100) if len(s) > n else None


def render(C):
    all_t = [t for g in COMMODITIES.values() for t, _, _ in g]
    with st.spinner("Loading commodity prices…"):
        try:
            pre = get_many(tuple(all_t), "1d", "5y")
        except Exception:
            pre = {}
    rows = []
    for grp, items in COMMODITIES.items():
        for t, name, unit in items:
            d = pre.get(t)
            if d is None or d.empty:
                d = load_one(t, "1d", "5y")
            if d is None or d.empty:
                rows.append({"Group": grp, "Commodity": name, "Ticker": t, "Last": None})
                continue
            c = d["Close"]
            rows.append({"Group": grp, "Commodity": name, "Ticker": t, "Unit": unit, "Last": float(c.iloc[-1]),
                         "1D": _chg(c, 1), "1W": _chg(c, 5), "1M": _chg(c, 21), "3M": _chg(c, 63), "1Y": _chg(c, 252),
                         "From 5Y high": float((c.iloc[-1] / c.max() - 1) * 100),
                         "Trend": c.tail(60).tolist()})
    tbl = pd.DataFrame(rows)
    g = st.segmented_control("Group", ["All"] + list(COMMODITIES), default="All", key="cm_grp",
                             label_visibility="collapsed") or "All"
    view = tbl if g == "All" else tbl[tbl["Group"] == g]
    cc = st.column_config
    with ui.panel("Commodity board", "futures prices from Yahoo · US$ unless stated"):
        st.dataframe(view.drop(columns=["Ticker"]), hide_index=True, width="stretch", height="content",
                     column_order=(["Commodity", "Last", "1D", "1M"] if C.mobile else
                                   ["Group", "Commodity", "Unit", "Last", "1D", "1W", "1M", "3M", "1Y", "From 5Y high", "Trend"]),
                     column_config={"Last": cc.NumberColumn(format="%,.2f"),
                                    **{k: cc.NumberColumn(format="%+.1f%%") for k in ("1D", "1W", "1M", "3M", "1Y", "From 5Y high")},
                                    "Trend": cc.LineChartColumn("60 days", width="small")})
        st.caption("Crude palm oil futures (FCPO, traded on Bursa) aren't available from Yahoo; soybean oil is the "
                   "closest free proxy. Rubber isn't available either.")

    opts = [r["Ticker"] for _, r in view.iterrows() if r["Last"] == r["Last"] and r["Last"] is not None]
    if not opts:
        return
    a, b = st.columns([1.5, 1]) if not C.mobile else (st.container(), st.container())
    sel = a.selectbox("Commodity", opts, key="cm_sel", format_func=lambda t: NAME[t],
                      index=opts.index("BZ=F") if "BZ=F" in opts and "cm_sel" not in st.session_state else 0)
    rng = b.segmented_control("Range", ["6M", "1Y", "2Y", "5Y"], default="1Y", key="cm_rng",
                              label_visibility="collapsed") or "1Y"
    d = pre.get(sel)
    d = d if d is not None and not d.empty else load_one(sel, "1d", "5y")
    left, right = (st.container(), st.container()) if C.mobile else st.columns([1.5, 1], gap="small")
    with left, ui.panel(NAME[sel], UNIT[sel]):
        x = d.tail({"6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260}[rng])
        fig = go.Figure(go.Scatter(x=x.index, y=x["Close"], line=dict(color=P["accent"], width=2), name=NAME[sel]))
        fig.add_hline(y=float(d["Close"].max()), line=dict(color=P["muted"], dash="dot"),
                      annotation_text="5-year high", annotation_position="top left")
        fig.update_layout(height=320, margin=dict(l=4, r=4, t=10, b=4), paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(size=11), hovermode="x unified")
        fig.update_yaxes(gridcolor=P["grid"], side="right")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key="cm_chart")
    with right, ui.panel("What a rise means for Bursa", "typical effect · measured correlation (1y)"):
        imp = IMPACT.get(sel)
        if not imp:
            st.markdown("<div class='muted'>No direct, well-known link to specific Bursa companies.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='font-size:13px;line-height:1.55'>{imp['why']}</div>", unsafe_allow_html=True)
            if imp["stocks"]:
                px = R.closes([sel] + [s for s, _, _ in imp["stocks"]], "2y")
                rc = R.returns(px[sel].dropna()) if sel in px else pd.Series(dtype=float)
                out = []
                for s, sign, why in imp["stocks"]:
                    corr = None
                    if s in px:
                        j = pd.concat([rc, R.returns(px[s].dropna())], axis=1).dropna().tail(252)
                        corr = float(j.iloc[:, 0].corr(j.iloc[:, 1])) if len(j) > 60 else None
                    s_last = px[s].dropna() if s in px else pd.Series(dtype=float)
                    chg = _chg(s_last, 1) if len(s_last) > 1 else None
                    ok = None if corr is None else (corr * sign > 0.05)
                    out.append(f"<tr><td><b>{company(s)}</b> <span class='muted'>{s}</span><div class='muted' "
                               f"style='font-size:11.5px'>{why}</div></td>"
                               f"<td style='color:{P['up'] if sign > 0 else P['down']}'>{'helps' if sign > 0 else 'hurts'}</td>"
                               f"<td class='mono'>{'N/A' if corr is None else f'{corr:+.2f}'}"
                               f"{'' if ok is None else (' ✓' if ok else ' ·')}</td>"
                               f"<td class='mono'>{'N/A' if chg is None else f'{chg:+.2f}%'}</td></tr>")
                st.markdown("<table class='ev'><tr><th>Stock</th><th>If price rises</th><th>Corr.</th><th>Today</th>"
                            "</tr>" + "".join(out) + "</table>", unsafe_allow_html=True)
                st.caption("Corr. = correlation of daily % moves over the last year (+1 move together, −1 opposite). ✓ = "
                           "the expected link showed up; · = weak or opposite last year (other news dominated). Daily "
                           "correlations between a futures price and a stock are usually modest.")
