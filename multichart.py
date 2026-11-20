"""Multi-Chart: watch up to 6 stocks side by side, each with its signal, score and key levels."""
import streamlit as st

import ui


LAYOUTS = {"3 × 2": 3, "2 × 3": 2, "1 column": 1}


def _set_default(i, universe, extra):
    key = f"mc_{i}"
    if key not in st.session_state:
        pool = list(dict.fromkeys(list(extra) + list(universe or []) + ["1155.KL", "1295.KL", "5347.KL",
                                                                         "1023.KL", "6888.KL", "5225.KL"]))
        st.session_state[key] = pool[i % len(pool)]


def render(display_name, normalise, P, pill, all_symbols, universe, stock_view, make_chart, LABEL_COLOR,
           CHART_PANELS, MC_TF, MOBILE):
    st.markdown("### 🧩 Multi-Chart")
    st.caption("Six charts at once. Pick any Bursa or US stock in each slot; one timeframe for all.")
    opts = all_symbols()
    watch = st.session_state.get("term_stocks", [])
    for i in range(6):
        _set_default(i, universe, watch)

    c1, c2, c3, c4 = st.columns([1.2, 1, 1.4, 1])
    tf = c1.segmented_control("Timeframe", list(MC_TF), default="1D", key="mc_tf") or "1D"
    layout = c2.selectbox("Layout", list(LAYOUTS), index=2 if MOBILE else 0, key="mc_layout")
    panels = c3.multiselect("Panels under each chart", CHART_PANELS, default=["Volume"], key="mc_panels")
    n = c4.selectbox("Charts", [2, 4, 6], index=2, key="mc_n")
    ncol = LAYOUTS[layout]

    with st.expander("✏️ Choose the stocks", expanded=False):
        pc = st.columns(3 if not MOBILE else 1)
        for i in range(n):
            cur = st.session_state[f"mc_{i}"]
            o = opts if cur in opts else [cur] + opts
            pc[i % len(pc)].selectbox(f"Chart {i + 1}", o, key=f"mc_{i}",
                                      format_func=lambda t: f"{display_name(t)} · {t}")
        typed = st.text_input("Not in the list? Type a code and pick the slot", placeholder="e.g. 0331 or PLTR",
                              key="mc_typed")
        slot = st.selectbox("Slot", list(range(1, n + 1)), key="mc_slot")
        def _put():
            t = st.session_state.get("mc_typed", "").strip()
            if t:
                st.session_state[f"mc_{st.session_state.get('mc_slot', 1) - 1}"] = normalise(t)
                st.session_state["mc_typed"] = ""
        st.button("Put it in that slot", on_click=_put)

    if ncol == 3 and n == 4:
        ncol = 2
    height = (300 if ncol == 1 and MOBILE else 290 if ncol == 3 else 340) + 70 * len(panels)
    rows = [list(range(r, min(r + ncol, n))) for r in range(0, n, ncol)]
    for row in rows:
        cols = st.columns(ncol)
        for j, i in enumerate(row):
            t = st.session_state[f"mc_{i}"]
            with cols[j]:
                v = stock_view(t, tf)
                if v is None:
                    st.markdown(f"<div class='card'><b>{display_name(t)}</b> <span class='muted'>{t}</span><br>"
                                f"<span class='muted'>Chart N/A: not enough price data.</span></div>",
                                unsafe_allow_html=True)
                    continue
                col = P["up"] if v["chg"] >= 0 else P["down"]
                p = v["plan"]
                sig = f"{v['label']} · {v['sc']:+d}"
                st.markdown(
                    f"<div class='card' style='padding:10px 12px;margin-bottom:4px'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;gap:6px;flex-wrap:wrap'>"
                    f"<div><b>{display_name(t)}</b> <span class='muted'>{t}</span></div>"
                    f"{pill(sig, LABEL_COLOR[v['label']])}</div>"
                    f"<div><span style='font-size:18px;font-weight:600'>{v['price']:,.3f}</span> "
                    f"<span style='color:{col}'>{v['chg']:+.2f}%</span></div>"
                    f"<div class='muted' style='font-size:12px'>S {p['s1']:.3f} · R {p['t1']:.3f} · "
                    f"stop {p['stop']:.3f}{' · ' + v['ax']['regime'] if v['ax'] else ''}</div></div>",
                    unsafe_allow_html=True)
                fig = make_chart(v["d"], v["sup"], v["res"], p, display_name(t), v["iv"],
                                 show_bars=min(v["bars"], 90), panels=panels,
                                 overlays=("EMA 20/50/200", "Support/resistance & zones"), mobile=True,
                                 height=height)
                fig.update_layout(showlegend=False)
                fig.update_xaxes(nticks=4 if ncol > 1 else 6)
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "displayModeBar": False},
                                key=f"mc_fig_{i}")
                b1, b2 = st.columns(2)
                b1.button("📊 Analysis", key=f"mc_an_{i}", width="stretch",
                          on_click=ui.goto, args=("analysis",), kwargs={"an_typed": t})
                b2.button("💼 Trade", key=f"mc_pt_{i}", width="stretch",
                          on_click=ui.goto, args=("paper",), kwargs={"pt_symbol": t, "pt_other": ""})
    st.caption("Lines: green = support, red = resistance, orange dashed = suggested stop, EMAs 20/50/200. "
               "📊 Analysis / 💼 Trade open that stock on those pages.")
