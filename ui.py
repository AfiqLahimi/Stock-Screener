"""Design system and shared UI pieces: theme CSS, top bar, page switching and the compact stock panel.

The colour palette lives in core.P and is unchanged; this file only decides how things are laid out.
"""
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st

import ta_plus
from core import (CHART_PANELS, LABEL_COLOR, MC_TF, P, display_name, make_chart, pill, regime_color, rgba,
                  stock_view)

import re

# Widget values that should survive switching pages. Only plain input widgets are listed: buttons, uploaders and
# table selections are read-only in Streamlit and must never be re-assigned.
_KEEP = re.compile(r"^(pick|an_typed|panels|overlays|extra_sel|scr_(filter|c_[a-z0-9]+|mode|cols)|ts_[a-z]+"
                   r"|term_(add|search|rng|ema|engine|cat|filter|layout|h|ov|pn|type)|mc_(\d|tf|layout|panels|n|slot)"
                   r"|pt_(symbol|other|side|type|lots|shares|price|use_sl|sl|use_tp|tp|manage|f\d|smyr|susd)"
                   r"|ptc_(tf|panels)|div_(amt|lots|mode|pick|shares|sort)|gl_(cat|q)|hm_(metric|size)|ipo_kw"
                   r"|pat_filter|picks_(mode|n)|tvhm_(h|src)|db_(idx|rng)|sb_[a-z]+|cmp_(syms|rng)|rk_src|cal_(b30|mon)|mac_fred|fin_(kind|per|all|row)|dcf_[a-z0-9]+|note_tag|nt_(q|tags)|cm_(sel|grp|rng)|w_(t_[a-z_]+|f_[a-z_]+|combo)|ai_(ov|pn)|fu_hist)$")


def detect_mobile():
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        ua = ""
    return any(k in ua for k in ("iPhone", "Android", "Mobile", "iPod"))


def is_mobile():
    return bool(st.session_state.get("_mobile", False))


def keep_state():
    """Streamlit forgets a page's widget values when you open another page. Re-assigning them at the start of
    every run keeps your picks (stock, timeframe, order ticket, trade settings…) when you come back."""
    for k in list(st.session_state.keys()):
        if not isinstance(k, str) or not _KEEP.match(k):
            continue
        try:
            st.session_state[k] = st.session_state[k]
        except Exception:
            pass


def goto(page, **state):
    """Use as a button callback: set some values, then open another page (handled in app.py)."""
    st.session_state.update(state)
    st.session_state["_goto"] = page


def status_chip(label, is_open):
    col = P["up"] if is_open else P["muted"]
    return (f"<span class='chip'><i style='background:{col};box-shadow:0 0 0 3px {rgba(col, .18)}'></i>"
            f"{label} {'open' if is_open else 'closed'}</span>")


def topbar(title, icon):
    now = pd.Timestamp.now(tz="Asia/Kuala_Lumpur")
    chips = status_chip("Bursa", ta_plus.market_open("1155.KL")) + status_chip("US", ta_plus.market_open("AAPL"))
    st.markdown(f"""
<div class="topbar">
  <div class="brand"><div class="logo">📈</div><div><div class="bname">Stock Screener &amp; Analysis</div>
  <div class="crumb">{icon} {title}</div></div></div>
  <div class="tb-right">{chips}<span class="chip muted">Updated {now:%d %b %H:%M} MYT · Bursa ~15 min delayed</span></div>
</div>""", unsafe_allow_html=True)


PHONE_ORDER = ("dashboard", "portfolio", "screener", "analysis", "terminal", "multichart", "paper", "risk", "compare",
               "calendar", "macro", "commodities", "notes", "picks", "heatmap", "dividends", "ipo", "tradingview", "learn")


def phone_nav(pages, current):
    """One swipeable row of page buttons under the top bar (phones hide the sidebar menu)."""
    with st.container(horizontal=True, key="phone_nav", gap="small"):
        for k in PHONE_ORDER:
            pg = pages[k]
            st.page_link(pg, label=pg.title.replace(" & Forecast", ""), icon=pg.icon)


TAPE = [("^KLSE", "KLCI"), ("^GSPC", "S&P 500"), ("^IXIC", "NASDAQ"), ("^DJI", "DOW"), ("MYR=X", "USD/MYR"),
        ("GC=F", "GOLD"), ("CL=F", "CRUDE"), ("^VIX", "VIX"), ("BTC-USD", "BTC")]


def ticker_tape():
    """Scrolling strip of the main markets under the top bar (one batched quote request, cached 60 s)."""
    try:
        import terminal
        q = terminal.load_quotes(tuple(t for t, _ in TAPE))
    except Exception:
        return
    items = []
    for t, name in TAPE:
        if t not in q.index or q.loc[t, "last"] != q.loc[t, "last"]:
            continue
        last, pct = float(q.loc[t, "last"]), q.loc[t, "pct"]
        col = P["muted"] if pct != pct else (P["up"] if pct >= 0 else P["down"])
        arrow = "" if pct != pct else ("▲" if pct >= 0 else "▼")
        dec = 4 if t == "MYR=X" else 2
        items.append(f"<span class='tk'><b>{name}</b><span class='mono'>{last:,.{dec}f}</span>"
                     f"<span class='mono' style='color:{col}'>{arrow} {'N/A' if pct != pct else f'{pct:+.2f}%'}</span></span>")
    if items:
        strip = "".join(items)
        st.markdown(f"<div class='tape'><div class='tape-track'>{strip}{strip}</div></div>", unsafe_allow_html=True)


@contextmanager
def guard(label):
    """Show a contained error instead of crashing the page when live data has a shape nobody expected.
    Streamlit's own rerun/stop signals are BaseException subclasses, so they still pass through."""
    try:
        yield
    except Exception as e:  # noqa: BLE001
        import traceback
        st.error(f"{label} couldn't be shown: {type(e).__name__}: {str(e)[:200]}. The rest of the app still works; "
                 "try again in a minute or pick another stock.")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception(e))[-3000:])


@contextmanager
def panel(title, sub=None, key=None):
    """A bordered panel with its own header bar (title left, note right), like a terminal window."""
    k = key or re.sub(r"\W+", "_", title.lower()).strip("_")
    with st.container(border=True, key=f"pnl_{k}"):
        sub_html = f"<span class='sub'>{sub}</span>" if sub else ""
        st.markdown(f"<div class='panel-h'><div class='ph-t'>{title}</div>{sub_html}</div>", unsafe_allow_html=True)
        yield


def section(title, sub=None):
    st.markdown(f"<div class='sec'><span>{title}</span>{f'<em>{sub}</em>' if sub else ''}</div>",
                unsafe_allow_html=True)


def inject_css():
    st.markdown(CSS.format(**P, up_ring=rgba(P["up"], .2)), unsafe_allow_html=True)


def render_stock_panel(ticker, key, use_plan_cb=None):
    """Chart of one stock with a short analysis box next to it (stacked on phones)."""
    t1, t2 = st.columns([1, 1.6])
    tf = t1.segmented_control("Chart timeframe", list(MC_TF), default="1D", key=f"{key}_tf") or "1D"
    panels = t2.multiselect("Panels", CHART_PANELS, default=["Volume", "RSI"], key=f"{key}_panels")
    v = stock_view(ticker, tf)
    if v is None:
        st.info(f"Chart: N/A. Not enough price data for {display_name(ticker)} on this timeframe.")
        return None
    MOBILE = is_mobile()
    fig = make_chart(v["d"], v["sup"], v["res"], v["plan"], display_name(ticker), v["iv"], show_bars=v["bars"],
                     panels=panels, mobile=MOBILE, height=(330 if MOBILE else 430) + 90 * len(panels))
    plan, d = v["plan"], v["d"]
    last = d.iloc[-1]
    risk = plan["entry"] - plan["stop"]
    rr = f"{plan['rr1']:.1f}" if np.isfinite(plan["rr1"]) else "N/A"
    pros = [t for p_, t in sorted(v["why"], key=lambda w: -w[0]) if p_ > 0][:3]
    cons = [t for p_, t in sorted(v["why"], key=lambda w: w[0]) if p_ < 0][:3]
    ax = v["ax"]
    box = f"""<div class="card">
<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">{pill(v['label'], LABEL_COLOR[v['label']])}
<span style="font-weight:600">Score {v['sc']:+d}</span>
{pill(f"{ax['regime']} · ADX {ax['adx']:.0f}", regime_color(ax['regime'])) if ax else pill("ADX N/A", P['muted'])}
<span class="muted">RSI {last['RSI']:.0f}</span></div>
<div style="margin:10px 0 6px;font-weight:600">{plan['action']}</div>
<table style="width:100%;font-size:13px">
<tr><td class="muted">Buy zone</td><td style="text-align:right">{plan['buy_lo']:.3f} – {plan['buy_hi']:.3f}</td></tr>
<tr><td class="muted">Entry</td><td style="text-align:right">{plan['entry']:.3f}</td></tr>
<tr><td class="muted">Stop loss</td><td style="text-align:right;color:{P['down']}">{plan['stop']:.3f}
 <span class="muted">({-risk / plan['entry'] * 100 if plan['entry'] else 0:.1f}%)</span></td></tr>
<tr><td class="muted">Target 1 / 2</td><td style="text-align:right;color:{P['up']}">{plan['t1']:.3f} / {plan['t2']:.3f}</td></tr>
<tr><td class="muted">Reward : risk (T1)</td><td style="text-align:right">{rr}</td></tr>
<tr><td class="muted">Support / resistance</td><td style="text-align:right">{plan['s1']:.3f} / {plan['t1']:.3f}</td></tr>
</table>
<div style="margin-top:8px;font-size:12.5px">{''.join(f"<div style='color:{P['up']}'>＋ {t}</div>" for t in pros)}
{''.join(f"<div style='color:{P['down']}'>－ {t}</div>" for t in cons)}</div>
<div class="muted" style="font-size:11.5px;margin-top:6px">{tf} candles · last {d.index[-1].strftime('%d %b %Y %H:%M')}.
Technical read only, not advice.</div></div>"""
    if MOBILE:
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=f"{key}_fig")
        st.markdown(box, unsafe_allow_html=True)
    else:
        cc1, cc2 = st.columns([2.3, 1])
        cc1.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True}, key=f"{key}_fig")
        cc2.markdown(box, unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    if use_plan_cb and plan["entry"] > plan["stop"]:
        b1.button("📋 Use this plan in the order ticket", key=f"{key}_use", width="stretch",
                  on_click=use_plan_cb, args=(ticker, plan))
    b2.button("📊 Open full analysis", key=f"{key}_full", width="stretch",
              on_click=goto, args=("analysis",), kwargs={"an_typed": ticker})
    return v


# ----------------------------------------------------------------------------
# Theme CSS. Same palette as before; tighter spacing, smaller radii, tabular numbers, sidebar navigation.
# ----------------------------------------------------------------------------
CSS = """
<style>
html, body, [data-testid="stAppViewContainer"]{{font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}}
.mono, .idx .v, .idx .c, [data-testid="stMetricValue"], [data-testid="stMetricDelta"], .head .price, .head .chg,
.pl b, .tk .mono{{font-family:SF Mono, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;font-variant-numeric:tabular-nums}}
/* ---------- ticker tape */
.tape{{overflow:hidden;white-space:nowrap;border:1px solid {border};border-radius:10px;background:{card};
  margin:-4px 0 14px;user-select:none;mask-image:linear-gradient(90deg,transparent,#000 4%,#000 96%,transparent)}}
.tape-track{{display:inline-block;padding:6px 0;animation:tape 70s linear infinite}}
.tape:hover .tape-track{{animation-play-state:paused}}
@keyframes tape{{from{{transform:translateX(0)}}to{{transform:translateX(-50%)}}}}
.tk{{display:inline-flex;align-items:baseline;gap:7px;margin:0 18px;font-size:12px}}
.tk b{{color:{muted};font-weight:600;letter-spacing:.04em}}
/* ---------- panels (bordered container with a header bar) */
[class*="st-key-pnl_"]{{background:{card};border:1px solid {border} !important;border-radius:12px !important;
  padding:11px 15px 14px !important;gap:.55rem !important;overflow:hidden}}
.panel-h{{display:flex;align-items:baseline;gap:10px;margin:-11px -15px 4px;padding:10px 15px 9px;
  border-bottom:1px solid {border};background:{card2}}}
.panel-h .ph-t{{font-size:11.5px;text-transform:uppercase;letter-spacing:.12em;color:{text};font-weight:700;
  line-height:1.3}}
.panel-h .sub{{font-size:11px;color:{muted};margin-left:auto;text-align:right}}
[class*="st-key-pnl_"] .card{{background:transparent;border:none;padding:0;margin:0}}
[class*="st-key-pnl_"] [data-testid="stMetric"]{{background:{bg}}}
.block-container{{padding-top:3.9rem;padding-bottom:2.5rem;max-width:1560px}}
h1,h2,h3,h4{{letter-spacing:-0.01em}}
h3{{font-size:1.25rem !important}}
/* ---------- top bar */
.topbar{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;padding:2px 0 12px;margin-bottom:14px;
  border-bottom:1px solid {border}}}
.topbar .brand{{display:flex;align-items:center;gap:11px}}
.topbar .logo{{width:34px;height:34px;border-radius:9px;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,{accent},{up});font-size:18px}}
.topbar .bname{{font-size:13px;font-weight:800;line-height:1.15;letter-spacing:.12em;text-transform:uppercase}}
.topbar .crumb{{font-size:12.5px;color:{muted};margin-top:1px}}
.tb-right{{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}}
.chip{{font-size:11.5px;color:{text};display:inline-flex;align-items:center;gap:7px;background:{card};
  border:1px solid {border};padding:4px 10px;border-radius:999px;white-space:nowrap}}
.chip.muted{{color:{muted}}}
.chip i{{width:7px;height:7px;border-radius:50%;display:inline-block}}
.sec{{display:flex;align-items:baseline;gap:10px;margin:14px 0 8px;font-size:11.5px;font-weight:700;
  text-transform:uppercase;letter-spacing:.12em;color:{muted}}}
.sec em{{font-style:normal;font-weight:400;text-transform:none;letter-spacing:0}}
/* ---------- metric cards */
[data-testid="stMetric"]{{background:{card};border:1px solid {border};border-radius:10px;padding:11px 15px}}
[data-testid="stMetricLabel"] p{{color:{muted} !important;font-size:10.5px !important;text-transform:uppercase;letter-spacing:.1em}}
[data-testid="stMetricValue"]{{font-size:19px !important;font-weight:700}}
[data-testid="stMetricDelta"]{{font-size:12px !important}}
/* ---------- tabs inside a page */
.stTabs [data-baseweb="tab-list"]{{gap:2px;border-bottom:1px solid {border}}}
.stTabs [data-baseweb="tab"]{{padding:7px 13px;border-radius:7px 7px 0 0;font-weight:600;font-size:13px;color:{muted}}}
.stTabs [aria-selected="true"]{{color:{text} !important;background:{card}}}
/* ---------- sidebar + navigation */
section[data-testid="stSidebar"]{{background:{card};border-right:1px solid {border}}}
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebarUserContent"]{{padding-top:.6rem}}
[data-testid="stNavSectionHeader"]{{font-size:11px !important;text-transform:uppercase;letter-spacing:.08em;
  color:{muted} !important;font-weight:700 !important;margin-top:6px}}
[data-testid="stSidebarNavLink"]{{border-radius:8px;padding:3px 10px;margin:1px 0}}
[data-testid="stSidebarNavLink"] span{{font-size:14px}}
[data-testid="stSidebarNavLink"][aria-current="page"]{{background:{card2};box-shadow:inset 3px 0 0 {accent}}}
[data-testid="stSidebarNavSeparator"]{{border-color:{border}}}
[data-testid="stExpander"] details{{border-radius:10px;border-color:{border}}}
/* ---------- generic pieces */
.card{{background:{card};border:1px solid {border};border-radius:12px;padding:14px 16px;margin-bottom:12px}}
.card h3{{font-size:15px !important;margin:0 0 8px;font-weight:600}}
.card .muted, .muted{{color:{muted};font-size:13px}}
.pill{{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12.5px;font-weight:600;letter-spacing:.02em}}
.head{{display:flex;flex-wrap:wrap;align-items:flex-end;gap:14px 30px}}
.head .name{{font-size:13px;color:{muted}}}
.head .price{{font-size:32px;font-weight:700;line-height:1.1}}
.head .chg{{font-size:15px;font-weight:600;margin-left:8px}}
.kv{{width:100%;border-collapse:collapse;border:none !important;margin:0 !important;display:table !important}}
.kv tr{{background:transparent !important;border:none !important}}
.kv td{{padding:6px 0 !important;border:none !important;border-bottom:1px solid {border} !important;font-size:13.5px;background:transparent !important}}
.kv tr:last-child td{{border-bottom:none}}
.kv td:first-child{{color:{muted}}}
.kv td:last-child{{text-align:right;font-weight:600}}
.reason{{display:flex;gap:10px;align-items:flex-start;padding:5px 0;font-size:13.5px;line-height:1.45}}
.reason b{{min-width:34px;text-align:right;font-weight:600}}
.note{{color:{muted};font-size:12px;line-height:1.5;margin-top:8px}}
.action{{font-size:15px;font-weight:700;margin:2px 0 8px;line-height:1.35}}
.when{{font-size:13.5px;line-height:1.6;margin-top:10px;color:{text};opacity:.9}}
a{{color:{accent}}}
.news{{padding:9px 0;border-bottom:1px solid {border}}}
.news:last-child{{border-bottom:none}}
.news a{{color:{text};text-decoration:none;font-weight:600;font-size:14px;line-height:1.45}}
.news a:hover{{color:{accent}}}
.news .meta{{color:{muted};font-size:12px;margin-top:4px}}
.news .pill{{font-size:11px;padding:1px 8px}}
.just{{margin-top:auto;padding-top:10px;font-size:13px;line-height:1.55;color:{text};opacity:.82;
  border-top:1px dashed {border};margin-top:10px}}
.live{{font-size:12px;color:{muted};display:inline-flex;align-items:center;gap:7px}}
.live i{{width:8px;height:8px;border-radius:50%;background:{up};display:inline-block;box-shadow:0 0 0 3px {up_ring}}}
[data-testid="stDataFrame"]{{border:1px solid {border};border-radius:10px}}
.foot{{color:{muted};font-size:12px;line-height:1.6;border-top:1px solid {border};padding-top:12px;margin-top:22px}}
/* ---------- shared report pieces (scores, evidence tables, alerts) */
.sg-row{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:12px}}
.sg{{background:{card};border:1px solid {border};border-radius:12px;padding:11px 14px}}
.sg-t{{font-size:10.5px;text-transform:uppercase;letter-spacing:.1em;color:{muted}}}
.sg-v{{font-family:SF Mono,ui-monospace,Menlo,Consolas,monospace;font-size:26px;font-weight:700;margin-top:4px}}
.sg-v span{{font-size:12px;color:{muted};margin-left:3px}}
.sg-s{{font-size:12px;color:{muted};margin-top:5px;line-height:1.4}}
.bar{{height:6px;background:{card2};border-radius:4px;overflow:hidden;margin-top:6px}}
.bar i{{display:block;height:100%;border-radius:4px}}
.gs-row{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin-bottom:10px}}
.ev{{width:100%;border-collapse:collapse;font-size:12.5px}}
.ev th{{text-align:left;color:{muted};font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  padding:6px 8px;border-bottom:1px solid {border}}}
.ev td{{padding:7px 8px;border-bottom:1px solid {border};vertical-align:top}}
.ev tr:last-child td{{border-bottom:none}}
.concl{{width:100%;border-collapse:collapse;font-size:13px}}
.concl td{{padding:7px 8px;border-bottom:1px solid {border};vertical-align:top;line-height:1.5}}
.concl td:first-child{{color:{muted};width:190px;font-weight:600}}
.hz{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
.hz > div{{background:{card};border:1px solid {border};border-radius:12px;padding:11px 14px}}
.op{{border:1px solid {border};border-left:3px solid {accent};border-radius:8px;padding:8px 11px;margin:6px 0;font-size:13px}}
@media (max-width: 768px){{.sg-row{{grid-template-columns:repeat(2,minmax(0,1fr)) !important}}
  .gs-row{{grid-template-columns:repeat(2,minmax(0,1fr)) !important}} .hz{{grid-template-columns:1fr !important}}
  .ev td:nth-child(4), .ev th:nth-child(4){{display:none}} .concl td:first-child{{width:110px}}}}

/* ---------- phone */
@media (max-width: 768px) {{
  html, body, [data-testid="stAppViewContainer"] {{font-size:15px}}
  .block-container{{padding:3.6rem .7rem 4rem .7rem !important}}
  .topbar{{gap:8px;padding-bottom:8px;margin-bottom:10px}}
  .topbar .logo{{width:28px;height:28px;font-size:15px;border-radius:8px}}
  .topbar .bname{{font-size:13.5px}}
  .tb-right{{margin-left:0;width:100%}}
  .chip{{font-size:10.5px;padding:3px 8px}}
  .chip.muted{{display:none}}
  .tape{{margin:0 0 10px}}
  [class*="st-key-pnl_"]{{padding:10px 12px 12px !important}}
  .panel-h{{margin:-10px -12px 4px;padding:9px 12px 8px;flex-wrap:wrap;row-gap:2px}}
  .panel-h .sub{{margin-left:0;width:100%;text-align:left;font-size:10.5px}}
  .stTabs [data-baseweb="tab-list"]{{overflow-x:auto;flex-wrap:nowrap;scrollbar-width:none;gap:2px}}
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{{display:none}}
  .stTabs [data-baseweb="tab"]{{padding:7px 10px;font-size:13.5px;white-space:nowrap;flex:0 0 auto}}
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]){{flex-wrap:wrap;gap:8px}}
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"]{{
     flex:1 1 calc(50% - 8px) !important;min-width:calc(50% - 8px) !important;width:calc(50% - 8px) !important}}
  [data-testid="stMetric"]{{padding:9px 11px}}
  [data-testid="stMetricValue"]{{font-size:19px !important}}
  div[style*="grid-template-columns"]{{grid-template-columns:1fr !important}}
  .card{{padding:12px 13px;margin-bottom:10px}}
  .head{{gap:8px 16px}}
  .head .price{{font-size:28px}}
  .head > div[style*="margin-left:auto"]{{margin-left:0 !important;width:100%}}
  .kv td{{font-size:13.5px;padding:6px 0 !important}}
  .note{{font-size:11.5px}}
  .reason{{font-size:13.5px}}
  .just{{font-size:12.5px}}
  [data-testid="stSegmentedControl"] button, [data-testid="stButtonGroup"] button{{padding:4px 9px;font-size:13px}}
  [data-testid="stButtonGroup"]{{overflow-x:auto;flex-wrap:nowrap}}
  .modebar-container{{display:none !important}}
  .st-key-phone_nav{{flex-wrap:nowrap !important;overflow-x:auto;scrollbar-width:none;margin:0 0 10px;padding:2px 0}}
  .st-key-phone_nav::-webkit-scrollbar{{display:none}}
  .st-key-phone_nav > div{{flex:0 0 auto !important;width:auto !important}}
  .st-key-phone_nav a{{background:{card};border:1px solid {border};border-radius:999px;padding:2px 11px !important;
     white-space:nowrap}}
  .st-key-phone_nav a[aria-current="page"], .st-key-phone_nav a:has(span[aria-current]){{border-color:{accent};
     background:{card2}}}
  .st-key-phone_nav a p, .st-key-phone_nav a span{{font-size:13.5px !important;white-space:nowrap}}
}}
</style>"""
