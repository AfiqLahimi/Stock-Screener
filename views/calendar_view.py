"""Calendar page: upcoming earnings and dividend dates for your watchlist and paper positions, plus
'watchlist intelligence' (what changed today on the stocks you follow)."""
import calendar as pycal

import numpy as np
import pandas as pd
import streamlit as st

import paper
import research as R
import ui
from core import BURSA, P, display_name, scan_universe
from views.analysis_plus import is_company

ICON = {"Earnings": "📊", "Ex-dividend": "💰", "Dividend paid": "💵"}


def _symbols(extra_bursa):
    wl = [t for t in st.session_state.get("term_stocks", []) if is_company(t)]
    try:
        pos = list(paper.load_account()["positions"])
    except Exception:
        pos = []
    out = list(dict.fromkeys(wl + pos + (list(BURSA) if extra_bursa else [])))
    return out[:60], set(pos)


def render(C):
    c1, c2 = st.columns([3, 1.4]) if not C.mobile else (st.container(), st.container())
    extra = c2.toggle("Include Bursa top 30", key="cal_b30", value=False)
    syms, held = _symbols(extra)
    c1.caption(f"Your watchlist{' + paper positions' if held else ''}{' + Bursa top 30' if extra else ''}: "
               f"{len(syms)} stocks. Dates come from Yahoo's company calendar and can change; companies confirm them "
               "in their Bursa / SEC announcements.")
    if not syms:
        st.info("Add stocks to your watchlist (Dashboard or Terminal) to see their events here.")
        return
    with st.spinner("Loading company calendars…"):
        ev = []
        for t in syms:
            e = R.events(t)
            for key, lab in (("earnings", "Earnings"), ("ex_dividend", "Ex-dividend"), ("dividend_pay", "Dividend paid")):
                if e.get(key) is not None:
                    ev.append({"date": pd.Timestamp(e[key]).normalize().tz_localize(None) if pd.Timestamp(e[key]).tzinfo
                               else pd.Timestamp(e[key]).normalize(), "ticker": t, "event": lab})
    today = pd.Timestamp.now().normalize()
    evdf = pd.DataFrame(ev)
    left, right = st.columns([1.25, 1]) if not C.mobile else (st.container(), st.container())

    with left, ui.panel("Upcoming events", "next 90 days"):
        if evdf.empty:
            st.markdown("<div class='muted'>N/A: Yahoo has no published dates for these stocks right now.</div>",
                        unsafe_allow_html=True)
        else:
            up = evdf[(evdf["date"] >= today) & (evdf["date"] <= today + pd.Timedelta(days=90))].sort_values("date")
            if up.empty:
                st.markdown("<div class='muted'>No published events in the next 90 days.</div>", unsafe_allow_html=True)
            rows = []
            for _, r in up.iterrows():
                days = (r["date"] - today).days
                when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
                note = {"Earnings": "results day: price can gap either way",
                        "Ex-dividend": "buy before this date to receive the dividend; price usually drops by about "
                                       "the dividend amount", "Dividend paid": "cash paid to holders"}[r["event"]]
                rows.append(f"<tr><td class='mono' style='white-space:nowrap'>{r['date']:%a %d %b}</td>"
                            f"<td style='white-space:nowrap'>{when}</td><td><b>{display_name(r['ticker'])}</b>"
                            f"{' <span class=chip>held</span>' if r['ticker'] in held else ''}</td>"
                            f"<td>{ICON[r['event']]} {r['event']}<div class='muted' style='font-size:11.5px'>{note}</div></td></tr>")
            if C.mobile:
                cards = []
                for _, r in up.iterrows():
                    days = (r["date"] - today).days
                    when = "today" if days == 0 else "tomorrow" if days == 1 else f"in {days} days"
                    cards.append(f"<div class='op'><span class='mono'>{r['date']:%a %d %b}</span> · {when}<br>"
                                 f"<b>{display_name(r['ticker'])}</b>{' <span class=chip>held</span>' if r['ticker'] in held else ''}"
                                 f" · {ICON[r['event']]} {r['event']}</div>")
                st.markdown("".join(cards), unsafe_allow_html=True)
            else:
                st.markdown("<table class='ev'><tr><th>Date</th><th></th><th>Stock</th><th>Event</th></tr>" +
                            "".join(rows) + "</table>", unsafe_allow_html=True)
            recent = evdf[(evdf["date"] < today) & (evdf["date"] >= today - pd.Timedelta(days=14))]
            if len(recent):
                st.caption("Last 14 days: " + " · ".join(f"{display_name(r.ticker)} {r.event.lower()} {r.date:%d %b}"
                                                          for r in recent.sort_values("date").itertuples()))

    with right, ui.panel("Month view", ""):
        mon = st.segmented_control("Month", ["This month", "Next month"], default="This month", key="cal_mon",
                                   label_visibility="collapsed") or "This month"
        base = today.replace(day=1) if mon == "This month" else (today.replace(day=1) + pd.offsets.MonthBegin(1))
        st.markdown(month_grid(base, evdf, today), unsafe_allow_html=True)

    with ui.panel("Watchlist intelligence", "what changed on your stocks (daily candles)"):
        wl = [t for t in syms if t in st.session_state.get("term_stocks", []) or t in held]
        intelligence(wl, evdf, today)

    with ui.panel("Economic calendar", "official sources"):
        st.markdown("<div class='muted' style='font-size:13px;line-height:1.6'>There is no reliable free feed of "
                    "economic-release dates, so this app doesn't guess them. Key official calendars: "
                    "<a href='https://www.bnm.gov.my/monetary-policy-statements' target='_blank'>Bank Negara MPC "
                    "statements</a> (OPR decisions) · <a href='https://www.dosm.gov.my/portal-main/release-calendar' "
                    "target='_blank'>DOSM release calendar</a> (Malaysia CPI, GDP, trade) · "
                    "<a href='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm' target='_blank'>US "
                    "Federal Reserve FOMC meetings</a> · <a href='https://www.bls.gov/schedule/news_release/' "
                    "target='_blank'>US BLS schedule</a> (CPI, jobs). The 🌐 Macro page shows the latest values."
                    "</div>", unsafe_allow_html=True)


def month_grid(first, evdf, today):
    y, m = first.year, first.month
    cal = pycal.Calendar(firstweekday=0).monthdatescalendar(y, m)
    by_day = {}
    if not evdf.empty:
        for r in evdf.itertuples():
            by_day.setdefault(r.date.date(), []).append(f"{ICON[r.event]} {display_name(r.ticker)}")
    head = "".join(f"<th>{d}</th>" for d in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"))
    rows = []
    for wk in cal:
        cells = []
        for d in wk:
            items = by_day.get(d, [])
            style = "opacity:.35" if d.month != m else ""
            ring = f"box-shadow:inset 0 0 0 1px {P['accent']};" if d == today.date() else ""
            cells.append(f"<td style='{style};{ring}'><div class='dn'>{d.day}</div>" +
                         "".join(f"<div class='ei'>{x}</div>" for x in items[:3]) +
                         (f"<div class='ei muted'>+{len(items) - 3}</div>" if len(items) > 3 else "") + "</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (f"<div class='muted' style='font-size:12px;margin-bottom:4px'>{first:%B %Y}</div>"
            f"<table class='mcal'><tr>{head}</tr>{''.join(rows)}</table>"
            f"<style>.mcal{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:11px}}"
            f".mcal th{{color:{P['muted']};font-weight:600;padding:4px}}"
            f".mcal td{{border:1px solid {P['border']};vertical-align:top;height:58px;padding:3px 4px;overflow:hidden}}"
            f".mcal .dn{{color:{P['muted']};font-size:10.5px}}.mcal .ei{{white-space:nowrap;overflow:hidden;"
            f"text-overflow:ellipsis}}</style>")


def intelligence(wl, evdf, today):
    if not wl:
        st.markdown("<div class='muted'>Your watchlist is empty.</div>", unsafe_allow_html=True)
        return
    s = scan_universe(tuple(wl), "1d", "2y", "trading days")
    if s.empty:
        st.markdown("<div class='muted'>N/A: no price data right now.</div>", unsafe_allow_html=True)
        return
    alerts = []
    for rd in s.to_dict("records"):
        n = rd["Name"]
        if rd["Breakout"]:
            alerts.append((2, "🚀", n, f"closed above its 20-day high (RVOL {rd['RVOL'] or 0:.1f}×)"))
        if rd["Breakdown"]:
            alerts.append((2, "🔻", n, "closed below its 20-day low"))
        if rd["New 52w high"]:
            alerts.append((2, "🏔️", n, "new 52-week high"))
        for base, lab in (("X 50/200", ("golden cross", "death cross")), ("X 20/50", ("EMA 20/50 bullish cross",
                                                                                         "EMA 20/50 bearish cross"))):
            dir_, age = rd[f"{base} dir"], rd[f"{base} age"]
            if dir_ and age is not None and age == age and age <= 3:
                alerts.append((2, "✨" if dir_ > 0 else "💀", n, f"{lab[0] if dir_ > 0 else lab[1]} {int(age)} bar{'' if int(age) == 1 else 's'} ago"))
        dir_, age = rd["X MACD dir"], rd["X MACD age"]
        if dir_ and age is not None and age == age and age <= 1:
            alerts.append((1, "〽️", n, f"MACD crossed {'above' if dir_ > 0 else 'below'} its signal"))
        if rd["RSI"] >= 70:
            alerts.append((1, "🔥", n, f"RSI {rd['RSI']:.0f}: overbought"))
        elif rd["RSI"] <= 30:
            alerts.append((1, "🧊", n, f"RSI {rd['RSI']:.0f}: oversold"))
        if (rd["RVOL"] or 0) >= 2:
            alerts.append((1, "📢", n, f"volume {rd['RVOL']:.1f}× normal"))
        if abs(rd["Change %"]) >= 4:
            alerts.append((1, "⚡", n, f"moved {rd['Change %']:+.1f}% today"))
        if not evdf.empty:
            soon = evdf[(evdf["ticker"] == rd["Ticker"]) & (evdf["date"] >= today) &
                        (evdf["date"] <= today + pd.Timedelta(days=7))]
            for e in soon.itertuples():
                alerts.append((2, ICON[e.event], n, f"{e.event.lower()} on {e.date:%a %d %b}"))
    if not alerts:
        st.markdown("<div class='muted'>Quiet day: no breakouts, crosses, extreme RSI, volume spikes or events within "
                    "a week on your watchlist.</div>", unsafe_allow_html=True)
    else:
        alerts.sort(key=lambda a: (-a[0], a[2]))
        st.markdown("".join(f"<div class='op' style='border-left-color:{P['warn'] if a[0] == 2 else P['accent']}'>"
                            f"{a[1]} <b>{a[2]}</b> · {a[3]}</div>" for a in alerts), unsafe_allow_html=True)
    sig = s[["Name", "Price", "Change %", "Signal", "Score", "RSI", "RVOL"]].sort_values("Score", ascending=False)
    st.dataframe(sig, hide_index=True, width="stretch", height="content",
                 column_config={"Price": st.column_config.NumberColumn(format="%.3f"),
                                "Change %": st.column_config.NumberColumn(format="%+.2f%%"),
                                "RSI": st.column_config.NumberColumn(format="%.0f"),
                                "RVOL": st.column_config.NumberColumn(format="%.1f×")})
