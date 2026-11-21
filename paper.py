"""Paper trading: a virtual account with market / limit / stop orders, stop-loss & take-profit,
Bursa board lots and fees, positions, history, performance stats and an equity curve.

No real orders are ever sent anywhere. The account is saved on the app server (paper_account.json)
and mirrored to this browser's local storage, and can be downloaded/restored as a JSON backup.
"""
from pxfmt import fp, pxf  # noqa: F401
import json
import os
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ta_plus

DEMO = os.environ.get("DEMO_DATA") == "1"
STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "paper_account.json")
LS_KEY = "paper_account_v1"
TZ = "Asia/Kuala_Lumpur"

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:  # library missing: server file + manual backup still work
    streamlit_js_eval = None


# ============================================================================
# Account storage
# ============================================================================
def _now():
    return pd.Timestamp.now(tz=TZ)


def new_account(myr=10000.0, usd=2000.0):
    return {"version": 1, "created": _now().isoformat(), "cash": {"MYR": float(myr), "USD": float(usd)},
            "start": {"MYR": float(myr), "USD": float(usd)}, "positions": {}, "orders": [], "trades": [],
            "equity": [], "fees": {"brk_pct": 0.08, "brk_min": 8.0, "platform": 0.0, "us_commission": 0.0}}


def _valid(a):
    return isinstance(a, dict) and {"cash", "positions", "orders", "trades"} <= set(a)


def load_account():
    if "pt_acct" in st.session_state:
        return st.session_state["pt_acct"]
    acct = None
    try:
        with open(STORE) as f:
            acct = json.load(f)
    except Exception:
        acct = None
    if not _valid(acct):
        acct = None
    st.session_state["pt_acct"] = acct or new_account()
    st.session_state["pt_from_disk"] = acct is not None
    return st.session_state["pt_acct"]


def save_account(acct):
    st.session_state["pt_acct"] = acct
    try:
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        tmp = STORE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(acct, f, default=str)
        os.replace(tmp, STORE)
    except Exception:
        pass
    st.session_state["pt_dirty"] = True


def browser_sync():
    """Mirror the account to this browser, and restore from it when the server copy was lost (app restart)."""
    if streamlit_js_eval is None:
        return
    acct = st.session_state.get("pt_acct")
    if not st.session_state.get("pt_from_disk") and not st.session_state.get("pt_ls_checked"):
        raw = streamlit_js_eval(js_expressions=f"localStorage.getItem('{LS_KEY}') || ''", key="pt_ls_get")
        if raw is not None:
            st.session_state["pt_ls_checked"] = True
            try:
                saved = json.loads(raw) if raw else None
            except Exception:
                saved = None
            if _valid(saved) and not acct.get("trades") and not acct.get("orders") and not acct.get("positions"):
                save_account(saved)
                st.session_state["pt_from_disk"] = True
                st.toast("Paper account restored from this browser.")
                st.rerun()
    if st.session_state.pop("pt_dirty", False) and acct is not None:
        payload = json.dumps(json.dumps(acct, default=str))
        streamlit_js_eval(js_expressions=f"localStorage.setItem('{LS_KEY}', {payload})",
                          key=f"pt_ls_set_{len(acct.get('trades', []))}_{len(acct.get('orders', []))}_{uuid.uuid4().hex[:6]}")


# ============================================================================
# Prices
# ============================================================================
def ccy(ticker):
    return "MYR" if ticker.endswith(".KL") else "USD"


def sym(c):
    return "RM" if c == "MYR" else "$"


def lot_of(ticker):
    return 100 if ticker.endswith(".KL") else 1


def _demo_px(t):
    rng = np.random.default_rng(abs(hash(t)) % 2**32)
    n = 120
    c = 5 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
    o = np.r_[c[0], c[:-1]]
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.99, "Close": c},
                        index=idx)


@st.cache_data(ttl=60, show_spinner=False, max_entries=150)
def daily(ticker):
    if DEMO:
        return _demo_px(ticker)
    try:
        import yfinance as yf
        d = yf.download(ticker, period="6mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close"]].dropna()
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)
        return d
    except Exception:
        return pd.DataFrame()


def last_price(ticker):
    d = daily(ticker)
    return float(d["Close"].iloc[-1]) if len(d) else None


def round_price(p, ticker):
    return ta_plus.round_tick(p, ticker) if ticker.endswith(".KL") else round(p, 2)


# ============================================================================
# Execution
# ============================================================================
def _fee(acct, value, ticker):
    return ta_plus.fees(value, ticker, {**acct["fees"]})["total"]


def _buy(acct, ticker, qty, price, when, reason, sl=None, tp=None):
    c = ccy(ticker)
    value = qty * price
    fee = _fee(acct, value, ticker)
    if value + fee > acct["cash"][c] + 1e-6:
        return False, f"Not enough {c} cash: need {sym(c)}{value + fee:,.2f}, have {sym(c)}{acct['cash'][c]:,.2f}"
    acct["cash"][c] -= value + fee
    pos = acct["positions"].get(ticker)
    if pos:
        tot = pos["qty"] * pos["avg"] + value + fee
        pos["qty"] += qty
        pos["avg"] = tot / pos["qty"]
        if sl is not None:
            pos["sl"] = sl
        if tp is not None:
            pos["tp"] = tp
    else:
        acct["positions"][ticker] = {"qty": qty, "avg": (value + fee) / qty, "opened": str(when), "sl": sl, "tp": tp}
    acct["trades"].append({"time": str(when), "ticker": ticker, "side": "BUY", "qty": qty, "price": price,
                           "fees": fee, "pnl": None, "reason": reason})
    return True, f"Bought {qty:,} {ticker} @ {fp(price)} (fees {sym(c)}{fee:.2f})"


def _sell(acct, ticker, qty, price, when, reason):
    pos = acct["positions"].get(ticker)
    if not pos or qty > pos["qty"]:
        return False, "You don't hold that many shares (paper trading has no short selling)."
    c = ccy(ticker)
    value = qty * price
    fee = _fee(acct, value, ticker)
    pnl = value - fee - pos["avg"] * qty
    acct["cash"][c] += value - fee
    pos["qty"] -= qty
    if pos["qty"] == 0:
        del acct["positions"][ticker]
        for o in acct["orders"]:  # cancel leftover sell orders for a closed position
            if o["ticker"] == ticker and o["side"] == "SELL" and o["status"] == "OPEN":
                o["status"], o["note"] = "CANCELLED", "position closed"
    acct["trades"].append({"time": str(when), "ticker": ticker, "side": "SELL", "qty": qty, "price": price,
                           "fees": fee, "pnl": pnl, "reason": reason})
    return True, f"Sold {qty:,} {ticker} @ {fp(price)}, P/L {sym(c)}{pnl:,.2f}"


def place_order(acct, ticker, side, otype, qty, price=None, sl=None, tp=None):
    lot = lot_of(ticker)
    if qty <= 0 or qty % lot:
        return False, f"Quantity must be a multiple of {lot}" + (" (1 lot = 100 shares on Bursa)." if lot == 100 else ".")
    now = _now()
    if side == "BUY" and (sl is not None or tp is not None):
        cur = last_price(ticker)
        fill_ref = cur if otype == "MARKET" or cur is None else (min(price, cur) if otype == "LIMIT" else max(price, cur))
        if fill_ref is not None and sl is not None and sl >= fill_ref:
            return False, f"Stop loss {fp(sl)} is at/above the price this would fill at (~{fp(fill_ref)}). " \
                          "Lower the stop or the entry."
        if fill_ref is not None and tp is not None and tp <= fill_ref:
            return False, f"Take profit {fp(tp)} is at/below the price this would fill at (~{fp(fill_ref)})."
    if side == "SELL":
        held = acct["positions"].get(ticker, {}).get("qty", 0)
        reserved = sum(o["qty"] for o in acct["orders"] if o["ticker"] == ticker and o["side"] == "SELL"
                       and o["status"] == "OPEN")
        if qty > held - reserved:
            return False, f"You can sell at most {max(held - reserved, 0):,} shares (held {held:,}, " \
                          f"{reserved:,} already in open sell orders)."
    if otype == "MARKET":
        px = last_price(ticker)
        if px is None:
            return False, "No price available for this symbol right now."
        px = round_price(px, ticker)
        if side == "BUY":
            ok, msg = _buy(acct, ticker, qty, px, now, "Market order", sl, tp)
        else:
            ok, msg = _sell(acct, ticker, qty, px, now, "Market order")
        if ok:
            acct["orders"].append({"id": uuid.uuid4().hex[:8], "ticker": ticker, "side": side, "type": "MARKET",
                                   "qty": qty, "price": None, "sl": sl, "tp": tp, "created": str(now),
                                   "status": "FILLED", "fill_price": px, "filled_at": str(now), "note": ""})
        return ok, msg + ("" if ta_plus.market_open(ticker) else " · market closed: filled at the last price")
    if price is None or price <= 0:
        return False, "Enter a valid limit/stop price."
    price = round_price(price, ticker)
    if side == "BUY":
        c = ccy(ticker)
        need = qty * price + _fee(acct, qty * price, ticker)
        if need > acct["cash"][c] + 1e-6:
            return False, f"Not enough {c} cash for this order ({sym(c)}{need:,.2f} needed)."
    acct["orders"].append({"id": uuid.uuid4().hex[:8], "ticker": ticker, "side": side, "type": otype, "qty": qty,
                           "price": price, "sl": sl, "tp": tp, "created": str(now), "status": "OPEN",
                           "fill_price": None, "filled_at": None, "note": ""})
    return True, f"{otype.title()} {side.lower()} order placed: {qty:,} {ticker} @ {fp(price)} (good till cancelled)"


def _touch(order, bar):
    """Fill price if this daily bar triggers the order, else None (gaps fill at the open)."""
    o, h, l = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
    p = order["price"]
    if order["side"] == "BUY" and order["type"] == "LIMIT" and l <= p:
        return min(p, o)
    if order["side"] == "SELL" and order["type"] == "LIMIT" and h >= p:
        return max(p, o)
    if order["side"] == "BUY" and order["type"] == "STOP" and h >= p:
        return max(p, o)
    if order["side"] == "SELL" and order["type"] == "STOP" and l <= p:
        return min(p, o)
    return None


def process(acct):
    """Fill pending orders and trigger stop-loss / take-profit using daily candles after they were placed.
    Uses only completed candles dated after the order/position date, plus the current price."""
    events = []
    for order in acct["orders"]:
        if order["status"] != "OPEN":
            continue
        d = daily(order["ticker"])
        if d.empty:
            continue
        created = pd.Timestamp(order["created"]).tz_localize(None).normalize()
        bars = d[d.index > created]
        fill, when = None, None
        for ts, bar in bars.iterrows():
            fill = _touch(order, bar)
            if fill is not None:
                when = ts
                break
        if fill is None:  # same-day: compare the current price only
            px = float(d["Close"].iloc[-1])
            live = {"Open": px, "High": px, "Low": px}
            fill = _touch(order, live)
            when = _now() if fill is not None else None
        if fill is None:
            continue
        fill = round_price(fill, order["ticker"])
        if order["side"] == "BUY":
            ok, msg = _buy(acct, order["ticker"], order["qty"], fill, when, f"{order['type'].title()} order",
                           order.get("sl"), order.get("tp"))
        else:
            ok, msg = _sell(acct, order["ticker"], order["qty"], fill, when, f"{order['type'].title()} order")
        order["status"] = "FILLED" if ok else "REJECTED"
        order["fill_price"], order["filled_at"] = (fill, str(when)) if ok else (None, None)
        order["note"] = "" if ok else msg
        events.append(msg)
    for t in list(acct["positions"]):
        pos = acct["positions"][t]
        if pos.get("sl") is None and pos.get("tp") is None:
            continue
        d = daily(t)
        if d.empty:
            continue
        since = pd.Timestamp(pos.get("bracket_from") or pos["opened"]).tz_localize(None).normalize()
        bars = list(d[d.index > since].iterrows())
        px = float(d["Close"].iloc[-1])
        bars.append((_now(), {"Open": px, "High": px, "Low": px}))
        for ts, bar in bars:
            o, h, l = float(bar["Open"]), float(bar["High"]), float(bar["Low"])
            if pos.get("sl") is not None and l <= pos["sl"]:  # stop checked first (conservative)
                ok, msg = _sell(acct, t, pos["qty"], round_price(min(pos["sl"], o), t), ts, "Stop loss")
                events.append("🛑 " + msg)
                break
            if pos.get("tp") is not None and h >= pos["tp"]:
                ok, msg = _sell(acct, t, pos["qty"], round_price(max(pos["tp"], o), t), ts, "Take profit")
                events.append("🎯 " + msg)
                break
    return events


def mark(acct):
    rows, val = [], {"MYR": 0.0, "USD": 0.0}
    for t, p in acct["positions"].items():
        px = last_price(t)
        px = p["avg"] if px is None else px
        c = ccy(t)
        mv = px * p["qty"]
        exit_fee = _fee(acct, mv, t)
        val[c] += mv
        upl = mv - exit_fee - p["avg"] * p["qty"]
        rows.append({"Ticker": t, "Qty": p["qty"], "Avg cost": p["avg"], "Last": px, "Market value": mv,
                     "Unrealised P/L": upl, "P/L %": upl / (p["avg"] * p["qty"]) * 100 if p["avg"] else 0,
                     "Stop loss": p.get("sl"), "Take profit": p.get("tp"), "Ccy": c})
    return pd.DataFrame(rows), val


def snapshot_equity(acct, val):
    today = _now().strftime("%Y-%m-%d")
    eq = {"date": today, "MYR": acct["cash"]["MYR"] + val["MYR"], "USD": acct["cash"]["USD"] + val["USD"]}
    if acct["equity"] and acct["equity"][-1]["date"] == today:
        changed = any(abs(acct["equity"][-1][k] - eq[k]) > 0.005 for k in ("MYR", "USD"))
        acct["equity"][-1] = eq
        return changed
    acct["equity"].append(eq)
    return True


def stats(acct):
    sells = [t for t in acct["trades"] if t["side"] == "SELL" and t["pnl"] is not None]
    if not sells:
        return None
    p = np.array([t["pnl"] for t in sells])
    wins, losses = p[p > 0], p[p <= 0]
    return {"n": len(p), "win_rate": len(wins) / len(p) * 100, "avg_win": wins.mean() if len(wins) else 0,
            "avg_loss": losses.mean() if len(losses) else 0,
            "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
            "best": p.max(), "worst": p.min()}


# ============================================================================
# UI
# ============================================================================
def _dv(key, default):
    """Pass a default only when the widget has no value yet (avoids Streamlit's double-default warning)."""
    return {} if key in st.session_state else {"value": default}


def _reset_cb():
    save_account(new_account(st.session_state.get("pt_smyr", 10000.0), st.session_state.get("pt_susd", 2000.0)))
    st.session_state["pt_confirm"] = False
    st.session_state["pt_flash"] = "New paper account created."
def _use_plan(ticker, plan):
    lot = lot_of(ticker)
    st.session_state["pt_prefill"] = {"ticker": ticker, "entry": round_price(plan["entry"], ticker),
                                      "stop": round_price(plan["stop"], ticker), "t1": round_price(plan["t1"], ticker),
                                      "qty": st.session_state.get("pt_lots", 1) * lot if lot == 100
                                      else st.session_state.get("pt_shares", 10)}


def render(display_name, normalise, P, pill, all_symbols, stock_panel):
    acct = load_account()
    events = process(acct)
    pos_df, val = mark(acct)
    if events or snapshot_equity(acct, val):
        save_account(acct)
        pos_df, val = mark(acct)
    for e in events:
        st.toast(e)
    browser_sync()

    st.markdown("### 💼 Paper trading")
    if st.session_state.get("pt_flash"):
        st.success(st.session_state.pop("pt_flash"))
    st.caption("Practise with virtual money: real prices, Bursa board lots and fees, no real orders. "
               "Limit/stop orders and stop-loss/take-profit are checked against daily candles each time you open "
               "the app.")

    # ---------------- account summary
    used = ["MYR"] + (["USD"] if any(ccy(t) == "USD" for t in acct["positions"])
                      or any(ccy(t["ticker"]) == "USD" for t in acct["trades"])
                      or any(ccy(o["ticker"]) == "USD" for o in acct["orders"]) else [])
    for c in used or ["MYR"]:
        eq = acct["cash"][c] + val[c]
        ret = (eq / acct["start"][c] - 1) * 100 if acct["start"][c] else 0
        upl = float(pos_df.loc[pos_df["Ccy"] == c, "Unrealised P/L"].sum()) if len(pos_df) else 0.0
        rpl = sum(t["pnl"] for t in acct["trades"] if t["pnl"] is not None and ccy(t["ticker"]) == c)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{c} account value", f"{sym(c)}{eq:,.2f}", f"{ret:+.2f}% since start")
        m2.metric("Cash available", f"{sym(c)}{acct['cash'][c]:,.2f}")
        m3.metric("Unrealised P/L", f"{sym(c)}{upl:,.2f}")
        m4.metric("Realised P/L", f"{sym(c)}{rpl:,.2f}")
    if "USD" not in used:
        st.caption(f"US stocks use a separate US$ account: ${acct['cash']['USD']:,.2f} cash. It shows here after "
                   "your first US trade.")

    tabs = st.tabs(["🧾 Trade", "📂 Positions", "⏳ Orders", "📜 History & stats", "⚙️ Account"])

    # ---------------- order ticket
    with tabs[0]:
        pre = st.session_state.pop("pt_prefill", None)
        if pre:
            st.session_state["pt_symbol"] = pre["ticker"]
            st.session_state["pt_side"] = "BUY"
            st.session_state["pt_type"] = "LIMIT"
            st.session_state["pt_price"] = float(pre["entry"])
            st.session_state["pt_use_sl"] = True
            st.session_state["pt_sl"] = float(pre["stop"])
            st.session_state["pt_use_tp"] = True
            st.session_state["pt_tp"] = float(pre["t1"])
            if pre.get("qty"):
                st.session_state["pt_lots" if pre["ticker"].endswith(".KL") else "pt_shares"] = \
                    max(int(pre["qty"]) // (100 if pre["ticker"].endswith(".KL") else 1), 1)
            st.session_state["pt_other"] = ""
            st.info("Order ticket filled from the trade plan (limit buy at entry, stop loss, target 1). Check it and press Place order.")
        opts = all_symbols()
        if st.session_state.get("pt_symbol") and st.session_state["pt_symbol"] not in opts:
            opts = [st.session_state["pt_symbol"]] + opts
        c1, c2 = st.columns([2.2, 1])
        ticker = c1.selectbox("Symbol (type to search every Bursa & US stock)", opts, key="pt_symbol",
                              format_func=lambda t: f"{display_name(t)}  ·  {t}")
        other = c2.text_input("…or a code not in the list", placeholder="e.g. 0331, 7123, PLTR", key="pt_other")
        if other.strip():
            ticker = normalise(other)
        if st.session_state.get("pt_last_sym") not in (None, ticker) and not pre:
            for k in ("pt_price", "pt_sl", "pt_tp", "pt_use_sl", "pt_use_tp"):
                st.session_state.pop(k, None)   # prices from another stock make no sense here
        st.session_state["pt_last_sym"] = ticker
        c = ccy(ticker)
        px = last_price(ticker)
        lot = lot_of(ticker)
        held = acct["positions"].get(ticker, {}).get("qty", 0)
        st.markdown(f"<div class='card' style='margin:6px 0 10px'><b>{display_name(ticker)}</b> "
                    f"<span class='muted'>{ticker}</span> &nbsp; last <b>{'N/A' if px is None else f'{sym(c)}{fp(px, True)}'}</b>"
                    f" &nbsp;·&nbsp; you hold <b>{held:,}</b> shares &nbsp;·&nbsp; "
                    f"{'market open' if ta_plus.market_open(ticker) else 'market closed'}</div>",
                    unsafe_allow_html=True)
        with st.expander(f"📈 Chart & analysis: {display_name(ticker)}", expanded=True):
            stock_panel(ticker, "ptc", use_plan_cb=_use_plan)
        d1, d2, d3 = st.columns(3)
        side = d1.segmented_control("Side", ["BUY", "SELL"], default="BUY", key="pt_side") or "BUY"
        otype = d2.segmented_control("Order type", ["MARKET", "LIMIT", "STOP"], default="MARKET",
                                     key="pt_type") or "MARKET"
        if lot == 100:
            lots = d3.number_input("Lots (1 lot = 100 shares)", min_value=1, step=1, key="pt_lots", **_dv("pt_lots", 1))
            qty = int(lots) * 100
        else:
            qty = int(d3.number_input("Shares", min_value=1, step=1, key="pt_shares", **_dv("pt_shares", 10)))
        price = None
        if otype != "MARKET":
            price = st.number_input(f"{'Limit' if otype == 'LIMIT' else 'Stop (trigger)'} price ({sym(c)})",
                                    min_value=0.0, step=0.005 if c == "MYR" else 0.01, format="%.3f", key="pt_price",
                                    **_dv("pt_price", float(px or 1.0)))
            st.caption({"LIMIT": "Buy limit fills at this price or lower; sell limit at this price or higher.",
                        "STOP": "Buy stop fills when price rises to this level (breakout entry); sell stop when "
                                "it falls to it."}[otype])
        sl = tp = None
        if side == "BUY":
            e1, e2, e3, e4 = st.columns([0.6, 1, 0.6, 1])
            if e1.checkbox("Stop loss", key="pt_use_sl"):
                sl = e2.number_input("Stop loss price", min_value=0.0, format="%.3f", key="pt_sl",
                                     label_visibility="collapsed", **_dv("pt_sl", round((px or 1) * 0.95, 3)))
            if e3.checkbox("Take profit", key="pt_use_tp"):
                tp = e4.number_input("Take profit price", min_value=0.0, format="%.3f", key="pt_tp",
                                     label_visibility="collapsed", **_dv("pt_tp", round((px or 1) * 1.10, 3)))
        ref = price if price else (px or 0)
        value = qty * ref
        fee = _fee(acct, value, ticker) if value else 0
        risk_txt = ""
        if sl and ref > sl:
            risk = qty * (ref - sl) + fee + _fee(acct, qty * sl, ticker)
            risk_txt = f" · risk to stop {sym(c)}{risk:,.2f} ({risk / max(acct['cash'][c] + val[c], 1) * 100:.1f}% of account)"
        st.markdown(f"<div class='note'>Estimated {'cost' if side == 'BUY' else 'proceeds'} "
                    f"{sym(c)}{value:,.2f} {'+' if side == 'BUY' else '−'} fees {sym(c)}{fee:,.2f} = "
                    f"<b>{sym(c)}{value + fee if side == 'BUY' else value - fee:,.2f}</b> · cash available "
                    f"{sym(c)}{acct['cash'][c]:,.2f}{risk_txt}</div>", unsafe_allow_html=True)
        if st.button(f"{'🟢 Place BUY' if side == 'BUY' else '🔴 Place SELL'} order", type="primary", width="stretch"):
            if sl is not None and ref and sl >= ref:
                st.error("Stop loss must be below the entry price.")
            elif tp is not None and ref and tp <= ref:
                st.error("Take profit must be above the entry price.")
            else:
                ok, msg = place_order(acct, ticker, side, otype, qty, price, sl, tp)
                if ok:
                    save_account(acct)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    # ---------------- positions
    with tabs[1]:
        if pos_df.empty:
            st.info("No open positions. Place a buy order in 🧾 Trade.")
        else:
            show = pos_df.copy()
            show.insert(1, "Name", [display_name(t) for t in show["Ticker"]])
            st.dataframe(show, hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.3f")
                                        for k in ("Avg cost", "Last", "Stop loss", "Take profit")}
                         | {k: st.column_config.NumberColumn(format="%,.2f") for k in ("Market value", "Unrealised P/L")}
                         | {"P/L %": st.column_config.NumberColumn(format="%+.2f%%")})
            st.caption("Avg cost includes buy fees. Unrealised P/L includes the estimated selling fees.")
            pt = st.selectbox("Manage position", list(acct["positions"]), format_func=lambda t: f"{display_name(t)} · {t}",
                              key="pt_manage")
            p = acct["positions"][pt]
            g1, g2, g3 = st.columns(3)
            nsl = g1.number_input("Stop loss (0 = none)", min_value=0.0, value=float(p.get("sl") or 0.0), format="%.3f",
                                  key=f"pt_esl_{pt}")
            ntp = g2.number_input("Take profit (0 = none)", min_value=0.0, value=float(p.get("tp") or 0.0),
                                  format="%.3f", key=f"pt_etp_{pt}")
            if g3.button("💾 Update SL / TP", width="stretch"):
                p["sl"], p["tp"] = (nsl or None), (ntp or None)
                p["bracket_from"] = str(_now())
                save_account(acct)
                st.success("Updated.")
                st.rerun()
            h1, h2 = st.columns(2)
            if h1.button(f"Close half ({(p['qty'] // 2) // lot_of(pt) * lot_of(pt):,} shares) at market", width="stretch",
                         disabled=(p["qty"] // 2) // lot_of(pt) == 0):
                ok, msg = place_order(acct, pt, "SELL", "MARKET", (p["qty"] // 2) // lot_of(pt) * lot_of(pt))
                (st.success if ok else st.error)(msg)
                if ok:
                    save_account(acct)
                    st.rerun()
            if h2.button(f"Close all ({p['qty']:,} shares) at market", width="stretch"):
                for o in acct["orders"]:
                    if o["ticker"] == pt and o["side"] == "SELL" and o["status"] == "OPEN":
                        o["status"], o["note"] = "CANCELLED", "closed manually"
                ok, msg = place_order(acct, pt, "SELL", "MARKET", p["qty"])
                (st.success if ok else st.error)(msg)
                if ok:
                    save_account(acct)
                    st.rerun()

    # ---------------- orders
    with tabs[2]:
        open_o = [o for o in acct["orders"] if o["status"] == "OPEN"]
        if not open_o:
            st.info("No open orders.")
        else:
            st.dataframe(pd.DataFrame([{"Placed": o["created"][:16], "Ticker": o["ticker"],
                                        "Name": display_name(o["ticker"]), "Side": o["side"], "Type": o["type"],
                                        "Qty": o["qty"], "Price": o["price"], "SL": o.get("sl"), "TP": o.get("tp")}
                                       for o in open_o]), hide_index=True, width="stretch")
            oid = st.selectbox("Cancel an order", [o["id"] for o in open_o],
                               format_func=lambda i: next(f"{o['side']} {o['qty']:,} {o['ticker']} @ {fp(o['price'])}"
                                                          for o in open_o if o["id"] == i), key="pt_cancel")
            if st.button("✖ Cancel order"):
                for o in acct["orders"]:
                    if o["id"] == oid:
                        o["status"], o["note"] = "CANCELLED", "cancelled by you"
                save_account(acct)
                st.rerun()
        done = [o for o in acct["orders"] if o["status"] != "OPEN"][-30:][::-1]
        if done:
            st.markdown("**Recent orders**")
            st.dataframe(pd.DataFrame([{"Placed": o["created"][:16], "Ticker": o["ticker"], "Side": o["side"],
                                        "Type": o["type"], "Qty": o["qty"], "Price": o["price"],
                                        "Status": o["status"], "Filled @": o.get("fill_price"),
                                        "Note": o.get("note", "")} for o in done]), hide_index=True, width="stretch")

    # ---------------- history & stats
    with tabs[3]:
        s = stats(acct)
        if s:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Closed trades", s["n"])
            k2.metric("Win rate", f"{s['win_rate']:.0f}%")
            k3.metric("Avg win / avg loss", f"{s['avg_win']:,.2f} / {s['avg_loss']:,.2f}")
            k4.metric("Profit factor", "∞" if np.isinf(s["profit_factor"]) else f"{s['profit_factor']:.2f}")
            st.caption("Profit factor = total profits ÷ total losses (above 1.5 is good). Win rate alone means little "
                       "without the size of wins vs losses.")
        if len(acct["equity"]) >= 2:
            eqd = pd.DataFrame(acct["equity"])
            fig = go.Figure()
            for c in used or ["MYR"]:
                fig.add_trace(go.Scatter(x=eqd["date"], y=eqd[c], name=f"{c} equity", mode="lines+markers"))
            fig.update_layout(height=300, margin=dict(l=8, r=8, t=30, b=8),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              title=dict(text="Account value (one point per day you open the app)", font=dict(size=13)),
                              legend=dict(orientation="h", y=1.15))
            fig.update_yaxes(gridcolor=P["grid"])
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        if acct["trades"]:
            tr = pd.DataFrame(acct["trades"][::-1])
            tr["time"] = tr["time"].str[:16]
            tr.insert(2, "name", [display_name(t) for t in tr["ticker"]])
            st.dataframe(tr, hide_index=True, width="stretch",
                         column_config={"price": st.column_config.NumberColumn(format="%.3f"),
                                        "fees": st.column_config.NumberColumn(format="%.2f"),
                                        "pnl": st.column_config.NumberColumn("realised P/L", format="%+,.2f")})
        else:
            st.info("No trades yet.")

    # ---------------- account settings
    with tabs[4]:
        st.markdown("**Fees used for paper trades**")
        f1, f2, f3, f4 = st.columns(4)
        fees = acct["fees"]
        nb = f1.number_input("Brokerage (%)", 0.0, 1.0, float(fees["brk_pct"]), 0.01, format="%.3f", key="pt_f1")
        nm = f2.number_input("Min brokerage (RM)", 0.0, 50.0, float(fees["brk_min"]), 1.0, key="pt_f2")
        npf = f3.number_input("Platform fee / order (RM)", 0.0, 20.0, float(fees["platform"]), 0.5, key="pt_f3")
        nus = f4.number_input("US commission / order ($)", 0.0, 20.0, float(fees["us_commission"]), 0.5, key="pt_f4")
        if (nb, nm, npf, nus) != (fees["brk_pct"], fees["brk_min"], fees["platform"], fees["us_commission"]):
            acct["fees"] = {"brk_pct": nb, "brk_min": nm, "platform": npf, "us_commission": nus}
            save_account(acct)
        st.caption("Clearing fee 0.03% and stamp duty (RM1 per RM1,000) are added automatically for Bursa.")
        st.divider()
        st.markdown("**Backup & restore**")
        st.download_button("⬇️ Download backup (JSON)", json.dumps(acct, indent=1, default=str),
                           file_name=f"paper_account_{_now():%Y%m%d}.json", mime="application/json")
        up = st.file_uploader("Restore from a backup file", type="json", key="pt_restore")
        if up is not None and st.button("Restore this backup"):
            try:
                data = json.load(up)
                if _valid(data):
                    save_account(data)
                    st.success("Account restored.")
                    st.rerun()
                else:
                    st.error("That file doesn't look like a paper account backup.")
            except Exception as e:
                st.error(f"Couldn't read the file: {e}")
        st.caption("Your account is saved on the app server and in this browser. The server copy can be lost when "
                   "the app restarts or is updated; it is then restored automatically from this browser. Phone and "
                   "PC browsers keep separate copies, so download a backup now and then.")
        st.divider()
        st.markdown("**Start a new account**")
        r1, r2, r3 = st.columns(3)
        smyr = r1.number_input("Starting cash (RM)", 0.0, 10_000_000.0, 10000.0, 1000.0, key="pt_smyr")
        susd = r2.number_input("Starting cash (US$)", 0.0, 10_000_000.0, 2000.0, 500.0, key="pt_susd")
        confirm = r3.checkbox("Yes, erase my current paper account", key="pt_confirm")
        st.button("🔄 Reset account", disabled=not confirm, on_click=_reset_cb)
