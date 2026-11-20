"""Analysis & Forecast page: full single-stock analysis."""
from core import *  # noqa: F401,F403
import engine
import learn
import ta_plus
import ui
from views import analysis_plus as AP


def render(C):
    universe, interval, period, unit, tf_name = C.universe, C.interval, C.period, C.unit, C.tf_name
    closed_only, all_symbols, MOBILE = C.closed_only, C.all_symbols, C.mobile
    opts = list(dict.fromkeys((list(universe) if universe else ["1155.KL"]) + all_symbols()))
    c1, c2, c3, c4 = st.columns([2.2, 1.4, 1, 1])
    pick = c1.selectbox("Stock", opts, format_func=lambda t: f"{display_name(t)}  ·  {t}", key="pick")
    typed = c2.text_input("Or type a ticker", placeholder="e.g. GAMUDA, 5398, NVDA", key="an_typed")
    horizon = c3.selectbox("Forecast ahead", [5, 10, 20, 30], index=1,
                           format_func=lambda v: f"{v} {unit.split()[-1]}")
    lookback = c4.selectbox("Trend based on", [30, 60, 120], index=1, format_func=lambda v: f"last {v} bars")
    ticker = normalise(typed) if typed.strip() else pick
    st.session_state["an_ticker"] = ticker
    cur = "RM" if ticker.endswith(".KL") else "$"

    with st.spinner(f"Fetching {ticker}..."):
        df_raw = load_one(ticker, interval, period)

    if df_raw.empty or len(df_raw) < 60:
        st.error(f"Not enough data found for **{ticker}** on this timeframe ({len(df_raw)} candles). For Bursa use the "
                 f"4-digit code + .KL (e.g. 1155.KL) or a name from the list; for US use the normal ticker (e.g. NVDA).")
    else:
        dq, cstat = ta_plus.data_quality(df_raw, interval, ticker)
        live_price = float(df_raw["Close"].iloc[-1])
        df = df_raw.iloc[:-1] if (closed_only and cstat["forming"]) else df_raw
        d, sup, res, sc, label, why = analyse(df)
        atr_mult = float(st.session_state.get("ts_atr", 1.0))
        plan = trade_plan(d, sup, res, label, unit, atr_mult=atr_mult)
        fc = trend_forecast(d, horizon=horizon, lookback=min(lookback, len(d) - 2))
        bar_word_a = {"1d": "day", "1wk": "week", "1mo": "month", "4h": "4-hour bar", "1h": "hour",
                      "15m": "bar"}.get(interval, "bar")
        bar_word = bar_word_a
        ax = adx_analysis(d, bar_word_a)
        va = volume_analysis(d, bar_word)
        vol = ta_plus.volatility(d)
        last, prev = d.iloc[-1], d.iloc[-2]
        price = float(last["Close"])
        chg = (price / float(prev["Close"]) - 1) * 100
        chg_col = P["up"] if chg >= 0 else P["down"]
        dir_col = {"Uptrend": P["up"], "Downtrend": P["down"]}.get(fc["direction"], P["muted"])
        vol_col = {"Low": P["up"], "Moderate": P["warn"], "High": P["down"]}.get(vol["cls"] if vol else "", P["muted"])

        # ---- status badges (data integrity)
        badges = []
        if cstat["forming"]:
            badges.append(pill(f"● Live candle forming · live {cur}{live_price:,.3f}"
                               + (" · signals use last completed candle" if closed_only else ""), P["warn"]))
        if cstat["stale"]:
            badges.append(pill("⚠ Data looks stale: check the symbol or try again later", P["down"]))
        badges.append(pill(f"Market {'open' if cstat['open'] else 'closed'} · {cstat['exchange']}", P["muted"]))
        st.markdown(f"""
<div class="card"><div class="head">
  <div><div class="name">{display_name(ticker)} · {ticker}</div>
       <div><span class="price">{price:,.3f}</span><span class="chg" style="color:{chg_col}">{chg:+.2f}%</span></div></div>
  <div><div class="name">Signal</div>{pill(label, LABEL_COLOR[label])}</div>
  <div><div class="name">Score</div><div style="font-size:20px;font-weight:600">{sc:+d}<span class="muted"> / 100</span></div></div>
  <div><div class="name">Trend strength (ADX)</div>{pill(f"{ax['regime']} · {ax['adx']:.0f}", regime_color(ax['regime'])) if ax else pill("N/A", P['muted'])}</div>
  <div><div class="name">Volatility (ATR)</div>{pill(f"{vol['cls']} · {vol['atrp']:.2f}%", vol_col) if vol else pill("N/A", P['muted'])}</div>
  <div><div class="name">RSI</div><div style="font-size:20px;font-weight:600">{last['RSI']:.0f}</div></div>
  <div style="margin-left:auto" class="muted">Last candle used: {d.index[-1].strftime('%d %b %Y %H:%M')}<br>{tf_name} · {len(d)} candles
   · fetched {pd.Timestamp(dq['Fetched']).strftime('%H:%M:%S') if dq['Fetched'] != 'n/a' else 'n/a'}</div>
</div><div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap">{''.join(badges)}</div></div>""",
                    unsafe_allow_html=True)

        # ---- chart controls
        k1, k2, k3, k4 = st.columns([0.8, 1.6, 1.4, 1.5])
        show_fc = k1.toggle("Trend forecast", value=True)
        extra_sel = k2.multiselect("Extra support / resistance", EXTRA_OPTS, default=[],
                                   placeholder="Swings, touches, volume profile, Fibonacci…", key="extra_sel")
        panels = k3.multiselect("Indicator panels", CHART_PANELS,
                                default=["Volume", "RSI"] if MOBILE else CHART_PANELS, key="panels")
        overlays = k4.multiselect("Price overlays", CHART_OVERLAYS, default=CHART_OVERLAYS, key="overlays")
        extra, extra_rows = compute_extra_levels(d, extra_sel) if extra_sel else ({}, [])
        st.plotly_chart(make_chart(d, sup, res, plan, display_name(ticker), interval, fc if show_fc else None,
                                   extra=extra, panels=panels, overlays=overlays, mobile=MOBILE),
                        width="stretch", config={"displaylogo": False, "scrollZoom": True})
        legend_bits = []
        if "Support/resistance & zones" in overlays:
            legend_bits.append("green/red lines = support/resistance (×N touches) · green band = buy zone · red band = "
                               "sell zone · orange dashed = stop loss")
        if show_fc:
            legend_bits.append("blue dashed = trend forecast with nominal 68%/95% bands (see Forecast accuracy)")
        if "Volume" in panels:
            legend_bits.append(f"volume: blue line = 20-bar average, <span style='color:{P['warn']}'>orange ⚡</span> = "
                               f"spike 2×+, 🚨 = 3×+")
        if "Breakout markers" in overlays:
            legend_bits.append("★ = 20-bar breakout/breakdown (coloured = volume-confirmed, grey = weak volume)")
        if "ADX" in panels:
            legend_bits.append("ADX panel: white = ADX, green = +DI, red = −DI")
        extra_legend = {
            "Swing highs/lows": "▼▲ = swing highs/lows",
            "Repeated touches (3+)": "teal ⟷ = levels touched 3+ times",
            "Volume profile": "right-side bars = volume by price, POC/VAH/VAL",
            "Fibonacci retracement": "gold dotted = Fibonacci levels",
        }
        legend_bits += [extra_legend[k] for k in extra_sel]
        st.markdown("<div class='note'>" + " · ".join(legend_bits) + "</div>", unsafe_allow_html=True)

        # ---- other timeframes (real data only; used by MTF tab, audit and summary)
        with st.spinner("Loading other timeframes…"):
            tf_rows, tf_data = {}, {}
            for key, lab, iv, per in ta_plus.TIMEFRAMES:
                tdf = df_raw if key == interval else load_one(ticker, iv, per)
                if tdf is None or tdf.empty:
                    tf_rows[key] = {"available": False, "candles": 0, "reason": "no data from the source"}
                    continue
                tst = ta_plus.candle_status(tdf, key, ticker)
                use = tdf.iloc[:-1] if (closed_only and tst["forming"]) else tdf
                tf_data[key] = use
                r_ = ta_plus.tf_analysis(use, add_indicators)
                r_["forming"] = tst["forming"]
                r_["stale"] = tst["stale"]
                tf_rows[key] = r_
        mtf = ta_plus.mtf_summary(tf_rows)
        pats_full = ta_plus.pattern_report(d, sup, res)
        bt_df, bt_m = ta_plus.walk_forward(d["Close"].values, fc["horizon"], fc["lookback"])

        F = score_factors(d, sup, res)
        tech = engine.assess(d, sup, res, mtf=mtf, patterns=learn.detect_patterns(d, 3), weights=AP.tech_weights())
        hz = engine.horizons(tech["x"])
        ops = engine.opportunities(tech["x"], sup, res, tech)
        AP.inject()
        tabs = st.tabs(["🧠 Summary", "🧭 Scorecard", "📐 Indicators", "🏦 Fundamentals", "🧮 Audit", "📋 Trade plan",
                        "⏱️ Timeframes", "📊 Volume", "📈 ADX", "🕯️ Patterns", "🎯 Forecast", "📏 Levels", "🧾 Data"])

        # ================= Summary
        with tabs[0]:
            head, secs = ta_plus.narrative(display_name(ticker), ticker, d, sup, res, plan, va, ax, mtf, fc, vol,
                                           pats_full, cstat, (bt_df, bt_m))
            st.markdown(f"""
<div class="card" style="border-left:3px solid {P['accent']}">
<h3>🧠 Analysis summary</h3>
<div style="font-size:15.5px;line-height:1.7">{head}</div>
<div class="note">Written automatically from the indicator values calculated on this page (rule-based; every number
comes from the data, nothing is invented). Not investment advice.</div></div>""", unsafe_allow_html=True)
            # ---------------- Key numbers at a glance (every analysis, with justification)
            def kcard(title, verdict, vcol, rows, why):
                trs = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
                return (f"<div class='card' style='margin:0;border-top:3px solid {vcol};display:flex;flex-direction:column'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:8px;"
                        f"flex-wrap:wrap'><h3 style='margin:0'>{title}</h3>{pill(verdict, vcol)}</div>"
                        f"<table class='kv' style='margin-top:10px'>{trs}</table>"
                        f"<div class='just'><b>Justification:</b> {why}</div></div>")

            nz = lambda v, f_="{:.3f}", na="N/A": na if v is None or (isinstance(v, float) and np.isnan(v)) else f_.format(v)  # noqa: E731
            cards_k = []

            # 1. Trade plan
            rr1v = plan["rr1"]
            rr_txt = "–" if np.isnan(rr1v) else f"1 : {rr1v:.2f}"
            rr_q = ("good (≥ 2)" if rr1v >= 2 else "acceptable (1.5–2)" if rr1v >= 1.5 else "poor (< 1.5)") \
                if not np.isnan(rr1v) else "not available"
            act_col = (P["up"] if "BUY" in plan["action"] and "DON'T" not in plan["action"] else
                       P["down"] if "AVOID" in plan["action"] else P["warn"] if "DON'T" in plan["action"] else P["muted"])
            cards_k.append(kcard(
                "📋 Trade plan", plan["action"].split(" ", 1)[1].title(), act_col,
                [("Buy zone", f"{cur}{plan['buy_lo']:.3f} – {plan['buy_hi']:.3f}"),
                 ("Reference entry", f"{cur}{plan['entry']:.3f}"),
                 ("Stop loss", f"{cur}{plan['stop']:.3f} ({(plan['stop'] / plan['entry'] - 1) * 100:.1f}%)"),
                 ("Target 1 / Target 2", f"{cur}{plan['t1']:.3f} / {cur}{plan['t2']:.3f}"),
                 ("Risk : Reward (T1)", rr_txt),
                 ("Sell zone", f"{cur}{plan['sell_lo']:.3f} – {plan['sell_hi']:.3f}")],
                f"Buy zone = nearest support {cur}{plan['s1']:.3f} ({sup[0][1] or 'projected'} touches) up to +0.5 ATR. "
                f"Stop sits {plan['atr_mult']:g} × ATR ({cur}{plan['atr']:.3f}) below support so normal daily noise "
                f"doesn't trigger it. Target 1 = nearest resistance ({res[0][1] or 'projected'} touches). "
                f"R:R is {rr_q}."))

            # 2. Trend forecast
            fc_col = {"Uptrend": P["up"], "Downtrend": P["down"]}.get(fc["direction"], P["muted"])
            bt_txt = (f"Back-tested direction accuracy {bt_m['dir_acc']:.0f}% over {bt_m['n']} past forecasts "
                      f"(skill vs 'no change' {bt_m['skill']:+.0f}%); the 95% band actually held the price "
                      f"{bt_m['cov95']:.0f}% of the time." if bt_m else "No back-test available.")
            reliab = ("reliable-ish" if bt_m and bt_m["dir_acc"] >= 60 and bt_m["skill"] > 0 else
                      "weak" if bt_m else "untested")
            cards_k.append(kcard(
                f"📈 Trend forecast · {fc['horizon']} {bar_word}s", f"{fc['arrow']} {fc['direction']}", fc_col,
                [("Projected price", f"{cur}{fc['centre'][-1]:.3f} ({fc['total']:+.1f}%)"),
                 ("Likely range (68%)", f"{cur}{fc['lo1'][-1]:.3f} – {fc['hi1'][-1]:.3f}"),
                 ("Wide range (95%)", f"{cur}{fc['lo2'][-1]:.3f} – {fc['hi2'][-1]:.3f}"),
                 ("Trend fit R²", f"{fc['r2']:.2f} ({fc['strength'].lower()})"),
                 ("Momentum", fc["momentum"]),
                 ("Forecast track record", reliab)],
                f"A straight trend line fitted to the last {fc['lookback']} {bar_word}s, extended forward. "
                f"R² {fc['r2']:.2f} = {'clean' if fc['r2'] >= 0.7 else 'moderate' if fc['r2'] >= 0.45 else 'choppy'} trend. "
                + bt_txt))

            # 3. Why this signal (original scoring)
            top = sorted(F, key=lambda f_: -abs(f_["Points"]))
            sig_rows = [(f_["Factor"], f"<span style='color:{P['up'] if f_['Points'] > 0 else P['down']}'>"
                                       f"{f_['Points']:+d}</span> · {f_['Reading']}") for f_ in top[:6]]
            cards_k.append(kcard(
                "🧮 Why this signal", f"{label} · {sc:+d}", LABEL_COLOR[label], sig_rows,
                f"Sum of {len(F)} scored factors = {sum(f_['Points'] for f_ in F):+d} (capped ±100). ≥ +40 Strong Buy, "
                f"≥ +15 Buy, ≤ −15 Sell, ≤ −40 Strong Sell. Trend (EMAs) "
                f"{sum(f_['Points'] for f_ in F if f_['Group'] == 'Trend'):+d}, momentum (RSI/MACD) "
                f"{sum(f_['Points'] for f_ in F if f_['Group'] == 'Momentum'):+d}, levels "
                f"{sum(f_['Points'] for f_ in F if f_['Group'] == 'Levels'):+d}. Full table in 🧮 Audit."))

            # 4. Volume
            if va:
                brk_v = (f"{va['brk']['type']} · {va['brk']['grade']} ({va['brk']['rvol']:.1f}×)" if va["brk"]
                         else "none in last 10 bars")
                ud_v = "∞" if np.isinf(va["ud"]) else f"{va['ud']:.2f}"
                cards_k.append(kcard(
                    "📊 Volume", va["verdict"].split(" ", 1)[1].split(":")[0], va["vcol"],
                    [("Latest volume", fmt_vol(va["vol"])), ("20-bar average", fmt_vol(va["avg"])),
                     ("Relative volume (RVOL)", f"{va['rvol']:.2f}×"),
                     ("Last 5 vs average", nz(va["part"], "{:.2f}×")),
                     ("Up / down volume (20)", ud_v), ("Breakout check", brk_v)],
                    " ".join(va["why"][:2]) + (" " + va["flags"][0][1] + "." if va["flags"] else "")))

            # 5. ADX
            if ax:
                cards_k.append(kcard(
                    "📐 Trend strength (ADX)", ax["regime"], regime_color(ax["regime"]),
                    [("ADX (14)", f"{ax['adx']:.1f} ({'rising' if ax['rising'] else 'falling' if ax['falling'] else 'flat'},"
                                  f" was {ax['adx5']:.1f})"),
                     ("+DI (buyers)", f"{ax['pdi']:.1f}"), ("−DI (sellers)", f"{ax['mdi']:.1f}"),
                     ("DI gap", f"{abs(ax['pdi'] - ax['mdi']):.1f} → {'buyers' if ax['pdi'] >= ax['mdi'] else 'sellers'} lead")],
                    f"ADX measures strength only: < 20 sideways, 20–25 weak, ≥ 25 strong. Direction comes from +DI vs −DI. "
                    + (ax["notes"][1] if len(ax["notes"]) > 1 else ax["notes"][0])))

            # 6. Volatility & position size
            if vol:
                cfg_s = {"capital": float(st.session_state.get("ts_capital", 10000.0)),
                         "risk_pct": float(st.session_state.get("ts_risk", 1.0)),
                         "max_loss": float(st.session_state.get("ts_maxloss", 0.0)),
                         "brk_pct": float(st.session_state.get("ts_brk", 0.08)),
                         "brk_min": float(st.session_state.get("ts_brkmin", 8.0)),
                         "platform": float(st.session_state.get("ts_plat", 0.0)),
                         "us_commission": float(st.session_state.get("ts_uscomm", 0.0))}
                e_ = ta_plus.round_tick(plan["entry"], ticker)
                s_ = ta_plus.round_tick(plan["stop"], ticker, "down")
                ps_ = ta_plus.position_size(e_, s_, ticker, cfg_s)
                sh_ = ps_["shares"] if ps_ else 0
                size_txt = (f"{sh_:,} shares" + (f" ({sh_ // 100} lots)" if ticker.endswith('.KL') else "")) if sh_ \
                    else "0 (stop too far for budget)"
                cards_k.append(kcard(
                    "🌡️ Volatility & position size", f"{vol['cls']} volatility", vol_col,
                    [("ATR (14)", f"{cur}{vol['atr']:.3f}"), ("ATR % of price", f"{vol['atrp']:.2f}%"),
                     ("Volatility rank", nz(vol["pct"], "{:.0f}th percentile")),
                     ("Risk budget", f"{cur}{ps_['budget']:,.2f}" if ps_ else "N/A"),
                     ("Position size", size_txt),
                     ("Required capital / max loss",
                      f"{cur}{ps_['cost']:,.0f} / {cur}{ps_['loss']:,.2f}" if sh_ else "–")],
                    f"Today's ATR% ranks {nz(vol['pct'], '{:.0f}')}th against this stock's last {vol['n']} candles. "
                    f"Risk per share = entry {cur}{e_:.3f} − stop {cur}{s_:.3f} = {cur}{e_ - s_:.3f}; budget "
                    f"{cfg_s['risk_pct']:g}% of {cur}{cfg_s['capital']:,.0f} ÷ that, rounded down to "
                    f"{'100-share lots' if ticker.endswith('.KL') else 'whole shares'}, fees included. Change the "
                    f"settings in 📋 Trade plan."))

            # 7. Multi-timeframe
            lab_k = {k: lab for k, lab, _, _ in ta_plus.TIMEFRAMES}
            mt_rows = [(lab_k[k], f"<span style='color:{P['up'] if 'Bullish' in r_['signal'] else P['down'] if 'Bearish' in r_['signal'] else P['muted']}'>"
                                  f"{r_['signal']}</span> · {r_['score']:+d}")
                       for k, r_ in tf_rows.items() if r_.get("available")]
            mt_col = {"Aligned bullish": P["up"], "Aligned bearish": P["down"], "Conflict": P["warn"]}.get(mtf["swing"], P["muted"])
            cards_k.append(kcard(
                "⏱️ Multi-timeframe", f"Swing: {mtf['swing']}", mt_col, mt_rows or [("Timeframes", "N/A")],
                f"Each timeframe scores trend, EMA alignment, RSI and MACD (±7). Weekly ×3, Daily ×2, 4H ×1 give a swing "
                f"score of {mtf['swing_score']:+d} / ±{mtf['swing_max']}. "
                + ("Timeframes disagree: follow the higher timeframe and wait for the lower one to line up."
                   if mtf["disagree"] else "No timeframes point in opposite directions.")))

            # 8. Candlestick patterns
            if pats_full:
                p0 = pats_full[0]
                perf0 = p0["perf"]
                pk_col = {"bull": P["up"], "bear": P["down"]}.get(p0["kind"], P["muted"])
                cards_k.append(kcard(
                    "🕯️ Candlestick pattern", f"{p0['name']}", pk_col,
                    [("When", "latest candle" if p0["bars_ago"] == 0 else f"{p0['bars_ago']} candle(s) ago"),
                     ("Type", {"bull": "Bullish", "bear": "Bearish"}.get(p0["kind"], "Neutral")),
                     ("Context", f"{p0['context']} ({p0['ctx_grade'].lower()})"),
                     ("Confirmation", p0["confirmation"]),
                     ("History in this stock", f"{perf0['win']:.0f}% worked over {perf0['n']} cases (random {perf0['base']:.0f}%)"
                      if perf0.get("win") is not None else f"insufficient (n={perf0['n']})"),
                     ("Other patterns (5 candles)", str(len(pats_full) - 1))],
                    p0["use"] + f" Detected because: {p0['rule']}."))
            else:
                cards_k.append(kcard("🕯️ Candlestick pattern", "None", P["muted"],
                                     [("Last 5 candles", "no recognised pattern")],
                                     "No pattern means no candle-based signal: rely on levels, trend and volume."))

            # 9. Forecast accuracy
            if bt_m:
                acc_col = P["up"] if bt_m["dir_acc"] >= 60 and bt_m["skill"] > 0 else (
                    P["warn"] if bt_m["dir_acc"] >= 50 else P["down"])
                cards_k.append(kcard(
                    "🎯 Forecast accuracy (back-test)", f"{bt_m['dir_acc']:.0f}% direction", acc_col,
                    [("Forecasts tested", f"{bt_m['n']}"), ("Directional accuracy", f"{bt_m['dir_acc']:.0f}% "
                                                                                   f"(always-up {bt_m['up_rate']:.0f}%)"),
                     ("MAE / RMSE", f"{cur}{bt_m['mae']:.3f} / {cur}{bt_m['rmse']:.3f} ({bt_m['mape']:.1f}%)"),
                     ("Skill vs 'no change'", f"{bt_m['skill']:+.0f}%"),
                     ("Band coverage 68% / 95%", f"{bt_m['cov68']:.0f}% / {bt_m['cov95']:.0f}%")],
                    f"Walk-forward test: at each past date the model saw only earlier data. Above 50% direction and "
                    f"positive skill mean it added value; bands are {ta_plus.calibration_note(bt_m)}."))

            st.markdown("<div class='sech' style='display:flex;align-items:center;gap:10px;margin:18px 0 8px'>"
                        "<span style='font-size:18px;font-weight:650'>📌 Key numbers from every analysis</span>"
                        f"<i style='flex:1;height:1px;background:{P['border']}'></i></div>", unsafe_allow_html=True)
            st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px'>"
                        + "".join(cards_k) + "</div>", unsafe_allow_html=True)
            st.markdown("<div class='sech' style='display:flex;align-items:center;gap:10px;margin:22px 0 8px'>"
                        "<span style='font-size:18px;font-weight:650'>📝 Detailed explanation</span>"
                        f"<i style='flex:1;height:1px;background:{P['border']}'></i></div>", unsafe_allow_html=True)
            order = ["Overall trend", "Short-term momentum", "Long-term trend", "Support & resistance",
                     "Volume behaviour", "RSI & MACD", "ADX trend strength", "Multi-timeframe", "Volatility",
                     "Bullish scenario", "Bearish scenario", "Suggested entry conditions",
                     "Risk factors & invalidation"]
            icons = {"Bullish scenario": "🟢", "Bearish scenario": "🔴", "Suggested entry conditions": "🎯",
                     "Risk factors & invalidation": "⚠️"}
            cards = "".join(
                f"<div class='card' style='margin:0'><div class='muted' style='font-size:12px;text-transform:uppercase;"
                f"letter-spacing:.05em'>{icons.get(k, '•')} {k}</div><div style='margin-top:6px;line-height:1.6'>"
                f"{secs[k]}</div></div>" for k in order if k in secs)
            st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px'>"
                        f"{cards}</div>", unsafe_allow_html=True)

        # ================= Signal audit
        # ================= Scorecard / Indicators / Fundamentals (Phase 5)
        with tabs[1]:
            AP.render_scorecard(ticker, tech, plan, sup, res, hz, ops, price, MOBILE)
        with tabs[2]:
            AP.render_indicators(ticker, display_name(ticker), d, tech, interval, MOBILE)
        with tabs[3]:
            AP.render_fundamentals(ticker, price, MOBILE)

        with tabs[4]:
            F = score_factors(d, sup, res)
            total = sum(f["Points"] for f in F)
            st.markdown(f"""
<div class="card"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
<h3 style="margin:0">🧮 How the score was calculated</h3>{pill(f"{label} · {sc:+d}", LABEL_COLOR[label])}</div>
<div class="muted" style="margin-top:8px">Raw total {total:+d} (capped to ±100). Labels: ≥ +40 STRONG BUY · +15…+39 BUY ·
−14…+14 NEUTRAL · −15…−39 SELL · ≤ −40 STRONG SELL. Each row shows the actual reading, the rule and the points.</div></div>""",
                        unsafe_allow_html=True)
            aud = pd.DataFrame([{k: f[k] for k in ("Group", "Factor", "Reading", "Rule", "Points")} for f in F])
            st.dataframe(aud, hide_index=True, width="stretch",
                         column_config={"Points": st.column_config.NumberColumn(format="%+d", width="small")})
            grp = aud.groupby("Group", sort=False)["Points"].sum()
            gtxt = " · ".join(f"<b>{g}</b> {v:+d}" for g, v in grp.items())
            st.markdown(f"<div class='note'>Subtotals: {gtxt}. The three EMA checks share one <b>Trend</b> group "
                        f"(max ±35) because they tend to move together: read them as one trend vote.</div>",
                        unsafe_allow_html=True)
            # context rows (not scored, avoids double counting)
            m_, sg_, h_ = float(last["MACD"]), float(last["MACDsig"]), float(last["MACDhist"])
            zero_cross = None
            mz = d["MACD"].tail(6).values
            for k_ in range(len(mz) - 1, 0, -1):
                if mz[k_] > 0 >= mz[k_ - 1]:
                    zero_cross = f"MACD line crossed ABOVE zero {len(mz) - 1 - k_} bar(s) ago"
                    break
                if mz[k_] < 0 <= mz[k_ - 1]:
                    zero_cross = f"MACD line crossed BELOW zero {len(mz) - 1 - k_} bar(s) ago"
                    break
            ctx_rows = [
                {"Context": "MACD line", "Reading": f"{m_:.4f}", "Meaning": "fast EMA(12) − slow EMA(26)"},
                {"Context": "MACD signal line", "Reading": f"{sg_:.4f}", "Meaning": "9-period EMA of the MACD line"},
                {"Context": "MACD histogram", "Reading": f"{h_:+.4f}",
                 "Meaning": "MACD − signal: above 0 = line above signal (bullish momentum); below 0 = bearish momentum"},
                {"Context": "MACD vs zero line", "Reading": "above zero" if m_ > 0 else "below zero",
                 "Meaning": ("positive MACD regime (fast EMA above slow EMA)" if m_ > 0 else
                             "negative MACD regime (fast EMA below slow EMA)")
                            + (f" · {zero_cross} (zero-line cross ≠ signal-line crossover)" if zero_cross else "")},
                {"Context": "ADX trend strength", "Reading": f"{ax['adx']:.1f} · {ax['regime']}" if ax else "N/A",
                 "Meaning": "strength only; direction from +DI/−DI"},
                {"Context": "Volume confirmation", "Reading": va["verdict"] if va else "N/A",
                 "Meaning": f"RVOL {va['rvol']:.2f}×" if va else ""},
                {"Context": "Multi-timeframe (swing)", "Reading": mtf["swing"],
                 "Meaning": f"bullish on {', '.join(mtf['bulls']) or 'none'} · bearish on {', '.join(mtf['bears']) or 'none'}"},
            ]
            st.markdown("**Context checks (shown for transparency, not added to the score)**")
            st.dataframe(pd.DataFrame(ctx_rows), hide_index=True, width="stretch")
            st.markdown("<div class='note'>Why not scored: the MACD zero line and ADX direction largely repeat what the EMA "
                        "trend factors already measure, and volume / multi-timeframe are reported separately so the "
                        "original score stays comparable. MACD rules used: line above signal = bullish momentum; line "
                        "crossing above signal = bullish crossover; line above zero = positive regime. These are "
                        "different events and are reported separately.</div>", unsafe_allow_html=True)

        # ================= Trade plan & sizing
        with tabs[5]:
            st.markdown("##### ⚙️ Your trade settings")
            s1c, s2c, s3c, s4c = st.columns(4)
            capital = s1c.number_input(f"Trading capital ({cur})", min_value=100.0, value=10000.0, step=500.0,
                                       key="ts_capital")
            risk_pct = s2c.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.25,
                                        key="ts_risk")
            max_loss = s3c.number_input(f"Max loss per trade ({cur}, 0 = no cap)", min_value=0.0, value=0.0,
                                        step=10.0, key="ts_maxloss")
            s4c.select_slider("Stop distance (× ATR below support)", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], value=1.0,
                              key="ts_atr")
            with st.expander("Broker fees" + (" (Bursa)" if ticker.endswith(".KL") else " (US)")):
                f1, f2, f3, f4 = st.columns(4)
                brk_pct = f1.number_input("Brokerage (%)", 0.0, 1.0, 0.08, 0.01, format="%.3f", key="ts_brk")
                brk_min = f2.number_input("Minimum brokerage (RM)", 0.0, 50.0, 8.0, 1.0, key="ts_brkmin")
                platform = f3.number_input("Platform fee per order (RM)", 0.0, 20.0, 0.0, 0.5, key="ts_plat")
                us_comm = f4.number_input("US commission per order ($)", 0.0, 20.0, 0.0, 0.5, key="ts_uscomm")
                st.caption("Bursa: clearing 0.03% (max RM1,000) and stamp duty RM1 per RM1,000 (max RM1,000) are "
                           "added automatically on both buy and sell. Ordinary-share brokerage is SST-exempt; the "
                           "platform fee gets 8% SST. Set brokerage to your broker's rate.")
            cfg = {"capital": capital, "risk_pct": risk_pct, "max_loss": max_loss, "brk_pct": brk_pct,
                   "brk_min": brk_min, "platform": platform, "us_commission": us_comm}

            rt = lambda p_, m_="nearest": ta_plus.round_tick(p_, ticker, m_)  # noqa: E731
            entry = rt(plan["entry"])
            stop = rt(plan["stop"], "down")
            t1, t2 = rt(plan["t1"], "down"), rt(plan["t2"], "down")
            buy_lo, buy_hi = rt(plan["buy_lo"]), rt(plan["buy_hi"])
            ps = ta_plus.position_size(entry, stop, ticker, cfg)
            atr_v = plan["atr"]
            if vol:
                st.markdown(f"""
<div class="card"><h3>🌡️ Volatility (ATR 14)</h3>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px">
 <div><div class="muted">ATR (14)</div><div style="font-size:20px;font-weight:600">{cur}{vol['atr']:.3f}</div></div>
 <div><div class="muted">ATR % of price</div><div style="font-size:20px;font-weight:600">{vol['atrp']:.2f}%</div></div>
 <div><div class="muted">Volatility</div>{pill(vol['cls'], vol_col)}<div class="muted" style="font-size:12px">
   {f"{vol['pct']:.0f}th percentile of last {vol['n']} candles" if vol['pct'] is not None else 'needs ≥ 60 candles'}</div></div>
 <div><div class="muted">ATR stop</div><div style="font-size:20px;font-weight:600">{cur}{stop:.3f}</div>
   <div class="muted" style="font-size:12px">support {cur}{plan['s1']:.3f} − {atr_mult:g} × ATR</div></div>
</div>
<div class="note">ATR = average size of one {bar_word}'s move (Wilder, 14). Classification compares today's ATR% with this
stock's own last {vol['n']} candles: bottom third = Low, middle = Moderate, top third = High. Because the stop sits a fixed
number of ATRs away, position size automatically shrinks when the stock is more volatile.</div></div>""",
                            unsafe_allow_html=True)
            if ps is None or ps["shares"] == 0:
                st.warning("With these settings the position size is 0: the stop is too far for your risk budget "
                           "(or price exceeds capital). Lower the ATR multiple, raise risk %, or choose a cheaper stock.")
            if ps:
                sh = ps["shares"]
                lot = ps["lot"]
                be = ta_plus.break_even(sh, entry, ticker, cfg) if sh else None
                p1 = ta_plus.net_pnl(sh, entry, t1, ticker, cfg) if sh else 0
                p2 = ta_plus.net_pnl(sh, entry, t2, ticker, cfg) if sh else 0
                half1, half2 = ta_plus.split_partial(sh, lot)
                part = (ta_plus.net_pnl(half1, entry, t1, ticker, cfg) + ta_plus.net_pnl(half2, entry, t2, ticker, cfg)
                        if half2 else p1)
                part_be = (ta_plus.net_pnl(half1, entry, t1, ticker, cfg) +
                           (ta_plus.net_pnl(half2, entry, be, ticker, cfg) if be else 0)) if half2 else p1
                loss = ps["loss"]
                rr_net1 = p1 / loss if loss > 0 else np.nan
                rr_net2 = p2 / loss if loss > 0 else np.nan
                trail_now = max(stop, float(d["Close"].tail(10).max()) - 2 * atr_v)
                lots_txt = f" ({sh // 100} lots)" if lot == 100 else ""
                fb = ps["buy_fees"]["total"]
                plan_rows = [
                    ("Entry price (reference)", f"{cur}{entry:.3f}"),
                    ("Buy zone", f"{cur}{buy_lo:.3f} – {cur}{buy_hi:.3f}"),
                    ("Stop loss", f"{cur}{stop:.3f} ({(stop / entry - 1) * 100:.1f}%)"),
                    ("Target 1", f"{cur}{t1:.3f} ({(t1 / entry - 1) * 100:+.1f}%)"),
                    ("Target 2", f"{cur}{t2:.3f} ({(t2 / entry - 1) * 100:+.1f}%)"),
                    ("Risk : Reward (gross)", "–" if np.isnan(plan['rr1']) else f"1 : {plan['rr1']:.2f} / 1 : {plan['rr2']:.2f}"),
                    ("Risk : Reward (after fees)", "–" if np.isnan(rr_net1) else f"1 : {rr_net1:.2f} / 1 : {rr_net2:.2f}"),
                ]
                size_rows = [
                    ("Risk budget", f"{cur}{ps['budget']:,.2f} ({ps['binding']})"),
                    ("Position size", f"{sh:,} shares{lots_txt}"),
                    ("Required capital", f"{cur}{ps['cost']:,.2f} (incl. {cur}{fb:,.2f} buy fees)"),
                    ("Maximum monetary risk", f"{cur}{loss:,.2f} if stopped (incl. fees both ways)"),
                    ("Break-even price", f"{cur}{be:.3f}" if be else "N/A"),
                    ("Net profit at T1 / T2", f"{cur}{p1:,.2f} / {cur}{p2:,.2f}"),
                ]
                kv = lambda rows: "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)  # noqa: E731
                pc1, pc2 = st.columns(2, gap="medium")
                when_html = re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", plan["when"])
                pc1.markdown(f"<div class='card'><h3>📋 Trade plan</h3><div class='action'>{plan['action']}</div>"
                             f"<table class='kv'>{kv(plan_rows)}</table><div class='when'>"
                             f"{when_html}</div>"
                             f"<div class='note'>Prices rounded to the exchange tick size"
                             f"{' (Bursa: 0.5 sen < RM1, 1 sen < RM10, 2 sen < RM100)' if ticker.endswith('.KL') else ''}."
                             f"</div></div>", unsafe_allow_html=True)
                pc2.markdown(f"<div class='card'><h3>🧮 Position sizing</h3><table class='kv'>{kv(size_rows)}</table>"
                             f"<div class='note'>Size = largest {'board-lot (100-share) ' if lot == 100 else ''}quantity "
                             f"whose loss at the stop, including buy and sell fees, stays within the smaller of "
                             f"{risk_pct:g}% of capital and your max-loss cap, and whose cost fits your capital.</div></div>",
                             unsafe_allow_html=True)
                mgmt = [
                    ("💰 Partial profit-taking",
                     f"Sell {half1:,} shares at T1 {cur}{t1:.3f} and {half2:,} at T2 {cur}{t2:.3f} → net "
                     f"{cur}{part:,.2f}. If the rest is stopped at break-even after T1: net {cur}{part_be:,.2f}."
                     if half2 else f"Position too small to split into board lots: sell all at T1 (net {cur}{p1:,.2f})."),
                    ("🔒 Trailing stop",
                     f"After T1, move the stop to break-even ({cur}{be:.3f}). Then trail it at the highest close − 2 × ATR "
                     f"(right now that would be {cur}{trail_now:.3f}); only ever move it up." if be else "N/A"),
                    ("⛔ Trade invalidation",
                     f"Exit if a {bar_word} closes below the stop {cur}{stop:.3f}; reassess if price closes below support "
                     f"{cur}{plan['s1']:.3f}, if ADX turns Strong Bearish, if a breakout entry falls back below "
                     f"{cur}{plan['t1']:.3f} on volume, or if T1 isn't reached within about "
                     f"{2 * (plan['bars_t1'] or 5)} {bar_word}s (time stop)."),
                ]
                st.markdown("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px'>"
                            + "".join(f"<div class='card' style='margin:0'><b>{a}</b><div style='margin-top:6px;"
                                      f"line-height:1.6'>{b}</div></div>" for a, b in mgmt) + "</div>",
                            unsafe_allow_html=True)
                if ps["buy_fees"].get("brokerage") is not None and sh:
                    fb_ = ps["buy_fees"]
                    st.caption(f"Buy-side fees for {sh:,} shares: brokerage {cur}{fb_['brokerage']:.2f} · clearing "
                               f"{cur}{fb_['clearing']:.2f} · stamp duty {cur}{fb_['stamp']:.2f}"
                               + (f" · platform {cur}{fb_['platform']:.2f}" if fb_['platform'] else "")
                               + ". Sell-side fees are similar and are included in the risk and profit numbers.")

                # ---- send to paper trading
                st.markdown("##### 💼 Practise this plan")
                st.button("Send this plan to 💼 Paper Trade (limit buy + stop loss + take profit)",
                          disabled=not sh, on_click=ui.goto, args=("paper",), kwargs={"pt_prefill": {
                              "ticker": ticker, "entry": entry, "stop": stop, "t1": t1, "qty": sh}})
                st.caption("Paper trading only: this app never places real orders.")

        # ================= Multi-timeframe
        with tabs[6]:
            lab_of = {k: lab for k, lab, _, _ in ta_plus.TIMEFRAMES}
            avail = [k for k, _, _, _ in ta_plus.TIMEFRAMES if tf_rows.get(k, {}).get("available")]
            missing = [f"{lab_of[k]} ({tf_rows[k].get('reason', 'n/a')})" for k, _, _, _ in ta_plus.TIMEFRAMES
                       if not tf_rows.get(k, {}).get("available")]
            sw_col = {"Aligned bullish": P["up"], "Aligned bearish": P["down"], "Conflict": P["warn"]}.get(
                mtf["swing"], P["muted"])
            sig_col = lambda s_: P["up"] if "Bullish" in s_ else (P["down"] if "Bearish" in s_ else P["muted"])  # noqa: E731
            cards = ""
            for k in avail:
                r_ = tf_rows[k]
                sw = " · swing priority" if k in ta_plus.SWING_WEIGHTS else ""
                cards += f"""<div class='card' style='margin:0;border-top:3px solid {sig_col(r_['signal'])}'>
<div style='display:flex;justify-content:space-between;align-items:center'><b>{lab_of[k]}</b>{pill(r_['signal'], sig_col(r_['signal']))}</div>
<div class='muted' style='font-size:12px;margin:2px 0 8px'>{r_['candles']} candles{sw}{' · last candle excluded (forming)' if (r_.get('forming') and closed_only) else ''}{' · ⚠ stale' if r_.get('stale') else ''}</div>
<table class='kv'><tr><td>Trend</td><td>{r_['trend']}</td></tr><tr><td>EMA alignment</td><td style='font-weight:500'>{r_['ema']}</td></tr>
<tr><td>RSI</td><td>{r_['rsi']:.1f} · {r_['rsi_txt']}</td></tr><tr><td>MACD</td><td style='font-weight:500'>{r_['macd_txt']}</td></tr>
<tr><td>Score</td><td>{r_['score']:+d} / ±7</td></tr></table></div>"""
            warn = ""
            if mtf["disagree"]:
                warn = (f"<div class='reason'><span>⚠️</span><span style='color:{P['warn']}'>Timeframes disagree: bullish on "
                        f"{', '.join(lab_of[k] for k in mtf['bulls'])}; bearish on "
                        f"{', '.join(lab_of[k] for k in mtf['bears'])}. Lower timeframes often pull back against "
                        f"the bigger trend; trade in the direction of the higher timeframe.</span></div>")
            st.markdown(f"""
<div class="card" style="border-left:3px solid {sw_col}">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
<h3 style="margin:0">⏱️ Multi-timeframe analysis</h3>{pill('Swing (W/D/4H): ' + mtf['swing'], sw_col)}</div>
<div class="muted" style="margin-top:8px">For swing trading the Weekly (×3), Daily (×2) and 4-Hour (×1) timeframes are weighted
most: weighted swing score {mtf['swing_score']:+d} / ±{mtf['swing_max']}.</div>{warn}</div>""", unsafe_allow_html=True)
            st.markdown(f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px'>"
                        f"{cards}</div>", unsafe_allow_html=True)
            if missing:
                st.caption("Not shown (insufficient real data, nothing is filled in): " + "; ".join(missing))
            st.markdown("<div class='note'>Per-timeframe score (±7): trend ±2 (price vs EMA50 and its slope) · EMA "
                        "alignment ±2 (±1 if EMA200 unavailable) · RSI +1 (55–70) / −1 (30–45) · MACD ±1 vs signal line "
                        "and ±1 vs zero line. ≥ +2 Bullish, ≥ +5 Strong Bullish (same for bearish). EMA200 needs ≥ 250 "
                        "candles and EMA50 ≥ 100, otherwise shown as N/A. 4-Hour candles are built from real 1-hour "
                        "candles (Yahoo has no 4H feed).</div>", unsafe_allow_html=True)

        # ================= Volume
        with tabs[7]:
            # ---- Volume analysis card
            if va is None:
                st.info("Not enough volume data for a volume analysis.")
            else:
                rv_col = P["warn"] if va["rvol"] >= SPIKE_RVOL else (P["up"] if va["rvol"] >= 1.2 else
                                                                     (P["muted"] if va["rvol"] >= 0.8 else P["down"]))
                ud_txt = "∞" if np.isinf(va["ud"]) else f"{va['ud']:.2f}"
                ud_col = P["up"] if va["ud"] >= 1.3 else (P["down"] if va["ud"] <= 0.77 else P["muted"])
                ud_lab = "accumulation" if va["ud"] >= 1.3 else ("distribution" if va["ud"] <= 0.77 else "balanced")
                part_txt = "–" if pd.isna(va["part"]) else f"{va['part']:.2f}×"
                if va["brk"]:
                    b = va["brk"]
                    gcol = {"Confirmed": P["up"] if b["type"] == "Breakout" else P["down"], "Moderate": P["warn"],
                            "Weak": P["muted"]}[b["grade"]]
                    when = "latest bar" if b["bars_ago"] == 0 else f"{b['bars_ago']} bar(s) ago"
                    brk_html = (f"{pill(b['type'] + ' · ' + b['grade'], gcol)}<div class='muted' style='margin-top:6px'>"
                                f"{b['type']} of the 20-bar {'high' if b['type'] == 'Breakout' else 'low'} "
                                f"{b['level']:.3f} ({when}) on <b>{b['rvol']:.1f}×</b> normal volume. "
                                + ("Strong volume = real buyers behind it." if b['grade'] == 'Confirmed' and b['type'] == 'Breakout'
                                   else "Strong volume = real selling behind it." if b['grade'] == 'Confirmed'
                                   else "Volume is so-so: wait for follow-through." if b['grade'] == 'Moderate'
                                   else "Low volume: higher risk of a fakeout.") + "</div>")
                else:
                    brk_html = "<div class='muted'>No 20-bar breakout or breakdown in the last 10 bars.</div>"
                flags_html = "".join(f"<div class='reason'><span>{ic}</span><span style='color:{c}'>{t}</span></div>"
                                     for ic, t, c in va["flags"]) or "<div class='muted'>Nothing unusual right now.</div>"
                why_html = "".join(f"<div class='reason'><span>•</span><span>{w}</span></div>" for w in va["why"])
                spk = va["spikes"][-6:][::-1]
                spk_html = "".join(
                    f"<div class='reason'><span style='min-width:70px'>{z['date'].strftime('%d %b')}</span>"
                    f"<b style='color:{P['warn']};min-width:52px'>{z['rvol']:.1f}×</b>"
                    f"<span style='color:{P['up'] if z['chg'] >= 0 else P['down']}'>{z['chg']:+.1f}%</span>"
                    f"<span class='muted'>{'🚨 unusual' if z['rvol'] >= UNUSUAL_RVOL else '⚡ spike'}</span></div>"
                    for z in spk) or "<div class='muted'>No spikes (2×+) in the last 60 bars.</div>"
                st.markdown(f"""
    <div class="card" style="border-left:3px solid {va['vcol']}">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
      <h3 style="margin:0">📊 Volume analysis</h3>{pill(va['verdict'], va['vcol'])}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0">
      <div><div class="muted">Volume (latest)</div><div style="font-size:20px;font-weight:600">{fmt_vol(va['vol'])}</div></div>
      <div><div class="muted">20-{bar_word} average</div><div style="font-size:20px;font-weight:600">{fmt_vol(va['avg'])}</div></div>
      <div><div class="muted">Relative volume (RVOL)</div><div style="font-size:20px;font-weight:600;color:{rv_col}">{va['rvol']:.2f}×</div></div>
      <div><div class="muted">Last 5 vs average</div><div style="font-size:20px;font-weight:600">{part_txt}</div></div>
      <div><div class="muted">Up / down volume (20)</div><div style="font-size:20px;font-weight:600;color:{ud_col}">{ud_txt}</div>
           <div class="muted" style="font-size:12px">{ud_lab}</div></div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:18px">
      <div><b>Is the move backed by volume?</b>{why_html}</div>
      <div><b>Breakout confirmation</b><div style="margin-top:8px">{brk_html}</div>
           <div style="margin-top:14px"><b>Unusual activity</b></div>{flags_html}</div>
      <div><b>Recent volume spikes (last 60 bars)</b>{spk_html}</div>
    </div>
    <div class="note">RVOL = today's volume ÷ average of the previous 20 {bar_word}s. 1× = normal, 2×+ = spike, 3×+ = unusual.
    Price moves on high volume are more trustworthy; moves on low volume often fade. Up/down volume above 1 = more shares
    traded on green candles than red (buyers in control).</div></div>""", unsafe_allow_html=True)

        # ================= ADX
        with tabs[8]:
            if not ax:
                st.info("Not enough candles for ADX (needs about 30).")
            # ---- ADX card
            if ax:
                rc = regime_color(ax["regime"])
                gauge_w = min(ax["adx"], 60) / 60 * 100
                di_total = ax["pdi"] + ax["mdi"] or 1
                pdi_w = ax["pdi"] / di_total * 100
                trend_arrow = "↗ rising" if ax["rising"] else ("↘ falling" if ax["falling"] else "→ flat")
                notes_html = "".join(f"<div class='reason'><span>•</span><span>{n}</span></div>" for n in ax["notes"])
                regimes = ["Strong Bearish Trend", "Weak Bearish Trend", "Sideways Market", "Weak Bullish Trend",
                           "Strong Bullish Trend"]
                scale = "".join(
                    f"<div style='flex:1;text-align:center;padding:7px 4px;border-radius:8px;font-size:12px;"
                    f"background:{rgba(regime_color(r_), .22 if r_ == ax['regime'] else .06)};"
                    f"color:{regime_color(r_) if r_ == ax['regime'] else P['muted']};"
                    f"font-weight:{700 if r_ == ax['regime'] else 500};"
                    f"border:1px solid {rgba(regime_color(r_), .6) if r_ == ax['regime'] else P['border']}'>{r_}</div>"
                    for r_ in regimes)
                st.markdown(f"""
    <div class="card" style="border-left:3px solid {rc}">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
      <h3 style="margin:0">📐 Trend strength (ADX 14)</h3>{pill(ax['regime'], rc)}</div>
    <div style="display:flex;gap:6px;margin:14px 0 16px;flex-wrap:wrap">{scale}</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:20px">
     <div>
      <div class="muted">ADX: how strong (not which way)</div>
      <div style="display:flex;align-items:baseline;gap:10px"><span style="font-size:28px;font-weight:700">{ax['adx']:.1f}</span>
       <span class="muted">{trend_arrow} (5 {bar_word_a}s ago {ax['adx5']:.1f})</span></div>
      <div style="position:relative;height:10px;border-radius:6px;background:{P['card2']};margin:8px 0 4px;overflow:hidden">
       <div style="position:absolute;left:0;top:0;bottom:0;width:{gauge_w:.0f}%;background:{rc};opacity:.85"></div>
       <div style="position:absolute;left:{20/60*100:.1f}%;top:0;bottom:0;width:2px;background:{P['bg']}"></div>
       <div style="position:absolute;left:{25/60*100:.1f}%;top:0;bottom:0;width:2px;background:{P['bg']}"></div></div>
      <div class="muted" style="font-size:11.5px;display:flex;justify-content:space-between">
       <span>0 · no trend</span><span>20</span><span>25 · strong</span><span>40+ · very strong</span></div>
     </div>
     <div>
      <div class="muted">+DI vs −DI: which side is winning (direction)</div>
      <div style="display:flex;justify-content:space-between;margin-top:6px;font-weight:600">
       <span style="color:{P['up']}">+DI {ax['pdi']:.1f} buyers</span><span style="color:{P['down']}">sellers −DI {ax['mdi']:.1f}</span></div>
      <div style="display:flex;height:10px;border-radius:6px;overflow:hidden;margin-top:6px">
       <div style="width:{pdi_w:.0f}%;background:{P['up']};opacity:.8"></div>
       <div style="flex:1;background:{P['down']};opacity:.8"></div></div>
      <div class="muted" style="font-size:11.5px;margin-top:4px">{'Buyers' if ax['pdi'] >= ax['mdi'] else 'Sellers'} have the upper hand
       (gap {abs(ax['pdi'] - ax['mdi']):.1f})</div>
     </div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-top:14px">
     <div><b>What the indicators say</b>{notes_html}</div>
     <div><b>How to trade this condition</b><div class="reason"><span>🧭</span><span>{ax['playbook']}</span></div></div>
    </div>
    <div class="note">ADX below 20 = sideways (direction ignored) · 20–25 = weak trend · 25+ = strong trend. The direction comes from
    +DI (buying pressure) vs −DI (selling pressure). ADX rises in strong up-trends <i>and</i> strong down-trends, so it never
    tells direction on its own. Chart: bottom panel, white = ADX, green = +DI, red = −DI.</div></div>""", unsafe_allow_html=True)
            st.write("")

        # ================= Patterns
        with tabs[9]:
            st.markdown("<div class='note'>Patterns in the last 5 candles. Each is checked against the trend before "
                        "it and nearby support/resistance; reversal patterns in the wrong place are marked low "
                        "reliability rather than turned into signals.</div>", unsafe_allow_html=True)
            if not pats_full:
                st.info("No candlestick pattern in the last 5 candles.")
            for p_ in pats_full:
                kc = {"bull": P["up"], "bear": P["down"]}.get(p_["kind"], P["muted"])
                kind_txt = {"bull": "Bullish", "bear": "Bearish"}.get(p_["kind"], "Neutral")
                perf = p_["perf"]
                if perf.get("win") is not None:
                    perf_txt = (f"{perf['n']} earlier times in this stock's history: price was "
                                f"{'higher' if p_['kind'] != 'bear' else 'lower'} {p_['fwd']} {bar_word}s later "
                                f"<b>{perf['win']:.0f}%</b> of the time (vs {perf['base']:.0f}% for any random "
                                f"{bar_word}), average move {perf['avg']:+.2f}%.")
                else:
                    perf_txt = f"Insufficient history (only {perf['n']} earlier occurrence(s); needs ≥ 5)."
                when = "latest candle" if p_["bars_ago"] == 0 else f"{p_['bars_ago']} candle(s) ago"
                use_col = (P["up"] if p_["use"].startswith("Actionable") else
                           P["down"] if p_["use"].startswith(("Ignore", "Low")) else P["warn"])
                st.markdown(f"""
<div class="card" style="border-left:3px solid {kc}">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
 <div><b style="font-size:16px">{p_['name']}</b> <span class="muted">· {when}</span></div>
 <div>{pill(kind_txt, kc)} {pill(p_['use'].split(':')[0], use_col)}</div></div>
<table class="kv" style="margin-top:10px">
<tr><td>Detection</td><td style="font-weight:500">{p_['rule']}</td></tr>
<tr><td>Context</td><td style="font-weight:500">{p_['context']} · context {p_['ctx_grade'].lower()}</td></tr>
<tr><td>Confirmation</td><td style="font-weight:500">{p_['confirmation']}</td></tr>
<tr><td>Historical performance</td><td style="font-weight:500">{perf_txt}</td></tr>
<tr><td>How to use</td><td style="font-weight:500">{p_['use']}</td></tr></table></div>""", unsafe_allow_html=True)
                with st.expander(f"Candle formation details · {p_['name']}"):
                    st.dataframe(pd.DataFrame(p_["candles"]), hide_index=True, width="stretch",
                                 column_config={c_: st.column_config.NumberColumn(format="%.3f")
                                                for c_ in ("Open", "High", "Low", "Close")}
                                 | {c_: st.column_config.NumberColumn(format="%.0f%%")
                                    for c_ in ("Body % of range", "Upper wick %", "Lower wick %")})
            st.markdown("<div class='note'>Confirmation = a later candle (within 3) closes beyond the pattern's high "
                        "(bullish) or low (bearish). Historical performance only uses earlier occurrences with a full "
                        "5-candle window afterwards (no look-ahead). See 🎓 Learn → Candlestick Patterns.</div>",
                        unsafe_allow_html=True)

        # ================= Forecast & accuracy
        with tabs[10]:
            AP.render_horizons(hz, ops, tech['x'])
            # ---- Forecast
            h = fc["horizon"]
            unit_short = unit.split()[-1]
            cross_up = first_cross(fc["centre"], plan["t1"], up=True)
            cross_dn = first_cross(fc["centre"], sup[0][0], up=False)
            if fc["direction"] == "Uptrend":
                hint = (f"If this pace holds, price could reach resistance <b>{plan['t1']:.3f}</b> in about "
                        f"<b>{cross_up} {unit_short}</b>, a likely area for sellers / taking profit."
                        if cross_up else f"At this pace, price stays below resistance {plan['t1']:.3f} within {h} {unit_short}.")
            elif fc["direction"] == "Downtrend":
                hint = (f"If this pace holds, price could drop to support <b>{sup[0][0]:.3f}</b> in about "
                        f"<b>{cross_dn} {unit_short}</b>. Watch there for a bounce (buy) or a break (more downside)."
                        if cross_dn else f"At this pace, price stays above support {sup[0][0]:.3f} within {h} {unit_short}.")
            else:
                hint = (f"No reliable trend: price is moving sideways. Range traders buy near support "
                        f"{sup[0][0]:.3f} and sell near resistance {plan['t1']:.3f}.")
            st.markdown(f"""
    <div class="card"><h3>Trend forecast · next {h} {unit_short}</h3>
    <div style="margin-bottom:10px">{pill(f"{fc['arrow']} {fc['direction']}", dir_col)} {pill(f"{fc['strength']} trend · R² {fc['r2']:.2f}", P['muted'])}</div>
    <table class="kv">
    <tr><td>Projected price</td><td>{fc['centre'][-1]:.3f} <span class="muted">({fc['total']:+.1f}%)</span></td></tr>
    <tr><td>Likely range (68%)</td><td>{fc['lo1'][-1]:.3f} – {fc['hi1'][-1]:.3f}</td></tr>
    <tr><td>Wide range (95%)</td><td>{fc['lo2'][-1]:.3f} – {fc['hi2'][-1]:.3f}</td></tr>
    <tr><td>Trend speed</td><td>{fc['per_bar']:+.2f}% per bar</td></tr>
    <tr><td>Momentum</td><td style="font-weight:500">{fc['momentum']}</td></tr>
    </table>
    <div style="margin-top:12px;font-size:14px;line-height:1.5">{hint}</div>
    <div class="note">How it works: a trend line is fitted to the last {fc['lookback']} bars and extended forward; the bands
    come from the stock's normal volatility. It shows where price goes <i>if the current trend continues</i>. News, results and
    market moves can break any trend, so treat it as a guide, not a prediction. R² near 1 = clean trend, near 0 = choppy.</div>
    </div>""", unsafe_allow_html=True)
            st.markdown("#### 🎯 How accurate has this forecast been? (walk-forward test)")
            if bt_m is None:
                st.info("Not enough history for an out-of-sample test.")
            else:
                cov_note = ta_plus.calibration_note(bt_m)
                skill_col = P["up"] if bt_m["skill"] > 0 else P["down"]
                st.markdown(f"""
<div class="card"><div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px">
 <div><div class="muted">Forecasts tested</div><div style="font-size:20px;font-weight:600">{bt_m['n']}</div></div>
 <div><div class="muted">Directional accuracy</div><div style="font-size:20px;font-weight:600">{bt_m['dir_acc']:.0f}%</div>
   <div class="muted" style="font-size:12px">'always up' would score {bt_m['up_rate']:.0f}%</div></div>
 <div><div class="muted">MAE</div><div style="font-size:20px;font-weight:600">{cur}{bt_m['mae']:.3f}</div>
   <div class="muted" style="font-size:12px">{bt_m['mape']:.1f}% of price</div></div>
 <div><div class="muted">RMSE</div><div style="font-size:20px;font-weight:600">{cur}{bt_m['rmse']:.3f}</div></div>
 <div><div class="muted">vs 'no change' guess</div><div style="font-size:20px;font-weight:600;color:{skill_col}">{bt_m['skill']:+.0f}%</div>
   <div class="muted" style="font-size:12px">naive MAE {cur}{bt_m['naive_mae']:.3f}</div></div>
 <div><div class="muted">Band coverage 68% / 95%</div><div style="font-size:20px;font-weight:600">{bt_m['cov68']:.0f}% / {bt_m['cov95']:.0f}%</div>
   <div class="muted" style="font-size:12px">{cov_note}</div></div>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:14px">
 <div><b>1 · Model fit (in-sample)</b><div class="muted" style="margin-top:4px">R² {fc['r2']:.2f} = how neatly a straight
   trend line fits the last {fc['lookback']} {bar_word}s. It says nothing about the future.</div></div>
 <div><b>2 · Forecast accuracy (out-of-sample)</b><div class="muted" style="margin-top:4px">At {bt_m['n']} past dates the
   model was fitted using only data up to that date and asked to predict {fc['horizon']} {bar_word}s ahead; results above
   compare those predictions with what actually happened.</div></div>
 <div><b>3 · Uncertainty (bands)</b><div class="muted" style="margin-top:4px">The 68%/95% bands are nominal ranges from
   recent volatility, not guaranteed probabilities. Historically they contained the real price {bt_m['cov68']:.0f}% /
   {bt_m['cov95']:.0f}% of the time ({cov_note}).</div></div>
</div></div>""", unsafe_allow_html=True)
                x_lab = [str(d.index[int(t_ + fc['horizon'])])[:10] for t_ in bt_df["t"]]
                figb = go.Figure()
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["hi95"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["lo95"], fill="tonexty", fillcolor=rgba(P["accent"], .12),
                                          line=dict(width=0), name="95% band (nominal)"))
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["pred"], name="Forecast", mode="lines+markers",
                                          line=dict(color=P["accent"], dash="dash"), marker=dict(size=4)))
                figb.add_trace(go.Scatter(x=x_lab, y=bt_df["actual"], name="Actual", mode="lines+markers",
                                          line=dict(color=P["text"]), marker=dict(size=4)))
                figb.update_layout(template="plotly_dark", height=340, margin=dict(l=8, r=8, t=30, b=8),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   title=dict(text=f"Forecast vs actual ({fc['horizon']} {bar_word}s ahead, out-of-sample)",
                                              font=dict(size=13)),
                                   legend=dict(orientation="h", y=1.12), hovermode="x unified",
                                   font=dict(family="Inter, sans-serif", color=P["text"]))
                figb.update_yaxes(gridcolor=P["grid"])
                st.plotly_chart(figb, width="stretch", config={"displaylogo": False})

                rows_tf = []
                for k, lab, _, _ in ta_plus.TIMEFRAMES:
                    if k not in tf_data or len(tf_data[k]) < fc["lookback"] + fc["horizon"] + 10:
                        continue
                    _, m_tf = ta_plus.walk_forward(tf_data[k]["Close"].values, fc["horizon"], fc["lookback"])
                    if m_tf:
                        rows_tf.append({"Timeframe": lab, "Tests": m_tf["n"], "Direction %": m_tf["dir_acc"],
                                        "MAE %": m_tf["mape"], "Skill vs no-change %": m_tf["skill"],
                                        "68% coverage": m_tf["cov68"], "95% coverage": m_tf["cov95"]})
                if rows_tf:
                    st.markdown(f"**Performance by timeframe** · {display_name(ticker)} · {fc['horizon']} bars ahead, "
                                f"{fc['lookback']}-bar fit")
                    pct = st.column_config.NumberColumn(format="%.0f%%")
                    st.dataframe(pd.DataFrame(rows_tf), hide_index=True, width="stretch",
                                 column_config={"Direction %": pct, "MAE %": st.column_config.NumberColumn(format="%.1f%%"),
                                                "Skill vs no-change %": st.column_config.NumberColumn(format="%+.0f%%"),
                                                "68% coverage": pct, "95% coverage": pct})
                if st.button(f"Run the same test for every stock in my list ({len(universe)})"):
                    with st.spinner("Back-testing…"):
                        data_u = get_many(tuple(universe), interval, period)
                        res_u = []
                        for t_, dfu in data_u.items():
                            if len(dfu) < fc["lookback"] + fc["horizon"] + 10:
                                continue
                            _, m_u = ta_plus.walk_forward(dfu["Close"].values, fc["horizon"], fc["lookback"])
                            if m_u:
                                res_u.append({"Name": display_name(t_), "Ticker": t_, "Tests": m_u["n"],
                                              "Direction %": m_u["dir_acc"], "MAE %": m_u["mape"],
                                              "Skill vs no-change %": m_u["skill"], "95% coverage": m_u["cov95"]})
                        st.session_state["bt_universe"] = (tf_name, fc["horizon"], fc["lookback"], res_u)
                if st.session_state.get("bt_universe"):
                    tfn, hz, lb, res_u = st.session_state["bt_universe"]
                    st.markdown(f"**Performance by stock** · {tfn} · {hz} bars ahead · {lb}-bar fit")
                    st.dataframe(pd.DataFrame(res_u).sort_values("Direction %", ascending=False), hide_index=True,
                                 width="stretch",
                                 column_config={"Direction %": st.column_config.NumberColumn(format="%.0f%%"),
                                                "MAE %": st.column_config.NumberColumn(format="%.1f%%"),
                                                "Skill vs no-change %": st.column_config.NumberColumn(format="%+.0f%%"),
                                                "95% coverage": st.column_config.NumberColumn(format="%.0f%%")})
                st.markdown("<div class='note'>Method: walk-forward, out-of-sample. Each test origin uses only candles up to "
                            "that date (no look-ahead), predicts the close N bars later, then moves forward by half the "
                            "horizon. Skill > 0% means the trend line beat simply guessing 'no change'. A directional "
                            "accuracy near 50% means the forecast has little predictive value for this stock.</div>",
                            unsafe_allow_html=True)

        # ================= Levels
        with tabs[11]:
            with st.expander("Support & resistance levels (table)", expanded=bool(extra_rows)):
                lv = pd.DataFrame(
                    [{"Type": "Resistance", "Level": l, "Touches": str(n) if n else "projected", "Distance %": (l / price - 1) * 100} for l, n in reversed(res)]
                    + [{"Type": "▶ Current price", "Level": price, "Touches": "", "Distance %": 0.0}]
                    + [{"Type": "Support", "Level": l, "Touches": str(n) if n else "projected", "Distance %": (l / price - 1) * 100} for l, n in sup])
                st.dataframe(lv, hide_index=True, width="stretch",
                             column_config={"Level": st.column_config.NumberColumn(format="%.3f"),
                                            "Distance %": st.column_config.NumberColumn(format="%+.1f%%")})
                if extra_rows:
                    st.markdown("**Extra levels**")
                    ex_df = pd.DataFrame(extra_rows).sort_values("Level", ascending=False)
                    st.dataframe(ex_df[["Method", "Role", "Level", "Distance %", "Note"]], hide_index=True,
                                 width="stretch",
                                 column_config={"Level": st.column_config.NumberColumn(format="%.3f"),
                                                "Distance %": st.column_config.NumberColumn(format="%+.1f%%")})

        # ================= Data
        with tabs[12]:
            st.markdown("#### 🧾 Data quality & source audit")
            st.dataframe(pd.DataFrame([{"Check": k, "Value": str(v)} for k, v in dq.items()]), hide_index=True,
                         width="stretch")
            st.markdown("""<div class='note'>
• <b>Source</b>: Yahoo Finance via the free <code>yfinance</code> library (no paid APIs). Bursa quotes are delayed about
15 minutes; US quotes are near real-time in market hours.<br>
• <b>Prices</b> are unadjusted for dividends (they match your broker's chart); Yahoo already adjusts for splits/bonus issues.<br>
• <b>Cleaning</b>: duplicate timestamps, rows with missing/zero prices and rows with High &lt; Low are removed; missing
volume is treated as 0.<br>
• <b>Forming candle</b>: while the market is open the latest candle can still change. With “Signals on completed
candles only” on, all indicators, patterns and signals use the last completed candle.<br>
• <b>Indicators</b> are calculated locally and deterministically (EMA, Wilder RSI/ATR/ADX, MACD 12/26/9). Values
that need more history than available are shown as N/A instead of being estimated.<br>
• <b>4-Hour</b> candles are aggregated from real 1-hour candles (4 per block within each trading day).</div>""",
                        unsafe_allow_html=True)
