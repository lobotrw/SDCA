"""Streamlit page for the composite risk + accumulation/distribution curve model."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import composite as cm

BAND_COLORS = {"Buy": "#22c55e", "Accumulate": "#95d81d", "Trim": "#f59e0b", "Sell": "#ff4444"}


@st.cache_data(ttl=6 * 3600, show_spinner="Loading app indicator data and Coin Metrics…")
def _inputs():
    site, coin, notes = None, None, []
    try:
        site = cm.load_site_data()
    except Exception as e:
        notes.append(f"Railway app data unavailable: {e}")
    try:
        coin = cm.load_coinmetrics()
    except Exception as e:
        notes.append(f"Coin Metrics unavailable: {e}")
    return site, coin, notes


@st.cache_data(ttl=6 * 3600, show_spinner="Building composite model (walk-forward fit takes ~20 s)…")
def _model(close, mode, enabled, onchain_source):
    site, coin, _ = _inputs()
    return cm.build(close, mode=mode, enabled=enabled, site=site, cm=coin, onchain_source=onchain_source)


def render(close):
    close = close[close.index < pd.Timestamp.now("UTC").tz_localize(None).normalize()]   # closed days only
    st.title("Composite risk + accumulation/distribution curve")
    c1, c2 = st.columns([3, 1])
    with c2:
        mode = st.radio("Mode", cm.MODES, format_func={"causal": "No look-ahead (default)",
                                                        "replica": "Replica of the Railway app (hindsight)"}.get)
        src = st.radio("On-chain source", ("coinmetrics", "site"),
                       index=0 if mode == "causal" else 1,
                       format_func={"coinmetrics": "Coin Metrics free proxy", "site": "Railway app series"}.get)
        enabled = tuple(k for k in cm.INDICATORS
                        if st.checkbox(cm.INDICATOR_LABELS[k], value=k in cm.DEFAULT_ENABLED, key=f"ind_{k}"))
        if not enabled:
            st.warning("Enable at least one indicator.")
            return
        cash = st.number_input("Your cash reserve, USD", 0.0, 1e9, 1000.0, 100.0)
        start = st.date_input("Backtest start", pd.Timestamp("2015-01-01"))
        capital = st.number_input("Starting capital, USD", 1.0, 1e9, 10000.0, 1000.0)
        st.caption("Curve: % of cash bought (+) or % of BTC sold (-) per day at each risk level.")
        edited = st.data_editor(pd.DataFrame({"risk %": cm.CURVE_RISK_NODES, "rate %/day": cm.CURVE_DEFAULT_VALUES}),
                                disabled=["risk %"], hide_index=True, key="curve")
        curve = tuple(float(np.clip(v, -10, 10)) for v in edited["rate %/day"])

    model = _model(close, mode, enabled, src)
    for n in model.notes:
        st.warning(n)
    t = cm.today_action(model, cash, curve)
    s, ledger = cm.backtest(model, str(start), capital, curve)
    f = model.frame

    with c2:
        st.subheader("Today's model action")
        st.metric(f"Composite risk ({t['date']})", f"{t['composite_risk']:.1f}%", t["band"], delta_color="off")
        st.metric("Composite Z", f"{t['composite_z']:+.2f}")
        st.metric("Curve rate", f"{t['curve_rate_pct_per_day']:+.2f}%/day")
        st.write(f"**{t['action']}**" + (f": spend ${t['spend_usd']:,.2f} ≈ {t['buy_btc']:.8f} BTC" if "spend_usd" in t else ""))
        st.write(f"Price ${t['price']:,.0f} · median ${t['median']:,.0f} · Q1% ${t['q01_rail']:,.0f} · Q99% ${t['q99_rail']:,.0f}")

    with c1:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        ax.set_yscale("log")
        ax.plot(f.index, f["close"], color="lightgrey", lw=0.8)
        g = f.dropna(subset=["comp_risk"])
        ax.scatter(g.index, g["close"], c=[BAND_COLORS[cm.composite_band(r)] for r in g["comp_risk"]], s=3)
        for col, ls in (("q01", ":"), ("q50", "--"), ("q99", ":")):
            ax.plot(f.index, f[col], ls, color="#4ea6c9", lw=1, label=col)
        ax.legend(loc="upper left")
        for lo, hi, name in ((0, 25, "Buy"), (25, 50, "Accumulate"), (50, 75, "Trim"), (75, 100, "Sell")):
            ax2.axhspan(lo, hi, color=BAND_COLORS[name], alpha=0.15)
        ax2.plot(g.index, g["comp_risk"], color="black", lw=0.8)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Composite risk %")
        st.pyplot(fig)

        st.subheader("Curve backtest")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Portfolio value", f"${s['portfolio_value']:,.0f}", f"{s['vs_lump_sum_pct']:+.1f}% vs lump sum")
        m2.metric("Lump sum value", f"${s['lump_sum_value']:,.0f}")
        m3.metric("Net BTC / cash", f"{s['net_btc']:.4f} BTC", f"${s['cash_reserve']:,.0f} cash", delta_color="off")
        m4.metric("Max drawdown", f"-{s['max_drawdown_pct']:.1f}%", f"lump -{s['max_drawdown_lump_pct']:.1f}%", delta_color="off")
        st.caption(f"{s['days']} days, buy {s['buy_days']} / sell {s['sell_days']}, avg rate {s['avg_daily_rate_pct']:+.2f}%/day")
        st.line_chart(ledger[["portfolio", "lump_sum", "cash"]])
        st.download_button("Download backtest CSV", ledger.to_csv(), "accum_dist_backtest.csv")
    st.caption("Model logic re-implemented from the Investing Campus SDCA signal web app. Not financial advice.")
