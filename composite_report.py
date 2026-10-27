"""Run the composite risk model end to end and write charts + numbers to out/composite/.

    python composite_report.py                       # causal (default) + replica comparison
    python composite_report.py --cash 5000 --start 2018-01-01 --capital 10000
    python composite_report.py --indicators price,sharpe,cbpl,onchain --onchain-source site
    python composite_report.py --curve "10,10,10,10,5,0,0,0,0,0,0,0,0,0,0,0,0,-0.5,-2,-4,-10"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import composite as cm  # noqa: E402

OUT = Path(__file__).resolve().parent / "out" / "composite"
BAND_COLORS = {"Buy": "#22c55e", "Accumulate": "#95d81d", "Trim": "#f59e0b", "Sell": "#ff4444"}


def _risk_colors(risk):
    return [BAND_COLORS.get(cm.composite_band(r), "#888") for r in risk]


def chart_composite(model, path):
    f = model.frame.dropna(subset=["comp_risk"])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
    ax.set_yscale("log")
    ax.plot(model.frame.index, model.frame["close"], color="lightgrey", lw=0.8, zorder=0)
    ax.scatter(f.index, f["close"], c=_risk_colors(f["comp_risk"]), s=3, zorder=2)
    for col, ls, lab in (("q01", ":", "Q1% rail"), ("q50", "--", "Median (Q50)"), ("q99", ":", "Q99% rail")):
        ax.plot(model.frame.index, model.frame[col], ls, color="#4ea6c9", lw=1, label=lab)
    ax.set_title(f"BTC with composite risk ({model.mode}; indicators: {', '.join(model.enabled)})")
    ax.legend(loc="upper left")
    ax.set_ylabel("USD (log)")
    for lo, hi, name in ((0, 25, "Buy"), (25, 50, "Accumulate"), (50, 75, "Trim"), (75, 100, "Sell")):
        ax2.axhspan(lo, hi, color=BAND_COLORS[name], alpha=0.15)
    ax2.plot(f.index, f["comp_risk"], color="black", lw=0.8)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Composite risk %")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_bands(model, path):
    f = model.frame
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_yscale("log")
    cols = ["q01", "q10", "q25", "q50", "q75", "q95", "q99"]
    colors = ["#173c70", "#21838e", "#38a04b", "#d6c420", "#e4741c", "#be0a12"]
    names = ["1-10% Fire sale", "10-25% Accumulate", "25-50% Value", "50-75% Above mid", "75-95% Hot", "95-99% Bubble"]
    for i in range(6):
        ax.fill_between(f.index, f[cols[i]], f[cols[i + 1]], color=colors[i], alpha=0.5, label=names[i], lw=0)
    ax.plot(f.index, f["close"], color="black", lw=0.9, label="BTC close")
    ax.set_title(f"Asymmetric Tail Curvature quantile bands ({model.mode})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_curve(curve, today_risk, path):
    xs = np.linspace(0, 100, 401)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(0, color="grey", lw=0.8)
    ax.plot(xs, cm.curve_rate(xs, curve), color="#d4a756", lw=2)
    ax.scatter(cm.CURVE_RISK_NODES, curve, color="#d4a756", zorder=3)
    if np.isfinite(today_risk):
        ax.axvline(today_risk, color="#4ea6c9", ls="--", label=f"today {today_risk:.1f}%")
        ax.legend()
    ax.set_xlabel("Composite risk %")
    ax.set_ylabel("% cash bought (+) / % BTC sold (-) per day")
    ax.set_title("Accumulation / distribution curve")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_backtest(ledger, summary, title, path):
    fig, ax = plt.subplots(figsize=(15, 7))
    rate = ledger["rate"].values
    trans = ax.get_xaxis_transform()
    ax.fill_between(ledger.index, 0, 1, where=rate > 0.05, color="#0f5e48", alpha=0.15, lw=0, transform=trans, step="post", label="buying")
    ax.fill_between(ledger.index, 0, 1, where=rate < -0.05, color="#892c3a", alpha=0.15, lw=0, transform=trans, step="post", label="selling")
    ax.plot(ledger.index, ledger["portfolio"], color="#d4a756", lw=2, label="Curve strategy")
    ax.plot(ledger.index, ledger["lump_sum"], color="grey", lw=1.4, label="Lump sum day 1")
    ax.plot(ledger.index, ledger["cash"].clip(lower=1), color="#4ea6c9", lw=1, label="Cash reserve (floored at $1 on log axis)".replace("$", "\\$"))
    ax.set_yscale("log")
    ax.set_title(f"{title}: \\${summary['portfolio_value']:,.0f} vs lump \\${summary['lump_sum_value']:,.0f} "
                 f"({summary['vs_lump_sum_pct']:+.1f}%), MDD {summary['max_drawdown_pct']:.1f}% vs {summary['max_drawdown_lump_pct']:.1f}%")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def chart_compare(causal, replica, path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    ax.set_yscale("log")
    ax.plot(causal.frame.index, causal.frame["close"], color="black", lw=0.8, label="BTC")
    ax.plot(replica.frame.index, replica.frame["q50"], color="#be0a12", lw=1.2, label="Median, published full-sample fit (replica)")
    ax.plot(causal.frame.index, causal.frame["q50"], color="#22c55e", lw=1.2, label="Median, walk-forward fit (causal)")
    ax.plot(replica.frame.index, replica.frame["q99"], color="#be0a12", lw=0.8, ls=":")
    ax.plot(causal.frame.index, causal.frame["q99"], color="#22c55e", lw=0.8, ls=":")
    ax.legend(loc="upper left")
    ax.set_title("Hindsight check: replica (app) vs causal (no look-ahead)")
    ax2.plot(replica.frame.index, replica.frame["comp_risk"], color="#be0a12", lw=0.8, label="Composite risk, replica")
    ax2.plot(causal.frame.index, causal.frame["comp_risk"], color="#22c55e", lw=0.8, label="Composite risk, causal")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=cm.MODES, default="causal")
    p.add_argument("--indicators", default=",".join(cm.DEFAULT_ENABLED), help="comma list of " + ",".join(cm.INDICATORS))
    p.add_argument("--onchain-source", choices=("coinmetrics", "site"), default=None)
    p.add_argument("--curve", default=None, help="21 comma-separated values for risk 0,5,...,100")
    p.add_argument("--cash", type=float, default=1000.0, help="cash reserve for today's action")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--no-compare", action="store_true", help="skip the replica/causal comparison")
    a = p.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    curve = cm.parse_curve(a.curve) if a.curve else cm.CURVE_DEFAULT_VALUES
    enabled = tuple(k.strip() for k in a.indicators.split(",") if k.strip())
    close = cm.load_close()
    site = cm.load_site_data()
    coin = cm.load_coinmetrics()
    print(f"price: {close.attrs.get('source')} {close.index[0].date()}..{close.index[-1].date()}")

    def run(mode, onchain_source=a.onchain_source, en=enabled):
        return cm.build(close, mode=mode, enabled=en, site=site, cm=coin, onchain_source=onchain_source)

    model = run(a.mode)
    today = cm.today_action(model, a.cash, curve)
    summary, ledger = cm.backtest(model, a.start, a.capital, curve)
    tag = a.mode
    chart_composite(model, OUT / f"composite_risk_{tag}.png")
    chart_bands(model, OUT / f"atc_bands_{tag}.png")
    chart_curve(curve, today.get("composite_risk", np.nan), OUT / "curve.png")
    chart_backtest(ledger, summary, f"Curve backtest ({tag})", OUT / f"backtest_{tag}.png")
    ledger.to_csv(OUT / f"backtest_{tag}.csv")
    model.frame.to_csv(OUT / f"series_{tag}.csv")
    result = {"mode": a.mode, "enabled": model.enabled, "onchain_source": model.frame.attrs["onchain_source"],
              "notes": model.notes, "today": today, "backtest": summary}

    if not a.no_compare:
        other = "replica" if a.mode == "causal" else "causal"
        om = run(other)
        causal, replica = (model, om) if a.mode == "causal" else (om, model)
        chart_compare(causal, replica, OUT / "causal_vs_replica.png")
        chart_backtest(*reversed(cm.backtest(om, a.start, a.capital, curve)), f"Curve backtest ({other})", OUT / f"backtest_{other}.png")
        grid = {}
        for atc_mode in ("replica", "causal"):
            for src in ("site", "coinmetrics"):
                mm = run(atc_mode, src)
                grid[f"atc={atc_mode}, onchain={src}"] = {
                    "today_risk": cm.today_action(mm, a.cash, curve).get("composite_risk"),
                    **{k: cm.backtest(mm, a.start, a.capital, curve)[0][k] for k in
                       ("portfolio_value", "lump_sum_value", "vs_lump_sum_pct", "max_drawdown_pct", "net_btc")}}
            pm = run(atc_mode, en=("price",))
            grid[f"atc={atc_mode}, price only"] = {
                "today_risk": cm.today_action(pm, a.cash, curve).get("composite_risk"),
                **{k: cm.backtest(pm, a.start, a.capital, curve)[0][k] for k in
                   ("portfolio_value", "lump_sum_value", "vs_lump_sum_pct", "max_drawdown_pct", "net_btc")}}
        result["comparison"] = grid
        result["today_other_mode"] = cm.today_action(om, a.cash, curve)

    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))
    print(f"charts written to {OUT}")
    return result


if __name__ == "__main__":
    main()
