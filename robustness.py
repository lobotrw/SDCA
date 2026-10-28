"""Robustness tests for the composite risk model and the 200-week SMA SDCA model.

    python robustness.py                        # full run -> out/robustness/
    python robustness.py --windows 500 --paths 600,300,300 --jobs 7

Part 1 runs backtests on real data: random windows (2-6 years, starting 2015 or later),
sensitivity checks (fees, next-day execution, perturbed curve or thresholds) and an
anchored walk-forward, where the curve is chosen on in-sample data only.
Part 2 is a Monte Carlo: real history up to the last close, then synthetic future paths
from a stationary block bootstrap, the same with half the drift, or a phase-aligned
halving-cycle bootstrap. The causal model runs forward on each path and never sees the
future. Nothing here tunes the strategy: the default curve and thresholds are tested as
they are.

Strategy sizing, the same as in the app: a positive daily rate buys that % of the current
cash, a negative rate sells that % of the current BTC.
SDCA 200W sizing: the original app gives signals but no position sizes, so we use
+5 %/day of cash on DCA-IN days and -2 %/day of BTC on DCA-OUT days. That is an
assumption, and the perturbation test varies it.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import composite as cm
import sdca

OUT = Path(__file__).resolve().parent / "out" / "robustness"
CAPITAL = 10000.0
SDCA_BUY_RATE = 5.0
SDCA_SELL_RATE = 2.0
HALVINGS = pd.to_datetime(["2012-11-28", "2016-07-09", "2020-05-11", "2024-04-20", "2028-03-28"])


# ====================================================================== simulation core
def simulate(price, rate, fee=0.0, lag=0, capital=CAPITAL):
    """Run a daily rate series. rate[t] > 0 buys rate% of current cash and rate[t] < 0 sells
    |rate|% of current BTC. lag=1 executes the day-t decision at the day-t+1 close.
    fee is a fraction of traded notional (fees + slippage). Returns (equity, n_trades)."""
    p = np.asarray(price, float)
    r = np.nan_to_num(np.asarray(rate, float), nan=0.0)
    if lag:
        r = np.r_[np.zeros(lag), r[:-lag]]
    cash, btc, trades = float(capital), 0.0, 0
    eq = np.empty(len(p))
    pl, rl = p.tolist(), r.tolist()
    keep = 1.0 - fee
    for i in range(len(pl)):
        x, px = rl[i], pl[i]
        if x > 1e-4:
            usd = cash * x / 100.0
            if usd > cash:
                usd = cash
            if usd > 0.005:
                cash -= usd
                btc += usd * keep / px
                trades += 1
        elif x < -1e-4:
            s = btc * (-x / 100.0)
            if s > 1e-12:
                btc -= s
                cash += s * px * keep
                trades += 1
        eq[i] = cash + btc * px
    return eq, trades


def lump_sum(price, fee=0.0, capital=CAPITAL):
    p = np.asarray(price, float)
    return capital * (1 - fee) / p[0] * p


def plain_dca(price, fee=0.0, capital=CAPITAL):
    """Same capital spent in equal daily amounts over the whole window."""
    p = np.asarray(price, float)
    a = capital / len(p)
    btc = np.cumsum(a * (1 - fee) / p)
    cash = capital - a * np.arange(1, len(p) + 1)
    return cash + btc * p


def max_drawdown(eq):
    eq = np.asarray(eq, float)
    peak = np.maximum.accumulate(eq)
    return float(np.max((peak - eq) / peak) * 100)


# ====================================================================== rates per variant
def composite_rate(risk, curve=cm.CURVE_DEFAULT_VALUES, shift=0.0):
    risk = np.asarray(risk, float)
    rate = cm.curve_rate(risk - shift, curve)
    return np.where(np.isfinite(risk), rate, 0.0)


def sdca_rate(z, cyc, in_thr=sdca.DEFAULT_IN_THRESHOLD, out_thr=sdca.DEFAULT_CYCLE_OUT_THRESHOLD,
              buy=SDCA_BUY_RATE, sell=SDCA_SELL_RATE):
    z = np.asarray(z, float)
    cyc = np.asarray(cyc, float)
    rate = np.where(np.isfinite(z) & (z < in_thr), buy, 0.0)
    return rate - np.where(np.isfinite(cyc) & (cyc > out_thr), sell, 0.0)


def perturb(rng, kind):
    """Random ±20 % perturbation of the variant's free parameters."""
    if kind == "composite":
        nodes = tuple(np.clip(np.array(cm.CURVE_DEFAULT_VALUES) * rng.uniform(0.8, 1.2, 21), -10, 10))
        return {"curve": nodes, "shift": rng.uniform(-5, 5)}      # ±5 risk pts = ±20 % of a 25-pt band
    tail = (1 - sdca.DEFAULT_CYCLE_OUT_THRESHOLD) * rng.uniform(0.8, 1.2)
    return {"in_thr": sdca.DEFAULT_IN_THRESHOLD * rng.uniform(0.8, 1.2), "out_thr": 1 - tail,
            "buy": SDCA_BUY_RATE * rng.uniform(0.8, 1.2), "sell": SDCA_SELL_RATE * rng.uniform(0.8, 1.2)}


def variant_rate(v, params=None):
    params = params or {}
    if v["kind"] == "composite":
        return composite_rate(v["risk"], params.get("curve", cm.CURVE_DEFAULT_VALUES), params.get("shift", 0.0))
    return sdca_rate(v["z"], v["cyc"], **{k: params[k] for k in ("in_thr", "out_thr", "buy", "sell") if k in params})


# ====================================================================== real-data variants
def real_variants(close=None, site=None, coin=None):
    close = cm.load_close() if close is None else close
    site = cm.load_site_data() if site is None else site
    coin = cm.load_coinmetrics() if coin is None else coin
    out = {}
    specs = {
        "causal_coinmetrics": dict(mode="causal", onchain_source="coinmetrics"),
        "causal_site_onchain": dict(mode="causal", onchain_source="site"),
        "replica_hindsight": dict(mode="replica", onchain_source="site"),
        "price_only_causal": dict(mode="causal", enabled=("price",)),
    }
    for name, kw in specs.items():
        m = cm.build(close, site=site, cm=coin, **kw)
        out[name] = {"kind": "composite", "price": m.frame["close"], "risk": m.frame["comp_risk"]}
    s = sdca.compute_sdca(close)
    out["sdca_200w"] = {"kind": "sdca", "price": s["BTC-USD"], "z": s["Z_Diff"], "cyc": s["Cycle_Pct"]}
    return close, out


def random_windows(index, n, start_min="2015-01-01", min_years=2, max_years=6, seed=0):
    rng = np.random.default_rng(seed)
    index = pd.DatetimeIndex(index)
    s0, end = index.searchsorted(pd.Timestamp(start_min)), len(index) - 1
    wins = []
    while len(wins) < n:
        length = int(rng.uniform(min_years, max_years) * 365.25)
        if s0 + length > end:
            continue
        a = int(rng.integers(s0, end - length + 1))
        wins.append((a, a + length))
    return wins


SETTINGS = {
    "base": dict(fee=0.0, lag=0),
    "fee_0.1%": dict(fee=0.001, lag=0),
    "fee_0.25%": dict(fee=0.0025, lag=0),
    "fee_0.5%": dict(fee=0.005, lag=0),
    "next_day": dict(fee=0.0, lag=1),
    "next_day+fee_0.25%": dict(fee=0.0025, lag=1),
    "perturbed_±20%": dict(fee=0.0, lag=0, perturb=True),
}


def window_study(variants, windows, settings=SETTINGS, seed=1):
    rows = []
    rng = np.random.default_rng(seed)
    pert = [{k: perturb(rng, k) for k in ("composite", "sdca")} for _ in windows]
    for name, v in variants.items():
        price = v["price"].values
        base_rate = variant_rate(v)
        for sname, st in settings.items():
            for wi, (a, b) in enumerate(windows):
                rate = variant_rate(v, pert[wi][v["kind"]]) if st.get("perturb") else base_rate
                p = price[a:b + 1]
                eq, trades = simulate(p, rate[a:b + 1], st["fee"], st["lag"])
                lump, dca = lump_sum(p, st["fee"]), plain_dca(p, st["fee"])
                rows.append(_row(name, sname, wi, v["price"].index[a], b - a, eq, lump, dca, trades))
    return pd.DataFrame(rows)


def _row(variant, setting, wid, start, days, eq, lump, dca, trades):
    return {"variant": variant, "setting": setting, "id": wid, "start": str(pd.Timestamp(start).date()),
            "days": int(days), "final": float(eq[-1]), "ret": float(eq[-1] / CAPITAL - 1),
            "lump_ret": float(lump[-1] / CAPITAL - 1), "dca_ret": float(dca[-1] / CAPITAL - 1),
            "excess_lump": float(eq[-1] / lump[-1] - 1), "excess_dca": float(eq[-1] / dca[-1] - 1),
            "mdd": max_drawdown(eq), "mdd_lump": max_drawdown(lump), "mdd_dca": max_drawdown(dca),
            "trades": int(trades)}


# ---------------------------------------------------------------------- walk-forward
def candidates(kind):
    if kind == "composite":
        base = np.array(cm.CURVE_DEFAULT_VALUES, float)
        out = []
        for kb in (0.5, 1.0, 2.0):
            for ks in (0.0, 0.5, 1.0, 2.0):
                c = np.where(base > 0, base * kb, base * ks)
                out.append({"curve": tuple(np.clip(c, -10, 10)), "label": f"buy x{kb}, sell x{ks}"})
        return out
    return [{"in_thr": i, "out_thr": o, "label": f"in {i}, out {o}"}
            for i in (-0.5, -1.0, -1.5) for o in (0.80, 0.85, 0.90, 9.0)]   # 9.0 = never sell


def walk_forward(variants, first_test="2018-01-01", test_years=2, step_months=3, is_start="2015-01-01", fee=0.0, lag=0):
    """Anchored walk-forward. For each test window the candidate with the best in-sample
    final value (is_start .. day before the test) is chosen and then run on the test window."""
    rows = []
    for name, v in variants.items():
        idx = v["price"].index
        price = v["price"].values
        cands = candidates(v["kind"])
        rates = [variant_rate(v, c) for c in cands]
        default_rate = variant_rate(v)
        t = pd.Timestamp(first_test)
        wid = 0
        while True:
            t_end = t + pd.DateOffset(years=test_years)
            if t_end > idx[-1] + pd.Timedelta(days=1):
                break
            a = idx.searchsorted(t)
            b = min(idx.searchsorted(t_end) - 1, len(idx) - 1)
            s = idx.searchsorted(pd.Timestamp(is_start))
            ins = [simulate(price[s:a], r[s:a], fee, lag)[0][-1] for r in rates]
            best = int(np.argmax(ins))
            p = price[a:b + 1]
            lump, dca = lump_sum(p, fee), plain_dca(p, fee)
            for label, r in (("walk_forward_selected", rates[best]), ("fixed_default", default_rate)):
                eq, tr = simulate(p, r[a:b + 1], fee, lag)
                row = _row(name, label, wid, idx[a], b - a, eq, lump, dca, tr)
                row["chosen"] = cands[best]["label"]
                rows.append(row)
            wid += 1
            t = t + pd.DateOffset(months=step_months)
    return pd.DataFrame(rows)


# ====================================================================== Monte Carlo
def stationary_bootstrap(logret, n, mean_block, rng):
    """Politis-Romano stationary bootstrap (circular, geometric block lengths)."""
    r = np.asarray(logret, float)
    out = np.empty(n)
    i = 0
    while i < n:
        start = int(rng.integers(len(r)))
        length = int(rng.geometric(1.0 / mean_block))
        take = np.take(r, np.arange(start, start + length), mode="wrap")[: n - i]
        out[i:i + len(take)] = take
        i += len(take)
    return out


def cycle_segments(close):
    """Daily log returns of each halving cycle (complete ones plus the current partial one)."""
    lr = np.log(close).diff()
    segs = []
    for h0, h1 in zip(HALVINGS[:-1], HALVINGS[1:]):
        s = lr[(lr.index >= h0) & (lr.index < h1)].dropna()
        if len(s) > 200:
            segs.append(s.values)
    return segs


def cycle_bootstrap(close, future_index, mean_block, rng, jitter=30):
    """Phase-aligned bootstrap. Each block is copied from a random historical cycle at the
    same position after its halving (± jitter days), so bull/bear phases stay in place."""
    segs = cycle_segments(close)
    complete_len = np.mean([len(s) for s in segs[:-1]])
    n = len(future_index)
    out = np.empty(n)
    halv = HALVINGS.values
    i = 0
    while i < n:
        d = future_index[i]
        h = halv[halv <= d.to_datetime64()].max()
        pos = (d - pd.Timestamp(h)).days
        nxt = halv[halv > d.to_datetime64()]
        cyc_len = (pd.Timestamp(nxt.min()) - pd.Timestamp(h)).days if len(nxt) else complete_len
        pool = [s for s in segs if len(s) > pos * len(s) / cyc_len + 1]
        seg = pool[int(rng.integers(len(pool)))]
        p = int(np.clip(pos * len(seg) / cyc_len + rng.integers(-jitter, jitter + 1), 0, len(seg) - 1))
        length = int(min(rng.geometric(1.0 / mean_block), len(seg) - p, n - i))
        out[i:i + length] = seg[p:p + length]
        i += length
    return out


def forward_atc(ext_close, n_real, base_bands, refit_every=30, min_history=1460, fit_stride=7):
    """Causal ATC bands for days n_real.. of an extended (real + synthetic) series. Fits at or
    before the last real day are reused from ``base_bands`` (the bands depend only on the date
    and the coefficients), and later fits use only data up to their own fit day."""
    idx = ext_close.index
    x = cm._atc_x(idx)
    y = np.log10(ext_close.values)
    coefs_log = base_bands.attrs.get("coefficients") or {}
    out = np.full((len(idx), len(cm.ATC_TAUS)), np.nan)
    out[:n_real] = base_bands.values[:n_real]
    for T in range(min_history - 1, len(idx), refit_every):
        end = min(len(idx), T + refit_every)
        if end <= n_real:
            continue
        if T < n_real:
            coefs = coefs_log[base_bands.index[T]]
        else:
            sel = np.arange(T, -1, -fit_stride)[::-1]
            coefs = cm.fit_tied_quantiles(x[sel], y[sel])
        lo = max(T, n_real)
        out[lo:end] = cm._bands_from_coefs(x[lo:end], coefs)
    return pd.DataFrame(out, index=idx, columns=base_bands.columns)


def fit_tied_onchain(real_frame, start="2015-01-01"):
    """Fit the non-price part of the default composite (mean of the CBPL and on-chain z) as
    a + b * price_z + AR(1) noise. This is an in-sample relationship, so MC results for the
    'tied' variant are only as good as this approximation."""
    f = real_frame.loc[start:, ["z_price", "z_cbpl", "z_onchain"]].dropna()
    nz = (f["z_cbpl"] + f["z_onchain"]) / 2
    b, a = np.polyfit(f["z_price"], nz, 1)
    u = nz - (a + b * f["z_price"])
    phi = float(np.corrcoef(u[:-1], u[1:])[0, 1])
    sigma = float(np.std(u.values[1:] - phi * u.values[:-1]))
    return {"a": float(a), "b": float(b), "phi": phi, "sigma": sigma, "u0": float(u.iloc[-1]),
            "r2": float(np.corrcoef(f["z_price"], nz)[0, 1] ** 2)}


_MC = {}


def _mc_init(state):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _MC.update(state)


def mc_path(args):
    regime, seed, mean_block = args
    st = _MC
    rng = np.random.default_rng(seed)
    close, H = st["close"], st["horizon"]
    fut_idx = pd.date_range(close.index[-1] + pd.Timedelta(days=1), periods=H)
    if regime == "cycles":
        lr = cycle_bootstrap(close, fut_idx, mean_block, rng)
    else:
        lr = stationary_bootstrap(st["pool"], H, mean_block, rng)
        if regime == "half_drift":
            lr = lr - st["pool_mean"] / 2
    fut = pd.Series(close.iloc[-1] * np.exp(np.cumsum(lr)), index=fut_idx)
    ext = pd.concat([close, fut])
    n0 = len(close)
    bands = forward_atc(ext, n0, st["bands"])
    pz = cm.atc_z(ext.values, bands).values
    pub = cm.atc_bands(ext, mode="replica")
    pz_pub = cm.atc_z(ext.values, pub).values
    tp = st["tied"]
    e = rng.normal(0, tp["sigma"], H)
    u = np.empty(H)
    prev = tp["u0"]
    for i in range(H):
        prev = tp["phi"] * prev + e[i]
        u[i] = prev
    nz = np.clip(tp["a"] + tp["b"] * pz[n0:] + u, -3, 3)
    sd = sdca.compute_sdca(ext)
    p = fut.values
    rates = {
        "price_only_causal": composite_rate(cm.risk_from_z(pz[n0:])),
        "tied_composite_causal": composite_rate(cm.risk_from_z(np.clip((pz[n0:] + 2 * nz) / 3, -3, 3))),
        "price_only_published_coefs": composite_rate(cm.risk_from_z(pz_pub[n0:])),
        "sdca_200w": sdca_rate(sd["Z_Diff"].values[n0:], sd["Cycle_Pct"].values[n0:]),
    }
    res = {"regime": regime, "seed": seed, "block": mean_block, "path_ret": float(p[-1] / close.iloc[-1] - 1)}
    curves = {"price": p[::7] / close.iloc[-1]}
    for sname, sset in (("base", (0.0, 0)), ("next_day+fee_0.25%", (0.0025, 1))):
        fee, lag = sset
        lump, dca = lump_sum(p, fee), plain_dca(p, fee)
        for v, r in rates.items():
            eq, tr = simulate(p, r, fee, lag)
            res[(v, sname)] = _row(v, sname, seed, fut_idx[0], H, eq, lump, dca, tr)
            if sname == "base":
                curves[v] = eq[::7] / CAPITAL
        if sname == "base":
            curves["lump_sum"] = lump[::7] / CAPITAL
            curves["dca"] = dca[::7] / CAPITAL
    res["curves"] = curves
    return res


def monte_carlo(close, base_bands, tied, paths=(600, 300, 300), horizon_years=4, jobs=7, seed=7,
                pool_start="2015-01-01", progress=True):
    from multiprocessing import Pool
    lr = np.log(close).diff().dropna()
    pool = lr[pool_start:].values
    state = {"close": close, "horizon": int(horizon_years * 365.25), "bands": base_bands, "tied": tied,
             "pool": pool, "pool_mean": float(pool.mean())}
    rng = np.random.default_rng(seed)
    tasks = []
    for regime, n in zip(("bootstrap", "half_drift", "cycles"), paths):
        for _ in range(n):
            tasks.append((regime, int(rng.integers(1 << 31)), int(rng.choice([30, 60, 90]))))
    results = []
    t0 = time.time()
    if jobs <= 1:
        _mc_init(state)
        it = map(mc_path, tasks)
    else:
        pl = Pool(jobs, initializer=_mc_init, initargs=(state,))
        it = pl.imap_unordered(mc_path, tasks, chunksize=2)
    for k, r in enumerate(it, 1):
        results.append(r)
        if progress and (k % 25 == 0 or k == len(tasks)):
            print(f"  MC {k}/{len(tasks)} paths, {time.time() - t0:.0f}s", flush=True)
    if jobs > 1:
        pl.close()
        pl.join()
    rows = []
    for r in results:
        for k, v in r.items():
            if isinstance(k, tuple):
                rows.append({**v, "regime": r["regime"], "block": r["block"], "path_ret": r["path_ret"]})
    return pd.DataFrame(rows), results, state


# ====================================================================== statistics
def summarize(df, by=("variant", "setting")):
    def q(s, p):
        return float(np.percentile(s, p))
    out = []
    for keys, g in df.groupby(list(by), sort=False):
        out.append({**dict(zip(by, keys if isinstance(keys, tuple) else (keys,))), "n": len(g),
                    "p_profit": float((g["ret"] > 0).mean()),
                    "p_beat_lump": float((g["excess_lump"] > 0).mean()),
                    "p_beat_dca": float((g["excess_dca"] > 0).mean()),
                    "ret_p5": q(g["ret"], 5), "ret_p50": q(g["ret"], 50), "ret_p95": q(g["ret"], 95),
                    "exl_p5": q(g["excess_lump"], 5), "exl_p50": q(g["excess_lump"], 50), "exl_p95": q(g["excess_lump"], 95),
                    "exd_p5": q(g["excess_dca"], 5), "exd_p50": q(g["excess_dca"], 50), "exd_p95": q(g["excess_dca"], 95),
                    "mdd_p50": q(g["mdd"], 50), "mdd_p95": q(g["mdd"], 95),
                    "mdd_lump_p50": q(g["mdd_lump"], 50), "mdd_dca_p50": q(g["mdd_dca"], 50),
                    "lump_ret_p50": q(g["lump_ret"], 50), "dca_ret_p50": q(g["dca_ret"], 50),
                    "p_lump_profit": float((g["lump_ret"] > 0).mean()), "p_dca_profit": float((g["dca_ret"] > 0).mean()),
                    "final_p5": q(g["final"], 5), "final_p50": q(g["final"], 50), "final_p95": q(g["final"], 95)})
    return pd.DataFrame(out)


def md_table(s, cols=("n", "p_profit", "p_beat_lump", "p_beat_dca", "ret_p5", "ret_p50", "ret_p95",
                      "exl_p5", "exl_p50", "exl_p95", "exd_p5", "exd_p50", "exd_p95", "mdd_p50", "mdd_p95")):
    keys = [c for c in s.columns if c not in cols and not pd.api.types.is_numeric_dtype(s[c])]
    head = keys + list(cols)
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in s.iterrows():
        cells = []
        for c in head:
            v = r[c]
            if c == "n":
                cells.append(str(int(v)))
            elif c.startswith("p_"):
                cells.append(f"{v:.0%}")
            elif c.startswith("mdd"):
                cells.append(f"{v:.0f}%")
            elif isinstance(v, float):
                cells.append(f"{v * 100:+.0f}%")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ====================================================================== charts
def charts(win, wf, mc_df, mc_results, out=OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    base = win[win.setting == "base"]
    variants = list(base.variant.unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col, title in zip(axes, ("ret", "excess_lump", "excess_dca"),
                              ("Return (x100 %)", "Excess vs lump sum", "Excess vs plain DCA")):
        data = [np.log1p(base[base.variant == v][col].clip(lower=-0.99)) for v in variants]
        ax.boxplot(data, tick_labels=[v.replace("_", "\n") for v in variants], showfliers=False)
        ax.axhline(0, color="red", lw=0.8)
        ax.set_title(f"Random windows: {title} (log1p scale)")
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    paths["windows_boxplots"] = out / "windows_boxplots.png"
    fig.savefig(paths["windows_boxplots"], dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(len(variants), 2, figsize=(14, 3 * len(variants)))
    for i, v in enumerate(variants):
        g = base[base.variant == v]
        for j, (col, lab) in enumerate((("excess_lump", "vs lump sum"), ("excess_dca", "vs DCA"))):
            ax = axes[i, j]
            ax.hist(np.log1p(g[col].clip(lower=-0.99)), bins=40, color="#4ea6c9")
            ax.axvline(0, color="red")
            ax.set_title(f"{v}: log(1+excess) {lab}  (beat {np.mean(g[col] > 0):.0%})", fontsize=9)
    fig.tight_layout()
    paths["windows_excess_hist"] = out / "windows_excess_hist.png"
    fig.savefig(paths["windows_excess_hist"], dpi=120)
    plt.close(fig)

    s = summarize(win)
    fig, ax = plt.subplots(figsize=(14, 5))
    settings = list(win.setting.unique())
    w = 0.8 / len(variants)
    for i, v in enumerate(variants):
        vals = [s[(s.variant == v) & (s.setting == st)]["p_beat_lump"].iloc[0] for st in settings]
        ax.bar(np.arange(len(settings)) + i * w, vals, w, label=v)
    ax.set_xticks(np.arange(len(settings)) + 0.4 - w / 2, settings, fontsize=8)
    ax.axhline(0.5, color="grey", ls="--")
    ax.set_ylabel("share of windows beating lump sum")
    ax.legend(fontsize=8)
    ax.set_title("Sensitivity: share of random windows beating lump sum")
    fig.tight_layout()
    paths["sensitivity_beat_lump"] = out / "sensitivity_beat_lump.png"
    fig.savefig(paths["sensitivity_beat_lump"], dpi=120)
    plt.close(fig)

    if mc_df is not None and len(mc_df):
        regimes = list(mc_df.regime.unique())
        mcv = ["price_only_causal", "tied_composite_causal", "price_only_published_coefs", "sdca_200w", "lump_sum", "dca"]
        fig, axes = plt.subplots(len(regimes), 1, figsize=(15, 5 * len(regimes)))
        for ax, rg in zip(np.atleast_1d(axes), regimes):
            rs = [r for r in mc_results if r["regime"] == rg]
            for v, color in zip(mcv, ("#22c55e", "#d4a756", "#be0a12", "#8b44c4", "grey", "black")):
                arr = np.array([r["curves"][v] for r in rs])
                x = np.arange(arr.shape[1]) * 7 / 365.25
                lo, med, hi = np.percentile(arr, [5, 50, 95], axis=0)
                ax.plot(x, med, color=color, lw=1.6, label=f"{v} median")
                if v in ("price_only_causal", "lump_sum", "tied_composite_causal"):
                    ax.fill_between(x, lo, hi, color=color, alpha=0.12)
            ax.set_yscale("log")
            ax.axhline(1, color="red", lw=0.7)
            ax.set_title(f"Monte Carlo fan chart, regime {rg} ({len(rs)} paths): portfolio / starting capital, 5-95% bands")
            ax.set_xlabel("years from last real close")
            ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()
        paths["mc_fan_charts"] = out / "mc_fan_charts.png"
        fig.savefig(paths["mc_fan_charts"], dpi=120)
        plt.close(fig)

        b = mc_df[mc_df.setting == "base"]
        fig, axes = plt.subplots(len(regimes), 2, figsize=(15, 4 * len(regimes)))
        for i, rg in enumerate(regimes):
            g = b[b.regime == rg]
            ax = np.atleast_2d(axes)[i, 0]
            for v in g.variant.unique():
                ax.hist(np.log10(g[g.variant == v]["final"]), bins=50, histtype="step", label=v)
            ax.hist(np.log10(CAPITAL * (1 + g[g.variant == "price_only_causal"]["lump_ret"])), bins=50,
                    histtype="step", color="black", ls="--", label="lump sum")
            ax.axvline(np.log10(CAPITAL), color="red")
            ax.set_title(f"{rg}: final value distribution, log10 USD from $10k")
            ax.legend(fontsize=7)
            ax = np.atleast_2d(axes)[i, 1]
            for v in g.variant.unique():
                ax.hist(np.log1p(g[g.variant == v]["excess_lump"].clip(lower=-0.99)), bins=50, histtype="step", label=v)
            ax.axvline(0, color="red")
            ax.set_title(f"{rg}: log(1+excess vs lump sum)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        paths["mc_distributions"] = out / "mc_distributions.png"
        fig.savefig(paths["mc_distributions"], dpi=120)
        plt.close(fig)

    if wf is not None and len(wf):
        fig, ax = plt.subplots(figsize=(14, 5))
        for v in wf.variant.unique():
            g = wf[(wf.variant == v) & (wf.setting == "walk_forward_selected")]
            ax.plot(pd.to_datetime(g.start), g.excess_lump * 100, marker="o", ms=3, label=v)
        ax.axhline(0, color="red")
        ax.set_ylabel("excess vs lump sum, % (2-year test window)")
        ax.set_title("Anchored walk-forward: settings chosen in-sample, tested out-of-sample")
        ax.legend(fontsize=8)
        fig.tight_layout()
        paths["walk_forward"] = out / "walk_forward.png"
        fig.savefig(paths["walk_forward"], dpi=120)
        plt.close(fig)
    return {k: str(v) for k, v in paths.items()}


# ====================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", type=int, default=500)
    ap.add_argument("--paths", default="600,300,300", help="MC paths for bootstrap,half_drift,cycles")
    ap.add_argument("--horizon-years", type=float, default=4)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--skip-mc", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    close, variants = real_variants()
    print(f"data {close.index[0].date()}..{close.index[-1].date()}, variants built in {time.time() - t0:.0f}s", flush=True)
    common = variants["causal_coinmetrics"]["price"].index
    wins = random_windows(common, a.windows)
    win = window_study(variants, wins)
    wf = walk_forward(variants)
    wf_fee = walk_forward(variants, fee=0.0025, lag=1)
    wf_fee["setting"] = wf_fee["setting"] + " (next_day+fee_0.25%)"
    wf = pd.concat([wf, wf_fee], ignore_index=True)
    print(f"windows + walk-forward done in {time.time() - t0:.0f}s", flush=True)
    win.to_csv(OUT / "windows.csv", index=False)
    wf.to_csv(OUT / "walk_forward.csv", index=False)

    mc_df, mc_res, tied = None, [], None
    if not a.skip_mc:
        m = cm.build(close, mode="causal", site=cm.load_site_data(), cm=cm.load_coinmetrics())
        base_bands = cm.atc_bands(close, mode="causal")
        tied = fit_tied_onchain(m.frame)
        paths = tuple(int(x) for x in a.paths.split(","))
        mc_df, mc_res, _ = monte_carlo(close, base_bands, tied, paths, a.horizon_years, a.jobs)
        mc_df.to_csv(OUT / "monte_carlo.csv", index=False)

    s_win = summarize(win)
    s_wf = summarize(wf)
    s_mc = summarize(mc_df, by=("regime", "variant", "setting")) if mc_df is not None else None
    chart_paths = charts(win, wf, mc_df, mc_res)
    result = {"generated": pd.Timestamp.now().isoformat(), "data_end": str(close.index[-1].date()),
              "n_windows": a.windows, "windows": s_win.to_dict("records"), "walk_forward": s_wf.to_dict("records"),
              "walk_forward_choices": wf[wf.setting == "walk_forward_selected"].groupby("variant")["chosen"].value_counts().to_dict(),
              "monte_carlo": s_mc.to_dict("records") if s_mc is not None else None, "tied_onchain_fit": tied,
              "charts": chart_paths, "runtime_s": time.time() - t0}
    result["walk_forward_choices"] = {f"{k[0]}: {k[1]}": int(v) for k, v in result["walk_forward_choices"].items()}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, default=float))
    md = ["# Robustness results", "", f"Data to {result['data_end']}. Capital $10,000. Returns and excess are fractions (+100% = doubled).", "",
          f"## 1. Random windows ({a.windows} windows, 2-6 years, starting 2015 or later; same windows for every variant)", "",
          md_table(s_win), "", "## Walk-forward (2-year test windows every quarter from 2018; settings chosen in-sample)", "",
          md_table(s_wf), ""]
    if s_mc is not None:
        md += ["## 2. Monte Carlo (forward from the last real close, 4 years)", "",
               md_table(s_mc, cols=("n", "p_profit", "p_beat_lump", "p_beat_dca", "ret_p5", "ret_p50", "ret_p95",
                                    "exl_p5", "exl_p50", "exl_p95", "exd_p5", "exd_p50", "exd_p95", "mdd_p50", "mdd_p95")), "",
               f"Tied on-chain model fit (in-sample, 2015+): {json.dumps(tied)}", ""]
    (OUT / "summary.md").write_text("\n".join(md))
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")
    return result


if __name__ == "__main__":
    main()
