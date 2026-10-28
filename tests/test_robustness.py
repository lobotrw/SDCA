import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composite as cm  # noqa: E402
import robustness as rb  # noqa: E402


def test_simulate_matches_composite_backtest():
    idx = pd.date_range("2020-01-01", periods=300)
    rng = np.random.default_rng(0)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, 300)))
    risk = np.clip(50 + np.cumsum(rng.normal(0, 4, 300)), 0, 100)
    frame = pd.DataFrame({"close": price, "comp_risk": risk, "comp_z": 0.0, "q50": 1.0, "q01": 1.0, "q99": 1.0}, index=idx)
    s, led = cm.backtest(cm.Model("causal", frame, ("price",)), "2020-01-01", 10000)
    eq, _ = rb.simulate(price, rb.composite_rate(risk))
    np.testing.assert_allclose(eq, led["portfolio"].values)


def test_fees_and_lag():
    p = np.array([100.0, 200.0, 100.0])
    eq0, _ = rb.simulate(p, [10, 0, 0])
    eqf, _ = rb.simulate(p, [10, 0, 0], fee=0.01)
    assert eqf[-1] < eq0[-1]
    assert eqf[-1] == pytest.approx(9000 + 1000 * 0.99)
    eql, _ = rb.simulate(p, [10, 0, 0], lag=1)            # buys at 200 instead of 100
    assert eql[-1] == pytest.approx(9000 + 1000 / 200 * 100)


def test_benchmarks():
    p = np.array([1.0, 2.0, 4.0, 2.0])
    np.testing.assert_allclose(rb.lump_sum(p, capital=1), [1, 2, 4, 2])
    d = rb.plain_dca(p, capital=4)                        # 1 per day
    assert d[-1] == pytest.approx(2 * (1 + 0.5 + 0.25 + 0.5))
    assert rb.max_drawdown([1, 2, 1, 3]) == pytest.approx(50)


def test_sdca_rate_and_perturb():
    r = rb.sdca_rate([-2, 0, 0, np.nan], [0.1, 0.5, 0.9, np.nan])
    np.testing.assert_allclose(r, [5, 0, -2, 0])
    rng = np.random.default_rng(1)
    for _ in range(50):
        c = rb.perturb(rng, "composite")
        assert all(abs(v) <= 10 for v in c["curve"]) and -5 <= c["shift"] <= 5
        s = rb.perturb(rng, "sdca")
        assert -1.2 <= s["in_thr"] <= -0.8 and 0.82 <= s["out_thr"] <= 0.88


def test_random_windows_bounds():
    idx = pd.date_range("2010-01-01", "2026-10-26")
    w = rb.random_windows(idx, 300, seed=3)
    assert len(w) == 300
    for a, b in w:
        assert idx[a] >= pd.Timestamp("2015-01-01") and b < len(idx)
        assert 2 * 365 <= b - a <= 6 * 365.25 + 1


def test_stationary_bootstrap_preserves_values_and_blocks():
    r = np.arange(1000, dtype=float)
    out = rb.stationary_bootstrap(r, 5000, 60, np.random.default_rng(0))
    assert len(out) == 5000 and set(out) <= set(r)
    consecutive = np.mean(np.diff(out) == 1)
    assert consecutive > 0.95                             # mostly contiguous blocks (mean 60)


def test_cycle_bootstrap_is_phase_aligned():
    idx = pd.date_range("2012-01-01", "2026-10-26")
    # returns encode the day-of-cycle so we can check alignment
    pos = np.zeros(len(idx))
    for h in rb.HALVINGS:
        pos = np.where(idx >= h, (idx - h).days, pos)
    close = pd.Series(np.exp(np.cumsum(pos * 1e-6)), index=idx)
    fut = pd.date_range("2026-10-27", periods=1461)
    out = rb.cycle_bootstrap(close, fut, 60, np.random.default_rng(0), jitter=0)
    src_pos = out / 1e-6
    fut_pos = np.array([(d - rb.HALVINGS[rb.HALVINGS <= d].max()).days for d in fut])
    # positions are rescaled to each source cycle's length, so allow a few % slack
    assert np.median(np.abs(src_pos - fut_pos) / 1440) < 0.05


def test_forward_atc_equals_full_causal_fit_on_real_data():
    idx = pd.date_range("2010-07-18", "2017-12-31")
    t = (idx - cm.ATC_GENESIS).days.values.astype(float)
    rng = np.random.default_rng(5)
    close = pd.Series(10 ** (-17 + 5.8 * np.log10(t)) * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))), index=idx)
    full = cm.atc_bands(close, mode="causal", refit_every=120)
    n_real = len(idx) - 400
    base = cm.atc_bands(close.iloc[:n_real], mode="causal", refit_every=120)
    fwd = rb.forward_atc(close, n_real, base, refit_every=120)
    np.testing.assert_allclose(fwd.values, full.values, rtol=1e-9, equal_nan=True)


def test_summarize_shares():
    df = pd.DataFrame({"variant": "v", "setting": "s", "ret": [0.1, -0.1], "excess_lump": [0.2, -0.3],
                       "excess_dca": [0.1, 0.1], "mdd": [10, 20], "mdd_lump": 30, "mdd_dca": 20,
                       "lump_ret": [0.0, 0.0], "dca_ret": 0.0, "final": [11000, 9000]})
    s = rb.summarize(df).iloc[0]
    assert s.p_profit == 0.5 and s.p_beat_lump == 0.5 and s.p_beat_dca == 1.0
    assert "| v | s |" in rb.md_table(rb.summarize(df))
