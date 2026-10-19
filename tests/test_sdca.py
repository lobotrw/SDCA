import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sdca  # noqa: E402


def synthetic_yahoo(start="2014-09-17", end="2026-10-19", seed=0):
    """Offline stand-in for Yahoo BTC-USD: a random walk starting near the real 2014 price."""
    idx = pd.date_range(start, end, freq="D")
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0015, 0.04, len(idx))
    return pd.Series(457.33 * np.exp(np.cumsum(steps)), index=idx, name="Close")


@pytest.fixture(scope="module")
def spliced():
    return sdca.splice_history(sdca.load_pre2014_history(), synthetic_yahoo())


def test_bundled_history_is_daily_and_ends_before_yahoo():
    old = sdca.load_pre2014_history()
    assert old.index.min() == pd.Timestamp("2010-07-18")
    assert old.index.max() == pd.Timestamp("2014-09-16")
    assert (old.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert (old > 0).all()


def test_splice_newer_wins_on_overlap():
    older = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3))
    newer = pd.Series([20.0, 30.0, 40.0], index=pd.date_range("2020-01-02", periods=3))
    out = sdca.splice_history(older, newer)
    assert out.tolist() == [1.0, 20.0, 30.0, 40.0]


def test_sma_starts_2014_with_extended_history(spliced):
    df = sdca.compute_sdca(spliced)
    first_sma = df["SMA_BTC"].first_valid_index()
    assert first_sma == pd.Timestamp("2014-05-17")      # 2010-07-18 + 1399 days
    assert first_sma < pd.Timestamp("2015-01-01")
    # every cycle top/bottom since 2015 is covered by a Z-score
    assert df.loc["2015-12-31":, "Z_Diff"].notna().all()


def test_no_lookahead(spliced):
    full = sdca.compute_sdca(spliced)
    for cut in ["2017-12-17", "2021-04-14", "2024-03-14"]:
        part = sdca.compute_sdca(spliced[:cut])
        pd.testing.assert_series_equal(part["Z_Diff"], full.loc[:cut, "Z_Diff"], check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(part["SMA_BTC"], full.loc[:cut, "SMA_BTC"], check_exact=False, rtol=1e-12)


def test_deviation_is_relative_not_dollar():
    base = synthetic_yahoo("2000-01-01", "2006-06-30", seed=1)
    a = sdca.compute_sdca(base)
    b = sdca.compute_sdca(base * 1000)          # same shape, 1000x the dollar price
    pd.testing.assert_series_equal(a["LogDev"], b["LogDev"], check_exact=False, atol=1e-9)
    pd.testing.assert_series_equal(a["Z_Diff"], b["Z_Diff"], check_exact=False, atol=1e-7)


@pytest.mark.parametrize("last_two, expected_sign", [((-0.20, -0.10), 1), ((0.50, 0.30), -1)])
def test_delta_sign_is_today_minus_yesterday(last_two, expected_sign):
    df = pd.DataFrame({"Z_Diff": [np.nan, 0.0, *last_two]})
    z, delta = sdca.z_change(df)
    assert z == last_two[1]
    assert delta == pytest.approx(last_two[1] - last_two[0])
    assert np.sign(delta) == expected_sign


def test_signal_thresholds():
    df = pd.DataFrame({"Z_Diff": [-2.0, -0.5, 0.0, 1.6, 3.0]})
    buy, sell = sdca.signals(df, -1.0, 1.5)
    assert buy.tolist() == [True, False, False, False, False]
    assert sell.tolist() == [False, False, False, True, True]


# ---- cycle-relative sell mode -------------------------------------------------------

def test_cycle_percentile_no_lookahead(spliced):
    full = sdca.compute_sdca(spliced)
    for cut in ["2018-01-01", "2021-06-30", "2025-01-15"]:
        part = sdca.compute_sdca(spliced[:cut])
        pd.testing.assert_series_equal(part["Cycle_Pct"], full.loc[:cut, "Cycle_Pct"])


def test_cycle_percentile_ignores_the_future():
    # changing values AFTER day t must not change the percentile on day t
    x = pd.Series(np.random.default_rng(3).normal(size=3000), index=pd.date_range("2010-01-01", periods=3000))
    y = x.copy(); y.iloc[2500:] += 100.0
    a = sdca.cycle_percentile(x); b = sdca.cycle_percentile(y)
    pd.testing.assert_series_equal(a.iloc[:2500], b.iloc[:2500])


def test_cycle_percentile_definition():
    x = pd.Series(np.arange(1, 11, dtype=float))
    p = sdca.cycle_percentile(x, window=5, min_periods=3)
    assert p.iloc[:2].isna().all()
    assert p.iloc[-1] == 1.0                                  # new high: above all 4 previous days
    x2 = pd.Series([5, 4, 3, 2, 1.0])
    assert sdca.cycle_percentile(x2, window=5, min_periods=3).iloc[-1] == 0.0   # new low
    x3 = pd.Series([1, 3, 2, 4, 2.5])
    assert sdca.cycle_percentile(x3, window=5, min_periods=3).iloc[-1] == 0.5   # above 1 and 2


def test_cycle_percentile_window_and_range(spliced):
    df = sdca.compute_sdca(spliced)
    p = df["Cycle_Pct"].dropna()
    assert p.between(0, 1).all()
    first_logdev = df["LogDev"].first_valid_index()
    # needs CYCLE_MIN_PERIODS valid LogDev values first
    assert p.index[0] == df["LogDev"].dropna().index[sdca.CYCLE_MIN_PERIODS - 1]
    assert p.index[0] > first_logdev


def test_signals_sell_modes():
    df = pd.DataFrame({"Z_Diff": [-2.0, 0.0, 2.0], "Cycle_Pct": [0.1, 0.9, 0.5]})
    buy_z, sell_z = sdca.signals(df, -1.0, 1.5, sell_mode="zscore")
    buy_c, sell_c = sdca.signals(df, -1.0, 0.85, sell_mode="cycle")
    assert buy_z.tolist() == buy_c.tolist() == [True, False, False]
    assert sell_z.tolist() == [False, False, True]
    assert sell_c.tolist() == [False, True, False]
    # default OUT threshold follows the mode (1.5 for Z, 0.85 for the percentile)
    assert sdca.signals(df, sell_mode="cycle")[1].tolist() == [False, True, False]
    assert sdca.signals(df, sell_mode="zscore")[1].tolist() == [False, False, True]
    with pytest.raises(ValueError):
        sdca.signals(df, sell_mode="nope")
