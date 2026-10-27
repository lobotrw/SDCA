import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import composite as cm  # noqa: E402


def synthetic_close(start="2010-07-18", end="2026-10-26", seed=1):
    """Power-law trend plus noise: stands in for BTC history offline."""
    idx = pd.date_range(start, end, freq="D")
    t = (idx - cm.ATC_GENESIS).days.values.astype(float)
    rng = np.random.default_rng(seed)
    noise = np.cumsum(rng.normal(0, 0.03, len(idx)))
    noise -= pd.Series(noise).rolling(365, min_periods=1).mean().values
    return pd.Series(10 ** (-17 + 5.8 * np.log10(t)) * np.exp(noise), index=idx)


def fake_site(index, seed=2):
    rng = np.random.default_rng(seed)
    start = int(index[0].value // 10**6)
    vals = lambda: list(np.round(np.clip(np.cumsum(rng.normal(0, 0.05, len(index))), -3, 3), 4))  # noqa: E731
    return {"SHARPE_Z": {"start": start, "values": vals()}, "CBPL_Z": {"start": start, "values": vals()},
            "ONCHAIN_Z": {"start": start, "values": vals()}, "BTC": []}


def test_published_coefficients_match_app_numbers():
    # Values printed by the app's own JavaScript for 2026-10-26 (run in Node).
    b = cm.atc_bands(pd.Series([79952.53], index=[pd.Timestamp("2026-10-26")]), mode="replica")
    assert b["q50"].iloc[0] == pytest.approx(126689.574476, rel=1e-9)
    assert b["q01"].iloc[0] == pytest.approx(71884.140827, rel=1e-9)
    assert b["q99"].iloc[0] == pytest.approx(212667.588453, rel=1e-9)
    z = cm.atc_z([79952.53], b)
    assert z.iloc[0] == pytest.approx(2.4368428, abs=1e-6)
    assert cm.eqm_risk([79952.53], b).iloc[0] == pytest.approx(6.5086454, abs=1e-6)


def test_atc_z_anchor_points():
    b = pd.DataFrame({"q01": [100.0], "q50": [1000.0], "q99": [10000.0]})
    assert cm.atc_z([100.0], b).iloc[0] == pytest.approx(3)
    assert cm.atc_z([1000.0], b).iloc[0] == pytest.approx(0)
    assert cm.atc_z([10000.0], b).iloc[0] == pytest.approx(-3)
    assert cm.atc_z([1e6], b).iloc[0] == pytest.approx(-3)        # clipped
    assert cm.atc_z([np.sqrt(1e5)], b).iloc[0] == pytest.approx(1.5)
    assert cm.atc_z([np.sqrt(1e7)], b).iloc[0] == pytest.approx(-1.5)


def test_risk_map_and_bands():
    assert cm.risk_from_z(3) == 0 and cm.risk_from_z(-3) == 100 and cm.risk_from_z(0) == 50
    assert cm.risk_from_z(0.7117809339914672) == pytest.approx(38.136984, abs=1e-5)
    assert [cm.composite_band(r) for r in (0, 24.9, 25, 49.9, 50, 75, 100)] == \
        ["Buy", "Buy", "Accumulate", "Accumulate", "Trim", "Sell", "Sell"]


def test_curve_default_and_interpolation():
    assert len(cm.CURVE_DEFAULT_VALUES) == len(cm.CURVE_RISK_NODES) == 21
    assert cm.curve_rate(0) == 10 and cm.curve_rate(100) == -10
    assert cm.curve_rate(17.5) == pytest.approx(7.5)
    assert cm.curve_rate(87.5) == pytest.approx(-1.25)
    assert cm.curve_rate(50) == 0
    assert cm.curve_rate(np.nan) == 0
    assert cm.parse_curve(",".join(["1"] * 21)) == (1.0,) * 21
    with pytest.raises(ValueError):
        cm.parse_curve("1,2,3")


def test_composite_requires_all_enabled():
    ind = pd.DataFrame({"price": [1.0, 2.0, np.nan], "cbpl": [3.0, np.nan, 1.0], "onchain": [-1.0, 0.0, 0.0]})
    z = cm.composite(ind, ("price", "cbpl", "onchain"))
    assert z.iloc[0] == pytest.approx(1.0) and np.isnan(z.iloc[1]) and np.isnan(z.iloc[2])
    z2 = cm.composite(ind, ("price", "onchain"), weights={"price": 3})
    assert z2.iloc[0] == pytest.approx((3 * 1 - 1) / 4)


def test_tied_quantile_fit_recovers_known_curve():
    rng = np.random.default_rng(0)
    x = np.linspace(-1.5, 0.8, 900)
    y = 3 + 2.5 * x - 0.2 * x * x + rng.normal(0, 0.1, x.size)
    c = cm.fit_tied_quantiles(x, y)
    assert c[0.5][0] == pytest.approx(3, abs=0.03)
    assert c[0.5][1] == pytest.approx(2.5, abs=0.05)
    assert c[0.5][2] == pytest.approx(-0.2, abs=0.05)
    assert c[0.01][0] < c[0.10][0] < c[0.25][0] < c[0.5][0] < c[0.75][0] < c[0.95][0] < c[0.99][0]
    assert c[0.01][2] == c[0.10][2] == c[0.25][2]      # tied curvature


def test_expanding_transforms_are_causal():
    v = np.random.default_rng(3).normal(size=800)
    full_r, full_z = cm.expanding_pct_rank(v, 50), cm.expanding_robust_z(v, 50)
    part_r, part_z = cm.expanding_pct_rank(v[:500], 50), cm.expanding_robust_z(v[:500], 50)
    np.testing.assert_allclose(full_r[:500], part_r)
    np.testing.assert_allclose(full_z[:500], part_z)
    assert np.isnan(full_r[:49]).all() and np.isfinite(full_r[49:]).all()


def test_sharpe_causal_rank_matches_raw_rank():
    raw = np.random.default_rng(4).normal(size=1000)
    full_rank = pd.Series(raw).rank(pct=True).values
    stored = pd.Series(np.clip(3 - 6 * full_rank, -3, 3))          # what the app embeds
    causal = cm.sharpe_z(stored, mode="causal", min_periods=10).values
    expected = 3 - 6 * cm.expanding_pct_rank(raw, 10)
    np.testing.assert_allclose(causal, expected)
    assert cm.sharpe_z(stored, mode="replica").equals(stored)


@pytest.fixture(scope="module")
def models():
    close = synthetic_close()
    site = fake_site(close.index)
    out = {}
    for mode in cm.MODES:
        out[mode] = cm.build(close, mode=mode, site=site, onchain_source="site", refit_every=90)
    return close, site, out


def test_causal_model_has_no_lookahead(models):
    close, site, out = models
    cut = pd.Timestamp("2021-06-30")
    part = cm.build(close[:cut], mode="causal", site=site, onchain_source="site", refit_every=90)
    a = out["causal"].frame.loc[:cut, ["q01", "q50", "q99", "atc_z", "comp_z", "comp_risk"]]
    b = part.frame[["q01", "q50", "q99", "atc_z", "comp_z", "comp_risk"]]
    pd.testing.assert_frame_equal(a, b)


def test_replica_uses_full_sample(models):
    close, site, out = models
    assert out["replica"].frame["q50"].notna().all()
    assert out["causal"].frame["q50"].iloc[:1459].isna().all()
    assert out["causal"].frame["q50"].iloc[1459:].notna().all()


def test_backtest_mechanics():
    idx = pd.date_range("2020-01-01", periods=4)
    frame = pd.DataFrame({"close": [100.0, 50.0, 200.0, 100.0], "comp_risk": [0.0, 50.0, 100.0, np.nan],
                          "comp_z": [3, 0, -3, np.nan], "q50": 1.0, "q01": 1.0, "q99": 1.0})
    model = cm.Model("causal", frame.set_index(idx), ("price",))
    s, led = cm.backtest(model, "2020-01-01", 1000)
    # day1 buy 10% of 1000 at 100 -> 1 BTC, cash 900; day2 rate 0; day3 sell 10% of 1 BTC at 200
    assert led["btc"].iloc[0] == pytest.approx(1.0)
    assert led["cash"].iloc[2] == pytest.approx(920.0)
    assert s["net_btc"] == pytest.approx(0.9)
    assert s["portfolio_value"] == pytest.approx(920 + 0.9 * 100)
    assert s["lump_sum_value"] == pytest.approx(1000.0)
    assert s["buy_days"] == 1 and s["sell_days"] == 1
    assert s["max_drawdown_pct"] == pytest.approx((1100 - 1010) / 1100 * 100)   # peak day3 1100 -> day4 1010
    assert s["max_drawdown_lump_pct"] == pytest.approx(50.0)


def test_today_action(models):
    _, _, out = models
    t = cm.today_action(out["causal"], 1000)
    assert t["band"] == cm.composite_band(t["composite_risk"])
    rate = float(cm.curve_rate(t["composite_risk"]))
    if rate > 0:
        assert t["spend_usd"] == pytest.approx(1000 * rate / 100)


def test_missing_indicator_is_skipped():
    close = synthetic_close(end="2016-12-31")
    m = cm.build(close, mode="replica", site={"BTC": []}, cm=None, onchain_source="site")
    assert m.enabled == ("price",)
    assert any("cbpl" in n for n in m.notes)
    assert m.frame["comp_z"].notna().any()


def test_parse_site_html():
    html = ('const SHARPE_Z = {"start":1282089600000,"values":[0.5,null]};\n'
            'const CBPL_Z = {"start":1282089600000,"values":[1]};\n'
            'const EMBEDDED_BTC_DATA = [[1282089600000,0.07]];')
    d = cm.parse_site_html(html)
    s = cm.site_series(d, "SHARPE_Z")
    assert s.index[0] == pd.Timestamp("2010-08-18") and np.isnan(s.iloc[1])
    assert d["ONCHAIN_Z"]["values"] == [] and d["BTC"] == [[1282089600000, 0.07]]
