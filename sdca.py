"""Core calculations for the Crypto SDCA System.

Kept free of Streamlit so it can be unit-tested and reused.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SMA_WINDOW = 200 * 7            # 200 weeks expressed in daily bars
Z_MIN_PERIODS = 365             # need one year of deviation history before a Z-score is shown
DEFAULT_IN_THRESHOLD = -1.0     # recalibrated for the log-deviation, no-look-ahead Z (see README)
DEFAULT_OUT_THRESHOLD = 1.5

DATA_DIR = Path(__file__).resolve().parent / "data"
PRE2014_CSV = DATA_DIR / "btc_pre2014.csv"


def load_pre2014_history(path=PRE2014_CSV):
    """Bundled daily BTC/USD closes 2010-07-18 .. 2014-09-16 (Coin Metrics community data, PriceUSD)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")["Close"].astype(float).sort_index()


def splice_history(older, newer):
    """Join two daily close series; ``newer`` wins wherever the dates overlap."""
    older = older.copy()
    newer = newer.copy()
    older.index = pd.to_datetime(older.index).tz_localize(None).normalize()
    newer.index = pd.to_datetime(newer.index).tz_localize(None).normalize()
    older = older[older.index < newer.index.min()]
    out = pd.concat([older, newer]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.name = "BTC-USD"
    return out


def compute_sdca(close, sma_window=SMA_WINDOW, z_min_periods=Z_MIN_PERIODS):
    """Return a DataFrame with the price, its 200-week SMA, the log deviation and its Z-score.

    * ``LogDev`` = ln(close / SMA). A relative (log) gap, so a 2013 deviation and a
      2025 deviation are comparable even though the dollar prices differ by 1000x.
    * ``Z_Diff`` is computed WITHOUT look-ahead: each day's Z uses only the mean and
      standard deviation of ``LogDev`` from the first SMA day up to and including that
      day (an expanding window). We prefer expanding over a fixed rolling window because
      a 200-week SMA cycle is ~4 years long; a rolling window short enough to be useful
      would only ever contain one or two cycles and would re-centre itself inside each
      bull/bear phase, flattening the very extremes we want to detect. The expanding
      window uses every cycle seen so far, and past values never change when new data
      arrives. ``z_min_periods`` (one year) avoids meaningless values from a tiny sample.
    """
    close = pd.Series(close, dtype=float).sort_index()
    df = pd.DataFrame({"BTC-USD": close})
    df["SMA_BTC"] = df["BTC-USD"].rolling(window=sma_window, min_periods=sma_window).mean()
    df["LogDev"] = np.log(df["BTC-USD"] / df["SMA_BTC"])
    exp = df["LogDev"].expanding(min_periods=z_min_periods)
    df["Z_Diff"] = (df["LogDev"] - exp.mean()) / exp.std()
    return df


def signals(df, in_threshold=DEFAULT_IN_THRESHOLD, out_threshold=DEFAULT_OUT_THRESHOLD):
    """Boolean Series for DCA-IN (Z below in_threshold) and DCA-OUT (Z above out_threshold)."""
    return df["Z_Diff"] < in_threshold, df["Z_Diff"] > out_threshold


def z_change(df):
    """Latest Z value and its day-over-day change (today minus yesterday)."""
    z = df["Z_Diff"].dropna()
    return float(z.iloc[-1]), float(z.iloc[-1] - z.iloc[-2])
