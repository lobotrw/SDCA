
### Crypto SDCA System

#### Overview
The Crypto SDCA System is designed for long-term investing in cryptocurrencies, employing systematic dollar-cost averaging (SDCA) principles. It utilizes historical price data and statistical metrics to automate investment decisions based on predefined buy and sell thresholds.

#### Key Features
- **Relative deviation**: Measures how far price is from the 200-week SMA as a log ratio, `ln(price / SMA)`, so deviations in 2015 and 2025 are comparable (a dollar gap made recent years dominate).
- **No look-ahead Z-score**: Normalizes the deviation with an *expanding* mean and standard deviation, so each day's Z-score uses only data available on that day and past values never change when new data arrives.
- **Gradient Color Visualization**: Displays the price chart with a gradient color background based on the Z-score of the difference, facilitating visual trend analysis.
![plot](images/image.png)

1. **200-week Moving Average**: Computes the 200-week (1,400-day) SMA to establish a baseline trend indicator.
2. **Z-score and Gradient Visualization**: Calculates `LogDev = ln(price / SMA)` and its Z-score `(LogDev - mean) / std`, where mean and std are expanding statistics over all days up to and including the current one (minimum one year of data). An expanding window was chosen over a rolling one because a 200-week cycle lasts ~4 years: a rolling window short enough to react would only hold one cycle and re-centre within it. This Z-score color-codes the price chart, highlighting periods of divergence from the long-term trend.
3. **Threshold-based Signals**: Identifies buying opportunities when the Z-score falls below the SDCA IN threshold (default **-1.0**), marked by blue diamonds (◆) below the price. It identifies selling opportunities when the Z-score rises above the SDCA OUT threshold (default **+1.5**), marked by red downward triangles (▼) above the price. Both thresholds are adjustable in the app.
![plot](images/image1.png)

#### Data
- **Yahoo Finance** `BTC-USD` daily closes (from 2014-09-17), downloaded live via `yfinance`.
- **Coin Metrics community data** (`PriceUSD`, https://github.com/coinmetrics/data, CC BY-NC 4.0) for 2010-07-18 to 2014-09-16, bundled in `data/btc_pre2014.csv`. Yahoo takes precedence wherever the two overlap; the prices agree to within 0.2% on the overlap day.
- With the extended history the 200-week SMA starts on **2014-05-17** (instead of mid-2018), and the Z-score from 2015-05-16, so the 2015, 2017, 2018/19, 2021 and 2022 cycles are all covered.

#### Default thresholds
With the log deviation and no-look-ahead Z the scale changed, so the defaults were recalibrated from -0.7 / +2.0 to **-1.0 / +1.5**. Historically this flags buys in mid/late 2015, Nov 2018 - Mar 2019, Mar - Apr 2020 and May 2022 - Oct 2023, and sells in May 2017 - Mar 2018 and Jan - Apr 2021. The 2024/25 peak sat much closer to the 200-week SMA in relative terms (about +150% at most, versus +1,500% in 2017), so it does not reach a sell signal at these defaults.

#### Sell signal method (selectable in the app)
Each cycle has peaked lower relative to the 200-week SMA: about 16x in 2017, about 6x in 2021 and about 2.5x in 2024/25. A Z-score measured against all history therefore never reached a sell level in 2024/25.
- **Cycle percentile (default):** sells when today's `ln(price / SMA)` is higher than a share (default **85%**) of the values from the last 1,460 days. That window is about one halving epoch (210,000 blocks, roughly 1,458 days). The rank is non-parametric, nothing is fitted, and it is causal. With the default it sells in 2017, early 2021 and Jul-Oct 2025 (first sell 2025-07-10, 88 days before the 2025-10-06 top), with no sells outside those periods. It works because it compares against the current cycle, so it means "the most stretched valuation in about 4 years". It is sensitive to the window: it stops firing in 2025 for windows over about 4.5 years.
- **Z-score (expanding):** the original sell rule, Z above the OUT threshold (default +1.5).

Buys always use the expanding Z-score. Three cycles are too few to validate any sell rule statistically, so treat the thresholds as judgement, not fitted truth.

#### Running
```bash
pip install -r requirements.txt
streamlit run SDCA_crypto.py
pytest tests          # unit tests for the calculation in sdca.py
```

4. **Automated Insights**: Provides a systematic approach to cryptocurrency investing, automating decisions based on statistical analysis of price trends.

#### Benefits
- **Objective Decision Making**: Removes emotional bias by relying on statistical metrics and predefined thresholds for buying and selling.
- **Long-term Focus**: Targets long-term investment opportunities by leveraging SMA and Z-score analysis.
- **Visual Clarity**: Enhances decision-making with visual representations of price trends.

#### Disclaimer
This system is for informational purposes only and should not be considered financial advice. Investors should conduct their own research or consult with a financial advisor before making investment decisions.
