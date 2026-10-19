
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from   matplotlib.colors import LinearSegmentedColormap
import streamlit as st

import sdca


st.set_page_config("Long Term Investing Cypto", "📈", layout="wide", initial_sidebar_state="collapsed")

# Function to fetch historical data
@st.cache_data(ttl=3600, show_spinner="Downloading BTC-USD prices…")
def get_historical_data(symbol):
    data = yf.download(symbol, period="max", interval='1d', auto_adjust=True, progress=False)
    close = data['Close']
    if isinstance(close, pd.DataFrame):   # yfinance >= 0.2.51 returns MultiIndex columns
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.dropna()

# Yahoo's BTC-USD history only starts on 2014-09-17, which means a 200-week SMA would only
# exist from mid-2018. We prepend a bundled daily history (Coin Metrics community data,
# 2010-07-18 .. 2014-09-16); Yahoo takes precedence wherever the two overlap.
yahoo_close = get_historical_data("BTC-USD")
close       = sdca.splice_history(sdca.load_pre2014_history(), yahoo_close)

# 200-week SMA, relative (log) deviation ln(price / SMA) and its no-look-ahead Z-score.
# See sdca.compute_sdca for the details of each step.
price_data = sdca.compute_sdca(close)

# Create a custom colormap with three colors
colors           = [(0, 'purple'), (0.35, '#33cff6'), (1, 'red')]
n_bins           = 100  # Discretizes the interpolation into bins
three_color_cmap = LinearSegmentedColormap.from_list('three_color_gradient', colors, N=n_bins)

# Plot BTC close price and SMA with gradient color background based on Z-score
fig, ax = plt.subplots(figsize=(17, 8), dpi=500)

# Thin grey price line so the history before the first Z-score is still visible
ax.plot(price_data.index, price_data['BTC-USD'], color='lightgrey', linewidth=0.8, zorder=0)

# Scatter plot with color based on Z-score
sc = ax.scatter(price_data.index, price_data['BTC-USD'],
                 c          = price_data['Z_Diff'], 
                 cmap       = three_color_cmap, 
                 alpha      = 0.8, 
                 edgecolor  = 'none')

# Plot the SMA line
ax.plot(price_data.index, price_data['SMA_BTC'], 
        label     = '200-week SMA', 
        color     = 'purple', 
        linewidth = 2)

# Add marks below/above the price where the Z-score crosses the thresholds
[c1, c2, c3, c4] = st.columns([1, 10, 4, 1])
c3.write("#")

c3.markdown("""<hr style="height:4px;border:none;color:#333;background-color:#1d9ff0;" /> """, unsafe_allow_html=True)



in_threshold  = c3.number_input("Threshold for SDCA IN", -3.0, 0.0, sdca.DEFAULT_IN_THRESHOLD, 0.1)
out_threshold = c3.number_input("Threshold for SDCA OUT", 0.0, 4.0, sdca.DEFAULT_OUT_THRESHOLD, 0.1)
below_threshold, above_threshold = sdca.signals(price_data, in_threshold, out_threshold)
ax.scatter(price_data.index[below_threshold], price_data['BTC-USD'][below_threshold]*0.7,
            color       = '#2788e8', 
            marker      = 'D', 
            label       = 'DCA IN', 
            edgecolor   = 'none')

ax.scatter(price_data.index[above_threshold], price_data['BTC-USD'][above_threshold]*1.35,
            color       = 'red', 
            marker      = 'v', 
            label       = 'DCA OUT', 
            edgecolor   = 'none')

# Add colorbar
cbar = plt.colorbar(sc, ax=ax)


cbar.set_label('Z-score of ln(price / 200-week SMA)')

# Set labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD, log scale)')
ax.set_yscale('log')
ax.set_title('BTC Close Price with 200-week SMA and Z-score (log deviation) Based Gradient')
ax.legend()


with c2:
    st.title("Crypto SDCA System")
    st.markdown("""<hr style="height:4px;border:none;color:#333;background-color:#1d9ff0;" /> """, unsafe_allow_html=True)
    st.pyplot(fig, width='stretch')

with c3:
    st.subheader("How it Works?")
    st.write(f'''

The Crypto SDCA System automates cryptocurrency investment decisions using systematic dollar-cost averaging (SDCA) principles. 

It computes the **200-week Simple Moving Average (SMA)**, measures how far price is from it in **relative terms**
(the log ratio ln(price / SMA), so a 2017 deviation and a 2025 deviation are comparable),
and turns that into a **Z-score** using only the data available up to each day (no look-ahead).
Price history since 2010 (Coin Metrics, extended with Yahoo Finance from 2014-09-17).

    
:blue[◆] Buy signals are identified when the Z-score drops below **"Threshold for SDCA IN" ({in_threshold:+.2f})**. 

:red[▼] Sell signals trigger when it exceeds **"Threshold for SDCA OUT" ({out_threshold:+.2f})**, 
aiming for disciplined, data-driven investment strategies.

''')

z_val, z_change = sdca.z_change(price_data)

c3.write("---")
c3.metric(label="**Current Z score**", 
    value=round(z_val, 2), 
    delta=round(z_change, 2),   # today minus yesterday: ▲ = Z went up
    delta_color="inverse")

c2.write("#")
c2.write("""
        ### Disclaimer:

This system is for informational purposes only and should not be considered financial advice. 
Investors should conduct their own research or consult with a financial advisor before making investment decisions.""")

with st.sidebar:
    st.write(f"""
### About
             
    𝗞𝗘𝗬 𝗙𝗘𝗔𝗧𝗨𝗥𝗘𝗦:

- **Data Source**: Yahoo Finance (BTC-USD, from 2014-09-17) spliced onto bundled Coin Metrics community daily prices back to 2010-07-18.
- **Indicator Calculation**: Computes the 200-week Simple Moving Average (SMA) to gauge long-term price trends.
- **Deviation Calculation**: Measures the relative (log) distance ln(price / SMA) instead of a dollar difference.
- **Z-score Calculation**: Normalizes the deviation with an expanding (no look-ahead) mean and standard deviation, so past values never change.
- **Gradient Color Visualization**: Visualizes price trends with a gradient color background based on Z-score values.
- **Threshold-based Alerts**: Marks potential entry points (buy signals) when the Z-score falls below {sdca.DEFAULT_IN_THRESHOLD:+.2f}, and exit points (sell signals) when it rises above {sdca.DEFAULT_OUT_THRESHOLD:+.2f} (defaults, adjustable).

*This system aims to provide objective investment decisions, reduce emotional bias, and optimize cryptocurrency investments over the long term.*

---
- **GitHub:**      https://github.com/VanHes1ng
- **TradingView:** https://www.tradingview.com/u/VanHe1sing/
- **Twitter:**     https://x.com/Van_He1sing    
- **Telegram:**    https://t.me/IvanKocherzhat

## Crated by @VanHelsing
                  
             """)
