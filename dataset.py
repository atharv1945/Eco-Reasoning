import yfinance as yf

# 1. Download NASDAQ-100 (Ticker: ^NDX)
# FinLSPM uses Jan 2001 - Nov 2024 
ndx = yf.download("^NDX", start="2001-01-01", end="2024-11-30")
ndx.to_csv("ndx_data.csv")

# 2. Download Bitcoin (Ticker: BTC-USD)
# FinLSPM uses Jan 2021 - Nov 2024 due to market maturity [cite: 481, 491]
btc = yf.download("BTC-USD", start="2021-01-01", end="2024-11-30")
btc.to_csv("btc_data.csv")

print("Datasets downloaded successfully: ndx_data.csv and btc_data.csv")