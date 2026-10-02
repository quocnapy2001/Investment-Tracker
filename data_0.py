import pandas as pd
import yfinance as yf
from datetime import datetime
import os

print(os.listdir())
print(os.getcwd())

# Transaction Data
transaction_data = pd.read_excel("Transaction Data.xlsx")
print(transaction_data.head())
print(transaction_data.tail())
print(transaction_data.columns)
print("\n")

# Get tickers
all_tickers = list(transaction_data["Ticker"].unique())
print(all_tickers)
print("\n")

blackList = []
filt_tickers = [i for i in all_tickers if i not in blackList]
print(f"The number of assets traded are {len(filt_tickers)}.")
print(filt_tickers)
print("\n")

# Price data
start_date = '2021-01-01'
end_date = datetime.today().strftime("%Y-%m-%d")

raw = yf.download(
    filt_tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)

price_data = (
    raw.swaplevel(0, 1, axis=1)   # (Ticker, Field)
       .sort_index(axis=1)
       .stack(level=0)            # Ticker → rows
       .rename_axis(["Date", "Ticker"])
       .reset_index()
       .set_index(["Ticker", "Date"])
       .sort_index()
)

close_prices = price_data["Close"]
close_prices.index = close_prices.index.set_levels(
    pd.to_datetime(close_prices.index.levels[1]),
    level=1)

# Monthly Cashflow
