import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.optimize import newton

from datetime import datetime
from pandas.tseries.offsets import BDay
import pandas_datareader.data as pdr
transaction_data = pd.read_excel("./Transaction Data.xlsx")
print(transaction_data.tail())
all_tickers = list(transaction_data['Ticker'].unique())

#Remove delisted stock
blackList = []

filt_tickers = [tick for tick in all_tickers if tick not in blackList]
print("You traded {} different stocks".format(len(all_tickers)))
filt_tickers
final_filtered = transaction_data[~transaction_data.Ticker.isin(blackList)]

###Collect the price history for all tickers
start_date = '2021-01-01'
end_date = '2026-03-01'

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
price_data
tx = transaction_data.copy()
tx["Date"] = pd.to_datetime(tx["Date"])
tx["Type"] = tx["Type"].str.upper()

# Signed shares & cash flow
tx["Signed_shares"] = tx["Shares"].where(
    tx["Type"] == "BUY", -tx["Shares"])

tx["cash_flow"] = tx["Total Cost ($)"].where(
    tx["Type"] == "SELL", -tx["Total Cost ($)"])

tx.tail(10)
close_prices = price_data["Close"]
close_prices.index = close_prices.index.set_levels(
    pd.to_datetime(close_prices.index.levels[1]),
    level=1)

close_prices