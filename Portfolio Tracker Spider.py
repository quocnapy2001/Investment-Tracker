# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 20:56:34 2026

@author: Owner
"""



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

### Monthly Buying
# Keep only Buy transactions 
buy_tx = tx[tx["Type"] == "BUY"].copy()

# Convert to monthy spending
monthly_spending = (buy_tx.set_index("Date").resample("M")["Total Cost ($)"].sum())

# Plot monthly spending
plt.figure(figsize=(12,6))
plt.plot(
    monthly_spending.index, 
    monthly_spending.values,
    marker = "o"
)
plt.title("Monthly Investment Spending")
plt.xlabel("Date")
plt.ylabel("Amount Spend ($)")
plt.grid(True)

# Add value labels
for x, y in zip(monthly_spending.index, monthly_spending.values):
    plt.annotate(
        f"${y:,.0f}",      # format number
        (x, y),            # point location
        textcoords="offset points",
        xytext=(0, 8),     # vertical offset
        ha="center",
        fontsize= 14
    )


plt.tight_layout()
plt.show()

### Net Monthly Cash Flow
# Convert to Monthy Cash Flow
monthly_net_cf = (tx.set_index("Date").resample("M")["cash_flow"].sum())

# Plot
plt.figure(figsize=(12,6))

plt.bar(
    monthly_net_cf.index,
    monthly_net_cf.values
)

plt.axhline(0, linewidth=1) # Zero line
plt.title("Net Monthly Cash Flow")
plt.xlabel("Date")
plt.ylabel("Monthly Cash Flow ($)")
plt.xticks(rotation=45)
plt.grid(True, axis="y")

# Value Label
for x, y in zip(monthly_net_cf.index, monthly_net_cf.values):
    if y != 0:
        plt.text(
            x,
            y,
            f"${y:,.0f}",
            ha = "center",
            va = "bottom" if y > 0 else "top",
            fontsize = 14
        )
        
plt.tight_layout()
plt.show()

###Portfolio Value
## Already have transaction data tx above:
## Build daily holdings per stock

# First transaction date
start_portfolio_date = tx["Date"].min()
prices = close_prices.unstack("Ticker")
prices = prices.loc[prices.index >= start_portfolio_date]

# Remove leading NaNs per ticker (IPO-safe)
prices = prices.apply(lambda s: s.loc[s.first_valid_index():])

# Forward-fill missing prices (market-closed days)
prices = prices.ffill()

# Aggregate trades by day and ticker:
daily_trades = (
    tx.groupby(["Date", "Ticker"])["Signed_shares"]
    .sum()
    .unstack("Ticker")
    .fillna(0)
)

# Reindex to daily frequency ad forward-fill holdings
daily_trades = daily_trades.reindex(
    prices.index,
    fill_value=0
)

daily_holdings = daily_trades.cumsum()

### Some issue to remember:
# NaNs are expected due to sparse trades and incomplete price histories.
# - No trade on a date ⇒ 0 shares traded (fill NaNs with 0 before cumsum)
# - Missing prices (IPOs, holidays) ⇒ forward-fill after first valid price
# - Assets only contribute to portfolio value once a position exists
# Explicit handling prevents artificial portfolio value drops.
### Porfolio Value
position_values = (
    prices
    .where(daily_holdings != 0, 0)   # only value when held
    .fillna(0)                       # no price → no valuation
    * daily_holdings
)

portfolio_value = position_values.sum(axis=1)

### Invested Capital
# Daily net cash flow
daily_cash_flow = (
    tx.groupby("Date")["cash_flow"]
    .sum()
    .reindex(portfolio_value.index, fill_value=0)
)

# Cumulative invested capital
invested_capital = (-daily_cash_flow).cumsum()

# Plot Portfolio Value + Invested Capital
plt.figure(figsize=(12, 6))

plt.plot(
    portfolio_value.index, 
    portfolio_value.values,
    label="Portfolio Value",
    linewidth=2
)

plt.plot(
    invested_capital.index,
    invested_capital.values,
    label="Invested Capital",
    linewidth=2
)

plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

### Asset Distribution
# Take the lastest date:
latest_date = position_values.index.max()
latest_positions = position_values.loc[latest_date]

# Remove 0 value:
latest_positions = latest_positions[latest_positions > 0]
# Mapping asset type
ticker_asset_type = (
    tx[["Ticker", "Asset"]]
    .drop_duplicates()
    .set_index("Ticker")["Asset"]
)

# Aggregate latest portfolio value by asset type
asset_type_values = latest_positions.groupby(
    ticker_asset_type.loc[latest_positions.index]
).sum()

# Define colors by asset type (aligned with index order)
colors = [
    "orange" if asset == "Equity" else "blue"
    for asset in asset_type_values.index
]


# Distribution by Asset Type (Pie Chart - %)
plt.pie(
    asset_type_values,
    labels=asset_type_values.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    wedgeprops={"edgecolor": "white"}
)
# Color ticker by asset
colors = [
    "orange" if ticker_asset_type[ticker] == "Equity" else "blue"
    for ticker in latest_positions.sort_values(ascending=True).index
]

# Distribution by ticker (Bar Chart - Value)
plt.figure(figsize=(12,6))
latest_positions.sort_values(ascending=True).plot(
    kind="bar",
    color=colors
)

plt.title("Porfolio Allocation by Ticker")
plt.xlabel("Ticker")
plt.ylabel("Market Value ($)")
plt.xticks(rotation=45)
plt.grid(True, axis="y")

plt.tight_layout()
plt.show()
# Aggregate portfolio position values by Asset Type over time.
asset_type_over_time = (
    position_values
    .groupby(ticker_asset_type, axis=1)
    .sum()
)

# Plot portfolio composition over time using a stacked area chart.
plt.figure(figsize=(12, 6))
plt.stackplot(
    asset_type_over_time.index,
    asset_type_over_time.T.values,
    labels=asset_type_over_time.columns
)

plt.title("Portfolio Composition Over Time by Asset Type")
plt.xlabel("Date")
plt.ylabel("Market Value ($)")
plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

#Plot Stacked % Over time

# Convert market values to portfolio weights (row-wise)
asset_type_pct = asset_type_over_time.div(
    asset_type_over_time.sum(axis=1),
    axis=0
)
asset_type_pct = asset_type_pct.fillna(0)

plt.figure(figsize=(12, 6))
plt.stackplot(
    asset_type_pct.index,
    asset_type_pct.T.values,
    labels=asset_type_pct.columns
)

plt.title("Portfolio Asset Allocation Over Time (%)")
plt.xlabel("Date")
plt.ylabel("Portfolio Weight")
plt.legend(loc="upper left")
plt.grid(True)

# Format y-axis as percentage
plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.0%}")
)

plt.tight_layout()
plt.show()

###Profit and Loss
# Total P&L
total_pnl = portfolio_value - invested_capital

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    total_pnl.index,
    total_pnl.values,
    label="Total Profit and Loss over time"
)

plt.title("Total Portfolio P&L Over Time")
plt.xlabel("Date")
plt.ylabel("P&L ($)")
plt.legend()
plt.grid(True)

# Format y-axis as dollars (no scientific notation)
plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"${x:,.0f}")
)

plt.tight_layout()
plt.show()

### Return over time
# Return Calculation
portfolio_return = total_pnl / invested_capital
portfolio_return = portfolio_return.replace([float("inf"), -float("inf")], 0)
portfolio_return = portfolio_return.fillna(0)

# Plot
plt.figure(figsize=(12, 6))

plt.plot(
    portfolio_return.index,
    portfolio_return.values,
    linewidth=2,
    label="Portfolio Return"
)

plt.title("Portfolio Return Over Time")
plt.xlabel("Date")
plt.ylabel("Return")
plt.legend()
plt.grid(True)

# Format as percentage
plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f"{y:.0%}")
)

plt.tight_layout()
plt.show()
# Money-weighted Return
def xirr(cash_flows):
    """
    cash_flows: pandas Series with Date index and cash flow values
                (negative = investment, positive = withdrawal)
    """
    dates = cash_flows.index
    amounts = cash_flows.values

    def npv(rate):
        return sum(
            amt / (1 + rate) ** ((d - dates[0]).days / 365)
            for amt, d in zip(amounts, dates)
        )

    return newton(npv, 0.1)

# Transaction cash flows (negative = buy, positive = sell)
cash_flows = tx.groupby("Date")["cash_flow"].sum().copy()

# Terminal portfolio value date
terminal_date = portfolio_value.index[-1]

# Add terminal portfolio value as a final inflow
if terminal_date in cash_flows.index:
    cash_flows.loc[terminal_date] += portfolio_value.iloc[-1]
else:
    cash_flows.loc[terminal_date] = portfolio_value.iloc[-1]

cash_flows = cash_flows.sort_index()
### All-in-one performance chart
plt.figure(figsize=(12, 6))

plt.plot(portfolio_value, label="Portfolio Value", linewidth=2)
plt.plot(invested_capital, label="Invested Capital", linestyle="--", linewidth=2)

# Shade profit/loss
plt.fill_between(
    portfolio_value.index,
    invested_capital,
    portfolio_value,
    where=(portfolio_value >= invested_capital),
    alpha=0.2,
    label="Profit"
)

plt.fill_between(
    portfolio_value.index,
    invested_capital,
    portfolio_value,
    where=(portfolio_value < invested_capital),
    alpha=0.2,
    label="Loss"
)

plt.title("Portfolio Value vs Invested Capital")
plt.ylabel("Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Money-Weighted Return Calculate above
portfolio_xirr = xirr(cash_flows)
print(f"Portfolio XIRR: {portfolio_xirr:.2%}")

###Realised & Unrealised Breakdown
# Transaction table
tx_sorted = tx.sort_values(["Ticker","Date"]).copy()
tx_sorted["Realised_PnL"] = 0.0

# Realised P&L Caluculation
def compute_realised_pnl_avg_cost(df):
    shares_held = 0.0
    cost_basis = 0.0
    realised = []

    for _, row in df.iterrows():
        qty = row["Shares"]
        price = row["Cost/Share ($)"]
        trade_type = row["Type"]

        if trade_type == "BUY":
            # Add to inventory
            shares_held += qty
            cost_basis += qty * price
            realised.append(0.0)

        elif trade_type == "SELL":
            if shares_held <= 0:
                realised.append(0.0)
                continue

            avg_cost = cost_basis / shares_held

            # Realised P&L on sold shares
            pnl = (price - avg_cost) * qty
            realised.append(pnl)

            # Remove sold inventory at average cost
            shares_held -= qty
            cost_basis -= avg_cost * qty

        else:
            realised.append(0.0)

    df = df.copy()
    df["Realised_PnL"] = realised
    return df

tx_realised = (
    tx_sorted
    .groupby("Ticker", group_keys=False)
    .apply(compute_realised_pnl_avg_cost)
)

# Aggregate realised P&L over time
daily_realised_pnl = (
    tx_realised.groupby("Date")["Realised_PnL"]
    .sum()
    .reindex(portfolio_value.index, fill_value=0)
)

cumulative_realised_pnl = daily_realised_pnl.cumsum()

# Unrealised P&L
unrealised_pnl = total_pnl - cumulative_realised_pnl
# Plot realised vs unrealised vs total P&L
plt.figure(figsize=(12, 6))

plt.plot(total_pnl, label="Total P&L", linewidth=2)
plt.plot(cumulative_realised_pnl, label="Realised P&L", linewidth=2)
plt.plot(unrealised_pnl, label="Unrealised P&L", linewidth=2)

plt.title("Portfolio P&L Breakdown")
plt.xlabel("Date")
plt.ylabel("P&L ($)")
plt.legend()
plt.grid(True)

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"${x:,.0f}")
)

plt.tight_layout()
plt.show()