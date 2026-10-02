# Investment Portfolio Tracker

A Python-based portfolio tracker that pulls transaction data from Excel, fetches live market prices from Yahoo Finance, and produces a full set of performance analytics plus a monthly optimisation strategy for future buys.

## View the Live Dashboard

**For a full overview of the portfolio, click here to open the live dashboard:**

### https://yourusername.github.io/reponame/

The dashboard gives you an interactive, at-a-glance view of:
- Current portfolio value, total P&L, and realised vs unrealised gains
- Portfolio value over time plotted against cost basis and invested capital
- Current allocation by asset type and by individual ticker
- P&L breakdown over time
- Full table of current holdings with per-asset performance

Open the link above before reading further. The dashboard is the fastest way to understand what this project produces.

## What the project does

Beyond the dashboard, the underlying notebook performs the following:

- Loads trade history from an Excel file
- Downloads price history for all traded tickers
- Tracks daily holdings, cost basis, and portfolio value over time
- Breaks down realised and unrealised P&L per asset and per sell order
- Computes Time-Weighted Return (TWR) and Money-Weighted Return (XIRR)
- Downloads dividend history and matches against holdings on ex-dates
- Runs a two-sleeve portfolio optimisation (Sortino for growth, min-variance for income)
- Backtests the optimisation strategy against actual trading decisions

## Project Stages

**Stage 1 - Data Foundation**
Loads transactions, downloads prices, cleans and reshapes.

**Stage 2 - Position Accounting**
Builds daily holdings, portfolio value, invested capital, and cost basis tracking.

**Stage 3 - Descriptive Analytics**
Charts and summaries covering monthly cash flow, allocation, P&L, returns, and dividends.

**Stage 4 - Decision Support**
Two-sleeve optimiser producing target weights and buy orders for new cash. Monthly rolling backtest compares the optimiser against actual decisions.

## Tech Stack

- Python, pandas, numpy
- yfinance for market data
- matplotlib for charts in the notebook
- scipy and scikit-learn for optimisation and covariance shrinkage
- plotly for the interactive dashboard

## Notes

- Prices are pulled with `auto_adjust=True`, so historical prices are adjusted for splits and dividends.
- The optimiser's expected returns are last year's annualised returns, which are noisy and not a forecast. The optimiser serves as an anchor for judgment, not a prescription.
- This project is a personal learning tool, not financial advice.
