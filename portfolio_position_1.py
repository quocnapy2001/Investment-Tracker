import pandas as pd
import matplotlib.pyplot as plt
from data_0 import *

# Transaction Cash Flow
tx = transaction_data.copy()
tx["Date"] = pd.to_datetime(tx["Date"])
tx["Type"] = tx["Type"].str.upper()

## Add sign to Share and Cost/Credit
tx["Signed_shares"] = tx["Shares"].where(
    tx["Type"] == "BUY", -tx["Shares"]
    )
tx["Cash_flow"] = tx["Total Cost/Credit ($)"].where(
    tx["Type"] == "SELL", -tx["Total Cost/Credit ($)"]
    )

## Monthy Buying
buy_tx = tx[tx["Type"] == "BUY"].copy()
buy_tx = buy_tx.set_index("Date")
monthly_spending = buy_tx.resample("M")["Total Cost/Credit ($)"].sum()

plt.figure(figsize = (12,6))
plt.plot(
    monthly_spending.index,
    monthly_spending.values,
    marker = "o"
)

plt.title("Monthly Investment Spending")
plt.xlabel("Date")
plt.ylabel("Amount Spend ($)")


plt.grid(True)
plt.tight_layout()
plt.show()

