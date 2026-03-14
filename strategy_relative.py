import pandas as pd
import numpy as np
import yfinance as yf

tickers = ["SPY", "GLD", "TLT", "QQQ"]
start_date = "2020-01-01"

prices = yf.download(
    tickers,
    start=start_date,
    auto_adjust=True,
    progress=False
)["Close"].dropna()

# Monthly prices
monthly_prices = prices.resample("ME").last()

# 3-month momentum
momentum = monthly_prices.pct_change(3)
print(momentum)
# # Pick best asset each month
# best_asset = momentum.idxmax(axis=1)

# # Build monthly strategy returns
# monthly_returns = monthly_prices.pct_change()

# strategy_returns = []
# dates = []

# for i in range(1, len(monthly_returns)):
#     date = monthly_returns.index[i]
#     prev_date = monthly_returns.index[i - 1]

#     chosen_asset = best_asset.loc[prev_date]
#     ret = monthly_returns.loc[date, chosen_asset]

#     strategy_returns.append(ret)
#     dates.append(date)

# strategy = pd.Series(strategy_returns, index=dates, name="rotation_strategy")

# # Benchmark: SPY buy & hold monthly
# spy_bh = monthly_returns["SPY"].loc[strategy.index]

# # Equity curves
# equity = pd.DataFrame({
#     "SPY_buy_hold": (1 + spy_bh.fillna(0)).cumprod(),
#     "Rotation_strategy": (1 + strategy.fillna(0)).cumprod()
# })

# print("Chosen asset by month:")
# print(best_asset.tail(12))
# print()

# print("Equity curves:")
# print(equity.tail())