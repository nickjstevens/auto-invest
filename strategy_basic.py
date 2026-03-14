import pandas as pd
import numpy as np
import yfinance as yf
import ffn

# ----------------------------
# 1. Download price data
# ----------------------------
tickers = ["SPY", "GLD", "TLT", "BTC-USD"]
start_date = "2020-01-01"
end_date = None  # use today's date

prices = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=True,
    progress=False
)["Close"]

prices = prices.dropna(how="all")

print("Price data:")
print(prices.tail())
print()

# ----------------------------
# 2. Basic return analysis
# ----------------------------
returns = prices.pct_change().dropna()

print("Daily returns summary:")
print(returns.describe())
print()

# ----------------------------
# 3. Portfolio performance stats with ffn
# ----------------------------
print("Performance stats:")
stats = prices.calc_stats()
print(stats.display())
print()

# ----------------------------
# 4. Simple moving average strategy
#    Buy when price > 200-day MA
#    Cash when price <= 200-day MA
# ----------------------------
asset = "SPY"

df = pd.DataFrame(index=prices.index)
df["price"] = prices[asset]
df["ret"] = df["price"].pct_change()
df["ma_200"] = df["price"].rolling(200).mean()

# Signal: 1 when above MA, else 0
df["signal"] = (df["price"] > df["ma_200"]).astype(int)

# Shift so today's signal applies to next day's return
df["strategy_ret"] = df["signal"].shift(1) * df["ret"]

# Equity curves
df["buy_hold"] = (1 + df["ret"].fillna(0)).cumprod()
df["strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

print(f"{asset} strategy preview:")
print(df.tail())
print()

# ----------------------------
# 5. Compare buy & hold vs strategy
# ----------------------------
comparison = pd.DataFrame({
    f"{asset}_buy_hold": df["buy_hold"],
    f"{asset}_ma200_strategy": df["strategy"]
}).dropna()

print("Buy & hold vs MA strategy stats:")
comparison_stats = comparison.calc_stats()
print(comparison_stats.display())