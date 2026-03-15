"""
Time-budgeted evaluation runner for prepared trading datasets.

Usage:
    uv run train.py

The runner samples random symbols and random windows from the prepared universe
until TIME_BUDGET_SECONDS is exhausted, producing a single SQN score suitable
for optimisation.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd
from prepare import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    GENERALIST_BASKET_SIZE,
    GENERALIST_WINDOWS_PER_CYCLE,
    MIN_TRADES_PER_FOLD,
    MIN_WINDOW_BARS,
    Strategy,
    TIME_BUDGET_SECONDS,
    discover_bundle,
    load_prices,
    run_time_budgeted_evaluation_loop,
    score_from_oos_folds,
)


class TrendFollowingStrategy(Strategy):
    def __init__(
        self,
        *,
        fast_ma_bars: int = 20,
        slow_ma_bars: int = 200,
        momentum_bars: int = 20,
        momentum_threshold: float = 0.0,
        breakout_bars: int = 10,
        rsi_bars: int = 14,
        rsi_entry_min: float = 55.0,
        rsi_entry_max: float = 70.0,
        rsi_exit_threshold: float = 45.0,
        position_size: float = 1.0,
        stop_loss_pct: float = 0.08,
        trailing_stop_pct: float = 0.12,
        risk_fraction: float = 0.02,
    ) -> None:
        super().__init__(risk_fraction=risk_fraction)
        self.fast_ma_bars = fast_ma_bars
        self.slow_ma_bars = slow_ma_bars
        self.momentum_bars = momentum_bars
        self.momentum_threshold = momentum_threshold
        self.breakout_bars = breakout_bars
        self.rsi_bars = rsi_bars
        self.rsi_entry_min = rsi_entry_min
        self.rsi_entry_max = rsi_entry_max
        self.rsi_exit_threshold = rsi_exit_threshold
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct

    def strategy_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["Close"].astype(float)
        ma_fast = close.rolling(self.fast_ma_bars, min_periods=self.fast_ma_bars).mean()
        ma_slow = close.rolling(self.slow_ma_bars, min_periods=self.slow_ma_bars).mean()
        momentum = close.pct_change(self.momentum_bars)
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        avg_gain = gains.rolling(self.rsi_bars, min_periods=self.rsi_bars).mean()
        avg_loss = losses.rolling(self.rsi_bars, min_periods=self.rsi_bars).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        breakout_level = close.shift(1).rolling(self.breakout_bars, min_periods=self.breakout_bars).max()

        buy_signal = (
            (ma_fast > ma_slow)
            & (momentum > self.momentum_threshold)
            & (rsi >= self.rsi_entry_min)
            & (rsi <= self.rsi_entry_max)
            & (close > breakout_level)
        )
        sell_signal = (ma_fast < ma_slow) | (momentum < -self.momentum_threshold) | (rsi < self.rsi_exit_threshold)

        signal = pd.Series(0.0, index=prices.index, dtype=float)
        signal = signal.mask(buy_signal, 1.0)
        signal = signal.mask(sell_signal, -1.0)
        return signal.rename("signal")

    def signal_to_position(self, prices: pd.DataFrame, signal: pd.Series) -> pd.Series:
        close = prices["Close"].astype(float)
        low = prices["Low"].astype(float)

        position = pd.Series(0.0, index=prices.index, dtype=float)
        in_trade = False
        entry_price = float("nan")
        highest_close = float("nan")

        for i in range(len(prices)):
            if in_trade:
                stop_hit = low.iloc[i] <= entry_price * (1.0 - self.stop_loss_pct)
                highest_close = max(highest_close, float(close.iloc[i]))
                trailing_stop_hit = close.iloc[i] <= highest_close * (1.0 - self.trailing_stop_pct)
                explicit_exit = signal.iloc[i] < 0

                if stop_hit or trailing_stop_hit or explicit_exit:
                    in_trade = False
                    entry_price = float("nan")
                    highest_close = float("nan")
                    position.iloc[i] = 0.0
                    continue

                position.iloc[i] = self.position_size
                continue

            if signal.iloc[i] > 0:
                in_trade = True
                entry_price = float(close.iloc[i])
                highest_close = entry_price
                position.iloc[i] = self.position_size

        return position.rename("position")


def train() -> None:
    start_time = time.perf_counter()
    deadline = start_time + TIME_BUDGET_SECONDS

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    bundle = discover_bundle(DEFAULT_OUTPUT_DIR)
    prices_by_symbol = {symbol: load_prices(bundle.output_dir, symbol) for symbol in bundle.symbols}

    strategy = TrendFollowingStrategy(
        fast_ma_bars=20,
        slow_ma_bars=200,
        momentum_bars=20,
        momentum_threshold=0.02,
        breakout_bars=10,
        rsi_bars=14,
        rsi_entry_min=55.0,
        rsi_entry_max=70.0,
        rsi_exit_threshold=45.0,
        position_size=1.0,
        stop_loss_pct=0.08,
        trailing_stop_pct=0.12,
    )

    combined_fold_trade_r, combined_samples, cycles = run_time_budgeted_evaluation_loop(
        symbols=bundle.symbols,
        prices_by_symbol=prices_by_symbol,
        deadline=deadline,
        rng=rng,
        min_window_bars=MIN_WINDOW_BARS,
        generalist_basket_size=GENERALIST_BASKET_SIZE,
        generalist_windows_per_cycle=GENERALIST_WINDOWS_PER_CYCLE,
        random_window_fn=strategy.random_window,
        evaluate_slice_fn=lambda window: strategy.evaluate_slice(
            window,
            trade_start_idx=int(window.attrs.get("trade_start_idx", 0)),
        ),
    )

    elapsed = time.perf_counter() - start_time

    valid_folds = sum(1 for fold in combined_fold_trade_r if len(fold) >= MIN_TRADES_PER_FOLD)
    combined_score = score_from_oos_folds(combined_fold_trade_r)
    combined_stats = strategy.collect_metrics(combined_samples)

    strategy.print_metrics(
        time_budget_seconds=TIME_BUDGET_SECONDS,
        elapsed_seconds=elapsed,
        cycles=cycles,
        universe_size=len(bundle.symbols),
        fold_count=len(combined_fold_trade_r),
        valid_folds=valid_folds,
        combined_score=combined_score,
        combined_stats=combined_stats,
    )


if __name__ == "__main__":
    train()
