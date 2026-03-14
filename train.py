"""
Time-budgeted evaluation runner for prepared trading datasets.

Usage:
    uv run train.py

The runner samples random symbols and random windows from the prepared universe
until TIME_BUDGET_SECONDS is exhausted, producing a single SQN score suitable
for optimisation.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prepare import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_FEE_BPS,
    DEFAULT_RANDOM_SEED,
    GENERALIST_BASKET_SIZE,
    GENERALIST_WINDOWS_PER_CYCLE,
    MAX_WINDOW_BARS,
    MIN_WINDOW_BARS,
    MIN_TRADES_PER_FOLD,
    MIN_VALID_FOLDS,
    run_time_budgeted_evaluation_loop,
    score_from_oos_folds,
)


DEFAULT_RISK_FRACTION = 0.02
TIME_BUDGET_SECONDS = float(os.getenv("TIME_BUDGET_SECONDS", "300"))


@dataclass(frozen=True)
class Bundle:
    symbols: list[str]
    output_dir: str


@dataclass(frozen=True)
class StrategyConfig:
    fast_ma_bars: int = 20
    slow_ma_bars: int = 200
    momentum_bars: int = 20
    momentum_threshold: float = 0.0
    position_size: float = 1.0
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.12


def discover_bundle(output_dir: str) -> Bundle:
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}. Run prepare.py first.")

    manifest_path = os.path.join(output_dir, "prep_universe.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        symbols = [str(item["symbol"]).upper() for item in payload.get("symbols", []) if item.get("symbol")]
    else:
        symbols = []

    if not symbols:
        symbols = [
            f.removeprefix("prices_").removesuffix(".parquet").upper()
            for f in os.listdir(output_dir)
            if f.startswith("prices_") and f.endswith(".parquet")
        ]

    symbols = sorted(set(symbols))
    if not symbols:
        raise FileNotFoundError(f"No prepared price files found in {output_dir}. Run prepare.py first.")

    return Bundle(symbols=symbols, output_dir=output_dir)


def load_prices(output_dir: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(output_dir, f"prices_{symbol}.parquet")
    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def strategy_signals(prices: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """
    Strategy hook that outputs discrete actions:
      -  1.0 = buy/enter long
      -  0.0 = hold
      - -1.0 = sell/exit
    """
    close = prices["Close"].astype(float)
    ma_fast = close.rolling(config.fast_ma_bars, min_periods=config.fast_ma_bars).mean()
    ma_slow = close.rolling(config.slow_ma_bars, min_periods=config.slow_ma_bars).mean()
    momentum = close.pct_change(config.momentum_bars)

    buy_signal = (ma_fast > ma_slow) & (momentum > config.momentum_threshold)
    sell_signal = (ma_fast < ma_slow) | (momentum < -config.momentum_threshold)

    signal = pd.Series(0.0, index=prices.index, dtype=float)
    signal = signal.mask(buy_signal, 1.0)
    signal = signal.mask(sell_signal, -1.0)
    return signal.rename("signal")


def signal_to_position(prices: pd.DataFrame, signal: pd.Series, config: StrategyConfig) -> pd.Series:
    """Convert buy/sell/hold signals into long-only position sizing with stop and trailing exits."""
    close = prices["Close"].astype(float)
    low = prices["Low"].astype(float)

    position = pd.Series(0.0, index=prices.index, dtype=float)
    in_trade = False
    entry_price = math.nan
    highest_close = math.nan

    for i in range(len(prices)):
        if in_trade:
            stop_hit = low.iloc[i] <= entry_price * (1.0 - config.stop_loss_pct)
            highest_close = max(highest_close, float(close.iloc[i]))
            trailing_stop_hit = close.iloc[i] <= highest_close * (1.0 - config.trailing_stop_pct)
            explicit_exit = signal.iloc[i] < 0

            if stop_hit or trailing_stop_hit or explicit_exit:
                in_trade = False
                entry_price = math.nan
                highest_close = math.nan
                position.iloc[i] = 0.0
                continue

            position.iloc[i] = config.position_size
            continue

        if signal.iloc[i] > 0:
            in_trade = True
            entry_price = float(close.iloc[i])
            highest_close = entry_price
            position.iloc[i] = config.position_size

    return position.rename("position")


def evaluate_slice(
    prices: pd.DataFrame,
    strategy_config: StrategyConfig,
    trade_start_idx: int = 0,
    risk_fraction: float = DEFAULT_RISK_FRACTION,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> dict[str, float | list[float]]:
    required = {"Open", "Close", "Low", "High"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price dataframe missing columns: {sorted(missing)}")

    df = prices.copy()
    if not 0 <= trade_start_idx < len(df):
        raise ValueError("trade_start_idx must be within the available price history.")

    signal = strategy_signals(df, strategy_config)
    position = signal_to_position(df, signal, strategy_config).clip(lower=0.0, upper=1.0).fillna(0.0)
    position.iloc[:trade_start_idx] = 0.0
    if len(position) != len(df):
        raise ValueError("strategy must return a series aligned with prices index.")

    ret = df["Close"].pct_change().fillna(0.0)
    traded_position = position.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(position.abs())
    fee_rate = fee_bps / 10000.0
    cost = turnover * fee_rate

    strategy_ret = traded_position * ret - cost
    strategy_ret_eval = strategy_ret.iloc[trade_start_idx:]
    position_eval = position.iloc[trade_start_idx:]
    df_eval = df.iloc[trade_start_idx:]

    equity = (1.0 + strategy_ret_eval).cumprod()

    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    max_drawdown = float(drawdown.min())

    years = max((df_eval.index[-1] - df_eval.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float((equity.iloc[-1] ** (1.0 / years)) - 1.0)

    vol = float(strategy_ret_eval.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * strategy_ret_eval.mean() / vol) if vol > 0 else math.nan

    trade_r: list[float] = []
    trade_duration_bars: list[float] = []
    trade_duration_days: list[float] = []
    in_trade = False
    entry_equity = 1.0
    entry_idx = 0
    for i in range(len(df_eval)):
        if not in_trade and position_eval.iloc[i] > 0 and (i == 0 or position_eval.iloc[i - 1] <= 0):
            in_trade = True
            entry_equity = float(equity.iloc[i - 1]) if i > 0 else 1.0
            entry_idx = i
        if in_trade and (position_eval.iloc[i] <= 0 or i == len(df_eval) - 1):
            exit_equity = float(equity.iloc[i])
            trade_ret = exit_equity / entry_equity - 1.0
            trade_r.append(float(trade_ret / risk_fraction))

            bars_open = float(i - entry_idx + 1)
            days_open = float((df_eval.index[i] - df_eval.index[entry_idx]).days)
            trade_duration_bars.append(bars_open)
            trade_duration_days.append(days_open)
            in_trade = False

    wins = sum(1 for r in trade_r if r > 0)
    win_rate = float(wins / len(trade_r)) if trade_r else math.nan

    return {
        "trade_r": trade_r,
        "trade_duration_bars": trade_duration_bars,
        "trade_duration_days": trade_duration_days,
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "num_trades": float(len(trade_r)),
        "win_rate": win_rate,
        "wins": float(wins),
    }


def random_window(prices: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(prices)
    if n < MIN_WINDOW_BARS:
        raise ValueError("Not enough history for random window sampling.")

    length = int(rng.integers(MIN_WINDOW_BARS, min(MAX_WINDOW_BARS, n) + 1))
    start_idx = int(rng.integers(0, n - length + 1))
    end_idx = start_idx + length

    history_plus_window = prices.iloc[:end_idx].copy()
    history_plus_window.attrs["trade_start_idx"] = start_idx
    return history_plus_window


def collect_metrics(samples: list[dict[str, float | list[float]]]) -> dict[str, float]:
    cagr = [float(m["cagr"]) for m in samples if np.isfinite(float(m["cagr"]))]
    mdd = [float(m["max_drawdown"]) for m in samples if np.isfinite(float(m["max_drawdown"]))]
    sharpe = [float(m["sharpe"]) for m in samples if np.isfinite(float(m["sharpe"]))]
    win_rate = [float(m["win_rate"]) for m in samples if np.isfinite(float(m["win_rate"]))]
    trade_r = [
        float(r)
        for sample in samples
        for r in sample["trade_r"]
        if np.isfinite(float(r))
    ]
    trade_duration_bars = [
        float(bars)
        for sample in samples
        for bars in sample["trade_duration_bars"]
        if np.isfinite(float(bars))
    ]
    trade_duration_days = [
        float(days)
        for sample in samples
        for days in sample["trade_duration_days"]
        if np.isfinite(float(days))
    ]

    return {
        "median_cagr": float(np.median(cagr)) if cagr else math.nan,
        "median_max_drawdown": float(np.median(mdd)) if mdd else math.nan,
        "median_sharpe": float(np.median(sharpe)) if sharpe else math.nan,
        "median_win_rate": float(np.median(win_rate)) if win_rate else math.nan,
        "median_trade_r": float(np.median(trade_r)) if trade_r else math.nan,
        "median_trade_duration_bars": float(np.median(trade_duration_bars)) if trade_duration_bars else math.nan,
        "median_trade_duration_days": float(np.median(trade_duration_days)) if trade_duration_days else math.nan,
    }


def train() -> None:
    start_time = time.perf_counter()
    deadline = start_time + TIME_BUDGET_SECONDS

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    bundle = discover_bundle(DEFAULT_OUTPUT_DIR)
    prices_by_symbol = {symbol: load_prices(bundle.output_dir, symbol) for symbol in bundle.symbols}

    strategy_config = StrategyConfig(
        fast_ma_bars=20,
        slow_ma_bars=200,
        momentum_bars=20,
        momentum_threshold=0.0,
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
        random_window_fn=random_window,
        evaluate_slice_fn=lambda window: evaluate_slice(
            window,
            strategy_config=strategy_config,
            trade_start_idx=int(window.attrs.get("trade_start_idx", 0)),
        ),
    )

    elapsed = time.perf_counter() - start_time

    valid_folds = sum(1 for fold in combined_fold_trade_r if len(fold) >= MIN_TRADES_PER_FOLD)
    combined_score = score_from_oos_folds(combined_fold_trade_r)
    combined_stats = collect_metrics(combined_samples)

    lines = [
        "Evaluation complete",
        f"time_budget_seconds      : {TIME_BUDGET_SECONDS:.1f}",
        f"elapsed_seconds          : {elapsed:.1f}",
        f"cycles_completed         : {cycles}",
        f"universe_size            : {len(bundle.symbols)}",
        f"combined_folds           : {len(combined_fold_trade_r)}",
        f"valid_folds_for_sqn      : {valid_folds}",
        f"min_valid_folds_required : {MIN_VALID_FOLDS}",
        (
            f"combined_score_sqn       : {combined_score:.6f}"
            if np.isfinite(combined_score)
            else "combined_score_sqn       : nan"
        ),
        (
            f"combined_median_cagr     : {combined_stats['median_cagr']:.2%}"
            if np.isfinite(combined_stats["median_cagr"])
            else "combined_median_cagr     : nan"
        ),
        (
            f"combined_median_drawdown : {combined_stats['median_max_drawdown']:.2%}"
            if np.isfinite(combined_stats["median_max_drawdown"])
            else "combined_median_drawdown : nan"
        ),
        (
            f"combined_median_sharpe   : {combined_stats['median_sharpe']:.3f}"
            if np.isfinite(combined_stats["median_sharpe"])
            else "combined_median_sharpe   : nan"
        ),
        (
            f"combined_median_win_rate : {combined_stats['median_win_rate']:.2%}"
            if np.isfinite(combined_stats["median_win_rate"])
            else "combined_median_win_rate : nan"
        ),
        (
            f"combined_median_trade_r  : {combined_stats['median_trade_r']:.3f}R"
            if np.isfinite(combined_stats["median_trade_r"])
            else "combined_median_trade_r  : nan"
        ),
        (
            f"median_trade_open_bars   : {combined_stats['median_trade_duration_bars']:.1f}"
            if np.isfinite(combined_stats["median_trade_duration_bars"])
            else "median_trade_open_bars   : nan"
        ),
        (
            f"median_trade_open_days   : {combined_stats['median_trade_duration_days']:.1f}"
            if np.isfinite(combined_stats["median_trade_duration_days"])
            else "median_trade_open_days   : nan"
        ),
    ]

    for line in lines:
        print(line)


if __name__ == "__main__":
    train()
