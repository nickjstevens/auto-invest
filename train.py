"""
Time-budgeted evaluation runner for prepared trading datasets.

Usage:
    uv run train.py

The runner evaluates a strategy in two modes:
  1) Generalist: many random windows sampled from a random basket of symbols.
  2) Specialist: one symbol sampled across explicit chronological regimes.

Evaluation continues until TIME_BUDGET_SECONDS is exhausted.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prepare import DEFAULT_OUTPUT_DIR, score_from_oos_folds


DEFAULT_RISK_FRACTION = 0.02
DEFAULT_FEE_BPS = 2.0
TIME_BUDGET_SECONDS = float(os.getenv("TIME_BUDGET_SECONDS", "300"))
DEFAULT_RANDOM_SEED = 42
MIN_WINDOW_BARS = 252
MAX_WINDOW_BARS = 756
GENERALIST_BASKET_SIZE = 12
GENERALIST_WINDOWS_PER_CYCLE = 32
SPECIALIST_WINDOWS_PER_REGIME = 24
REGIME_COUNT = 4


@dataclass(frozen=True)
class Bundle:
    symbols: list[str]
    specialist_symbol: str
    output_dir: str


def discover_bundle(output_dir: str) -> Bundle:
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}. Run prepare.py first.")

    manifest_path = os.path.join(output_dir, "prep_universe.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        symbols = [str(item["symbol"]).upper() for item in payload.get("symbols", []) if item.get("symbol")]
        specialist_symbol = str(payload.get("specialist_symbol") or "").upper()
    else:
        symbols = []
        specialist_symbol = ""

    if not symbols:
        symbols = [
            f.removeprefix("prices_").removesuffix(".parquet").upper()
            for f in os.listdir(output_dir)
            if f.startswith("prices_") and f.endswith(".parquet")
        ]

    symbols = sorted(set(symbols))
    if not symbols:
        raise FileNotFoundError(f"No prepared price files found in {output_dir}. Run prepare.py first.")

    if specialist_symbol not in symbols:
        specialist_symbol = "GLD" if "GLD" in symbols else symbols[0]

    return Bundle(symbols=symbols, specialist_symbol=specialist_symbol, output_dir=output_dir)


def load_prices(output_dir: str, symbol: str) -> pd.DataFrame:
    path = os.path.join(output_dir, f"prices_{symbol}.parquet")
    prices = pd.read_parquet(path)
    prices.index = pd.to_datetime(prices.index)
    return prices.sort_index()


def strategy_positions(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy hook.

    Simple regime-aware trend baseline:
      - Above 200DMA and positive 20-day momentum -> long 1.0
      - Otherwise flat 0.0
    """
    close = prices["Close"].astype(float)
    ma_fast = close.rolling(20, min_periods=20).mean()
    ma_slow = close.rolling(200, min_periods=200).mean()
    momentum = close.pct_change(20)

    long_signal = (ma_fast > ma_slow) & (momentum > 0)
    return long_signal.astype(float).reindex(prices.index).fillna(0.0).rename("position")


def evaluate_slice(
    prices: pd.DataFrame,
    risk_fraction: float = DEFAULT_RISK_FRACTION,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> dict[str, float | list[float]]:
    required = {"Open", "Close", "Low"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price dataframe missing columns: {sorted(missing)}")

    df = prices.copy()
    position = strategy_positions(df).clip(lower=0.0, upper=1.0).fillna(0.0)
    if len(position) != len(df):
        raise ValueError("strategy_positions must return a series aligned with prices index.")

    ret = df["Close"].pct_change().fillna(0.0)
    traded_position = position.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(position.abs())
    fee_rate = fee_bps / 10000.0
    cost = turnover * fee_rate

    strategy_ret = traded_position * ret - cost
    equity = (1.0 + strategy_ret).cumprod()

    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    max_drawdown = float(drawdown.min())

    years = max((df.index[-1] - df.index[0]).days / 365.25, 1 / 365.25)
    total_return = float(equity.iloc[-1] - 1.0)
    cagr = float((equity.iloc[-1] ** (1.0 / years)) - 1.0)

    vol = float(strategy_ret.std(ddof=1))
    sharpe = float(np.sqrt(252.0) * strategy_ret.mean() / vol) if vol > 0 else math.nan

    trade_r: list[float] = []
    in_trade = False
    entry_equity = 1.0
    for i in range(len(df)):
        if not in_trade and position.iloc[i] > 0 and (i == 0 or position.iloc[i - 1] <= 0):
            in_trade = True
            entry_equity = float(equity.iloc[i - 1]) if i > 0 else 1.0
        if in_trade and (position.iloc[i] <= 0 or i == len(df) - 1):
            exit_equity = float(equity.iloc[i])
            trade_ret = exit_equity / entry_equity - 1.0
            trade_r.append(float(trade_ret / risk_fraction))
            in_trade = False

    wins = sum(1 for r in trade_r if r > 0)
    win_rate = float(wins / len(trade_r)) if trade_r else math.nan

    return {
        "trade_r": trade_r,
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
    return prices.iloc[start_idx : start_idx + length]


def specialist_regime_windows(prices: pd.DataFrame, rng: np.random.Generator) -> list[pd.DataFrame]:
    windows: list[pd.DataFrame] = []
    n = len(prices)
    if n < MIN_WINDOW_BARS:
        return windows

    boundaries = np.linspace(0, n, REGIME_COUNT + 1, dtype=int)
    for regime_id in range(REGIME_COUNT):
        lo = int(boundaries[regime_id])
        hi = int(boundaries[regime_id + 1])
        regime_slice = prices.iloc[lo:hi]
        if len(regime_slice) < MIN_WINDOW_BARS:
            continue
        for _ in range(SPECIALIST_WINDOWS_PER_REGIME):
            windows.append(random_window(regime_slice, rng))
    return windows


def collect_metrics(samples: list[dict[str, float | list[float]]]) -> dict[str, float]:
    cagr = [float(m["cagr"]) for m in samples if np.isfinite(float(m["cagr"]))]
    mdd = [float(m["max_drawdown"]) for m in samples if np.isfinite(float(m["max_drawdown"]))]
    sharpe = [float(m["sharpe"]) for m in samples if np.isfinite(float(m["sharpe"]))]
    win_rate = [float(m["win_rate"]) for m in samples if np.isfinite(float(m["win_rate"]))]

    return {
        "median_cagr": float(np.median(cagr)) if cagr else math.nan,
        "median_max_drawdown": float(np.median(mdd)) if mdd else math.nan,
        "median_sharpe": float(np.median(sharpe)) if sharpe else math.nan,
        "median_win_rate": float(np.median(win_rate)) if win_rate else math.nan,
    }


def train() -> None:
    start_time = time.perf_counter()
    deadline = start_time + TIME_BUDGET_SECONDS

    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    bundle = discover_bundle(DEFAULT_OUTPUT_DIR)
    prices_by_symbol = {symbol: load_prices(bundle.output_dir, symbol) for symbol in bundle.symbols}

    generalist_fold_trade_r: list[list[float]] = []
    generalist_samples: list[dict[str, float | list[float]]] = []
    specialist_fold_trade_r: list[list[float]] = []
    specialist_samples: list[dict[str, float | list[float]]] = []

    specialist_prices = prices_by_symbol[bundle.specialist_symbol]
    cycles = 0

    while time.perf_counter() < deadline:
        cycles += 1

        basket_size = min(GENERALIST_BASKET_SIZE, len(bundle.symbols))
        basket = rng.choice(bundle.symbols, size=basket_size, replace=False)

        for symbol in basket:
            prices = prices_by_symbol[str(symbol)]
            if len(prices) < MIN_WINDOW_BARS:
                continue

            for _ in range(GENERALIST_WINDOWS_PER_CYCLE):
                if time.perf_counter() >= deadline:
                    break
                window = random_window(prices, rng)
                metrics = evaluate_slice(window)
                generalist_fold_trade_r.append(metrics["trade_r"])
                generalist_samples.append(metrics)

        for window in specialist_regime_windows(specialist_prices, rng):
            if time.perf_counter() >= deadline:
                break
            metrics = evaluate_slice(window)
            specialist_fold_trade_r.append(metrics["trade_r"])
            specialist_samples.append(metrics)

    elapsed = time.perf_counter() - start_time

    generalist_score = score_from_oos_folds(generalist_fold_trade_r)
    specialist_score = score_from_oos_folds(specialist_fold_trade_r)

    generalist_stats = collect_metrics(generalist_samples)
    specialist_stats = collect_metrics(specialist_samples)

    print("Evaluation complete")
    print(f"time_budget_seconds      : {TIME_BUDGET_SECONDS:.1f}")
    print(f"elapsed_seconds          : {elapsed:.1f}")
    print(f"cycles_completed         : {cycles}")
    print(f"universe_size            : {len(bundle.symbols)}")
    print(f"specialist_symbol        : {bundle.specialist_symbol}")
    print(f"generalist_folds         : {len(generalist_fold_trade_r)}")
    print(f"specialist_folds         : {len(specialist_fold_trade_r)}")
    print(
        f"generalist_score_sqn     : {generalist_score:.6f}"
        if np.isfinite(generalist_score)
        else "generalist_score_sqn     : nan"
    )
    print(
        f"specialist_score_sqn     : {specialist_score:.6f}"
        if np.isfinite(specialist_score)
        else "specialist_score_sqn     : nan"
    )
    print(
        f"combined_score_sqn       : {np.nanmedian([generalist_score, specialist_score]):.6f}"
        if np.isfinite(generalist_score) or np.isfinite(specialist_score)
        else "combined_score_sqn       : nan"
    )

    for label, stats in (("generalist", generalist_stats), ("specialist", specialist_stats)):
        print(
            f"{label}_median_cagr      : {stats['median_cagr']:.2%}"
            if np.isfinite(stats["median_cagr"])
            else f"{label}_median_cagr      : nan"
        )
        print(
            f"{label}_median_drawdown  : {stats['median_max_drawdown']:.2%}"
            if np.isfinite(stats["median_max_drawdown"])
            else f"{label}_median_drawdown  : nan"
        )
        print(
            f"{label}_median_sharpe    : {stats['median_sharpe']:.3f}"
            if np.isfinite(stats["median_sharpe"])
            else f"{label}_median_sharpe    : nan"
        )
        print(
            f"{label}_median_win_rate  : {stats['median_win_rate']:.2%}"
            if np.isfinite(stats["median_win_rate"])
            else f"{label}_median_win_rate  : nan"
        )


if __name__ == "__main__":
    train()
