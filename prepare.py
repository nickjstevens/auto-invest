"""
Prepare full-history trading data and reusable evaluation utilities.

The output bundle is intentionally broad: it stores complete daily OHLCV history
for a symbol universe, then training/evaluation scripts can sample random windows
from those full histories.

Outputs (default: ~/.cache/auto-invest/prep):
  - prices_<SYMBOL>.parquet    : full downloaded OHLCV history per symbol
  - prep_universe.json         : universe metadata and evaluation description

Evaluation metric:
  score = median(SQN of net R-multiples on each OOS fold)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_SPECIALIST_SYMBOL = "GLD"
DEFAULT_START = "1990-01-01"
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "auto-invest", "prep")
DEFAULT_FEE_BPS = 2.0
DEFAULT_RANDOM_SEED = 42
MIN_WINDOW_BARS = 252
MAX_WINDOW_BARS = 756
GENERALIST_BASKET_SIZE = 12
GENERALIST_WINDOWS_PER_CYCLE = 32
TIME_BUDGET_SECONDS = 60
DEFAULT_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "IEF",
    "LQD",
    "HYG",
    "GLD",
    "SLV",
    "USO",
    "XLE",
    "XLK",
    "XLF",
    "XLI",
    "XLP",
    "XLV",
    "XLY",
    "XLB",
    "XLU",
    "VNQ",
    "EEM",
    "EFA",
    "FXI",
    "ARKK",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
]


def parse_symbols(raw_symbols: str | None) -> list[str]:
    if not raw_symbols:
        return DEFAULT_SYMBOLS
    symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
    if not symbols:
        raise ValueError("No valid symbols parsed from --symbols.")
    return list(dict.fromkeys(symbols))


def download_price_history(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance."""
    df = yf.download(
        tickers=symbol,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data downloaded for symbol={symbol!r}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns from Yahoo data: {missing}")

    out = df[required].copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    return out.sort_index().dropna()


MIN_TRADES_PER_FOLD = 5
MIN_VALID_FOLDS = 25
FOLD_SHRINKAGE = 25.0


def sqn(net_r_multiples: Sequence[float], min_trades: int = MIN_TRADES_PER_FOLD) -> float:
    """System Quality Number for a sequence of net R-multiples."""
    arr = np.asarray(net_r_multiples, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < max(2, min_trades):
        return math.nan

    stdev = float(arr.std(ddof=1))
    if stdev == 0.0:
        return math.nan

    return float(np.sqrt(arr.size) * arr.mean() / stdev)


def score_from_oos_folds(
    oos_net_r_multiples: Iterable[Sequence[float]],
    min_trades_per_fold: int = MIN_TRADES_PER_FOLD,
    min_valid_folds: int = MIN_VALID_FOLDS,
    fold_shrinkage: float = FOLD_SHRINKAGE,
) -> float:
    """Evaluation metric: sample-size-aware median SQN across folds."""
    per_fold_sqn = [sqn(fold_values, min_trades=min_trades_per_fold) for fold_values in oos_net_r_multiples]
    per_fold_sqn = [v for v in per_fold_sqn if np.isfinite(v)]
    valid_folds = len(per_fold_sqn)
    if valid_folds < min_valid_folds:
        return math.nan

    raw_median = float(np.median(per_fold_sqn))
    fold_confidence = float(np.sqrt(valid_folds / (valid_folds + fold_shrinkage)))
    return raw_median * fold_confidence


def run_time_budgeted_evaluation_loop(
    *,
    symbols: Sequence[str],
    prices_by_symbol: dict[str, pd.DataFrame],
    deadline: float,
    rng: np.random.Generator,
    min_window_bars: int,
    generalist_basket_size: int,
    generalist_windows_per_cycle: int,
    random_window_fn: Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame],
    evaluate_slice_fn: Callable[[pd.DataFrame], dict[str, float | list[float]]],
) -> tuple[list[list[float]], list[dict[str, float | list[float]]], int]:
    """Run the shared time-budgeted random-window evaluation loop."""
    combined_fold_trade_r: list[list[float]] = []
    combined_samples: list[dict[str, float | list[float]]] = []
    cycles = 0

    while time.perf_counter() < deadline:
        cycles += 1

        basket_size = min(generalist_basket_size, len(symbols))
        basket = rng.choice(symbols, size=basket_size, replace=False)

        for symbol in basket:
            prices = prices_by_symbol[str(symbol)]
            if len(prices) < min_window_bars:
                continue

            for _ in range(generalist_windows_per_cycle):
                if time.perf_counter() >= deadline:
                    break

                window = random_window_fn(prices, rng)
                metrics = evaluate_slice_fn(window)
                combined_fold_trade_r.append(metrics["trade_r"])
                combined_samples.append(metrics)

    return combined_fold_trade_r, combined_samples, cycles


def save_preparation_bundle(
    prices_by_symbol: dict[str, pd.DataFrame],
    specialist_symbol: str,
    output_dir: str,
) -> None:
    """Persist prices and universe metadata for downstream optimisation scripts."""
    os.makedirs(output_dir, exist_ok=True)

    summaries: list[dict[str, str | int]] = []
    for symbol, prices in sorted(prices_by_symbol.items()):
        prices_path = os.path.join(output_dir, f"prices_{symbol}.parquet")
        prices.to_parquet(prices_path)
        summaries.append(
            {
                "symbol": symbol,
                "rows": int(len(prices)),
                "date_start": prices.index.min().strftime("%Y-%m-%d"),
                "date_end": prices.index.max().strftime("%Y-%m-%d"),
                "prices": os.path.basename(prices_path),
            }
        )
        print(f"Saved prices  : {prices_path}")

    prep_payload = {
        "universe_size": len(summaries),
        "specialist_symbol": specialist_symbol,
        "symbols": summaries,
        "evaluation_metric": {
            "name": "median_fold_sqn",
            "formula": "score = median(SQN on valid folds) * sqrt(valid_folds / (valid_folds + 25))",
            "sqn": "SQN = sqrt(n) * mean(R) / std(R, ddof=1)",
            "constraints": {
                "min_trades_per_fold": MIN_TRADES_PER_FOLD,
                "min_valid_folds": MIN_VALID_FOLDS,
            },
            "function": "score_from_oos_folds",
        },
    }

    prep_path = os.path.join(output_dir, "prep_universe.json")
    with open(prep_path, "w", encoding="utf-8") as f:
        json.dump(prep_payload, f, indent=2)

    print(f"Saved metadata: {prep_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare multi-symbol full-history trading data.")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--specialist-symbol", type=str, default=DEFAULT_SPECIALIST_SYMBOL)
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = parse_symbols(args.symbols)
    specialist_symbol = args.specialist_symbol.upper().strip()
    if specialist_symbol not in symbols:
        symbols.append(specialist_symbol)

    prices_by_symbol: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}

    for symbol in symbols:
        print(f"Downloading {symbol} daily full history from Yahoo Finance...")
        try:
            prices = download_price_history(symbol=symbol, start=args.start, end=args.end)
        except Exception as exc:  # noqa: BLE001
            failures[symbol] = str(exc)
            print(f"Failed {symbol}: {exc}")
            continue

        prices_by_symbol[symbol] = prices
        print(f"Downloaded {len(prices)} rows for {symbol}: {prices.index.min().date()} to {prices.index.max().date()}")

    if not prices_by_symbol:
        raise RuntimeError("No symbols were downloaded successfully.")

    if specialist_symbol not in prices_by_symbol:
        specialist_symbol = sorted(prices_by_symbol.keys())[0]
        print(f"Specialist symbol unavailable; falling back to {specialist_symbol}.")

    save_preparation_bundle(prices_by_symbol=prices_by_symbol, specialist_symbol=specialist_symbol, output_dir=args.output_dir)

    if failures:
        print("\nSome symbols failed download:")
        for symbol, error in sorted(failures.items()):
            print(f"  - {symbol}: {error}")


if __name__ == "__main__":
    main()
