"""
Prepare trading data and evaluation utilities for autoresearch-style optimisation.

This script currently focuses on GLD daily data and writes a compact preparation
bundle that can be consumed by optimisation loops.

Outputs (default: ~/.cache/auto-invest/prep):
  - prices_GLD.parquet          : OHLCV history
  - folds_GLD.json              : walk-forward OOS fold definitions
  - prep_GLD.json               : metadata + metric description

Evaluation metric:
  score = median(SQN of net R-multiples on each OOS fold)

Usage:
    python prepare.py
    python prepare.py --symbol GLD --start 2006-01-01 --end 2026-01-01
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_SYMBOL = "GLD"
DEFAULT_START = "2004-11-18"  # GLD inception window
DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "auto-invest", "prep")


@dataclass(frozen=True)
class Fold:
    """Represents one walk-forward OOS fold."""

    fold_id: int
    train_start: str
    train_end: str
    oos_start: str
    oos_end: str


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

    # yfinance can return a MultiIndex for columns even for one ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns from Yahoo data: {missing}")

    out = df[required].copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "Date"
    out = out.sort_index().dropna()
    return out


def make_walk_forward_folds(
    index: pd.DatetimeIndex,
    n_folds: int,
    min_train_bars: int,
    oos_bars: int,
) -> list[Fold]:
    """Create anchored walk-forward folds on a datetime index."""
    if len(index) < min_train_bars + oos_bars:
        raise ValueError(
            f"Not enough bars ({len(index)}) for min_train_bars={min_train_bars} and oos_bars={oos_bars}."
        )

    max_possible = (len(index) - min_train_bars) // oos_bars
    if max_possible < 1:
        raise ValueError("Cannot create even one OOS fold with provided settings.")

    n_folds = min(n_folds, max_possible)
    folds: list[Fold] = []

    for i in range(n_folds):
        train_end_idx = min_train_bars - 1 + i * oos_bars
        oos_start_idx = train_end_idx + 1
        oos_end_idx = oos_start_idx + oos_bars - 1

        train_start = index[0]
        train_end = index[train_end_idx]
        oos_start = index[oos_start_idx]
        oos_end = index[oos_end_idx]

        folds.append(
            Fold(
                fold_id=i,
                train_start=train_start.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                oos_start=oos_start.strftime("%Y-%m-%d"),
                oos_end=oos_end.strftime("%Y-%m-%d"),
            )
        )

    return folds


def sqn(net_r_multiples: Sequence[float]) -> float:
    """System Quality Number for a sequence of net R-multiples."""
    arr = np.asarray(net_r_multiples, dtype=float)
    arr = arr[np.isfinite(arr)]

    if arr.size < 2:
        return math.nan

    stdev = float(arr.std(ddof=1))
    if stdev == 0.0:
        return math.nan

    return float(np.sqrt(arr.size) * arr.mean() / stdev)


def score_from_oos_folds(oos_net_r_multiples: Iterable[Sequence[float]]) -> float:
    """
    Evaluation metric for optimisation:
        score = median(SQN of net R-multiples on each OOS fold)
    """
    per_fold_sqn = [sqn(fold_values) for fold_values in oos_net_r_multiples]
    per_fold_sqn = [v for v in per_fold_sqn if np.isfinite(v)]
    if not per_fold_sqn:
        return math.nan
    return float(np.median(per_fold_sqn))


def save_preparation_bundle(
    symbol: str,
    prices: pd.DataFrame,
    folds: list[Fold],
    output_dir: str,
) -> None:
    """Persist data and metadata for downstream optimisation scripts."""
    os.makedirs(output_dir, exist_ok=True)

    prices_path = os.path.join(output_dir, f"prices_{symbol}.parquet")
    folds_path = os.path.join(output_dir, f"folds_{symbol}.json")
    prep_path = os.path.join(output_dir, f"prep_{symbol}.json")

    prices.to_parquet(prices_path)

    with open(folds_path, "w", encoding="utf-8") as f:
        json.dump([asdict(fold) for fold in folds], f, indent=2)

    prep_payload = {
        "symbol": symbol,
        "rows": int(len(prices)),
        "date_start": prices.index.min().strftime("%Y-%m-%d"),
        "date_end": prices.index.max().strftime("%Y-%m-%d"),
        "files": {
            "prices": os.path.basename(prices_path),
            "folds": os.path.basename(folds_path),
        },
        "evaluation_metric": {
            "name": "median_fold_sqn",
            "formula": "score = median(SQN of net R-multiples on each OOS fold)",
            "sqn": "SQN = sqrt(n) * mean(R) / std(R, ddof=1)",
            "function": "score_from_oos_folds",
        },
    }

    with open(prep_path, "w", encoding="utf-8") as f:
        json.dump(prep_payload, f, indent=2)

    print(f"Saved prices  : {prices_path}")
    print(f"Saved folds   : {folds_path}")
    print(f"Saved metadata: {prep_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare trading data for optimisation.")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL)
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=8, help="Maximum number of OOS folds.")
    parser.add_argument("--min-train-bars", type=int, default=504, help="Initial IS window length.")
    parser.add_argument("--oos-bars", type=int, default=126, help="Bars per OOS fold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    symbol = args.symbol.upper().strip()
    print(f"Downloading {symbol} daily data from Yahoo Finance...")
    prices = download_price_history(symbol=symbol, start=args.start, end=args.end)
    print(f"Downloaded {len(prices)} rows from {prices.index.min().date()} to {prices.index.max().date()}.")

    folds = make_walk_forward_folds(
        index=prices.index,
        n_folds=args.n_folds,
        min_train_bars=args.min_train_bars,
        oos_bars=args.oos_bars,
    )
    print(f"Built {len(folds)} walk-forward OOS folds.")

    save_preparation_bundle(symbol=symbol, prices=prices, folds=folds, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
