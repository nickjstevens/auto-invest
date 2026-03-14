"""
Minimal optimisation/evaluation loop for a baseline trading strategy.

This script consumes the preparation bundle from ``prepare.py`` and evaluates a
simple long-only moving-average crossover strategy over walk-forward OOS folds.

Evaluation metric (required by project):
    score = score_from_oos_folds(oos_net_r_multiples)

Usage:
    uv run python prepare.py
    uv run python train.py --symbol GLD --fast 20 --slow 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prepare import DEFAULT_OUTPUT_DIR, Fold, score_from_oos_folds


@dataclass(frozen=True)
class StrategyParams:
    fast: int = 20
    slow: int = 100
    stop_pct: float = 0.02
    fee_bps: float = 2.0


def load_bundle(symbol: str, output_dir: str) -> tuple[pd.DataFrame, list[Fold]]:
    """Load prepared prices + fold definitions produced by prepare.py."""
    symbol = symbol.upper().strip()
    prices_path = os.path.join(output_dir, f"prices_{symbol}.parquet")
    folds_path = os.path.join(output_dir, f"folds_{symbol}.json")

    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Missing prices file: {prices_path}. Run prepare.py first.")
    if not os.path.exists(folds_path):
        raise FileNotFoundError(f"Missing folds file: {folds_path}. Run prepare.py first.")

    prices = pd.read_parquet(prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    with open(folds_path, "r", encoding="utf-8") as f:
        raw_folds = json.load(f)

    folds = [Fold(**item) for item in raw_folds]
    if not folds:
        raise RuntimeError("No folds found in preparation bundle.")

    return prices, folds


def compute_signals(close: pd.Series, params: StrategyParams) -> pd.DataFrame:
    """Compute moving-average crossover signals for long-only trading."""
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["fast_ma"] = close.rolling(params.fast, min_periods=params.fast).mean()
    out["slow_ma"] = close.rolling(params.slow, min_periods=params.slow).mean()
    out["signal"] = (out["fast_ma"] > out["slow_ma"]).astype(int)
    out["cross_up"] = (out["signal"].diff().fillna(0) > 0)
    out["cross_down"] = (out["signal"].diff().fillna(0) < 0)
    return out


def trade_net_r_multiples(prices: pd.DataFrame, params: StrategyParams) -> list[float]:
    """
    Turn MA-cross events into net R-multiples per trade.

    - Entry: next bar open after cross_up.
    - Exit: next bar open after cross_down, or same-bar stop-loss breach.
    - Risk (1R): ``entry_price * stop_pct``.
    - Fees: charged on entry + exit in basis points.
    """
    if params.fast >= params.slow:
        raise ValueError("Expected fast < slow for MA crossover.")

    required = {"Open", "Low", "Close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Price dataframe missing columns: {sorted(missing)}")

    sig = compute_signals(prices["Close"], params)
    df = prices.join(sig[["cross_up", "cross_down"]], how="left").fillna(False)

    net_r: list[float] = []
    in_position = False
    entry_price = math.nan
    stop_price = math.nan

    fee_rate = params.fee_bps / 10000.0

    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_open = float(df.iloc[i + 1]["Open"])

        if not in_position and bool(row["cross_up"]):
            entry_price = next_open
            stop_price = entry_price * (1.0 - params.stop_pct)
            in_position = True
            continue

        if not in_position:
            continue

        # Intrabar stop check first; assumes worst-case fill exactly at stop.
        if float(row["Low"]) <= stop_price:
            exit_price = stop_price
            gross_return = exit_price / entry_price - 1.0
            net_return = gross_return - 2.0 * fee_rate
            r_multiple = net_return / params.stop_pct
            net_r.append(float(r_multiple))
            in_position = False
            continue

        if bool(row["cross_down"]):
            exit_price = next_open
            gross_return = exit_price / entry_price - 1.0
            net_return = gross_return - 2.0 * fee_rate
            r_multiple = net_return / params.stop_pct
            net_r.append(float(r_multiple))
            in_position = False

    # Close open position on final close to keep fold accounting deterministic.
    if in_position:
        exit_price = float(df.iloc[-1]["Close"])
        gross_return = exit_price / entry_price - 1.0
        net_return = gross_return - 2.0 * fee_rate
        r_multiple = net_return / params.stop_pct
        net_r.append(float(r_multiple))

    return net_r


def evaluate_on_folds(prices: pd.DataFrame, folds: list[Fold], params: StrategyParams) -> tuple[float, list[list[float]]]:
    """Evaluate strategy across OOS folds and score with score_from_oos_folds."""
    oos_trades_per_fold: list[list[float]] = []

    for fold in folds:
        oos_slice = prices.loc[fold.oos_start : fold.oos_end]
        if oos_slice.empty:
            oos_trades_per_fold.append([])
            continue
        fold_r = trade_net_r_multiples(oos_slice, params)
        oos_trades_per_fold.append(fold_r)

    score = score_from_oos_folds(oos_trades_per_fold)
    return score, oos_trades_per_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an autoresearch-style strategy training loop.")
    parser.add_argument("--symbol", type=str, default="GLD")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fast", type=int, default=20, help="Starting fast moving average window")
    parser.add_argument("--slow", type=int, default=100, help="Starting slow moving average window")
    parser.add_argument("--stop-pct", type=float, default=0.02, help="Starting stop distance as fraction of entry")
    parser.add_argument("--fee-bps", type=float, default=2.0, help="One-way fee in basis points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for parameter mutations")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap for loop iterations (default: run forever)",
    )
    return parser.parse_args()


def mutate_params(base: StrategyParams, rng: random.Random) -> StrategyParams:
    """Generate a nearby candidate strategy for the next training step."""
    fast = max(2, base.fast + rng.randint(-5, 5))
    slow = max(fast + 2, base.slow + rng.randint(-10, 10))
    stop_pct = min(0.15, max(0.005, base.stop_pct + rng.uniform(-0.003, 0.003)))
    return StrategyParams(
        fast=fast,
        slow=slow,
        stop_pct=stop_pct,
        fee_bps=base.fee_bps,
    )


def train() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    params = StrategyParams(
        fast=args.fast,
        slow=args.slow,
        stop_pct=args.stop_pct,
        fee_bps=args.fee_bps,
    )

    prices, folds = load_bundle(symbol=args.symbol, output_dir=args.output_dir)
    best_params = params
    best_score = math.nan

    step = 0
    print("Starting strategy training loop (press Ctrl+C to stop).")
    while True:
        step += 1
        candidate = params if step == 1 else mutate_params(best_params, rng)

        try:
            score, fold_trades = evaluate_on_folds(prices=prices, folds=folds, params=candidate)
        except ValueError as err:
            print(f"step={step:06d} candidate={candidate} skipped: {err}")
            continue

        has_best = np.isfinite(best_score)
        if np.isfinite(score) and (not has_best or score > best_score):
            best_score = score
            best_params = candidate
            status = "NEW_BEST"
        else:
            status = "keep"

        total_trades = sum(len(t) for t in fold_trades)
        print(
            f"step={step:06d} status={status} score={score:.6f} "
            f"best={best_score:.6f} trades={total_trades} params={candidate}"
        )

        if args.max_steps is not None and step >= args.max_steps:
            print("Reached --max-steps; stopping loop.")
            break

        time.sleep(0.2)

    print("Final best strategy")
    print(f"symbol            : {args.symbol.upper()}")
    print(f"num_folds         : {len(folds)}")
    print(f"best_params       : {best_params}")
    print(
        f"best_score_median : {best_score:.6f}"
        if np.isfinite(best_score)
        else "best_score_median : nan"
    )

train()
