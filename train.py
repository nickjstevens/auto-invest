"""
Simple evaluation runner for prepared trading datasets.

Usage:
    uv run train.py

This script intentionally keeps the workflow minimal so an external agent can
rewrite/replace strategy logic directly in this file.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prepare import DEFAULT_OUTPUT_DIR, Fold, score_from_oos_folds


DEFAULT_RISK_FRACTION = 0.02
DEFAULT_FEE_BPS = 2.0


@dataclass(frozen=True)
class Bundle:
    symbol: str
    prices_path: str
    folds_path: str


def discover_bundle(output_dir: str) -> Bundle:
    """Pick the most recently modified symbol bundle from the output directory."""
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}. Run prepare.py first.")

    price_files = [f for f in os.listdir(output_dir) if f.startswith("prices_") and f.endswith(".parquet")]
    if not price_files:
        raise FileNotFoundError(f"No prepared price files found in {output_dir}. Run prepare.py first.")

    candidates: list[Bundle] = []
    for filename in price_files:
        symbol = filename.removeprefix("prices_").removesuffix(".parquet")
        prices_path = os.path.join(output_dir, filename)
        folds_path = os.path.join(output_dir, f"folds_{symbol}.json")
        if os.path.exists(folds_path):
            candidates.append(Bundle(symbol=symbol, prices_path=prices_path, folds_path=folds_path))

    if not candidates:
        raise FileNotFoundError(f"Found prices files in {output_dir}, but no matching folds_*.json files.")

    latest = max(candidates, key=lambda b: os.path.getmtime(b.prices_path))
    return latest


def load_bundle(bundle: Bundle) -> tuple[pd.DataFrame, list[Fold]]:
    prices = pd.read_parquet(bundle.prices_path)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    with open(bundle.folds_path, "r", encoding="utf-8") as f:
        raw_folds = json.load(f)

    folds = [Fold(**item) for item in raw_folds]
    if not folds:
        raise RuntimeError("No folds found in preparation bundle.")

    return prices, folds


def strategy_positions(prices: pd.DataFrame) -> pd.Series:
    """
    Strategy hook.

    By default this is always long (1.0). Replace this function when testing
    other strategies; keep output aligned to ``prices.index`` and between 0..1.
    """
    return pd.Series(1.0, index=prices.index, name="position")


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

    # Trade segmentation: each positive step in position starts a trade.
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
    losses = sum(1 for r in trade_r if r <= 0)
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
        "losses": float(losses),
    }


def train() -> None:
    bundle = discover_bundle(DEFAULT_OUTPUT_DIR)
    prices, folds = load_bundle(bundle)

    fold_trade_r: list[list[float]] = []
    fold_cagrs: list[float] = []
    fold_mdds: list[float] = []
    fold_sharpes: list[float] = []
    fold_win_rates: list[float] = []

    total_trades = 0
    total_wins = 0

    for fold in folds:
        oos_slice = prices.loc[fold.oos_start : fold.oos_end]
        if oos_slice.empty:
            fold_trade_r.append([])
            continue

        metrics = evaluate_slice(oos_slice)
        trade_r = metrics["trade_r"]
        fold_trade_r.append(trade_r)

        total_trades += int(metrics["num_trades"])
        total_wins += int(metrics["wins"])

        for k, collector in [
            ("cagr", fold_cagrs),
            ("max_drawdown", fold_mdds),
            ("sharpe", fold_sharpes),
            ("win_rate", fold_win_rates),
        ]:
            value = float(metrics[k])
            if np.isfinite(value):
                collector.append(value)

    score = score_from_oos_folds(fold_trade_r)

    print("Evaluation complete")
    print(f"symbol                : {bundle.symbol}")
    print(f"folds                 : {len(folds)}")
    print(f"score_median_fold_sqn : {score:.6f}" if np.isfinite(score) else "score_median_fold_sqn : nan")
    print(f"total_trades          : {total_trades}")
    print(f"winning_trades        : {total_wins}")
    if total_trades > 0:
        print(f"win_rate              : {total_wins / total_trades:.2%}")
    print(
        f"median_cagr           : {np.median(fold_cagrs):.2%}"
        if fold_cagrs
        else "median_cagr           : nan"
    )
    print(
        f"median_max_drawdown   : {np.median(fold_mdds):.2%}"
        if fold_mdds
        else "median_max_drawdown   : nan"
    )
    print(
        f"median_sharpe         : {np.median(fold_sharpes):.3f}"
        if fold_sharpes
        else "median_sharpe         : nan"
    )
    print(
        f"median_fold_win_rate  : {np.median(fold_win_rates):.2%}"
        if fold_win_rates
        else "median_fold_win_rate  : nan"
    )


if __name__ == "__main__":
    train()
