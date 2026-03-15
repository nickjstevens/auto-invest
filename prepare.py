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
from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Bundle:
    symbols: list[str]
    output_dir: str


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


class Strategy(ABC):
    """Base strategy with shared evaluation and reporting utilities."""

    def __init__(
        self,
        *,
        risk_fraction: float = 0.02,
        fee_bps: float = DEFAULT_FEE_BPS,
    ) -> None:
        self.risk_fraction = risk_fraction
        self.fee_bps = fee_bps

    @abstractmethod
    def strategy_signals(self, prices: pd.DataFrame) -> pd.Series:
        """Return discrete signal actions (1.0 buy, 0.0 hold, -1.0 sell)."""

    @abstractmethod
    def signal_to_position(self, prices: pd.DataFrame, signal: pd.Series) -> pd.Series:
        """Map strategy signals to a position series aligned to prices."""

    def evaluate_slice(
        self,
        prices: pd.DataFrame,
        trade_start_idx: int = 0,
    ) -> dict[str, float | list[float]]:
        required = {"Open", "Close", "Low", "High"}
        missing = required - set(prices.columns)
        if missing:
            raise ValueError(f"Price dataframe missing columns: {sorted(missing)}")

        df = prices.copy()
        if not 0 <= trade_start_idx < len(df):
            raise ValueError("trade_start_idx must be within the available price history.")

        signal = self.strategy_signals(df)
        position = self.signal_to_position(df, signal).clip(lower=0.0, upper=1.0).fillna(0.0)
        position.iloc[:trade_start_idx] = 0.0
        if len(position) != len(df):
            raise ValueError("strategy must return a series aligned with prices index.")

        ret = df["Close"].pct_change().fillna(0.0)
        traded_position = position.shift(1).fillna(0.0)

        turnover = position.diff().abs().fillna(position.abs())
        fee_rate = self.fee_bps / 10000.0
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
                trade_r.append(float(trade_ret / self.risk_fraction))

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

    def random_window(self, prices: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        n = len(prices)
        if n < MIN_WINDOW_BARS:
            raise ValueError("Not enough history for random window sampling.")

        length = int(rng.integers(MIN_WINDOW_BARS, min(MAX_WINDOW_BARS, n) + 1))
        start_idx = int(rng.integers(0, n - length + 1))
        end_idx = start_idx + length

        history_plus_window = prices.iloc[:end_idx].copy()
        history_plus_window.attrs["trade_start_idx"] = start_idx
        return history_plus_window

    def collect_metrics(self, samples: list[dict[str, float | list[float]]]) -> dict[str, float]:
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

    def evaluation_report_lines(
        self,
        *,
        time_budget_seconds: float,
        elapsed_seconds: float,
        cycles: int,
        universe_size: int,
        fold_count: int,
        valid_folds: int,
        combined_score: float,
        combined_stats: dict[str, float],
    ) -> list[str]:
        return [
            "Evaluation complete",
            f"time_budget_seconds      : {time_budget_seconds:.1f}",
            f"elapsed_seconds          : {elapsed_seconds:.1f}",
            f"cycles_completed         : {cycles}",
            f"universe_size            : {universe_size}",
            f"combined_folds           : {fold_count}",
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


    def print_metrics(
        self,
        *,
        time_budget_seconds: float,
        elapsed_seconds: float,
        cycles: int,
        universe_size: int,
        fold_count: int,
        valid_folds: int,
        combined_score: float,
        combined_stats: dict[str, float],
    ) -> None:
        for line in self.evaluation_report_lines(
            time_budget_seconds=time_budget_seconds,
            elapsed_seconds=elapsed_seconds,
            cycles=cycles,
            universe_size=universe_size,
            fold_count=fold_count,
            valid_folds=valid_folds,
            combined_score=combined_score,
            combined_stats=combined_stats,
        ):
            print(line)


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
