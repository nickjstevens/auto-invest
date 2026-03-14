# auto-invest

Uses autoresearch-style loops for an investing algorithm. 

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. This is modified for trading strategies. The core idea is that you're not touching any of the Python files like you normally would as a trading strategy backtester. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. 

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data for a basket of symbols), and runtime utilities (dataloader, evaluation). Not modified. This downloads full daily OHLCV history for a basket of symbols and saves `prices_<SYMBOL>.parquet` files plus `prep_universe.json` under `~/.cache/auto-invest/prep` by default.
- **`train.py`** — the single file the agent edits. Contains the trading strategy, and training loop. Everything is fair game: architecture, hyperparameters, etc. **This file is edited and iterated on by the agent**. Uses a fixed `TIME_BUDGET_SECONDS` budget and repeatedly samples random windows from full histories to evaluate against the scoring metric (median fold SQN over sampled windows). Indicators and signals can use all available prior history up to each sampled window end (backward-looking only, no future leakage). Trading and scoring remain constrained to the sampled random window.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock), regardless of the details of your compute. The metric is **sqn** — higher is better.

## Quick start

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond a few small packages. One file, one metric.


## License

MIT