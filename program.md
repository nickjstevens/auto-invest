# autoresearch

This is an experiment to have the LLM do its own research and optimise a trading strategy.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `auto-invest/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b auto-invest/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, trading strategy, training loop.
4. **Verify data exists**: Check that `~/.cache/auto-invest/prep` contains data. If not, tell the human to run `uv run prepare.py`. The file `prep_universe.json` summarises the data available. 
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single computer. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, trading strategy, hyperparameters, model size, use of indicators (moving averages, RSI, ichimoku, momentum, etc).

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants (time budget, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `score_from_oos_folds` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest combined_score_sqn.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the strategy, the hyperparameters, the model, the indicators, create composite indicators, etc. The only constraint is that the code runs without crashing and finishes within the time budget.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 combined_score_sqn improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 combined_score_sqn improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is (a moving average cross strategy).

## Output format

Once the script finishes it prints a summary like this:

```
---
Evaluation complete
time_budget_seconds.     : 300.0
elapsed_seconds          : 300.0
cycles_completed         : 61
universe_size            : 32
combined_folds           : 23372
valid_folds_for_sqn      : 18714
min_valid_folds_required : 25
combined_score_sqn       : 0.278988
combined_median_cagr     : 0.39%
combined_median_drawdown : -9.73%
combined_median_sharpe   : 0.150
combined_median_win_rate : 38.46%
combined_median_trade_r  : -0.200R
median_trade_open_bars   : 6.0
median_trade_open_days   : 7.0
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^combined_score_sqn:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 10 columns:

```
commit	combined_score_sqn	combined_median_cagr combined_median_drawdown	combined_median_sharpe	combined_median_win_rate	combined_median_trade_r	median_trade_open_bars  status	description
```

1. git commit hash (short, 7 chars)
2. combined_score_sqn achieved (e.g. 1.234567) — use 0.000000 for crashes
3. combined_median_cagr (e.g. 12.3%) — use 0.0 for crashes
4. combined_median_drawdown (e.g. -12.3%) — use 0.0 for crashes
5. combined_median_sharpe (e.g. 1.23) — use 0.0 for crashes
6. combined_median_win_rate (e.g. 12.3%) — use 0.0 for crashes
7. combined_median_trade_r (e.g. -12.3R) — use 0.0 for crashes
8. median_trade_open_bars (e.g. 12) — use 0.0 for crashes
9. status: `keep`, `discard`, or `crash`
10. short text description of what this experiment tried as an overview of the trading strategy

Example:

```
commit	combined_score_sqn	combined_median_cagr combined_median_drawdown	combined_median_sharpe	combined_median_win_rate	combined_median_trade_r	median_trade_open_bars  status	description
a1b2c3d	0.123456	12.3%	-12.3%	1.23	12.3%	-12.3R	12 keep	baseline
b2c3d4e	1.123456	13.3%	-13.3%	1.33	13.3%	-13.3R	23	keep	ichimoku TK cross strategy
c3d4e5f	1.005000	13.3%	-13.3%	1.33	13.3%	-13.3R	23	discard	simple moving average cross strategy
d4e5f6g	0.000000	0.0%	0.0%	0.0	0.0%	0.0R	0.0   crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `auto-invest/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If combined_score_sqn improved (higher), you "advance" the branch, keeping the git commit
9. If combined_score_sqn is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes, use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
