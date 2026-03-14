# auto-invest

Uses autoresearch-style loops for an investing algorithm.

## Workflow

1. Prepare a full-history universe (many symbols):

```bash
uv run prepare.py
```

This downloads full daily OHLCV history for a basket of symbols and saves
`prices_<SYMBOL>.parquet` files plus `prep_universe.json` under
`~/.cache/auto-invest/prep` by default.

2. Run time-budgeted evaluation:

```bash
uv run train.py
```

`train.py` uses a fixed `TIME_BUDGET_SECONDS` budget and repeatedly samples
random windows from full histories to evaluate:

- Indicators and signals can use all available prior history up to each sampled
  window end (backward-looking only, no future leakage).
- Trading and scoring remain constrained to the sampled random window.

- **Generalist mode**: random basket of symbols and many random windows.
- **Specialist mode**: one specialist symbol (default GLD) across chronological
  regimes, each with random windows.

Scoring remains median fold SQN over sampled windows.
