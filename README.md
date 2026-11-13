# Polymarket NVIDIA CEO price forecaster

Entry for make.fun bounty `polymarket-nvidia-ceo-forecast-01`: forecast the
Yes price of every trading market of Polymarket's "Next CEO of NVIDIA?"
event at every minute of the 4 hours after each checkpoint, as five
quantiles (5%, 25%, 50%, 75%, 95%), scored by mean pinball loss.

## Method

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads each case in `/input` (`$MAKEFUN_INPUT_DIR` locally) and writes one
`<id>.out` per case to `/output` (`$MAKEFUN_OUTPUT_DIR` locally).

- **Median: persistence.** The price stays where it was at the checkpoint.
  Every drift / mean-reversion variant tested made the practice loss worse.
- **Width: empirical h-step quantiles.** For each horizon h, the quantiles
  of past h-step price changes over the last 8 hours (480 minutes),
  recentered so their median is 0, blended 85/15 with the house baseline's
  normal quantiles (sigma 0.0025 per sqrt-minute). The empirical distribution
  captures the heavy tails, tick bounce and mean reversion that a normal
  random walk misses.
- **Conditional volatility.** All widths are scaled by (recent 2h 10-minute
  realized vol / lookback 10-minute realized vol) ** 0.5, clipped to
  [0.7, 1.6], so the forecast widens within minutes when the market gets
  jumpy.
- Short history (<60 prices) falls back to the house baseline.

## Validation (local backtest, six practice cases)

Actual prices for the practice horizons were fetched from Polymarket's
public price history (`clob.polymarket.com/prices-history`, fidelity=1)
and scored with the official pinball loss. The harness reproduces the
house baseline's published practice loss exactly (0.363589).

| model | pooled pinball loss |
|---|---|
| house baseline | 0.363589 |
| **this program** | **0.285119** |

It beats the baseline on all six practice cases (p00 0.491 vs 0.536,
p01 0.312 vs 0.380, p02 0.372 vs 0.408, p03 0.282 vs 0.369,
p04 0.199 vs 0.299, p05 0.121 vs 0.238). 90% coverage 0.85–0.97.

Runtime: ~0.08 s per case locally (limit 8 s). Two runs on the same case
write byte-identical output. Output files ~180–240 KB (limit 1 MiB).

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
