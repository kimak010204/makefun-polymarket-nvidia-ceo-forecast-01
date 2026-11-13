#!/usr/bin/env python3
"""Forecast Polymarket's next NVIDIA CEO prices (make.fun bounty
polymarket-nvidia-ceo-forecast-01).

Reads each case in the input directory (one case per file <id>.in) and writes
one forecast file <id>.out per case to the output directory.

Method (tuned and validated on the six practice cases, local backtest):
  - Median: persistence -- the price stays where it was at the checkpoint.
    (Every drift/mean-reversion variant tested made the practice loss worse.)
  - Width: for each horizon h, the empirical quantiles of past h-step price
    changes over the last 8 hours (480 minutes), recentered so their median
    is 0, blended 85/15 with the house baseline's normal quantiles
    (sigma 0.0025 per sqrt-minute). The empirical distribution captures the
    heavy tails, tick bounce and mean reversion the normal random walk misses.
  - Conditional vol: all widths are scaled by (recent 2h 10-minute realized
    vol / lookback 10-minute realized vol) ** 0.5, clipped to [0.7, 1.6], so
    the forecast widens within minutes when the market gets jumpy.
  - Short history (<60 prices): fall back to the house baseline.
  - No network, no randomness, standard library only, deterministic:
    two runs on the same case write identical bytes.

Local backtest on the six practice cases (actuals fetched from Polymarket's
public price history, official pinball loss):
  house baseline 0.363589 | this program 0.283753 (beats baseline on all
  six cases; median error and 90% coverage also reported in BUILD_LOG.md).

Usage (local): mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
In the grader: python3 /bot/main.py reads /input, writes /output.
"""

import bisect
import json
import math
import os
from pathlib import Path

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# --- tuned parameters (see BUILD_LOG.md for the validation) ---
LOOKBACK_MIN = 480       # empirical h-step changes from the last 8 hours
W_EMP = 0.85             # weight of empirical deltas vs baseline normal deltas
BASE_SIGMA = 0.0025      # house baseline spread per sqrt-minute
GAMMA = 0.5              # conditional-vol exponent
VOL_CLIP = (0.7, 1.6)    # conditional-vol scaling limits
GRID_H = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 180, 240]
MIN_DIFFS = 30           # minimum overlapping samples for a grid horizon


def _clamp6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = min(1.0, max(0.0, x))
    return math.floor(v * 1e6 + 0.5) / 1e6 + 0.0


def _empirical_quantiles(vals, levels):
    """Linear-interpolation empirical quantiles of a list of numbers."""
    s = sorted(vals)
    n = len(s)
    out = []
    for lv in levels:
        if n == 1:
            out.append(s[0])
            continue
        pos = lv * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(s[lo] * (1.0 - frac) + s[hi] * frac)
    return out


def _baseline_rows(x, horizon):
    rows = []
    for h in range(1, horizon + 1):
        s = BASE_SIGMA * math.sqrt(h)
        rows.append([_clamp6(x + z * s) for z in Z])
    return rows


def _hstep_grid(prices, lookback):
    """qgrid[h] = recentered empirical quantiles of h-step changes.

    Uses overlapping h-step diffs from the last `lookback` minutes.
    Recentered so the median delta is 0 (persistence median).
    """
    n = len(prices)
    seg = prices[max(0, n - lookback):]
    m = len(seg)
    qgrid = {}
    for h in GRID_H:
        if m - h < MIN_DIFFS:
            continue
        diffs = [seg[i] - seg[i - h] for i in range(h, m)]
        qs = _empirical_quantiles(diffs, QUANTILES)
        med = qs[2]
        qgrid[h] = [v - med for v in qs]
    return qgrid


def _interp_delta(qgrid, h):
    """Interpolate the recentered quantile deltas at horizon h (in sqrt(h))."""
    hs = sorted(qgrid)
    if h <= hs[0]:
        return qgrid[hs[0]]
    if h >= hs[-1]:
        return qgrid[hs[-1]]
    j = bisect.bisect_left(hs, h)
    h0, h1 = hs[j - 1], hs[j]
    w = (math.sqrt(h) - math.sqrt(h0)) / (math.sqrt(h1) - math.sqrt(h0))
    d0, d1 = qgrid[h0], qgrid[h1]
    return [d0[k] * (1.0 - w) + d1[k] * w for k in range(5)]


def _rms_diffs(step, seg):
    ds = [seg[i] - seg[i - step] for i in range(step, len(seg))]
    if not ds:
        return 0.0
    return math.sqrt(sum(d * d for d in ds) / len(ds))


def _cond_vol_factor(prices):
    """Recent (2h) vs lookback 10-minute realized-vol ratio, ^ GAMMA, clipped."""
    n = len(prices)
    if n < 240:
        return 1.0
    s_rec = _rms_diffs(10, prices[-121:])
    s_lb = _rms_diffs(10, prices[max(0, n - LOOKBACK_MIN):])
    if s_lb <= 0.0 or s_rec <= 0.0:
        return 1.0
    f = (s_rec / s_lb) ** GAMMA
    return min(VOL_CLIP[1], max(VOL_CLIP[0], f))


def forecast_market(prices, horizon):
    """Forecast rows (horizon x 5 quantiles) for one market's price list."""
    if not prices:
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
    x = prices[-1]
    if len(prices) < 60:
        return _baseline_rows(x, horizon)
    qgrid = _hstep_grid(prices, LOOKBACK_MIN)
    if not qgrid:
        return _baseline_rows(x, horizon)
    factor = _cond_vol_factor(prices)
    rows = []
    for h in range(1, horizon + 1):
        d = _interp_delta(qgrid, h)
        base = [z * BASE_SIGMA * math.sqrt(h) for z in Z]
        row = []
        for k in range(5):
            delta = (W_EMP * d[k] + (1.0 - W_EMP) * base[k]) * factor
            row.append(_clamp6(x + delta))
        for k in range(1, 5):
            if row[k] < row[k - 1]:
                row[k] = row[k - 1]
        rows.append(row)
    return rows


def main():
    in_dir = Path(INPUT_DIR)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in in_dir.iterdir()
                    if p.is_file() and p.name.endswith(".in"))
    for path in inputs:
        case_id = path.name[:-3]
        with open(path, "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = int(case.get("horizon_minutes", 240))
        forecasts = []
        for m in case["markets"]:
            rows = forecast_market(m["prices"], horizon)
            forecasts.append({"id": m["id"], "q": rows})
        out_path = out_dir / (case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
