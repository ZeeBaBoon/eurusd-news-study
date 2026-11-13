"""Sanity checks we run before trusting the data."""

import numpy as np
import pandas as pd

from .prices import M1_TICK_DIR, fix_histdata_clock, load_histdata_raw

# Periods where HistData overlaps Dukascopy. The short ones are the weeks where US
# and EU clocks disagree: a DST bug only shows up there, not in Jan or Jul.
PERIODS = [("2016-01-04", "2016-01-29"), ("2016-03-14", "2016-03-25"), ("2016-07-04", "2016-07-29"),
           ("2016-10-31", "2016-11-04"), ("2018-03-12", "2018-03-23"), ("2018-10-29", "2018-11-02"),
           ("2019-03-11", "2019-03-29"), ("2019-07-01", "2019-07-26"), ("2019-10-28", "2019-11-01"),
           ("2022-01-03", "2022-01-28"), ("2024-07-01", "2024-07-26")]


def histdata_clock_offset(max_lag: int = 120, fixed: bool = True) -> pd.DataFrame:
    """Compare HistData bars with Dukascopy bars in months where both exist.
    If HistData's UTC timestamps are right, 1-minute returns line up at lag 0.
    (fixed=False shows the raw export's +60 min error in EU summer months.)"""
    h = load_histdata_raw()
    if fixed:
        h = fix_histdata_clock(h)
    duka = pd.concat([pd.read_parquet(p, columns=["bid_close"]) for p in sorted(M1_TICK_DIR.glob("*.parquet"))])
    duka = duka[~duka.index.duplicated()].sort_index()
    rows = []
    for lo, hi in PERIODS:
        a = h.close[lo:hi]
        b = duka.bid_close[lo:hi]
        if a.empty or b.empty:
            continue
        m = f"{lo}..{hi[5:]}"
        grid = pd.date_range(min(a.index[0], b.index[0]), max(a.index[-1], b.index[-1]), freq="min")
        ra = a.reindex(grid).ffill().diff()
        rb = b.reindex(grid).ffill().diff()
        corr = {lag: ra.corr(rb.shift(lag)) for lag in range(-max_lag, max_lag + 1)}
        best = max(corr, key=lambda k: corr[k] if np.isfinite(corr[k]) else -1)
        both = a.index.intersection(b.index)
        rows.append({"period": m, "best_lag_min": best, "corr_at_best": round(corr[best], 3),
                     "corr_at_0": round(corr[0], 3),
                     "median_abs_price_diff_pips": round(((a[both] - b[both]).abs().median()) / 1e-4, 2)})
    return pd.DataFrame(rows)


def summarize(ev: pd.DataFrame) -> str:
    out = []
    prim = ev[ev.is_primary]
    out.append(f"rows (indicator releases): {len(ev):,} | market events (clusters): {prim.cluster_id.nunique():,}")
    out.append(f"period: {ev.release_utc.min():%Y-%m-%d} -> {ev.release_utc.max():%Y-%m-%d}")
    out.append("\ntime source:\n" + ev.time_source.value_counts().to_string())
    out.append("\ntime check (primary rows):\n" + prim.time_check.value_counts(dropna=False).to_string())
    out.append("\nby indicator:\n" + ev.groupby("indicator").agg(
        n=("event_id", "size"),
        has_surprise=("surprise", lambda s: round(s.notna().mean(), 2)),
        ok=("time_check", lambda s: round((s == "ok").mean(), 2)),
        suspect=("time_check", lambda s: int((s == "suspect").sum())),
        med_abs_5m_pips=("ret_5m_pips", lambda s: round(s.abs().median(), 1)),
    ).to_string())
    return "\n".join(out)
