"""Module 2b: does the news reaction depend on the market regime?

Pre-registered (fixed BEFORE looking at results, 2026-11-13):
Three regime measures, each known at trade time (only data from BEFORE the release):
  vol          last 20 days' realised volatility vs. its own history so far -> high / low
  trend        20-day efficiency ratio |net move| / sum|daily moves| vs. history -> trending / ranging
  sensitivity  how much this indicator moved EUR/USD per unit of surprise over its
               last 10 releases, vs. that indicator's own history -> high / low
Outcome: the Module 2 continuation measure (>=10-pip move in first 5 min, further
move by minute 60; + continues, - rebounds). No rule grid.
A regime bucket PASSES only if: same sign in train (2016-21) and test (2022+),
|t| > 2 over all years, and |mean| > 2 pips (bigger than costs).
"""

import numpy as np
import pandas as pd

from . import config
from .baseline import SPLIT, START, _stats, build_windows, continuation
from .prices import load_prices

VOL_DAYS = 20
TREND_DAYS = 20
SENS_RELEASES = 10
MIN_HISTORY_DAYS = 250
PASS_T = 2.0
PASS_PIPS = 2.0


def daily_closes(m1: pd.DataFrame) -> pd.Series:
    """Last mid (bid if no ask) of each UTC day with trading."""
    mid = m1.bid_close.where(m1.ask_close.isna(), (m1.bid_close + m1.ask_close) / 2)
    d = mid.resample("1D").last().dropna()
    return d[d.index.dayofweek < 5]


def _above_past_median(x: pd.Series, min_periods: int) -> pd.Series:
    """1 if today's value > median of all EARLIER values, 0 if not, NaN if too little history."""
    past_med = x.expanding(min_periods=min_periods).median().shift(1)
    return pd.Series(np.where(past_med.isna() | x.isna(), np.nan, (x > past_med).astype(float)), index=x.index)


def market_regimes(closes: pd.Series) -> pd.DataFrame:
    """Per day D: regime known at the START of day D (uses closes up to D-1 only)."""
    r = np.log(closes).diff()
    vol = r.rolling(VOL_DAYS).std()
    net = (closes - closes.shift(TREND_DAYS)).abs()
    path = closes.diff().abs().rolling(TREND_DAYS).sum()
    eff = net / path
    out = pd.DataFrame({"vol_high": _above_past_median(vol, MIN_HISTORY_DAYS),
                        "trending": _above_past_median(eff, MIN_HISTORY_DAYS)})
    out.index = out.index + pd.Timedelta("1D")   # known from the next day on
    return out


def sensitivity(ev: pd.DataFrame) -> pd.Series:
    """Per release row: sum|5-min reaction| / sum|surprise_z| over the indicator's
    previous SENS_RELEASES releases (not including this one), then high/low vs. that
    indicator's own earlier values."""
    ev = ev.sort_values("release_utc")
    out = pd.Series(np.nan, index=ev.index)
    for _, g in ev.groupby("indicator"):
        ok = g.surprise_z.notna() & g.ret_5m_pips.notna()
        react = g.ret_5m_pips.abs().where(ok)
        surpr = g.surprise_z.abs().where(ok)
        s = (react.rolling(SENS_RELEASES, min_periods=SENS_RELEASES).sum()
             / surpr.rolling(SENS_RELEASES, min_periods=SENS_RELEASES).sum()).shift(1)
        out[g.index] = _above_past_median(s, SENS_RELEASES)
    return out


def run() -> str:
    ev_all = pd.read_parquet(config.PROCESSED / "events.parquet")
    ev_all["sens_high"] = sensitivity(ev_all)          # history from 2010 on
    ev = ev_all[ev_all.is_primary & (ev_all.release_utc >= START)].sort_values("release_utc")

    m1 = load_prices()
    reg = market_regimes(daily_closes(m1))
    day = pd.DatetimeIndex(ev.release_utc.dt.normalize())
    # latest regime known on or before the release date (Monday uses Friday's close)
    ev = ev.join(reg.reindex(day, method="ffill").set_index(ev.index))

    m1 = m1[m1.index >= START - pd.Timedelta("1D")]
    t = pd.DatetimeIndex(ev.release_utc)
    W, p0 = build_windows(m1, t)
    c = continuation(W, p0, t).merge(ev[["release_utc", "vol_high", "trending", "sens_high"]],
                                     left_on="time", right_on="release_utc")

    lines = [f"big-move news events (>=10 pips in 5 min), 2016+: {len(c)}",
             f"pass rule: same sign train & test, |t| > {PASS_T}, |mean| > {PASS_PIPS} pips\n",
             f"{'regime':22s} {'train n':>7s} {'mean':>6s} {'test n':>7s} {'mean':>6s} "
             f"{'all n':>6s} {'mean':>6s} {'t':>5s}  verdict"]
    rows = []
    for col, names in (("vol_high", ("low vol", "high vol")), ("trending", ("ranging", "trending")),
                       ("sens_high", ("low sensitivity", "high sensitivity"))):
        for val, name in enumerate(names):
            g = c[c[col] == val]
            tr, te, al = (_stats(g.after[m]) for m in (g.time < SPLIT, g.time >= SPLIT, g.time.notna()))
            passed = (np.sign(tr["mean"]) == np.sign(te["mean"]) and abs(al["t"]) > PASS_T
                      and abs(al["mean"]) > PASS_PIPS)
            rows.append({"regime": name, "train_n": tr["n"], "train_mean": tr["mean"], "test_n": te["n"],
                         "test_mean": te["mean"], "all_n": al["n"], "all_mean": al["mean"], "all_t": al["t"],
                         "pass": passed})
            lines.append(f"{name:22s} {tr['n']:7d} {tr['mean']:+6.1f} {te['n']:7d} {te['mean']:+6.1f} "
                         f"{al['n']:6d} {al['mean']:+6.1f} {al['t']:+5.1f}  {'PASS' if passed else 'fail'}")
        lines.append("")
    lines.append(f"unknown regime (too little history): vol {c.vol_high.isna().sum()}, "
                 f"trend {c.trending.isna().sum()}, sensitivity {c.sens_high.isna().sum()}")
    pd.DataFrame(rows).to_csv(config.PROCESSED / "regimes_result.csv", index=False)
    return "\n".join(lines)
