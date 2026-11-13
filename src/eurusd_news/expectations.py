"""Module 2c: fast entry (~60 s after release) + what the market expected beforehand.

Pre-registered (fixed BEFORE looking at results, 2026-11-13):
Entry   close of the release minute (bar t0, i.e. ~60 s after the release).
Trigger |mid move from just before release to entry| >= 5 pips.
Outcome further mid move by minute 60 in the move's direction (+ continues, - rebounds);
        2016+ also with real costs: follow = buy ask / sell bid (and mirrored), fade = opposite.
Factors (only data from BEFORE the release):
  A drift   mid move over the 4 h before release, |drift| >= 5 pips:
            'priced in'  = drift same direction as the news move  (hypothesis: rebound)
            'against'    = drift opposite to the news move         (hypothesis: continuation)
  B nervous pre_vol_60m of this release vs. median of that indicator's EARLIER releases.
Pass = same sign 2016-21 and 2022+, |t| > 2 over 2016+, |mean| > 2 pips.
2010-2015 (bid only, no costs) is reported as an extra untouched check.
"""

import numpy as np
import pandas as pd

from . import config
from .baseline import SPLIT, START, _stats
from .prices import load_prices
from .regimes import PASS_PIPS, PASS_T, _above_past_median

TRIGGER = 5.0
DRIFT_MIN = 5.0
DRIFT_H = 240
HORIZON = 60
MAX_STALE = pd.Timedelta(minutes=5)
MIN = pd.Timedelta(minutes=1)


def _at(index: pd.DatetimeIndex, values: np.ndarray, t: pd.DatetimeIndex) -> np.ndarray:
    """Close of the last bar that STARTS before t (i.e. the price known at time t), NaN if stale."""
    j = index.searchsorted(t) - 1
    ok = (j >= 0)
    jj = np.clip(j, 0, len(index) - 1)
    stale = (t - index[jj]) > MAX_STALE
    return np.where(ok & ~stale, values[jj], np.nan)


def event_moves(ev: pd.DataFrame, m1: pd.DataFrame, entry_min: int = 1) -> pd.DataFrame:
    idx = m1.index
    bid = m1.bid_close.to_numpy()
    ask = m1.ask_close.to_numpy()
    mid = np.where(np.isnan(ask), bid, (bid + ask) / 2)
    t0 = pd.DatetimeIndex(ev.release_utc)
    te, th = t0 + entry_min * MIN, t0 + HORIZON * MIN
    p0, pe, ph = (_at(idx, mid, t) for t in (t0, te, th))
    pd_ = _at(idx, mid, t0 - DRIFT_H * MIN)
    move = (pe - p0) / config.PIP
    s = np.sign(move)
    out = pd.DataFrame({"time": t0, "indicator": ev.indicator.to_numpy(), "move": move,
                        "after": (ph - pe) / config.PIP * s,
                        "drift": (p0 - pd_) / config.PIP}, index=ev.index)
    be, ae, bh, ah = _at(idx, bid, te), _at(idx, ask, te), _at(idx, bid, th), _at(idx, ask, th)
    long_follow = np.where(s > 0, bh - ae, be - ah) / config.PIP     # follow the move, pay spread
    long_fade = np.where(s > 0, be - ah, bh - ae) / config.PIP       # bet on rebound, pay spread
    out["pnl_follow"], out["pnl_fade"] = long_follow, long_fade
    out["spread_entry"] = (ae - be) / config.PIP
    return out


def run() -> str:
    ev = pd.read_parquet(config.PROCESSED / "events.parquet")
    ev = ev[ev.is_primary].sort_values("release_utc")
    ev["nervous"] = np.nan
    for _, g in ev.groupby("indicator"):
        ev.loc[g.index, "nervous"] = _above_past_median(g.pre_vol_60m_pips, 8)

    c = event_moves(ev, load_prices())
    c["nervous"] = ev.nervous
    c = c[np.isfinite(c.after) & (c.move.abs() >= TRIGGER)]
    aligned = np.sign(c.drift) == np.sign(c.move)
    c["drift_grp"] = np.where(c.drift.abs() < DRIFT_MIN, "flat",
                              np.where(aligned, "priced in", "against"))
    c.loc[c.drift.isna(), "drift_grp"] = np.nan

    old, new = c[c.time < START], c[c.time >= START]
    tr, te = new.time < SPLIT, new.time >= SPLIT

    def row(name, g_old, g_new, m_tr, m_te, cost=True):
        o, a, b, al = _stats(g_old.after), _stats(g_new.after[m_tr]), _stats(g_new.after[m_te]), _stats(g_new.after)
        passed = (np.sign(a["mean"]) == np.sign(b["mean"]) and abs(al["t"]) > PASS_T
                  and abs(al["mean"]) > PASS_PIPS)
        fol, fad = g_new.pnl_follow.mean(), g_new.pnl_fade.mean()
        return (f"{name:18s} {o['n']:5d} {o['mean']:+6.1f} | {a['n']:5d} {a['mean']:+6.1f} "
                f"{b['n']:5d} {b['mean']:+6.1f} | {al['mean']:+6.1f} {al['t']:+5.1f} | "
                f"{fol:+6.1f} {fad:+6.1f}  {'PASS' if passed else 'fail'}")

    hdr = (f"{'':18s} {'2010-15':>12s} | {'2016-21':>12s} {'2022+':>12s} | {'2016+ mean':>6s} {'t':>5s} | "
           f"{'follow':>6s} {'fade':>6s}\n{'':18s} {'n':>5s} {'mean':>6s} | {'n':>5s} {'mean':>6s} "
           f"{'n':>5s} {'mean':>6s} | {'':>6s} {'':>5s} | {'(net)':>6s} {'(net)':>6s}")
    lines = [f"entry ~60 s after release, trigger >= {TRIGGER:g} pips in the release minute, "
             f"outcome = further move by minute {HORIZON} (+ continue, - rebound)",
             f"median spread at entry (2016+): {new.spread_entry.median():.1f} pips, "
             f"90th pct {new.spread_entry.quantile(.9):.1f}\n", hdr,
             row("ALL", old, new, tr, te), ""]
    for col, vals in (("drift_grp", ("priced in", "against", "flat")), ("nervous", (0.0, 1.0))):
        for v in vals:
            name = {0.0: "calm before", 1.0: "nervous before"}.get(v, v)
            go, gn = old[old[col] == v], new[new[col] == v]
            lines.append(row(name, go, gn, gn.time < SPLIT, gn.time >= SPLIT)
                         + ("   (not a test)" if v == "flat" else ""))
        lines.append("")
    c.to_csv(config.PROCESSED / "expectations_events.csv", index=False)
    return "\n".join(lines)
