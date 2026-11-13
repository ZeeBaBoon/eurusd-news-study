"""Module 2d: does the central-bank cycle tell us WHEN to follow and when to fade?

Pre-registered (fixed BEFORE looking at results, 2026-11-13):
Active  the Fed or the ECB changed its policy rate in the 182 days BEFORE the release day
        (rate changes taken from our own calendar: actual != previous for the Fed funds rate
        and the ECB main refinancing rate).
Measure entry 5 min after release, move >= 10 pips, further move by minute 60
        (+ continues, - rebounds). Costs: real bid/ask (2016+).
Prediction  active -> moves continue (follow); hold -> moves rebound (fade).
Pass    active mean > 0 AND hold mean < 0 in BOTH 2016-21 and 2022+,
        t of (active - hold) > 2 over 2016+, and the switching strategy
        (follow if active, fade if hold) > 2 pips per trade after costs over 2016+.
2010-2015 (bid only, no costs) is reported as an extra untouched check.
"""

import numpy as np
import pandas as pd

from . import config
from .baseline import SPLIT, START, _stats
from .expectations import event_moves
from .prices import load_prices

LOOKBACK = pd.Timedelta(days=182)
ENTRY_MIN = 5
TRIGGER = 10.0
PASS_T = 2.0
PASS_PIPS = 2.0


def rate_change_dates(cal: pd.DataFrame) -> pd.DatetimeIndex:
    r = cal[cal.indicator.isin(["us_fed_rate", "ea_ecb_rate"]) & (cal.actual != cal.previous)]
    return pd.DatetimeIndex(r.release_utc.dt.normalize()).sort_values()


def cycle_active(days: pd.DatetimeIndex, changes: pd.DatetimeIndex) -> np.ndarray:
    """True if a rate change happened on a day in [day - LOOKBACK, day) -- strictly before."""
    hi = changes.searchsorted(days, side="left")              # changes before this day
    lo = changes.searchsorted(days - LOOKBACK, side="left")
    return hi > lo


def _diff_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return (a.mean() - b.mean()) / se


def run() -> str:
    cal = pd.read_parquet(config.INTERIM / "calendar_clean.parquet")
    ev = pd.read_parquet(config.PROCESSED / "events.parquet")
    ev = ev[ev.is_primary].sort_values("release_utc")
    c = event_moves(ev, load_prices(), entry_min=ENTRY_MIN)
    c = c[np.isfinite(c.after) & (c.move.abs() >= TRIGGER)].copy()
    c["active"] = cycle_active(pd.DatetimeIndex(c.time.dt.normalize()), rate_change_dates(cal))
    c["strategy"] = np.where(c.active, c.pnl_follow, c.pnl_fade)

    periods = {"2010-15": c.time < START, "2016-21": (c.time >= START) & (c.time < SPLIT),
               "2022+": c.time >= SPLIT, "2016+": c.time >= START}
    lines = [f"entry {ENTRY_MIN} min after release, move >= {TRIGGER:g} pips, further move by minute 60 "
             "(+ continue, - rebound; mid, no costs)\n",
             f"{'':8s} {'active n':>8s} {'mean':>6s} {'hold n':>7s} {'mean':>6s} {'diff t':>7s} "
             f"{'switch strategy (net)':>22s}"]
    res = {}
    for name, m in periods.items():
        g = c[m]
        a, h = _stats(g.after[g.active]), _stats(g.after[~g.active])
        s = _stats(g.strategy)
        res[name] = (a, h, _diff_t(g.after[g.active], g.after[~g.active]), s)
        strat = f"{s['mean']:+6.1f} (n={s['n']}, t={s['t']:+.1f})" if s["n"] else "n/a (no ask before 2016)"
        lines.append(f"{name:8s} {a['n']:8d} {a['mean']:+6.1f} {h['n']:7d} {h['mean']:+6.1f} "
                     f"{res[name][2]:+7.1f}   {strat}")
    ok_signs = all(res[p][0]["mean"] > 0 and res[p][1]["mean"] < 0 for p in ("2016-21", "2022+"))
    passed = ok_signs and res["2016+"][2] > PASS_T and res["2016+"][3]["mean"] > PASS_PIPS
    lines += ["", f"predicted signs in both periods: {ok_signs} | "
              f"diff t 2016+: {res['2016+'][2]:+.1f} | switch net 2016+: {res['2016+'][3]['mean']:+.1f} pips",
              f"VERDICT: {'PASS' if passed else 'fail'}",
              f"share of events in 'active': " + ", ".join(
                  f"{p} {c.active[m].mean():.0%}" for p, m in periods.items() if p != "2016+")]
    c.to_csv(config.PROCESSED / "ratecycle_events.csv", index=False)
    return "\n".join(lines)
