"""Attach what the price did around each release -> the final event table.

Conventions (all times UTC, 1-minute bars labelled by their START minute):
- t0 = release minute. Bar t0 covers [t0, t0+1min) = the first minute after news.
- p0 = last mid close BEFORE t0 (the price just before the release).
- ret_{h}m = mid close at the end of minute t0+h-1, minus p0, in pips.
- hi_60m / lo_60m = highest high / lowest low in the first 60 minutes, minus p0.
- pre_vol_60m = std of 1-minute mid changes in the hour before t0 (pips).

Time check: a correctly timed release usually shows the biggest 1-minute move of
the surrounding hour in bar t0 or t0+1. Minutes of OTHER known releases are ignored.
- ok:      release-minute move >= 3x normal noise and the dominant move nearby
- suspect: almost nothing at t0 (< 2x noise) but a big spike (>= 6x) elsewhere within
           +-30 min or exactly +-60 min -> the time is probably wrong; check by hand
- unclear: no clear reaction. Time may be right; the market just didn't care much.
"""

import numpy as np
import pandas as pd

from . import config
from .prices import with_mid

HORIZONS = (1, 5, 15, 30, 60, 240)
PRE_WINDOW = 60
CHECK_WINDOW = 30
SPIKE_MULT = 3.0        # 'ok' needs a release-minute move >= 3x normal noise
SUSPECT_T0_MAX = 2.0    # 'suspect': release minute < 2x noise ...
SUSPECT_PEAK_MIN = 6.0  # ... while another minute nearby is >= 6x noise
MAX_STALE_MIN = 5
MIN = pd.Timedelta(minutes=1)


def _window_features(mid: pd.DataFrame, t0: pd.Timestamp, other_releases=()) -> dict:
    w = mid.loc[t0 - PRE_WINDOW * MIN: t0 + max(HORIZONS) * MIN]
    nan = {"p0": np.nan}
    pre = w[w.index < t0]
    if pre.empty or t0 - pre.index[-1] > MAX_STALE_MIN * MIN:
        return nan
    p0 = pre.mid_close.iloc[-1]
    post = w[w.index >= t0]
    f = {"p0": p0, "price_source": pre.source.iloc[-1],
         "minutes_since_last_bar": (t0 - pre.index[-1]) / MIN - 1}

    for h in HORIZONS:
        upto = post[post.index <= t0 + (h - 1) * MIN]
        f[f"ret_{h}m_pips"] = (upto.mid_close.iloc[-1] - p0) / config.PIP if len(upto) else np.nan

    first_hour = post[post.index < t0 + 60 * MIN]
    f["hi_60m_pips"] = (first_hour.mid_high.max() - p0) / config.PIP if len(first_hour) else np.nan
    f["lo_60m_pips"] = (first_hour.mid_low.min() - p0) / config.PIP if len(first_hour) else np.nan
    f["n_bars_60m"] = len(first_hour)

    d = (w.mid_close.diff() / config.PIP)
    pre_vol = d[w.index < t0].iloc[1:].std()
    f["pre_vol_60m_pips"] = pre_vol

    near = d[(d.index >= t0 - CHECK_WINDOW * MIN) & (d.index <= t0 + CHECK_WINDOW * MIN)].abs()
    # Ignore minutes that belong to OTHER known releases (e.g. German PMI 30 min
    # before the euro area PMI), otherwise their move looks like a timing error.
    for t in other_releases:
        near = near[(near.index < t) | (near.index > t + MIN)]
    at_t0 = near[(near.index >= t0) & (near.index <= t0 + MIN)]
    r0 = at_t0.max() if len(at_t0) else np.nan
    rp = near.max() if len(near) else np.nan
    # Also look exactly +-60 min away: the typical daylight-saving mistake.
    dst = d[d.index.isin([t0 - 60 * MIN, t0 - 59 * MIN, t0 + 60 * MIN, t0 + 61 * MIN])].abs()
    r_dst = dst.max() if len(dst) else np.nan
    if len(dst) and r_dst > rp:
        rp, peak_at = r_dst, dst.idxmax()
    else:
        peak_at = near.idxmax() if len(near) else None
    f["peak_offset_min"] = (peak_at - t0) / MIN if peak_at is not None else np.nan
    ok_vol = bool(pre_vol and pre_vol > 0)
    f["spike_ratio"] = r0 / pre_vol if ok_vol else np.nan
    f["peak_ratio"] = rp / pre_vol if ok_vol else np.nan
    if ok_vol and r0 >= SPIKE_MULT * pre_vol and r0 >= 0.5 * rp:
        f["time_check"] = "ok"
    elif ok_vol and r0 < SUSPECT_T0_MAX * pre_vol and rp >= SUSPECT_PEAK_MIN * pre_vol:
        f["time_check"] = "suspect"
    else:
        f["time_check"] = "unclear"

    f["spread_pre_pips"] = pre.spread_close.iloc[-1] / config.PIP
    first5 = post[post.index < t0 + 5 * MIN]
    f["spread_max_5m_pips"] = first5.spread_max.max() / config.PIP if len(first5) else np.nan
    return f


def build_event_table(cal: pd.DataFrame, m1: pd.DataFrame) -> pd.DataFrame:
    mid = with_mid(m1)
    clusters = cal.release_utc.drop_duplicates().sort_values()
    win = CHECK_WINDOW * MIN

    def others(t):
        return [u for u in clusters[(clusters >= t - win) & (clusters <= t + win)] if u != t]

    feats = pd.DataFrame([{"release_utc": t, **_window_features(mid, t, others(t))} for t in clusters])

    ev = cal.copy()
    # Several indicators published in the same minute = one market event (cluster).
    ev["cluster_id"] = ev.release_utc.dt.strftime("%Y%m%dT%H%MZ")
    ev["n_in_cluster"] = ev.groupby("release_utc").indicator.transform("size")
    rank = ev.sort_values(["release_utc", "priority", "currency"]).groupby("release_utc").cumcount()
    ev["is_primary"] = rank.reindex(ev.index).eq(0)
    # Other catalog releases within +-30 minutes (e.g. ECB decision -> press conference).
    times = clusters.to_numpy()
    lo = np.searchsorted(times, (ev.release_utc - 30 * MIN).to_numpy(), side="left")
    hi = np.searchsorted(times, (ev.release_utc + 30 * MIN).to_numpy(), side="right")
    ev["n_other_releases_30m"] = hi - lo - 1
    ev["utc_hour"] = ev.release_utc.dt.hour + ev.release_utc.dt.minute / 60
    ev["weekday"] = ev.release_utc.dt.day_name().str[:3]

    return ev.merge(feats, on="release_utc", how="left")
