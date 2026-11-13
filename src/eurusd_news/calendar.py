"""Load the ForexFactory calendar snapshot and turn it into a clean release list.

Three repairs happen here (see `assign_times`):
1. Timezone: the scraper wrote Tehran wall-clock times with outdated offsets
   (Iran dropped daylight saving in 2022). We keep the wall-clock and re-apply
   the correct Asia/Tehran rules.
2. Lost times: ~45% of rows say 00:00 because the time was not captured. We take
   the time from another indicator of the same release (e.g. NFP from Average
   Hourly Earnings), else from the official schedule in `catalog.py`.
3. Values like '151K', '0.3%', '<0.25%' become numbers.
"""

import re
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config
from .catalog import BY_FF_NAME, BY_KEY

_VALUE_RE = re.compile(r"^\s*[<>]?\s*(-?\d+(?:\.\d+)?)\s*([KMBT%]?)\s*$")
_MULT = {"": 1.0, "%": 1.0, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

# Surprise is standardised by the spread of *earlier* surprises only (no look-ahead).
SURPRISE_MIN_HISTORY = 8


def download_calendar() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=config.CALENDAR_REPO, filename=config.CALENDAR_FILE,
        repo_type="dataset", revision=config.CALENDAR_REV,
        local_dir=config.RAW / "calendar",
    ))


def parse_value(s) -> float:
    """'151K' -> 151000.0, '0.3%' -> 0.3, '<0.25%' -> 0.25, '' -> nan."""
    if not isinstance(s, str):
        return np.nan
    m = _VALUE_RE.match(s)
    if not m:
        return np.nan
    return float(m.group(1)) * _MULT[m.group(2)]


def _scheduled_utc(key: str, d) -> pd.Timestamp:
    ind = BY_KEY[key]
    local = datetime.combine(d, ind.scheduled_time(d), tzinfo=ZoneInfo(ind.tz))
    return pd.Timestamp(local).tz_convert("UTC")


def assign_times(df: pd.DataFrame) -> pd.DataFrame:
    """Needs columns: wall (naive Tehran wall-clock), indicator, group.
    Adds: release_date, release_utc, time_source, sched_diff_min."""
    df = df.copy()
    missing = df.wall.dt.time == time(0, 0)
    # The Tehran calendar date equals the issuer's local date for every event in
    # our catalog (all are published between 09:30 Frankfurt and 14:15 New York).
    df["release_date"] = df.wall.dt.date

    ff_utc = (df.wall.where(~missing)
              .dt.tz_localize(config.CALENDAR_DISPLAY_TZ, ambiguous="NaT", nonexistent="NaT")
              .dt.tz_convert("UTC"))
    sibling = ff_utc.groupby([df.release_date, df.group]).transform("first")
    sched = pd.Series([_scheduled_utc(k, d) for k, d in zip(df.indicator, df.release_date)],
                      index=df.index, dtype="datetime64[ns, UTC]")

    df["release_utc"] = ff_utc.fillna(sibling).fillna(sched).dt.floor("min")
    df["time_source"] = np.select(
        [ff_utc.notna(), sibling.notna()], ["ff", "ff_sibling"], default="schedule")
    df["sched_diff_min"] = ((df.release_utc - sched).dt.total_seconds() / 60).round().astype(int)
    return df


def add_surprise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("release_utc").copy()
    df["surprise"] = df.actual_v - df.forecast_v
    past_std = df.groupby("indicator").surprise.transform(
        lambda s: s.expanding(min_periods=SURPRISE_MIN_HISTORY).std().shift(1))
    df["surprise_z"] = df.surprise / past_std.replace(0, np.nan)
    return df


def clean_calendar(raw_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, dtype=str)
    keys = [BY_FF_NAME.get((c, e)) for c, e in zip(raw.Currency, raw.Event)]
    raw["indicator"] = [k.key if k else None for k in keys]
    df = raw[raw.indicator.notna()].copy()

    df["wall"] = pd.to_datetime(df.DateTime.str[:19])  # drop the unreliable offset
    df["group"] = df.indicator.map(lambda k: BY_KEY[k].group)
    df["priority"] = df.indicator.map(lambda k: BY_KEY[k].priority)
    df = assign_times(df)
    df = df[(df.release_utc >= config.START) & (df.release_utc < config.END)]

    before = len(df)
    df = df.drop_duplicates(["indicator", "release_date"])
    if len(df) != before:
        print(f"calendar: dropped {before - len(df)} duplicate rows")

    for col in ("Actual", "Forecast", "Previous"):
        df[col.lower() + "_v"] = df[col].map(parse_value)
    df = add_surprise(df)

    tz = df.indicator.map(lambda k: BY_KEY[k].tz)
    df["release_local"] = [t.tz_convert(z).strftime("%Y-%m-%d %H:%M %Z")
                           for t, z in zip(df.release_utc, tz)]
    df["event_id"] = df.indicator + "_" + df.release_date.astype(str)

    out = df.rename(columns={"Currency": "currency", "Event": "ff_name", "Impact": "ff_impact",
                             "Actual": "actual", "Forecast": "forecast", "Previous": "previous"})
    cols = ["event_id", "indicator", "currency", "ff_name", "ff_impact", "group", "priority",
            "release_utc", "release_local", "time_source", "sched_diff_min",
            "actual", "forecast", "previous", "actual_v", "forecast_v", "previous_v",
            "surprise", "surprise_z"]
    return out[cols].sort_values(["release_utc", "priority"]).reset_index(drop=True)
