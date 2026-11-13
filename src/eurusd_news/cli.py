"""Command line: `uv run eurusd-news <step>`.

Steps: calendar | prices | events | qa | all
"""

import argparse

import pandas as pd

from . import config
from .align import build_event_table
from .calendar import clean_calendar, download_calendar
from .prices import build_prices, load_prices
from .qa import histdata_clock_offset, summarize

CAL_FILE = config.INTERIM / "calendar_clean.parquet"
EVENTS_FILE = config.PROCESSED / "events.parquet"


def step_calendar():
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    cal = clean_calendar(download_calendar())
    cal.to_parquet(CAL_FILE)
    print(f"calendar: {len(cal):,} releases -> {CAL_FILE}")
    print(cal.time_source.value_counts().to_string())


def step_prices(keep_raw: bool):
    m1 = build_prices(keep_raw)
    print(f"prices: {len(m1):,} minute bars -> {m1.index.min()} .. {m1.index.max()}")
    print(m1.source.value_counts().to_string())


def step_events():
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    ev = build_event_table(pd.read_parquet(CAL_FILE), load_prices())
    ev.to_parquet(EVENTS_FILE)
    ev.to_csv(EVENTS_FILE.with_suffix(".csv"), index=False)
    print(f"events: {len(ev):,} rows -> {EVENTS_FILE} (+ .csv)")


def step_qa():
    print("HistData vs Dukascopy clock check (best lag should be 0):")
    print(histdata_clock_offset().to_string(index=False))
    print()
    print(summarize(pd.read_parquet(EVENTS_FILE)))


def main():
    p = argparse.ArgumentParser(prog="eurusd-news")
    p.add_argument("step", choices=["calendar", "prices", "events", "qa", "all", "baseline", "regimes", "expectations", "ratecycle", "figures"])
    p.add_argument("--keep-raw", action="store_true", help="keep raw tick files (~2.5 GB)")
    a = p.parse_args()
    if a.step in ("calendar", "all"):
        step_calendar()
    if a.step in ("prices", "all"):
        step_prices(a.keep_raw)
    if a.step in ("events", "all"):
        step_events()
    if a.step in ("qa", "all"):
        step_qa()
    if a.step == "baseline":
        from .baseline import run
        print(run())
    if a.step == "regimes":
        from .regimes import run
        print(run())
    if a.step == "expectations":
        from .expectations import run
        print(run())
    if a.step == "ratecycle":
        from .ratecycle import run
        print(run())
    if a.step == "figures":
        from .figures import run
        run()


if __name__ == "__main__":
    main()
