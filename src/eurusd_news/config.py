"""Paths, data sources and study period. Everything that is a 'setting' lives here."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"            # downloads exactly as published (never edited)
INTERIM = DATA / "interim"    # our intermediate files (e.g. 1-minute bars)
PROCESSED = DATA / "processed"  # final tables

# Study period. The calendar snapshot ends 2025-04-07, so that is our real end.
START = pd.Timestamp("2010-01-01", tz="UTC")
END = pd.Timestamp("2025-04-08", tz="UTC")  # exclusive

# Hugging Face datasets, pinned to an exact commit so results are reproducible.
CALENDAR_REPO = "Ehsanrs2/Forex_Factory_Calendar"
CALENDAR_REV = "5cf3d34732b7995b3bc455139495257e64b2e1b4"
CALENDAR_FILE = "forex_factory_cache.csv"
# The scraper ran with ForexFactory's display timezone set to Tehran.
CALENDAR_DISPLAY_TZ = "Asia/Tehran"

# Dukascopy ticks (bid + ask) -> used from 2016 on.
TICKS_REPO = "mito0o852/dukascopy-ticks"
TICKS_REV = "bf19dbd89c732f010e20db7c148922ba02b2e33b"
TICKS_DIR = "data/EURUSD"
TICKS_START = pd.Timestamp("2016-01-01", tz="UTC")

# HistData 1-minute bars (bid only, already converted to UTC) -> used before 2016.
HISTDATA_REPO = "elthariel/histdata_fx_1m"
HISTDATA_REV = "6fd3d8b7cf25f17301e5bace246d10d365e3004c"
HISTDATA_FILE = "eurusd/ticks.parquet"

PIP = 1e-4  # EUR/USD pip
