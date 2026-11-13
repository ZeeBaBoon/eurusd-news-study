"""Build one table of 1-minute EUR/USD bars (UTC) for 2010 -> 2025.

- 2016+: Dukascopy ticks (every bid/ask quote). We aggregate them into 1-minute
  bid and ask bars, plus the widest spread seen in each minute.
- before 2016: HistData 1-minute bars. These are BID ONLY, so spreads are unknown.

Output columns: bid_open/high/low/close, ask_open/high/low/close, spread_max,
n_ticks, source. ask_* and spread_max are NaN for HistData minutes.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

M1_FILE = config.INTERIM / "eurusd_m1.parquet"
M1_TICK_DIR = config.INTERIM / "m1_dukascopy"
_TICK_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.parquet$")
BAR_COLS = [f"{side}_{f}" for side in ("bid", "ask") for f in ("open", "high", "low", "close")]


def ticks_to_m1(ticks: pd.DataFrame) -> pd.DataFrame:
    """Columns in: timestamp (ms since epoch, UTC), bidPrice, askPrice."""
    t = ticks.sort_values("timestamp", kind="stable")
    minute = pd.to_datetime(t.timestamp.to_numpy(), unit="ms", utc=True).floor("min")
    t = pd.DataFrame({"minute": minute, "bid": t.bidPrice.to_numpy(), "ask": t.askPrice.to_numpy()})
    t["spread"] = t.ask - t.bid
    g = t.groupby("minute")
    out = pd.concat({
        "bid_open": g.bid.first(), "bid_high": g.bid.max(), "bid_low": g.bid.min(), "bid_close": g.bid.last(),
        "ask_open": g.ask.first(), "ask_high": g.ask.max(), "ask_low": g.ask.min(), "ask_close": g.ask.last(),
        "spread_max": g.spread.max(), "n_ticks": g.size(),
    }, axis=1)
    out.index = pd.DatetimeIndex(out.index).as_unit("ns")
    out.index.name = "minute"
    return out


def tick_files_needed() -> list[str]:
    """Tick files (each covers ~30 days) that overlap [TICKS_START, END)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(config.TICKS_REPO, repo_type="dataset", revision=config.TICKS_REV)
    keep = []
    for f in files:
        m = _TICK_NAME.search(f)
        if not f.startswith(config.TICKS_DIR + "/") or not m:
            continue
        lo, hi = (pd.Timestamp(x, tz="UTC") for x in m.groups())
        if hi >= config.TICKS_START and lo < config.END:
            keep.append(f)
    return sorted(keep)


def build_dukascopy_m1(keep_raw: bool = False) -> None:
    """Download each tick file once, aggregate to 1-minute bars, cache the result.
    Raw tick files (~2.5 GB in total) are deleted after aggregation unless keep_raw."""
    from huggingface_hub import hf_hub_download

    M1_TICK_DIR.mkdir(parents=True, exist_ok=True)
    files = tick_files_needed()
    for i, f in enumerate(files, 1):
        out = M1_TICK_DIR / Path(f).name
        if out.exists():
            continue
        raw = Path(hf_hub_download(config.TICKS_REPO, f, repo_type="dataset",
                                   revision=config.TICKS_REV, local_dir=config.RAW / "ticks"))
        ticks = pd.read_parquet(raw, columns=["timestamp", "bidPrice", "askPrice"])
        ticks_to_m1(ticks).to_parquet(out)
        if not keep_raw:
            raw.unlink()
        print(f"  ticks {i}/{len(files)} {Path(f).name}: {len(ticks):,} ticks", flush=True)


def load_histdata_raw() -> pd.DataFrame:
    """Full HistData file (bid-only 1-minute bars, UTC), indexed by minute."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(config.HISTDATA_REPO, config.HISTDATA_FILE, repo_type="dataset",
                        revision=config.HISTDATA_REV, local_dir=config.RAW / "histdata")
    h = pd.read_parquet(p)
    h = h.set_index(pd.DatetimeIndex(h.pop("ts")).tz_convert("UTC").as_unit("ns").rename("minute"))
    return h.sort_index()


HISTDATA_DST_SWITCH = pd.Timestamp("2019-01-01", tz="UTC")


def fix_histdata_clock(h: pd.DataFrame) -> pd.DataFrame:
    """The Hugging Face HistData export is 60 minutes LATE during summer time, and
    WHICH summer time changed over the years (verified against Dukascopy ticks in
    the weeks where US and EU clocks disagree, see `qa.histdata_clock_offset`):
      - up to Nov 2018: US daylight-saving calendar (2nd Sun Mar -> 1st Sun Nov)
      - from Mar 2019:  EU summer-time calendar (last Sun Mar -> last Sun Oct)
    Pre-2016 events confirm the US rule: in US-only-summer weeks the news spike sat
    exactly 60 min after the release in 30 of 34 major releases before this fix.
    Clock changes happen on weekends while FX is closed, so the edges are clean."""
    us = np.array([bool(t.dst()) for t in h.index.tz_convert("America/New_York")])
    eu = np.array([bool(t.dst()) for t in h.index.tz_convert("Europe/Berlin")])
    late = np.where(h.index < HISTDATA_DST_SWITCH, us, eu)
    out = h.copy()
    out.index = h.index - pd.to_timedelta(np.where(late, 60, 0), unit="min")
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def build_prices(keep_raw: bool = False) -> pd.DataFrame:
    config.INTERIM.mkdir(parents=True, exist_ok=True)
    build_dukascopy_m1(keep_raw)
    duka = pd.concat([pd.read_parquet(p) for p in sorted(M1_TICK_DIR.glob("*.parquet"))])
    duka = duka[~duka.index.duplicated()].sort_index()
    duka = duka[(duka.index >= config.TICKS_START) & (duka.index < config.END)]
    duka["source"] = "dukascopy"

    h = fix_histdata_clock(load_histdata_raw())
    h = h[(h.index >= config.START) & (h.index < config.TICKS_START)]
    hist = pd.DataFrame(index=h.index, columns=BAR_COLS + ["spread_max", "n_ticks"], dtype=float)
    for f in ("open", "high", "low", "close"):
        hist[f"bid_{f}"] = h[f]
    hist["source"] = "histdata"

    m1 = pd.concat([hist, duka]).sort_index()
    m1["n_ticks"] = m1.n_ticks.astype(float)
    m1.to_parquet(M1_FILE)
    return m1


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(M1_FILE)


def with_mid(m1: pd.DataFrame) -> pd.DataFrame:
    """Mid = (bid + ask) / 2 where ask exists, else bid (HistData)."""
    out = pd.DataFrame(index=m1.index)
    for f in ("open", "high", "low", "close"):
        bid, ask = m1[f"bid_{f}"], m1[f"ask_{f}"]
        out[f"mid_{f}"] = np.where(ask.notna(), (bid + ask) / 2, bid)
    out["spread_close"] = m1.ask_close - m1.bid_close
    out["spread_max"] = m1.spread_max
    out["source"] = m1.source
    return out
