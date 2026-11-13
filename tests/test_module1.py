import numpy as np
import pandas as pd
import pytest

from eurusd_news.align import build_event_table
from eurusd_news.calendar import add_surprise, assign_times, parse_value
from eurusd_news.catalog import BY_KEY
from eurusd_news.prices import fix_histdata_clock, ticks_to_m1


def test_histdata_clock_fix_us_rule_before_2019_eu_rule_after():
    idx = pd.DatetimeIndex([
        "2015-03-06 14:00", "2015-03-12 14:00",   # US clocks changed 8 Mar 2015 (EU: 29 Mar)
        "2015-10-28 14:00", "2015-11-03 14:00",   # US-only summer week, then US back to winter
        "2019-03-15 14:00", "2019-04-01 14:00",   # 2019+: EU rule (EU changed 31 Mar)
    ], tz="UTC")
    out = fix_histdata_clock(pd.DataFrame({"close": range(6)}, index=idx))
    assert list(out.index.strftime("%Y-%m-%d %H:%M")) == [
        "2015-03-06 14:00", "2015-03-12 13:00", "2015-10-28 13:00", "2015-11-03 14:00",
        "2019-03-15 14:00", "2019-04-01 13:00"]


@pytest.mark.parametrize("s,v", [("151K", 151e3), ("0.3%", 0.3), ("-131.4B", -131.4e9),
                                 ("<0.25%", 0.25), ("4.1%", 4.1), ("54.8", 54.8),
                                 ("", np.nan), (None, np.nan), ("2.50|3.5", np.nan)])
def test_parse_value(s, v):
    out = parse_value(s)
    assert (np.isnan(v) and np.isnan(out)) or out == pytest.approx(v)


def _rows(*rows):
    df = pd.DataFrame(rows, columns=["wall", "indicator"])
    df["wall"] = pd.to_datetime(df.wall)
    df["group"] = df.indicator.map(lambda k: BY_KEY[k].group)
    return assign_times(df)


def test_tehran_wall_clock_uses_current_rules():
    # 2025: the snapshot labelled this +04:30, but Iran has had no DST since 2022.
    # ISM 10:00 New York (EDT) = 14:00 UTC = 17:30 Tehran (+03:30).
    df = _rows(("2025-04-01 17:30:00", "us_ism_mfg"))
    assert df.release_utc[0] == pd.Timestamp("2025-04-01 14:00", tz="UTC")
    assert df.time_source[0] == "ff" and df.sched_diff_min[0] == 0


def test_missing_time_from_sibling_then_schedule():
    df = _rows(("2025-03-07 00:00:00", "us_nfp"),             # lost time
               ("2025-03-07 17:00:00", "us_avg_earnings"),    # same release, has time
               ("2024-07-05 00:00:00", "us_unemployment"))    # lost, no sibling
    assert df.release_utc[0] == pd.Timestamp("2025-03-07 13:30", tz="UTC")  # 08:30 EST
    assert df.time_source[0] == "ff_sibling"
    assert df.release_utc[2] == pd.Timestamp("2024-07-05 12:30", tz="UTC")  # 08:30 EDT
    assert df.time_source[2] == "schedule"


def test_schedule_eras():
    df = _rows(("2022-06-09 00:00:00", "ea_ecb_rate"), ("2022-07-21 00:00:00", "ea_ecb_rate"))
    assert df.release_utc[0] == pd.Timestamp("2022-06-09 11:45", tz="UTC")   # 13:45 CEST
    assert df.release_utc[1] == pd.Timestamp("2022-07-21 12:15", tz="UTC")   # 14:15 CEST


def test_surprise_z_uses_only_past():
    n = 20
    df = pd.DataFrame({"indicator": "us_nfp",
                       "release_utc": pd.date_range("2020-01-01", periods=n, freq="30D", tz="UTC"),
                       "actual_v": np.r_[np.arange(n - 1) % 3, 1000.0], "forecast_v": 0.0})
    out = add_surprise(df)
    # The huge last surprise must not affect the scaling of any earlier row.
    df2 = df.copy(); df2.loc[n - 1, "actual_v"] = 1.0
    out2 = add_surprise(df2)
    pd.testing.assert_series_equal(out.surprise_z.iloc[:-1], out2.surprise_z.iloc[:-1])
    assert out.surprise_z.iloc[:8].isna().all()


def test_ticks_to_m1():
    base = pd.Timestamp("2020-01-02 10:00", tz="UTC").value // 10**6
    ticks = pd.DataFrame({"timestamp": [base + 1000, base + 30000, base + 59999, base + 60000],
                          "bidPrice": [1.1000, 1.1010, 1.0990, 1.2000],
                          "askPrice": [1.1002, 1.1013, 1.0991, 1.2001]})
    m1 = ticks_to_m1(ticks)
    r = m1.iloc[0]
    assert len(m1) == 2 and m1.index[0] == pd.Timestamp("2020-01-02 10:00", tz="UTC")
    assert (r.bid_open, r.bid_high, r.bid_low, r.bid_close) == (1.1000, 1.1010, 1.0990, 1.0990)
    assert r.spread_max == pytest.approx(0.0003) and r.n_ticks == 3


def _synthetic_m1(t0, jump_at, jump_pips=20.0):
    idx = pd.date_range(t0 - pd.Timedelta("2h"), t0 + pd.Timedelta("5h"), freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    bid = 1.1 + np.cumsum(rng.normal(0, 0.3e-4, len(idx)))
    bid[idx >= jump_at] += jump_pips * 1e-4
    m1 = pd.DataFrame(index=idx)
    for f in ("open", "high", "low", "close"):
        m1[f"bid_{f}"] = bid
        m1[f"ask_{f}"] = bid + 0.5e-4
    m1["spread_max"] = 0.5e-4
    m1["n_ticks"] = 10.0
    m1["source"] = "dukascopy"
    return m1


def _cal(t0):
    return pd.DataFrame({"event_id": ["x"], "indicator": ["us_nfp"], "currency": ["USD"],
                         "priority": [1], "release_utc": [t0]})


def test_event_features_and_time_check():
    t0 = pd.Timestamp("2024-07-05 12:30", tz="UTC")
    ev = build_event_table(_cal(t0), _synthetic_m1(t0, jump_at=t0)).iloc[0]
    assert ev.time_check == "ok" and ev.peak_offset_min == 0
    assert 15 < ev.ret_1m_pips < 25            # the 20-pip jump shows up in the first minute
    assert ev.spread_pre_pips == pytest.approx(0.5)
    assert ev.is_primary and ev.n_in_cluster == 1

    dst = build_event_table(_cal(t0), _synthetic_m1(t0, jump_at=t0 + pd.Timedelta("60min"))).iloc[0]
    assert dst.time_check == "suspect" and dst.peak_offset_min == 60   # daylight-saving error
    off = build_event_table(_cal(t0), _synthetic_m1(t0, jump_at=t0 + pd.Timedelta("10min"))).iloc[0]
    assert off.time_check == "suspect" and off.peak_offset_min == 10
    quiet = build_event_table(_cal(t0), _synthetic_m1(t0, jump_at=t0, jump_pips=0)).iloc[0]
    assert quiet.time_check == "unclear"
