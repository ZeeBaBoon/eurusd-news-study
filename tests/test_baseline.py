import numpy as np
import pandas as pd
import pytest

from eurusd_news.baseline import FIELDS, L, news_free_mask, simulate

PIP = 1e-4
SPREAD = 0.2  # pips


def _window(mids):
    """One price path: mids[k] = mid at bar k (flat bars), spread 0.2 pips."""
    W = np.full((1, L, len(FIELDS)), np.nan)
    m = np.asarray(mids, dtype=float)
    n = len(m)
    for name in FIELDS:
        side = -1 if name.startswith("bid") else 1
        W[0, :n, FIELDS.index(name)] = m + side * SPREAD / 2 * PIP
    return W


def _run(mids, **kw):
    args = dict(delay=1, thresh=5, mode="fade", T=10, S=10, H=30) | kw
    return simulate(_window(mids), np.array([1.1000]), **args).iloc[0]


def test_no_trade_below_threshold():
    assert not _run([1.1003] * 40).traded


def test_fade_long_hits_target():
    # price drops 10 pips at release, then rebounds 20 pips -> fade = long, target +10
    r = _run([1.0990] + [1.0995] * 3 + [1.1010] * 30)
    assert r.traded and r.direction == 1 and r.exit == "target"
    assert r.pnl_pips == pytest.approx(10.0)


def test_follow_short_stopped_out_pays_spread_and_gap():
    # price drops 10 pips; follow = short. Price then jumps 15 pips up in one bar:
    # stop at +10 is jumped over -> fills at the worse bar open.
    r = _run([1.0990, 1.1005] + [1.1005] * 30, mode="follow")
    assert r.direction == -1 and r.exit == "stop"
    entry_bid = 1.0990 - 0.1 * PIP
    exit_ask = 1.1005 + 0.1 * PIP
    assert r.pnl_pips == pytest.approx((entry_bid - exit_ask) / PIP)   # -15.2, worse than -10


def test_bar_touching_both_counts_as_stop():
    W = _window([1.0990] + [1.0990] * 30)
    W[0, 1, FIELDS.index("bid_high")] = 1.1010   # target touched ...
    W[0, 1, FIELDS.index("bid_low")] = 1.0970    # ... and stop touched in the same bar
    r = simulate(W, np.array([1.1000]), 1, 5, "fade", 10, 10, 30).iloc[0]
    assert r.exit == "stop"


def test_time_limit_exit_at_bid():
    r = _run([1.0990] + [1.0993] * 40, H=30)
    assert r.exit == "time"
    assert r.pnl_pips == pytest.approx(3.0 - SPREAD)  # +3 pips minus the spread paid


def test_news_free_mask():
    times = pd.DatetimeIndex(["2020-03-02 10:00", "2020-03-02 13:00", "2020-03-03 13:00"], tz="UTC")
    news = pd.Series(pd.DatetimeIndex(["2020-03-02 10:30"], tz="UTC"))
    unknown = pd.DataFrame({"date": [pd.Timestamp("2020-03-03").date()], "currency": ["USD"]})
    assert list(news_free_mask(times, news, unknown)) == [False, True, False]
