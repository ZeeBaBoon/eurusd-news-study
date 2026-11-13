import numpy as np
import pandas as pd

from eurusd_news.expectations import _at


def test_price_at_t_uses_only_bars_started_before_t():
    idx = pd.date_range("2020-01-01 13:00", periods=5, freq="1min", tz="UTC")
    v = np.array([1.0, 2, 3, 4, 5])
    t = pd.DatetimeIndex([idx[2]])                 # 13:02 -> bar 13:01 (closes at 13:02) is the latest known
    assert _at(idx, v, t)[0] == 2
    assert np.isnan(_at(idx, v, pd.DatetimeIndex([idx[0]]))[0])          # nothing before first bar
    assert np.isnan(_at(idx, v, pd.DatetimeIndex([idx[-1] + pd.Timedelta("10min")]))[0])  # stale
