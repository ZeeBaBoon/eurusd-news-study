import numpy as np
import pandas as pd

from eurusd_news.regimes import _above_past_median, market_regimes, sensitivity


def test_above_past_median_ignores_today_and_future():
    x = pd.Series([1.0, 2, 3, 4, 100])
    a = _above_past_median(x, 2)
    assert np.isnan(a[0]) and np.isnan(a[1])
    assert a[2] == 1 and a[3] == 1          # 3 > median(1,2); 4 > median(1,2,3)
    x2 = x.copy(); x2[4] = -100            # changing the LAST value must not change earlier ones
    pd.testing.assert_series_equal(a[:4], _above_past_median(x2, 2)[:4])


def test_market_regimes_use_only_past_days():
    idx = pd.bdate_range("2015-01-01", periods=400, tz="UTC")
    rng = np.random.default_rng(1)
    closes = pd.Series(1.1 + np.cumsum(rng.normal(0, 0.005, 400)), index=idx)
    a = market_regimes(closes)
    shocked = closes.copy(); shocked.iloc[-1] *= 1.2   # huge move on the last day
    b = market_regimes(shocked)
    # The regime known on the last day itself must not see that day's close.
    pd.testing.assert_frame_equal(a.iloc[:-1], b.iloc[:-1])


def test_sensitivity_excludes_current_release():
    n = 30
    ev = pd.DataFrame({"indicator": "us_nfp",
                       "release_utc": pd.date_range("2015-01-01", periods=n, freq="30D", tz="UTC"),
                       "surprise_z": 1.0, "ret_5m_pips": np.r_[np.full(n - 1, 5.0), 500.0]})
    s = sensitivity(ev)
    ev2 = ev.copy(); ev2.loc[n - 1, "ret_5m_pips"] = 5.0
    pd.testing.assert_series_equal(s, sensitivity(ev2))   # last reaction can't affect itself
