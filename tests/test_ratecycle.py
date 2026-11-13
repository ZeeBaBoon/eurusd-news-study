import pandas as pd

from eurusd_news.ratecycle import cycle_active


def test_cycle_active_only_counts_changes_strictly_before():
    ch = pd.DatetimeIndex(["2020-03-16"], tz="UTC")
    days = pd.DatetimeIndex(["2020-03-15", "2020-03-16", "2020-03-17", "2020-09-13", "2020-09-16"], tz="UTC")
    assert cycle_active(days, ch).tolist() == [False, False, True, True, False]
