"""Module 2: rule-based baseline. Is there any edge at all?

Rule (one market event -> at most one trade):
  1. Wait `delay` minutes after the release.
  2. move = mid price now - mid price just before the release (pips).
  3. If |move| >= `thresh` pips, trade:
       fade   = bet on a rebound (opposite to the move)
       follow = bet the move continues
  4. Exits set at entry: target +T pips, stop -S pips, time limit H minutes.

Realism (spot EUR/USD, before turbo mechanics):
  - buy at the ask, sell at the bid -> the spread is paid on every trade
  - if a bar touches both stop and target, assume the stop came first
  - a stop that is jumped over fills at the worse bar open (slippage)
  - only 2016+ data (real bid/ask); Dukascopy minutes with no quotes = no fill

Control: random moments at the SAME UTC time of day, with no relevant news
nearby, put through the identical rule. Edge = news result minus control result.
Parameters are chosen on TRAIN (2016-2021) and judged once on TEST (2022+).
"""

from itertools import product

import numpy as np
import pandas as pd

from . import config
from .calendar import download_calendar
from .prices import load_prices

DELAYS = (1, 5)
THRESHOLDS = (5, 10, 20)
TARGETS = (10, 20, 30)
STOPS = (10, 20, 30)
HOLDS = (30, 120, 240)
MODES = ("fade", "follow")
SPLIT = pd.Timestamp("2022-01-01", tz="UTC")
START = config.TICKS_START
CONTROLS_PER_EVENT = 15
MIN_TRADES = 30
L = max(DELAYS) + max(HOLDS)
FIELDS = ["bid_open", "bid_high", "bid_low", "bid_close", "ask_open", "ask_high", "ask_low", "ask_close"]
MIN = pd.Timedelta(minutes=1)


# ---------------------------------------------------------------- price windows
def build_windows(m1: pd.DataFrame, times: pd.DatetimeIndex):
    """W[i, k, f] = field f of the bar starting at times[i] + k minutes (NaN if no quotes).
    p0[i] = mid close of the last bar before times[i] (NaN if older than 5 min)."""
    W = np.full((len(times), L, len(FIELDS)), np.nan)
    p0 = np.full(len(times), np.nan)
    idx = m1.index
    vals = m1[FIELDS].to_numpy()
    mid = ((m1.bid_close + m1.ask_close) / 2).to_numpy()
    for i, t in enumerate(times):
        j = idx.searchsorted(t)                       # first bar >= t
        if j == 0 or t - idx[j - 1] > 5 * MIN:
            continue
        p0[i] = mid[j - 1]
        k = idx.searchsorted(t + L * MIN)
        sub = idx[j:k]
        W[i, ((sub - t) // MIN).astype(int)] = vals[j:k]
    return W, p0


# ---------------------------------------------------------------- simulation
def _first(mask: np.ndarray) -> np.ndarray:
    """Index of first True per row, or a large number if none."""
    return np.where(mask.any(1), mask.argmax(1), 10**9)


def simulate(W, p0, delay, thresh, mode, T, S, H) -> pd.DataFrame:
    """Returns per-row: traded (bool), direction (+1 long / -1 short), pnl_pips, exit."""
    f = {n: W[:, :, i] for i, n in enumerate(FIELDS)}
    e = delay - 1                                          # entry at the close of this bar
    mid_e = (f["bid_close"][:, e] + f["ask_close"][:, e]) / 2
    move = (mid_e - p0) / config.PIP
    traded = np.isfinite(move) & (np.abs(move) >= thresh)
    direction = np.where(mode == "fade", -np.sign(move), np.sign(move))

    path = slice(e + 1, e + 1 + H)
    pip = config.PIP
    # long: enter at ask, exit at bid
    el = f["ask_close"][:, e]
    lt = _first(f["bid_high"][:, path] >= (el + T * pip)[:, None])
    ls = _first(f["bid_low"][:, path] <= (el - S * pip)[:, None])
    # short: enter at bid, exit at ask
    es = f["bid_close"][:, e]
    st = _first(f["ask_low"][:, path] <= (es - T * pip)[:, None])
    ss = _first(f["ask_high"][:, path] >= (es + S * pip)[:, None])

    is_long = direction > 0
    entry = np.where(is_long, el, es)
    t_idx = np.where(is_long, lt, st)
    s_idx = np.where(is_long, ls, ss)

    # time-limit exit: last quoted close on the exit side within the holding window
    exit_close = np.where(is_long[:, None], f["bid_close"][:, path], f["ask_close"][:, path])
    last = pd.DataFrame(exit_close).ffill(axis=1).to_numpy()[:, -1]

    rows = np.arange(len(W))
    stop_open = np.where(is_long[:, None], f["bid_open"][:, path], f["ask_open"][:, path])
    s_safe = np.minimum(s_idx, H - 1)
    gap_open = stop_open[rows, s_safe]
    stop_level = entry - direction * S * pip
    # a stop that was jumped over fills at the (worse) bar open
    stop_fill = np.where(is_long, np.fmin(stop_level, gap_open), np.fmax(stop_level, gap_open))

    stop_first = (s_idx < 10**9) & (s_idx <= t_idx)        # tie -> stop (conservative)
    target_first = (t_idx < 10**9) & ~stop_first
    exit_px = np.select([stop_first, target_first], [stop_fill, entry + direction * T * pip], last)
    pnl = (exit_px - entry) * direction / pip
    traded &= np.isfinite(pnl)
    kind = np.select([stop_first, target_first], ["stop", "target"], "time")
    return pd.DataFrame({"traded": traded, "direction": direction, "move_pips": move,
                         "pnl_pips": np.where(traded, pnl, np.nan), "exit": kind})


# ---------------------------------------------------------------- control times
def news_free_mask(times: pd.DatetimeIndex, news_utc: pd.Series, unknown: pd.DataFrame) -> np.ndarray:
    """True where no USD/EUR medium/high-impact release is near `times`.
    Known time: +-60 min. Unknown time (calendar lost it): block that currency's
    usual release hours on that date (USD 12:00-15:30 UTC, EUR 06:00-11:30 UTC)."""
    news = np.sort(news_utc.to_numpy())
    lo = np.searchsorted(news, (times - 60 * MIN).to_numpy())
    hi = np.searchsorted(news, (times + 60 * MIN).to_numpy(), side="right")
    ok = hi == lo
    blocked = {(d, c) for d, c in zip(unknown.date, unknown.currency)}
    hours = times.hour + times.minute / 60
    for i, t in enumerate(times):
        d = t.date()
        if ((d, "USD") in blocked and 12 <= hours[i] <= 15.5) or \
           ((d, "EUR") in blocked and 6 <= hours[i] <= 11.5):
            ok[i] = False
    return ok


def load_news_times():
    raw = pd.read_csv(download_calendar(), dtype=str)
    raw = raw[raw.Currency.isin(["USD", "EUR"]) & raw.Impact.isin(["High Impact Expected", "Medium Impact Expected"])]
    wall = pd.to_datetime(raw.DateTime.str[:19])
    missing = wall.dt.strftime("%H:%M:%S").isin(["00:00:00", "23:59:59"])
    known = (wall[~missing].dt.tz_localize(config.CALENDAR_DISPLAY_TZ, ambiguous="NaT", nonexistent="NaT")
             .dt.tz_convert("UTC").dropna())
    unknown = pd.DataFrame({"date": wall[missing].dt.date, "currency": raw.Currency[missing]})
    return known, unknown


def sample_controls(event_times: pd.Series, m1_index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    known, unknown = load_news_times()
    rng = np.random.default_rng(seed)
    days = pd.date_range(START, config.END - pd.Timedelta("1D"), freq="B", tz="UTC")
    rows = []
    for t in event_times:
        tod = t - t.normalize()
        cand = days[rng.integers(0, len(days), CONTROLS_PER_EVENT * 4)] + tod
        cand = cand[np.abs((cand - t) / pd.Timedelta("1D")) > 1]
        cand = cand[news_free_mask(cand, known, unknown)][:CONTROLS_PER_EVENT]
        rows += [{"time": c, "matched_event": t} for c in cand]
    ctrl = pd.DataFrame(rows).drop_duplicates("time")
    return ctrl[ctrl.time.isin(m1_index)].reset_index(drop=True)  # market must be open


# ---------------------------------------------------------------- evaluation
def _stats(pnl: pd.Series) -> dict:
    n = int(pnl.notna().sum())
    m = pnl.mean()
    sd = pnl.std()
    return {"n": n, "mean": m, "win": (pnl > 0).mean() if n else np.nan,
            "t": m / (sd / np.sqrt(n)) if n > 1 and sd > 0 else np.nan}


def run_grid(ev_W, ev_p0, ev_time, ct_W, ct_p0, ct_time) -> pd.DataFrame:
    out = []
    for delay, thresh, mode, T, S, H in product(DELAYS, THRESHOLDS, MODES, TARGETS, STOPS, HOLDS):
        r = {"delay": delay, "thresh": thresh, "mode": mode, "T": T, "S": S, "H": H}
        for name, W, p0, times in (("news", ev_W, ev_p0, ev_time), ("ctrl", ct_W, ct_p0, ct_time)):
            sim = simulate(W, p0, delay, thresh, mode, T, S, H)
            pnl = sim.pnl_pips.where(sim.traded)
            for part, mask in (("train", times < SPLIT), ("test", times >= SPLIT)):
                for k, v in _stats(pnl[mask]).items():
                    r[f"{name}_{part}_{k}"] = v
        out.append(r)
    res = pd.DataFrame(out)
    for part in ("train", "test"):
        res[f"edge_{part}"] = res[f"news_{part}_mean"] - res[f"ctrl_{part}_mean"]
    return res


def continuation(W, p0, times, delay=5, horizon=60, thresh=10) -> pd.DataFrame:
    """Parameter-free view: after a move of >= thresh pips in the first `delay`
    minutes, how much further does price go in the move's direction by `horizon`?
    Positive = continuation, negative = rebound. Mid prices, no costs."""
    mid = (W[:, :, FIELDS.index("bid_close")] + W[:, :, FIELDS.index("ask_close")]) / 2
    m_e = pd.DataFrame(mid[:, :delay]).ffill(axis=1).to_numpy()[:, -1]
    m_h = pd.DataFrame(mid[:, :horizon]).ffill(axis=1).to_numpy()[:, -1]
    move = (m_e - p0) / config.PIP
    after = (m_h - m_e) / config.PIP * np.sign(move)
    big = np.isfinite(after) & (np.abs(move) >= thresh)
    return pd.DataFrame({"time": times[big], "move": move[big], "after": after[big]})


def report(ev_W, ev_p0, ev_time, ct_W, ct_p0, ct_time, res: pd.DataFrame) -> str:
    out = []
    out.append(f"news events: {len(ev_time):,} (2016+) | control moments: {len(ct_time):,}")
    out.append("\n1) After a >=10-pip move in the first 5 min: further move by minute 60 "
               "(+ = continues, - = rebounds), mid, no costs")
    for name, W, p0, t in (("news", ev_W, ev_p0, ev_time), ("control", ct_W, ct_p0, ct_time)):
        c = continuation(W, p0, t)
        s = _stats(c.after)
        out.append(f"   {name:8s} n={s['n']:5d}  mean {s['mean']:+.1f} pips  "
                   f"share continuing {s['win']:.0%}  t={s['t']:+.1f}")

    out.append(f"\n2) Rule grid: {len(res)} parameter combinations (costs included)")
    for part in ("train", "test"):
        ok = res[res[f"news_{part}_n"] >= MIN_TRADES]
        out.append(f"   {part}: configs with news mean > 0: {(ok[f'news_{part}_mean'] > 0).mean():.0%} | "
                   f"control mean > 0: {(ok[f'ctrl_{part}_mean'] > 0).mean():.0%}")

    ok = res[res.news_train_n >= MIN_TRADES].sort_values("news_train_t", ascending=False)
    cols = ["delay", "thresh", "mode", "T", "S", "H",
            "news_train_n", "news_train_mean", "news_train_t", "ctrl_train_mean",
            "news_test_n", "news_test_mean", "news_test_t", "ctrl_test_mean"]
    out.append("\n3) Best 10 configs chosen on TRAIN (2016-2021), then their TEST (2022+) result:")
    out.append(ok[cols].head(10).round(2).to_string(index=False))
    best_test = ok.head(10)
    out.append(f"\n   of these 10, positive in test: {(best_test.news_test_mean > 0).sum()} | "
               f"beat control in test: {(best_test.edge_test > 0).sum()}")
    return "\n".join(out)


def run(seed: int = 0):
    ev = pd.read_parquet(config.PROCESSED / "events.parquet")
    ev = ev[ev.is_primary & (ev.release_utc >= START)].sort_values("release_utc")
    m1 = load_prices()
    m1 = m1[m1.index >= START - pd.Timedelta("1D")]
    ev_time = pd.DatetimeIndex(ev.release_utc)
    ctrl = sample_controls(ev.release_utc, m1.index, seed)
    ct_time = pd.DatetimeIndex(ctrl.time)
    ev_W, ev_p0 = build_windows(m1, ev_time)
    ct_W, ct_p0 = build_windows(m1, ct_time)
    res = run_grid(ev_W, ev_p0, ev_time, ct_W, ct_p0, ct_time)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.PROCESSED / "baseline_grid.csv", index=False)
    return report(ev_W, ev_p0, ev_time, ct_W, ct_p0, ct_time, res)
