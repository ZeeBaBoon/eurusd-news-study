"""Figures for REPORT.md -> reports/figures/*.png  (`uv run eurusd-news figures`)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .baseline import SPLIT, START, build_windows, sample_controls  # noqa: E402
from .prices import load_prices  # noqa: E402

OUT = config.ROOT / "reports" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#86b6ef", "#3987e5", "#184f95"]          # one hue, light -> dark (old -> new)
GRAY = "#9a9994"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 200,
})


def _title(ax, title, sub):
    ax.set_title(title, pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def fig_event_example(ev, m1):
    e = ev[(ev.indicator == "us_nfp") & ev.is_primary & (ev.release_utc >= START) & (ev.time_check == "ok")]
    e = e.loc[e.ret_5m_pips.abs().idxmax()]
    t0 = e.release_utc
    w = m1.loc[t0 - pd.Timedelta("30min"): t0 + pd.Timedelta("120min")]
    mid = (w.bid_close + w.ask_close) / 2
    x = (w.index - t0) / pd.Timedelta("1min") + 1          # bar close time
    y = (mid - e.p0) / config.PIP
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axvline(0, color=INK2, lw=1)
    ax.axhline(0, color=GRAY, lw=1)
    ax.plot(x, y, color=BLUE)
    ax.text(1.5, ax.get_ylim()[1] * 0.92, "release", color=INK2, fontsize=9)
    ax.set_xlabel("minutes from release")
    ax.set_ylabel("EUR/USD vs. pre-release price (pips)")
    _title(ax, f"One event: US Non-Farm Payrolls, {t0:%d %b %Y}",
           f"actual {e.actual} vs. forecast {e.forecast} - minute mid prices (Dukascopy)")
    _save(fig, "1_event_example.png")


def fig_reaction_path(ev, m1, seed=0):
    ev = ev[ev.is_primary & (ev.release_utc >= START)]
    m1 = m1[m1.index >= START - pd.Timedelta("1D")]
    t_ev = pd.DatetimeIndex(ev.release_utc)
    ctrl = sample_controls(ev.release_utc, m1.index, seed)
    t_ct = pd.DatetimeIndex(ctrl.time)

    def paths(times):
        W, p0 = build_windows(m1, times)
        mid = (W[:, :60, 3] + W[:, :60, 7]) / 2
        mid = pd.DataFrame(mid).ffill(axis=1).to_numpy()
        move = (mid - p0[:, None]) / config.PIP
        s = np.sign(move[:, 4])
        keep = np.isfinite(move[:, 59]) & (np.abs(move[:, 4]) >= 10)
        return (move * s[:, None])[keep], times[keep]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(1, 61)
    pe, te = paths(t_ev)
    pc, _ = paths(t_ct)
    series = [("news 2016-21", pe[te < SPLIT], BLUE), ("news 2022+", pe[te >= SPLIT], ORANGE),
              ("random news-free moments", pc, GRAY)]
    for lab, p, col in series:
        m = p.mean(0)
        se = p.std(0) / np.sqrt(len(p))
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=col, alpha=0.10, lw=0)
        ax.plot(x, m, color=col, label=f"{lab} (n={len(p)})")
        ax.plot(x[-1], m[-1], "o", ms=7, color=col, mec=SURFACE, mew=2)
        ax.annotate(f"{m[-1]:.1f}", (x[-1], m[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.axvline(5, color=INK2, lw=1)
    ax.text(5.6, 1, "entry\n(minute 5)", color=INK2, fontsize=9, va="bottom")
    ax.set_xlim(0, 66)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("minutes from release")
    ax.set_ylabel("move in the direction of the first 5 min (pips)")
    ax.legend(loc="lower right")
    _title(ax, "After minute 5, news moves drift no more than random moves",
           "average path after a >= 10-pip move in the first 5 minutes; bands = 95% confidence")
    _save(fig, "2_reaction_path.png")


def fig_overfitting():
    r = pd.read_csv(config.PROCESSED / "baseline_grid.csv")
    r = r[(r.news_train_n >= 30) & (r.news_test_n >= 30)]
    top = r.nlargest(10, "news_train_t")
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    lim = np.nanmax(np.abs(r[["news_train_mean", "news_test_mean"]].to_numpy())) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, lw=1)
    ax.axhline(0, color=INK2, lw=1); ax.axvline(0, color=INK2, lw=1)
    ax.scatter(r.news_train_mean, r.news_test_mean, s=22, color=GRAY, alpha=0.55, lw=0,
               label=f"all rules (n={len(r)})")
    ax.scatter(top.news_train_mean, top.news_test_mean, s=50, color=BLUE, edgecolor=SURFACE, lw=1.5,
               label="10 best rules on train", zorder=3)
    ax.text(lim * 0.95, lim * 0.88, "same result\nin both periods", ha="right", color=INK2, fontsize=9)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("train 2016-21: average pips per trade")
    ax.set_ylabel("test 2022+: average pips per trade")
    ax.legend(loc="lower right")
    _title(ax, "The best rules on train don't hold up on test",
           "324 fade/follow rules with stops, targets and holding times; real bid/ask costs")
    _save(fig, "3_overfitting.png")


def fig_sign_flip():
    rows = []
    reg = pd.read_csv(config.PROCESSED / "regimes_result.csv")
    for _, x in reg.iterrows():
        rows.append(("Market regime (2b)", x.regime, x.train_mean, x.test_mean))
    ex = pd.read_csv(config.PROCESSED / "expectations_events.csv", parse_dates=["time"])
    ex = ex[ex.time >= START]
    for lab, m in (("drift 'priced in'", ex.drift_grp == "priced in"), ("drift against move", ex.drift_grp == "against"),
                   ("calm before", ex.nervous == 0), ("nervous before", ex.nervous == 1)):
        g = ex[m]
        rows.append(("Before the release (2c)", lab, g.after[g.time < SPLIT].mean(), g.after[g.time >= SPLIT].mean()))
    rc = pd.read_csv(config.PROCESSED / "ratecycle_events.csv", parse_dates=["time"])
    rc = rc[rc.time >= START]
    for lab, v in (("Fed/ECB active", True), ("Fed/ECB on hold", False)):
        g = rc[rc.active == v]
        rows.append(("Rate cycle (2d)", lab, g.after[g.time < SPLIT].mean(), g.after[g.time >= SPLIT].mean()))
    d = pd.DataFrame(rows, columns=["group", "label", "train", "test"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axvspan(-2, 2, color=GRAY, alpha=0.12, lw=0)
    ax.axvline(0, color=INK2, lw=1)
    y, ticks, labels = 0, [], []
    for grp, g in d.groupby("group", sort=False):
        ax.text(-9.6, y - 0.1, grp, fontweight="bold", fontsize=9.5, va="bottom")
        y -= 0.9
        for _, r in g.iterrows():
            ax.plot([r.train, r.test], [y, y], color=GRAY, lw=1.5, zorder=1)
            ax.plot(r.train, y, "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=2)
            ax.plot(r.test, y, "o", ms=8, color=ORANGE, mec=SURFACE, mew=2, zorder=2)
            ticks.append(y); labels.append(r.label)
            y -= 1
        y -= 0.4
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(-9.6, 9.6)
    ax.set_xlabel("further move by minute 60 (pips):  - rebound | + continuation")
    ax.set_ylim(ticks[-1] - 0.7, 0.6)
    ax.plot([], [], "o", ms=8, color=BLUE, label="train 2016-21")
    ax.plot([], [], "o", ms=8, color=ORANGE, label="test 2022+")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=GRAY, alpha=0.25, label="within +-2 pips: smaller than costs"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=3)
    _title(ax, "Most splits flip sign between periods",
           "pre-registered filters, each known before the release; entry 5 min (2b, 2d) or 1 min (2c) after")
    _save(fig, "4_sign_flip.png")


def fig_decay(ev):
    ev = ev[ev.is_primary].dropna(subset=["ret_5m_pips", "ret_60m_pips"])
    after = (ev.ret_60m_pips - ev.ret_5m_pips) * np.sign(ev.ret_5m_pips)
    per = {"2010-15": ev.release_utc < START, "2016-21": (ev.release_utc >= START) & (ev.release_utc < SPLIT),
           "2022+": ev.release_utc >= SPLIT}
    th = [10, 20, 30]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.26
    for i, (p, m) in enumerate(per.items()):
        vals = [after[m & (ev.ret_5m_pips.abs() >= t)] for t in th]
        means = [v.mean() for v in vals]
        xs = np.arange(len(th)) + (i - 1) * w
        ax.bar(xs, means, width=w - 0.03, color=SEQ[i], label=p, zorder=2)
        for xx, v, mm in zip(xs, vals, means):
            ax.text(xx, mm + (0.3 if mm >= 0 else -0.3), f"{mm:+.1f}\nn={len(v)}", ha="center",
                    va="bottom" if mm >= 0 else "top", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(len(th)), [f">= {t} pips" for t in th])
    ax.set_xlabel("size of the move in the first 5 minutes")
    ax.set_ylabel("further move by minute 60 (pips)")
    ax.set_ylim(min(ax.get_ylim()[0], 0) - 2.5, ax.get_ylim()[1] + 2)
    ax.legend(loc="upper left", ncol=3)
    _title(ax, "Big news moves used to continue - that edge has faded",
           "average further move after entering at minute 5, mid prices, before costs")
    _save(fig, "5_decay.png")


def run():
    ev = pd.read_parquet(config.PROCESSED / "events.parquet")
    m1 = load_prices()
    fig_event_example(ev, m1)
    fig_reaction_path(ev, m1)
    fig_overfitting()
    fig_sign_flip()
    fig_decay(ev)
