"""Which releases we study, and when each one is officially published.

ForexFactory's impact labels change over the years (e.g. euro area CPI flash is
'medium' most years), so we select events by (currency, name), never by label.

The schedule is only a fallback: about 45% of rows in the calendar snapshot lost
their time. When that happens we use the official local release time, converted
to UTC with the issuer's own daylight-saving rules.
"""

from dataclasses import dataclass
from datetime import date, time

NY = "America/New_York"
FRA = "Europe/Berlin"
ALWAYS = date(1900, 1, 1)


def _t(s: str) -> time:
    h, m = map(int, s.split(":"))
    return time(h, m)


@dataclass(frozen=True)
class Indicator:
    key: str                 # our short name
    currency: str            # USD or EUR
    ff_names: tuple[str, ...]  # ForexFactory event name(s)
    tz: str                  # timezone the issuer publishes in
    times: tuple[tuple[date, time], ...]  # (valid from, local release time)
    group: str               # indicators published in the same release
    priority: int            # 1 = the headline number of its release

    def scheduled_time(self, d: date) -> time:
        current = self.times[0][1]
        for start, t in self.times:
            if d >= start:
                current = t
        return current


def _ind(key, cur, names, tz, times, group, prio):
    if isinstance(times, str):
        times = ((ALWAYS, _t(times)),)
    else:
        times = tuple((d, _t(t)) for d, t in times)
    names = (names,) if isinstance(names, str) else tuple(names)
    return Indicator(key, cur, names, tz, times, group, prio)


ECB_2022 = date(2022, 7, 21)   # ECB moved decision 13:45 -> 14:15, presser 14:30 -> 14:45
FOMC_2013 = date(2013, 7, 1)   # FOMC statement 14:15 -> 14:00 (12:30 on 2011-12 presser days)

CATALOG: list[Indicator] = [
    # ---- United States (New York time) ----
    _ind("us_nfp", "USD", "Non-Farm Employment Change", NY, "08:30", "us_jobs", 1),
    _ind("us_unemployment", "USD", "Unemployment Rate", NY, "08:30", "us_jobs", 2),
    _ind("us_avg_earnings", "USD", "Average Hourly Earnings m/m", NY, "08:30", "us_jobs", 3),
    _ind("us_core_cpi_mm", "USD", "Core CPI m/m", NY, "08:30", "us_cpi", 1),
    _ind("us_cpi_yy", "USD", "CPI y/y", NY, "08:30", "us_cpi", 2),
    _ind("us_cpi_mm", "USD", "CPI m/m", NY, "08:30", "us_cpi", 3),
    _ind("us_core_pce_mm", "USD", "Core PCE Price Index m/m", NY, "08:30", "us_pce", 1),
    _ind("us_gdp_advance", "USD", "Advance GDP q/q", NY, "08:30", "us_gdp", 1),
    _ind("us_retail", "USD", "Retail Sales m/m", NY, "08:30", "us_retail", 1),
    _ind("us_core_retail", "USD", "Core Retail Sales m/m", NY, "08:30", "us_retail", 2),
    _ind("us_ism_mfg", "USD", "ISM Manufacturing PMI", NY, "10:00", "us_ism_mfg", 1),
    _ind("us_ism_services", "USD", "ISM Services PMI", NY, "10:00", "us_ism_svc", 1),
    _ind("us_fed_rate", "USD", "Federal Funds Rate", NY,
         ((ALWAYS, "14:15"), (FOMC_2013, "14:00")), "us_fomc", 1),
    # ---- Euro area / Germany (Frankfurt time) ----
    _ind("ea_ecb_rate", "EUR", "Main Refinancing Rate", FRA,
         ((ALWAYS, "13:45"), (ECB_2022, "14:15")), "ea_ecb", 1),
    _ind("ea_ecb_press", "EUR", "ECB Press Conference", FRA,
         ((ALWAYS, "14:30"), (ECB_2022, "14:45")), "ea_ecb_press", 1),
    _ind("ea_cpi_flash", "EUR", "CPI Flash Estimate y/y", FRA, "11:00", "ea_cpi", 1),
    _ind("ea_core_cpi_flash", "EUR", "Core CPI Flash Estimate y/y", FRA, "11:00", "ea_cpi", 2),
    _ind("ea_gdp_prelim_flash", "EUR", "Prelim Flash GDP q/q", FRA, "11:00", "ea_gdp_prelim", 1),
    _ind("ea_gdp_flash", "EUR", "Flash GDP q/q", FRA, "11:00", "ea_gdp_flash", 1),
    _ind("de_pmi_mfg_flash", "EUR", "German Flash Manufacturing PMI", FRA, "09:30", "de_pmi", 1),
    _ind("de_pmi_svc_flash", "EUR", "German Flash Services PMI", FRA, "09:30", "de_pmi", 2),
    _ind("ea_pmi_mfg_flash", "EUR", "Flash Manufacturing PMI", FRA, "10:00", "ea_pmi", 1),
    _ind("ea_pmi_svc_flash", "EUR", "Flash Services PMI", FRA, "10:00", "ea_pmi", 2),
    _ind("de_ifo", "EUR", "German ifo Business Climate", FRA, "10:00", "de_ifo", 1),
    _ind("de_zew", "EUR", "German ZEW Economic Sentiment", FRA, "11:00", "de_zew", 1),
]

BY_FF_NAME: dict[tuple[str, str], Indicator] = {
    (ind.currency, name): ind for ind in CATALOG for name in ind.ff_names
}
BY_KEY: dict[str, Indicator] = {ind.key: ind for ind in CATALOG}
