"""International Cost of Capital app.

Compares a project's cost of equity, cost of debt and WACC in USD and in the
currency of a selected country, using live data from:
  * Damodaran  - US industry unlevered betas, BBB (Baa2) borrowing spread
  * OECD       - statutory corporate income tax rates
  * Trading Economics - 10-year government bond yields

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import io
import re
import unicodedata

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
URL_BETAS = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls"
URL_SPREAD = "https://pages.stern.nyu.edu/adamodar/New_Home_Page/datafile/ratings.html"
URL_TAX = (
    "https://www.oecd.org/en/publications/corporate-tax-statistics-2024_9c27d6e8-en/"
    "full-report/statutory-corporate-income-tax-rates_440f9286.html"
)
URL_BONDS = "https://tradingeconomics.com/bonds"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DEFAULT_INDUSTRY = "Software (System & Application)"
DEFAULT_COUNTRY = "United States"
DEFAULT_ERP = 6.5      # %
DEFAULT_DEBT = 30      # %

# Offline fallbacks. They are used ONLY if a live fetch fails, and are always
# labelled as fallbacks in the "Data sources" table. Override them manually.
FALLBACK_SPREAD = 0.0111          # Baa2/BBB, large non-financial firms, Damodaran Jan-2026
FALLBACK_SPREAD_ASOF = "January 2026 (bundled snapshot)"
FALLBACK_US_YIELD = 0.0500        # approx. level of the US 10Y in late Oct 2026 (placeholder)
FALLBACK_BETA = 1.20              # generic placeholder, only used if betas.xls cannot be read
# Approximate combined statutory CIT rates (OECD Corporate Tax Statistics 2024).
# Bundled snapshot for offline use - verify against the OECD page.
FALLBACK_TAX = {
    "United States": 0.2577, "Canada": 0.262, "Mexico": 0.30, "Brazil": 0.34,
    "Argentina": 0.35, "Chile": 0.27, "Colombia": 0.35, "Peru": 0.295,
    "United Kingdom": 0.25, "Germany": 0.299, "France": 0.2583, "Italy": 0.24,
    "Spain": 0.25, "Netherlands": 0.258, "Switzerland": 0.196, "Sweden": 0.206,
    "Norway": 0.22, "Denmark": 0.22, "Ireland": 0.125, "Poland": 0.19,
    "Turkey": 0.25, "Japan": 0.2974, "South Korea": 0.264, "China": 0.25,
    "India": 0.2517, "Indonesia": 0.22, "Singapore": 0.17, "Australia": 0.30,
    "New Zealand": 0.28, "South Africa": 0.27, "Saudi Arabia": 0.20,
    "Israel": 0.23, "Hungary": 0.09,
}

CURRENCY = {
    "United States": "USD", "Canada": "CAD", "Mexico": "MXN", "Brazil": "BRL",
    "Argentina": "ARS", "Chile": "CLP", "Colombia": "COP", "Peru": "PEN",
    "United Kingdom": "GBP", "Switzerland": "CHF", "Sweden": "SEK", "Norway": "NOK",
    "Denmark": "DKK", "Poland": "PLN", "Czech Republic": "CZK", "Czechia": "CZK",
    "Hungary": "HUF", "Romania": "RON", "Turkey": "TRY", "Russia": "RUB",
    "Japan": "JPY", "South Korea": "KRW", "China": "CNY", "India": "INR",
    "Indonesia": "IDR", "Thailand": "THB", "Malaysia": "MYR", "Philippines": "PHP",
    "Vietnam": "VND", "Singapore": "SGD", "Hong Kong": "HKD", "Taiwan": "TWD",
    "Pakistan": "PKR", "Australia": "AUD", "New Zealand": "NZD", "South Africa": "ZAR",
    "Egypt": "EGP", "Nigeria": "NGN", "Kenya": "KES", "Israel": "ILS",
    "Saudi Arabia": "SAR", "Iceland": "ISK",
}
for _c in ["Germany", "France", "Italy", "Spain", "Netherlands", "Belgium", "Austria",
           "Finland", "Ireland", "Portugal", "Greece", "Slovakia", "Slovenia",
           "Lithuania", "Latvia", "Estonia", "Luxembourg", "Croatia", "Cyprus", "Malta"]:
    CURRENCY[_c] = "EUR"

_ALIASES = {
    "turkiye": "turkey", "korea": "south korea", "republic of korea": "south korea",
    "korea, republic of": "south korea", "czechia": "czech republic",
    "slovak republic": "slovakia", "united states of america": "united states",
    "us": "united states", "usa": "united states", "russian federation": "russia",
    "china (people's republic of)": "china", "people's republic of china": "china",
    "peoples republic of china": "china", "hong kong, china": "hong kong",
    "viet nam": "vietnam", "uk": "united kingdom",
}


# --------------------------------------------------------------------------- #
# Core finance (pure functions - these are what verify.py tests)
# --------------------------------------------------------------------------- #
def relever_beta(beta_u: float, debt_to_equity: float, tax: float) -> float:
    """Hamada: beta_L = beta_U * (1 + (1 - t) * D/E)."""
    return beta_u * (1.0 + (1.0 - tax) * debt_to_equity)


def fx_factor(y_us: float, y_fc: float) -> float:
    """Conversion factor (1 + y_FC) / (1 + y_US).

    The 10-year government yield differential is used as the proxy for the
    expected inflation / interest-rate differential between the two currencies.
    """
    return (1.0 + y_fc) / (1.0 + y_us)


def compute_wacc(
    beta_u: float,
    tax: float,
    erp: float,
    debt_share: float,
    y_us: float,
    y_fc: float,
    spread_usd: float,
    spread_fc_override: float | None = None,
) -> dict:
    """Cost of equity, cost of debt and WACC in USD and the foreign currency.

    All rates are decimals (0.05 = 5%). Shared inputs (beta, tax, ERP, debt
    share) are identical in both currencies; only the risk-free rate and the
    currency conversion differ.
    """
    if not 0.0 <= debt_share < 1.0:
        raise ValueError("debt_share must be in [0, 1)")
    w_d = debt_share
    w_e = 1.0 - debt_share
    de = w_d / w_e
    beta_l = relever_beta(beta_u, de, tax)

    # --- USD ---
    ke_usd = y_us + beta_l * erp
    kd_usd = y_us + spread_usd
    wacc_usd = w_e * ke_usd + w_d * kd_usd * (1.0 - tax)

    # --- Foreign currency (interest-rate-parity style conversion) ---
    f = fx_factor(y_us, y_fc)
    ke_fc = f * (1.0 + ke_usd) - 1.0
    spread_fc_auto = spread_usd * f            # = [(1+Kd_USD) * f - 1] - y_FC
    spread_fc = spread_fc_auto if spread_fc_override is None else spread_fc_override
    kd_fc = y_fc + spread_fc
    wacc_fc = w_e * ke_fc + w_d * kd_fc * (1.0 - tax)

    return {
        "beta_u": beta_u, "beta_l": beta_l, "de": de, "tax": tax, "erp": erp,
        "w_e": w_e, "w_d": w_d, "fx_factor": f,
        "usd": {"rf": y_us, "ke": ke_usd, "spread": spread_usd, "kd": kd_usd,
                "kd_at": kd_usd * (1 - tax), "wacc": wacc_usd},
        "fc": {"rf": y_fc, "ke": ke_fc, "spread": spread_fc, "kd": kd_fc,
               "kd_at": kd_fc * (1 - tax), "wacc": wacc_fc,
               "spread_auto": spread_fc_auto,
               "spread_is_override": spread_fc_override is not None},
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def norm(name: str) -> str:
    """Normalise country names so OECD / Trading Economics names can be matched."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s).strip().lower()
    return _ALIASES.get(s, s)


def to_float(x) -> float | None:
    if x is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None


def pct(x: float | None, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def _get(url: str) -> bytes:
    r = requests.get(url, headers=HEADERS, timeout=(8, 30))
    r.raise_for_status()
    return r.content


def _text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# --------------------------------------------------------------------------- #
# Parsers (separate from network code so they can be unit-tested)
# --------------------------------------------------------------------------- #
def parse_betas_xls(content: bytes) -> tuple[pd.DataFrame, str]:
    """Return (DataFrame[Industry, Unlevered beta, ...], as-of text)."""
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    for name, raw in sheets.items():
        hdr = None
        for i in range(min(len(raw), 40)):
            if any(str(c).strip().lower() == "industry name" for c in raw.iloc[i]):
                hdr = i
                break
        if hdr is None:
            continue
        cols = [str(c).strip() for c in raw.iloc[hdr]]
        df = raw.iloc[hdr + 1:].copy()
        df.columns = cols
        low = {c.lower(): c for c in cols}
        if "unlevered beta" not in low:
            continue
        ind_col, ub_col = low["industry name"], low["unlevered beta"]
        out = pd.DataFrame({
            "Industry": df[ind_col].astype(str).str.strip(),
            "Unlevered beta": pd.to_numeric(df[ub_col], errors="coerce"),
        })
        for extra in ["number of firms", "beta", "d/e ratio", "effective tax rate"]:
            if extra in low:
                out[low[extra]] = pd.to_numeric(df[low[extra]], errors="coerce")
        out = out[out["Unlevered beta"].notna() & ~out["Industry"].str.lower().str.startswith("total market")]
        out = out[out["Industry"].ne("nan")].reset_index(drop=True)
        head_txt = " ".join(str(v) for v in raw.iloc[:hdr].values.ravel() if str(v) != "nan")
        m = re.search(r"as of\s+([A-Za-z]+\.?\s*\d{4})", head_txt, re.I) or \
            re.search(r"(?:date updated|updated)[:\s]+([^\n]+?)(?:\s{2,}|$)", head_txt, re.I)
        return out, (m.group(1).strip() if m else "date not found in file")
    raise ValueError("Could not find an 'Industry Name' / 'Unlevered beta' table in betas.xls")


def parse_spread_html(html: str) -> tuple[float, str]:
    """Baa2/BBB spread for large non-financial firms (left-hand table)."""
    tables = pd.read_html(io.StringIO(html), header=None)
    for t in tables:
        for row in t.fillna("").astype(str).values.tolist():
            for j, cell in enumerate(row):
                if re.search(r"Baa2\s*/\s*BBB", cell, re.I):
                    for c2 in row[j + 1:]:
                        m = re.search(r"(\d+(?:\.\d+)?)\s*%", c2)
                        if m:
                            asof = re.search(r"as of\s+([A-Za-z]+\s+\d{4})",
                                             re.sub(r"<[^>]+>", " ", html), re.I)
                            return float(m.group(1)) / 100.0, (asof.group(1) if asof else "date not found")
    raise ValueError("Baa2/BBB row not found on Damodaran ratings page")


def parse_te_html(html: str) -> dict[str, tuple[float, str]]:
    """{normalised country: (yield as decimal, table date text)} from tradingeconomics.com/bonds."""
    out: dict[str, tuple[float, str, str]] = {}
    for t in pd.read_html(io.StringIO(html)):
        cols = [str(c).strip().lower() for c in t.columns]
        if "yield" not in cols:
            continue
        yi = cols.index("yield")
        di = cols.index("date") if "date" in cols else None
        for _, r in t.iterrows():
            name = re.sub(r"\s+", " ", str(r.iloc[0])).strip()
            y = to_float(r.iloc[yi])
            if not name or name.lower() == "nan" or y is None:
                continue
            key = norm(name)
            if key not in out:
                out[key] = (y / 100.0, str(r.iloc[di]).strip() if di is not None else "", name)
    if not out:
        raise ValueError("No yield table found on Trading Economics page")
    return out


def parse_oecd_html(html: str) -> tuple[dict[str, float], str]:
    """{normalised country: rate} using the latest-year column of the OECD rate table."""
    best: tuple[dict[str, float], str] | None = None
    for t in pd.read_html(io.StringIO(html), header=None):
        t = t.fillna("").astype(str)
        if t.shape[0] < 20 or t.shape[1] < 2:
            continue
        year_row = None
        for i in range(min(6, len(t))):
            if sum(bool(re.fullmatch(r"(19|20)\d\d(\.0)?", c.strip())) for c in t.iloc[i]) >= 3:
                year_row = i
                break
        if year_row is None:
            continue
        years = {j: int(float(c)) for j, c in enumerate(t.iloc[year_row].str.strip())
                 if re.fullmatch(r"(19|20)\d\d(\.0)?", c)}
        col = max(years, key=years.get)
        rates: dict[str, float] = {}
        for _, r in t.iloc[year_row + 1:].iterrows():
            v = to_float(r.iloc[col])
            if v is not None and r.iloc[0].strip() and r.iloc[0].strip().lower() != "nan":
                rates[norm(r.iloc[0])] = v
        if "united states" in rates and len(rates) >= 20:
            scale = 1.0 if max(rates.values()) <= 1.0 else 100.0
            return {k: v / scale for k, v in rates.items()}, str(years[col])
    raise ValueError("OECD statutory tax-rate table not found in page HTML")


# --------------------------------------------------------------------------- #
# Fetchers - cached 1h across sessions on success (failures are not cached)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_betas():
    df, asof = parse_betas_xls(_get(URL_BETAS))
    return {"df": df, "asof": asof, "fetched_at": _now()}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_spread():
    val, asof = parse_spread_html(_text(_get(URL_SPREAD)))
    return {"value": val, "asof": asof, "fetched_at": _now()}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_yields():
    return {"map": parse_te_html(_text(_get(URL_BONDS))), "fetched_at": _now()}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_tax():
    rates, year = parse_oecd_html(_text(_get(URL_TAX)))
    return {"map": rates, "asof": f"OECD Corporate Tax Statistics 2024, {year} rates", "fetched_at": _now()}


def _safe(fn):
    try:
        return {"ok": True, **fn()}
    except Exception as e:  # network error, layout change, blocked, ...
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def load_all(force: bool = False) -> dict:
    if force:
        st.cache_data.clear()
        st.session_state.pop("data", None)
    if "data" not in st.session_state:
        with st.spinner("Fetching Damodaran, OECD and Trading Economics data..."):
            st.session_state["data"] = {
                "betas": _safe(_fetch_betas), "spread": _safe(_fetch_spread),
                "yields": _safe(_fetch_yields), "tax": _safe(_fetch_tax),
            }
    return st.session_state["data"]


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def override_input(label, fetched, fallback, key, *, pct_units=True, step=0.01, lo=-5.0, hi=100.0,
                   help_=None):
    """Checkbox + number box. Returns (value used, source label). Values are decimals for % inputs."""
    forced = fetched is None
    scale = 100.0 if pct_units else 1.0
    c1, c2 = st.columns([1, 1])
    with c1:
        on = st.checkbox(f"Override: {label}", value=forced, disabled=forced, key=f"on_{key}", help=help_)
    start = (fallback if forced else fetched) * scale
    with c2:
        val = st.number_input(label, min_value=lo, max_value=hi, value=float(round(start, 4)), step=step,
                              format="%.3f", key=f"val_{key}_{round(start, 6)}",
                              label_visibility="collapsed", disabled=not (on or forced))
    if forced:
        return val / scale, "manual (live fetch failed)"
    return (val / scale, "manual override") if on else (fetched, "fetched")


def build_table(res: dict, ccy: str) -> pd.DataFrame:
    u, f = res["usd"], res["fc"]
    rows = [
        ("10-yr government yield (risk-free proxy)", u["rf"], f["rf"], "pct"),
        ("Unlevered beta", res["beta_u"], res["beta_u"], "num"),
        ("Debt / Equity", res["de"], res["de"], "num"),
        ("Relevered beta", res["beta_l"], res["beta_l"], "num"),
        ("Equity risk premium", res["erp"], res["erp"], "pct"),
        ("Cost of equity", u["ke"], f["ke"], "pct"),
        ("BBB credit spread", u["spread"], f["spread"], "pct"),
        ("Pre-tax cost of debt", u["kd"], f["kd"], "pct"),
        ("Project tax rate", res["tax"], res["tax"], "pct"),
        ("After-tax cost of debt", u["kd_at"], f["kd_at"], "pct"),
        ("Equity weight E/V", res["w_e"], res["w_e"], "pct"),
        ("Debt weight D/V", res["w_d"], res["w_d"], "pct"),
        ("WACC", u["wacc"], f["wacc"], "pct"),
    ]
    return pd.DataFrame(rows, columns=["Item", "USD", ccy, "kind"])


def fmt_table(df: pd.DataFrame, ccy: str) -> pd.DataFrame:
    out = df.copy()
    for c in ["USD", ccy]:
        out[c] = [pct(v) if k == "pct" else f"{v:.3f}" for v, k in zip(df[c], df["kind"])]
    return out.drop(columns="kind")


def all_industries(betas: pd.DataFrame, **kw) -> pd.DataFrame:
    rows = []
    for _, r in betas.iterrows():
        res = compute_wacc(beta_u=float(r["Unlevered beta"]), **kw)
        rows.append({
            "Industry": r["Industry"], "Unlevered beta": r["Unlevered beta"],
            "Relevered beta": res["beta_l"],
            "Cost of equity (USD)": res["usd"]["ke"], "Cost of equity (FC)": res["fc"]["ke"],
            "WACC (USD)": res["usd"]["wacc"], "WACC (FC)": res["fc"]["wacc"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="International Cost of Capital", layout="wide")
    st.title("International Cost of Equity & WACC")
    st.caption("Compare a project's cost of capital in USD and in the currency of a selected country. "
               "Same beta, tax rate, equity risk premium and debt share in both currencies.")

    with st.sidebar:
        st.header("Data")
        refresh = st.button("🔄 Refresh data", width="stretch",
                            help="Clears the cache and re-fetches all four sources.")
    D = load_all(force=refresh)
    betas_r, spread_r, yields_r, tax_r = D["betas"], D["spread"], D["yields"], D["tax"]

    # ---- Selectors -------------------------------------------------------- #
    industries = list(betas_r["df"]["Industry"]) if betas_r["ok"] else [DEFAULT_INDUSTRY]
    ycountries = sorted({v[2] for v in yields_r["map"].values()}) if yields_r["ok"] else sorted(FALLBACK_TAX)
    if DEFAULT_COUNTRY not in ycountries:
        ycountries = [DEFAULT_COUNTRY] + ycountries

    with st.sidebar:
        st.header("Project assumptions")
        industry = st.selectbox("Industry (Damodaran US)", industries,
                                index=industries.index(DEFAULT_INDUSTRY) if DEFAULT_INDUSTRY in industries else 0)
        country = st.selectbox("Target country", ycountries, index=ycountries.index(DEFAULT_COUNTRY))
        erp = st.number_input("Equity risk premium (%)", 0.0, 20.0, DEFAULT_ERP, 0.1, format="%.2f") / 100
        debt = st.slider("Target debt share D/V (%)", 0, 90, DEFAULT_DEBT, 1) / 100
        tax_choice = st.radio("Project tax rate", ["United States", "Selected country"], horizontal=True,
                              help="One tax rate is applied to both currencies.")

    is_us = norm(country) == "united states"
    ccy = CURRENCY.get(country, "Local currency")
    ccy_label = "USD (same)" if is_us else ccy

    # ---- Fetched values --------------------------------------------------- #
    f_beta = None
    if betas_r["ok"]:
        f_beta = float(betas_r["df"].loc[betas_r["df"]["Industry"] == industry, "Unlevered beta"].iloc[0])
    tax_map = tax_r["map"] if tax_r["ok"] else {norm(k): v for k, v in FALLBACK_TAX.items()}
    f_tax_us = tax_map.get("united states")
    f_tax_fc = tax_map.get(norm(country))
    f_spread = spread_r["value"] if spread_r["ok"] else None
    ymap = yields_r["map"] if yields_r["ok"] else {}
    f_y_us = ymap["united states"][0] if "united states" in ymap else None
    f_y_fc = ymap[norm(country)][0] if norm(country) in ymap else None

    # ---- Manual overrides ------------------------------------------------- #
    with st.expander("Manual overrides (optional) - replace fetched data", expanded=False):
        st.caption("Tick a box to replace the fetched value. If a fetch failed, the manual box is required.")
        beta_u, beta_src = override_input("Unlevered beta", f_beta, FALLBACK_BETA, "beta",
                                          pct_units=False, step=0.01, lo=0.0, hi=5.0)
        tax_us, tax_us_src = override_input("US corporate tax rate (%)", f_tax_us, FALLBACK_TAX["United States"], "taxus")
        if is_us:
            tax_fc, tax_fc_src = tax_us, tax_us_src
            st.caption("Selected country is the United States - the country tax rate equals the US rate.")
        else:
            tax_fc, tax_fc_src = override_input(f"{country} corporate tax rate (%)", f_tax_fc, FALLBACK_TAX["United States"], "taxfc")
        spread_usd, spread_src = override_input("USD BBB spread (%)", f_spread, FALLBACK_SPREAD, "spread", lo=0.0, hi=20.0)
        y_us, y_us_src = override_input("US 10-yr yield (%)", f_y_us, FALLBACK_US_YIELD, "yus")
        if is_us:
            y_fc, y_fc_src = y_us, y_us_src
            st.caption("Selected country is the United States - foreign yield equals the US yield.")
        else:
            y_fc, y_fc_src = override_input(f"{country} 10-yr yield (%)", f_y_fc, y_us, "yfc")

        auto_spread = spread_usd * fx_factor(y_us, y_fc)
        st.markdown("**Foreign-currency BBB spread**")
        sp_on = st.toggle("Override foreign-currency BBB spread", value=False,
                          help="Off = auto: USD spread x (1+y_FC)/(1+y_US). On = enter your own.")
        if sp_on:
            sp_fc = st.number_input("Foreign-currency BBB spread (%)", 0.0, 50.0,
                                    float(round(auto_spread * 100, 4)), 0.01, format="%.3f",
                                    key="sp_fc_val") / 100
        else:
            sp_fc = None
            st.caption(f"Auto-calculated foreign-currency spread: **{pct(auto_spread, 3)}**")

    tax = tax_us if tax_choice == "United States" else tax_fc
    tax_src = tax_us_src if tax_choice == "United States" else tax_fc_src

    # ---- Warnings about missing data ------------------------------------- #
    for nm, r in [("Damodaran betas", betas_r), ("Damodaran BBB spread", spread_r),
                  ("Trading Economics yields", yields_r), ("OECD tax rates", tax_r)]:
        if not r["ok"]:
            st.warning(f"{nm}: live fetch failed ({r['error']}). Use manual overrides"
                       + (" or the bundled fallback shown in the sources table." if nm != "Damodaran betas" else "."))
    if yields_r["ok"] and f_y_fc is None and not is_us:
        st.warning(f"No Trading Economics yield found for {country}; enter it manually.")

    # ---- Results ---------------------------------------------------------- #
    res = compute_wacc(beta_u, tax, erp, debt, y_us, y_fc, spread_usd, sp_fc)
    tbl = build_table(res, ccy_label)

    st.subheader(f"{industry}  |  {country}  |  tax rate: {tax_choice} ({pct(tax)})")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🇺🇸 USD")
        a, b = st.columns(2)
        a.metric("Cost of equity", pct(res["usd"]["ke"]))
        b.metric("WACC", pct(res["usd"]["wacc"]))
        st.metric("After-tax cost of debt", pct(res["usd"]["kd_at"]))
    with c2:
        st.markdown(f"### {country} - {ccy_label}")
        a, b = st.columns(2)
        a.metric("Cost of equity", pct(res["fc"]["ke"]), delta=f"{(res['fc']['ke'] - res['usd']['ke']) * 100:+.2f} pp vs USD")
        b.metric("WACC", pct(res["fc"]["wacc"]), delta=f"{(res['fc']['wacc'] - res['usd']['wacc']) * 100:+.2f} pp vs USD")
        st.metric("After-tax cost of debt", pct(res["fc"]["kd_at"]),
                  delta=f"{(res['fc']['kd_at'] - res['usd']['kd_at']) * 100:+.2f} pp vs USD")
    st.dataframe(fmt_table(tbl, ccy_label), width="stretch", hide_index=True)
    if sp_on:
        st.info("Foreign-currency BBB spread is a manual override; USD results are unaffected by it.")

    with st.expander("Main assumptions & formulas", expanded=False):
        st.markdown(f"""
- **Shared inputs (both currencies):** unlevered beta, one project tax rate (**{tax_choice}**), equity risk premium
  ({pct(erp)}) and debt share ({pct(debt, 0)}).
- **Relevered beta:** β_L = β_U × (1 + (1 − t) × D/E), with D/E = D/V ÷ E/V.
- **USD:** Cost of equity = y_US + β_L × ERP; cost of debt = y_US + BBB spread; WACC = E/V × Ke + D/V × Kd × (1 − t).
- **Foreign currency:** the 10-yr government yield gap is the proxy for the expected inflation/rate gap.
  Conversion factor = (1 + y_FC) / (1 + y_US) = **{res['fx_factor']:.4f}**.
  Ke_FC = factor × (1 + Ke_USD) − 1. Kd_FC = y_FC + spread_FC, where the auto spread_FC = spread_USD × factor
  (equivalent to converting the USD cost of debt with the same factor). WACC_FC uses the same weights and tax rate.
- **Spread override:** replaces only the foreign-currency spread; USD results never change.
- Yields are used as risk-free proxies; no country default/sovereign spread is added beyond what is in the ERP input.
- Project tax rate is a statutory rate (OECD combined central + sub-central rate), not an effective rate.
""")

    # ---- All-industry comparison ----------------------------------------- #
    st.subheader("All-industry comparison")
    allind = None
    if betas_r["ok"]:
        allind = all_industries(betas_r["df"], tax=tax, erp=erp, debt_share=debt, y_us=y_us, y_fc=y_fc,
                                spread_usd=spread_usd, spread_fc_override=sp_fc)
        if beta_src == "manual override":
            st.caption("Note: the manual beta override applies to the selected industry only; the table uses Damodaran betas.")
        show = allind.copy()
        show.insert(0, "Selected", np.where(show["Industry"] == industry, "◀", ""))
        sty = show.style.format({c: "{:.2%}" for c in show.columns if "WACC" in c or "Cost of equity" in c}
                                | {"Unlevered beta": "{:.3f}", "Relevered beta": "{:.3f}"})
        st.dataframe(sty, width="stretch", hide_index=True)
        st.bar_chart(allind.sort_values("WACC (USD)").set_index("Industry")[["WACC (USD)", "WACC (FC)"]])
    else:
        st.info("Industry betas could not be fetched, so the all-industry table is unavailable. Press Refresh to retry.")

    # ---- Sources ---------------------------------------------------------- #
    st.subheader("Data sources, dates and values used")
    te_date = ymap.get(norm(country), (None, "", ""))[1] if ymap else ""
    src_rows = [
        ("Unlevered beta", URL_BETAS, beta_src, f"{beta_u:.3f}",
         betas_r.get("asof", "n/a") if betas_r["ok"] else "n/a", betas_r.get("fetched_at", "-")),
        ("USD BBB spread (Baa2/BBB)", URL_SPREAD, spread_src,
         pct(spread_usd, 3), spread_r.get("asof", FALLBACK_SPREAD_ASOF) if spread_r["ok"] else FALLBACK_SPREAD_ASOF + " (not used unless typed)",
         spread_r.get("fetched_at", "-")),
        ("US corporate tax rate", URL_TAX, tax_us_src, pct(tax_us),
         tax_r.get("asof", "bundled approximate snapshot") if tax_r["ok"] else "bundled approximate snapshot", tax_r.get("fetched_at", "-")),
        (f"{country} corporate tax rate", URL_TAX, tax_fc_src, pct(tax_fc),
         tax_r.get("asof", "bundled approximate snapshot") if tax_r["ok"] else "bundled approximate snapshot", tax_r.get("fetched_at", "-")),
        ("US 10-yr yield", URL_BONDS, y_us_src, pct(y_us, 3),
         (ymap.get("united states", (0, "", ""))[1] or "n/a") if ymap else "n/a", yields_r.get("fetched_at", "-")),
        (f"{country} 10-yr yield", URL_BONDS, y_fc_src, pct(y_fc, 3), te_date or "n/a", yields_r.get("fetched_at", "-")),
        ("Foreign-currency BBB spread", "calculated" if not sp_on else "manual", "manual override" if sp_on else "auto",
         pct(res["fc"]["spread"], 3), "-", "-"),
    ]
    src = pd.DataFrame(src_rows, columns=["Input", "Source", "Status", "Value used", "Data date / as-of", "Fetched at"])
    st.dataframe(src, width="stretch", hide_index=True,
                 column_config={"Source": st.column_config.LinkColumn("Source")})
    st.caption("Status 'fetched' = live value from the source; 'manual override' = typed by the user; "
               "'manual (live fetch failed)' = fetch failed and a typed value is being used.")

    # ---- Downloads -------------------------------------------------------- #
    st.subheader("Download results")
    res_df = tbl.drop(columns="kind")
    inputs_df = pd.DataFrame({
        "Input": ["Industry", "Country", "Currency", "Tax rate basis", "Tax rate", "ERP", "Debt share D/V",
                  "Unlevered beta", "USD BBB spread", "US 10Y", f"{country} 10Y", "FC spread override on"],
        "Value": [industry, country, ccy_label, tax_choice, tax, erp, debt, beta_u, spread_usd, y_us, y_fc, sp_on],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        inputs_df.to_excel(xw, sheet_name="Inputs", index=False)
        res_df.to_excel(xw, sheet_name="Results", index=False)
        if allind is not None:
            allind.to_excel(xw, sheet_name="All industries", index=False)
        src.to_excel(xw, sheet_name="Sources", index=False)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Results (CSV)", res_df.to_csv(index=False).encode(), "cost_of_capital_results.csv", "text/csv")
    if allind is not None:
        d2.download_button("⬇ All industries (CSV)", allind.to_csv(index=False).encode(), "all_industries.csv", "text/csv")
    d3.download_button("⬇ Full workbook (Excel)", buf.getvalue(), "cost_of_capital.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
