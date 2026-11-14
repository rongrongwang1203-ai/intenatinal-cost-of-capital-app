# intenatinal-cost-of-capital-app
intenatinal-cost-of-capital-app
# International Cost of Capital App

A Streamlit app that compares a project's **cost of equity, cost of debt and WACC in USD and in the currency of a selected country**, using the same beta, tax rate, equity risk premium and debt share in both currencies.

## Data sources (fetched live, with dates shown in the app)
| Input | Source |
|---|---|
| US industry unlevered betas | Damodaran `betas.xls` ("Unlevered beta" column) |
| USD BBB spread (Baa2/BBB, large non-financial firms) | Damodaran ratings page |
| Corporate tax rates | OECD Corporate Tax Statistics 2024 (statutory rates) |
| 10-year government yields (US and selected country) | Trading Economics `/bonds` |

A **Refresh data** button clears the cache and re-fetches everything. Every fetched input can be overridden manually. If a fetch fails the app says so and requires a manual value (the tax table and BBB spread also have a bundled, clearly-labelled fallback snapshot).

## Features
- Defaults: Software (System & Application), United States, US tax rate, 6.5% ERP, 30% debt.
- Choose industry, target country, ERP, debt share; switch the project tax rate between the US and the selected country.
- Manual overrides: unlevered beta, US / country tax rates, USD BBB spread, both 10-yr yields, and (separate switch) the foreign-currency BBB spread. With the switch off, the auto-calculated foreign spread is shown.
- Side-by-side USD vs foreign-currency results, an assumptions panel, an all-industry comparison (table + chart), a sources/dates table, and CSV/Excel downloads.

## Formulas
- D/E = (D/V)/(E/V); relevered beta = β_U × (1 + (1 − t) × D/E)
- USD: Ke = y_US + β_L × ERP; Kd = y_US + BBB spread; WACC = E/V·Ke + D/V·Kd·(1 − t)
- Foreign currency: f = (1 + y_FC)/(1 + y_US); Ke_FC = f·(1 + Ke_USD) − 1; auto spread_FC = spread_USD × f (so Kd_FC = y_FC + spread_FC = f·(1 + Kd_USD) − 1); WACC_FC uses the same weights and tax rate.

The currency-conversion step lives in `compute_wacc()` in `app.py`; adjust it there if your course formulas differ.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
python verify.py        # calculation checks + parser tests + app smoke test
```

## Deploy
1. Create a GitHub repo and upload `app.py`, `requirements.txt`, `README.md` (optionally `verify.py`).
2. On https://share.streamlit.io choose *New app*, pick the repo, branch `main`, main file `app.py`, then *Deploy*.

## Verification (see `verify.py`)
- Independent hand calculation matches the app.
- Equal government yields (foreign spread override off) → identical USD and foreign results.
- Zero debt → WACC = cost of equity (and relevered beta = unlevered beta).
- Overriding the foreign-currency spread leaves all USD results unchanged.

## Limitations
- Scrapers depend on the sources' page layouts; Trading Economics and OECD may block automated requests or change layout, in which case use manual overrides.
- Statutory tax rates are used, not effective rates. Yields are risk-free proxies and no sovereign default spread is added.
