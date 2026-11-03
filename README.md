# Nifty 50 Volatility Modeling: GARCH, GJR-GARCH and EGARCH

A 10-cell Jupyter notebook that downloads daily index data from Yahoo Finance, fits three conditional-volatility models, compares them, and simulates forward volatility and return paths.

## Contents

1. [Overview](#overview)
2. [Models](#models)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Notebook walkthrough](#notebook-walkthrough)
6. [Configuration](#configuration)
7. [Interpreting the output](#interpreting-the-output)
8. [Known limitations](#known-limitations)
9. [Troubleshooting](#troubleshooting)
10. [Extending the notebook](#extending-the-notebook)
11. [Disclaimer](#disclaimer)

---

## Overview

| Item | Detail |
|---|---|
| Instrument | Nifty 50 (`^NSEI`); swap to `^NSEBANK` for BankNifty |
| Data source | Yahoo Finance via `yfinance` |
| Frequency | Daily (`1d`) |
| Window | 2006-01-01 to today |
| Returns | Log returns, scaled to percent (x100) |
| Error distribution | Student-t (captures fat tails) |
| Models | GARCH(1,1), GJR-GARCH(1,1,1), EGARCH(1,1,1) |
| Simulation | 1,000 paths, 22 trading days (about 1 month) ahead, per model |

**Data note:** Yahoo's `^NSEI` history starts around September 2007, so although the start date is set to 2006, the effective sample is roughly 19 years. The first cell of output prints the actual first and last dates so you can confirm.

## Models

All three models describe the conditional variance sigma^2_t of returns as a function of past shocks and past variance.

**GARCH(1,1)**

```
sigma^2_t = omega + alpha * eps^2_(t-1) + beta * sigma^2_(t-1)
```

- Symmetric: a +3% and a -3% shock raise variance equally.
- Stationary if `alpha + beta < 1`. Long-run variance is `omega / (1 - alpha - beta)`.

**GJR-GARCH(1,1,1)**

```
sigma^2_t = omega + (alpha + gamma * I_(t-1)) * eps^2_(t-1) + beta * sigma^2_(t-1)
```

- `I = 1` when the previous shock was negative, else 0.
- `gamma > 0` means bad news raises variance more than good news (the leverage effect).
- Stationary if `alpha + gamma/2 + beta < 1`.

**EGARCH(1,1,1)**

```
ln sigma^2_t = omega + alpha * (|z_(t-1)| - E|z|) + gamma * z_(t-1) + beta * ln sigma^2_(t-1)
```

- Models log variance, so positivity is automatic and no parameter constraints are needed.
- `gamma < 0` indicates the leverage effect (sign convention differs from GJR).
- Large shocks have an exponential impact, so upper-tail forecasts can be extreme.

| | GARCH | GJR | EGARCH |
|---|---|---|---|
| Asymmetry | No | Yes (threshold) | Yes (smooth) |
| Models | variance | variance | log variance |
| Parameter constraints | Yes | Yes | No |
| Multi-step forecast | Closed form | Closed form | Simulation |

## Requirements

- Python 3.9+
- Internet access (Yahoo Finance download)
- Packages: `yfinance`, `arch`, `statsmodels`, `pandas`, `numpy`, `matplotlib`

Cell 1 installs the packages with `pip`. If you manage environments yourself:

```bash
pip install yfinance arch statsmodels pandas numpy matplotlib
```

## Quick start

1. Open a new Jupyter notebook (JupyterLab, VS Code, Colab, etc.).
2. Paste each cell from the walkthrough below into its own cell, in order.
3. Run all cells top to bottom (`Run All`).
4. Total runtime is typically under a minute on a laptop, dominated by the download and the simulations.

Always run cells in order; later cells depend on `r`, `fits`, `vol` and `sims` defined earlier.

## Notebook walkthrough

| Cell | Purpose | Key outputs |
|---|---|---|
| 1 | Install and import libraries, set plot size | None |
| 2 | Download daily prices with `yf.download`, flatten the column index, keep `Close` | First/last date, row count, price chart |
| 3 | Compute log returns `100 * ln(P_t / P_(t-1))` | Descriptive stats, returns chart |
| 4 | Fit GARCH, GJR-GARCH and EGARCH with a constant mean and Student-t errors | Full summary of the GJR model |
| 5 | Build a comparison table | Log-likelihood, AIC, BIC, gamma, gamma p-value |
| 6 | Plot annualized conditional volatility (`sigma_t * sqrt(252)`) for all three models | Overlay chart, last 5 rows |
| 7 | Zoom into stress windows (2008-2009 and 2020) | Two side-by-side charts |
| 8 | Ljung-Box test on squared standardized residuals at lags 10 and 20 | p-values per model |
| 9 | Simulate 1,000 forward paths of 22 days per model and plot volatility paths | Three-panel path chart with mean line |
| 10 | Compare simulated volatility bands and 22-day cumulative return distributions | Band chart, histogram, summary table |

## Configuration

| What to change | Where | Notes |
|---|---|---|
| Ticker | Cell 2, `yf.download("^NSEI", ...)` | Use `^NSEBANK` for BankNifty |
| Start date | Cell 2, `start` | Limited by what Yahoo provides |
| Error distribution | Cell 4, `dist="t"` | Options include `"normal"`, `"skewt"`, `"ged"` |
| Mean model | Cell 4, `mean="Constant"` | Try `"Zero"` or `"AR"` with `lags=1` |
| Model orders | Cell 4, `specs` | Increase `p`, `q` to test higher orders |
| Forecast horizon | Cell 9, `H` | Default 22 trading days |
| Number of paths | Cell 9, `N` | Default 1,000; raise for smoother percentiles |
| Annualization | Cells 6, 9 | Uses 252 trading days; Indian markets are close to this |
| Stress windows | Cell 7 | Edit the date tuples |

## Interpreting the output

**Cell 5, the comparison table**

- Lower AIC/BIC is better; higher log-likelihood is better.
- `gamma_pval` below 0.05 means asymmetry is statistically significant.
- Expected sign: GJR `gamma > 0`, EGARCH `gamma < 0`. For equity indices, both asymmetric models usually beat plain GARCH, and the gap between them is typically small. Choose on diagnostics and on whether you need closed-form multi-step forecasts.
- The plain GARCH row has no gamma, so it shows `NaN`.

**Cell 6 and 7, conditional volatility**

- Volatility spikes should line up with known stress events (2008 financial crisis, 2020 COVID crash).
- Asymmetric models usually react faster to sharp falls than plain GARCH.
- Values are annualized percentages. A reading of 20 means about 20% annualized volatility.

**Cell 8, residual diagnostics**

- p-values above 0.05 suggest the model has absorbed the volatility clustering.
- Small p-values mean leftover ARCH effects; consider higher orders or a different distribution.

**Cell 9 and 10, simulation**

- Each blue line in Cell 9 is one simulated volatility path; the red line is the mean across paths. Mean paths drift toward the model's long-run volatility.
- Cell 10's shaded bands show the 5th to 95th percentile of simulated volatility by day.
- The summary table reports average annualized volatility, the 95th percentile of volatility, and the 5% 22-day value-at-risk in percent (the 5th percentile of cumulative simulated returns; more negative means a larger tail loss).
- EGARCH will often show a fatter upper tail because shocks enter exponentially.

## Known limitations

- **Data quality:** Yahoo data can contain gaps, adjustments, or occasional bad prints. Inspect `px` before trusting results.
- **History length:** Nifty data begins around 2007 on Yahoo, not 2006.
- **Single-asset, in-sample fit:** Models are fit on the full sample, so the charts in Cells 6 and 7 are in-sample. They do not show out-of-sample forecasting skill.
- **No backtest:** The notebook does not evaluate forecast accuracy (for example, rolling forecasts versus realized volatility or VaR exceedance tests).
- **Constant parameters:** Parameters are assumed fixed across regimes, which may not hold across 19 years.
- **Simulated VaR is model-based:** It depends on the fitted distribution and parameters and is not a guarantee of actual losses.
- **Daily close data only:** Intraday and overnight-gap dynamics are not modeled.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `ConvergenceWarning` during fit | Returns not scaled; confirm they are in percent. Or raise iterations: `fit(disp="off", options={"maxiter": 500})` |
| Empty or short DataFrame from `yf.download` | Rate limiting or ticker issue. Re-run, update `yfinance` (`pip install -U yfinance`), or check the ticker |
| `KeyError: 'Close'` | Column index not flattened. Keep the `df.columns.get_level_values(0)` line in Cell 2 |
| `gamma[1]` missing in the table | Normal for plain GARCH; only GJR and EGARCH have it |
| Cell 9 `forecast` errors on simulation | Make sure the `arch` version is recent and `method="simulation"` is used; analytic forecasts are not available for EGARCH multi-step |
| Cell 9 slow | Reduce `N` (for example, 500) |
| Plots not showing | Add `%matplotlib inline` at the top of Cell 1 |
| Network blocked in sandbox | Allow outbound access to Yahoo Finance, or download the CSV locally and load it with `pd.read_csv` |

## Extending the notebook

- **BankNifty or other indices:** Change the ticker and rerun; compare persistence (`alpha + beta`) across instruments.
- **Skewed-t errors:** Use `dist="skewt"` to capture return skewness.
- **Rolling out-of-sample test:** Refit on an expanding window and compare one-day-ahead forecasts to realized squared returns or a realized-volatility proxy.
- **Volatility for options work:** Compare model volatility forecasts with India VIX (`^INDIAVIX`) to study the gap between realized and implied volatility, useful when choosing strikes for volatility-based option strategies.
- **Mean models:** Add `AR(1)` terms if the returns show autocorrelation.
- **Exports:** Save outputs with `vol.to_csv("conditional_vol.csv")` and `summary.to_csv("sim_summary.csv")`.

## Disclaimer

This notebook is for education and research. It is not investment advice. Volatility models are estimates built on historical data and can fail in regimes unlike those in the sample. Validate any model out-of-sample before using it in trading or risk decisions.
