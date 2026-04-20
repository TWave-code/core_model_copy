# CORE - Collateralized Onchain Risk Engine

A quantitative framework for computing the **Capital Requirement Ratio (CRR)** across over-collateralised DeFi lending protocols. The model combines ARMA-GARCH price simulation, copula-based cross-asset correlation, an optional compound Poisson jump component, and full liquidation mechanics to estimate the **Expected Loss (EL)** of bad-debt exposure — the primary risk metric — together with concentration diagnostics based on the Herfindahl-Hirschman Index (HHI) of borrower exposures.

---

## Supported Protocols

| Protocol | Data Source |
|---|---|
| **Morpho** | Parquet files |
| **SparkLend** | Parquet files |
| **Maple** | Parquet files |
| **Galaxy** | Parquet files |
| **Anchorage** | Parquet files |

---

## Architecture

```
main.py               Entry point — orchestrates the full pipeline
│
├── importer.py       Protocol-specific data loaders (users + market data) and prices / orderbook data
│
├── calibrator.py     ARMA / GARCH-family model selection, diagnostics, backtesting
│   └── backtester.py Rolling VaR backtests (Kupiec + Christoffersen)
│
├── forecaster.py     Monte Carlo price simulation (Forecaster + Simulator)
│   └── aggregator.py Cross-asset copula construction (Gaussian / t-Copula)
│
└── liquidator.py     Liquidation mechanics + bad debt / CRR calculation
```

### Pipeline

| Step | Module | Description |
|---|---|---|
| 1 | `importer.py` | Fetch live borrower positions and market parameters plus price and orderbook data for each modelled token |
| 2 | `calibrator.py` | Fit ARMA(p,q)-GARCH-family models on daily log returns; select best specification by BIC; validate with ARCH-LM diagnostics and rolling Kupiec / Christoffersen backtests |
| 3 | `calibrator.py` | Optionally fit a compound Poisson jump process with Student-t jump sizes to tail return observations |
| 4 | `forecaster.py` / `aggregator.py` | Generate `N_MC` correlated price scenarios via a Gaussian or t-Copula; optionally decompose to hourly resolution using a Brownian bridge |
| 5 | `liquidator.py` | For each scenario, apply protocol-specific liquidation rules, compute liquidator profit (after gas and slippage), and accumulate bad debt |
| 6 | `liquidator.py` | Compute the Expected Loss (EL) and concentration-adjusted CRR from the bad-debt distribution |

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `PROTOCOL` | `MORPHO` | Target protocol |
| `NETWORK` | `ETHEREUM` | Target network (useless in this script) |
| `FORECAST_STEP` | `14` | Forecast horizon (days) |
| `TRAIN_SIZE` | `180` | Rolling training window (days) |
| `N_MC` | `10 000` | Monte Carlo scenarios |
| `PERC` | `0.975` | VaR / ES confidence level |
| `COPULA_TYPE` | `T-COPULA` | Cross-asset dependence structure (`GAUSSIAN` or `T-COPULA`) |
| `HOURLY_CONV` | `False` | Decompose daily returns to hourly via Brownian bridge |
| `USE_LOG_RETURNS` | `True` | Use log returns instead of simple returns |
| `JUMPS` | `False` | Include compound Poisson jump component |
| `FOCUS_ON_NEGATIVE` | `False` | Restrict jump simulation to downside only |
| `VOL_FLOOR_PCT` | `0.75` | Floor GARCH forecast vol at this percentile of the full historical rolling vol |
| `WORST_CASE` | `False` | Use worst-case LTVs instead of observed LTVs |
| `LOAN_TOKEN` | `USDC` | Filter positions by loan token (`ALL` = no filter) |
| `SEED` | `0` | Global random seed |

---

## Volatility Models

The calibrator performs a grid search over GARCH-family specifications, each tested with Normal, Student-t, and Skewed-t innovations. The winning model is selected by **BIC** and must:

1. Pass residual diagnostics: Ljung-Box on standardised residuals and squared residuals, plus ARCH-LM test
2. Pass rolling VaR backtests: **Kupiec** (unconditional coverage) and **Christoffersen** (conditional coverage / independence) at `backtest_alpha = 1 - PERC`

Models tested (in order of preference):

| Model | Characteristic |
|---|---|
| FIGARCH(1,1) | Long-memory volatility |
| GJR-GARCH(1,1) | Asymmetric response to negative shocks |
| GARCH(1,1) | Standard volatility clustering |
| EGARCH(1,1) | Leverage effects, log-variance formulation |

If no model passes both backtests, a **soft fallback** selects the candidate whose rolling exceedance rate is closest to `backtest_alpha`, rather than discarding GARCH entirely.

The GARCH gate is determined by an **ARCH-LM test** on the return residuals: GARCH is only fitted when heteroskedasticity is statistically detected, regardless of whether the mean model residuals are already white noise.

### Volatility Floor

To prevent capital requirements from collapsing during low-volatility regimes, the GARCH conditional volatility forecast is floored at the `VOL_FLOOR_PCT` percentile of the 21-day rolling realised volatility computed over the **full historical series** (not just the training window). This decouples the floor from the training window choice and provides a stable long-run anchor.

---

## Liquidation Mechanics

### Morpho
Partial liquidation up to the repayment amount `R_req` that restores the position exactly to the liquidation threshold:

```
R_req = (LT × CV − D) / (LT × (1 + bonus) − 1)
```

### Aave / SparkLend
Close-factor liquidation based on Health Factor:
- **HF > 0.95** → 50 % of outstanding debt repaid
- **HF ≤ 0.95** → 100 % of outstanding debt repaid

### Liquidator Profitability Constraint
Liquidation is only executed if the liquidator makes a non-negative profit:

```
proceeds = (1 − swap_fee − slippage) × (1 + bonus) × R_req
profit   = proceeds − R_req − gas_fee_usd  ≥ 0
```

If the constraint is not met, the position is marked as bad debt equal to the full exposure at default (EAD). Subsequent scenario steps where partial recovery becomes viable reduce the net bad debt figure.

### Slippage
Slippage is modelled from aggregated real-time order books across 12+ CEX venues and Uniswap V3. Available depth at each price level is used to compute the average execution price for the required liquidation size. The order book is dynamic over the forecast horizon in the sense that the liquidity is consumed as liquidations happen.

---

## Multi-Loan Positions (Aave / SparkLend)

When a wallet borrows multiple assets, `explode_by_loan_token()` creates one synthetic sub-position per loan token, re-weighting collateral by each loan's share of total borrow. This preserves LTV and Health Factor invariants:

```
LTV_token = D_token / (C × share)  =  D_total / C       (unchanged)
HF_token  = (C × share × LT) / D_token                   (unchanged)
```

Setting `LOAN_TOKEN = "ALL"` analyses the full mixed-borrow portfolio; setting it to a specific token isolates that market.

---

## Risk Metrics

| Metric | Definition |
|---|---|
| **CRR (EL)** | Mean (Net Bad Debt / Total Exposure) across all `N_MC` scenarios — the Basel Expected Loss analog; the primary headline metric |
| **HHI** | Herfindahl-Hirschman Index of borrower exposures: `Σ (borrow_i / total_borrow)²`; ranges from 0 (perfectly granular) to 1 (single borrower) |
| **PL** | `PERC`-quantile of the fraction of positions liquidated |
| **PD** | `PERC`-quantile of the fraction of positions generating bad debt |
| **Delta LTV** | `PERC`-quantile of the maximum LTV overshoot above the liquidation threshold |

CRR (EL) is the headline metric. It equals `PD × LGD × EAD` in Basel notation — the expected cost of lending expressed as a fraction of total exposure. It is stable under borrower concentration and directly comparable across protocols and market segments. VaR and ES of the bad-debt distribution are computed internally and available as diagnostics but are not the primary output.

---

## Installation

Requires **Python 3.10+**.

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

---

## Usage

Configure the parameters at the top of `main.py`, then:

```bash
python main.py
```

### Example Output

```
Loading Market and Users Data...

Processing Token: WBTC

Importing Prices DataFrame...
Last Close Price of WBTC: 71755.40

ARCH-LM test p-value: 0.0000 → GARCH warranted
Selected GARCH model: GJR-GARCH

Running Monte-Carlo Simulations for Prices...

Simulated Prices Statistics for WBTC:
            2026-04-10     2026-04-23
mean        71,760.99      72,132.22
std          1,511.87       6,727.74
min         66,598.84      50,401.83
max         78,439.61      99,453.16

Running Monte-Carlo for Liquidations...

CRR as EL (Basel):     2.34%
HHI (concentration):   0.142
PL         at 99.50%:  50.00%
PD         at 99.50%:  50.00%
Delta LTV  at 99.50%:  26.50%
```

---

## Project Structure

```
CORE/
├── main.py
├── importer.py
├── calibrator.py
├── forecaster.py
├── aggregator.py
├── backtester.py
├── liquidator.py
├── requirements.txt
├── README.md
├── inputs/
│   ├── galaxy_users.csv
│   ├── galaxy_market.csv
│   ├── anchorage_users.csv
│   └── anchorage_market.csv
├── documentation/
    └── internal_model_doc.md
```
