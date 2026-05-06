# CORE - Collateralized Onchain Risk Engine

A quantitative framework for computing the **Capital Requirement Ratio (CRR)** across over-collateralised DeFi lending protocols. The model combines ARMA-GARCH price simulation, copula-based cross-asset correlation, an optional compound Poisson jump component, and full liquidation mechanics to estimate the **Expected Loss (EL)** of bad-debt exposure (the primary risk metric) together with concentration diagnostics based on the Herfindahl-Hirschman Index (HHI) of borrower exposures.

Note that CRR is an expected loss, not a tail loss by construction. However, the inputs that generate bad debt in the model are deliberately conservative: volatility is floored at its 75th historical percentile, liquidity is consumed cumulatively across sequential liquidations without replenishment, and joint tail events across collateral assets are modelled using a t-Copula that assigns materially higher probability to simultaneous crashes than standard correlation assumptions: a bad debt event in this model already presupposes a severe stress scenario.

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

## Data Sources

The model draws from three distinct data layers. Each is fetched independently and at a different cadence.

### 1 — Protocol position data

Borrower-level positions (collateral amounts, debt, LTV, liquidation threshold, liquidation bonus) are fetched from parquet files.

### 2 — Price data

All collateral price histories are downloaded from **Yahoo Finance** (`yfinance`, `period="max"`) at calibration time. For the purposes of this script, however, a precomputed snapshot of these prices is loaded from a parquet file.

### 3 — Order book / liquidity data

Order book depth is uploaded from a parquet files. Routing depends on the collateral token:

| Collateral token | Venue type | Source | Notes |
|---|---|---|---|
| **CBBTC** | DEX | Uniswap V3 | Pool `0xfB...43ef` (cbBTC/USDC, Base) — on-chain pool state |
| **HYPE** (and variants) | DEX | HyperLiquid | Native HyperLiquid order book |
| **ETH and LSTs** (WETH, WEETH, STETH, WSTETH, RETH) | CEX | Aggregated | Proxied via ETH spot book across 11 venues |
| **BTC and wrappers** (WBTC, LBTC, TBTC) | CEX | Aggregated | Proxied via BTC spot book across 11 venues |
| **SOL** | CEX | Aggregated | Direct SOL spot book across 11 venues |
| **All other tokens** | CEX | Aggregated | Direct spot book across 11 venues |

CEX aggregation covers: **Binance, Bybit, OKX, Kraken, Coinbase, Gate.io, KuCoin, Huobi, Bitget, Bitfinex, Crypto.com**.

Liquidity is consumed **cumulatively** across liquidation events within a scenario: each successive liquidation starts from the point in the book where the previous one left off, rather than assuming a fully replenished book.

---

## Architecture

```
main.py               Entry point — orchestrates the full pipeline
│
├── importer.py       Protocol-specific data loaders (users + market data) plus prices and orderbook data
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
| 1 | `importer.py` | Fetch borrower positions and market parameters plus price and orderbook data for each modelled token |
| 2 | `calibrator.py` | Fit ARMA(p,q)-GARCH-family models on daily log returns; select best specification by BIC; validate with ARCH-LM diagnostics and rolling Kupiec / Christoffersen backtests |
| 3 | `calibrator.py` | Optionally fit a compound Poisson jump process with Student-t jump sizes to tail return observations |
| 4 | `forecaster.py` / `aggregator.py` | Generate `N_MC` correlated price scenarios via a Gaussian or t-Copula; optionally decompose to hourly resolution using a Brownian bridge |
| 5 | `liquidator.py` | For each scenario, apply protocol-specific liquidation rules, compute liquidator profit (after gas and slippage), and accumulate bad debt. Compute finally risk metrics |

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `PROTOCOL` | `MORPHO` | Target protocol |
| `NETWORK` | `ETHEREUM` | Target network |
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
| `LINDY_ALPHA` | `0.0` | Lindy vol scaling exponent — `0.0` = disabled; `0.5` = square-root decay (recommended starting point) |
| `LINDY_REF_DAYS` | `1825` | Reference history length (days) at which the Lindy factor equals 1.0 (≈ 5 years) |
| `LINDY_MAX_FACTOR` | `2.0` | Hard cap on the Lindy vol multiplier |
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

### Lindy Volatility Scaling (optional)

When `LINDY_ALPHA > 0`, a multiplicative uncertainty premium is applied to the GARCH conditional volatility forecast for assets whose price history is shorter than `LINDY_REF_DAYS`:

```
lindy_factor = min(LINDY_MAX_FACTOR,  max(1.0,  (LINDY_REF_DAYS / n_obs) ^ LINDY_ALPHA))
vol_forecast = vol_forecast × lindy_factor
```

The rationale is that GARCH parameters estimated on a short history carry wide confidence intervals, and the sample may cover only a single market regime. Tokens with longer histories (≥ `LINDY_REF_DAYS` days) receive a factor of 1.0 — no adjustment. Tokens with shorter histories receive a factor > 1.0, scaled by the exponent `LINDY_ALPHA`:

| Token (examples with α = 0.5, ref = 1825 d) | n_obs | Lindy factor |
|---|---|---|
| BTC / ETH (≥ 5 years) | ≥ 1825 | 1.00 |
| SOL (≈ 3 years) | ≈ 1095 | 1.29 |
| WIF / PENDLE (≈ 2 years) | ≈ 730 | 1.58 |
| HYPE (≈ 6 months) | ≈ 180 | 2.00 (capped) |

`LINDY_ALPHA = 0.0` (the default) disables the feature entirely — the factor is always 1.0 with no effect on model output.

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

## Risk Metrics

| Metric | Definition |
|---|---|
| **CRR (EL)** | Mean (Net Bad Debt / Total Exposure) across all `N_MC` scenarios — the Basel Expected Loss analog; the primary headline metric |
| **HHI** | Herfindahl-Hirschman Index of borrower exposures: `Σ (borrow_i / total_borrow)²`; ranges from 0 (perfectly granular) to 1 (single borrower) |
| **PL** | `PERC`-quantile of the fraction of positions liquidated |
| **PD** | `PERC`-quantile of the fraction of positions generating bad debt |
| **Delta LTV** | `PERC`-quantile of the maximum LTV overshoot above the liquidation threshold |

CRR (EL) is the headline metric. It equals `PD × LGD` in Basel notation — the expected cost of lending expressed as a fraction of total exposure. It is stable under borrower concentration and directly comparable across protocols and market segments. VaR and ES of the bad-debt distribution are computed internally and available as diagnostics but are not the primary output.

---

## Further Developments

The following extensions are under consideration for future model iterations:

- Idle capital risk — stablecoins: extension of the CRR framework to cover idle capital held in stablecoins, which is currently excluded from the model perimeter.
- Idle capital risk and collateral — RWA tokens: integration of tokenised real-world assets both as a form of idle capital and as accepted collateral, accounting for the distinct risk structure of these instruments relative to native crypto assets.
- PT tokens: extension of the liquidation and price simulation framework to cover fixed-rate DeFi instruments whose price dynamics depend on both interest rate movements and protocol credit risk.
- DEX liquidity integration: broader coverage of decentralised exchange venues for order book depth aggregation, relevant for collateral tokens whose liquidity resides primarily or exclusively on-chain.

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
HHI (concentration):   14.2%
PL         at 97.50%:  50.00%
PD         at 97.50%:  50.00%
Delta LTV  at 97.50%:  26.50%
```

---

## Input Files (Galaxy / Anchorage)

For protocols without a live API, place CSV files in the `inputs/` folder:

```
inputs/
├── galaxy_users.csv
├── galaxy_market.csv
├── anchorage_users.csv
└── anchorage_market.csv
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