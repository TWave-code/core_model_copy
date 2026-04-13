# CORE — Methodology

---

## TL;DR

The model answers three practical questions:
- **How often do positions get liquidated?** (Probability of Liquidation, PL)
- **How often do liquidations fail to cover outstanding debt?** (Probability of Default, PD)
- **How much capital is needed to absorb tail losses?** (Capital Requirement Ratio, CRR)

---

## 0. Glossary

| Term | Definition |
|---|---|
| **CRR** | Capital Requirement Ratio — bad debt as a fraction of total exposure, at a given confidence level |
| **PL** | Probability of Liquidation — fraction of positions liquidated in a given scenario |
| **PD** | Probability of Default — fraction of positions generating unrecovered bad debt |
| **VaR** | Value at Risk — α-quantile of the loss distribution |
| **ES** | Expected Shortfall — expected loss conditional on exceeding VaR |
| **LTV** | Loan-to-Value ratio — outstanding debt / collateral value |
| **LT** | Liquidation Threshold — LTV level at which a position becomes eligible for liquidation |
| **HF** | Health Factor — (collateral value × LT) / debt; position is unsafe when HF < 1 |
| **EAD** | Exposure at Default — outstanding debt at the moment bad debt is first recorded |

---

## 1. Overview

The model integrates four components into a single pipeline:

1. **Calibrator** — fits and validates ARMA-GARCH models on historical return data
2. **Forecaster / Simulator** — generates correlated Monte Carlo price paths
3. **Aggregator** — constructs cross-asset copula dependence
4. **Liquidator** — simulates protocol-specific liquidation mechanics and computes bad debt

All components share a common data flow: calibrated model parameters drive the simulation, simulated prices drive the liquidation engine, and liquidation outcomes are aggregated into tail risk metrics.

---

## 2. Data Inputs

For each market, the model reconstructs the full on-chain state at the simulation start date.

**Position-level inputs (per borrower):**
- Collateral amounts and token identities
- Borrowed amounts and loan token identities
- Current LTV and Health Factor
- Liquidation threshold and liquidation bonus

**Market-level inputs (per collateral asset):**
- Daily OHLCV price history (up to 4 years, sourced from Yahoo Finance)
- Oracle price at simulation start
- Real-time order book depth from 12+ CEX venues and Uniswap V3

**Protocol parameters:**
- Close factor rules (Aave / SparkLend)
- Partial liquidation formula parameters (Morpho)
- Margin call thresholds and cure probabilities (Maple / Galaxy)
- Gas fee and swap fee assumptions

---

## 3. Return Dynamics

### 3.1 Log Returns

All models operate on daily log returns:

```
r_t = log(P_t / P_{t-1})
```

Log returns are preferred over simple returns for their additive property over time and their approximate symmetry for moderate price changes.

### 3.2 Mean Model — ARMA(p, q)

Where autocorrelation structure is present in the return series, a mean model of the form

```
r_t = c + φ_1 r_{t-1} + ... + φ_p r_{t-p} + ε_t + θ_1 ε_{t-1} + ... + θ_q ε_{t-q}
```

is fitted via maximum likelihood. The order (p, q) is selected by **BIC** over a grid search with p, q ≤ 5. If ARMA residuals already exhibit white noise (Ljung-Box test), the mean model is retained only for its residuals, which are then passed to the volatility model.

### 3.3 Volatility Model — GARCH Family

Whether a GARCH model is warranted is determined by an **ARCH-LM test** on the return series (or ARMA residuals). If heteroskedasticity is detected at p < 0.05, the following specifications are considered in order:

| Model | Variance Equation |
|---|---|
| **FIGARCH(1,1)** | Long-memory: fractional integration in the lag polynomial |
| **GJR-GARCH(1,1)** | `σ²_t = ω + (α + γ·𝟙[ε_{t-1}<0])·ε²_{t-1} + β·σ²_{t-1}` |
| **GARCH(1,1)** | `σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}` |
| **EGARCH(1,1)** | `log σ²_t = ω + α·(|z_{t-1}| − E|z|) + γ·z_{t-1} + β·log σ²_{t-1}` |

Each specification is tested with three innovation distributions: Normal, Student-t, and Skewed Student-t. The winning combination minimises BIC.

### 3.4 Model Validation

A candidate model is accepted only if it passes all three residual diagnostics:
- **Ljung-Box** on standardised residuals (no remaining autocorrelation in mean)
- **Ljung-Box** on squared standardised residuals (no remaining ARCH effects)
- **ARCH-LM test** on standardised residuals (no remaining heteroskedasticity)

Accepted candidates are then subject to a **rolling 1-step-ahead VaR backtest**. The model is trained on a window of `TRAIN_SIZE` days and the 1-day-ahead VaR is computed at level `backtest_alpha = 1 - PERC`. The window then rolls forward by 1 day, producing approximately (`N_history` − `TRAIN_SIZE`) non-overlapping hit observations — roughly 1 280 over a 4-year history with a 180-day training window. Two statistical tests are applied to the resulting hit sequence:

- **Kupiec POF test** — tests unconditional coverage: does the observed exceedance rate match `backtest_alpha`?
- **Christoffersen test** — tests conditional coverage: are exceedances independent over time?

Note: the backtest evaluates 1-step-ahead VaR only, while the simulation uses a `FORECAST_STEP`-day horizon. These are separate concerns — 1-step-ahead backtesting is the industry standard for model validation and provides far more statistical power than rolling by `FORECAST_STEP` days (which would yield only ~90 non-overlapping windows).

Both tests must have p-values ≥ 0.05 for a model to be accepted. If no model passes both tests, a **soft fallback** selects the candidate whose rolling exceedance rate is closest to `backtest_alpha`, ensuring a GARCH model is always used for assets with detected heteroskedasticity.

### 3.5 Volatility Floor

GARCH models are conditionally adaptive: they produce low volatility forecasts during calm market regimes, which can cause capital requirements to collapse precisely when the risk environment is benign but not necessarily safe. To prevent this procyclicality, the GARCH conditional volatility forecast is bounded below by:

```
vol_floor = Percentile(rolling_21d_std(r_t, full history), VOL_FLOOR_PCT)
```

The floor is computed from the **full historical price series** (not the training window), ensuring it remains stable regardless of the window size used for model estimation. `VOL_FLOOR_PCT = 0.75` corresponds to the 75th percentile of historical realised volatility, keeping the model in the upper half of observed conditions.

The 21-day rolling window is a convention, not a derived parameter. It approximates one trading month (21 business days), which is the standard lookback used in risk management for estimating "current" realized volatility: it's long enough to smooth out daily noise but short enough to be responsive to regime changes.

### 3.6 Jump Component (Optional)

When `JUMPS = True`, a compound Poisson jump process is added to the return:

```
J_t = N_t × j_t

N_t ~ Poisson(λ)         jump occurrence
j_t ~ Student-t(df, μ_j, σ_j)   jump size
```

Parameters are estimated from the tail of the historical return distribution. By default, bilateral tails are used (returns below the 2.5th or above the 97.5th percentile). Setting `FOCUS_ON_NEGATIVE = True` restricts calibration to the left tail only, and clips simulated jumps to be non-positive, which is more conservative for liquidation risk.

---

## 4. Cross-Asset Correlation

When multiple collateral tokens are present in a market, the model captures their joint tail behaviour via a **copula**. The key steps are:

**Step 1 — Spearman rank correlation.** Pairwise Spearman correlations are computed on the standardised GARCH residuals. Spearman correlation is preferred over Pearson because it is robust to outliers and captures monotonic dependence without requiring linearity.

**Step 2 — Conversion to copula correlation.** Spearman correlations are converted to the linear correlations used inside the copula via:

```
ρ_copula = 2 · sin(π · ρ_spearman / 6)
```

**Step 3 — PSD enforcement.** If the resulting correlation matrix is not positive semi-definite (due to rounding or incomplete pairwise observations), it is regularised using the Rebonato-Jäckel eigenvalue flooring method.

**Step 4 — Copula sampling.** Two copula types are supported:

- **Gaussian copula** — joint Normal dependence; correctly captures linear correlation but underestimates tail co-movement.
- **t-Copula** — joint Student-t dependence; adds tail dependence, meaning assets are more likely to crash together in extreme scenarios. This is the recommended setting for crypto collateral.

For single-token markets, uniform samples are drawn independently.

---

## 5. Price Simulation

For each of the `N_MC` Monte Carlo scenarios:

1. Sample a row of `FORECAST_STEP` correlated uniform variates from the copula.
2. Transform each uniform to an innovation via the fitted distribution's inverse CDF (t or Normal).
3. Combine with GARCH conditional volatility forecast and ARMA mean forecast:

```
r̂_t = μ̂_t + σ̂_t · z_t + J_t
```

4. Reconstruct prices from cumulative log returns:

```
P_t = P_0 · exp(Σ r̂_s for s=1 to t)
```

Cumulative log returns are clipped to [−log(5), log(5)] to prevent numerical overflow in extreme scenarios.

**Brownian Bridge (optional).** When `HOURLY_CONV = True`, each daily return is decomposed into 24 hourly sub-returns using a Brownian bridge conditioned on the daily endpoint. This allows the liquidation engine to check for threshold crossings at hourly frequency, better capturing intraday liquidation dynamics. This is the more realistic and less conservative configuration; daily-only simulation should be preferred when continuous liquidation cannot be assumed.

---

## 6. Liquidation Engine

The liquidation engine processes each borrower position step by step along each simulated price path.

### 6.1 State Update

At each time step, collateral values are recomputed using simulated prices, and LTVs and Health Factors are updated accordingly.

### 6.2 Margin Call Logic (Maple / Galaxy)

For products with a margin call mechanism, when a position breaches the margin call threshold, the borrower is given a probabilistic opportunity to self-cure. The cure probability is calibrated from historical data provided by the asset manager. If cure occurs, the borrower posts additional collateral to restore the LTV to the margin call threshold. If no cure occurs, the position proceeds to liquidation.

### 6.3 Liquidation Trigger

A position is eligible for liquidation when:

```
LTV ≥ LT   ↔   HF < 1
```

### 6.4 Repayment Amount

**Morpho** uses partial liquidation, restoring the position exactly to the liquidation threshold:

```
R_req = (LT × CV − D) / (LT × (1 + bonus) − 1)
```

where CV is collateral value and D is outstanding debt.

**Aave / SparkLend** use a close-factor approach:
- HF > 0.95 → repay 50 % of outstanding debt
- HF ≤ 0.95 → repay 100 % of outstanding debt

In all cases, `R_req` is bounded by available collateral.

### 6.5 Profitability Constraint

A liquidation is only executed if the liquidator earns a non-negative profit:

```
proceeds = (1 − swap_fee − slippage) × (1 + bonus) × R_req
profit   = proceeds − R_req − gas_fee_usd
```

If `profit < 0`, no liquidation occurs and the full outstanding debt is recorded as bad debt at that step.

### 6.6 Slippage Modelling

Slippage is computed from a synthetic order book aggregated across 12+ CEX venues (Binance, Bybit, OKX, Kraken, Coinbase, Gate.io, KuCoin, Huobi, Bitget, Bitfinex, Crypto.com) and Uniswap V3. For a required liquidation of size `R_req`, the model:

1. Identifies available sell-side liquidity at prices ≤ the simulated price
2. Computes the average execution price by consuming order book depth sequentially
3. Derives slippage as the deviation of average execution price from the mid price

The order book is fetched once at simulation start and held static over the forecast horizon. This is a conservative assumption: in a real stress event, liquidity is likely to thin further as prices decline.

### 6.7 Bad Debt Accounting

Bad debt is recorded using "count once" semantics:
- When a position first becomes unsafe and cannot be liquidated profitably, the full EAD is recorded as bad debt
- In subsequent steps, if the liquidation becomes profitable (e.g., volatility subsides), any recovered amount reduces the outstanding bad debt
- Net bad debt = EAD − cumulative recoveries

This correctly distinguishes between exposure at default and net economic loss.

---

## 7. Risk Metrics

Scenario-level net bad debt figures are aggregated into tail risk metrics at the `PERC` confidence level:

| Metric | Definition |
|---|---|
| **CRR (VaR)** | α-quantile of (Net Bad Debt / Total Exposure) |
| **CRR (ES)** | E[Net Bad Debt / Total Exposure \| scenario ≥ VaR] |
| **PL** | α-quantile of the fraction of positions liquidated per scenario |
| **PD** | α-quantile of the fraction of positions with net bad debt > 0 |
| **Delta LTV** | α-quantile of the maximum LTV overshoot above LT across all positions |

ES is the preferred metric for capital setting because it is coherent (subadditive) and more sensitive to the severity of tail scenarios, not just their frequency.

---

## 8. Model Calibration Choices and Limitations

### Calibration choices

| Choice | Rationale |
|---|---|
| BIC for model selection | Penalises complexity more heavily than AIC; prevents overfitting on short training windows |
| ARCH-LM gate for GARCH | Ljung-Box on levels detects mean autocorrelation, not variance clustering; ARCH-LM is the correct pre-test for GARCH |
| 1-step-ahead backtest rolling by 1 day | Rolling by `FORECAST_STEP` days produces only ~90 non-overlapping windows over 4 years — too sparse for reliable Kupiec / Christoffersen tests. Rolling by 1 day gives ~1280 non-overlapping 1-day hits, providing proper statistical power. The 1-step-ahead horizon is the standard for VaR model validation; the multi-step simulation horizon is a separate concern. |
| Soft backtest fallback | Hard rejection of all GARCH models when none passes formal tests causes a regression to constant-volatility forecasting; the least-bad GARCH candidate is always preferable |
| t-Copula over Gaussian | Crypto assets exhibit strong tail co-dependence; a Gaussian copula underestimates the probability of simultaneous crashes |
| Volatility floor from full history | A floor computed from the training window is self-referential and moves with window size; anchoring to the full series gives a stable long-run stress reference |
| Spearman over Pearson for correlation | Robust to outliers; captures monotonic relationships without requiring linearity |

### Known limitations

| Limitation | Impact |
|---|---|
| Static order books | Liquidity may deteriorate during stress; the model likely understates slippage in severe scenarios |
| No cascading liquidations | Large-scale simultaneous selling depresses prices further; second-order market impact is not modelled |
| No interest rate dynamics | Borrow rates spike during high-utilisation stress events; fixed-rate assumption is optimistic |
| Oracle risk not modelled | Stale oracles or oracle manipulation are not captured |
| Stablecoin collateral filtered | Peg risk for USDC, USDT, and crypto-backed stablecoins is not modelled; positions using stablecoin collateral are excluded from price simulation |

---

**Key observations:**
- Hourly liquidation granularity materially reduces CRR relative to daily, reflecting realistic continuous liquidation dynamics. Daily-only simulation is the more conservative, prudential configuration.
- The CORE consistently produces higher CRR estimates than the Lending Model, particularly under daily assumptions, due to more explicit slippage and profitability modelling.
- In addition to CRR, the model provides PL, PD, and Delta LTV, giving a richer view of the risk drivers beyond a single capital number.

---

*End of Documentation*
