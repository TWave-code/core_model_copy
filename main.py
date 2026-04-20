import json
import os
import pandas as pd
import numpy as np

from calibrator import Calibrator
from forecaster import Simulator
from liquidator import Liquidator
from config import load_params
import importer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def _load_protection_usd(
    protocol: str
) -> float:
    """
    Read inputs/protocol_defense.json and return the total USD protection
    (sum of present loss absorbers) for the given protocol.
    Returns 0.0 if the file is missing or the protocol has no entry.
    """
    defense_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "inputs", "protocol_defense.json"
    )
    try:
        with open(defense_path, "r") as f:
            data = json.load(f)
        return float(data.get(protocol.upper(), {}).get("total_protection_usd", 0))
    except Exception:
        return 0.0


# 0.A - SET FUND AND MODEL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# Parameters are loaded from inputs/default_params.json via config.load_params().
# To customise a run, either:
#   (a) edit the values below directly, or
#   (b) pass a flat JSON file:  load_params(path="my_run.json", overrides={...})
# ─────────────────────────────────────────────────────────────────────────────

_p = load_params(overrides={
    # ── Override any parameter here to deviate from the defaults ──
    # "PROTOCOL":    "AAVE",
    # "LOAN_TOKEN":  "USDT",
    # "N_MC":        5000,
})

PROTOCOL        = _p["PROTOCOL"]        # SYRUP | MORPHO | AAVE | SPARKLEND | GALAXY | ANCHORAGE
NETWORK         = _p["NETWORK"]         # ETHEREUM | ARBITRUM | OPTIMISM (USELESS VARIABLE IN THIS SCRIPT)
MORPHO_MARKET   = _p["MORPHO_MARKET"]   # CBBTC | WETH  (MORPHO only)
GALAXY_TYPE     = _p["GALAXY_TYPE"]     # WITH CLASS A | NO CLASS A  (GALAXY only)
LOAN_TOKEN      = _p["LOAN_TOKEN"]      # USDC | USDT | USDS | DAI | WETH | WEETH | ALL
TIME_INTERVAL   = _p["TIME_INTERVAL"]   # 1d | 1h
FORECAST_STEP   = _p["FORECAST_STEP"]   # forecast horizon in days
TRAIN_SIZE      = _p["TRAIN_SIZE"]      # GARCH rolling training window in days
N_MC            = _p["N_MC"]            # Monte Carlo scenarios
LIQ_ANALYSIS    = _p["LIQ_ANALYSIS"]   # YES | NO

HOURLY_CONV     = _p["HOURLY_CONV"]       # Brownian-bridge hourly decomposition
USE_LOG_RETURNS = _p["USE_LOG_RETURNS"]   # log returns vs simple returns
JUMPS           = _p["JUMPS"]             # compound Poisson jump component
FOCUS_ON_NEGATIVE = _p["FOCUS_ON_NEGATIVE"]  # downside-only jumps
COPULA_TYPE     = _p["COPULA_TYPE"]       # T-COPULA | GAUSSIAN
PERC            = _p["PERC"]             # confidence level for VaR/ES diagnostics
SEED            = _p["SEED"]             # global random seed
WORST_CASE      = _p["WORST_CASE"]       # worst-case LTVs

results = {}

GAS_FEE_USD   = _p["GAS_FEE_USD"]    # liquidation gas cost (USD)
SWAP_FEE_USD  = _p["SWAP_FEE_USD"]   # DEX swap fee (decimal, e.g. 0.005 = 0.5 %)
VOL_FLOOR_PCT = _p["VOL_FLOOR_PCT"]  # GARCH vol floor percentile

MC_TRIGGER     = _p["MC_TRIGGER"]      # margin-call LTV trigger (SYRUP / ANCHORAGE only)
MC_TARGET_LTV  = _p["MC_TARGET_LTV"]  # restore-to LTV on cure; None = initial LTV
MC_CURE_PROB   = _p["MC_CURE_PROB"]   # probability borrower posts collateral when margin-called

print("\nLoading Market and Users Data...\n")
users_df, market_df = importer.load_protocol_data(
    protocol = PROTOCOL,
    network = NETWORK,
    morpho_market = MORPHO_MARKET,
    loan_token = LOAN_TOKEN,
    galaxy_type = GALAXY_TYPE
)
if WORST_CASE:
    users_df = importer.change_user_ltvs(users_df, market_df)

collateral_list = market_df['token_symbol'].unique()

# 1 DATA LOADING
prices_df = importer.load_price_data(collateral_list)

for collateral in collateral_list:
    
    print(f"\nProcessing Token: {collateral.upper()}\n")

    TICKER = collateral.upper()
    scenario = int((1 - PERC) * N_MC)

    print("\nImporting Prices DataFrame...\n")

    prices = prices_df[collateral].dropna()
    prices.name = collateral.upper()

    last_close = float(prices.iloc[-1])
    print(f"\nLast Close Price of {collateral.upper()}: {np.round(last_close, 4)}\n")

    # 2. CALIBRATE MEAN / VOLATILITY MODELS

    # Calculate the best mean+vol model
    calibrator = Calibrator(
        price_series=prices,
        seed=SEED
    )
    best_arima_fitted, best_garch_fitted, arima_spec, garch_spec = calibrator.total_fitter(
        use_log_returns=USE_LOG_RETURNS,
        use_arma_model=False,
        use_vol_model=True,
        train_size=TRAIN_SIZE,
        forecast_step=FORECAST_STEP
    )
    if best_garch_fitted is None:
        best_arima_fitted, best_garch_fitted, arima_spec, garch_spec = calibrator.total_fitter(
            use_log_returns=USE_LOG_RETURNS,
            use_arma_model=True,
            use_vol_model=True,
            train_size=TRAIN_SIZE,
            forecast_step=FORECAST_STEP
        )

    print(f"\nBest ARIMA order for {collateral.upper()}: {arima_spec}\n")
    print(f"\nBest GARCH order for {collateral.upper()}: {garch_spec}\n")

    # Add jumps if requested
    if JUMPS:
        if HOURLY_CONV:
            prices_df_hourly, _ = importer.load_data_yahoo(
                ticker = TICKER,
                period = "max",
                time_interval = "1h"
            )
            prices_jumps = prices_df_hourly["Close"]
            prices_jumps.name = collateral.upper()
        else:
            prices_jumps = prices.copy()
        returns, log_returns = Calibrator.calculate_returns(prices_jumps)
        all_returns = log_returns if USE_LOG_RETURNS else returns
        JUMP_PARAMS = Calibrator.fit_poisson_intensity(
            hist_series=all_returns,
            lower_q=0.025,
            upper_q=0.975,
            focus_on_negative=FOCUS_ON_NEGATIVE,   # always bilateral calibration
        )
        JUMP_PARAMS["focus_on_negative"] = FOCUS_ON_NEGATIVE  # controls simulation clipping only
    else:
        JUMP_PARAMS = None

    # 3. SIMULATE PRICES
    # Obtain simulated prices DataFrame (N_MC x FORECAST_STEP)
    simulator = Simulator(
        prices, 
        arima_spec,
        garch_spec,
        SEED
    )
    arima_model, garch_model, residuals = simulator.arma_garch_refitter(
        TRAIN_SIZE,
        USE_LOG_RETURNS
    )

    results[collateral.upper()] = {
        'token': collateral.upper(),
        'prices': prices,
        'arima_model': arima_model,
        'garch_model': garch_model,
        'residuals': residuals
    }

all_simulated_prices = Simulator.simulate_prices(
    result_per_token = results,
    copula_type = COPULA_TYPE,
    forecasted_step = FORECAST_STEP,
    use_log_returns = USE_LOG_RETURNS,
    use_brownian_bridge = HOURLY_CONV,
    jump_parameters = JUMP_PARAMS,
    n_sims = N_MC,
    seed = SEED,
    market_df = market_df,
    vol_floor_pct = VOL_FLOOR_PCT
)

# 4. DATAFRAME FORMATTING
for token in all_simulated_prices:

    last_close = float(results[token]['prices'].iloc[-1])

    token_simulated_prices = all_simulated_prices[token]
    
    FREQ = '1h' if HOURLY_CONV else '1D'
    PERIODS = FORECAST_STEP * 24 if HOURLY_CONV else FORECAST_STEP
    token_simulated_prices.columns = pd.date_range(
        start=prices_df.index[-1],
        periods=PERIODS,
        freq=FREQ
    )

    stats_df = token_simulated_prices.describe(percentiles=[PERC])
    print(f"\nSimulated Prices Statistics for {token.upper()}:\n")
    print(stats_df.loc[['mean', 'std', 'min', 'max']])



# 5. IF REQUESTED, PERFORM A LIQUIDATION ANALYSES TO ESTIMATE BAD DEBT EXPOSURE

if LIQ_ANALYSIS.upper() == "YES":

    print("\nRunning Monte-Carlo for Liquidations...")

    DEBT = []
    all_liquidations = {}
    liquidation_summary = pd.DataFrame(index=['VALUES'])
    bad_debt_df = pd.DataFrame(index=range(N_MC))

    init_positions = Liquidator(
        borrowers_df = users_df,
        market_df = market_df
    )

    PROTECTION_USD = _load_protection_usd(PROTOCOL)
    if PROTECTION_USD > 0:
        print(f"\nDefense mechanisms loaded: ${PROTECTION_USD/1e6:.1f}M protection applied to net bad debt.\n")

    init_positions.simulate_liquidations(
        all_prices = all_simulated_prices,
        product = PROTOCOL,
        swap_fee = SWAP_FEE_USD,
        gas_fee_usd = GAS_FEE_USD,
        perc = PERC,
        protection_usd = PROTECTION_USD,
        margin_call_trigger    = MC_TRIGGER,
        margin_call_target_ltv = MC_TARGET_LTV,
        margin_call_cure_prob  = MC_CURE_PROB,
    )