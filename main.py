import pandas as pd
import numpy as np

from calibrator import Calibrator
from forecaster import Simulator
from liquidator import Liquidator
import importer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 0.A - SET FUND AND MODEL PARAMETERS

PROTOCOL = "SYRUP"            # can choose between MORPHO, GALAXY, ANCHORAGE, MAPLE AND SPARKLEND
MORPHO_MARKET = "CBBTC"        # can choose between CBBTC and WETH (only for MORPHO)
LOAN_TOKEN = "USDC"            # can choose between USDC, USDT, USDS and DAI
TIME_INTERVAL = "1d"           # Time interval for price data (1d, 1h, etc.)
FORECAST_STEP = 14             # ° days forecast in order to have something
TRAIN_SIZE = 90                # Train on 90 days (medium term) to capture volatility dynamics without overfitting on too recent data
N_MC = 10000                   # Monte Carlo simulation for the forecasted prices
LIQ_ANALYSIS = "YES"           # Set to NO if you want only to forecast prices
SAVE_RESULTS = "NO"            # Set to YES if you want to save outputs in csv format

HOURLY_CONV = False            # Whether to convert daily volatility to hourly (True) or keep daily (False)
USE_LOG_RETURNS = True         # Whether to use log returns (True) or simple returns (False)
JUMPS = False                  # Whether to include jumps in the price simulations
FOCUS_ON_NEGATIVE = False      # If True, only downside jumps are applied (more conservative)
COPULA_TYPE = "T-COPULA"       # Options: "GAUSSIAN", "T-COPULA"
PERC = 0.975                   # Percentile for liquidation risk metrics (0.95, 0.975, 0.995, ...)
SEED = 0                       # Random seed for reproducibility
WORST_CASE = False             # Whether to run worst-case scenario for LTVs (True) or average-case (False)

results = {}

GAS_FEE_USD = 10.0
SWAP_FEE_USD = 0.005
VOL_FLOOR_PCT = 0.75

users_df, market_df = importer.load_protocol_data(
    PROTOCOL, 
    LOAN_TOKEN, 
    morpho_market=MORPHO_MARKET,
    galaxy_type="no-class-a"
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

    init_positions.simulate_liquidations(
        all_prices = all_simulated_prices,
        product = PROTOCOL,
        swap_fee = SWAP_FEE_USD,
        gas_fee_usd = GAS_FEE_USD,
        perc = PERC
    )