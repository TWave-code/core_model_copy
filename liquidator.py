# ============================================================
# Liquidator: A class to analyze borrower positions and liquidation risks

# This class provides tools to compute the bad debt exposure for each scenario
# of the simulated price paths.
# Swap fee --> Fixed 
# Gas fee --> Fixed
# Slippage --> Sphere based or Hyperlend based or Uniswap V3-based
# ============================================================

import pandas as pd
import numpy as np

import importer


class Liquidator:


    def __init__(
        self,
        borrowers_df: pd.DataFrame,
        market_df: pd.DataFrame,
        *,
        seed: int = 0
    ) -> None:
        """Initializes the Liquidator with borrower and market data.
        Compute for every modeled collateral its sell orderbook."""
        
        # New user final DataFrame after cleaning
        users_final = borrowers_df.copy()
        market_final = market_df.copy()
        
        LT = users_final['lltv'].reset_index(drop=True)
        LTV = users_final['ltv'].reset_index(drop=True)
        HF = users_final['health_factor'].reset_index(drop=True)
        LB = users_final['liquidation_incentive'].reset_index(drop=True)
        tokens = market_df['token_symbol'].to_list()
        oracle_prices_dict = market_final.set_index('token_symbol')['oracle_price'].to_dict()

        all_sell_orderbooks = importer.load_orderbook_data(tokens)

        self.HF = HF
        self.LT = LT
        self.LTV = LTV
        self.LB = LB
        self.seed = seed
        self.user_df = users_final
        self.oracle_prices = oracle_prices_dict
        self.sell_orderbooks = all_sell_orderbooks

    
    @staticmethod
    def slippage_calculator(
        ticks_df: pd.DataFrame,
        amount_liq_usd: np.ndarray,
        sim_price: float
    ) -> np.ndarray:
        
        N = amount_liq_usd.shape[0]
        slippage = np.zeros(N)
        add_slippage = np.zeros(N)

        prices = ticks_df['price'].to_numpy(dtype=np.float64)
        liquidity = ticks_df['liquidity'].to_numpy(dtype=np.float64)

        mask = prices <= sim_price
        prices = prices[mask]
        liquidity = liquidity[mask]
        
        cum_liq = np.cumsum(liquidity)
        cum_value = np.cumsum(liquidity * prices)

        if prices.size == 0:
            return np.ones_like(amount_liq_usd)

        available_liq = cum_liq[-1]

        overflow = amount_liq_usd > available_liq
        add_slippage[overflow] = (amount_liq_usd[overflow] - available_liq) / amount_liq_usd[overflow]

        liq_eff = np.minimum(amount_liq_usd, available_liq)

        idx = np.minimum(
            np.searchsorted(cum_liq, liq_eff, side="left"),
            len(cum_liq) - 1
        )
        value_used = cum_value[idx]
        liq_used = cum_liq[idx]
        avg_price = value_used / liq_used

        price_impact = (sim_price - avg_price) / sim_price
        slippage = np.minimum(price_impact + add_slippage, 1.0)

        slippage[amount_liq_usd == 0.0] = 0.0

        return slippage


    @staticmethod
    def slippage_calculator_cum(
        ticks_df: pd.DataFrame,
        amount_liq_usd: np.ndarray,
        sim_price: float,
        already_consumed: float = 0.0
    ) -> np.ndarray:

        N = amount_liq_usd.shape[0]
        slippage = np.zeros(N)
        add_slippage = np.zeros(N)

        prices    = ticks_df['price'].to_numpy(dtype=np.float64)
        liquidity = ticks_df['liquidity'].to_numpy(dtype=np.float64)

        mask      = prices <= sim_price
        prices    = prices[mask]
        liquidity = liquidity[mask]

        if prices.size == 0:
            return np.ones_like(amount_liq_usd)

        cum_liq   = np.cumsum(liquidity)
        cum_value = np.cumsum(liquidity * prices)

        total_available   = cum_liq[-1]
        effective_avail   = max(total_available - already_consumed, 0.0)

        # Overflow: amount exceeds what remains in the book after prior consumption
        overflow           = amount_liq_usd > effective_avail
        add_slippage[overflow] = (
            (amount_liq_usd[overflow] - effective_avail) / amount_liq_usd[overflow]
        )

        liq_eff      = np.minimum(amount_liq_usd, effective_avail)
        total_needed = already_consumed + liq_eff          # absolute position in book

        idx_end  = np.minimum(
            np.searchsorted(cum_liq, total_needed, side="left"),
            len(cum_liq) - 1
        )
        idx_base = np.minimum(
            np.searchsorted(cum_liq, np.full(N, already_consumed), side="left"),
            len(cum_liq) - 1
        )

        value_used = cum_value[idx_end] - cum_value[idx_base]
        liq_used   = cum_liq[idx_end]   - cum_liq[idx_base]

        safe_liq = np.where(liq_used > 0, liq_used, 1e-12)
        avg_price = value_used / safe_liq

        price_impact         = (sim_price - avg_price) / sim_price
        slippage             = np.minimum(price_impact + add_slippage, 0.9999)
        slippage[amount_liq_usd == 0.0] = 0.0

        return slippage


    @staticmethod
    def get_delta(
        cols, 
        default_delta=pd.Timedelta(hours=1)
    ):
        if len(cols) > 1:
            return cols[1] - cols[0]
        
        if hasattr(cols, "freq") and cols.freq is not None:
            return cols.freq
        
        inferred = pd.infer_freq(cols)
        if inferred is not None:
            return pd.Timedelta(inferred)
        
        return default_delta



    def simulate_liquidations(
        self,
        all_prices: dict,
        product: str,
        *,
        swap_fee: float = 0.0,
        gas_fee_usd: float = 0.0,
        perc: float = 0.995
    ) -> dict:
        """
        Profit-aware simulation with 'count bad debt once' semantics.

        Bad debt accounting:
        - On the FIRST step a wallet is unsafe and not executed, record EAD = R_req (once) and mark it defaulted.
        - Later, if a defaulted wallet is executed, treat R as a recovery up to outstanding EAD.
        - Report: EAD_total, recoveries_total, net_bad_debt_total = EAD_total - recoveries_total.

        Liquidation rule (same as before):
        - Must bring LTV to exactly liq_threshold in one shot (R_req).
        - Execute only if feasible and profitable (after swap fee, slippage, and gas).
        - Slippage scales with R_req relative to max position USD at that step, capped at max_slippage.

        Returns:
        result['summary']: scenario-level aggregates with:
            ['bad_debt_ead_total','recoveries_total','net_bad_debt_total',
            'debt_repaid_total','collateral_liquidated_total',
            'final_total_debt','final_total_collateral']
        result['per_step'] (optional): (scenario, step) with
            ['new_ead_step','recoveries_step','net_bad_debt_outstanding_step',
            'debt_repaid_step','collateral_liquidated_step','executions_step']
        """      
        users_final = self.user_df.copy().reset_index(drop=True)
        users_final = users_final.set_index("wallet_address")

        liq_bonus = users_final['liquidation_incentive']
        first_token_prices = next(iter(all_prices.values())) 
        N_SCEN, N_FORECAST = first_token_prices.shape

        all_prices = {k.upper(): v for k, v in all_prices.items()}
        modeled_tokens = set(all_prices.keys())

        for token, prices in all_prices.items():
        
            if prices is None or prices.empty:
                continue

            oracle = self.oracle_prices[token]

            cols = prices.columns
            
            delta = Liquidator.get_delta(prices.columns)

            t0 = cols.min() - delta
            t0_col = pd.DataFrame(
                oracle,
                index=prices.index,
                columns=[t0]
            )

            prices = pd.concat([t0_col, prices], axis=1)
            prices = prices.sort_index(axis=1)

            all_prices[token] = prices

        def _base_token(col: str, suffix: str) -> str:
            return col.replace(suffix, "").upper()

        supply_cols = [
            c for c in users_final.columns
            if c.endswith("_supply") and "total_supply" not in c
        ]
        borrow_cols = [
            c for c in users_final.columns
            if c.endswith("_borrow") and "total_borrow" not in c
        ]
        borrow_usd_cols = [f"{c}_usd" for c in borrow_cols]
        TOT_DEBT = users_final[borrow_usd_cols].fillna(0).sum(axis=1).sum(axis=0)
        # print(f"TOT DEBT: {TOT_DEBT}")

        supply_tokens = {c: _base_token(c, "_supply") for c in supply_cols}   # col -> token
        borrow_tokens = {c: _base_token(c, "_borrow") for c in borrow_cols}

        # Modeled columns (we have price paths for these)
        mod_supply_cols = [c for c in supply_cols if supply_tokens[c] in modeled_tokens]
        mod_borrow_cols = [c for c in borrow_cols if borrow_tokens[c] in modeled_tokens]

        # Unmodeled columns (static in USD — no price path)
        unmod_supply_cols = [c for c in supply_cols if supply_tokens[c] not in modeled_tokens]
        unmod_borrow_cols = [c for c in borrow_cols if borrow_tokens[c] not in modeled_tokens]

        unmod_supply_usd_cols = [f"{c}_usd" for c in unmod_supply_cols]
        unmod_borrow_usd_cols = [f"{c}_usd" for c in unmod_borrow_cols]

        # Static unmodeled totals per wallet  (N_wallets,)
        unmod_supply_vec = users_final[unmod_supply_usd_cols].fillna(0).sum(axis=1).values  # (W,)
        unmod_borrow_vec = users_final[unmod_borrow_usd_cols].fillna(0).sum(axis=1).values 

        N_BORROW = users_final.shape[0]
        T = N_FORECAST + 1
        
        supply_usd_tensor = np.zeros((N_BORROW, N_SCEN, T))
        borrow_usd_tensor = np.zeros((N_BORROW, N_SCEN, T))
        for col in mod_supply_cols:
            
            token = supply_tokens[col].upper()
            qty = users_final[col].fillna(0).values[:, None, None]     # (W,1,1)
            prices = all_prices[token].values[None, :, :]     # (1,S,T)

            supply_usd_tensor += qty * prices

        for col in mod_borrow_cols:
            
            token = borrow_tokens[col].upper()
            qty = users_final[col].fillna(0).values[:, None, None]
            prices = all_prices[token].values[None, :, :]

            borrow_usd_tensor += qty * prices

        supply_usd_tensor += unmod_supply_vec[:, None, None]
        borrow_usd_tensor += unmod_borrow_vec[:, None, None]
        
        one_plus_bonus = np.array(self.LB, dtype=np.float64)
        denom = -1.0 + self.LT.values * one_plus_bonus

        if np.any(denom >= 0):
            raise ValueError("Invalid params: -1 + LT*(1+bonus) must be < 0.")

        debt_repaid_totals   = np.zeros(N_SCEN)
        collat_liq_totals    = np.zeros(N_SCEN)
        ead_totals           = np.zeros(N_SCEN)  # sum of first defaults
        recoveries_totals    = np.zeros(N_SCEN)
        final_debt_totals    = np.zeros(N_SCEN)
        final_collat_totals  = np.zeros(N_SCEN)
        max_delta_ltv        = np.zeros(N_SCEN)
        max_pct_loss         = np.zeros(N_SCEN)
        pct_user_liq         = np.zeros(N_SCEN)
        pct_user_default     = np.zeros(N_SCEN) 

        for s in range(N_SCEN):

            # Default tracking
            defaulted       = np.zeros(N_BORROW, dtype=bool)
            ead_outstanding = np.zeros(N_BORROW, dtype=np.float64)  # per-wallet EAD not yet recovered
            each_user_loss  = np.zeros(N_BORROW, dtype=np.float64)
            ever_liquidated = np.zeros(N_BORROW, dtype=bool)

            consumed_per_token = {token: 0.0 for token in modeled_tokens}

            scen_repaid         = 0.0
            scen_colliq         = 0.0
            scen_ead            = 0.0
            scen_recv           = 0.0
            n_users_defaulting  = 0

            ltv_list = [0.0]
            D_adj  = np.zeros(N_BORROW)
            CV_adj = np.zeros(N_BORROW)
            for t in range(T):
                D = borrow_usd_tensor[:, s, t] + D_adj
                CV = supply_usd_tensor[:, s, t] + CV_adj
                    
                hf = (CV * self.LT.values) / np.maximum(D, 1e-12)
                unsafe = hf < 1.0

                if unsafe.any():
                    ltv_list.append((self.LT[unsafe] - self.LTV[unsafe]).max())

                if "AAVE" in product.upper() or "SPARK" in product.upper():
                    # Close factor is applied to debt (amount the liquidator repays).
                    # Seized collateral = R_req × (1 + bonus), handled via one_plus_bonus downstream.
                    close_factor = np.where(hf < 0.95, 1.0, 0.5)
                    R_req = close_factor * D
                else:
                    R_req = (self.LT * CV - D) / denom

                R_req = np.where(unsafe, np.maximum(R_req, 0.0), 0.0)
                R_req = np.minimum(R_req, D)
                
                R_cap_collat = CV / one_plus_bonus
                R_req = np.minimum(R_req, R_cap_collat)

                feasible = unsafe & (R_req > 0) & (R_req <= R_cap_collat)

                # do a for cicle to search for the best slippage opportunity
                best_profit = np.full(N_BORROW, -np.inf)
                best_price = np.zeros(N_BORROW)
                best_token_arr  = np.full(N_BORROW, "", dtype=object)

                if any(unsafe):
                    for token in modeled_tokens:

                        P = all_prices[token].values[s, t]
                        
                        slippage = Liquidator.slippage_calculator_cum(
                            self.sell_orderbooks[token],
                            R_req,
                            P,
                            already_consumed=consumed_per_token[token]
                        )
                        # slippage = Liquidator.slippage_calculator(
                        #     self.sell_orderbooks[token],
                        #     R_req,
                        #     P
                        # )

                        proceeds = (1.0 - swap_fee - slippage) * one_plus_bonus * R_req
                        profit = proceeds - R_req - gas_fee_usd

                        better = profit > best_profit
                        best_profit[better] = profit[better]
                        best_price[better] = P
                        best_token_arr[better] = token

                    profit = best_profit
                    P = best_price
                else:
                    profit = 0.0

                profitable = profit >= 0.0
                do_exec = feasible & profitable
                for token in modeled_tokens:
                    token_mask = do_exec & (best_token_arr == token)
                    consumed_per_token[token] += float(np.sum(R_req[token_mask]))

                ever_liquidated |= do_exec

                # Execute liquidations
                R = np.where(do_exec, R_req, 0.0)
                seized_collat_usd = (one_plus_bonus * R)
                each_user_loss += liq_bonus * R
               
                D_adj  -= R
                CV_adj -= seized_collat_usd

                step_repaid = float(np.sum(R))
                step_colliq = float(np.sum(seized_collat_usd))

                # ---- EAD once, recoveries later ----
                # New defaults this step (unsafe, not executed, and not already defaulted)
                new_default_mask = unsafe & (~do_exec) & (~defaulted)
                n_users_defaulting += int(new_default_mask.sum())
                # review the default probability integration here later
                new_ead = np.zeros(N_BORROW, dtype=np.float64)
                new_ead[new_default_mask] = R_req[new_default_mask]  # count once
                defaulted[new_default_mask] = True
                ead_outstanding[new_default_mask] += new_ead[new_default_mask]
                step_new_ead = float(np.sum(new_ead))
                scen_ead += step_new_ead
                # print(f"Scen EAD: {scen_ead}")

                # Recoveries from liquidations of already-defaulted wallets
                recov_mask = do_exec & defaulted
                # Recovery is the debt actually repaid, capped by outstanding EAD
                recov_amt = np.minimum(R[recov_mask], ead_outstanding[recov_mask])
                ead_outstanding[recov_mask] -= recov_amt
                step_recov = float(np.sum(recov_amt))
                scen_recv += step_recov

                scen_repaid += step_repaid
                scen_colliq += step_colliq

            den = np.where(D > 0, D, np.nan)
            
            max_pct_loss[s] = float(np.nanmax(each_user_loss / den))
            debt_repaid_totals[s]   = scen_repaid
            collat_liq_totals[s]    = scen_colliq
            ead_totals[s]           = scen_ead
            recoveries_totals[s]    = scen_recv
            final_debt_totals[s]    = float(np.sum(D - R))
            # final_collat_totals[s]  = float(np.sum(q))
            max_delta_ltv[s]        = max(ltv_list)
            pct_user_liq[s]         = int(ever_liquidated.sum()) / N_BORROW
            pct_user_default[s]     = n_users_defaulting / N_BORROW

        net_bad_debt_total = np.maximum(0.0, ead_totals - recoveries_totals)
       
        scen_names = np.arange(N_SCEN)           # or whatever label you want
        summary = pd.DataFrame({
            'scenario': scen_names,
            'bad_debt_ead_total': ead_totals,
            'recoveries_total': recoveries_totals,
            'net_bad_debt_total': net_bad_debt_total,
            'debt_repaid_total': debt_repaid_totals,
            'collateral_liquidated_total': collat_liq_totals,
            'final_total_debt': final_debt_totals,
            'final_total_collateral': final_collat_totals,
            'max_delta_ltv': max_delta_ltv,
            'max_pct_loss': max_pct_loss,
            'prob_of_liq': pct_user_liq,
            'prob_of_default': pct_user_default
        })

        summary_df = summary[['scenario', 'max_delta_ltv', 'net_bad_debt_total', 
                              'max_pct_loss', 'prob_of_liq', 'prob_of_default']].copy().reset_index(drop=True)
        
        bad_debt_var = np.quantile(summary_df['net_bad_debt_total'], perc, method='inverted_cdf')
        bad_debt_es = summary_df['net_bad_debt_total'][summary_df['net_bad_debt_total'] >= bad_debt_var].mean()
        crr = bad_debt_var / TOT_DEBT
        es = bad_debt_es / TOT_DEBT

        bins = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 1.0]
        labels = [
            "<50%", "50–60%", "60–70%", "70–75%",
            "75–80%", "80–85%", ">85%"
        ]

        # Create bucket once
        users_final['ltv_bucket'] = pd.cut(
            users_final['ltv'],
            bins=bins,
            labels=labels,
            right=False
        )

        ltv_bucket_stats = (
            users_final
            .groupby('ltv_bucket', observed=True)
            .agg(
                positions=('ltv', 'size'),
                total_borrow_usd=('total_borrow_usd', 'sum'),
                avg_ltv=('ltv', 'mean')
            )
            .sort_index()
        )

        print("\n--------------------------------\n")
        print("Initial LTV bucket distribution:\n")
        for bucket, row in ltv_bucket_stats.iterrows():
            print(f"{row['positions']:>4} positions in {bucket:<7}\n"
                f"Borrowed: ${row['total_borrow_usd']:,.0f}\n"
                f"Borrow concentration: {row['total_borrow_usd']/TOT_DEBT:.2%}\n"
                f"Avg LTV: {row['avg_ltv']:.2%}\n"
            )
        print("--------------------------------\n")

        print(f"\nCRR as ES at {perc:.2%}: {es:.4%}")
        print(f"CRR as VaR at {perc:.2%}: {crr:.4%}")
        print(f"PL at {perc:.2%}: {np.quantile(summary_df['prob_of_liq'], perc, method='inverted_cdf'):.4%}")
        print(f"PD at {perc:.2%}: {np.quantile(summary_df['prob_of_default'], perc, method='inverted_cdf'):.4%}")
        print(f"Delta LTV at {perc:.2%}: {np.quantile(summary_df['max_delta_ltv'], perc, method='inverted_cdf'):.4%}\n")
        
        return summary_df
