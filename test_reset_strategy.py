"""Compare current DCA strategy vs Reset strategy (close both when both maxed DCA).

Reset strategy: When both long and short hit max DCA count simultaneously,
close both positions at market and re-enter fresh base positions.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
from numba import njit

from src.common.data_manager import DataManager
from src.optimization.ga_engine import (
    ABS_CAP_DCA_CONST,
    P_L_PRICE_DEVIATION, P_L_TAKE_PROFIT, P_L_MAX_DCA,
    P_L_DEV_MULT, P_L_VOL_MULT, P_L_SL_RATIO,
    P_S_PRICE_DEVIATION, P_S_TAKE_PROFIT, P_S_MAX_DCA,
    P_S_DEV_MULT, P_S_VOL_MULT, P_S_SL_RATIO,
    _update_mdd_mark_to_market,
    bars_per_day,
)


@njit(cache=True, fastmath=True)
def run_dual_reset(
    opens, closes, x1s, x2s, params, n_bars, sharpe_interval, cooldown_bars,
    initial_capital, fee_rate, slip_rate, fixed_base_margin, fixed_dca_margin,
    fixed_leverage
):
    """Same as run_dual_simulation but with reset logic:
    After each DCA fill, check if both sides are at max DCA.
    If so, close both positions at current price and re-enter base.
    """
    lev = fixed_leverage

    l_dev = params[P_L_PRICE_DEVIATION]
    l_tp = params[P_L_TAKE_PROFIT]
    l_max_dca = int(params[P_L_MAX_DCA] + 0.5)
    l_dev_mult = params[P_L_DEV_MULT]
    l_vol_mult = params[P_L_VOL_MULT]
    l_base_m = fixed_base_margin
    l_dca_m = fixed_dca_margin

    s_dev = params[P_S_PRICE_DEVIATION]
    s_tp = params[P_S_TAKE_PROFIT]
    s_max_dca = int(params[P_S_MAX_DCA])
    s_dev_mult = params[P_S_DEV_MULT]
    s_vol_mult = params[P_S_VOL_MULT]
    s_base_m = fixed_base_margin
    s_dca_m = fixed_dca_margin

    l_sl_target = params[P_L_SL_RATIO]
    s_sl_target = params[P_S_SL_RATIO]

    balance = initial_capital

    l_active = True
    l_start_fee = (l_base_m * lev) * fee_rate
    if (l_base_m + l_start_fee) > balance:
        l_active = False
        l_max_dca = 0

    s_active = True
    s_start_fee = (s_base_m * lev) * fee_rate
    if (l_base_m + l_start_fee + s_base_m + s_start_fee) > balance:
        s_active = False
        s_max_dca = 0

    # DCA Trigger Calculation
    l_dca_ratios = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)
    l_dca_vols = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)
    s_dca_ratios = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)
    s_dca_vols = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)

    curr_ratio = 1.0
    curr_step_dev = l_dev
    curr_vol = 1.0
    for k in range(ABS_CAP_DCA_CONST):
        if k < l_max_dca:
            curr_ratio = curr_ratio * (1.0 - curr_step_dev)
            l_dca_ratios[k] = curr_ratio
            l_dca_vols[k] = l_dca_m * curr_vol
            curr_step_dev *= l_dev_mult
            curr_vol *= l_vol_mult

    curr_ratio = 1.0
    curr_step_dev = s_dev
    curr_vol = 1.0
    for k in range(ABS_CAP_DCA_CONST):
        if k < s_max_dca:
            curr_ratio = curr_ratio * (1.0 + curr_step_dev)
            s_dca_ratios[k] = curr_ratio
            s_dca_vols[k] = s_dca_m * curr_vol
            curr_step_dev *= s_dev_mult
            curr_vol *= s_vol_mult

    l_amt, l_cost, l_avg, l_dca_cnt = 0.0, 0.0, 0.0, 0
    l_base_price = 0.0
    s_amt, s_cost, s_avg, s_dca_cnt = 0.0, 0.0, 0.0, 0
    s_base_price = 0.0

    peak_combined_equity = initial_capital
    max_dd = 0.0
    total_trades = 0.0
    reset_count = 0.0

    sharpe_last_equity = initial_capital
    sharpe_sum_r = 0.0
    sharpe_sum_sq = 0.0
    sharpe_cnt = 0

    start_open = opens[0]

    l_wait_until = -1
    s_wait_until = -1

    # First entry
    if l_active:
        trigger_p = start_open
        fill_p = trigger_p * (1.0 + slip_rate)
        notional = l_base_m * lev
        fee = notional * fee_rate
        balance -= (l_base_m + fee)
        l_amt = notional / fill_p
        l_cost = l_base_m
        l_avg = (l_cost * lev) / l_amt
        l_base_price = fill_p

    if s_active:
        trigger_p = start_open
        fill_p = trigger_p * (1.0 - slip_rate)
        notional = s_base_m * lev
        fee = notional * fee_rate
        balance -= (s_base_m + fee)
        s_amt = notional / fill_p
        s_cost = s_base_m
        s_avg = (s_cost * lev) / s_amt
        s_base_price = fill_p

    peak_combined_equity, max_dd = _update_mdd_mark_to_market(
        start_open, balance, l_amt, l_cost, s_amt, s_cost, lev,
        peak_combined_equity, max_dd, initial_capital
    )

    prev_close = start_open
    path_points = np.zeros(5, dtype=np.float32)

    for i in range(n_bars):
        curr_open = opens[i]

        # New Long entry
        if l_active and l_amt == 0 and i >= l_wait_until:
            trigger_p = curr_open
            fill_p = trigger_p * (1.0 + slip_rate)
            margin = l_base_m
            notional = margin * lev
            fee = notional * fee_rate
            req_cash = margin + fee
            if balance >= req_cash:
                balance -= req_cash
                l_amt = notional / fill_p
                l_cost = margin
                l_avg = (l_cost * lev) / l_amt
                l_dca_cnt = 0
                l_base_price = fill_p
            else:
                l_active = False

        # New Short entry
        if s_active and s_amt == 0 and i >= s_wait_until:
            trigger_p = curr_open
            fill_p = trigger_p * (1.0 - slip_rate)
            margin = s_base_m
            notional = margin * lev
            fee = notional * fee_rate
            req_cash = margin + fee
            if balance >= req_cash:
                balance -= req_cash
                s_amt = notional / fill_p
                s_cost = margin
                s_avg = (s_cost * lev) / s_amt
                s_dca_cnt = 0
                s_base_price = fill_p
            else:
                s_active = False

        peak_combined_equity, max_dd = _update_mdd_mark_to_market(
            curr_open, balance, l_amt, l_cost, s_amt, s_cost, lev,
            peak_combined_equity, max_dd, initial_capital
        )

        path_points[0] = prev_close
        path_points[1] = opens[i]
        path_points[2] = x1s[i]
        path_points[3] = x2s[i]
        path_points[4] = closes[i]
        prev_close = path_points[4]

        for p_idx in range(4):
            start_p = path_points[p_idx]
            end_p = path_points[p_idx + 1]
            seg_min = min(start_p, end_p)
            seg_max = max(start_p, end_p)
            is_down = start_p > end_p

            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                start_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                peak_combined_equity, max_dd, initial_capital
            )

            # ── Long Logic ──
            if l_active and l_amt > 0:
                while True:
                    l_tp_p = l_avg * (1.0 + l_tp)
                    l_sl_p = -1.0
                    if l_base_price > 0.0:
                        l_sl_p = l_base_price * (1.0 - l_sl_target)

                    best_dist = 1.0e20
                    action = 0
                    best_p = start_p

                    if is_down and l_dca_cnt < l_max_dca:
                        target = l_base_price * l_dca_ratios[l_dca_cnt]
                        if target >= seg_min and target <= seg_max:
                            dist = start_p - target
                            if dist < best_dist:
                                best_dist = dist
                                best_p = target
                                action = 1

                    if is_down and l_sl_p > 0.0 and l_sl_p >= seg_min and l_sl_p <= seg_max:
                        dist = start_p - l_sl_p
                        if dist < best_dist:
                            best_dist = dist
                            best_p = l_sl_p
                            action = 2

                    if l_tp_p >= seg_min and l_tp_p <= seg_max:
                        dist = abs(start_p - l_tp_p)
                        if dist < best_dist:
                            best_dist = dist
                            best_p = l_tp_p
                            action = 4

                    if action == 0:
                        break

                    if action == 1:  # DCA
                        margin = l_dca_vols[l_dca_cnt]
                        notional = margin * lev
                        fee = notional * fee_rate
                        req_cash = margin + fee
                        if balance >= req_cash:
                            balance -= req_cash
                            eff_p = best_p * (1.0 + slip_rate)
                            l_amt += notional / eff_p
                            l_cost += margin
                            l_dca_cnt += 1
                            l_avg = (l_cost * lev) / l_amt

                            # ── RESET CHECK ──
                            if l_dca_cnt >= l_max_dca and s_amt > 0 and s_dca_cnt >= s_max_dca:
                                # Close long
                                exit_fill_l = best_p * (1.0 - slip_rate)
                                val_l = l_amt * exit_fill_l
                                pnl_l = val_l - (l_cost * lev)
                                fee_l = val_l * fee_rate
                                ret_l = l_cost + pnl_l - fee_l
                                if ret_l < 0:
                                    ret_l = 0
                                balance += ret_l

                                # Close short
                                exit_fill_s = best_p * (1.0 + slip_rate)
                                val_s = s_amt * exit_fill_s
                                pnl_s = (s_cost * lev) - val_s
                                fee_s = val_s * fee_rate
                                ret_s = s_cost + pnl_s - fee_s
                                if ret_s < 0:
                                    ret_s = 0
                                balance += ret_s

                                total_trades += 2
                                reset_count += 1

                                l_amt = 0.0
                                l_cost = 0.0
                                l_base_price = 0.0
                                s_amt = 0.0
                                s_cost = 0.0
                                s_base_price = 0.0

                                peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                    best_p, balance, 0.0, 0.0, 0.0, 0.0, lev,
                                    peak_combined_equity, max_dd, initial_capital
                                )

                                # Re-enter both
                                re_p = best_p
                                # Long re-entry
                                notional_l = l_base_m * lev
                                fee_l2 = notional_l * fee_rate
                                if balance >= (l_base_m + fee_l2 + s_base_m + (s_base_m * lev * fee_rate)):
                                    balance -= (l_base_m + fee_l2)
                                    l_amt = notional_l / (re_p * (1.0 + slip_rate))
                                    l_cost = l_base_m
                                    l_avg = (l_cost * lev) / l_amt
                                    l_dca_cnt = 0
                                    l_base_price = re_p * (1.0 + slip_rate)

                                    # Short re-entry
                                    notional_s = s_base_m * lev
                                    fee_s2 = notional_s * fee_rate
                                    balance -= (s_base_m + fee_s2)
                                    s_amt = notional_s / (re_p * (1.0 - slip_rate))
                                    s_cost = s_base_m
                                    s_avg = (s_cost * lev) / s_amt
                                    s_dca_cnt = 0
                                    s_base_price = re_p * (1.0 - slip_rate)

                                break  # Exit while loop after reset

                            start_p = best_p
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p
                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                start_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                                peak_combined_equity, max_dd, initial_capital
                            )
                        else:
                            start_p = best_p - 0.000001
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p
                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                best_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                                peak_combined_equity, max_dd, initial_capital
                            )

                    else:  # SL or TP
                        exit_fill = best_p * (1.0 - slip_rate)
                        val = l_amt * exit_fill
                        pnl = val - (l_cost * lev)
                        fee = val * fee_rate
                        ret = l_cost + pnl - fee
                        if ret < 0:
                            ret = 0
                        balance += ret
                        total_trades += 1
                        l_amt = 0.0
                        l_cost = 0.0
                        l_base_price = 0.0

                        peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                            best_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                            peak_combined_equity, max_dd, initial_capital
                        )

                        if action == 2:
                            l_wait_until = i + cooldown_bars
                            break
                        elif action == 4:
                            margin = l_base_m
                            notional = margin * lev
                            fee = notional * fee_rate
                            req_cash = margin + fee
                            if balance >= req_cash:
                                balance -= req_cash
                                eff_p = best_p * (1.0 + slip_rate)
                                l_amt = notional / eff_p
                                l_cost = margin
                                l_avg = (l_cost * lev) / l_amt
                                l_dca_cnt = 0
                                l_base_price = eff_p
                                start_p = best_p
                                if is_down:
                                    seg_max = start_p
                                else:
                                    seg_min = start_p
                                peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                    best_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                                    peak_combined_equity, max_dd, initial_capital
                                )
                            else:
                                l_active = False
                                break

            # ── Short Logic ──
            if s_active and s_amt > 0:
                while True:
                    s_tp_p = s_avg * (1.0 - s_tp)
                    s_sl_p = -1.0
                    if s_base_price > 0.0:
                        s_sl_p = s_base_price * (1.0 + s_sl_target)

                    best_dist = 1.0e20
                    action = 0
                    best_p = start_p

                    if (not is_down) and s_dca_cnt < s_max_dca:
                        target = s_base_price * s_dca_ratios[s_dca_cnt]
                        if target >= seg_min and target <= seg_max:
                            dist = target - start_p
                            if dist < best_dist:
                                best_dist = dist
                                best_p = target
                                action = 1

                    if (not is_down) and s_sl_p > 0.0 and s_sl_p >= seg_min and s_sl_p <= seg_max:
                        dist = s_sl_p - start_p
                        if dist < best_dist:
                            best_dist = dist
                            best_p = s_sl_p
                            action = 2

                    if s_tp_p >= seg_min and s_tp_p <= seg_max:
                        dist = abs(start_p - s_tp_p)
                        if dist < best_dist:
                            best_dist = dist
                            best_p = s_tp_p
                            action = 4

                    if action == 0:
                        break

                    if action == 1:  # DCA
                        margin = s_dca_vols[s_dca_cnt]
                        notional = margin * lev
                        fee = notional * fee_rate
                        req_cash = margin + fee
                        if balance >= req_cash:
                            balance -= req_cash
                            eff_p = best_p * (1.0 - slip_rate)
                            s_amt += notional / eff_p
                            s_cost += margin
                            s_dca_cnt += 1
                            s_avg = (s_cost * lev) / s_amt

                            # ── RESET CHECK ──
                            if s_dca_cnt >= s_max_dca and l_amt > 0 and l_dca_cnt >= l_max_dca:
                                # Close long
                                exit_fill_l = best_p * (1.0 - slip_rate)
                                val_l = l_amt * exit_fill_l
                                pnl_l = val_l - (l_cost * lev)
                                fee_l = val_l * fee_rate
                                ret_l = l_cost + pnl_l - fee_l
                                if ret_l < 0:
                                    ret_l = 0
                                balance += ret_l

                                # Close short
                                exit_fill_s = best_p * (1.0 + slip_rate)
                                val_s = s_amt * exit_fill_s
                                pnl_s = (s_cost * lev) - val_s
                                fee_s = val_s * fee_rate
                                ret_s = s_cost + pnl_s - fee_s
                                if ret_s < 0:
                                    ret_s = 0
                                balance += ret_s

                                total_trades += 2
                                reset_count += 1

                                l_amt = 0.0
                                l_cost = 0.0
                                l_base_price = 0.0
                                s_amt = 0.0
                                s_cost = 0.0
                                s_base_price = 0.0

                                peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                    best_p, balance, 0.0, 0.0, 0.0, 0.0, lev,
                                    peak_combined_equity, max_dd, initial_capital
                                )

                                # Re-enter both
                                re_p = best_p
                                notional_l = l_base_m * lev
                                fee_l2 = notional_l * fee_rate
                                if balance >= (l_base_m + fee_l2 + s_base_m + (s_base_m * lev * fee_rate)):
                                    balance -= (l_base_m + fee_l2)
                                    l_amt = notional_l / (re_p * (1.0 + slip_rate))
                                    l_cost = l_base_m
                                    l_avg = (l_cost * lev) / l_amt
                                    l_dca_cnt = 0
                                    l_base_price = re_p * (1.0 + slip_rate)

                                    notional_s = s_base_m * lev
                                    fee_s2 = notional_s * fee_rate
                                    balance -= (s_base_m + fee_s2)
                                    s_amt = notional_s / (re_p * (1.0 - slip_rate))
                                    s_cost = s_base_m
                                    s_avg = (s_cost * lev) / s_amt
                                    s_dca_cnt = 0
                                    s_base_price = re_p * (1.0 - slip_rate)

                                break

                            start_p = best_p
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p
                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                start_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                                peak_combined_equity, max_dd, initial_capital
                            )
                        else:
                            start_p = best_p + 0.000001
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p
                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                best_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                                peak_combined_equity, max_dd, initial_capital
                            )

                    else:  # SL or TP
                        exit_fill = best_p * (1.0 + slip_rate)
                        val = s_amt * exit_fill
                        pnl = (s_cost * lev) - val
                        fee = val * fee_rate
                        ret = s_cost + pnl - fee
                        if ret < 0:
                            ret = 0
                        balance += ret
                        total_trades += 1
                        s_amt = 0.0
                        s_cost = 0.0
                        s_base_price = 0.0

                        peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                            best_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                            peak_combined_equity, max_dd, initial_capital
                        )

                        if action == 2:
                            s_wait_until = i + cooldown_bars
                            break
                        elif action == 4:
                            margin = s_base_m
                            notional = margin * lev
                            fee = notional * fee_rate
                            req_cash = margin + fee
                            if balance >= req_cash:
                                balance -= req_cash
                                eff_p = best_p * (1.0 - slip_rate)
                                s_amt = notional / eff_p
                                s_cost = margin
                                s_avg = (s_cost * lev) / s_amt
                                s_dca_cnt = 0
                                s_base_price = eff_p
                                start_p = best_p
                                if is_down:
                                    seg_max = start_p
                                else:
                                    seg_min = start_p
                                peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                    best_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                                    peak_combined_equity, max_dd, initial_capital
                                )
                            else:
                                s_active = False
                                break

            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                end_p, balance, l_amt, l_cost, s_amt, s_cost, lev,
                peak_combined_equity, max_dd, initial_capital
            )

        # Sharpe Sampling
        combined = balance
        if l_amt > 0:
            l_val = l_amt * closes[i]
            combined += l_cost + (l_val - (l_cost * lev))
        if s_amt > 0:
            s_val = s_amt * closes[i]
            combined += s_cost + ((s_cost * lev) - s_val)

        if sharpe_interval > 0 and ((i + 1) % sharpe_interval == 0):
            if sharpe_last_equity > 0:
                profit_delta = combined - sharpe_last_equity
                step_ret = profit_delta / sharpe_last_equity if sharpe_last_equity > 0 else 0.0
                sharpe_sum_r += step_ret
                sharpe_sum_sq += (step_ret * step_ret)
                sharpe_cnt += 1
            sharpe_last_equity = combined

    final_equity = balance
    if l_amt > 0:
        final_equity += l_cost + (l_amt * closes[n_bars - 1] - (l_cost * lev))
    if s_amt > 0:
        final_equity += s_cost + ((s_cost * lev) - s_amt * closes[n_bars - 1])

    net_profit = final_equity - initial_capital
    roi = (net_profit / initial_capital) * 100.0
    mdd = max_dd * 100.0

    sharpe = 0.0
    if sharpe_cnt > 1:
        mean_ret = sharpe_sum_r / sharpe_cnt
        variance = (sharpe_sum_sq / sharpe_cnt) - (mean_ret * mean_ret)
        if variance > 0.0000001:
            volatility = math.sqrt(variance)
            sharpe = mean_ret / volatility

    return roi, mdd, net_profit, total_trades, final_equity, sharpe, reset_count


def params_to_genome(params_data: dict) -> np.ndarray:
    """Convert params JSON to genome array."""
    genome = np.zeros(12, dtype=np.float32)
    lp = params_data["parameters"]["long"]
    sp = params_data["parameters"]["short"]
    genome[P_L_PRICE_DEVIATION] = lp["price_deviation"]
    genome[P_L_TAKE_PROFIT] = lp["take_profit"]
    genome[P_L_MAX_DCA] = lp["max_dca"]
    genome[P_L_DEV_MULT] = lp["dev_multiplier"]
    genome[P_L_VOL_MULT] = lp["vol_multiplier"]
    genome[P_L_SL_RATIO] = lp["stop_loss"]
    genome[P_S_PRICE_DEVIATION] = sp["price_deviation"]
    genome[P_S_TAKE_PROFIT] = sp["take_profit"]
    genome[P_S_MAX_DCA] = sp["max_dca"]
    genome[P_S_DEV_MULT] = sp["dev_multiplier"]
    genome[P_S_VOL_MULT] = sp["vol_multiplier"]
    genome[P_S_SL_RATIO] = sp["stop_loss"]
    return genome


def run_comparison(ticker: str, params_path: Path):
    from src.optimization.ga_engine import (
        run_dual_simulation, SimulationConfig, sharpe_interval_bars,
        timeframe_minutes,
    )

    with open(params_path) as f:
        params_data = json.load(f)

    cfg = SimulationConfig(data_years=5.5)
    genome = params_to_genome(params_data)

    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")
    print(f"  Long:  max_dca={int(genome[P_L_MAX_DCA])}, tp={genome[P_L_TAKE_PROFIT]*100:.2f}%, sl={genome[P_L_SL_RATIO]*100:.2f}%")
    print(f"  Short: max_dca={int(genome[P_S_MAX_DCA])}, tp={genome[P_S_TAKE_PROFIT]*100:.2f}%, sl={genome[P_S_SL_RATIO]*100:.2f}%")

    df = DataManager.fetch_data(ticker, timeframe=cfg.timeframe, years=cfg.data_years)
    if df is None or len(df) < 1000:
        print(f"  ⚠️ Insufficient data, skipping")
        return None

    opens = df['Open'].values.astype(np.float32)
    closes = df['Close'].values.astype(np.float32)
    is_bull = closes >= opens
    x1s = np.where(is_bull, df['Low'].values, df['High'].values).astype(np.float32)
    x2s = np.where(is_bull, df['High'].values, df['Low'].values).astype(np.float32)

    n_bars = len(df)
    days = n_bars / bars_per_day(cfg.timeframe)
    months = max(days / 30.0, 0.1)

    s_interval = int(sharpe_interval_bars(cfg.timeframe, cfg.sharpe_days))
    cooldown_bars = int((cfg.cooldown_hours * 60) / timeframe_minutes(cfg.timeframe))

    # Current strategy
    roi1, mdd1, _, trades1, _, sharpe1 = run_dual_simulation(
        opens, closes, x1s, x2s, genome, n_bars, s_interval, cooldown_bars,
        cfg.initial_capital, cfg.fee_rate, cfg.slip_rate,
        cfg.fixed_base_margin, cfg.fixed_dca_margin, cfg.fixed_leverage
    )
    mpr1 = roi1 / months

    # Reset strategy
    roi2, mdd2, _, trades2, _, sharpe2, resets = run_dual_reset(
        opens, closes, x1s, x2s, genome, n_bars, s_interval, cooldown_bars,
        cfg.initial_capital, cfg.fee_rate, cfg.slip_rate,
        cfg.fixed_base_margin, cfg.fixed_dca_margin, cfg.fixed_leverage
    )
    mpr2 = roi2 / months

    print(f"\n  {'Metric':<20} {'Current':>12} {'Reset':>12} {'Diff':>12}")
    print(f"  {'-'*56}")
    print(f"  {'MPR':<20} {mpr1:>11.2f}% {mpr2:>11.2f}% {mpr2-mpr1:>+11.2f}%")
    print(f"  {'ROI':<20} {roi1:>11.1f}% {roi2:>11.1f}% {roi2-roi1:>+11.1f}%")
    print(f"  {'MDD':<20} {mdd1:>11.2f}% {mdd2:>11.2f}% {mdd2-mdd1:>+11.2f}%")
    print(f"  {'Sharpe':<20} {sharpe1:>12.3f} {sharpe2:>12.3f} {sharpe2-sharpe1:>+12.3f}")
    print(f"  {'Trades':<20} {trades1:>12.0f} {trades2:>12.0f} {trades2-trades1:>+12.0f}")
    print(f"  {'Resets':<20} {'N/A':>12} {resets:>12.0f}")

    return {
        "ticker": ticker,
        "current": {"mpr": mpr1, "mdd": mdd1, "sharpe": sharpe1, "roi": roi1},
        "reset": {"mpr": mpr2, "mdd": mdd2, "sharpe": sharpe2, "roi": roi2, "resets": resets},
    }


def main():
    params_dir = Path("data/params")
    tickers_config = json.load(open("config/optimize_config.json"))
    tickers = tickers_config.get("tickers", [])

    print("=" * 60)
    print("  DCA Strategy Comparison: Current vs Reset")
    print("  Reset = close both when both sides hit max DCA")
    print("=" * 60)

    results = []
    for ticker in tickers:
        safe_name = ticker.replace("/", "_")
        params_path = params_dir / f"{safe_name}.json"
        if not params_path.exists():
            print(f"\n  ⚠️ {ticker}: No params file, skipping")
            continue
        r = run_comparison(ticker, params_path)
        if r:
            results.append(r)

    if results:
        print(f"\n\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Ticker':<12} {'Cur MPR':>8} {'Rst MPR':>8} {'Diff':>8} {'Resets':>7}")
        print(f"  {'-'*45}")
        for r in results:
            c = r["current"]
            s = r["reset"]
            diff = s["mpr"] - c["mpr"]
            print(f"  {r['ticker']:<12} {c['mpr']:>7.1f}% {s['mpr']:>7.1f}% {diff:>+7.1f}% {s['resets']:>7.0f}")


if __name__ == "__main__":
    main()
