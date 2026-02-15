"""유전 알고리즘 핵심 엔진 (CPU 병렬화 기반 DCA 전략 최적화 - Apple Silicon 호환)."""

from __future__ import annotations

import gc
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit, prange, float32, int32
from numba.core.errors import NumbaPerformanceWarning
from tabulate import tabulate
from tqdm.auto import tqdm

from src.common.data_manager import DataManager

# 경고 메시지 제어
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


# ==========================================
# 1. Configuration & Timeframe Utilities
# ==========================================

TIMEFRAME_TO_MIN = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240,
    "1d": 1440
}


def timeframe_minutes(tf: str) -> int:
    if tf not in TIMEFRAME_TO_MIN:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return TIMEFRAME_TO_MIN[tf]


def bars_per_day(tf: str) -> float:
    return 1440.0 / float(timeframe_minutes(tf))


def sharpe_interval_bars(tf: str, sharpe_days: int = 14) -> int:
    return int(bars_per_day(tf) * sharpe_days)


def trades_per_month(tf: str, base_tf: str = "1m", base_trades: int = 10) -> float:
    scale = timeframe_minutes(base_tf) / timeframe_minutes(tf)
    return base_trades * math.sqrt(scale)


# ==========================================
# 2. Configuration Dataclasses
# ==========================================

@dataclass
class SimulationConfig:
    """시뮬레이션 상수 설정."""
    timeframe: str = "1m"
    data_years: int = 5
    start_date_str: Optional[str] = None
    
    initial_capital: float = 1000.0
    fee_rate: float = 0.0005
    slip_rate: float = 0.0003
    
    fixed_base_margin: float = 5.0
    fixed_dca_margin: float = 5.0
    
    fixed_alloc_long: float = 0.6
    fixed_lev_long: int = 25
    fixed_lev_short: int = 20
    
    cooldown_hours: int = 6
    sharpe_days: int = 14
    
    # 안전 장치 상수 (OKX-style SL)
    abs_cap_dca: int = 15
    dca_sl_gap: float = 0.005
    liq_buffer: float = 0.98
    max_sl_price_cap: float = 0.95
    min_sl_price: float = 0.005


@dataclass
class GAConfig:
    """유전 알고리즘 설정."""
    min_pop_size: int = 2000
    max_pop_size: int = 4000
    
    elite_ratio: float = 0.05
    
    growth_interval: int = 10
    growth_multiplier: float = 1.10
    
    max_generations: int = 50000
    max_patience_limit: int = 100
    
    tournament_size: int = 3


# --- Parameter Indices (Total: 12) ---
P_L_PRICE_DEVIATION = 0
P_L_TAKE_PROFIT = 1
P_L_MAX_DCA = 2
P_L_DEV_MULT = 3
P_L_VOL_MULT = 4
P_L_SL_RATIO = 5

P_S_PRICE_DEVIATION = 6
P_S_TAKE_PROFIT = 7
P_S_MAX_DCA = 8
P_S_DEV_MULT = 9
P_S_VOL_MULT = 10
P_S_SL_RATIO = 11

GENOME_SIZE = 12

# Parameter bounds
PARAM_BOUNDS_HOST = np.array([
    [0.005, 0.02], [0.01, 0.11], [4, 7], [1.05, 1.6], [1.3, 2.1], [0.005, 0.30],  # Long
    [0.005, 0.02], [0.01, 0.11], [4, 7], [1.05, 1.6], [1.3, 2.1], [0.005, 0.30],  # Short
], dtype=np.float32)

ABSOLUTE_MIN_TRADES = 10.0

# 상수 캡
ABS_CAP_DCA_CONST = 15


# ==========================================
# 3. Simulation Logic (JIT Functions)
# ==========================================

@njit(cache=True, fastmath=True)
def _update_mdd_mark_to_market(
    mark_p,
    l_balance, l_amt, l_cost, l_lev,
    s_balance, s_amt, s_cost, s_lev,
    peak_equity, max_dd,
    initial_capital
):
    l_eq = l_balance
    if l_amt > 0.0:
        l_val = l_amt * mark_p
        l_eq += l_cost + (l_val - (l_cost * l_lev))

    s_eq = s_balance
    if s_amt > 0.0:
        s_val = s_amt * mark_p
        s_eq += s_cost + ((s_cost * s_lev) - s_val)

    combined = l_eq + s_eq

    if combined > peak_equity:
        peak_equity = combined

    dd = (peak_equity - combined) / initial_capital
    if dd > max_dd:
        max_dd = dd

    return peak_equity, max_dd


@njit(cache=True, fastmath=True)
def run_dual_simulation(
    opens, closes, x1s, x2s, params, n_bars, sharpe_interval, cooldown_bars,
    initial_capital, fee_rate, slip_rate, fixed_base_margin, fixed_dca_margin,
    fixed_alloc_long, fixed_lev_long, fixed_lev_short
):
    long_ratio = fixed_alloc_long
    short_ratio = 1.0 - long_ratio

    l_lev = fixed_lev_long
    s_lev = fixed_lev_short

    # 파라미터 로딩
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

    l_balance = initial_capital * long_ratio
    s_balance = initial_capital * short_ratio

    # 초기 진입 자금 확인
    l_active = True
    l_start_fee = (l_base_m * l_lev) * fee_rate
    if (l_base_m + l_start_fee) > l_balance:
        l_active = False
        l_max_dca = 0

    s_active = True
    s_start_fee = (s_base_m * s_lev) * fee_rate
    if (s_base_m + s_start_fee) > s_balance:
        s_active = False
        s_max_dca = 0

    # DCA Trigger Calculation
    l_dca_ratios = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)
    l_dca_vols = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)
    s_dca_ratios = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)
    s_dca_vols = np.zeros(ABS_CAP_DCA_CONST, dtype=np.float32)

    # Long Pre-calculation
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

    # Short Pre-calculation
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

    # 변수 초기화
    l_start_cap = l_balance
    s_start_cap = s_balance

    l_amt, l_cost, l_avg, l_dca_cnt = 0.0, 0.0, 0.0, 0
    l_base_price = 0.0

    s_amt, s_cost, s_avg, s_dca_cnt = 0.0, 0.0, 0.0, 0
    s_base_price = 0.0

    peak_combined_equity = initial_capital
    max_dd = 0.0

    total_trades = 0.0

    sharpe_last_equity = initial_capital
    sharpe_sum_r = 0.0
    sharpe_sum_sq = 0.0
    sharpe_cnt = 0

    start_open = opens[0]

    l_wait_until = -1
    s_wait_until = -1

    # 첫 진입 (Base)
    if l_active:
        trigger_p = start_open
        fill_p = trigger_p * (1.0 + slip_rate)

        notional = l_base_m * l_lev
        fee = notional * fee_rate

        l_balance -= (l_base_m + fee)
        l_amt = notional / fill_p
        l_cost = l_base_m
        l_avg = (l_cost * l_lev) / l_amt
        l_base_price = fill_p

    if s_active:
        trigger_p = start_open
        fill_p = trigger_p * (1.0 - slip_rate)

        notional = s_base_m * s_lev
        fee = notional * fee_rate

        s_balance -= (s_base_m + fee)
        s_amt = notional / fill_p
        s_cost = s_base_m
        s_avg = (s_cost * s_lev) / s_amt
        s_base_price = fill_p

    peak_combined_equity, max_dd = _update_mdd_mark_to_market(
        start_open,
        l_balance, l_amt, l_cost, l_lev,
        s_balance, s_amt, s_cost, s_lev,
        peak_combined_equity, max_dd,
        initial_capital
    )

    prev_close = start_open
    path_points = np.zeros(5, dtype=np.float32)

    # 메인 루프
    for i in range(n_bars):
        curr_open = opens[i]

        # Long 신규 진입
        if l_active and l_amt == 0 and i >= l_wait_until:
            trigger_p = curr_open
            fill_p = trigger_p * (1.0 + slip_rate)
            margin = l_base_m
            notional = margin * l_lev
            fee = notional * fee_rate
            req_cash = margin + fee

            if l_balance >= req_cash:
                l_balance -= req_cash
                l_amt = notional / fill_p
                l_cost = margin
                l_avg = (l_cost * l_lev) / l_amt
                l_dca_cnt = 0
                l_base_price = fill_p
            else:
                l_active = False

        # Short 신규 진입
        if s_active and s_amt == 0 and i >= s_wait_until:
            trigger_p = curr_open
            fill_p = trigger_p * (1.0 - slip_rate)
            margin = s_base_m
            notional = margin * s_lev
            fee = notional * fee_rate
            req_cash = margin + fee

            if s_balance >= req_cash:
                s_balance -= req_cash
                s_amt = notional / fill_p
                s_cost = margin
                s_avg = (s_cost * s_lev) / s_amt
                s_dca_cnt = 0
                s_base_price = fill_p
            else:
                s_active = False

        peak_combined_equity, max_dd = _update_mdd_mark_to_market(
            curr_open,
            l_balance, l_amt, l_cost, l_lev,
            s_balance, s_amt, s_cost, s_lev,
            peak_combined_equity, max_dd,
            initial_capital
        )

        # High/Low Path Logic
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
                start_p,
                l_balance, l_amt, l_cost, l_lev,
                s_balance, s_amt, s_cost, s_lev,
                peak_combined_equity, max_dd,
                initial_capital
            )

            # Long Logic
            if l_active and l_amt > 0:
                while True:
                    l_tp_p = l_avg * (1.0 + l_tp)

                    l_sl_p = -1.0
                    if l_base_price > 0.0:
                        l_sl_p = l_base_price * (1.0 - l_sl_target)

                    best_dist = 1.0e20
                    action = 0
                    best_p = start_p

                    # DCA
                    if is_down and l_dca_cnt < l_max_dca:
                        target = l_base_price * l_dca_ratios[l_dca_cnt]
                        if target >= seg_min and target <= seg_max:
                            dist = start_p - target
                            if dist < best_dist:
                                best_dist = dist
                                best_p = target
                                action = 1

                    # SL
                    if is_down and l_sl_p > 0.0 and l_sl_p >= seg_min and l_sl_p <= seg_max:
                        dist = start_p - l_sl_p
                        if dist < best_dist:
                            best_dist = dist
                            best_p = l_sl_p
                            action = 2

                    # TP
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
                        notional = margin * l_lev
                        fee = notional * fee_rate
                        req_cash = margin + fee

                        if l_balance >= req_cash:
                            l_balance -= req_cash
                            eff_p = best_p * (1.0 + slip_rate)
                            l_amt += notional / eff_p
                            l_cost += margin
                            l_dca_cnt += 1
                            l_avg = (l_cost * l_lev) / l_amt

                            start_p = best_p
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p

                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                start_p,
                                l_balance, l_amt, l_cost, l_lev,
                                s_balance, s_amt, s_cost, s_lev,
                                peak_combined_equity, max_dd,
                                initial_capital
                            )
                        else:
                            start_p = best_p - 0.000001
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p

                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                best_p,
                                l_balance, l_amt, l_cost, l_lev,
                                s_balance, s_amt, s_cost, s_lev,
                                peak_combined_equity, max_dd,
                                initial_capital
                            )

                    else:  # SL or TP
                        exit_fill = best_p * (1.0 - slip_rate)
                        val = l_amt * exit_fill
                        pnl = val - (l_cost * l_lev)
                        fee = val * fee_rate

                        ret = l_cost + pnl - fee
                        if ret < 0:
                            ret = 0

                        l_balance += ret
                        l_start_cap = l_balance
                        total_trades += 1

                        l_amt = 0.0
                        l_cost = 0.0
                        l_base_price = 0.0

                        peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                            best_p,
                            l_balance, l_amt, l_cost, l_lev,
                            s_balance, s_amt, s_cost, s_lev,
                            peak_combined_equity, max_dd,
                            initial_capital
                        )

                        if action == 2:  # Stop Loss
                            l_wait_until = i + cooldown_bars
                            break

                        elif action == 4:  # Take Profit
                            margin = l_base_m
                            notional = margin * l_lev
                            fee = notional * fee_rate
                            req_cash = margin + fee

                            if l_balance >= req_cash:
                                l_balance -= req_cash
                                eff_p = best_p * (1.0 + slip_rate)

                                l_amt = notional / eff_p
                                l_cost = margin
                                l_avg = (l_cost * l_lev) / l_amt
                                l_dca_cnt = 0
                                l_base_price = eff_p

                                start_p = best_p
                                if is_down:
                                    seg_max = start_p
                                else:
                                    seg_min = start_p

                                peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                    best_p,
                                    l_balance, l_amt, l_cost, l_lev,
                                    s_balance, s_amt, s_cost, s_lev,
                                    peak_combined_equity, max_dd,
                                    initial_capital
                                )
                            else:
                                l_active = False
                                break

            # Short Logic
            if s_active and s_amt > 0:
                while True:
                    s_tp_p = s_avg * (1.0 - s_tp)

                    s_sl_p = -1.0
                    if s_base_price > 0.0:
                        s_sl_p = s_base_price * (1.0 + s_sl_target)

                    best_dist = 1.0e20
                    action = 0
                    best_p = start_p

                    # DCA
                    if (not is_down) and s_dca_cnt < s_max_dca:
                        target = s_base_price * s_dca_ratios[s_dca_cnt]
                        if target >= seg_min and target <= seg_max:
                            dist = target - start_p
                            if dist < best_dist:
                                best_dist = dist
                                best_p = target
                                action = 1

                    # SL
                    if (not is_down) and s_sl_p > 0.0 and s_sl_p >= seg_min and s_sl_p <= seg_max:
                        dist = s_sl_p - start_p
                        if dist < best_dist:
                            best_dist = dist
                            best_p = s_sl_p
                            action = 2

                    # TP
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
                        notional = margin * s_lev
                        fee = notional * fee_rate
                        req_cash = margin + fee

                        if s_balance >= req_cash:
                            s_balance -= req_cash
                            eff_p = best_p * (1.0 - slip_rate)
                            s_amt += notional / eff_p
                            s_cost += margin
                            s_dca_cnt += 1
                            s_avg = (s_cost * s_lev) / s_amt

                            start_p = best_p
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p

                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                start_p,
                                l_balance, l_amt, l_cost, l_lev,
                                s_balance, s_amt, s_cost, s_lev,
                                peak_combined_equity, max_dd,
                                initial_capital
                            )
                        else:
                            start_p = best_p + 0.000001
                            if is_down:
                                seg_max = start_p
                            else:
                                seg_min = start_p

                            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                best_p,
                                l_balance, l_amt, l_cost, l_lev,
                                s_balance, s_amt, s_cost, s_lev,
                                peak_combined_equity, max_dd,
                                initial_capital
                            )

                    else:  # SL or TP
                        exit_fill = best_p * (1.0 + slip_rate)
                        val = s_amt * exit_fill
                        pnl = (s_cost * s_lev) - val

                        fee = val * fee_rate
                        ret = s_cost + pnl - fee
                        if ret < 0:
                            ret = 0

                        s_balance += ret
                        s_start_cap = s_balance
                        total_trades += 1

                        s_amt = 0.0
                        s_cost = 0.0
                        s_base_price = 0.0

                        peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                            best_p,
                            l_balance, l_amt, l_cost, l_lev,
                            s_balance, s_amt, s_cost, s_lev,
                            peak_combined_equity, max_dd,
                            initial_capital
                        )

                        if action == 2:  # Stop Loss
                            s_wait_until = i + cooldown_bars
                            break

                        elif action == 4:  # Take Profit
                            margin = s_base_m
                            notional = margin * s_lev
                            fee = notional * fee_rate
                            req_cash = margin + fee

                            if s_balance >= req_cash:
                                s_balance -= req_cash
                                eff_p = best_p * (1.0 - slip_rate)

                                s_amt = notional / eff_p
                                s_cost = margin
                                s_avg = (s_cost * s_lev) / s_amt
                                s_dca_cnt = 0
                                s_base_price = eff_p

                                start_p = best_p
                                if is_down:
                                    seg_max = start_p
                                else:
                                    seg_min = start_p

                                peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                                    best_p,
                                    l_balance, l_amt, l_cost, l_lev,
                                    s_balance, s_amt, s_cost, s_lev,
                                    peak_combined_equity, max_dd,
                                    initial_capital
                                )
                            else:
                                s_active = False
                                break

            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                end_p,
                l_balance, l_amt, l_cost, l_lev,
                s_balance, s_amt, s_cost, s_lev,
                peak_combined_equity, max_dd,
                initial_capital
            )

        # Equity Update
        l_eq = l_balance
        if l_amt > 0:
            l_val = l_amt * closes[i]
            l_eq += l_cost + (l_val - (l_cost * l_lev))

        s_eq = s_balance
        if s_amt > 0:
            s_val = s_amt * closes[i]
            s_eq += s_cost + ((s_cost * s_lev) - s_val)

        combined = l_eq + s_eq

        # Sharpe Sampling
        if sharpe_interval > 0 and ((i + 1) % sharpe_interval == 0):
            if sharpe_last_equity > 0:
                profit_delta = combined - sharpe_last_equity
                step_ret = profit_delta / initial_capital
                sharpe_sum_r += step_ret
                sharpe_sum_sq += (step_ret * step_ret)
                sharpe_cnt += 1
            sharpe_last_equity = combined

    final_equity = l_balance + s_balance
    if l_amt > 0:
        final_equity += l_cost + (l_amt * closes[n_bars - 1] - (l_cost * l_lev))
    if s_amt > 0:
        final_equity += s_cost + ((s_cost * s_lev) - s_amt * closes[n_bars - 1])

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

    return roi, mdd, net_profit, total_trades, final_equity, long_ratio, sharpe


# ==========================================
# 4. Parameter Legalization
# ==========================================

@njit(cache=True, fastmath=True)
def validate_and_fix_side(
    balance_limit,
    lev,
    base_m,
    dca_m,
    vol_mult,
    dev,
    dev_mult,
    target_dca,
    sl_pct_gene,
    is_long,
    fee_rate, min_sl_price, dca_sl_gap, liq_buffer, max_sl_price_cap
):
    valid_dca_count = 0
    valid_deepest_dev_ratio = 0.0

    curr_margin = base_m

    current_balance = balance_limit - (base_m * (1.0 + fee_rate * lev))
    if current_balance < 0:
        return 0.0, float(min_sl_price)

    next_dca_margin = dca_m
    curr_price_ratio = 1.0
    curr_step_dev = dev

    curr_pos_notional = base_m * lev
    curr_pos_amt_approx = curr_pos_notional
    curr_avg_ratio = 1.0

    for i in range(1, int(target_dca) + 1):
        if i > ABS_CAP_DCA_CONST:
            break

        if is_long:
            curr_price_ratio = curr_price_ratio * (1.0 - curr_step_dev)
        else:
            curr_price_ratio = curr_price_ratio * (1.0 + curr_step_dev)

        req_notional = next_dca_margin * lev
        req_fee = req_notional * fee_rate
        cost_req = next_dca_margin + req_fee

        if current_balance < cost_req:
            break

        valid_dca_count = i
        if is_long:
            valid_deepest_dev_ratio = 1.0 - curr_price_ratio
        else:
            valid_deepest_dev_ratio = curr_price_ratio - 1.0

        current_balance -= cost_req

        new_amt = req_notional / curr_price_ratio
        total_amt = curr_pos_amt_approx + new_amt
        curr_avg_ratio = (curr_avg_ratio * curr_pos_amt_approx + curr_price_ratio * new_amt) / total_amt

        curr_pos_amt_approx = total_amt
        curr_margin += next_dca_margin

        curr_step_dev *= dev_mult
        next_dca_margin *= vol_mult

    total_wallet = current_balance + curr_margin
    wallet_per_amt = total_wallet / curr_pos_amt_approx

    if is_long:
        liq_ratio = curr_avg_ratio - wallet_per_amt
        max_move = 1.0 - liq_ratio
    else:
        liq_ratio = curr_avg_ratio + wallet_per_amt
        max_move = liq_ratio - 1.0

    abs_max_sl = max_move * liq_buffer
    if abs_max_sl > max_sl_price_cap:
        abs_max_sl = max_sl_price_cap
    if abs_max_sl < min_sl_price:
        abs_max_sl = min_sl_price

    min_sl = float(min_sl_price)
    if valid_dca_count > 0:
        min_sl = valid_deepest_dev_ratio + dca_sl_gap
        if min_sl < min_sl_price:
            min_sl = min_sl_price

    if min_sl >= abs_max_sl:
        final_sl = abs_max_sl
    else:
        sl = sl_pct_gene
        if sl < min_sl:
            sl = min_sl
        if sl > abs_max_sl:
            sl = abs_max_sl
        final_sl = sl

    if final_sl < min_sl_price:
        final_sl = min_sl_price
    if final_sl > max_sl_price_cap:
        final_sl = max_sl_price_cap

    return float(valid_dca_count), float(final_sl)


@njit(cache=True, fastmath=True)
def legalize_genome(
    genome,
    initial_capital, fixed_alloc_long, fixed_lev_long, fixed_lev_short,
    fixed_base_margin, fixed_dca_margin,
    fee_rate, min_sl_price, dca_sl_gap, liq_buffer, max_sl_price_cap
):
    # Long
    l_alloc = fixed_alloc_long * initial_capital
    l_res_dca, l_res_sl = validate_and_fix_side(
        l_alloc,
        fixed_lev_long,
        fixed_base_margin,
        fixed_dca_margin,
        genome[P_L_VOL_MULT],
        genome[P_L_PRICE_DEVIATION],
        genome[P_L_DEV_MULT],
        genome[P_L_MAX_DCA],
        genome[P_L_SL_RATIO],
        1,
        fee_rate, min_sl_price, dca_sl_gap, liq_buffer, max_sl_price_cap
    )
    genome[P_L_MAX_DCA] = np.float32(l_res_dca)
    genome[P_L_SL_RATIO] = np.float32(l_res_sl)

    # Short
    s_alloc = (1.0 - fixed_alloc_long) * initial_capital
    s_res_dca, s_res_sl = validate_and_fix_side(
        s_alloc,
        fixed_lev_short,
        fixed_base_margin,
        fixed_dca_margin,
        genome[P_S_VOL_MULT],
        genome[P_S_PRICE_DEVIATION],
        genome[P_S_DEV_MULT],
        genome[P_S_MAX_DCA],
        genome[P_S_SL_RATIO],
        0,
        fee_rate, min_sl_price, dca_sl_gap, liq_buffer, max_sl_price_cap
    )
    genome[P_S_MAX_DCA] = np.float32(s_res_dca)
    genome[P_S_SL_RATIO] = np.float32(s_res_sl)


# ==========================================
# 5. Parallel Evaluation & Evolution Functions
# ==========================================

@njit(parallel=True, cache=True, fastmath=True)
def evaluate_population(
    population, results, opens, closes, x1s, x2s,
    days, trades_per_month_value, sharpe_interval, cooldown_bars,
    initial_capital, fee_rate, slip_rate, fixed_base_margin, fixed_dca_margin,
    fixed_alloc_long, fixed_lev_long, fixed_lev_short,
    min_sl_price, dca_sl_gap, liq_buffer, max_sl_price_cap
):
    """병렬로 전체 population 평가."""
    pop_size = population.shape[0]
    n_bars = opens.shape[0]
    
    for idx in prange(pop_size):
        genome = population[idx].copy()
        
        # Legalize genome
        legalize_genome(
            genome,
            initial_capital, fixed_alloc_long, fixed_lev_long, fixed_lev_short,
            fixed_base_margin, fixed_dca_margin,
            fee_rate, min_sl_price, dca_sl_gap, liq_buffer, max_sl_price_cap
        )

        r, mdd, _, num, _, _, sharpe = run_dual_simulation(
            opens, closes, x1s, x2s, genome, n_bars, sharpe_interval, cooldown_bars,
            initial_capital, fee_rate, slip_rate, fixed_base_margin, fixed_dca_margin,
            fixed_alloc_long, fixed_lev_long, fixed_lev_short
        )

        months = days / 30.0
        if months < 0.1:
            months = 0.1

        mpr = 0.0
        if months > 0.0:
            mpr = r / months

        req = months * trades_per_month_value
        if req < ABSOLUTE_MIN_TRADES:
            req = ABSOLUTE_MIN_TRADES

        pen = 1.0
        if num < req:
            pen *= (num / req)

        risk_factor = 1.0 + (mdd / 50.0) ** 2

        sharpe_mult = 1.0
        if sharpe > 0:
            sharpe_mult = 1.0 + (sharpe * 10.0)

        fit = (1 / risk_factor) * pen * sharpe_mult

        results[idx, 0] = fit
        results[idx, 1] = r
        results[idx, 2] = mpr
        results[idx, 3] = mdd
        results[idx, 4] = sharpe


@njit(cache=True, fastmath=True)
def apply_bounds(val, idx, bounds):
    min_v = bounds[idx, 0]
    max_v = bounds[idx, 1]
    if val < min_v:
        return min_v
    if val > max_v:
        return max_v
    return val


@njit(cache=True, fastmath=True)
def mutate_gene(val, idx, bounds, rng):
    r = rng
    if r < 0.5:
        factor = 0.85 + np.random.random() * 0.3
        val *= factor
    else:
        shift = np.random.random() * 0.02 - 0.01
        val += shift

    return apply_bounds(val, idx, bounds)


@njit(parallel=True, cache=True, fastmath=True)
def evolve_population(old_pop, new_pop, fitness, bounds, mut_rate, sorted_indices, elite_count, pop_size):
    """병렬로 진화 연산 수행."""
    
    for tid in prange(pop_size):
        # Elitism
        if tid < elite_count:
            parent_idx = sorted_indices[tid]
            for i in range(GENOME_SIZE):
                new_pop[tid, i] = old_pop[parent_idx, i]
            continue

        # Tournament selection
        best_p1 = -1
        best_fit1 = -1e20

        for _ in range(3):
            rand_idx = int(np.random.random() * pop_size)
            if rand_idx >= pop_size:
                rand_idx = pop_size - 1
            score = fitness[rand_idx, 0]
            if score > best_fit1:
                best_fit1 = score
                best_p1 = rand_idx

        best_p2 = -1
        best_fit2 = -1e20
        for _ in range(3):
            rand_idx = int(np.random.random() * pop_size)
            if rand_idx >= pop_size:
                rand_idx = pop_size - 1
            score = fitness[rand_idx, 0]
            if score > best_fit2:
                best_fit2 = score
                best_p2 = rand_idx

        split_point = 1 + int(np.random.random() * (GENOME_SIZE - 2))

        for i in range(GENOME_SIZE):
            val = 0.0
            if i < split_point:
                val = old_pop[best_p1, i]
            else:
                val = old_pop[best_p2, i]

            if np.random.random() < mut_rate:
                val = mutate_gene(val, i, bounds, np.random.random())

            new_pop[tid, i] = val


@njit(parallel=True, cache=True)
def init_population(pop, bounds):
    """병렬로 초기 population 생성."""
    pop_size = pop.shape[0]
    
    for idx in prange(pop_size):
        for i in range(GENOME_SIZE):
            low = bounds[i, 0]
            high = bounds[i, 1]
            rand_val = np.random.random()
            pop[idx, i] = low + rand_val * (high - low)


# ==========================================
# 6. Result Classes
# ==========================================

@dataclass
class OptimizationResult:
    """최적화 결과를 담는 데이터 클래스."""
    symbol: str
    timeframe: str
    
    # 성능 지표
    mpr: float  # Monthly Percentage Return
    mdd: float  # Maximum Drawdown
    sharpe: float
    fitness: float
    
    # Long 파라미터
    long_price_deviation: float
    long_take_profit: float
    long_max_dca: int
    long_dev_multiplier: float
    long_vol_multiplier: float
    long_stop_loss: float
    
    # Short 파라미터
    short_price_deviation: float
    short_take_profit: float
    short_max_dca: int
    short_dev_multiplier: float
    short_vol_multiplier: float
    short_stop_loss: float
    
    # 고정값
    long_allocation: float
    short_allocation: float
    long_leverage: int
    short_leverage: int
    base_margin_ratio: float  # initial_capital 대비 비율
    dca_margin_ratio: float   # initial_capital 대비 비율
    
    # 메타 정보
    generation: int
    created_at: str  # ISO 8601 format
    data_years: int
    
    def to_dict(self) -> Dict:
        """JSON 저장용 딕셔너리로 변환."""
        return {
            "meta": {
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "created_at": self.created_at,
                "generation": self.generation,
                "data_years": self.data_years,
            },
            "performance": {
                "mpr": round(self.mpr, 4),
                "mdd": round(self.mdd, 4),
                "sharpe": round(self.sharpe, 4),
                "fitness": round(self.fitness, 6),
            },
            "parameters": {
                "long": {
                    "price_deviation": round(self.long_price_deviation, 6),
                    "take_profit": round(self.long_take_profit, 6),
                    "max_dca": self.long_max_dca,
                    "dev_multiplier": round(self.long_dev_multiplier, 4),
                    "vol_multiplier": round(self.long_vol_multiplier, 4),
                    "stop_loss": round(self.long_stop_loss, 6),
                },
                "short": {
                    "price_deviation": round(self.short_price_deviation, 6),
                    "take_profit": round(self.short_take_profit, 6),
                    "max_dca": self.short_max_dca,
                    "dev_multiplier": round(self.short_dev_multiplier, 4),
                    "vol_multiplier": round(self.short_vol_multiplier, 4),
                    "stop_loss": round(self.short_stop_loss, 6),
                },
            },
            "fixed_settings": {
                "long_allocation": self.long_allocation,
                "short_allocation": self.short_allocation,
                "long_leverage": self.long_leverage,
                "short_leverage": self.short_leverage,
                "base_margin_ratio": round(self.base_margin_ratio, 6),
                "dca_margin_ratio": round(self.dca_margin_ratio, 6),
            },
        }


# ==========================================
# 7. Reporter
# ==========================================

class Reporter:
    @staticmethod
    def print_checkpoint(ticker: str, gen: int, genome: np.ndarray, stats: Dict, sim_config: SimulationConfig):
        if genome is None or stats['mpr'] == 0:
            return

        l_ratio = sim_config.fixed_alloc_long * 100.0
        s_ratio = (1.0 - sim_config.fixed_alloc_long) * 100.0
        l_lev = sim_config.fixed_lev_long
        s_lev = sim_config.fixed_lev_short

        l_dca_count = int(genome[P_L_MAX_DCA])
        s_dca_count = int(genome[P_S_MAX_DCA])

        table_data = [
            ["Allocation (Fixed)", f"{l_ratio:.1f}%", f"{s_ratio:.1f}%"],
            ["Leverage (Fixed)", f"{l_lev}x", f"{s_lev}x"],
            ["Price Deviation", f"{genome[P_L_PRICE_DEVIATION] * 100.0:.2f}%", f"{genome[P_S_PRICE_DEVIATION] * 100.0:.2f}%"],
            ["Take Profit (Avg%)", f"{genome[P_L_TAKE_PROFIT] * 100.0:.2f}%", f"{genome[P_S_TAKE_PROFIT] * 100.0:.2f}%"],
            ["Max DCA (Raw)", f"{l_dca_count} times", f"{s_dca_count} times"],
            ["Dev Multiplier", f"x{genome[P_L_DEV_MULT]:.2f}", f"x{genome[P_S_DEV_MULT]:.2f}"],
            ["Vol Multiplier", f"x{genome[P_L_VOL_MULT]:.2f}", f"x{genome[P_S_VOL_MULT]:.2f}"],
            ["Stop Loss (Base Order %)", f"{genome[P_L_SL_RATIO] * 100.0:.2f}%", f"{genome[P_S_SL_RATIO] * 100.0:.2f}%"],
            ["Base Margin", f"${sim_config.fixed_base_margin:.1f}", f"${sim_config.fixed_base_margin:.1f}"],
            ["DCA Margin", f"${sim_config.fixed_dca_margin:.1f}", f"${sim_config.fixed_dca_margin:.1f}"],
        ]
        tqdm.write(f"\n[{ticker}] Gen {gen} Best Result:")
        tqdm.write(tabulate(table_data, headers=["Parameter", "Long Value", "Short Value"], tablefmt="simple"))
        tqdm.write("👉 Metrics (Simple Interest, Bi-Weekly Sharpe):")
        tqdm.write(f"   MPR: {stats['mpr']:.1f}% | MDD: {stats['mdd']:.2f}% | Sharpe: {stats.get('sharpe', 0.0):.3f}")
        tqdm.write("-" * 50)


# ==========================================
# 8. GA Engine Class
# ==========================================

class GAEngine:
    """유전 알고리즘 기반 DCA 전략 최적화 엔진 (CPU 병렬화 버전)."""
    
    def __init__(
        self,
        sim_config: Optional[SimulationConfig] = None,
        ga_config: Optional[GAConfig] = None,
        params_dir: Optional[str] = None,
    ):
        self.sim_config = sim_config or SimulationConfig()
        self.ga_config = ga_config or GAConfig()
        
        # params 디렉토리 설정
        if params_dir:
            self.params_dir = Path(params_dir)
        else:
            # 프로젝트 루트의 params 폴더
            self.params_dir = self._find_project_root() / "params"
        
        self.params_dir.mkdir(parents=True, exist_ok=True)
        
        # CPU 코어 수 확인
        self.num_cores = os.cpu_count() or 4
        print(f"[Info] Using {self.num_cores} CPU cores for parallel processing")
    
    @staticmethod
    def _find_project_root() -> Path:
        """프로젝트 루트 디렉토리 찾기."""
        start = Path(__file__).resolve()
        markers = {".git", "pyproject.toml", "requirements.txt", "setup.cfg", "setup.py"}
        for parent in [start.parent, *start.parents]:
            for marker in markers:
                if (parent / marker).exists():
                    return parent
        return start.parent.parent.parent
    
    def _df_to_arrays(self, df, timeframe: str) -> Dict:
        """DataFrame을 NumPy 배열로 변환."""
        opens = df['Open'].values.astype(np.float32)
        closes = df['Close'].values.astype(np.float32)
        lows = df['Low'].values.astype(np.float32)
        highs = df['High'].values.astype(np.float32)

        is_bull = closes >= opens
        x1 = np.where(is_bull, lows, highs)
        x2 = np.where(is_bull, highs, lows)

        bpd = bars_per_day(timeframe)

        return {
            'Open': opens,
            'Close': closes,
            'X1': x1.astype(np.float32),
            'X2': x2.astype(np.float32),
            'Days': float(len(df) / bpd)
        }
    
    def _genome_to_result(
        self,
        ticker: str,
        genome: np.ndarray,
        stats: Dict,
        generation: int
    ) -> OptimizationResult:
        """Genome을 OptimizationResult로 변환."""
        return OptimizationResult(
            symbol=ticker,
            timeframe=self.sim_config.timeframe,
            mpr=float(stats['mpr']),
            mdd=float(stats['mdd']),
            sharpe=float(stats.get('sharpe', 0.0)),
            fitness=float(stats['fitness']),
            long_price_deviation=float(genome[P_L_PRICE_DEVIATION]),
            long_take_profit=float(genome[P_L_TAKE_PROFIT]),
            long_max_dca=int(genome[P_L_MAX_DCA]),
            long_dev_multiplier=float(genome[P_L_DEV_MULT]),
            long_vol_multiplier=float(genome[P_L_VOL_MULT]),
            long_stop_loss=float(genome[P_L_SL_RATIO]),
            short_price_deviation=float(genome[P_S_PRICE_DEVIATION]),
            short_take_profit=float(genome[P_S_TAKE_PROFIT]),
            short_max_dca=int(genome[P_S_MAX_DCA]),
            short_dev_multiplier=float(genome[P_S_DEV_MULT]),
            short_vol_multiplier=float(genome[P_S_VOL_MULT]),
            short_stop_loss=float(genome[P_S_SL_RATIO]),
            long_allocation=self.sim_config.fixed_alloc_long,
            short_allocation=1.0 - self.sim_config.fixed_alloc_long,
            long_leverage=self.sim_config.fixed_lev_long,
            short_leverage=self.sim_config.fixed_lev_short,
            base_margin_ratio=self.sim_config.fixed_base_margin / self.sim_config.initial_capital,
            dca_margin_ratio=self.sim_config.fixed_dca_margin / self.sim_config.initial_capital,
            generation=generation,
            created_at=datetime.utcnow().isoformat() + "Z",
            data_years=self.sim_config.data_years,
        )
    
    def _save_result(self, result: OptimizationResult) -> str:
        """결과를 JSON 파일로 저장."""
        # 파일명: BTC_USDT.json (슬래시를 언더스코어로)
        filename = result.symbol.replace("/", "_") + ".json"
        filepath = self.params_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def _legalize_genome_host(self, genome: np.ndarray) -> np.ndarray:
        """호스트에서 genome 합법화."""
        cfg = self.sim_config
        result = genome.copy()
        legalize_genome(
            result,
            cfg.initial_capital, cfg.fixed_alloc_long, cfg.fixed_lev_long, cfg.fixed_lev_short,
            cfg.fixed_base_margin, cfg.fixed_dca_margin,
            cfg.fee_rate, cfg.min_sl_price, cfg.dca_sl_gap, cfg.liq_buffer, cfg.max_sl_price_cap
        )
        return result
    
    def optimize_ticker(self, ticker: str) -> Optional[OptimizationResult]:
        """단일 티커에 대한 최적화 수행."""
        cfg = self.sim_config
        ga = self.ga_config
        
        tpm = np.float32(trades_per_month(cfg.timeframe))
        s_interval = np.int32(sharpe_interval_bars(cfg.timeframe, cfg.sharpe_days))
        bars_cooldown = np.int32((cfg.cooldown_hours * 60) / timeframe_minutes(cfg.timeframe))
        
        print(f"\n🚀 Processing {ticker} (Timeframe={cfg.timeframe}, CPU Parallel Mode)...")
        
        df = DataManager.fetch_data(
            ticker,
            timeframe=cfg.timeframe,
            years=cfg.data_years,
            start_date_str=cfg.start_date_str
        )
        
        if df is None or len(df) < 1000:
            print(f"⚠️ {ticker}: 데이터 부족, 스킵")
            return None
        
        data = self._df_to_arrays(df, cfg.timeframe)
        bounds = PARAM_BOUNDS_HOST.copy()
        
        # 랜덤 시드 설정
        np.random.seed(int(time.time()) % (2**31))
        
        curr_pop_size = ga.min_pop_size
        
        # Population 초기화
        pop_curr = np.zeros((ga.max_pop_size, GENOME_SIZE), dtype=np.float32)
        pop_next = np.zeros((ga.max_pop_size, GENOME_SIZE), dtype=np.float32)
        results = np.zeros((ga.max_pop_size, 5), dtype=np.float32)
        
        # 초기 population 생성
        init_population(pop_curr[:curr_pop_size], bounds)
        
        best_stats = {'fitness': -1e9, 'mpr': 0, 'mdd': 0, 'sharpe': 0}
        best_genome_host = np.zeros(GENOME_SIZE, dtype=np.float32)
        patience = 0
        final_gen = 0
        
        pbar = tqdm(
            range(1, ga.max_generations + 1),
            desc=f"Pop: {curr_pop_size} | MPR: 0.0% | MDD: 0.0% | Sharpe: 0.00 | Pat: 0",
            bar_format="{desc} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for gen in pbar:
            final_gen = gen
            
            # Population 평가 (병렬)
            evaluate_population(
                pop_curr[:curr_pop_size], results[:curr_pop_size],
                data['Open'], data['Close'], data['X1'], data['X2'],
                np.float32(data['Days']),
                tpm,
                s_interval,
                bars_cooldown,
                np.float32(cfg.initial_capital),
                np.float32(cfg.fee_rate),
                np.float32(cfg.slip_rate),
                np.float32(cfg.fixed_base_margin),
                np.float32(cfg.fixed_dca_margin),
                np.float32(cfg.fixed_alloc_long),
                np.int32(cfg.fixed_lev_long),
                np.int32(cfg.fixed_lev_short),
                np.float32(cfg.min_sl_price),
                np.float32(cfg.dca_sl_gap),
                np.float32(cfg.liq_buffer),
                np.float32(cfg.max_sl_price_cap)
            )
            
            # 현재 세대의 평가 크기 저장 (진화 연산에 사용)
            eval_pop_size = curr_pop_size
            
            fitness_scores = results[:eval_pop_size, 0]
            sorted_indices = np.argsort(fitness_scores)[::-1].astype(np.int32)
            
            best_idx = sorted_indices[0]
            curr_best_fit = fitness_scores[best_idx]
            
            if curr_best_fit > best_stats['fitness']:
                best_stats['fitness'] = curr_best_fit
                best_stats['mpr'] = results[best_idx, 2]
                best_stats['mdd'] = results[best_idx, 3]
                best_stats['sharpe'] = results[best_idx, 4]
                
                best_genome_host = pop_curr[best_idx].copy()
                patience = 0
                
                if curr_pop_size > ga.min_pop_size:
                    pop_curr[0] = pop_curr[best_idx]
                    curr_pop_size = ga.min_pop_size
                    # eval_pop_size는 변경하지 않음 - sorted_indices가 이미 원래 크기로 계산됨
            else:
                patience += 1
            
            # 체크포인트
            if patience == ga.max_patience_limit // 2:
                converted_genome = self._legalize_genome_host(best_genome_host)
                Reporter.print_checkpoint(ticker, gen, converted_genome, best_stats, cfg)
                
                pop_curr[0] = pop_curr[best_idx]
                init_population(pop_curr[1:curr_pop_size], bounds)
            
            if patience >= ga.max_patience_limit:
                break
            
            progress = patience / ga.max_patience_limit
            mut_rate = (1.0 / GENOME_SIZE) + (0.8 * (progress ** 4))
            
            elite_count = int(eval_pop_size * ga.elite_ratio)
            if elite_count < 1:
                elite_count = 1
            
            # 진화 (병렬)
            full_sorted_indices = np.zeros(ga.max_pop_size, dtype=np.int32)
            full_sorted_indices[:eval_pop_size] = sorted_indices
            
            evolve_population(
                pop_curr[:eval_pop_size],
                pop_next[:eval_pop_size],
                results[:eval_pop_size],
                bounds,
                np.float32(mut_rate),
                full_sorted_indices,
                np.int32(elite_count),
                np.int32(eval_pop_size)
            )
            
            # Population 성장 (다음 세대용)
            if curr_pop_size < ga.max_pop_size and patience > 0 and patience % ga.growth_interval == 0:
                new_size = min(int(curr_pop_size * ga.growth_multiplier), ga.max_pop_size)
                if new_size > curr_pop_size:
                    init_population(pop_next[curr_pop_size:new_size], bounds)
                    curr_pop_size = new_size
            
            pop_curr, pop_next = pop_next, pop_curr
            
            pbar.set_description(
                f"Pop: {curr_pop_size} | MPR: {best_stats['mpr']:.1f}% | "
                f"MDD: {best_stats['mdd']:.1f}% | Sharpe: {best_stats['sharpe']:.2f} | Pat: {patience}"
            )
            
            if gen % 100 == 0:
                converted_genome = self._legalize_genome_host(best_genome_host)
                Reporter.print_checkpoint(ticker, gen, converted_genome, best_stats, cfg)
        
        # 최종 결과
        tqdm.write(f"\n🏁 Finished: {ticker}")
        final_genome = self._legalize_genome_host(best_genome_host)
        Reporter.print_checkpoint(ticker, final_gen, final_genome, best_stats, cfg)
        
        # 결과 생성
        result = self._genome_to_result(ticker, final_genome, best_stats, final_gen)
        
        gc.collect()
        
        return result
    
    def run(self, tickers: List[str]) -> Dict[str, OptimizationResult]:
        """모든 티커에 대해 최적화 실행 및 결과 저장."""
        if not tickers:
            raise ValueError("tickers 리스트가 비어있습니다.")
        
        cfg = self.sim_config
        tpm = trades_per_month(cfg.timeframe)
        s_interval = sharpe_interval_bars(cfg.timeframe, cfg.sharpe_days)
        bars_cooldown = int((cfg.cooldown_hours * 60) / timeframe_minutes(cfg.timeframe))
        
        print(f"[Config] Timeframe={cfg.timeframe}, Trades/Month={tpm:.3f}, SharpeIntervalBars={s_interval}")
        print(f"[Config] SL Cooldown = {cfg.cooldown_hours} Hours ({bars_cooldown} bars) - OKX Style")
        print(f"[Config] Params will be saved to: {self.params_dir}")
        
        results = {}
        
        for ticker in tickers:
            result = self.optimize_ticker(ticker)
            
            if result:
                filepath = self._save_result(result)
                results[ticker] = result
                print(f"✅ {ticker} 파라미터 저장완료: {filepath}")
        
        return results
