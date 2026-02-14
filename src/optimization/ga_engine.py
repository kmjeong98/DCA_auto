"""유전 알고리즘 핵심 엔진 (GPU 기반 DCA 전략 최적화)."""

from __future__ import annotations

import gc
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import cuda, float32, int32
from numba.core.errors import NumbaPerformanceWarning
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
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


# ==========================================
# 3. GPU Constants (Module-level for Numba)
# ==========================================

# These will be set before kernel execution
CONST_INITIAL_CAPITAL = 1000.0
CONST_FEE_RATE = 0.0005
CONST_SLIP_RATE = 0.0003
CONST_FIXED_BASE_MARGIN = 5.0
CONST_FIXED_DCA_MARGIN = 5.0
CONST_FIXED_ALLOC_LONG = 0.6
CONST_FIXED_LEV_LONG = 25
CONST_FIXED_LEV_SHORT = 20
CONST_ABS_CAP_DCA = 15
CONST_DCA_SL_GAP = 0.005
CONST_LIQ_BUFFER = 0.98
CONST_MAX_SL_PRICE_CAP = 0.95
CONST_MIN_SL_PRICE = 0.005


# ==========================================
# 4. Simulation Logic (Device Functions)
# ==========================================

@cuda.jit(device=True, inline=True)
def _update_mdd_mark_to_market(
    mark_p,
    l_balance, l_amt, l_cost, l_lev,
    s_balance, s_amt, s_cost, s_lev,
    peak_equity, max_dd
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

    dd = (peak_equity - combined) / CONST_INITIAL_CAPITAL
    if dd > max_dd:
        max_dd = dd

    return peak_equity, max_dd


@cuda.jit(device=True, inline=True)
def run_dual_simulation(opens, closes, x1s, x2s, params, n_bars, sharpe_interval, cooldown_bars):
    long_ratio = CONST_FIXED_ALLOC_LONG
    short_ratio = 1.0 - long_ratio

    l_lev = CONST_FIXED_LEV_LONG
    s_lev = CONST_FIXED_LEV_SHORT

    # 파라미터 로딩
    l_dev = params[P_L_PRICE_DEVIATION]
    l_tp = params[P_L_TAKE_PROFIT]
    l_max_dca = int(params[P_L_MAX_DCA] + 0.5)
    l_dev_mult = params[P_L_DEV_MULT]
    l_vol_mult = params[P_L_VOL_MULT]
    l_base_m = CONST_FIXED_BASE_MARGIN
    l_dca_m = CONST_FIXED_DCA_MARGIN

    s_dev = params[P_S_PRICE_DEVIATION]
    s_tp = params[P_S_TAKE_PROFIT]
    s_max_dca = int(params[P_S_MAX_DCA])
    s_dev_mult = params[P_S_DEV_MULT]
    s_vol_mult = params[P_S_VOL_MULT]
    s_base_m = CONST_FIXED_BASE_MARGIN
    s_dca_m = CONST_FIXED_DCA_MARGIN

    l_sl_target = params[P_L_SL_RATIO]
    s_sl_target = params[P_S_SL_RATIO]

    l_balance = CONST_INITIAL_CAPITAL * long_ratio
    s_balance = CONST_INITIAL_CAPITAL * short_ratio

    # 초기 진입 자금 확인
    l_active = True
    l_start_fee = (l_base_m * l_lev) * CONST_FEE_RATE
    if (l_base_m + l_start_fee) > l_balance:
        l_active = False
        l_max_dca = 0

    s_active = True
    s_start_fee = (s_base_m * s_lev) * CONST_FEE_RATE
    if (s_base_m + s_start_fee) > s_balance:
        s_active = False
        s_max_dca = 0

    # DCA Trigger Calculation
    l_dca_ratios = cuda.local.array(CONST_ABS_CAP_DCA, dtype=float32)
    l_dca_vols = cuda.local.array(CONST_ABS_CAP_DCA, dtype=float32)
    s_dca_ratios = cuda.local.array(CONST_ABS_CAP_DCA, dtype=float32)
    s_dca_vols = cuda.local.array(CONST_ABS_CAP_DCA, dtype=float32)

    # Long Pre-calculation
    curr_ratio = 1.0
    curr_step_dev = l_dev
    curr_vol = 1.0

    for k in range(CONST_ABS_CAP_DCA):
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

    for k in range(CONST_ABS_CAP_DCA):
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

    peak_combined_equity = CONST_INITIAL_CAPITAL
    max_dd = 0.0

    total_trades = 0.0

    sharpe_last_equity = CONST_INITIAL_CAPITAL
    sharpe_sum_r = 0.0
    sharpe_sum_sq = 0.0
    sharpe_cnt = 0

    start_open = opens[0]

    l_wait_until = -1
    s_wait_until = -1

    # 첫 진입 (Base)
    if l_active:
        trigger_p = start_open
        fill_p = trigger_p * (1.0 + CONST_SLIP_RATE)

        notional = l_base_m * l_lev
        fee = notional * CONST_FEE_RATE

        l_balance -= (l_base_m + fee)
        l_amt = notional / fill_p
        l_cost = l_base_m
        l_avg = (l_cost * l_lev) / l_amt
        l_base_price = fill_p

    if s_active:
        trigger_p = start_open
        fill_p = trigger_p * (1.0 - CONST_SLIP_RATE)

        notional = s_base_m * s_lev
        fee = notional * CONST_FEE_RATE

        s_balance -= (s_base_m + fee)
        s_amt = notional / fill_p
        s_cost = s_base_m
        s_avg = (s_cost * s_lev) / s_amt
        s_base_price = fill_p

    peak_combined_equity, max_dd = _update_mdd_mark_to_market(
        start_open,
        l_balance, l_amt, l_cost, l_lev,
        s_balance, s_amt, s_cost, s_lev,
        peak_combined_equity, max_dd
    )

    prev_close = start_open
    path_points = cuda.local.array(5, dtype=float32)

    # 메인 루프
    for i in range(n_bars):
        curr_open = opens[i]

        # Long 신규 진입
        if l_active and l_amt == 0 and i >= l_wait_until:
            trigger_p = curr_open
            fill_p = trigger_p * (1.0 + CONST_SLIP_RATE)
            margin = l_base_m
            notional = margin * l_lev
            fee = notional * CONST_FEE_RATE
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
            fill_p = trigger_p * (1.0 - CONST_SLIP_RATE)
            margin = s_base_m
            notional = margin * s_lev
            fee = notional * CONST_FEE_RATE
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
            peak_combined_equity, max_dd
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
                peak_combined_equity, max_dd
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
                        fee = notional * CONST_FEE_RATE
                        req_cash = margin + fee

                        if l_balance >= req_cash:
                            l_balance -= req_cash
                            eff_p = best_p * (1.0 + CONST_SLIP_RATE)
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
                                peak_combined_equity, max_dd
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
                                peak_combined_equity, max_dd
                            )

                    else:  # SL or TP
                        exit_fill = best_p * (1.0 - CONST_SLIP_RATE)
                        val = l_amt * exit_fill
                        pnl = val - (l_cost * l_lev)
                        fee = val * CONST_FEE_RATE

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
                            peak_combined_equity, max_dd
                        )

                        if action == 2:  # Stop Loss
                            l_wait_until = i + cooldown_bars
                            break

                        elif action == 4:  # Take Profit
                            margin = l_base_m
                            notional = margin * l_lev
                            fee = notional * CONST_FEE_RATE
                            req_cash = margin + fee

                            if l_balance >= req_cash:
                                l_balance -= req_cash
                                eff_p = best_p * (1.0 + CONST_SLIP_RATE)

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
                                    peak_combined_equity, max_dd
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
                        fee = notional * CONST_FEE_RATE
                        req_cash = margin + fee

                        if s_balance >= req_cash:
                            s_balance -= req_cash
                            eff_p = best_p * (1.0 - CONST_SLIP_RATE)
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
                                peak_combined_equity, max_dd
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
                                peak_combined_equity, max_dd
                            )

                    else:  # SL or TP
                        exit_fill = best_p * (1.0 + CONST_SLIP_RATE)
                        val = s_amt * exit_fill
                        pnl = (s_cost * s_lev) - val

                        fee = val * CONST_FEE_RATE
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
                            peak_combined_equity, max_dd
                        )

                        if action == 2:  # Stop Loss
                            s_wait_until = i + cooldown_bars
                            break

                        elif action == 4:  # Take Profit
                            margin = s_base_m
                            notional = margin * s_lev
                            fee = notional * CONST_FEE_RATE
                            req_cash = margin + fee

                            if s_balance >= req_cash:
                                s_balance -= req_cash
                                eff_p = best_p * (1.0 - CONST_SLIP_RATE)

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
                                    peak_combined_equity, max_dd
                                )
                            else:
                                s_active = False
                                break

            peak_combined_equity, max_dd = _update_mdd_mark_to_market(
                end_p,
                l_balance, l_amt, l_cost, l_lev,
                s_balance, s_amt, s_cost, s_lev,
                peak_combined_equity, max_dd
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
                step_ret = profit_delta / CONST_INITIAL_CAPITAL
                sharpe_sum_r += step_ret
                sharpe_sum_sq += (step_ret * step_ret)
                sharpe_cnt += 1
            sharpe_last_equity = combined

    final_equity = l_balance + s_balance
    if l_amt > 0:
        final_equity += l_cost + (l_amt * closes[n_bars - 1] - (l_cost * l_lev))
    if s_amt > 0:
        final_equity += s_cost + ((s_cost * s_lev) - s_amt * closes[n_bars - 1])

    net_profit = final_equity - CONST_INITIAL_CAPITAL
    roi = (net_profit / CONST_INITIAL_CAPITAL) * 100.0
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
# 5. Parameter Legalization
# ==========================================

@cuda.jit(device=True, inline=True)
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
    is_long
):
    valid_dca_count = 0
    valid_deepest_dev_ratio = 0.0

    curr_margin = base_m

    current_balance = balance_limit - (base_m * (1.0 + CONST_FEE_RATE * lev))
    if current_balance < 0:
        return 0.0, float(CONST_MIN_SL_PRICE)

    next_dca_margin = dca_m
    curr_price_ratio = 1.0
    curr_step_dev = dev

    curr_pos_notional = base_m * lev
    curr_pos_amt_approx = curr_pos_notional
    curr_avg_ratio = 1.0

    for i in range(1, int(target_dca) + 1):
        if i > CONST_ABS_CAP_DCA:
            break

        if is_long:
            curr_price_ratio = curr_price_ratio * (1.0 - curr_step_dev)
        else:
            curr_price_ratio = curr_price_ratio * (1.0 + curr_step_dev)

        req_notional = next_dca_margin * lev
        req_fee = req_notional * CONST_FEE_RATE
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

    abs_max_sl = max_move * CONST_LIQ_BUFFER
    if abs_max_sl > CONST_MAX_SL_PRICE_CAP:
        abs_max_sl = CONST_MAX_SL_PRICE_CAP
    if abs_max_sl < CONST_MIN_SL_PRICE:
        abs_max_sl = CONST_MIN_SL_PRICE

    min_sl = float(CONST_MIN_SL_PRICE)
    if valid_dca_count > 0:
        min_sl = valid_deepest_dev_ratio + CONST_DCA_SL_GAP
        if min_sl < CONST_MIN_SL_PRICE:
            min_sl = CONST_MIN_SL_PRICE

    if min_sl >= abs_max_sl:
        final_sl = abs_max_sl
    else:
        sl = sl_pct_gene
        if sl < min_sl:
            sl = min_sl
        if sl > abs_max_sl:
            sl = abs_max_sl
        final_sl = sl

    if final_sl < CONST_MIN_SL_PRICE:
        final_sl = CONST_MIN_SL_PRICE
    if final_sl > CONST_MAX_SL_PRICE_CAP:
        final_sl = CONST_MAX_SL_PRICE_CAP

    return float(valid_dca_count), float(final_sl)


@cuda.jit(device=True, inline=True)
def legalize_genome(genome):
    # Long
    l_alloc = CONST_FIXED_ALLOC_LONG * CONST_INITIAL_CAPITAL
    l_res_dca, l_res_sl = validate_and_fix_side(
        l_alloc,
        CONST_FIXED_LEV_LONG,
        CONST_FIXED_BASE_MARGIN,
        CONST_FIXED_DCA_MARGIN,
        genome[P_L_VOL_MULT],
        genome[P_L_PRICE_DEVIATION],
        genome[P_L_DEV_MULT],
        genome[P_L_MAX_DCA],
        genome[P_L_SL_RATIO],
        1
    )
    genome[P_L_MAX_DCA] = float32(l_res_dca)
    genome[P_L_SL_RATIO] = float32(l_res_sl)

    # Short
    s_alloc = (1.0 - CONST_FIXED_ALLOC_LONG) * CONST_INITIAL_CAPITAL
    s_res_dca, s_res_sl = validate_and_fix_side(
        s_alloc,
        CONST_FIXED_LEV_SHORT,
        CONST_FIXED_BASE_MARGIN,
        CONST_FIXED_DCA_MARGIN,
        genome[P_S_VOL_MULT],
        genome[P_S_PRICE_DEVIATION],
        genome[P_S_DEV_MULT],
        genome[P_S_MAX_DCA],
        genome[P_S_SL_RATIO],
        0
    )
    genome[P_S_MAX_DCA] = float32(s_res_dca)
    genome[P_S_SL_RATIO] = float32(s_res_sl)


# ==========================================
# 6. GPU Kernels
# ==========================================

@cuda.jit(fastmath=True)
def evaluate_kernel(population, results, o, c, x1s, x2s, days, trades_per_month_value, sharpe_interval, cooldown_bars):
    idx = cuda.grid(1)
    if idx < population.shape[0]:
        genome = cuda.local.array(GENOME_SIZE, dtype=float32)
        for i in range(GENOME_SIZE):
            genome[i] = population[idx, i]

        legalize_genome(genome)

        r, mdd, _, num, _, _, sharpe = run_dual_simulation(
            o, c, x1s, x2s, genome, o.shape[0], sharpe_interval, cooldown_bars
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

        risk_factor = 1.0 + math.pow(mdd / 50.0, 2)

        sharpe_mult = 1.0
        if sharpe > 0:
            sharpe_mult = 1.0 + (sharpe * 10.0)

        fit = 0.0
        if mpr > 0.0:
            fit = (1 / risk_factor) * pen * sharpe_mult
        else:
            fit = (mpr * risk_factor) * pen

        results[idx, 0] = fit
        results[idx, 1] = r
        results[idx, 2] = mpr
        results[idx, 3] = mdd
        results[idx, 4] = sharpe


@cuda.jit(device=True)
def get_rand(rng_states, tid):
    return xoroshiro128p_uniform_float32(rng_states, tid)


@cuda.jit(device=True)
def apply_bounds_gpu(val, idx, bounds):
    min_v = bounds[idx, 0]
    max_v = bounds[idx, 1]
    if val < min_v:
        return min_v
    if val > max_v:
        return max_v
    return val


@cuda.jit(device=True)
def mutate_gene_gpu(val, idx, bounds, rng_states, tid):
    r = get_rand(rng_states, tid)
    if r < 0.5:
        factor = 0.85 + get_rand(rng_states, tid) * 0.3
        val *= factor
    else:
        shift = get_rand(rng_states, tid) * 0.02 - 0.01
        val += shift

    return apply_bounds_gpu(val, idx, bounds)


@cuda.jit(fastmath=True)
def evolve_kernel(rng_states, old_pop, new_pop, fitness, bounds, mut_rate, sorted_indices, elite_count, pop_size):
    tid = cuda.grid(1)
    if tid >= pop_size:
        return

    # Elitism
    if tid < elite_count:
        parent_idx = sorted_indices[tid]
        for i in range(GENOME_SIZE):
            new_pop[tid, i] = old_pop[parent_idx, i]
        return

    # Tournament selection
    best_p1 = -1
    best_fit1 = -1e20

    for _ in range(3):
        rand_idx = int(get_rand(rng_states, tid) * pop_size)
        if rand_idx >= pop_size:
            rand_idx = pop_size - 1
        score = fitness[rand_idx, 0]
        if score > best_fit1:
            best_fit1 = score
            best_p1 = rand_idx

    best_p2 = -1
    best_fit2 = -1e20
    for _ in range(3):
        rand_idx = int(get_rand(rng_states, tid) * pop_size)
        if rand_idx >= pop_size:
            rand_idx = pop_size - 1
        score = fitness[rand_idx, 0]
        if score > best_fit2:
            best_fit2 = score
            best_p2 = rand_idx

    split_point = 1 + int(get_rand(rng_states, tid) * (GENOME_SIZE - 2))

    for i in range(GENOME_SIZE):
        val = 0.0
        if i < split_point:
            val = old_pop[best_p1, i]
        else:
            val = old_pop[best_p2, i]

        if get_rand(rng_states, tid) < mut_rate:
            val = mutate_gene_gpu(val, i, bounds, rng_states, tid)

        new_pop[tid, i] = val


@cuda.jit(fastmath=True)
def init_population_kernel(pop, bounds, rng_states):
    idx = cuda.grid(1)
    if idx < pop.shape[0]:
        for i in range(GENOME_SIZE):
            low = bounds[i, 0]
            high = bounds[i, 1]
            rand_val = get_rand(rng_states, idx)
            pop[idx, i] = low + rand_val * (high - low)


@cuda.jit
def legalize_genome_kernel(genome):
    legalize_genome(genome)


# ==========================================
# 7. Result Classes
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
# 8. Reporter
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
# 9. GA Engine Class
# ==========================================

class GAEngine:
    """유전 알고리즘 기반 DCA 전략 최적화 엔진."""
    
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
        
        # GPU 체크
        if not cuda.is_available():
            raise RuntimeError("GPU가 필요합니다. CUDA 지원 GPU를 확인해주세요.")
        
        self._update_global_constants()
    
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
    
    def _update_global_constants(self):
        """GPU 커널에서 사용할 전역 상수 업데이트."""
        global CONST_INITIAL_CAPITAL, CONST_FEE_RATE, CONST_SLIP_RATE
        global CONST_FIXED_BASE_MARGIN, CONST_FIXED_DCA_MARGIN
        global CONST_FIXED_ALLOC_LONG, CONST_FIXED_LEV_LONG, CONST_FIXED_LEV_SHORT
        global CONST_ABS_CAP_DCA, CONST_DCA_SL_GAP, CONST_LIQ_BUFFER
        global CONST_MAX_SL_PRICE_CAP, CONST_MIN_SL_PRICE
        
        CONST_INITIAL_CAPITAL = self.sim_config.initial_capital
        CONST_FEE_RATE = self.sim_config.fee_rate
        CONST_SLIP_RATE = self.sim_config.slip_rate
        CONST_FIXED_BASE_MARGIN = self.sim_config.fixed_base_margin
        CONST_FIXED_DCA_MARGIN = self.sim_config.fixed_dca_margin
        CONST_FIXED_ALLOC_LONG = self.sim_config.fixed_alloc_long
        CONST_FIXED_LEV_LONG = self.sim_config.fixed_lev_long
        CONST_FIXED_LEV_SHORT = self.sim_config.fixed_lev_short
        CONST_ABS_CAP_DCA = self.sim_config.abs_cap_dca
        CONST_DCA_SL_GAP = self.sim_config.dca_sl_gap
        CONST_LIQ_BUFFER = self.sim_config.liq_buffer
        CONST_MAX_SL_PRICE_CAP = self.sim_config.max_sl_price_cap
        CONST_MIN_SL_PRICE = self.sim_config.min_sl_price
    
    def _df_to_gpu(self, df, timeframe: str) -> Dict:
        """DataFrame을 GPU 배열로 변환."""
        opens = df['Open'].values.astype(np.float32)
        closes = df['Close'].values.astype(np.float32)
        lows = df['Low'].values.astype(np.float32)
        highs = df['High'].values.astype(np.float32)

        is_bull = closes >= opens
        x1 = np.where(is_bull, lows, highs)
        x2 = np.where(is_bull, highs, lows)

        bpd = bars_per_day(timeframe)

        return {
            'Open': cuda.to_device(opens),
            'Close': cuda.to_device(closes),
            'X1': cuda.to_device(x1.astype(np.float32)),
            'X2': cuda.to_device(x2.astype(np.float32)),
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
    
    def optimize_ticker(self, ticker: str) -> Optional[OptimizationResult]:
        """단일 티커에 대한 최적화 수행."""
        cfg = self.sim_config
        ga = self.ga_config
        
        tpm = float32(trades_per_month(cfg.timeframe))
        s_interval = int32(sharpe_interval_bars(cfg.timeframe, cfg.sharpe_days))
        bars_cooldown = int32((cfg.cooldown_hours * 60) / timeframe_minutes(cfg.timeframe))
        
        print(f"\n🚀 Processing {ticker} (Timeframe={cfg.timeframe}, GPU Evolution Mode)...")
        
        df = DataManager.fetch_data(
            ticker,
            timeframe=cfg.timeframe,
            years=cfg.data_years,
            start_date_str=cfg.start_date_str
        )
        
        if df is None or len(df) < 1000:
            print(f"⚠️ {ticker}: 데이터 부족, 스킵")
            return None
        
        d_data = self._df_to_gpu(df, cfg.timeframe)
        d_bounds = cuda.to_device(PARAM_BOUNDS_HOST)
        
        curr_pop_size = ga.min_pop_size
        rng_states = create_xoroshiro128p_states(curr_pop_size, seed=int(time.time()))
        
        d_pop_curr = cuda.device_array((ga.max_pop_size, GENOME_SIZE), dtype=np.float32)
        d_pop_next = cuda.device_array((ga.max_pop_size, GENOME_SIZE), dtype=np.float32)
        d_results = cuda.device_array((ga.max_pop_size, 5), dtype=np.float32)
        d_sorted_indices = cuda.device_array(ga.max_pop_size, dtype=np.int32)
        
        block = 32
        grid = (curr_pop_size + block - 1) // block
        init_population_kernel[grid, block](d_pop_curr[:curr_pop_size], d_bounds, rng_states)
        
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
            grid = (curr_pop_size + block - 1) // block
            
            evaluate_kernel[grid, block](
                d_pop_curr[:curr_pop_size], d_results[:curr_pop_size],
                d_data['Open'], d_data['Close'], d_data['X1'], d_data['X2'],
                float32(d_data['Days']),
                tpm,
                s_interval,
                bars_cooldown
            )
            
            h_results = d_results[:curr_pop_size].copy_to_host()
            fitness_scores = h_results[:, 0]
            
            sorted_indices_host = np.argsort(fitness_scores)[::-1].astype(np.int32)
            d_sorted_indices[:curr_pop_size].copy_to_device(sorted_indices_host)
            
            best_idx = sorted_indices_host[0]
            curr_best_fit = fitness_scores[best_idx]
            
            if curr_best_fit > best_stats['fitness']:
                best_stats['fitness'] = curr_best_fit
                best_stats['mpr'] = h_results[best_idx, 2]
                best_stats['mdd'] = h_results[best_idx, 3]
                best_stats['sharpe'] = h_results[best_idx, 4]
                
                best_genome_host = d_pop_curr[best_idx].copy_to_host()
                patience = 0
                
                if curr_pop_size > ga.min_pop_size:
                    d_pop_curr[0] = d_pop_curr[best_idx]
                    curr_pop_size = ga.min_pop_size
                    rng_states = create_xoroshiro128p_states(curr_pop_size, seed=int(time.time()))
            else:
                patience += 1
            
            if curr_pop_size < ga.max_pop_size and patience > 0 and patience % ga.growth_interval == 0:
                new_size = min(int(curr_pop_size * ga.growth_multiplier), ga.max_pop_size)
                if new_size > curr_pop_size:
                    rng_states = create_xoroshiro128p_states(new_size, seed=int(time.time()))
                    added = new_size - curr_pop_size
                    grid_add = (added + block - 1) // block
                    init_population_kernel[grid_add, block](d_pop_curr[curr_pop_size:new_size], d_bounds, rng_states)
                    curr_pop_size = new_size
            
            # 체크포인트
            if patience == ga.max_patience_limit // 2:
                temp_genome = cuda.to_device(best_genome_host)
                legalize_genome_kernel[1, 1](temp_genome)
                converted_genome = temp_genome.copy_to_host()
                Reporter.print_checkpoint(ticker, gen, converted_genome, best_stats, cfg)
                
                d_pop_curr[0] = d_pop_curr[best_idx]
                rng_states = create_xoroshiro128p_states(curr_pop_size, seed=int(time.time()))
                grid_full = (curr_pop_size + block - 1) // block
                init_population_kernel[grid_full, block](d_pop_curr[1:curr_pop_size], d_bounds, rng_states)
            
            if patience >= ga.max_patience_limit:
                break
            
            progress = patience / ga.max_patience_limit
            mut_rate = (1.0 / GENOME_SIZE) + (0.8 * (progress ** 4))
            
            elite_count = int(curr_pop_size * ga.elite_ratio)
            if elite_count < 1:
                elite_count = 1
            
            grid = (curr_pop_size + block - 1) // block
            evolve_kernel[grid, block](
                rng_states,
                d_pop_curr[:curr_pop_size],
                d_pop_next[:curr_pop_size],
                d_results[:curr_pop_size],
                d_bounds,
                float32(mut_rate),
                d_sorted_indices,
                int32(elite_count),
                int32(curr_pop_size)
            )
            
            d_pop_curr, d_pop_next = d_pop_next, d_pop_curr
            
            pbar.set_description(
                f"Pop: {curr_pop_size} | MPR: {best_stats['mpr']:.1f}% | "
                f"MDD: {best_stats['mdd']:.1f}% | Sharpe: {best_stats['sharpe']:.2f} | Pat: {patience}"
            )
            
            if gen % 100 == 0:
                temp_genome = cuda.to_device(best_genome_host)
                legalize_genome_kernel[1, 1](temp_genome)
                converted_genome = temp_genome.copy_to_host()
                Reporter.print_checkpoint(ticker, gen, converted_genome, best_stats, cfg)
        
        # 최종 결과
        tqdm.write(f"\n🏁 Finished: {ticker}")
        temp_genome = cuda.to_device(best_genome_host)
        legalize_genome_kernel[1, 1](temp_genome)
        final_genome = temp_genome.copy_to_host()
        Reporter.print_checkpoint(ticker, final_gen, final_genome, best_stats, cfg)
        
        # 결과 생성
        result = self._genome_to_result(ticker, final_genome, best_stats, final_gen)
        
        # 정리
        del d_pop_curr, d_pop_next, d_results, d_data, rng_states, d_sorted_indices, d_bounds
        gc.collect()
        
        return result
    
    def run(self, tickers: Optional[List[str]] = None) -> Dict[str, OptimizationResult]:
        """모든 티커에 대해 최적화 실행 및 결과 저장."""
        if not tickers:
            raise ValueError("tickers 파라미터는 필수입니다. run(tickers=['BTC/USDT', ...])")
        
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


# ==========================================
# 10. Entry Point
# ==========================================

def main():
    """메인 실행 함수."""
    # 기본 설정으로 엔진 생성
    sim_config = SimulationConfig(
        timeframe="1m",
        data_years=5,
    )
    
    ga_config = GAConfig(
        tickers=("DOGE/USDT", "ZEC/USDT", "SUI/USDT", "ETH/USDT", "BTC/USDT", "SOL/USDT"),
    )
    
    engine = GAEngine(sim_config=sim_config, ga_config=ga_config)
    results = engine.run()
    
    print(f"\n🎉 최적화 완료! 총 {len(results)}개 티커 처리됨.")


if __name__ == "__main__":
    main()
