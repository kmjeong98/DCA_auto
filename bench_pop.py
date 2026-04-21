"""Benchmark population sizes: measure time per generation."""
import time
import numpy as np
from dotenv import load_dotenv
load_dotenv("config/.env")

from src.optimization.ga_engine import (
    GAEngine, SimulationConfig, GAConfig,
    evaluate_population, evolve_population, init_population,
    PARAM_BOUNDS_HOST, GENOME_SIZE,
    trades_per_month, sharpe_interval_bars, timeframe_minutes,
)
from src.common.data_manager import DataManager

TICKER = "ETH/USDT"
POP_SIZES = [500, 1000, 1500, 2000, 3000, 4000, 5000]
WARMUP_GENS = 2
BENCH_GENS = 5

cfg = SimulationConfig()
ga = GAConfig()

print(f"Loading data: {TICKER} ({cfg.data_years}y, {cfg.timeframe})...")
df = DataManager.fetch_data(TICKER, timeframe=cfg.timeframe, years=cfg.data_years)
engine = GAEngine(sim_config=cfg, ga_config=ga)
data = engine._df_to_arrays(df, cfg.timeframe)
print(f"Bars: {data['Open'].shape[0]:,}  Days: {data['Days']:.0f}\n")

tpm = trades_per_month(cfg.timeframe)
s_interval = sharpe_interval_bars(cfg.timeframe, cfg.sharpe_days)
bars_cooldown = int((cfg.cooldown_hours * 60) / timeframe_minutes(cfg.timeframe))
bounds = PARAM_BOUNDS_HOST.copy()

print(f"{'Pop':>5s}  {'Eval(ms)':>9s}  {'Evolve(ms)':>10s}  {'Total(ms)':>10s}  {'Gen/min':>7s}")
print("-" * 55)

for pop_size in POP_SIZES:
    pop_curr = np.zeros((pop_size, GENOME_SIZE), dtype=np.float32)
    pop_next = np.zeros((pop_size, GENOME_SIZE), dtype=np.float32)
    results = np.zeros((pop_size, 5), dtype=np.float32)

    init_population(pop_curr, bounds)

    # Warmup (JIT compile + cache)
    for _ in range(WARMUP_GENS):
        evaluate_population(
            pop_curr, results,
            data['Open'], data['Close'], data['X1'], data['X2'],
            np.float32(data['Days']), tpm, s_interval, bars_cooldown,
            np.float32(cfg.initial_capital), np.float32(cfg.fee_rate),
            np.float32(cfg.slip_rate), np.float32(cfg.fixed_base_margin),
            np.float32(cfg.fixed_dca_margin), np.int32(cfg.fixed_leverage),
            np.float32(cfg.min_sl_price), np.float32(cfg.dca_sl_gap),
            np.float32(cfg.liq_buffer), np.float32(cfg.max_sl_price_cap),
        )
        sorted_idx = np.argsort(results[:, 0])[::-1].astype(np.int32)
        full_sorted = np.zeros(pop_size, dtype=np.int32)
        full_sorted[:] = sorted_idx
        evolve_population(
            pop_curr, pop_next, results, bounds,
            np.float32(0.1), full_sorted, np.int32(max(1, pop_size // 20)), np.int32(pop_size),
        )
        pop_curr, pop_next = pop_next, pop_curr

    # Benchmark
    eval_times = []
    evolve_times = []
    for _ in range(BENCH_GENS):
        t0 = time.perf_counter()
        evaluate_population(
            pop_curr, results,
            data['Open'], data['Close'], data['X1'], data['X2'],
            np.float32(data['Days']), tpm, s_interval, bars_cooldown,
            np.float32(cfg.initial_capital), np.float32(cfg.fee_rate),
            np.float32(cfg.slip_rate), np.float32(cfg.fixed_base_margin),
            np.float32(cfg.fixed_dca_margin), np.int32(cfg.fixed_leverage),
            np.float32(cfg.min_sl_price), np.float32(cfg.dca_sl_gap),
            np.float32(cfg.liq_buffer), np.float32(cfg.max_sl_price_cap),
        )
        t1 = time.perf_counter()
        eval_times.append(t1 - t0)

        sorted_idx = np.argsort(results[:, 0])[::-1].astype(np.int32)
        full_sorted = np.zeros(pop_size, dtype=np.int32)
        full_sorted[:] = sorted_idx

        t2 = time.perf_counter()
        evolve_population(
            pop_curr, pop_next, results, bounds,
            np.float32(0.1), full_sorted, np.int32(max(1, pop_size // 20)), np.int32(pop_size),
        )
        t3 = time.perf_counter()
        evolve_times.append(t3 - t2)
        pop_curr, pop_next = pop_next, pop_curr

    avg_eval = np.mean(eval_times) * 1000
    avg_evolve = np.mean(evolve_times) * 1000
    avg_total = avg_eval + avg_evolve
    gen_per_min = 60000 / avg_total

    print(f"{pop_size:>5d}  {avg_eval:>9.1f}  {avg_evolve:>10.1f}  {avg_total:>10.1f}  {gen_per_min:>7.1f}")
