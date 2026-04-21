"""Dry-run optimization test with modified bounds (max_dca=2, price_dev max=0.05)."""
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("config/.env")

from src.optimization.ga_engine import (
    GAEngine, SimulationConfig, GAConfig,
    PARAM_BOUNDS_HOST, P_L_PRICE_DEVIATION, P_L_MAX_DCA,
    P_S_PRICE_DEVIATION, P_S_MAX_DCA,
)

CONFIG_PATH = "config/optimize_config.json"

config = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
sim_config = SimulationConfig.from_dict(config.get("simulation", {}))
ga_config = GAConfig.from_dict(config.get("ga", {}))
tickers = ["ETH/USDT"]

# Override bounds for this test
PARAM_BOUNDS_HOST[P_L_MAX_DCA, 0] = 2.0      # Long max_dca min: 3 → 2
PARAM_BOUNDS_HOST[P_S_MAX_DCA, 0] = 2.0      # Short max_dca min: 3 → 2
PARAM_BOUNDS_HOST[P_L_PRICE_DEVIATION, 1] = 0.05  # Long price_dev max: 0.03 → 0.05
PARAM_BOUNDS_HOST[P_S_PRICE_DEVIATION, 1] = 0.05  # Short price_dev max: 0.03 → 0.05

print("=== DRY RUN TEST ===")
print(f"  Max DCA bounds: [{PARAM_BOUNDS_HOST[P_L_MAX_DCA, 0]}, {PARAM_BOUNDS_HOST[P_L_MAX_DCA, 1]}]")
print(f"  Price Dev bounds: [{PARAM_BOUNDS_HOST[P_L_PRICE_DEVIATION, 0]}, {PARAM_BOUNDS_HOST[P_L_PRICE_DEVIATION, 1]}]")

engine = GAEngine(sim_config=sim_config, ga_config=ga_config)
engine.run(tickers=tickers, dry_run=True)
