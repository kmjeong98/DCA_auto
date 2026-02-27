"""GA optimization runner script (periodic execution)."""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("config/.env")

from src.optimization.ga_engine import GAEngine, SimulationConfig, GAConfig

CONFIG_PATH = "config/optimize_config.json"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load optimize_config.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Run: cp config/optimize_config.example.json config/optimize_config.json"
        )
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config = load_config()

    sim_config = SimulationConfig.from_dict(config.get("simulation", {}))
    ga_config = GAConfig.from_dict(config.get("ga", {}))
    tickers = config.get("tickers", [])

    if not tickers:
        raise ValueError("tickers list is empty. Check config/optimize_config.json.")

    engine = GAEngine(sim_config=sim_config, ga_config=ga_config)
    engine.run(tickers=tickers)


if __name__ == "__main__":
    main()
