"""GA 최적화 실행 스크립트 (주기적 구동)."""

import json
from pathlib import Path

from src.optimization.ga_engine import GAEngine, SimulationConfig, GAConfig

CONFIG_PATH = "optimize_config.json"


def load_config(path: str = CONFIG_PATH) -> dict:
    """optimize_config.json 로드."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Run: cp optimize_config.example.json optimize_config.json"
        )
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config = load_config()

    sim_config = SimulationConfig.from_dict(config.get("simulation", {}))
    ga_config = GAConfig.from_dict(config.get("ga", {}))
    tickers = config.get("tickers", [])

    if not tickers:
        raise ValueError("tickers 목록이 비어있습니다. optimize_config.json을 확인하세요.")

    engine = GAEngine(sim_config=sim_config, ga_config=ga_config)
    engine.run(tickers=tickers)


if __name__ == "__main__":
    main()
