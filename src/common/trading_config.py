"""Trading config loader (config.json)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.common.logger import setup_logger


@dataclass
class SymbolConfig:
    """Per-symbol configuration."""
    symbol: str      # "BTC/USDT"
    weight: float    # capital allocation ratio


@dataclass
class TradingConfig:
    """Global trading configuration."""
    cooldown_hours: int
    symbols: Dict[str, SymbolConfig]  # key: "BTC_USDT"

    @classmethod
    def load(cls, path: str = "config/config.json") -> "TradingConfig":
        """Load config.json file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        symbols: Dict[str, SymbolConfig] = {}
        for key, val in data.get("symbols", {}).items():
            # BTC_USDT -> BTC/USDT
            symbol = key.replace("_", "/")
            symbols[key] = SymbolConfig(
                symbol=symbol,
                weight=float(val.get("weight", 0)),
            )

        return cls(
            cooldown_hours=int(data.get("cooldown_hours", 6)),
            symbols=symbols,
        )

    def get_symbol_names(self) -> List[str]:
        """Return in ["BTC/USDT", "ETH/USDT", ...] format."""
        return [cfg.symbol for cfg in self.symbols.values()]

    def get_safe_names(self) -> List[str]:
        """Return ["BTC_USDT", "ETH_USDT", ...] key format."""
        return list(self.symbols.keys())

    def get_weight(self, symbol: str) -> float:
        """Get weight for symbol. symbol can be in "BTC/USDT" or "BTC_USDT" format."""
        safe_key = symbol.replace("/", "_")
        cfg = self.symbols.get(safe_key)
        if cfg is None:
            raise KeyError(f"Symbol {symbol} not in config")
        return cfg.weight

    def get_symbol_capital(self, symbol: str, total_balance: float) -> float:
        """Calculate capital allocated to symbol."""
        return total_balance * self.get_weight(symbol)

    def validate(self) -> None:
        """Validate configuration."""
        logger = setup_logger("config", "data/logs/config.log")

        if not self.symbols:
            raise ValueError("No symbols configured")

        # Check that params files exist
        params_dir = Path("data/params")
        missing = []
        for safe_name in self.symbols:
            param_file = params_dir / f"{safe_name}.json"
            if not param_file.exists():
                missing.append(safe_name)

        if missing:
            raise FileNotFoundError(
                f"Missing params files for: {missing}. "
                f"Run main_optimize.py first."
            )

        # Display weight total (no normalization)
        total_weight = sum(cfg.weight for cfg in self.symbols.values())
        logger.info(f"Total weight: {total_weight:.4f} ({len(self.symbols)} symbols)")
        if total_weight > 1.0:
            logger.warning(
                f"Total weight {total_weight:.4f} > 1.0 - "
                f"overleverage may occur"
            )
