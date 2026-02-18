"""트레이딩 설정 로더 (config.json)."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.common.logger import setup_logger


@dataclass
class SymbolConfig:
    """심볼별 설정."""
    symbol: str      # "BTC/USDT"
    weight: float    # 자본 배분 비율


@dataclass
class TradingConfig:
    """트레이딩 전역 설정."""
    cooldown_hours: int
    fee_rate: float
    symbols: Dict[str, SymbolConfig]  # key: "BTC_USDT"

    @classmethod
    def load(cls, path: str = "config.json") -> "TradingConfig":
        """config.json 파일 로드."""
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
            fee_rate=float(data.get("fee_rate", 0.0005)),
            symbols=symbols,
        )

    def get_symbol_names(self) -> List[str]:
        """["BTC/USDT", "ETH/USDT", ...] 형식으로 반환."""
        return [cfg.symbol for cfg in self.symbols.values()]

    def get_safe_names(self) -> List[str]:
        """["BTC_USDT", "ETH_USDT", ...] 키 형식으로 반환."""
        return list(self.symbols.keys())

    def get_weight(self, symbol: str) -> float:
        """심볼의 weight 조회. symbol은 "BTC/USDT" 또는 "BTC_USDT" 형식."""
        safe_key = symbol.replace("/", "_")
        cfg = self.symbols.get(safe_key)
        if cfg is None:
            raise KeyError(f"Symbol {symbol} not in config")
        return cfg.weight

    def get_symbol_capital(self, symbol: str, total_balance: float) -> float:
        """심볼에 할당할 자본 계산."""
        return total_balance * self.get_weight(symbol)

    def validate(self) -> None:
        """설정 유효성 검증."""
        logger = setup_logger("config", "logs/config.log")

        if not self.symbols:
            raise ValueError("No symbols configured")

        # params 파일 존재 확인
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

        # weight 합계 표시 (정규화하지 않음)
        total_weight = sum(cfg.weight for cfg in self.symbols.values())
        logger.info(f"Total weight: {total_weight:.4f} ({len(self.symbols)} symbols)")
        if total_weight > 1.0:
            logger.warning(
                f"Total weight {total_weight:.4f} > 1.0 - "
                f"overleverage may occur"
            )
