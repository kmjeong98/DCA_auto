"""트레이딩 상태 저장 및 복구."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.logger import setup_logger
from src.trading.strategy import PositionState


class StateManager:
    """포지션 상태 영속화 관리."""

    def __init__(self, state_dir: str = "data/state") -> None:
        """
        StateManager 초기화.

        Args:
            state_dir: 상태 파일 저장 디렉토리
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("state_manager", "data/logs/state_manager.log")

    def _get_state_path(self, symbol: str) -> Path:
        """심볼별 상태 파일 경로."""
        safe_symbol = symbol.replace("/", "_")
        return self.state_dir / f"{safe_symbol}_state.json"

    def save_state(
        self,
        symbol: str,
        long_state: PositionState,
        short_state: PositionState,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        심볼 상태 저장.

        Args:
            symbol: 심볼
            long_state: Long 포지션 상태
            short_state: Short 포지션 상태
            extra_data: 추가 데이터 (잔고 등)
        """
        state = {
            "symbol": symbol,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "long": long_state.to_dict(),
            "short": short_state.to_dict(),
        }

        if extra_data:
            state["extra"] = extra_data

        path = self._get_state_path(symbol)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self.logger.debug(f"State saved: {symbol}")

    def load_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        심볼 상태 로드.

        Args:
            symbol: 심볼

        Returns:
            {"long": PositionState, "short": PositionState, "extra": {...}} 또는 None
        """
        path = self._get_state_path(symbol)

        if not path.exists():
            self.logger.info(f"No state file for {symbol}")
            return None

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "long": PositionState.from_dict(data["long"]),
                "short": PositionState.from_dict(data["short"]),
                "extra": data.get("extra", {}),
                "updated_at": data.get("updated_at"),
            }

        except Exception as e:
            self.logger.error(f"Failed to load state for {symbol}: {e}")
            return None

    def delete_state(self, symbol: str) -> None:
        """심볼 상태 파일 삭제."""
        path = self._get_state_path(symbol)
        if path.exists():
            path.unlink()
            self.logger.info(f"State deleted: {symbol}")

    def list_symbols_with_state(self) -> list:
        """상태 파일이 있는 심볼 목록."""
        symbols = []
        for path in self.state_dir.glob("*_state.json"):
            name = path.stem.replace("_state", "").replace("_", "/")
            symbols.append(name)
        return symbols


class TradeLogger:
    """거래 이력 로깅."""

    def __init__(self, log_dir: str = "data/logs/trades") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        date_str = datetime.now().strftime("%Y%m")
        return self.log_dir / f"{safe_symbol}_{date_str}.jsonl"

    def log_trade(
        self,
        symbol: str,
        event_type: str,
        side: str,
        data: Dict[str, Any],
    ) -> None:
        """
        거래 이벤트 로깅.

        Args:
            symbol: 심볼
            event_type: 이벤트 타입 (ENTRY, DCA, TP, SL 등)
            side: "long" 또는 "short"
            data: 추가 데이터
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "event": event_type,
            "side": side,
            **data,
        }

        path = self._get_log_path(symbol)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_entry(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        margin: float,
    ) -> None:
        """진입 로깅."""
        self.log_trade(symbol, "ENTRY", side, {
            "price": price,
            "amount": amount,
            "margin": margin,
        })

    def log_dca(
        self,
        symbol: str,
        side: str,
        level: int,
        price: float,
        amount: float,
        margin: float,
        new_avg: float,
    ) -> None:
        """DCA 체결 로깅."""
        self.log_trade(symbol, "DCA", side, {
            "level": level,
            "price": price,
            "amount": amount,
            "margin": margin,
            "new_avg_price": new_avg,
        })

    def log_tp(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        pnl: float,
    ) -> None:
        """TP 체결 로깅."""
        self.log_trade(symbol, "TP", side, {
            "price": price,
            "amount": amount,
            "pnl": pnl,
        })

    def log_sl(
        self,
        symbol: str,
        side: str,
        price: float,
        amount: float,
        pnl: float,
    ) -> None:
        """SL 체결 로깅."""
        self.log_trade(symbol, "SL", side, {
            "price": price,
            "amount": amount,
            "pnl": pnl,
        })
