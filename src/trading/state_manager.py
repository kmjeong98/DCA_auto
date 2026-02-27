"""Trading state persistence and recovery."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.logger import setup_logger
from src.trading.strategy import PositionState


class StateManager:
    """Position state persistence manager."""

    def __init__(self, state_dir: str = "data/state") -> None:
        """
        Initialize StateManager.

        Args:
            state_dir: Directory for state files
        """
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("state_manager", "data/logs/state_manager.log")

    def _get_state_path(self, symbol: str) -> Path:
        """Return state file path for symbol."""
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
        Save symbol state.

        Args:
            symbol: Symbol
            long_state: Long position state
            short_state: Short position state
            extra_data: Additional data (balance, etc.)
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
        Load symbol state.

        Args:
            symbol: Symbol

        Returns:
            {"long": PositionState, "short": PositionState, "extra": {...}} or None
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
        """Delete state file for symbol."""
        path = self._get_state_path(symbol)
        if path.exists():
            path.unlink()
            self.logger.info(f"State deleted: {symbol}")

    def list_symbols_with_state(self) -> list:
        """List symbols that have a state file."""
        symbols = []
        for path in self.state_dir.glob("*_state.json"):
            name = path.stem.replace("_state", "").replace("_", "/")
            symbols.append(name)
        return symbols


class TradeLogger:
    """Trade history logger."""

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
        Log a trade event.

        Args:
            symbol: Symbol
            event_type: Event type (ENTRY, DCA, TP, SL, etc.)
            side: "long" or "short"
            data: Additional data
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
        """Log entry."""
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
        """Log DCA fill."""
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
        """Log TP fill."""
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
        """Log SL fill."""
        self.log_trade(symbol, "SL", side, {
            "price": price,
            "amount": amount,
            "pnl": pnl,
        })
