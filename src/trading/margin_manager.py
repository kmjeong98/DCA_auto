"""Per-coin margin persistence and update management."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.logger import setup_logger


class MarginManager:
    """Per-coin margin (capital) manager.

    - Saves allocated capital for each coin to data/margins/{SYMBOL}_margin.json
    - Restores saved capital on program restart (power recovery)
    - Only allows margin increases, ignores decreases (preserves DCA characteristics)
    """

    def __init__(self, margin_dir: str = "data/margins") -> None:
        self.margin_dir = Path(margin_dir)
        self.margin_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("margin_manager", "data/logs/margin_manager.log")

    def _get_path(self, symbol: str) -> Path:
        """Return margin file path for symbol."""
        safe_symbol = symbol.replace("/", "_")
        return self.margin_dir / f"{safe_symbol}_margin.json"

    def load_or_init(
        self,
        symbol: str,
        weight: float,
        total_balance: float,
    ) -> float:
        """
        Load margin file or create initial one.

        Args:
            symbol: Symbol (e.g. "BTC/USDT")
            weight: weight from config.json
            total_balance: Binance total balance

        Returns:
            Allocated capital
        """
        path = self._get_path(symbol)

        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                capital = float(data.get("capital", 0))
                if capital > 0:
                    self.logger.info(
                        f"[{symbol}] Loaded margin from file: ${capital:.2f}"
                    )
                    return capital
            except Exception as e:
                self.logger.warning(f"[{symbol}] Failed to load margin file: {e}")

        # Initial creation
        capital = total_balance * weight
        self.save(symbol, capital, total_balance, weight)
        self.logger.info(
            f"[{symbol}] Initialized margin: ${capital:.2f} "
            f"(balance=${total_balance:.2f} × weight={weight:.4f})"
        )
        return capital

    def save(
        self,
        symbol: str,
        capital: float,
        total_balance: float,
        weight: float,
    ) -> None:
        """Save margin file."""
        path = self._get_path(symbol)
        data = {
            "symbol": symbol,
            "capital": capital,
            "total_balance_at_update": total_balance,
            "weight_at_update": weight,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def try_update(
        self,
        symbol: str,
        weight: float,
        current_capital: float,
        total_balance: float,
        force: bool = False,
    ) -> float:
        """
        Attempt margin update (increase only, force allows decrease).

        Args:
            symbol: Symbol
            weight: weight from config.json
            current_capital: Currently allocated capital
            total_balance: Latest Binance balance
            force: If True, also allows decrease (e.g. when config weight changes)

        Returns:
            Updated capital (keeps current if decreased, unless force=True)
        """
        new_capital = total_balance * weight

        if force or new_capital >= current_capital:
            self.save(symbol, new_capital, total_balance, weight)
            self.logger.info(
                f"[{symbol}] Margin updated: ${current_capital:.2f} → ${new_capital:.2f}"
                f"{' (forced by config change)' if force else ''}"
            )
            return new_capital

        # Ignore decrease (recoverable given DCA characteristics)
        self.logger.info(
            f"[{symbol}] Margin decrease ignored: "
            f"${current_capital:.2f} → ${new_capital:.2f} (keeping current)"
        )
        return current_capital

    def load(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load margin file."""
        path = self._get_path(symbol)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def delete(self, symbol: str) -> None:
        """Remove margin file when coin is deleted."""
        path = self._get_path(symbol)
        if path.exists():
            path.unlink()
            self.logger.info(f"[{symbol}] Margin file deleted")
