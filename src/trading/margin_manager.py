"""코인별 마진 영속화 및 업데이트 관리."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.logger import setup_logger


class MarginManager:
    """코인별 마진(자본) 관리.

    - 각 코인의 할당 자본을 data/margins/{SYMBOL}_margin.json에 저장
    - 프로그램 재시작 시 저장된 capital로 복구 (정전 복구)
    - 마진 증가만 허용, 감소는 무시 (DCA 특성 보호)
    """

    def __init__(self, margin_dir: str = "data/margins") -> None:
        self.margin_dir = Path(margin_dir)
        self.margin_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("margin_manager", "data/logs/margin_manager.log")

    def _get_path(self, symbol: str) -> Path:
        """심볼별 마진 파일 경로."""
        safe_symbol = symbol.replace("/", "_")
        return self.margin_dir / f"{safe_symbol}_margin.json"

    def load_or_init(
        self,
        symbol: str,
        weight: float,
        total_balance: float,
    ) -> float:
        """
        마진 파일 로드 또는 초기 생성.

        Args:
            symbol: 심볼 (예: "BTC/USDT")
            weight: config.json의 weight
            total_balance: Binance 총 잔고

        Returns:
            할당된 capital
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

        # 초기 생성
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
        """마진 파일 저장."""
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
    ) -> float:
        """
        마진 업데이트 시도 (증가만 허용).

        Args:
            symbol: 심볼
            weight: config.json의 weight
            current_capital: 현재 할당된 capital
            total_balance: Binance 최신 잔고

        Returns:
            업데이트된 capital (감소 시 기존 값 유지)
        """
        new_capital = total_balance * weight

        if new_capital >= current_capital:
            self.save(symbol, new_capital, total_balance, weight)
            self.logger.info(
                f"[{symbol}] Margin updated: ${current_capital:.2f} → ${new_capital:.2f}"
            )
            return new_capital

        # 감소 무시 (DCA 특성상 복구 가능)
        self.logger.info(
            f"[{symbol}] Margin decrease ignored: "
            f"${current_capital:.2f} → ${new_capital:.2f} (keeping current)"
        )
        return current_capital

    def load(self, symbol: str) -> Optional[Dict[str, Any]]:
        """마진 파일 로드."""
        path = self._get_path(symbol)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def delete(self, symbol: str) -> None:
        """코인 삭제 시 마진 파일 제거."""
        path = self._get_path(symbol)
        if path.exists():
            path.unlink()
            self.logger.info(f"[{symbol}] Margin file deleted")
