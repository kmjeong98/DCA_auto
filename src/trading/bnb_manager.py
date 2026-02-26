"""Futures 지갑 BNB 잔고 자동 유지 (수수료 10% 할인용)."""

import logging
from typing import Any, Dict, Optional

from src.common.api_client import APIClient
from src.common.logger import setup_logger


class BnbManager:
    """Futures 지갑의 BNB 잔고를 일정 수준으로 유지한다.

    Binance Futures Convert API를 사용하여 USDT → BNB 전환.
    Mainnet에서만 동작하며, 실패 시 로그만 남기고 트레이딩에 영향을 주지 않는다.
    """

    TARGET_USD = 20.0
    THRESHOLD_USD = 15.0

    def __init__(self, api: APIClient, testnet: bool = True) -> None:
        self.api = api
        self.testnet = testnet
        self.logger = setup_logger("bnb_manager", "data/logs/bnb_manager.log")

    # ── public ───────────────────────────────────────────────

    def check_and_refill(self) -> None:
        """BNB 잔고를 확인하고 부족하면 USDT → BNB 전환."""
        if self.testnet:
            return

        try:
            bnb_balance = self._get_bnb_balance()
            bnb_price = self._get_bnb_price()

            if bnb_price <= 0:
                self.logger.warning("BNB 가격 조회 실패, 스킵")
                return

            bnb_value_usd = bnb_balance * bnb_price

            if bnb_value_usd >= self.THRESHOLD_USD:
                self.logger.info(
                    f"BNB 잔고 충분: {bnb_balance:.4f} BNB (${bnb_value_usd:.2f})"
                )
                return

            # 부족분 계산
            deficit_usd = self.TARGET_USD - bnb_value_usd
            self.logger.info(
                f"BNB 잔고 부족: {bnb_balance:.4f} BNB (${bnb_value_usd:.2f}), "
                f"${deficit_usd:.2f} 충전 시도"
            )

            # Convert API로 USDT → BNB 전환
            self._convert_usdt_to_bnb(deficit_usd)

        except Exception as e:
            self.logger.error(f"BNB 충전 실패: {e}")

    # ── private ──────────────────────────────────────────────

    def _get_bnb_balance(self) -> float:
        """Futures 지갑의 BNB 가용 잔고 조회."""
        balances = self.api.client.balance()
        for b in balances:
            if b["asset"] == "BNB":
                return float(b["availableBalance"])
        return 0.0

    def _get_bnb_price(self) -> float:
        """BNB/USDT 마크 가격 조회."""
        try:
            return self.api.get_mark_price("BNBUSDT")
        except Exception:
            return 0.0

    def _check_convert_available(self) -> Optional[Dict[str, Any]]:
        """USDT → BNB 전환 가능 여부 및 한도 확인."""
        result = self.api.client.sign_request(
            "GET",
            "/fapi/v1/convert/exchangeInfo",
            {"fromAsset": "USDT", "toAsset": "BNB"},
        )
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return None

    def _get_quote(self, from_amount: float) -> Optional[Dict[str, Any]]:
        """USDT → BNB 전환 견적 요청."""
        result = self.api.client.sign_request(
            "POST",
            "/fapi/v1/convert/getQuote",
            {
                "fromAsset": "USDT",
                "toAsset": "BNB",
                "fromAmount": str(from_amount),
            },
        )
        if isinstance(result, dict) and "quoteId" in result:
            return result
        return None

    def _accept_quote(self, quote_id: str) -> Optional[Dict[str, Any]]:
        """견적 수락하여 전환 실행."""
        result = self.api.client.sign_request(
            "POST",
            "/fapi/v1/convert/acceptQuote",
            {"quoteId": quote_id},
        )
        return result if isinstance(result, dict) else None

    def _convert_usdt_to_bnb(self, usdt_amount: float) -> None:
        """USDT → BNB 전환 전체 흐름."""
        # 1. 전환 가능 여부 확인
        info = self._check_convert_available()
        if info is None:
            self.logger.warning("USDT → BNB 전환 불가 (exchangeInfo 없음)")
            return

        min_from = float(info.get("fromAssetMinAmount", 0))
        max_from = float(info.get("fromAssetMaxAmount", 0))

        if usdt_amount < min_from:
            self.logger.info(
                f"전환 금액 ${usdt_amount:.2f}이 최소 ${min_from:.2f} 미만, 스킵"
            )
            return
        if max_from > 0 and usdt_amount > max_from:
            usdt_amount = max_from

        # 2. 견적 요청
        quote = self._get_quote(usdt_amount)
        if quote is None:
            self.logger.warning(f"견적 요청 실패 (amount=${usdt_amount:.2f})")
            return

        quote_id = quote["quoteId"]
        to_amount = quote.get("toAmount", "?")
        ratio = quote.get("ratio", "?")
        self.logger.info(
            f"견적 수신: {usdt_amount:.2f} USDT → {to_amount} BNB (rate={ratio})"
        )

        # 3. 견적 수락
        accept = self._accept_quote(quote_id)
        if accept is None:
            self.logger.warning("견적 수락 실패")
            return

        status = accept.get("orderStatus", "UNKNOWN")
        order_id = accept.get("orderId", "?")
        self.logger.info(
            f"전환 완료: orderId={order_id}, status={status}"
        )
