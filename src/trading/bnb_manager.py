"""Auto-maintain BNB balance in Futures wallet (for 10% fee discount)."""

import logging
from typing import Any, Dict, Optional

from src.common.api_client import APIClient
from src.common.logger import setup_logger


class BnbManager:
    """Maintains BNB balance in the Futures wallet at a stable level.

    Uses Binance Futures Convert API to convert USDT → BNB.
    Only operates on mainnet; failures are logged without affecting trading.
    """

    TARGET_USD = 20.0
    THRESHOLD_USD = 15.0

    def __init__(self, api: APIClient, testnet: bool = True) -> None:
        self.api = api
        self.testnet = testnet
        self.logger = setup_logger("bnb_manager", "data/logs/bnb_manager.log")

    # ── public ───────────────────────────────────────────────

    def check_and_refill(self) -> None:
        """Check BNB balance and convert USDT → BNB if insufficient."""
        if self.testnet:
            return

        try:
            bnb_balance = self._get_bnb_balance()
            bnb_price = self._get_bnb_price()

            if bnb_price <= 0:
                self.logger.warning("Failed to fetch BNB price, skipping")
                return

            bnb_value_usd = bnb_balance * bnb_price

            if bnb_value_usd >= self.THRESHOLD_USD:
                self.logger.info(
                    f"BNB balance sufficient: {bnb_balance:.4f} BNB (${bnb_value_usd:.2f})"
                )
                return

            # Calculate deficit
            deficit_usd = self.TARGET_USD - bnb_value_usd
            self.logger.info(
                f"BNB balance insufficient: {bnb_balance:.4f} BNB (${bnb_value_usd:.2f}), "
                f"attempting to top up ${deficit_usd:.2f}"
            )

            # Convert USDT → BNB via Convert API
            self._convert_usdt_to_bnb(deficit_usd)

        except Exception as e:
            self.logger.error(f"BNB refill failed: {e}")

    # ── private ──────────────────────────────────────────────

    def _get_bnb_balance(self) -> float:
        """Fetch available BNB balance in Futures wallet."""
        balances = self.api.client.balance()
        for b in balances:
            if b["asset"] == "BNB":
                return float(b["availableBalance"])
        return 0.0

    def _get_bnb_price(self) -> float:
        """Fetch BNB/USDT mark price."""
        try:
            return self.api.get_mark_price("BNBUSDT")
        except Exception:
            return 0.0

    def _check_convert_available(self) -> Optional[Dict[str, Any]]:
        """Check whether USDT → BNB conversion is available and get limits."""
        result = self.api.client.sign_request(
            "GET",
            "/fapi/v1/convert/exchangeInfo",
            {"fromAsset": "USDT", "toAsset": "BNB"},
        )
        if isinstance(result, list) and len(result) > 0:
            return result[0]
        return None

    def _get_quote(self, from_amount: float) -> Optional[Dict[str, Any]]:
        """Request USDT → BNB conversion quote."""
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
        """Accept quote to execute conversion."""
        result = self.api.client.sign_request(
            "POST",
            "/fapi/v1/convert/acceptQuote",
            {"quoteId": quote_id},
        )
        return result if isinstance(result, dict) else None

    def _convert_usdt_to_bnb(self, usdt_amount: float) -> None:
        """Full flow for USDT → BNB conversion."""
        # 1. Check conversion availability
        info = self._check_convert_available()
        if info is None:
            self.logger.warning("USDT → BNB conversion unavailable (no exchangeInfo)")
            return

        min_from = float(info.get("fromAssetMinAmount", 0))
        max_from = float(info.get("fromAssetMaxAmount", 0))

        if usdt_amount < min_from:
            self.logger.info(
                f"Conversion amount ${usdt_amount:.2f} is below minimum ${min_from:.2f}, skipping"
            )
            return
        if max_from > 0 and usdt_amount > max_from:
            usdt_amount = max_from

        # 2. Request quote
        quote = self._get_quote(usdt_amount)
        if quote is None:
            self.logger.warning(f"Quote request failed (amount=${usdt_amount:.2f})")
            return

        quote_id = quote["quoteId"]
        to_amount = quote.get("toAmount", "?")
        ratio = quote.get("ratio", "?")
        self.logger.info(
            f"Quote received: {usdt_amount:.2f} USDT → {to_amount} BNB (rate={ratio})"
        )

        # 3. Accept quote
        accept = self._accept_quote(quote_id)
        if accept is None:
            self.logger.warning("Quote acceptance failed")
            return

        status = accept.get("orderStatus", "UNKNOWN")
        order_id = accept.get("orderId", "?")
        self.logger.info(
            f"Conversion complete: orderId={order_id}, status={status}"
        )
