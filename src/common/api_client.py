"""거래소 API 연결 및 공통 호출."""

import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import ccxt

load_dotenv()


class APIClient:
    """Binance Futures API 클라이언트 (ccxt 기반)."""

    # Testnet URLs
    TESTNET_BASE = "https://testnet.binancefuture.com"

    def __init__(self, testnet: bool = True) -> None:
        """
        API 클라이언트 초기화.

        Args:
            testnet: True면 Testnet, False면 Mainnet
        """
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        options = {
            "defaultType": "future",
            "adjustForTimeDifference": True,
        }

        self.exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": options,
        })

        # Testnet URL 직접 설정 (sandbox mode 대신)
        if testnet:
            self.exchange.urls["api"]["fapiPublic"] = self.TESTNET_BASE + "/fapi/v1"
            self.exchange.urls["api"]["fapiPrivate"] = self.TESTNET_BASE + "/fapi/v1"
            self.exchange.urls["api"]["fapiPublicV2"] = self.TESTNET_BASE + "/fapi/v2"
            self.exchange.urls["api"]["fapiPrivateV2"] = self.TESTNET_BASE + "/fapi/v2"

        self.testnet = testnet
        self._markets_loaded = False

    def _ensure_markets(self) -> None:
        """시장 정보 로드 (최초 1회)."""
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        레버리지 설정.

        Args:
            symbol: 심볼 (예: "BTC/USDT")
            leverage: 레버리지 배수

        Returns:
            API 응답
        """
        self._ensure_markets()
        return self.exchange.set_leverage(leverage, symbol)

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> Dict[str, Any]:
        """
        마진 모드 설정.

        Args:
            symbol: 심볼
            mode: "isolated" 또는 "cross"

        Returns:
            API 응답
        """
        self._ensure_markets()
        try:
            return self.exchange.set_margin_mode(mode, symbol)
        except ccxt.ExchangeError as e:
            # 이미 설정된 경우 무시
            if "No need to change margin type" in str(e):
                return {"info": "already_set"}
            raise

    def set_position_mode(self, hedge_mode: bool = True) -> Dict[str, Any]:
        """
        포지션 모드 설정 (Hedge Mode / One-way Mode).

        Args:
            hedge_mode: True면 Hedge Mode (양방향 동시), False면 One-way

        Returns:
            API 응답
        """
        self._ensure_markets()
        try:
            return self.exchange.fapiPrivatePostPositionSideDual({
                "dualSidePosition": "true" if hedge_mode else "false"
            })
        except ccxt.ExchangeError as e:
            if "No need to change position side" in str(e):
                return {"info": "already_set"}
            raise

    def get_balance(self) -> float:
        """
        USDT 가용 잔고 조회.

        Returns:
            가용 USDT 잔고
        """
        self._ensure_markets()
        balance = self.exchange.fetch_balance()
        return float(balance.get("USDT", {}).get("free", 0))

    def get_total_balance(self) -> float:
        """
        USDT 총 잔고 조회 (포지션 포함).

        Returns:
            총 USDT 잔고
        """
        self._ensure_markets()
        balance = self.exchange.fetch_balance()
        return float(balance.get("USDT", {}).get("total", 0))

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        현재 포지션 조회.

        Args:
            symbol: 특정 심볼만 조회 (None이면 전체)

        Returns:
            포지션 리스트
        """
        self._ensure_markets()
        positions = self.exchange.fetch_positions([symbol] if symbol else None)
        # 수량이 있는 포지션만 필터
        return [p for p in positions if float(p.get("contracts", 0)) > 0]

    def get_position(self, symbol: str, side: str) -> Optional[Dict[str, Any]]:
        """
        특정 심볼/방향 포지션 조회.

        Args:
            symbol: 심볼
            side: "long" 또는 "short"

        Returns:
            포지션 정보 또는 None
        """
        positions = self.get_positions(symbol)
        for p in positions:
            if p.get("side") == side:
                return p
        return None

    def place_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        position_side: str = "LONG",
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        시장가 주문.

        Args:
            symbol: 심볼
            side: "buy" 또는 "sell"
            amount: 수량 (contracts)
            position_side: "LONG" 또는 "SHORT" (Hedge Mode)
            reduce_only: 청산 전용 여부

        Returns:
            주문 결과
        """
        self._ensure_markets()
        params = {"positionSide": position_side}
        if reduce_only:
            params["reduceOnly"] = True

        return self.exchange.create_order(
            symbol=symbol,
            type="market",
            side=side,
            amount=amount,
            params=params,
        )

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        position_side: str = "LONG",
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """
        지정가 주문.

        Args:
            symbol: 심볼
            side: "buy" 또는 "sell"
            amount: 수량
            price: 가격
            position_side: "LONG" 또는 "SHORT"
            reduce_only: 청산 전용 여부

        Returns:
            주문 결과
        """
        self._ensure_markets()
        params = {"positionSide": position_side}
        if reduce_only:
            params["reduceOnly"] = True

        return self.exchange.create_order(
            symbol=symbol,
            type="limit",
            side=side,
            amount=amount,
            price=price,
            params=params,
        )

    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        position_side: str = "LONG",
    ) -> Dict[str, Any]:
        """
        스탑로스 주문 (시장가 트리거).

        Args:
            symbol: 심볼
            side: "buy" (숏 청산) 또는 "sell" (롱 청산)
            amount: 수량
            stop_price: 트리거 가격
            position_side: "LONG" 또는 "SHORT"

        Returns:
            주문 결과
        """
        self._ensure_markets()
        return self.exchange.create_order(
            symbol=symbol,
            type="stop_market",
            side=side,
            amount=amount,
            params={
                "stopPrice": stop_price,
                "positionSide": position_side,
                "reduceOnly": True,
            },
        )

    def place_take_profit(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        position_side: str = "LONG",
    ) -> Dict[str, Any]:
        """
        테이크프로핏 주문 (시장가 트리거).

        Args:
            symbol: 심볼
            side: "buy" (숏 청산) 또는 "sell" (롱 청산)
            amount: 수량
            stop_price: 트리거 가격
            position_side: "LONG" 또는 "SHORT"

        Returns:
            주문 결과
        """
        self._ensure_markets()
        return self.exchange.create_order(
            symbol=symbol,
            type="take_profit_market",
            side=side,
            amount=amount,
            params={
                "stopPrice": stop_price,
                "positionSide": position_side,
                "reduceOnly": True,
            },
        )

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        주문 취소.

        Args:
            symbol: 심볼
            order_id: 주문 ID

        Returns:
            취소 결과
        """
        self._ensure_markets()
        return self.exchange.cancel_order(order_id, symbol)

    def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """
        심볼의 모든 미체결 주문 취소.

        Args:
            symbol: 심볼

        Returns:
            취소된 주문 리스트
        """
        self._ensure_markets()
        try:
            return self.exchange.cancel_all_orders(symbol)
        except ccxt.ExchangeError:
            return []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        미체결 주문 조회.

        Args:
            symbol: 심볼 (None이면 전체)

        Returns:
            미체결 주문 리스트
        """
        self._ensure_markets()
        return self.exchange.fetch_open_orders(symbol)

    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        주문 상태 조회.

        Args:
            symbol: 심볼
            order_id: 주문 ID

        Returns:
            주문 정보
        """
        self._ensure_markets()
        return self.exchange.fetch_order(order_id, symbol)

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        현재가 조회.

        Args:
            symbol: 심볼

        Returns:
            티커 정보
        """
        self._ensure_markets()
        return self.exchange.fetch_ticker(symbol)

    def get_mark_price(self, symbol: str) -> float:
        """
        마크 가격 조회.

        Args:
            symbol: 심볼

        Returns:
            마크 가격
        """
        ticker = self.get_ticker(symbol)
        return float(ticker.get("markPrice", ticker.get("last", 0)))

    def get_min_order_amount(self, symbol: str) -> float:
        """
        최소 주문 수량 조회.

        Args:
            symbol: 심볼

        Returns:
            최소 수량
        """
        self._ensure_markets()
        market = self.exchange.market(symbol)
        return float(market.get("limits", {}).get("amount", {}).get("min", 0.001))

    def get_precision(self, symbol: str) -> Dict[str, int]:
        """
        가격/수량 소수점 정밀도 조회.

        Args:
            symbol: 심볼

        Returns:
            {"price": int, "amount": int}
        """
        self._ensure_markets()
        market = self.exchange.market(symbol)
        return {
            "price": market.get("precision", {}).get("price", 2),
            "amount": market.get("precision", {}).get("amount", 3),
        }

    def round_price(self, symbol: str, price: float) -> float:
        """가격을 심볼의 정밀도에 맞게 반올림."""
        precision = self.get_precision(symbol)
        return round(price, precision["price"])

    def round_amount(self, symbol: str, amount: float) -> float:
        """수량을 심볼의 정밀도에 맞게 반올림."""
        precision = self.get_precision(symbol)
        return round(amount, precision["amount"])
