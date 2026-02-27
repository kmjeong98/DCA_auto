"""Exchange API connection and common calls (based on binance-futures-connector)."""

import math
import os
from typing import Any, Dict, List, Optional

from binance.error import ClientError
from binance.um_futures import UMFutures
from dotenv import load_dotenv

load_dotenv()


class APIClient:
    """Binance Futures API client (based on binance-futures-connector)."""

    TESTNET_BASE = "https://testnet.binancefuture.com"
    MAINNET_BASE = "https://fapi.binance.com"

    def __init__(self, testnet: bool = True) -> None:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        base_url = self.TESTNET_BASE if testnet else self.MAINNET_BASE

        self.client = UMFutures(
            key=api_key,
            secret=api_secret,
            base_url=base_url,
        )

        self.testnet = testnet
        self._exchange_info: Optional[Dict[str, Any]] = None
        self._symbol_cache: Dict[str, Dict[str, Any]] = {}

    def _ensure_exchange_info(self) -> None:
        """Load exchange info (once on first call)."""
        if self._exchange_info is None:
            self._exchange_info = self.client.exchange_info()
            for s in self._exchange_info["symbols"]:
                self._symbol_cache[s["symbol"]] = s

    def _to_binance_symbol(self, symbol: str) -> str:
        """Convert BTC/USDT -> BTCUSDT."""
        return symbol.replace("/", "")

    def _get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch symbol info."""
        self._ensure_exchange_info()
        binance_symbol = self._to_binance_symbol(symbol)
        if binance_symbol not in self._symbol_cache:
            raise ValueError(f"Unknown symbol: {symbol}")
        return self._symbol_cache[binance_symbol]

    def _get_filters(self, symbol: str) -> Dict[str, Dict[str, Any]]:
        """Return symbol filter info as a dictionary."""
        info = self._get_symbol_info(symbol)
        return {f["filterType"]: f for f in info["filters"]}

    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage."""
        binance_symbol = self._to_binance_symbol(symbol)
        return self.client.change_leverage(symbol=binance_symbol, leverage=leverage)

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> Dict[str, Any]:
        """Set margin mode."""
        binance_symbol = self._to_binance_symbol(symbol)
        margin_type = "ISOLATED" if mode == "isolated" else "CROSSED"
        try:
            return self.client.change_margin_type(symbol=binance_symbol, marginType=margin_type)
        except ClientError as e:
            if e.error_code == -4046:
                return {"info": "already_set"}
            raise

    def set_position_mode(self, hedge_mode: bool = True) -> Dict[str, Any]:
        """Set position mode (Hedge Mode / One-way Mode)."""
        try:
            return self.client.change_position_mode(
                dualSidePosition="true" if hedge_mode else "false"
            )
        except ClientError as e:
            if "No need to change position side" in str(e.error_message):
                return {"info": "already_set"}
            if e.error_code == -4067:
                # Cannot change due to open orders — treat as already set
                return {"info": "already_set"}
            raise

    def get_balance(self) -> float:
        """Fetch USDT available balance."""
        balances = self.client.balance()
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["availableBalance"])
        return 0.0

    def get_total_balance(self) -> float:
        """Fetch USDT total balance."""
        balances = self.client.balance()
        for b in balances:
            if b["asset"] == "USDT":
                return float(b["balance"])
        return 0.0

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch current positions."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = self._to_binance_symbol(symbol)

        raw_positions = self.client.get_position_risk(**kwargs)

        positions = []
        for p in raw_positions:
            amount = abs(float(p.get("positionAmt", 0)))
            if amount > 0:
                position_side = p.get("positionSide", "BOTH")
                if position_side == "LONG":
                    side = "long"
                elif position_side == "SHORT":
                    side = "short"
                else:
                    side = "long" if float(p.get("positionAmt", 0)) > 0 else "short"

                positions.append({
                    "symbol": p["symbol"],
                    "side": side,
                    "contracts": amount,
                    "entryPrice": float(p.get("entryPrice", 0)),
                    "markPrice": float(p.get("markPrice", 0)),
                    "unrealizedPnl": float(p.get("unRealizedProfit", 0)),
                    "positionSide": position_side,
                })

        return positions

    def get_position(self, symbol: str, side: str) -> Optional[Dict[str, Any]]:
        """Fetch position for a specific symbol/side."""
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
        """Place market order."""
        binance_symbol = self._to_binance_symbol(symbol)
        amount = self.round_amount(symbol, amount)
        kwargs: Dict[str, Any] = {
            "symbol": binance_symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": amount,
            "positionSide": position_side,
            "newOrderRespType": "RESULT",
        }
        if reduce_only:
            kwargs["reduceOnly"] = "true"

        result = self.client.new_order(**kwargs)
        return self._normalize_order_response(result)

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        position_side: str = "LONG",
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        """Place limit order."""
        binance_symbol = self._to_binance_symbol(symbol)
        amount = self.round_amount(symbol, amount)
        price = self.round_price(symbol, price)
        kwargs: Dict[str, Any] = {
            "symbol": binance_symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "quantity": amount,
            "price": price,
            "timeInForce": "GTC",
            "positionSide": position_side,
        }
        if reduce_only:
            kwargs["reduceOnly"] = "true"

        result = self.client.new_order(**kwargs)
        return self._normalize_order_response(result)

    def place_stop_loss(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_price: float,
        position_side: str = "LONG",
    ) -> Dict[str, Any]:
        """Place stop-loss order (Algo Order STOP_MARKET — mark price trigger)."""
        binance_symbol = self._to_binance_symbol(symbol)
        amount = self.round_amount(symbol, amount)
        stop_price = self.round_price(symbol, stop_price)

        result = self.client.sign_request(
            "POST",
            "/fapi/v1/algoOrder",
            {
                "algoType": "CONDITIONAL",
                "symbol": binance_symbol,
                "side": side.upper(),
                "positionSide": position_side,
                "type": "STOP_MARKET",
                "quantity": amount,
                "triggerPrice": stop_price,
                "workingType": "MARK_PRICE",
            },
        )
        return self._normalize_algo_order_response(result)

    def place_take_profit(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        position_side: str = "LONG",
    ) -> Dict[str, Any]:
        """Place take-profit order (LIMIT). In Hedge Mode, direction is set via positionSide."""
        binance_symbol = self._to_binance_symbol(symbol)
        amount = self.round_amount(symbol, amount)
        price = self.round_price(symbol, price)
        result = self.client.new_order(
            symbol=binance_symbol,
            side=side.upper(),
            type="LIMIT",
            quantity=amount,
            price=price,
            positionSide=position_side,
            timeInForce="GTC",
        )
        return self._normalize_order_response(result)

    def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel order."""
        binance_symbol = self._to_binance_symbol(symbol)
        return self.client.cancel_order(symbol=binance_symbol, orderId=int(order_id))

    def cancel_algo_order(self, symbol: str, algo_id: str) -> Dict[str, Any]:
        """Cancel algo order."""
        return self.client.sign_request(
            "DELETE",
            "/fapi/v1/algoOrder",
            {"algoId": int(algo_id)},
        )

    def get_open_algo_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open algo orders."""
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = self._to_binance_symbol(symbol)
        result = self.client.sign_request("GET", "/fapi/v1/openAlgoOrders", params)
        orders = result.get("orders", []) if isinstance(result, dict) else result
        return [self._normalize_algo_order_response(o) for o in orders]

    def cancel_all_orders(self, symbol: str) -> List[Dict[str, Any]]:
        """Cancel all open orders for a symbol (regular + algo)."""
        binance_symbol = self._to_binance_symbol(symbol)
        try:
            self.client.cancel_open_orders(symbol=binance_symbol)
        except ClientError:
            pass
        # Also cancel algo orders
        try:
            algo_orders = self.get_open_algo_orders(symbol)
            for ao in algo_orders:
                try:
                    self.cancel_algo_order(symbol, str(ao["id"]))
                except Exception:
                    pass
        except Exception:
            pass
        return []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch open orders."""
        kwargs = {}
        if symbol:
            kwargs["symbol"] = self._to_binance_symbol(symbol)

        orders = self.client.get_orders(**kwargs)
        return [self._normalize_order_response(o) for o in orders]

    def get_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Fetch order status."""
        binance_symbol = self._to_binance_symbol(symbol)
        result = self.client.query_order(symbol=binance_symbol, orderId=int(order_id))
        return self._normalize_order_response(result)

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch current price."""
        binance_symbol = self._to_binance_symbol(symbol)
        return self.client.ticker_price(symbol=binance_symbol)

    def get_mark_price(self, symbol: str) -> float:
        """Fetch mark price."""
        binance_symbol = self._to_binance_symbol(symbol)
        result = self.client.mark_price(symbol=binance_symbol)
        return float(result["markPrice"])

    def get_min_order_amount(self, symbol: str) -> float:
        """Fetch minimum order quantity."""
        filters = self._get_filters(symbol)
        lot_size = filters.get("LOT_SIZE", {})
        return float(lot_size.get("minQty", 0.001))

    def get_precision(self, symbol: str) -> Dict[str, int]:
        """Fetch price/quantity decimal precision."""
        filters = self._get_filters(symbol)

        tick_size = float(filters.get("PRICE_FILTER", {}).get("tickSize", "0.01"))
        step_size = float(filters.get("LOT_SIZE", {}).get("stepSize", "0.001"))

        price_precision = self._step_to_precision(tick_size)
        amount_precision = self._step_to_precision(step_size)

        return {"price": price_precision, "amount": amount_precision}

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to symbol precision."""
        filters = self._get_filters(symbol)
        tick_size = float(filters.get("PRICE_FILTER", {}).get("tickSize", "0.01"))
        return self._round_step(price, tick_size)

    def round_amount(self, symbol: str, amount: float) -> float:
        """Round amount to symbol precision."""
        filters = self._get_filters(symbol)
        step_size = float(filters.get("LOT_SIZE", {}).get("stepSize", "0.001"))
        return self._round_step(amount, step_size)

    def get_account_equity(self) -> float:
        """Fetch totalMarginBalance (actual equity including unrealized PnL)."""
        account = self.client.account()
        return float(account["totalMarginBalance"])

    def new_listen_key(self) -> str:
        """Create listen key for User Data Stream."""
        result = self.client.new_listen_key()
        return result["listenKey"]

    def renew_listen_key(self, listen_key: str) -> None:
        """Renew listen key."""
        self.client.renew_listen_key(listenKey=listen_key)

    def close_listen_key(self, listen_key: str) -> None:
        """Delete listen key."""
        self.client.close_listen_key(listenKey=listen_key)

    @staticmethod
    def _step_to_precision(step: float) -> int:
        """Convert stepSize/tickSize to decimal precision."""
        if step >= 1:
            return 0
        return max(0, int(round(-math.log10(step))))

    @staticmethod
    def _round_step(value: float, step: float) -> float:
        """Floor value to step unit."""
        if step <= 0:
            return value
        precision = max(0, int(round(-math.log10(step))))
        return round(math.floor(value / step) * step, precision)

    @staticmethod
    def _normalize_order_response(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize order response to a format compatible with existing code."""
        return {
            "id": str(raw.get("orderId", "")),
            "symbol": raw.get("symbol", ""),
            "side": raw.get("side", "").lower(),
            "type": raw.get("type", ""),
            "status": raw.get("status", ""),
            "price": float(raw.get("price", 0)),
            "average": float(raw.get("avgPrice", 0)),
            "amount": float(raw.get("origQty", 0)),
            "filled": float(raw.get("executedQty", 0)),
            "positionSide": raw.get("positionSide", ""),
            "reduceOnly": raw.get("reduceOnly", False),
            "raw": raw,
        }

    @staticmethod
    def _normalize_algo_order_response(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize algo order response to a format compatible with existing code."""
        return {
            "id": str(raw.get("algoId", "")),
            "symbol": raw.get("symbol", ""),
            "side": raw.get("side", "").lower(),
            "type": raw.get("orderType", raw.get("type", "")),
            "status": raw.get("algoStatus", ""),
            "price": float(raw.get("triggerPrice", 0)),
            "average": 0.0,
            "amount": float(raw.get("quantity", 0)),
            "filled": 0.0,
            "positionSide": raw.get("positionSide", ""),
            "reduceOnly": raw.get("reduceOnly", False),
            "raw": raw,
        }
