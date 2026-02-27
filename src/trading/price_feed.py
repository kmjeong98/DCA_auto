"""Real-time price feed via WebSocket (based on binance-futures-connector)."""

import json
import threading
import time
from typing import Callable, Dict, List, Optional

from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from src.common.logger import setup_logger


class PriceFeed:
    """Binance Futures WebSocket price feed."""

    MAINNET_WS = "wss://fstream.binance.com"
    TESTNET_WS = "wss://fstream.binancefuture.com"

    def __init__(
        self,
        symbols: List[str],
        on_price_update: Callable[[str, float], None],
        testnet: bool = True,
    ) -> None:
        self.symbols = symbols
        self.on_price_update = on_price_update
        self.testnet = testnet

        self.ws_url = self.TESTNET_WS if testnet else self.MAINNET_WS
        self.logger = setup_logger("price_feed", "data/logs/price_feed.log")

        self._prices: Dict[str, float] = {}
        self._lock = threading.Lock()

        self._ws_client: Optional[UMFuturesWebsocketClient] = None
        self._running = False

        # Symbol mapping (BTC/USDT -> btcusdt)
        self._stream_to_symbol: Dict[str, str] = {}
        for symbol in symbols:
            stream_name = symbol.replace("/", "").lower()
            self._stream_to_symbol[stream_name] = symbol

    def _on_message(self, _, message: str) -> None:
        """Handle WebSocket message."""
        try:
            data = json.loads(message)

            # combined stream format
            if "stream" in data:
                data = data["data"]

            if data.get("e") == "markPriceUpdate":
                raw_symbol = data["s"].lower()
                mark_price = float(data["p"])

                symbol = self._stream_to_symbol.get(raw_symbol)
                if symbol:
                    with self._lock:
                        self._prices[symbol] = mark_price

                    try:
                        self.on_price_update(symbol, mark_price)
                    except Exception as e:
                        self.logger.error(f"Price callback error: {e}")

        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON: {message[:100]}")
        except Exception as e:
            self.logger.error(f"Message processing error: {e}")

    def _on_close(self, _) -> None:
        """Handle WebSocket connection close."""
        self.logger.warning("WebSocket closed")
        if self._running:
            self.logger.info("Reconnecting in 5s...")
            time.sleep(5)
            self._connect()

    def _on_error(self, _, error) -> None:
        """Handle WebSocket error."""
        self.logger.error(f"WebSocket error: {error}")

    def _on_open(self, _) -> None:
        """WebSocket connection established."""
        self.logger.info(f"WebSocket connected, symbols: {self.symbols}")

    def _connect(self) -> None:
        """Connect and subscribe via WebSocket (Combined Stream — single connection)."""
        self._ws_client = UMFuturesWebsocketClient(
            stream_url=self.ws_url,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
            on_open=self._on_open,
            is_combined=True,
        )

        for symbol in self.symbols:
            stream_name = symbol.replace("/", "").lower()
            self._ws_client.mark_price(symbol=stream_name, speed=1)

        self.logger.info(f"Subscribed to {len(self.symbols)} mark price streams (combined)")

    def start(self) -> None:
        """Start WebSocket."""
        if self._running:
            return

        self._running = True
        self._connect()
        self.logger.info("PriceFeed started")

        # Wait for connection
        time.sleep(2)

    def stop(self) -> None:
        """Stop WebSocket."""
        self._running = False
        if self._ws_client:
            self._ws_client.stop()
        self.logger.info("PriceFeed stopped")

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price (cached)."""
        with self._lock:
            return self._prices.get(symbol)

    def get_all_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        with self._lock:
            return self._prices.copy()

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._ws_client is not None and self._running

    def add_symbol(self, symbol: str) -> None:
        """Add symbol (requires reconnection)."""
        if symbol not in self.symbols:
            stream_name = symbol.replace("/", "").lower()
            self.symbols.append(symbol)
            self._stream_to_symbol[stream_name] = symbol

            if self._running:
                self.stop()
                time.sleep(1)
                self.start()


class OrderUpdateFeed:
    """User data stream (order/position updates)."""

    MAINNET_WS = "wss://fstream.binance.com"
    TESTNET_WS = "wss://fstream.binancefuture.com"

    def __init__(
        self,
        listen_key: str,
        on_order_update: Callable[[Dict], None],
        on_position_update: Callable[[Dict], None],
        testnet: bool = True,
    ) -> None:
        self.listen_key = listen_key
        self.on_order_update = on_order_update
        self.on_position_update = on_position_update
        self.testnet = testnet

        self.ws_url = self.TESTNET_WS if testnet else self.MAINNET_WS
        self.logger = setup_logger("order_feed", "data/logs/order_feed.log")

        self._ws_client: Optional[UMFuturesWebsocketClient] = None
        self._running = False

    def _on_message(self, _, message: str) -> None:
        """Handle message."""
        try:
            data = json.loads(message)
            event_type = data.get("e")

            if event_type == "ORDER_TRADE_UPDATE":
                order_data = data.get("o", {})
                self.on_order_update({
                    "symbol": order_data.get("s"),
                    "order_id": order_data.get("i"),
                    "client_order_id": order_data.get("c"),
                    "side": order_data.get("S"),
                    "position_side": order_data.get("ps"),
                    "type": order_data.get("o"),
                    "status": order_data.get("X"),
                    "price": float(order_data.get("p", 0)),
                    "avg_price": float(order_data.get("ap", 0)),
                    "amount": float(order_data.get("q", 0)),
                    "filled": float(order_data.get("z", 0)),
                    "pnl": float(order_data.get("rp", 0)),
                })

            elif event_type == "ACCOUNT_UPDATE":
                update_data = data.get("a", {})
                positions = update_data.get("P", [])
                for pos in positions:
                    self.on_position_update({
                        "symbol": pos.get("s"),
                        "position_side": pos.get("ps"),
                        "amount": float(pos.get("pa", 0)),
                        "entry_price": float(pos.get("ep", 0)),
                        "unrealized_pnl": float(pos.get("up", 0)),
                    })

        except Exception as e:
            self.logger.error(f"Message error: {e}")

    def _on_close(self, _) -> None:
        self.logger.warning("WebSocket closed")
        if self._running:
            time.sleep(5)
            self._connect()

    def _on_error(self, _, error) -> None:
        self.logger.error(f"WebSocket error: {error}")

    def _on_open(self, _) -> None:
        self.logger.info("User data stream connected")

    def _connect(self) -> None:
        self._ws_client = UMFuturesWebsocketClient(
            stream_url=self.ws_url,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
            on_open=self._on_open,
        )
        self._ws_client.user_data(listen_key=self.listen_key)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._connect()

    def stop(self) -> None:
        self._running = False
        if self._ws_client:
            self._ws_client.stop()
