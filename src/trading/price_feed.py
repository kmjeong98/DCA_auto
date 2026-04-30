"""User Data Stream (order/position updates) via WebSocket.

Mark prices are fetched via REST polling in the main loop — see
TradingExecutor.run() — so no mark-price WebSocket is used here.
"""

import json
import threading
import time
from typing import Callable, Dict, Optional

from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient

from src.common.logger import setup_logger


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
        self._reconnect_lock = threading.Lock()

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

    def _reconnect(self) -> None:
        if not self._reconnect_lock.acquire(blocking=False):
            return
        try:
            if not self._running:
                return
            time.sleep(5)
            try:
                if self._ws_client:
                    self._ws_client.stop()
            except Exception:
                pass
            self._connect()
        finally:
            self._reconnect_lock.release()

    def _on_close(self, _) -> None:
        self.logger.warning("WebSocket closed")
        self._reconnect()

    def _on_error(self, _, error) -> None:
        self.logger.error(f"WebSocket error: {error}")
        self._reconnect()

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
