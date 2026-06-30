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
        # Monotonic timestamp of the last inbound frame (data OR ping/pong).
        # Binance pushes a PING every ~3 min, so a healthy stream refreshes
        # this even when no orders fill. The executor's watchdog uses it to
        # detect a silently-dead stream (connected but no frames). 0.0 = never
        # seen a frame yet -> treated as infinitely stale by seconds_since_activity.
        self._last_activity: float = 0.0

    def _mark_activity(self) -> None:
        self._last_activity = time.monotonic()

    def seconds_since_activity(self) -> float:
        """Seconds since the last inbound frame; inf if none seen yet."""
        if self._last_activity == 0.0:
            return float("inf")
        return time.monotonic() - self._last_activity

    def _on_message(self, _, message: str) -> None:
        """Handle message."""
        self._mark_activity()
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

    def _on_ping(self, _, data) -> None:
        # Binance sends a PING every ~3 min; the library auto-replies PONG.
        # We only stamp it as our liveness heartbeat.
        self._mark_activity()

    def _on_pong(self, _) -> None:
        self._mark_activity()

    def _on_open(self, _) -> None:
        self._mark_activity()
        self.logger.info("User data stream connected")

    def _connect(self) -> None:
        # Connect directly to the user-data stream path. Binance starts pushing
        # account events immediately on connect — no SUBSCRIBE needed.
        #
        # The previous approach connected to the bare host and then called
        # user_data() (which sends {"method":"SUBSCRIBE","params":[listenKey]}).
        # That stopped delivering events around 2026-04-24: the socket connected,
        # the SUBSCRIBE was ACKed, ping/pong stayed alive, but no ORDER_TRADE_UPDATE
        # frames ever arrived. The listenKey-in-path form is the documented way
        # and reconnects naturally pick up an updated key via self.listen_key.
        stream_url = f"{self.ws_url}/ws/{self.listen_key}"
        self._ws_client = UMFuturesWebsocketClient(
            stream_url=stream_url,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
            on_open=self._on_open,
            on_ping=self._on_ping,
            on_pong=self._on_pong,
        )

    def update_listen_key(self, listen_key: str) -> None:
        """Swap in a new listen key and reconnect with it in the stream URL.

        Called by the executor when it regenerates the listen key (e.g. after a
        renewal failure). Without this the feed keeps connecting with the stale
        key and stays silent — no ORDER_TRADE_UPDATE arrives. _connect() builds
        the URL from self.listen_key, so the reconnect picks up the new key.
        """
        if not listen_key or listen_key == self.listen_key:
            return
        self.listen_key = listen_key
        self.logger.info("Listen key updated — reconnecting user data stream")
        self._reconnect()

    def force_reconnect(self, listen_key: Optional[str] = None) -> None:
        """Force a reconnect, optionally swapping in a fresh listen key.

        Used by the executor's staleness watchdog. The stream can stay
        connected (ping/pong alive) yet stop delivering ORDER_TRADE_UPDATE
        frames; _on_close/_on_error never fire in that case, so nothing else
        triggers recovery. Unlike update_listen_key this reconnects even when
        the key is unchanged (Binance returns the same key while it's valid).
        Runs on a daemon thread so the caller (main loop) isn't blocked by
        _reconnect's backoff sleep.
        """
        if not self._running:
            return
        if listen_key:
            self.listen_key = listen_key
        threading.Thread(target=self._reconnect, daemon=True).start()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._connect()

    def stop(self) -> None:
        self._running = False
        if self._ws_client:
            self._ws_client.stop()
