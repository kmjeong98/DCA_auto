"""Offline recovery: reconstruct missed TP/SL/DCA/ENTRY events from exchange history.

Called on restart (after `StateManager.load_state` but before `_sync_with_exchange`)
to fetch trade history since the bot was last alive, classify each filled order as
ENTRY/DCA/TP/SL, and return a time-sorted event list that the caller replays to
bring trade logs and runtime state (especially `last_sl_time` cooldown) up to date.

Pagination/grouping pattern adapted from fix_trade_logs.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from src.common.api_client import APIClient
from src.trading.strategy import PositionState


@dataclass
class ReconEvent:
    """A single reconstructed fill event to replay."""
    kind: str          # "ENTRY" | "DCA" | "TP" | "SL"
    side: str          # "long" | "short"
    price: float
    qty: float
    pnl: float
    timestamp: datetime
    order_id: str


class OfflineRecovery:
    """Fetch and classify offline fills for a single symbol."""

    MAX_LOOKBACK_DAYS = 30
    WINDOW_MS = 7 * 24 * 60 * 60 * 1000  # 7-day fetch windows

    def __init__(self, api: APIClient, symbol: str, logger) -> None:
        self.api = api
        self.symbol = symbol
        self.logger = logger

    def fetch_trades_since(self, start_ms: int) -> List[Dict[str, Any]]:
        """Fetch trades from start_ms to now using 7-day sliding windows.

        Caps total lookback at MAX_LOOKBACK_DAYS to keep API usage bounded on
        long-offline restarts.
        """
        now_ms = int(time.time() * 1000)
        min_allowed = now_ms - self.MAX_LOOKBACK_DAYS * 24 * 60 * 60 * 1000
        if start_ms < min_allowed:
            self.logger.warning(
                f"Offline recovery capped at {self.MAX_LOOKBACK_DAYS}d "
                f"(requested window was longer)"
            )
            start_ms = min_allowed

        all_trades: List[Dict[str, Any]] = []
        seen_ids: Set[int] = set()

        cursor = start_ms
        while cursor < now_ms:
            end = min(cursor + self.WINDOW_MS, now_ms)

            trades = self.api.get_account_trades(
                self.symbol, start_time=cursor, end_time=end, limit=1000,
            )
            for t in trades:
                tid = t["id"]
                if tid not in seen_ids:
                    seen_ids.add(tid)
                    all_trades.append(t)

            # Paginate within this window when hitting the 1000-trade cap
            while len(trades) == 1000:
                last_id = trades[-1]["id"]
                trades = self.api.get_account_trades(
                    self.symbol, from_id=last_id, limit=1000,
                )
                trades = [
                    t for t in trades
                    if t["id"] != last_id and t["time"] <= end
                ]
                for t in trades:
                    tid = t["id"]
                    if tid not in seen_ids:
                        seen_ids.add(tid)
                        all_trades.append(t)
                time.sleep(0.05)

            cursor = end + 1
            time.sleep(0.1)

        all_trades.sort(key=lambda t: t["time"])
        return all_trades

    def group_by_order(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Group trades by orderId, summing qty and realizedPnl.

        Uses the weighted average fill price (sum(price*qty)/sum(qty)) rather
        than the first trade's price — for market orders that fill across
        multiple price levels this matters.
        """
        grouped: Dict[str, Dict[str, Any]] = {}
        for t in trades:
            oid = str(t["orderId"])
            qty = float(t["qty"])
            price = float(t["price"])
            if oid not in grouped:
                grouped[oid] = {
                    "order_id": oid,
                    "symbol": t["symbol"],
                    "side": t["side"],              # BUY / SELL
                    "positionSide": t["positionSide"],  # LONG / SHORT
                    "qty": 0.0,
                    "notional": 0.0,
                    "pnl": 0.0,
                    "time": t["time"],              # earliest fill time
                }
            g = grouped[oid]
            g["qty"] += qty
            g["notional"] += price * qty
            g["pnl"] += float(t.get("realizedPnl", 0) or 0)
            if t["time"] < g["time"]:
                g["time"] = t["time"]

        for g in grouped.values():
            g["price"] = g["notional"] / g["qty"] if g["qty"] > 0 else 0.0

        return grouped

    @staticmethod
    def _is_opening(order: Dict[str, Any]) -> Optional[str]:
        """Return 'long'/'short' if order is a position-increasing fill, else None.

        Mirrors SymbolTrader._is_opening_order:
          BUY + LONG  → long entry/DCA
          SELL + SHORT → short entry/DCA
        """
        pos_side = order.get("positionSide", "")
        order_side = order.get("side", "")
        if pos_side == "LONG" and order_side == "BUY":
            return "long"
        if pos_side == "SHORT" and order_side == "SELL":
            return "short"
        return None

    @staticmethod
    def _closing_side(order: Dict[str, Any]) -> Optional[str]:
        """Return 'long'/'short' if order closes a position, else None."""
        pos_side = order.get("positionSide", "")
        order_side = order.get("side", "")
        if pos_side == "LONG" and order_side == "SELL":
            return "long"
        if pos_side == "SHORT" and order_side == "BUY":
            return "short"
        return None

    def classify_events(
        self,
        orders_by_id: Dict[str, Dict[str, Any]],
        long_state: PositionState,
        short_state: PositionState,
        seen_order_ids: Set[str],
    ) -> List[ReconEvent]:
        """Classify each order into ENTRY/DCA/TP/SL and return time-sorted events.

        Matching priority:
          1. orderId == state.tp_order_id  → TP (authoritative)
          2. orderId == state.sl_order_id  → SL (authoritative)
          3. opening order (BUY+LONG / SELL+SHORT) → ENTRY if no prior position
             for that side in our replay timeline, else DCA
          4. closing order not matching tp/sl id → TP if pnl >= 0 else SL
             (best-effort fallback for orders cancelled/re-placed while offline)

        Orders in `seen_order_ids` are skipped (already logged).
        """
        # Snapshot tp/sl ids from saved state before any mutation
        tp_ids = {
            "long": long_state.tp_order_id,
            "short": short_state.tp_order_id,
        }
        sl_ids = {
            "long": long_state.sl_order_id,
            "short": short_state.sl_order_id,
        }

        # Track whether each side currently has a reconstructed position in timeline.
        # Seed from saved state: if state.active and amount>0, a position existed
        # before the offline window, so the first opening order of that side is DCA.
        active_in_timeline = {
            "long": long_state.active and long_state.amount > 0,
            "short": short_state.active and short_state.amount > 0,
        }

        # Sort by time before walking so state transitions are in order
        ordered = sorted(orders_by_id.values(), key=lambda o: o["time"])

        events: List[ReconEvent] = []
        for order in ordered:
            oid = order["order_id"]
            if oid in seen_order_ids:
                continue
            if order["qty"] <= 0:
                continue

            ts = datetime.fromtimestamp(order["time"] / 1000.0, tz=timezone.utc)

            # 1. TP id match
            matched = False
            for side in ("long", "short"):
                if tp_ids[side] and oid == tp_ids[side]:
                    events.append(ReconEvent(
                        kind="TP", side=side,
                        price=order["price"], qty=order["qty"],
                        pnl=order["pnl"], timestamp=ts, order_id=oid,
                    ))
                    active_in_timeline[side] = False
                    matched = True
                    break
            if matched:
                continue

            # 2. SL id match
            for side in ("long", "short"):
                if sl_ids[side] and oid == sl_ids[side]:
                    events.append(ReconEvent(
                        kind="SL", side=side,
                        price=order["price"], qty=order["qty"],
                        pnl=order["pnl"], timestamp=ts, order_id=oid,
                    ))
                    active_in_timeline[side] = False
                    matched = True
                    break
            if matched:
                continue

            # 3. Opening order → ENTRY or DCA
            open_side = self._is_opening(order)
            if open_side is not None:
                if active_in_timeline[open_side]:
                    kind = "DCA"
                else:
                    kind = "ENTRY"
                    active_in_timeline[open_side] = True
                events.append(ReconEvent(
                    kind=kind, side=open_side,
                    price=order["price"], qty=order["qty"],
                    pnl=order["pnl"], timestamp=ts, order_id=oid,
                ))
                continue

            # 4. Closing order without id match → fallback TP/SL by pnl sign
            close_side = self._closing_side(order)
            if close_side is not None:
                kind = "TP" if order["pnl"] >= 0 else "SL"
                events.append(ReconEvent(
                    kind=kind, side=close_side,
                    price=order["price"], qty=order["qty"],
                    pnl=order["pnl"], timestamp=ts, order_id=oid,
                ))
                active_in_timeline[close_side] = False
                self.logger.warning(
                    f"Fallback-classified {close_side.upper()} close order {oid} "
                    f"as {kind} by pnl sign (pnl={order['pnl']:.4f})"
                )
                continue

            # Unrecognized — log and skip
            self.logger.warning(
                f"Unclassifiable order {oid}: side={order['side']}, "
                f"positionSide={order['positionSide']}"
            )

        return events
