"""Order execution and balance management."""

import json
import os
import signal
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.common.api_client import APIClient
from src.common.config_loader import ConfigLoader
from src.common.logger import setup_logger
from src.common.trading_config import TradingConfig
from src.trading.bnb_manager import BnbManager
from src.trading.margin_manager import MarginManager
from src.trading.offline_recovery import OfflineRecovery, ReconEvent
from src.trading.price_feed import PriceFeed, OrderUpdateFeed
from src.trading.state_manager import StateManager, TradeLogger
from src.trading.strategy import DCAStrategy, DCALevel, PositionState


class SymbolTrader:
    """Per-symbol trading manager."""

    def __init__(
        self,
        symbol: str,
        params: Dict[str, Any],
        api_client: APIClient,
        capital: float,
        weight: float,
        margin_manager: MarginManager,
        config_loader: ConfigLoader,
        safe_name: str,
        cooldown_hours: int = 6,
        config_path: str = "config/config.json",
    ) -> None:
        self.symbol = symbol
        self.api = api_client
        self.capital = capital
        self.weight = weight
        self.margin_manager = margin_manager
        self.config_loader = config_loader
        self._params_safe_name = safe_name
        self.cooldown_hours = cooldown_hours
        self._weight_changed = False
        self._config_path = config_path

        # Initialize strategy
        self.strategy = DCAStrategy(params)

        # Position state
        self.long_state = PositionState(side="long")
        self.short_state = PositionState(side="short")

        # Logging
        safe_symbol = symbol.replace("/", "_")
        self.logger = setup_logger(f"trader_{safe_symbol}", f"data/logs/trader_{safe_symbol}.log")
        self.trade_logger = TradeLogger()

        # State management
        self.state_manager = StateManager()

        # Current price
        self._current_price: float = 0.0
        self._lock = threading.Lock()

        # Balance snapshot (for PnL calculation)
        self._last_equity: float = 0.0
        self._cached_equity: float = 0.0

        # active_params directory
        self._active_params_dir = Path("data/active_params")

        # Initialization complete flag
        self._initialized = False

        # Pending removal flag (symbol removed from config)
        self._marked_for_removal = False

        # Pending order retry list
        self._pending_sl: Set[str] = set()   # "long" / "short"
        self._pending_tp: Set[str] = set()   # "long" / "short"
        self._pending_dca: List[tuple] = []  # [(side, DCALevel), ...]

    def initialize(self) -> None:
        """Configure exchange and sync initial positions."""
        self.logger.info(f"Initializing {self.symbol}...")

        # Power recovery: use active_params file if available
        active_params = self._load_active_params()
        if active_params:
            self.strategy = DCAStrategy(active_params)
            self.logger.info("Loaded params from active_params (recovery)")
        else:
            # First start: save current params to active_params
            self._save_active_params(self.strategy.params)
            self.logger.info("Saved initial params to active_params")

        # Set cross margin mode
        try:
            self.api.set_margin_mode(self.symbol, "cross")
        except Exception as e:
            self.logger.warning(f"Margin mode setting: {e}")

        # Set leverage
        lev = self.strategy.leverage
        try:
            self.api.set_leverage(self.symbol, lev)
        except Exception as e:
            self.logger.warning(f"Leverage setting: {e}")

        # Load saved state
        saved_state = self.state_manager.load_state(self.symbol)
        if saved_state:
            self.long_state = saved_state["long"]
            self.short_state = saved_state["short"]
            self.logger.info("Loaded saved state")

        # Fetch equity before offline recovery — _handle_tp may call
        # _try_update_margin during replay, which needs _cached_equity.
        try:
            equity = self.api.get_account_equity()
            self._last_equity = equity
            self._cached_equity = equity
        except Exception as e:
            self.logger.warning(f"Initial equity snapshot failed: {e}")

        # Reconstruct offline events (TP/SL/DCA fills that occurred while down).
        # Must run BEFORE _sync_with_exchange — needs saved tp/sl order_ids
        # intact to classify events. Safe no-op when no saved state exists.
        if saved_state:
            try:
                self._reconstruct_offline_events(saved_state.get("updated_at"))
            except Exception as e:
                self.logger.error(f"Offline recovery failed, continuing: {e}")

        # Sync with exchange positions
        self._sync_with_exchange()

        # Before price feed starts, fetch current price via REST (for monitor display)
        if self._current_price <= 0:
            try:
                self._current_price = self.api.get_mark_price(self.symbol)
            except Exception:
                pass

        # Immediately reflect sync result in state file (for monitor display)
        self._save_state()

        self._initialized = True
        self.logger.info(
            f"Initialized {self.symbol} - Capital: ${self.capital:.2f}, "
            f"Leverage: {lev}x"
        )

    # ── Offline recovery ──

    _RECOVERY_SLACK_MS = 5 * 60 * 1000  # 5-minute slack for clock skew / save-lag
    _TRADE_LOG_DIR = Path("data/logs/trades")

    def _recovery_start_ms(self, saved_updated_at: Optional[str]) -> int:
        """Lower bound of exchange-history fetch window.

        Picks max(saved_state.updated_at, latest_trade_log_ts) minus a 5-min
        slack. Falls back to MAX_LOOKBACK_DAYS ago when neither is available.
        """
        candidates: List[int] = []

        if saved_updated_at:
            try:
                dt = datetime.fromisoformat(saved_updated_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                candidates.append(int(dt.timestamp() * 1000))
            except Exception:
                pass

        last_log_ts = self._latest_trade_log_timestamp()
        if last_log_ts is not None:
            candidates.append(last_log_ts)

        now_ms = int(time.time() * 1000)
        if not candidates:
            return now_ms - OfflineRecovery.MAX_LOOKBACK_DAYS * 24 * 60 * 60 * 1000
        return max(candidates) - self._RECOVERY_SLACK_MS

    def _recent_month_log_paths(self) -> List[Path]:
        """Current + previous month JSONL paths for this symbol (for dedup/scan)."""
        safe = self.symbol.replace("/", "_")
        now = datetime.now(timezone.utc)
        prev = now.replace(day=1) - timedelta(days=1)
        months = [now.strftime("%Y%m"), prev.strftime("%Y%m")]
        return [self._TRADE_LOG_DIR / f"{safe}_{m}.jsonl" for m in months]

    def _latest_trade_log_timestamp(self) -> Optional[int]:
        """Scan this symbol's current + previous month JSONL for the newest event."""
        latest: Optional[int] = None
        for path in self._recent_month_log_paths():
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        ts = datetime.fromisoformat(rec["timestamp"])
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=timezone.utc)
                        ms = int(ts.timestamp() * 1000)
                        if latest is None or ms > latest:
                            latest = ms
            except Exception as e:
                self.logger.warning(f"Trade log scan error for {path.name}: {e}")
        return latest

    def _seen_order_ids(self) -> Set[str]:
        """Collect order_ids already recorded in the last two months of trade logs."""
        seen: Set[str] = set()
        for path in self._recent_month_log_paths():
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        oid = rec.get("order_id")
                        if oid:
                            seen.add(str(oid))
            except Exception as e:
                self.logger.warning(f"seen_order_ids scan error for {path.name}: {e}")
        return seen

    def _apply_offline_dca(self, event: ReconEvent) -> None:
        """Replay a DCA fill detected during offline recovery.

        Mirrors _process_dca_fill but (a) uses the reconstructed fill price/qty
        directly, (b) skips TP re-placement — the subsequent _reconcile_orders
        will handle TP re-placement with the final avg_price, and (c) writes
        the trade log with the exchange fill timestamp.
        """
        state = self.long_state if event.side == "long" else self.short_state

        # If saved state has no prior position, treat as ENTRY-equivalent seed
        # (rare — happens if the saved state was stale/missing entry info).
        if not state.active or state.amount <= 0:
            state.active = True
            state.amount = event.qty
            state.cost = event.price * event.qty
            state.avg_price = event.price
            if state.base_price == 0:
                state.base_price = event.price
            state.dca_count = 0
            self.trade_logger.log_entry(
                self.symbol, event.side, event.price, event.qty,
                event.price * event.qty / max(self.strategy.get_leverage(event.side), 1),
                order_id=event.order_id,
                timestamp=event.timestamp,
            )
            self.logger.info(
                f"[offline-recovery] Seeded {event.side.upper()} from ENTRY-like "
                f"fill {event.order_id} @ {event.price:.4f} x {event.qty:.4f}"
            )
            return

        new_amount, new_cost, new_avg = self.strategy.calculate_avg_price(
            state.amount, state.cost, event.qty, event.price * event.qty,
        )
        state.amount = new_amount
        state.cost = new_cost
        state.avg_price = new_avg
        state.dca_count += 1

        level = state.dca_count
        leverage = self.strategy.get_leverage(event.side)
        margin = event.price * event.qty / leverage if leverage > 0 else 0.0

        self.logger.info(
            f"[offline-recovery] DCA{level} {event.side.upper()} - "
            f"Price: {event.price:.4f}, Qty: {event.qty:.4f}, "
            f"New Avg: {new_avg:.4f}, Total: {new_amount:.4f}"
        )
        self.trade_logger.log_dca(
            self.symbol, event.side, level,
            event.price, event.qty, margin, new_avg,
            order_id=event.order_id,
            timestamp=event.timestamp,
        )

    def _reconstruct_offline_events(self, saved_updated_at: Optional[str]) -> None:
        """Fetch exchange trade history since last-alive time and replay missed fills.

        Flow:
          1. Compute start_ms from saved_updated_at + latest trade-log timestamp.
          2. Fetch user trades, group by orderId.
          3. Classify into ReconEvents (TP/SL by tp_order_id/sl_order_id match,
             otherwise by positionSide/side).
          4. Skip events whose order_id is already in the trade log (dedup).
          5. Replay each event via _handle_tp / _handle_sl / _apply_offline_dca.
        """
        start_ms = self._recovery_start_ms(saved_updated_at)
        seen_ids = self._seen_order_ids()

        recovery = OfflineRecovery(self.api, self.symbol, self.logger)
        trades = recovery.fetch_trades_since(start_ms)
        if not trades:
            self.logger.info("[offline-recovery] No trades found in window")
            return

        orders = recovery.group_by_order(trades)
        events = recovery.classify_events(
            orders, self.long_state, self.short_state, seen_ids,
        )
        if not events:
            self.logger.info(
                f"[offline-recovery] {len(orders)} orders fetched, "
                f"0 new events after dedup"
            )
            return

        self.logger.info(
            f"[offline-recovery] Replaying {len(events)} missed event(s) "
            f"from {len(orders)} order(s)"
        )

        for ev in events:
            try:
                if ev.kind == "ENTRY":
                    # Only fires when saved state was missing the original entry.
                    # Just log it — state will reconcile via _sync_with_exchange.
                    leverage = self.strategy.get_leverage(ev.side)
                    margin = ev.price * ev.qty / leverage if leverage > 0 else 0.0
                    self.trade_logger.log_entry(
                        self.symbol, ev.side, ev.price, ev.qty, margin,
                        order_id=ev.order_id,
                        timestamp=ev.timestamp,
                    )
                elif ev.kind == "DCA":
                    self._apply_offline_dca(ev)
                elif ev.kind == "TP":
                    self._handle_tp(
                        ev.side,
                        fill_data={"avg_price": ev.price, "pnl": ev.pnl},
                        log_timestamp=ev.timestamp,
                    )
                elif ev.kind == "SL":
                    self._handle_sl(
                        ev.side,
                        fill_data={"avg_price": ev.price, "pnl": ev.pnl},
                        log_timestamp=ev.timestamp,
                    )
            except Exception as e:
                self.logger.error(
                    f"[offline-recovery] Failed to replay {ev.kind} {ev.side} "
                    f"{ev.order_id}: {e}"
                )

    def _sync_with_exchange(self) -> None:
        """Sync local state with exchange positions."""
        try:
            positions = self.api.get_positions(self.symbol)

            # Check exchange positions
            exchange_long_amt = 0.0
            exchange_short_amt = 0.0

            for pos in positions:
                side = pos.get("side", "").lower()
                amount = float(pos.get("contracts", 0))
                entry_price = float(pos.get("entryPrice", 0))

                if side == "long" and amount > 0:
                    exchange_long_amt = amount
                    self.long_state.active = True
                    self.long_state.amount = amount
                    if entry_price > 0:
                        self.long_state.avg_price = entry_price
                        self.long_state.cost = entry_price * amount
                        if self.long_state.base_price == 0:
                            self.long_state.base_price = entry_price

                elif side == "short" and amount > 0:
                    exchange_short_amt = amount
                    self.short_state.active = True
                    self.short_state.amount = amount
                    if entry_price > 0:
                        self.short_state.avg_price = entry_price
                        self.short_state.cost = entry_price * amount
                        if self.short_state.base_price == 0:
                            self.short_state.base_price = entry_price

            # Reset if local state is active but no exchange position exists
            if exchange_long_amt == 0.0 and self.long_state.active:
                self.logger.warning("Long state was active but no exchange position — resetting")
                sl_time = self.long_state.last_sl_time
                self.long_state.reset()
                self.long_state.last_sl_time = sl_time  # preserve cooldown

            if exchange_short_amt == 0.0 and self.short_state.active:
                self.logger.warning("Short state was active but no exchange position — resetting")
                sl_time = self.short_state.last_sl_time
                self.short_state.reset()
                self.short_state.last_sl_time = sl_time  # preserve cooldown

            self.logger.info(
                f"Synced - Long: {self.long_state.amount:.4f}, "
                f"Short: {self.short_state.amount:.4f}"
            )

            # ── Reconcile order IDs (clean up fills that occurred while offline) ──
            self._reconcile_orders()

        except Exception as e:
            self.logger.error(f"Sync error: {e}")

    def periodic_sync(self) -> None:
        """Periodically check for state/exchange mismatches and fix them.

        Detects cases where local state says a position is active but
        the exchange has no position (e.g. TP/SL filled while stream was down).
        Checks TP/SL order status to determine which event occurred.
        """
        try:
            positions = self.api.get_positions(self.symbol)
            exchange_sides = set()
            for pos in positions:
                side = pos.get("side", "").lower()
                amount = float(pos.get("contracts", 0))
                if amount > 0:
                    exchange_sides.add(side)

            for side in ("long", "short"):
                state = self.long_state if side == "long" else self.short_state
                if state.active and side not in exchange_sides:
                    # Check TP order status to distinguish TP vs SL
                    was_tp = False
                    if state.tp_order_id:
                        try:
                            tp_order = self.api.get_order(self.symbol, state.tp_order_id)
                            if tp_order.get("status") == "FILLED":
                                was_tp = True
                        except Exception:
                            pass

                    if was_tp:
                        self.logger.warning(
                            f"[periodic_sync] {side.upper()} closed by TP — re-entering"
                        )
                        self._handle_tp(side)
                    else:
                        self.logger.warning(
                            f"[periodic_sync] {side.upper()} closed by SL — cooldown"
                        )
                        self._handle_sl(side)

            # Clean up orphan algo orders (e.g. SL cancel failed during DCA fill)
            tracked_algo_ids: set = set()
            for side in ("long", "short"):
                state = self.long_state if side == "long" else self.short_state
                if state.sl_order_id:
                    tracked_algo_ids.add(state.sl_order_id)
            try:
                algo_orders = self.api.get_open_algo_orders(self.symbol)
                for ao in algo_orders:
                    ao_id = str(ao["id"])
                    if ao_id not in tracked_algo_ids:
                        self.logger.warning(
                            f"[periodic_sync] Cancelling orphan algo order {ao_id}"
                        )
                        try:
                            self.api.cancel_algo_order(self.symbol, ao_id)
                        except Exception as e:
                            self.logger.error(
                                f"[periodic_sync] Cancel orphan algo order error: {e}"
                            )
            except Exception as e:
                self.logger.warning(f"[periodic_sync] Fetch algo orders error: {e}")

            # Missing SL detection: active position without an SL order_id.
            # Happens when a previous SL placement failed (e.g. -2021) and the
            # pending_sl retry queue was lost on restart, or the exchange SL
            # was cancelled externally. Re-place; if price already breached
            # the SL level, _place_sl_order will force a market close.
            for side in ("long", "short"):
                state = self.long_state if side == "long" else self.short_state
                if state.active and state.amount > 0 and not state.sl_order_id:
                    self.logger.warning(
                        f"[periodic_sync] {side.upper()} active but no SL order "
                        f"— placing"
                    )
                    self._place_sl_order(side)

        except Exception as e:
            self.logger.error(f"Periodic sync error: {e}")

    def _is_opening_order(self, order: Dict[str, Any], side: str) -> bool:
        """Determine if an order is a position-increasing (DCA) order.

        In Hedge Mode:
          LONG position: BUY + LONG = entry/DCA, SELL + LONG = close
          SHORT position: SELL + SHORT = entry/DCA, BUY + SHORT = close
        """
        pos_side = order.get("positionSide", "")
        order_side = order.get("side", "")

        if side == "long":
            return pos_side == "LONG" and order_side == "buy"
        else:
            return pos_side == "SHORT" and order_side == "sell"

    def _reconcile_orders(self) -> None:
        """Full reconciliation between exchange open orders and local state.

        1. Remove DCA order IDs from state that were filled while offline
        2. Classify orphaned orders: DCA (position-increasing) adopted into state, TP/SL (closing) cancelled
        3. Cancel SL/TP and re-place based on current position
        4. Correct dca_count to max_dca - remaining DCA count
        """
        try:
            open_orders = self.api.get_open_orders(self.symbol)
            open_ids = {str(o["id"]) for o in open_orders}
        except Exception as e:
            self.logger.error(f"Reconcile orders error: {e}")
            return

        # Fetch algo orders (SL)
        algo_open_ids: set = set()
        try:
            algo_orders = self.api.get_open_algo_orders(self.symbol)
            algo_open_ids = {str(o["id"]) for o in algo_orders}
        except Exception as e:
            self.logger.warning(f"Reconcile algo orders error: {e}")

        # ── Step 1: Clean up stale order IDs per side ──
        tracked_ids: set = set()

        for side in ["long", "short"]:
            state = self.long_state if side == "long" else self.short_state

            if not state.active:
                # No position — all orders are invalid
                if state.dca_orders or state.sl_order_id or state.tp_order_id:
                    self.logger.info(f"[{side}] No position — clearing stale order IDs")
                    state.dca_orders = []
                    state.sl_order_id = None
                    state.tp_order_id = None
                continue

            # DCA: order not on exchange = filled → remove from list (no recalculation)
            before = len(state.dca_orders)
            state.dca_orders = [
                d for d in state.dca_orders
                if d.order_id and d.order_id in open_ids
            ]
            removed = before - len(state.dca_orders)
            if removed > 0:
                state.dca_count += removed
                self.logger.info(
                    f"[{side}] Reconciled {removed} DCA fills "
                    f"(dca_count → {state.dca_count})"
                )

            # Collect tracked IDs
            for dca in state.dca_orders:
                if dca.order_id:
                    tracked_ids.add(dca.order_id)

            # SL: managed as algo order — check in algo_open_ids
            if state.sl_order_id and state.sl_order_id not in algo_open_ids:
                self.logger.info(f"[{side}] SL algo order {state.sl_order_id} no longer open")
                state.sl_order_id = None

            # TP: not on exchange means filled or expired → clear ID
            if state.tp_order_id and state.tp_order_id not in open_ids:
                self.logger.info(f"[{side}] TP order {state.tp_order_id} no longer open")
                state.tp_order_id = None
            if state.tp_order_id:
                tracked_ids.add(state.tp_order_id)

        # ── Step 2: Classify orphaned orders — DCA adopted, TP/SL cancelled ──
        for order in open_orders:
            order_id = str(order["id"])
            if order_id in tracked_ids:
                continue

            # Determine which side this belongs to
            adopted = False
            for side in ["long", "short"]:
                state = self.long_state if side == "long" else self.short_state
                if not state.active:
                    continue

                if self._is_opening_order(order, side):
                    # DCA order → adopt into state
                    price = float(order.get("price", 0))
                    amount = float(order.get("amount", 0))
                    leverage = self.strategy.get_leverage(side)
                    margin = amount * price / leverage if leverage > 0 else 0

                    level = len(state.dca_orders) + 1
                    state.dca_orders.append(DCALevel(
                        level=level,
                        trigger_price=price,
                        margin=margin,
                        order_id=order_id,
                    ))
                    tracked_ids.add(order_id)
                    adopted = True
                    self.logger.info(
                        f"[{side}] Adopted orphaned DCA order {order_id} "
                        f"(price={price:.2f}, qty={amount})"
                    )
                    break

            if not adopted:
                # Closing order or unmatched → cancel
                self.logger.warning(
                    f"Cancelling orphaned order {order_id} "
                    f"({order.get('type')} {order.get('side')} "
                    f"qty={order.get('amount')})"
                )
                try:
                    self.api.cancel_order(self.symbol, order_id)
                except Exception as e:
                    self.logger.error(f"Cancel orphaned order error: {e}")

        # ── Step 2b: Cancel orphaned algo orders (duplicate SL) ──
        tracked_algo_ids: set = set()
        for side in ["long", "short"]:
            state = self.long_state if side == "long" else self.short_state
            if state.sl_order_id:
                tracked_algo_ids.add(state.sl_order_id)
        try:
            for ao in algo_orders:
                ao_id = str(ao["id"])
                if ao_id not in tracked_algo_ids:
                    self.logger.warning(f"Cancelling orphaned algo order {ao_id}")
                    try:
                        self.api.cancel_algo_order(self.symbol, ao_id)
                    except Exception as e:
                        self.logger.error(f"Cancel orphaned algo order error: {e}")
        except Exception:
            pass

        # ── Step 3: Re-place SL/TP for active positions + correct dca_count ──
        for side in ["long", "short"]:
            state = self.long_state if side == "long" else self.short_state
            if not state.active or state.amount <= 0:
                continue

            # Correct dca_count: max_dca - remaining DCA count
            params = self.strategy.get_side_params(side)
            max_dca = int(params.get("max_dca", 0))
            remaining = len(state.dca_orders)
            inferred_count = max_dca - remaining
            if inferred_count > state.dca_count:
                self.logger.info(
                    f"[{side}] Inferred dca_count: {state.dca_count} → {inferred_count} "
                    f"(max_dca={max_dca}, remaining={remaining})"
                )
                state.dca_count = inferred_count

            # Re-place SL (Algo Order)
            if state.sl_order_id:
                try:
                    self.api.cancel_algo_order(self.symbol, state.sl_order_id)
                except Exception:
                    pass
                state.sl_order_id = None
            self._place_sl_order(side)

            # Re-place TP
            if state.tp_order_id:
                try:
                    self.api.cancel_order(self.symbol, state.tp_order_id)
                except Exception:
                    pass
                state.tp_order_id = None
            self._place_tp_order(side)

    def _save_state(self) -> None:
        """Save current state."""
        try:
            # Pending retry info
            pending = []
            for side in self._pending_sl:
                pending.append(f"SL {side.upper()}")
            for side in self._pending_tp:
                pending.append(f"TP {side.upper()}")
            for side, dca in self._pending_dca:
                pending.append(f"DCA{dca.level} {side.upper()}")

            self.state_manager.save_state(
                self.symbol,
                self.long_state,
                self.short_state,
                {
                    "capital": self.capital,
                    "current_price": self._current_price,
                    "pending_retries": pending,
                },
            )
        except Exception as e:
            self.logger.error(f"Save state error: {e}")

    def _reload_weight_from_config(self) -> None:
        """Re-read and apply current symbol's weight from config.json."""
        try:
            cfg_data = json.loads(Path(self._config_path).read_text(encoding="utf-8"))
            symbols_cfg = cfg_data.get("symbols", {})
            if self._params_safe_name in symbols_cfg:
                new_weight = float(symbols_cfg[self._params_safe_name].get("weight", self.weight))
                if new_weight != self.weight:
                    self.logger.info(
                        f"Weight updated from config: {self.weight:.4f} → {new_weight:.4f}"
                    )
                    self.weight = new_weight
                    self._weight_changed = True
        except Exception as e:
            self.logger.warning(f"Config weight reload failed: {e}")

    def update_cached_equity(self, equity: float) -> None:
        """Update cached equity from polling loop (no API call here)."""
        self._cached_equity = equity

    def _try_update_margin(self) -> None:
        """Attempt margin + params update.

        Margin: both sides inactive only.
        Params: no active side with dca > 0.
        Called from 5-min poll and from _handle_tp/_handle_sl after reset.
        """
        both_inactive = not self.long_state.active and not self.short_state.active

        self._reload_weight_from_config()

        if self._cached_equity <= 0:
            return

        # Margin update: both sides must be inactive
        if both_inactive:
            try:
                force = self._weight_changed
                self.capital = self.margin_manager.try_update(
                    self.symbol, self.weight, self.capital, self._cached_equity, force=force
                )
                self._weight_changed = False
            except Exception as e:
                self.logger.error(f"Margin update error: {e}")

        # Params update: no active side with DCA > 0
        has_dca = (
            (self.long_state.active and self.long_state.dca_count > 0)
            or (self.short_state.active and self.short_state.dca_count > 0)
        )
        if not has_dca:
            self._try_update_params()

    @staticmethod
    def _trading_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract only trading-relevant parameters (exclude meta/performance)."""
        return {
            "parameters": params.get("parameters"),
            "fixed_settings": params.get("fixed_settings"),
        }

    def _try_update_params(self) -> None:
        """Detect and replace parameter changes when both sides inactive."""
        try:
            new_params = self.config_loader.load(self._params_safe_name)
            if new_params == self.strategy.params:
                return  # Ignore if completely identical

            trading_changed = (
                self._trading_params(new_params)
                != self._trading_params(self.strategy.params)
            )

            # Always refresh even if only meta/performance changed
            self.strategy = DCAStrategy(new_params)
            self._save_active_params(new_params)

            if trading_changed:
                lev = self.strategy.leverage
                try:
                    self.api.set_leverage(self.symbol, lev)
                except Exception:
                    pass
                self.logger.info(
                    f"Params updated for {self.symbol} (leverage: {lev}x)"
                )
            else:
                self.logger.info(
                    f"Params meta/performance refreshed for {self.symbol}"
                )
        except Exception as e:
            self.logger.error(f"Params update error: {e}")

    def _try_hot_update_params(self) -> None:
        """Hot-update parameters when DCA count is 0.

        Skip if both sides inactive (handled by _try_update_params()).
        Skip if any active side has dca_count > 0 (unsafe).
        If all active sides have dca_count == 0, cancel and re-place DCA/TP/SL.
        """
        # Collect active sides
        active_sides = []
        if self.long_state.active:
            active_sides.append("long")
        if self.short_state.active:
            active_sides.append("short")

        # Both sides inactive → handled by existing logic
        if not active_sides:
            return

        # Cannot update if any active side has dca_count > 0
        for side in active_sides:
            state = self.long_state if side == "long" else self.short_state
            if state.dca_count > 0:
                return

        # Load and compare new parameters
        try:
            new_params = self.config_loader.load(self._params_safe_name)
            if new_params == self.strategy.params:
                return  # Ignore if completely identical
        except Exception as e:
            self.logger.error(f"Hot params load error: {e}")
            return

        trading_changed = (
            self._trading_params(new_params)
            != self._trading_params(self.strategy.params)
        )

        # Always refresh even on meta/performance-only changes
        self.strategy = DCAStrategy(new_params)
        self._save_active_params(new_params)

        if not trading_changed:
            self.logger.info(
                f"Params meta/performance refreshed for {self.symbol}"
            )
            return

        # Only re-place orders if trading params changed
        lev = self.strategy.leverage
        try:
            self.api.set_leverage(self.symbol, lev)
        except Exception:
            pass

        for side in active_sides:
            state = self.long_state if side == "long" else self.short_state

            # Cancel DCA orders
            for dca in state.dca_orders:
                if dca.order_id:
                    try:
                        self.api.cancel_order(self.symbol, dca.order_id)
                    except Exception:
                        pass
            state.dca_orders = []

            # Cancel TP
            if state.tp_order_id:
                try:
                    self.api.cancel_order(self.symbol, state.tp_order_id)
                except Exception:
                    pass
                state.tp_order_id = None

            # Cancel SL (Algo Order)
            if state.sl_order_id:
                try:
                    self.api.cancel_algo_order(self.symbol, state.sl_order_id)
                    state.sl_order_id = None
                except Exception as e:
                    self.logger.warning(f"Failed to cancel old SL {state.sl_order_id}: {e}")

            # Re-place with new parameters
            self._place_sl_order(side)
            self._place_tp_order(side)
            self._place_dca_orders(side)

        self._save_state()
        self.logger.info(
            f"Hot params updated for {self.symbol} (leverage: {lev}x, "
            f"sides: {', '.join(s.upper() for s in active_sides)})"
        )

    def _save_active_params(self, params: Dict[str, Any]) -> None:
        """Save currently active parameters to active_params."""
        try:
            self._active_params_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._active_params_dir / f"{self._params_safe_name}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Save active params error: {e}")

    def _load_active_params(self) -> Optional[Dict[str, Any]]:
        """Load parameters from active_params (for power recovery)."""
        file_path = self._active_params_dir / f"{self._params_safe_name}.json"
        if not file_path.exists():
            return None
        try:
            with file_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Load active params error: {e}")
            return None

    def _enter_position(self, side: str, ref_price: float = 0.0) -> bool:
        """Enter base position via market order.

        Args:
            ref_price: Reference price for quantity calculation (e.g. TP price
                       on re-entry). Skips mark-price lookup when provided,
                       shaving off one REST round-trip.
        """
        # Do not enter new positions for symbols pending removal
        if self._marked_for_removal:
            return False

        state = self.long_state if side == "long" else self.short_state

        if state.active:
            return False

        # Cooldown check
        if not self.strategy.should_enter(state, self.cooldown_hours):
            return False

        try:
            # Calculate margin
            margin = self.strategy.calculate_base_margin(self.capital, side)
            leverage = self.strategy.get_leverage(side)

            # Use ref_price (TP re-entry) → _current_price → API fallback
            price = ref_price if ref_price > 0 else self._current_price
            if price <= 0:
                price = self.api.get_mark_price(self.symbol)

            # Calculate quantity
            notional = margin * leverage
            amount = notional / price
            amount = self.api.round_amount(self.symbol, amount)

            min_amount = self.api.get_min_order_amount(self.symbol)
            if amount < min_amount:
                self.logger.warning(f"Amount {amount} < min {min_amount}")
                return False

            # Market order
            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "buy" if side == "long" else "sell"

            order = self.api.place_market_order(
                self.symbol, order_side, amount, position_side,
            )

            fill_price = float(order.get("average", 0))
            filled_amount = float(order.get("filled", 0))
            entry_order_id = str(order.get("id")) if order.get("id") is not None else None

            # Fallback: if no fill info in response, use mark price / order quantity
            if fill_price <= 0:
                fill_price = self.api.get_mark_price(self.symbol)
                self.logger.warning(f"No fill price in response, using mark price: {fill_price}")
            if filled_amount <= 0:
                filled_amount = amount
                self.logger.warning(f"No filled amount in response, using order amount: {filled_amount}")

            # Update state
            state.active = True
            state.amount = filled_amount
            state.cost = fill_price * filled_amount  # notional cost
            state.avg_price = fill_price
            state.base_price = fill_price
            state.dca_count = 0
            state.entry_time = datetime.now(timezone.utc)

            self.logger.info(
                f"ENTRY {side.upper()} - Price: {fill_price:.2f}, "
                f"Amount: {filled_amount:.4f}, Margin: {margin:.2f}"
            )
            self.trade_logger.log_entry(
                self.symbol, side, fill_price, filled_amount, margin,
                order_id=entry_order_id,
            )

            # Place DCA orders
            self._place_dca_orders(side)

            # Place SL order
            self._place_sl_order(side)

            # Place TP order
            self._place_tp_order(side)

            # Update balance snapshot (baseline after entry)
            if self._cached_equity > 0:
                self._last_equity = self._cached_equity

            self._save_state()
            return True

        except Exception as e:
            self.logger.error(f"Entry error ({side}): {e}")
            return False

    def _place_dca_orders(self, side: str) -> None:
        """Place DCA limit orders."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.base_price <= 0:
            return

        # Calculate DCA levels
        dca_levels = self.strategy.calculate_dca_levels(
            state.base_price, side, self.capital
        )

        position_side = "LONG" if side == "long" else "SHORT"
        order_side = "buy" if side == "long" else "sell"

        for dca in dca_levels:
            try:
                leverage = self.strategy.get_leverage(side)
                notional = dca.margin * leverage
                amount = notional / dca.trigger_price
                amount = self.api.round_amount(self.symbol, amount)
                price = self.api.round_price(self.symbol, dca.trigger_price)

                order = self.api.place_limit_order(
                    self.symbol, order_side, amount, price, position_side,
                )

                dca.order_id = str(order.get("id"))
                state.dca_orders.append(dca)

                self.logger.info(
                    f"DCA{dca.level} {side.upper()} placed - "
                    f"Price: {price:.2f}, Amount: {amount:.4f}"
                )

            except Exception as e:
                self.logger.error(f"DCA order error: {e}")
                self._pending_dca.append((side, dca))

    def _place_sl_order(self, side: str) -> None:
        """Place SL order (STOP_MARKET — mark price trigger). Schedule retry on failure.

        If the SL level has already been breached by the current market price
        (Binance error -2021 "Order would immediately trigger"), force a market
        close of the position through `_force_sl_market_close`.
        """
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.amount <= 0:
            self._pending_sl.discard(side)
            return

        sl_price = 0.0
        try:
            sl_price = self.strategy.calculate_sl_price(state.base_price, side)
            sl_price = self.api.round_price(self.symbol, sl_price)

            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "sell" if side == "long" else "buy"

            order = self.api.place_stop_loss(
                self.symbol, order_side, sl_price, position_side,
            )

            state.sl_order_id = str(order.get("id"))
            self._pending_sl.discard(side)

            self.logger.info(
                f"SL {side.upper()} placed - Price: {sl_price:.2f}, "
                f"closePosition=true"
            )

        except Exception as e:
            err_str = str(e)
            # -2021 "Order would immediately trigger" → SL level already breached
            if "-2021" in err_str or "immediately trigger" in err_str.lower():
                self.logger.warning(
                    f"SL price {sl_price:.4f} already breached by mark → "
                    f"force market close {side.upper()}"
                )
                self._pending_sl.discard(side)
                self._force_sl_market_close(side)
            else:
                self.logger.error(f"SL order error ({side}): {e}")
                self._pending_sl.add(side)

    def _force_sl_market_close(self, side: str) -> None:
        """Force market close a position whose SL level is already breached.

        Routes through `_handle_sl` so the trade is logged as an SL event with
        proper cooldown and state cleanup.
        """
        state = self.long_state if side == "long" else self.short_state
        if not state.active or state.amount <= 0:
            return

        position_side = "LONG" if side == "long" else "SHORT"
        order_side = "sell" if side == "long" else "buy"
        amount = self.api.round_amount(self.symbol, state.amount)

        try:
            order = self.api.place_market_order(
                self.symbol, order_side, amount, position_side,
            )
            fill_price = float(order.get("average", 0))
            if fill_price <= 0:
                fill_price = self._current_price
            filled = float(order.get("filled", amount))

            self.logger.info(
                f"Forced SL market close {side.upper()} - "
                f"Price: {fill_price:.4f}, Amount: {filled:.4f}"
            )

            # Route through _handle_sl for consistent logging + cooldown.
            # realized pnl unknown from market order response → let fallback
            # path in _handle_sl estimate via cached equity diff.
            fill_data = {"avg_price": fill_price, "pnl": 0.0}
            self._handle_sl(side, fill_data)
        except Exception as e:
            self.logger.error(f"Force SL market close error ({side}): {e}")

    def has_pending_orders(self) -> bool:
        """Check if there are pending retry orders."""
        return bool(self._pending_sl or self._pending_tp or self._pending_dca)

    def retry_pending_orders(self) -> None:
        """Retry failed SL/TP/DCA orders (called periodically from outside)."""
        changed = False

        # SL retry
        for side in list(self._pending_sl):
            state = self.long_state if side == "long" else self.short_state
            if not state.active or state.amount <= 0 or state.sl_order_id:
                self._pending_sl.discard(side)
                continue
            self.logger.info(f"Retrying SL {side.upper()}...")
            self._place_sl_order(side)
            if side not in self._pending_sl:
                changed = True

        # TP retry
        for side in list(self._pending_tp):
            state = self.long_state if side == "long" else self.short_state
            if not state.active or state.amount <= 0 or state.tp_order_id:
                self._pending_tp.discard(side)
                continue
            self.logger.info(f"Retrying TP {side.upper()}...")
            self._place_tp_order(side)
            if side not in self._pending_tp:
                changed = True

        # DCA retry
        remaining: List[tuple] = []
        for side, dca in self._pending_dca:
            state = self.long_state if side == "long" else self.short_state
            if not state.active or state.base_price <= 0:
                continue
            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "buy" if side == "long" else "sell"
            try:
                leverage = self.strategy.get_leverage(side)
                notional = dca.margin * leverage
                amount = notional / dca.trigger_price
                amount = self.api.round_amount(self.symbol, amount)
                price = self.api.round_price(self.symbol, dca.trigger_price)

                order = self.api.place_limit_order(
                    self.symbol, order_side, amount, price, position_side,
                )
                dca.order_id = str(order.get("id"))
                state.dca_orders.append(dca)
                changed = True
                self.logger.info(
                    f"Retry DCA{dca.level} {side.upper()} placed - "
                    f"Price: {price:.2f}, Amount: {amount:.4f}"
                )
            except Exception as e:
                self.logger.error(f"Retry DCA order error: {e}")
                remaining.append((side, dca))
        self._pending_dca = remaining

        if changed:
            self._save_state()

    def _place_tp_order(self, side: str) -> None:
        """Place TP order (LIMIT — fills at exact price). Schedule retry on failure."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.amount <= 0:
            self._pending_tp.discard(side)
            return

        try:
            tp_price = self.strategy.calculate_tp_price(state.avg_price, side)
            tp_price = self.api.round_price(self.symbol, tp_price)

            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "sell" if side == "long" else "buy"
            amount = self.api.snap_amount(self.symbol, state.amount)

            order = self.api.place_take_profit(
                self.symbol, order_side, amount, tp_price, position_side,
            )

            state.tp_order_id = str(order.get("id"))
            self._pending_tp.discard(side)

            self.logger.info(
                f"TP {side.upper()} placed - Price: {tp_price:.2f}, "
                f"Amount: {amount:.4f}"
            )

        except Exception as e:
            self.logger.error(f"TP order error ({side}): {e}")
            self._pending_tp.add(side)

    def _cancel_side_orders(self, side: str) -> None:
        """Cancel all orders for one side."""
        state = self.long_state if side == "long" else self.short_state

        # Cancel DCA orders
        for dca in state.dca_orders:
            if dca.order_id:
                try:
                    self.api.cancel_order(self.symbol, dca.order_id)
                except Exception:
                    pass
        state.dca_orders = []

        # Cancel SL (Algo Order)
        if state.sl_order_id:
            try:
                self.api.cancel_algo_order(self.symbol, state.sl_order_id)
            except Exception:
                pass
            state.sl_order_id = None

        # Cancel TP
        if state.tp_order_id:
            try:
                self.api.cancel_order(self.symbol, state.tp_order_id)
            except Exception:
                pass
            state.tp_order_id = None

    def _cancel_old_orders(
        self,
        old_dca_orders: List,
        old_sl_order_id: Optional[str],
    ) -> None:
        """Cancel open orders from previous position (skip TP as it was already filled)."""
        for dca in old_dca_orders:
            if dca.order_id:
                try:
                    self.api.cancel_order(self.symbol, dca.order_id)
                except Exception:
                    pass
        if old_sl_order_id:
            try:
                self.api.cancel_algo_order(self.symbol, old_sl_order_id)
            except Exception:
                pass

    def _handle_tp(
        self,
        side: str,
        fill_data: Optional[Dict[str, Any]] = None,
        log_timestamp: Optional[datetime] = None,
    ) -> None:
        """Handle TP fill — re-entry first, cleanup after.

        Args:
            side: "long" or "short"
            fill_data: Order fill data from exchange (ORDER_TRADE_UPDATE).
                       Contains accurate `avg_price` and `pnl` (realized).
                       None if called from periodic_sync fallback.
            log_timestamp: Override for the trade log event time. Used by
                       offline recovery to record the actual exchange fill
                       time rather than the replay time.
        """
        state = self.long_state if side == "long" else self.short_state

        # Save old state before reset
        old_amount = state.amount
        old_dca_orders = state.dca_orders
        old_sl_order_id = state.sl_order_id
        old_tp_order_id = state.tp_order_id

        # Use exchange-provided fill data when available (most accurate)
        if fill_data and float(fill_data.get("avg_price", 0)) > 0:
            close_price = float(fill_data["avg_price"])
            realized_pnl = float(fill_data.get("pnl", 0))
        else:
            # Fallback (periodic_sync path) — use calculated tp_price
            close_price = self.strategy.calculate_tp_price(state.avg_price, side)
            realized_pnl = (
                self._cached_equity - self._last_equity
                if self._cached_equity > 0 else 0.0
            )

        # ── ① Immediate re-entry (top priority) ──
        state.reset()
        self._try_update_margin()  # refresh capital before re-entry
        self._enter_position(side, ref_price=close_price)

        # ── ② Clean up previous orders (after market order sent) ──
        self._cancel_old_orders(old_dca_orders, old_sl_order_id)

        self.logger.info(
            f"TP HIT {side.upper()} - Fill Price: {close_price:.2f}, "
            f"PnL: {realized_pnl:.2f}"
        )
        self.trade_logger.log_tp(
            self.symbol, side, close_price, old_amount, realized_pnl,
            order_id=old_tp_order_id,
            timestamp=log_timestamp,
        )

        self._save_state()

    def _handle_sl(
        self,
        side: str,
        fill_data: Optional[Dict[str, Any]] = None,
        log_timestamp: Optional[datetime] = None,
    ) -> None:
        """Handle SL fill.

        Args:
            side: "long" or "short"
            fill_data: Order fill data from exchange (ORDER_TRADE_UPDATE).
                       Contains accurate `avg_price` and `pnl` (realized).
                       None if called from periodic_sync fallback.
            log_timestamp: Override for the trade log event time. When set,
                       also used as `last_sl_time` so the cooldown starts
                       from the actual exchange fill time rather than replay.
        """
        state = self.long_state if side == "long" else self.short_state

        # SL already filled → cancel only old DCA + TP (skip SL)
        old_dca_orders = state.dca_orders
        old_tp_order_id = state.tp_order_id
        old_sl_order_id = state.sl_order_id
        state.dca_orders = []
        state.sl_order_id = None
        state.tp_order_id = None

        for dca in old_dca_orders:
            if dca.order_id:
                try:
                    self.api.cancel_order(self.symbol, dca.order_id)
                except Exception:
                    pass
        if old_tp_order_id:
            try:
                self.api.cancel_order(self.symbol, old_tp_order_id)
            except Exception:
                pass

        # Use exchange-provided fill data when available (most accurate)
        if fill_data and float(fill_data.get("avg_price", 0)) > 0:
            close_price = float(fill_data["avg_price"])
            realized_pnl = float(fill_data.get("pnl", 0))
        else:
            # Fallback (periodic_sync path) — stale, less accurate
            close_price = self._current_price
            realized_pnl = (
                self._cached_equity - self._last_equity
                if self._cached_equity > 0 else 0.0
            )

        self.logger.info(
            f"SL HIT {side.upper()} - Fill Price: {close_price:.2f}, "
            f"PnL: {realized_pnl:.2f}"
        )
        self.trade_logger.log_sl(
            self.symbol, side, close_price, state.amount, realized_pnl,
            order_id=old_sl_order_id,
            timestamp=log_timestamp,
        )

        # Reset state (preserve cooldown)
        state.reset()
        state.last_sl_time = log_timestamp or datetime.now(timezone.utc)
        self._try_update_margin()  # refresh capital for next entry

        self._save_state()

    def _check_and_handle_dca_fills(self, side: str) -> None:
        """Detect and handle DCA fills (polling backup)."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or not state.dca_orders:
            return

        try:
            open_orders = self.api.get_open_orders(self.symbol)
            open_order_ids = {str(o["id"]) for o in open_orders}

            filled_dcas = []
            remaining_dcas = []

            for dca in state.dca_orders:
                if dca.order_id and dca.order_id not in open_order_ids:
                    filled_dcas.append(dca)
                else:
                    remaining_dcas.append(dca)

            for dca in filled_dcas:
                # Query exchange for actual fill data
                try:
                    order_info = self.api.get_order(self.symbol, dca.order_id)
                    poll_data = {
                        "filled": float(order_info.get("filled", 0)),
                        "avg_price": float(order_info.get("average", 0)),
                        "status": "FILLED",
                    }
                except Exception as e:
                    self.logger.warning(f"Failed to query DCA order {dca.order_id}: {e}")
                    # Fallback to calculated values
                    leverage = self.strategy.get_leverage(side)
                    calc_amount = dca.margin * leverage / dca.trigger_price
                    poll_data = {
                        "filled": self.api.round_amount(self.symbol, calc_amount) + dca.last_filled,
                        "avg_price": dca.trigger_price,
                        "status": "FILLED",
                    }
                self._process_dca_fill(side, dca, poll_data)

            state.dca_orders = remaining_dcas
            if filled_dcas:
                self._save_state()

        except Exception as e:
            self.logger.error(f"DCA check error: {e}")

    def _process_dca_fill(self, side: str, dca: DCALevel, data: Dict[str, Any]) -> None:
        """Handle DCA fill (partial or full) — recalculate avg, re-place SL/TP."""
        state = self.long_state if side == "long" else self.short_state
        is_final = (data.get("status") == "FILLED")

        # Use actual filled qty from exchange (cumulative - previous cumulative)
        cumulative_filled = data.get("filled", 0)
        add_amount = cumulative_filled - dca.last_filled
        add_amount = self.api.round_amount(self.symbol, add_amount)
        dca.last_filled = cumulative_filled

        if add_amount <= 0:
            return

        # Use actual avg fill price (limit order fills at same price, no error)
        fill_price = data.get("avg_price", dca.trigger_price)
        if fill_price <= 0:
            fill_price = dca.trigger_price

        new_amount, new_cost, new_avg = self.strategy.calculate_avg_price(
            state.amount, state.cost, add_amount, fill_price * add_amount,
        )

        state.amount = new_amount
        state.cost = new_cost
        state.avg_price = new_avg

        # dca_count increments only on final fill
        if is_final:
            state.dca_count += 1

        fill_type = "FILLED" if is_final else "PARTIAL"
        self.logger.info(
            f"DCA{dca.level} {fill_type} {side.upper()} - "
            f"Price: {fill_price:.2f}, Qty: {add_amount:.4f}, "
            f"New Avg: {new_avg:.2f}, Total: {new_amount:.4f}"
        )
        if is_final:
            self.trade_logger.log_dca(
                self.symbol, side, dca.level,
                fill_price, cumulative_filled, dca.margin, new_avg,
                order_id=dca.order_id,
            )

        # SL uses closePosition=true — no re-placement needed on DCA fill
        # (trigger price is based on base_price which doesn't change)

        # Cancel TP → re-place (new avg_price, increased amount)
        if state.tp_order_id:
            try:
                self.api.cancel_order(self.symbol, state.tp_order_id)
                state.tp_order_id = None
            except Exception as e:
                self.logger.warning(f"Failed to cancel old TP {state.tp_order_id}: {e}")
        self._place_tp_order(side)

    def on_order_filled(self, order_id: str, data: Dict[str, Any]) -> None:
        """Detect order fills from User Data Stream."""
        status = data.get("status", "FILLED")
        with self._lock:
            for side in ["long", "short"]:
                state = self.long_state if side == "long" else self.short_state

                # Check TP fill (only on FILLED)
                if status == "FILLED" and state.tp_order_id == order_id:
                    self._handle_tp(side, data)
                    return

                # Check SL fill (only on FILLED)
                if status == "FILLED" and state.sl_order_id == order_id:
                    self._handle_sl(side, data)
                    return

                # Check DCA fill (partial or full)
                for dca in list(state.dca_orders):
                    if dca.order_id == order_id:
                        is_final = (status == "FILLED")
                        if is_final:
                            state.dca_orders.remove(dca)
                        self._process_dca_fill(side, dca, data)
                        self._save_state()
                        return

    def on_price_update(self, price: float) -> None:
        """Price update callback."""
        with self._lock:
            self._current_price = price

        if not self._initialized:
            return

        # TP/SL are handled by exchange orders → on_order_filled path
        # on_price_update only triggers new entries when idle
        if not self.long_state.active:
            if self.strategy.should_enter(self.long_state, self.cooldown_hours):
                self._enter_position("long")

        if not self.short_state.active:
            if self.strategy.should_enter(self.short_state, self.cooldown_hours):
                self._enter_position("short")

    def get_status(self) -> Dict[str, Any]:
        """Current state summary."""
        return {
            "symbol": self.symbol,
            "capital": self.capital,
            "current_price": self._current_price,
            "long": {
                "active": self.long_state.active,
                "amount": self.long_state.amount,
                "avg_price": self.long_state.avg_price,
                "dca_count": self.long_state.dca_count,
            },
            "short": {
                "active": self.short_state.active,
                "amount": self.short_state.amount,
                "avg_price": self.short_state.avg_price,
                "dca_count": self.short_state.dca_count,
            },
        }

    def shutdown(self) -> None:
        """Shutdown handler."""
        self._save_state()
        self.logger.info(f"Shutdown {self.symbol}")


class TradingExecutor:
    """Main trading executor."""

    def __init__(
        self,
        config: TradingConfig,
        testnet: bool = True,
        config_path: str = "config/config.json",
    ) -> None:
        self.config = config
        self.testnet = testnet

        self.logger = setup_logger("executor", "data/logs/executor.log")
        self.config_loader = ConfigLoader()

        # API client
        self.api = APIClient(testnet=testnet)

        # Margin manager
        self.margin_manager = MarginManager()

        # BNB balance manager (for fee discount, mainnet only)
        self._bnb_manager = BnbManager(self.api, testnet=testnet)

        # Symbol list
        self.symbols = config.get_symbol_names()

        if not self.symbols:
            raise ValueError("No symbols to trade")

        # Per-symbol trader
        self.traders: Dict[str, SymbolTrader] = {}

        # WebSocket feeds
        self.price_feed: Optional[PriceFeed] = None
        self.order_feed: Optional[OrderUpdateFeed] = None
        self._listen_key: Optional[str] = None
        self._listen_key_timer: Optional[threading.Timer] = None

        # Running state
        self._running = False
        self._shutdown_event = threading.Event()

        # config.json hot reload
        self._config_path: str = config_path
        self._config_mtime: float = 0.0
        self._pending_removals: Set[str] = set()  # Symbols pending removal (in symbol format)

    @staticmethod
    def _raw_to_symbol(raw_symbol: str) -> str:
        """Convert BTCUSDT → BTC/USDT."""
        # Handle common USDT pairs
        if raw_symbol.endswith("USDT"):
            base = raw_symbol[:-4]
            return f"{base}/USDT"
        return raw_symbol

    def _on_price_update(self, symbol: str, price: float) -> None:
        """Price update callback (called from PriceFeed)."""
        if symbol in self.traders:
            self.traders[symbol].on_price_update(price)

    def _on_order_update(self, data: Dict[str, Any]) -> None:
        """Order update callback (called from OrderUpdateFeed)."""
        raw_symbol = data.get("symbol", "")
        symbol = self._raw_to_symbol(raw_symbol)
        status = data.get("status", "")
        order_id = str(data.get("order_id", ""))

        if symbol in self.traders and status in ("FILLED", "PARTIALLY_FILLED"):
            self.logger.info(f"Order {status}: {symbol} #{order_id}")
            self.traders[symbol].on_order_filled(order_id, data)

    def _on_position_update(self, data: Dict[str, Any]) -> None:
        """Position update callback (for logging)."""
        self.logger.debug(f"Position update: {data}")

    # ---- config.json hot reload ----

    def _check_config_update(self) -> None:
        """Detect and process config.json changes (mtime comparison)."""
        try:
            current_mtime = os.path.getmtime(self._config_path)
        except OSError:
            return  # Ignore if file is inaccessible

        if current_mtime == self._config_mtime:
            return  # No changes

        self._config_mtime = current_mtime
        self.logger.info("config.json change detected, reloading...")

        try:
            new_config = TradingConfig.load(self._config_path)
        except Exception as e:
            self.logger.error(f"Config reload failed: {e}")
            return

        # Compare current vs new symbols
        old_symbols = set(self.config.symbols.keys())  # safe_name: "BTC_USDT"
        new_symbols = set(new_config.symbols.keys())

        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols
        common = old_symbols & new_symbols

        # Apply cooldown_hours changes
        if new_config.cooldown_hours != self.config.cooldown_hours:
            self.logger.info(
                f"Cooldown changed: {self.config.cooldown_hours}h → "
                f"{new_config.cooldown_hours}h"
            )
            for trader in self.traders.values():
                trader.cooldown_hours = new_config.cooldown_hours

        # Detect weight changes → attempt immediate capital recalculation
        for safe_name in common:
            old_weight = self.config.symbols[safe_name].weight
            new_weight = new_config.symbols[safe_name].weight
            if old_weight != new_weight:
                symbol = new_config.symbols[safe_name].symbol
                if symbol in self.traders:
                    self.traders[symbol].weight = new_weight
                    self.traders[symbol]._weight_changed = True
                    self.logger.info(
                        f"{symbol} weight changed: {old_weight:.4f} → {new_weight:.4f}"
                    )
                    # Apply immediately if both inactive, defer to next margin update if active
                    self.traders[symbol]._try_update_margin()

        # Add symbols
        need_reconnect = False
        for safe_name in added:
            sym_cfg = new_config.symbols[safe_name]
            self.logger.info(f"Adding new symbol: {sym_cfg.symbol}")
            if self._add_trader(safe_name, sym_cfg):
                need_reconnect = True

        # Remove symbols — register for pending removal, not immediate shutdown
        for safe_name in removed:
            symbol = self.config.symbols[safe_name].symbol
            if symbol in self.traders:
                self._mark_trader_for_removal(symbol)

        # Replace config object
        self.config = new_config

        # Reconnect PriceFeed (only when symbols added)
        if need_reconnect:
            self._reconnect_price_feed()

    def _add_trader(self, safe_name: str, sym_cfg: Any) -> bool:
        """Add new trader at runtime. Returns True on success."""
        symbol = sym_cfg.symbol

        if symbol in self.traders:
            self.logger.warning(f"{symbol} already exists, skipping add")
            return False

        try:
            # Load parameters
            params = self.config_loader.load(safe_name)

            # Initialize margin
            total_balance = self.api.get_account_equity()
            capital = self.margin_manager.load_or_init(
                symbol, sym_cfg.weight, total_balance
            )

            trader = SymbolTrader(
                symbol=symbol,
                params=params,
                api_client=self.api,
                capital=capital,
                weight=sym_cfg.weight,
                margin_manager=self.margin_manager,
                config_loader=self.config_loader,
                safe_name=safe_name,
                cooldown_hours=self.config.cooldown_hours,
                config_path=self._config_path,
            )
            trader.initialize()
            self.traders[symbol] = trader

            self.logger.info(f"Trader added: {symbol} (${capital:.2f})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add trader {symbol}: {e}")
            return False

    def _mark_trader_for_removal(self, symbol: str) -> None:
        """Register symbol for pending removal (blocks new entries)."""
        if symbol not in self.traders:
            return

        self._pending_removals.add(symbol)
        self.traders[symbol]._marked_for_removal = True
        self.logger.info(
            f"{symbol} marked for removal (waiting for positions to close)"
        )

    def _remove_trader(self, symbol: str) -> None:
        """Fully remove trader (call after confirming both sides inactive)."""
        trader = self.traders.get(symbol)
        if not trader:
            return

        # Save state + shutdown
        trader.shutdown()

        # Cancel all open orders
        try:
            self.api.cancel_all_orders(symbol)
        except Exception as e:
            self.logger.error(f"Cancel orders error for {symbol}: {e}")

        del self.traders[symbol]
        self._pending_removals.discard(symbol)

        self.logger.info(f"Trader removed: {symbol}")

        # Reconnect PriceFeed (subscription symbols changed)
        self._reconnect_price_feed()

    def _process_pending_removals(self) -> None:
        """Actually remove pending symbols where both sides are inactive."""
        for symbol in list(self._pending_removals):
            trader = self.traders.get(symbol)
            if not trader:
                self._pending_removals.discard(symbol)
                continue

            if not trader.long_state.active and not trader.short_state.active:
                self.logger.info(
                    f"{symbol} both sides inactive, removing trader"
                )
                self._remove_trader(symbol)

    def _reconnect_price_feed(self) -> None:
        """Reconnect PriceFeed based on current traders."""
        if not self.price_feed:
            return

        current_symbols = list(self.traders.keys())
        if not current_symbols:
            self.logger.warning("No symbols left, stopping price feed")
            self.price_feed.stop()
            return

        self.logger.info(f"Reconnecting PriceFeed: {current_symbols}")
        self.price_feed.stop()
        time.sleep(1)

        self.price_feed = PriceFeed(
            symbols=current_symbols,
            on_price_update=self._on_price_update,
            testnet=self.testnet,
        )
        self.price_feed.start()

    def _start_listen_key_renewal(self) -> None:
        """Renew Listen Key every 30 minutes."""
        if not self._running or not self._listen_key:
            return
        try:
            self.api.renew_listen_key(self._listen_key)
            self.logger.debug("Listen key renewed")
        except Exception as e:
            self.logger.error(f"Listen key renewal failed: {e}")
            # Attempt to generate new key
            try:
                self._listen_key = self.api.new_listen_key()
                self.logger.info("New listen key created")
            except Exception as e2:
                self.logger.error(f"New listen key failed: {e2}")

        # Schedule next renewal (30 minutes)
        if self._running:
            self._listen_key_timer = threading.Timer(
                30 * 60, self._start_listen_key_renewal
            )
            self._listen_key_timer.daemon = True
            self._listen_key_timer.start()

    def _start_order_retry_loop(self) -> None:
        """Start background retry thread (15-second interval)."""
        def _retry_worker():
            while self._running and not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=15)
                if self._shutdown_event.is_set():
                    break
                for trader in list(self.traders.values()):
                    if trader.has_pending_orders():
                        trader.retry_pending_orders()

        t = threading.Thread(target=_retry_worker, daemon=True)
        t.start()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers."""
        def handle_shutdown(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    def run(self) -> None:
        """Start main trading loop."""
        self.logger.info("=" * 50)
        self.logger.info("Starting Trading Executor")
        self.logger.info(f"Testnet: {self.testnet}")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info("=" * 50)

        self._running = True
        self._setup_signal_handlers()

        try:
            # Set Hedge Mode
            self.api.set_position_mode(hedge_mode=True)

            # Fetch total balance from Binance
            total_balance = self.api.get_account_equity()
            self.logger.info(f"Total wallet balance: ${total_balance:.2f}")

            # Display per-symbol capital allocation
            for safe_name, sym_cfg in self.config.symbols.items():
                cap = total_balance * sym_cfg.weight
                self.logger.info(
                    f"  {sym_cfg.symbol}: weight={sym_cfg.weight:.4f}, "
                    f"capital=${cap:.2f}"
                )

            # Initialize traders
            for safe_name, sym_cfg in self.config.symbols.items():
                symbol = sym_cfg.symbol
                try:
                    # Load parameters
                    params = self.config_loader.load(safe_name)

                    # Load or initialize margin
                    capital = self.margin_manager.load_or_init(
                        symbol, sym_cfg.weight, total_balance
                    )

                    trader = SymbolTrader(
                        symbol=symbol,
                        params=params,
                        api_client=self.api,
                        capital=capital,
                        weight=sym_cfg.weight,
                        margin_manager=self.margin_manager,
                        config_loader=self.config_loader,
                        safe_name=safe_name,
                        cooldown_hours=self.config.cooldown_hours,
                        config_path=self._config_path,
                    )
                    trader.initialize()
                    self.traders[symbol] = trader

                    self.logger.info(f"Trader initialized: {symbol} (${capital:.2f})")

                except Exception as e:
                    self.logger.error(f"Failed to init {symbol}: {e}")

            if not self.traders:
                raise RuntimeError("No traders initialized")

            # Log symbol trading specs
            self.logger.info("=== Symbol Trading Specs ===")
            for symbol in self.traders:
                specs = self.api.get_symbol_specs(symbol)
                self.logger.info(
                    f"  {symbol}: stepSize={specs['stepSize']}, "
                    f"tickSize={specs['tickSize']}, minQty={specs['minQty']}"
                )

            # Start price feed
            self.price_feed = PriceFeed(
                symbols=list(self.traders.keys()),
                on_price_update=self._on_price_update,
                testnet=self.testnet,
            )
            self.price_feed.start()

            # Start User Data Stream
            try:
                self._listen_key = self.api.new_listen_key()
                self.order_feed = OrderUpdateFeed(
                    listen_key=self._listen_key,
                    on_order_update=self._on_order_update,
                    on_position_update=self._on_position_update,
                    testnet=self.testnet,
                )
                self.order_feed.start()
                self._start_listen_key_renewal()
                self.logger.info("User Data Stream started")
            except Exception as e:
                self.logger.warning(f"User Data Stream failed: {e} (using polling fallback)")

            self.logger.info("Trading started")

            # Save initial config.json mtime
            try:
                self._config_mtime = os.path.getmtime(self._config_path)
            except OSError:
                self._config_mtime = 0.0

            # Start order retry background thread
            self._start_order_retry_loop()

            # Main loop (status monitoring + polling backup)
            poll_counter = 0
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=60)

                if not self._shutdown_event.is_set():
                    poll_counter += 1
                    self._log_status()

                    # Process pending removals (every 60 seconds)
                    if self._pending_removals:
                        self._process_pending_removals()

                    # Every minute: watchdog — reconnect PriceFeed if ticks are stale.
                    # Mark prices stream once per second; silence > 120s means the
                    # WebSocket died without firing _on_close (see past incidents
                    # on 2026-02-26, 03-16, 03-25, 04-06, 04-15).
                    if self.price_feed:
                        stale = self.price_feed.seconds_since_last_tick()
                        if stale > 120:
                            self.logger.warning(
                                f"PriceFeed stale ({stale:.0f}s since last tick) — reconnecting"
                            )
                            self._reconnect_price_feed()

                    # Every 5 minutes: detect config changes + DCA polling backup + hot param update + sync
                    if poll_counter % 5 == 0:
                        self._check_config_update()

                        # Fetch equity once, distribute to all traders
                        try:
                            cached_equity = self.api.get_account_equity()
                        except Exception:
                            cached_equity = 0.0

                        for trader in self.traders.values():
                            trader.update_cached_equity(cached_equity)
                            trader._try_update_margin()
                            trader._check_and_handle_dca_fills("long")
                            trader._check_and_handle_dca_fills("short")
                            trader._try_hot_update_params()
                            trader.periodic_sync()

                    # Every 10 minutes: check and refill BNB balance + save equity snapshot
                    if poll_counter % 10 == 0:
                        self._bnb_manager.check_and_refill()
                        self._save_equity_snapshot()

        except Exception as e:
            self.logger.error(f"Executor error: {e}")
            raise

        finally:
            self.shutdown()

    _SNAPSHOT_DIR = Path("data/balance_snapshots")

    def _save_equity_snapshot(self) -> None:
        """Save equity snapshot to JSONL file (for 24h/monthly PnL)."""
        try:
            equity = self.api.get_account_equity()
            if equity <= 0:
                return
            self._SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc)
            filename = f"balance_{now.strftime('%Y%m')}.jsonl"
            record = json.dumps({"t": now.isoformat(), "v": round(equity, 4)})
            with (self._SNAPSHOT_DIR / filename).open("a", encoding="utf-8") as f:
                f.write(record + "\n")
        except Exception as e:
            self.logger.error(f"Snapshot save error: {e}")

    def _log_status(self) -> None:
        """Log current state to file."""
        for symbol, trader in self.traders.items():
            status = trader.get_status()
            self.logger.info(
                f"{symbol} [${status['capital']:.2f}] - "
                f"Price: {status['current_price']:.2f} | "
                f"Long: {status['long']['amount']:.4f} @ {status['long']['avg_price']:.2f} "
                f"(DCA: {status['long']['dca_count']}) | "
                f"Short: {status['short']['amount']:.4f} @ {status['short']['avg_price']:.2f} "
                f"(DCA: {status['short']['dca_count']})"
            )

    def shutdown(self) -> None:
        """Safe shutdown."""
        self.logger.info("Shutting down...")
        self._running = False

        # Stop Listen Key renewal timer
        if self._listen_key_timer:
            self._listen_key_timer.cancel()

        # Stop User Data Stream
        if self.order_feed:
            self.order_feed.stop()

        # Delete Listen Key
        if self._listen_key:
            try:
                self.api.close_listen_key(self._listen_key)
            except Exception:
                pass

        # Stop price feed
        if self.price_feed:
            self.price_feed.stop()

        # Shutdown traders
        for trader in self.traders.values():
            trader.shutdown()

        self.logger.info("Shutdown complete")
