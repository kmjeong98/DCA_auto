"""Order execution and balance management."""

import json
import os
import signal
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.common.api_client import APIClient
from src.common.config_loader import ConfigLoader
from src.common.logger import setup_logger
from src.common.trading_config import TradingConfig
from src.trading.bnb_manager import BnbManager
from src.trading.margin_manager import MarginManager
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

        # Sync with exchange positions
        self._sync_with_exchange()

        # Initialize balance snapshot (PnL baseline)
        try:
            self._last_equity = self.api.get_account_equity()
        except Exception as e:
            self.logger.warning(f"Initial equity snapshot failed: {e}")

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
                        if self.long_state.base_price == 0:
                            self.long_state.base_price = entry_price

                elif side == "short" and amount > 0:
                    exchange_short_amt = amount
                    self.short_state.active = True
                    self.short_state.amount = amount
                    if entry_price > 0:
                        self.short_state.avg_price = entry_price
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

    def _try_update_margin(self) -> None:
        """Attempt margin update when both sides are inactive."""
        if self.long_state.active or self.short_state.active:
            return  # Skip if either side is active

        # Apply latest weight from config
        self._reload_weight_from_config()

        try:
            new_balance = self.api.get_account_equity()
            force = self._weight_changed
            self.capital = self.margin_manager.try_update(
                self.symbol, self.weight, self.capital, new_balance, force=force
            )
            self._weight_changed = False
        except Exception as e:
            self.logger.error(f"Margin update error: {e}")

        # Also attempt parameter update
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
                except Exception:
                    pass
                state.sl_order_id = None

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

    def _enter_position(self, side: str) -> bool:
        """Enter base position via market order."""
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

            # Fetch current price
            price = self._current_price
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
            state.cost = margin
            state.avg_price = fill_price
            state.base_price = fill_price
            state.dca_count = 0
            state.entry_time = datetime.now(timezone.utc)

            self.logger.info(
                f"ENTRY {side.upper()} - Price: {fill_price:.2f}, "
                f"Amount: {filled_amount:.4f}, Margin: {margin:.2f}"
            )
            self.trade_logger.log_entry(self.symbol, side, fill_price, filled_amount, margin)

            # Place DCA orders
            self._place_dca_orders(side)

            # Place SL order
            self._place_sl_order(side)

            # Place TP order
            self._place_tp_order(side)

            # Update balance snapshot (baseline after entry)
            try:
                self._last_equity = self.api.get_account_equity()
            except Exception:
                pass

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
        """Place SL order (STOP_MARKET — mark price trigger). Schedule retry on failure."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.amount <= 0:
            self._pending_sl.discard(side)
            return

        try:
            sl_price = self.strategy.calculate_sl_price(state.base_price, side)
            sl_price = self.api.round_price(self.symbol, sl_price)

            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "sell" if side == "long" else "buy"

            order = self.api.place_stop_loss(
                self.symbol, order_side, state.amount, sl_price, position_side,
            )

            state.sl_order_id = str(order.get("id"))
            self._pending_sl.discard(side)

            self.logger.info(
                f"SL {side.upper()} placed - Price: {sl_price:.2f}, "
                f"Amount: {state.amount:.4f}"
            )

        except Exception as e:
            self.logger.error(f"SL order error ({side}): {e}")
            self._pending_sl.add(side)

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

            order = self.api.place_take_profit(
                self.symbol, order_side, state.amount, tp_price, position_side,
            )

            state.tp_order_id = str(order.get("id"))
            self._pending_tp.discard(side)

            self.logger.info(
                f"TP {side.upper()} placed - Price: {tp_price:.2f}, "
                f"Amount: {state.amount:.4f}"
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

    def _try_update_margin_with_equity(self, equity: float) -> None:
        """Update margin using externally provided equity (avoids duplicate REST calls)."""
        if self.long_state.active or self.short_state.active:
            return
        if equity <= 0:
            return

        # Apply latest weight from config
        self._reload_weight_from_config()

        try:
            force = self._weight_changed
            self.capital = self.margin_manager.try_update(
                self.symbol, self.weight, self.capital, equity, force=force
            )
            self._weight_changed = False
        except Exception as e:
            self.logger.error(f"Margin update error: {e}")
        self._try_update_params()

    def _handle_tp(self, side: str) -> None:
        """Handle TP fill — re-entry first, cleanup after."""
        state = self.long_state if side == "long" else self.short_state

        # ── ① Immediate re-entry (top priority) ──
        # Temporarily save old order ID, reset → enter market order immediately
        old_amount = state.amount
        old_dca_orders = state.dca_orders
        old_sl_order_id = state.sl_order_id
        state.reset()

        self._enter_position(side)

        # ── ② Clean up previous orders (after market order sent) ──
        self._cancel_old_orders(old_dca_orders, old_sl_order_id)

        # ── ③ Calculate PnL (called once) ──
        try:
            new_equity = self.api.get_account_equity()
            pnl = new_equity - self._last_equity
            self._last_equity = new_equity
        except Exception:
            new_equity = 0.0
            pnl = 0.0

        self.logger.info(
            f"TP HIT {side.upper()} - Price: {self._current_price:.2f}, "
            f"PnL: {pnl:.2f}"
        )
        self.trade_logger.log_tp(
            self.symbol, side, self._current_price, old_amount, pnl
        )

        # ── ④ Update margin (reuse equity, avoid duplicate calls) ──
        self._try_update_margin_with_equity(new_equity)

        self._save_state()

    def _handle_sl(self, side: str) -> None:
        """Handle SL fill."""
        state = self.long_state if side == "long" else self.short_state

        # SL already filled → cancel only old DCA + TP (skip SL)
        old_dca_orders = state.dca_orders
        old_tp_order_id = state.tp_order_id
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

        # Balance-based PnL calculation (once)
        try:
            new_equity = self.api.get_account_equity()
            pnl = new_equity - self._last_equity
            self._last_equity = new_equity
        except Exception:
            new_equity = 0.0
            pnl = 0.0

        self.logger.info(
            f"SL HIT {side.upper()} - Price: {self._current_price:.2f}, "
            f"PnL: {pnl:.2f}"
        )
        self.trade_logger.log_sl(
            self.symbol, side, self._current_price, state.amount, pnl
        )

        # Reset state (preserve cooldown)
        state.reset()
        state.last_sl_time = datetime.now(timezone.utc)

        # Update margin (reuse equity)
        self._try_update_margin_with_equity(new_equity)

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
                self._process_dca_fill(side, dca)

            state.dca_orders = remaining_dcas
            if filled_dcas:
                self._save_state()

        except Exception as e:
            self.logger.error(f"DCA check error: {e}")

    def _process_dca_fill(self, side: str, dca: DCALevel) -> None:
        """Handle DCA fill — recalculate avg, re-place SL/TP."""
        state = self.long_state if side == "long" else self.short_state

        leverage = self.strategy.get_leverage(side)
        add_notional = dca.margin * leverage
        add_amount = add_notional / dca.trigger_price

        new_amount, new_cost, new_avg = self.strategy.calculate_avg_price(
            state.amount, state.cost, add_amount, dca.margin, side
        )

        state.amount = new_amount
        state.cost = new_cost
        state.avg_price = new_avg
        state.dca_count += 1

        self.logger.info(
            f"DCA{dca.level} FILLED {side.upper()} - "
            f"Price: {dca.trigger_price:.2f}, New Avg: {new_avg:.2f}, "
            f"Amount: {new_amount:.4f}"
        )
        self.trade_logger.log_dca(
            self.symbol, side, dca.level,
            dca.trigger_price, add_amount, dca.margin, new_avg
        )

        # Cancel SL → re-place (same base_price, increased amount) — Algo Order
        if state.sl_order_id:
            try:
                self.api.cancel_algo_order(self.symbol, state.sl_order_id)
            except Exception:
                pass
            state.sl_order_id = None
        self._place_sl_order(side)

        # Cancel TP → re-place (new avg_price, increased amount)
        if state.tp_order_id:
            try:
                self.api.cancel_order(self.symbol, state.tp_order_id)
            except Exception:
                pass
            state.tp_order_id = None
        self._place_tp_order(side)

    def on_order_filled(self, order_id: str, data: Dict[str, Any]) -> None:
        """Detect order fills from User Data Stream."""
        with self._lock:
            for side in ["long", "short"]:
                state = self.long_state if side == "long" else self.short_state

                # Check TP fill
                if state.tp_order_id == order_id:
                    self._handle_tp(side)
                    return

                # Check SL fill
                if state.sl_order_id == order_id:
                    self._handle_sl(side)
                    return

                # Check DCA fill
                for dca in list(state.dca_orders):
                    if dca.order_id == order_id:
                        state.dca_orders.remove(dca)
                        self._process_dca_fill(side, dca)
                        self._save_state()
                        return

    def on_price_update(self, price: float) -> None:
        """Price update callback."""
        with self._lock:
            self._current_price = price

        if not self._initialized:
            return

        # Handle Long position
        if self.long_state.active:
            if self.strategy.check_tp_hit(price, self.long_state):
                self._handle_tp("long")
            elif self.strategy.check_sl_hit(price, self.long_state):
                self._handle_sl("long")
        else:
            if self.strategy.should_enter(self.long_state, self.cooldown_hours):
                self._enter_position("long")

        # Handle Short position
        if self.short_state.active:
            if self.strategy.check_tp_hit(price, self.short_state):
                self._handle_tp("short")
            elif self.strategy.check_sl_hit(price, self.short_state):
                self._handle_sl("short")
        else:
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

        if symbol in self.traders and status == "FILLED":
            self.logger.info(f"Order filled: {symbol} #{order_id}")
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

                    # Every 5 minutes: detect config changes + DCA polling backup + hot param update
                    if poll_counter % 5 == 0:
                        self._check_config_update()

                        for trader in self.traders.values():
                            trader._check_and_handle_dca_fills("long")
                            trader._check_and_handle_dca_fills("short")
                            trader._try_hot_update_params()

                    # Every 10 minutes: check and refill BNB balance
                    if poll_counter % 10 == 0:
                        self._bnb_manager.check_and_refill()

        except Exception as e:
            self.logger.error(f"Executor error: {e}")
            raise

        finally:
            self.shutdown()

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
