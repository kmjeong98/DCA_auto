"""DCA strategy logic (parameter-injection style)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DCALevel:
    """DCA order level."""
    level: int  # DCA sequence number (1, 2, 3, ...)
    trigger_price: float  # trigger price
    margin: float  # margin amount
    order_id: Optional[str] = None  # exchange order ID


@dataclass
class PositionState:
    """Position state tracker."""
    side: str  # "long" or "short"
    active: bool = False  # whether position is active
    amount: float = 0.0  # held quantity (contracts)
    cost: float = 0.0  # total margin invested
    avg_price: float = 0.0  # average entry price
    base_price: float = 0.0  # initial entry price (DCA/SL reference)
    dca_count: int = 0  # number of DCA fills executed
    last_sl_time: Optional[datetime] = None  # timestamp of last SL
    entry_time: Optional[datetime] = None  # entry timestamp

    # Managed orders
    dca_orders: List[DCALevel] = field(default_factory=list)
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None

    def reset(self) -> None:
        """Reset after position close."""
        self.active = False
        self.amount = 0.0
        self.cost = 0.0
        self.avg_price = 0.0
        self.base_price = 0.0
        self.dca_count = 0
        self.entry_time = None
        self.dca_orders = []
        self.sl_order_id = None
        self.tp_order_id = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize."""
        return {
            "side": self.side,
            "active": self.active,
            "amount": self.amount,
            "cost": self.cost,
            "avg_price": self.avg_price,
            "base_price": self.base_price,
            "dca_count": self.dca_count,
            "last_sl_time": self.last_sl_time.isoformat() if self.last_sl_time else None,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "dca_orders": [
                {"level": d.level, "trigger_price": d.trigger_price, "margin": d.margin, "order_id": d.order_id}
                for d in self.dca_orders
            ],
            "sl_order_id": self.sl_order_id,
            "tp_order_id": self.tp_order_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PositionState":
        """Deserialize."""
        state = cls(side=data["side"])
        state.active = data.get("active", False)
        state.amount = data.get("amount", 0.0)
        state.cost = data.get("cost", 0.0)
        state.avg_price = data.get("avg_price", 0.0)
        state.base_price = data.get("base_price", 0.0)
        state.dca_count = data.get("dca_count", 0)

        if data.get("last_sl_time"):
            state.last_sl_time = datetime.fromisoformat(data["last_sl_time"])
        if data.get("entry_time"):
            state.entry_time = datetime.fromisoformat(data["entry_time"])

        state.dca_orders = [
            DCALevel(
                level=d["level"],
                trigger_price=d["trigger_price"],
                margin=d["margin"],
                order_id=d.get("order_id"),
            )
            for d in data.get("dca_orders", [])
        ]
        state.sl_order_id = data.get("sl_order_id")
        state.tp_order_id = data.get("tp_order_id")

        return state


class DCAStrategy:
    """Bidirectional DCA strategy calculator."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize strategy.

        Args:
            params: Optimized parameter JSON
                {
                    "parameters": {"long": {...}, "short": {...}},
                    "fixed_settings": {...}
                }
        """
        self.params = params

        # Extract parameters
        self.long_params = params["parameters"]["long"]
        self.short_params = params["parameters"]["short"]
        self.fixed = params["fixed_settings"]

        # Fixed settings
        self.leverage = self.fixed["leverage"]
        self.base_margin = self.fixed["base_margin_ratio"]
        self.dca_margin = self.fixed["dca_margin_ratio"]

    def get_side_params(self, side: str) -> Dict[str, Any]:
        """Return parameters for the given side."""
        return self.long_params if side == "long" else self.short_params

    def get_leverage(self, side: str = "") -> int:
        """Return leverage (Binance uses one per symbol, side is ignored)."""
        return self.leverage

    def calculate_base_margin(self, capital: float, side: str = "") -> float:
        """
        Calculate base order margin.

        Args:
            capital: Total capital
            side: Unused (kept for call compatibility)

        Returns:
            Base margin amount
        """
        return capital * self.base_margin

    def calculate_dca_levels(
        self,
        base_price: float,
        side: str,
        capital: float,
    ) -> List[DCALevel]:
        """
        Calculate DCA levels (ported from ga_engine.py logic).

        Long: triggers when price falls below base_price * (1 - cumulative_deviation)
        Short: triggers when price rises above base_price * (1 + cumulative_deviation)

        Args:
            base_price: Initial entry price
            side: "long" or "short"
            capital: Total capital

        Returns:
            List of DCA levels
        """
        params = self.get_side_params(side)

        price_deviation = params["price_deviation"]
        dev_multiplier = params["dev_multiplier"]
        vol_multiplier = params["vol_multiplier"]
        max_dca = int(params["max_dca"])

        base_dca_margin = capital * self.dca_margin

        levels: List[DCALevel] = []

        # Geometric deviation calculation
        curr_ratio = 1.0
        curr_step_dev = price_deviation
        curr_vol = 1.0

        for i in range(max_dca):
            if side == "long":
                curr_ratio = curr_ratio * (1.0 - curr_step_dev)
                trigger_price = base_price * curr_ratio
            else:  # short
                curr_ratio = curr_ratio * (1.0 + curr_step_dev)
                trigger_price = base_price * curr_ratio

            margin = base_dca_margin * curr_vol

            levels.append(DCALevel(
                level=i + 1,
                trigger_price=trigger_price,
                margin=margin,
            ))

            curr_step_dev *= dev_multiplier
            curr_vol *= vol_multiplier

        return levels

    def calculate_tp_price(self, avg_price: float, side: str) -> float:
        """
        Calculate take profit price.

        Args:
            avg_price: Average entry price
            side: "long" or "short"

        Returns:
            TP price
        """
        params = self.get_side_params(side)
        tp_ratio = params["take_profit"]

        if side == "long":
            return avg_price * (1.0 + tp_ratio)
        else:
            return avg_price * (1.0 - tp_ratio)

    def calculate_sl_price(self, base_price: float, side: str) -> float:
        """
        Calculate stop loss price (based on base_price).

        Args:
            base_price: Initial entry price
            side: "long" or "short"

        Returns:
            SL price
        """
        params = self.get_side_params(side)
        sl_ratio = params["stop_loss"]

        if side == "long":
            return base_price * (1.0 - sl_ratio)
        else:
            return base_price * (1.0 + sl_ratio)

    def calculate_position_amount(
        self,
        margin: float,
        price: float,
        side: str,
    ) -> float:
        """
        Calculate position quantity purchasable with margin.

        Args:
            margin: Margin to invest
            price: Current price
            side: "long" or "short"

        Returns:
            Position quantity (contracts)
        """
        leverage = self.get_leverage(side)
        notional = margin * leverage
        return notional / price

    def calculate_avg_price(
        self,
        current_amount: float,
        current_cost: float,
        add_amount: float,
        add_margin: float,
        side: str,
    ) -> Tuple[float, float, float]:
        """
        Calculate new average price after DCA.

        Args:
            current_amount: Current held quantity
            current_cost: Current margin invested
            add_amount: Additional quantity
            add_margin: Additional margin
            side: "long" or "short"

        Returns:
            (new_amount, new_cost, new_avg_price)
        """
        leverage = self.get_leverage(side)

        new_amount = current_amount + add_amount
        new_cost = current_cost + add_margin
        new_avg = (new_cost * leverage) / new_amount

        return new_amount, new_cost, new_avg

    def should_enter(
        self,
        state: PositionState,
        cooldown_hours: int = 6,
    ) -> bool:
        """
        Check whether new entry is allowed (cooldown check).

        Args:
            state: Current position state
            cooldown_hours: Wait time after SL

        Returns:
            True if entry is allowed
        """
        # Cannot enter if position already active
        if state.active:
            return False

        # Enter immediately if no SL history
        if state.last_sl_time is None:
            return True

        # Cooldown check
        now = datetime.now(timezone.utc)
        elapsed = (now - state.last_sl_time).total_seconds() / 3600
        return elapsed >= cooldown_hours

    def check_tp_hit(self, current_price: float, state: PositionState) -> bool:
        """
        Check whether TP has been reached.

        Args:
            current_price: Current price
            state: Position state

        Returns:
            True if TP reached
        """
        if not state.active or state.amount <= 0:
            return False

        tp_price = self.calculate_tp_price(state.avg_price, state.side)

        if state.side == "long":
            return current_price >= tp_price
        else:
            return current_price <= tp_price

    def check_sl_hit(self, current_price: float, state: PositionState) -> bool:
        """
        Check whether SL has been reached.

        Args:
            current_price: Current price
            state: Position state

        Returns:
            True if SL reached
        """
        if not state.active or state.amount <= 0:
            return False

        sl_price = self.calculate_sl_price(state.base_price, state.side)

        if state.side == "long":
            return current_price <= sl_price
        else:
            return current_price >= sl_price

    def check_dca_triggered(
        self,
        current_price: float,
        state: PositionState,
    ) -> Optional[DCALevel]:
        """
        Check DCA trigger.

        Args:
            current_price: Current price
            state: Position state

        Returns:
            Triggered DCA level or None
        """
        if not state.active:
            return None

        # Check only the first unfilled DCA
        for dca in state.dca_orders:
            if dca.order_id is not None:
                # Already has an order on the exchange — skip (fill handled by exchange)
                continue

            if state.side == "long" and current_price <= dca.trigger_price:
                return dca
            elif state.side == "short" and current_price >= dca.trigger_price:
                return dca

        return None
