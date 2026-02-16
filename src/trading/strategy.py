"""DCA 전략 로직 (파라미터 주입형)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DCALevel:
    """DCA 주문 레벨."""
    level: int  # DCA 순서 (1, 2, 3, ...)
    trigger_price: float  # 트리거 가격
    margin: float  # 마진 금액
    order_id: Optional[str] = None  # 거래소 주문 ID


@dataclass
class PositionState:
    """포지션 상태 추적."""
    side: str  # "long" 또는 "short"
    active: bool = False  # 포지션 활성 여부
    amount: float = 0.0  # 보유 수량 (contracts)
    cost: float = 0.0  # 투입 마진 합계
    avg_price: float = 0.0  # 평균 진입가
    base_price: float = 0.0  # 최초 진입가 (DCA/SL 기준)
    dca_count: int = 0  # 체결된 DCA 횟수
    last_sl_time: Optional[datetime] = None  # 마지막 SL 시각
    entry_time: Optional[datetime] = None  # 진입 시각

    # 관리 중인 주문들
    dca_orders: List[DCALevel] = field(default_factory=list)
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None

    def reset(self) -> None:
        """포지션 청산 후 리셋."""
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
        """직렬화."""
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
        """역직렬화."""
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
    """양방향 DCA 전략 계산기."""

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        전략 초기화.

        Args:
            params: 최적화된 파라미터 JSON
                {
                    "parameters": {"long": {...}, "short": {...}},
                    "fixed_settings": {...}
                }
        """
        self.params = params

        # 파라미터 추출
        self.long_params = params["parameters"]["long"]
        self.short_params = params["parameters"]["short"]
        self.fixed = params["fixed_settings"]

        # 고정 설정
        self.long_allocation = self.fixed["long_allocation"]
        self.short_allocation = self.fixed["short_allocation"]
        self.long_leverage = self.fixed["long_leverage"]
        self.short_leverage = self.fixed["short_leverage"]
        self.base_margin = self.fixed["base_margin_ratio"]
        self.dca_margin = self.fixed["dca_margin_ratio"]

    def get_side_params(self, side: str) -> Dict[str, Any]:
        """방향별 파라미터 반환."""
        return self.long_params if side == "long" else self.short_params

    def get_leverage(self, side: str) -> int:
        """방향별 레버리지."""
        return self.long_leverage if side == "long" else self.short_leverage

    def get_allocation(self, side: str) -> float:
        """방향별 자본 배분율."""
        return self.long_allocation if side == "long" else self.short_allocation

    def calculate_base_margin(self, capital: float, side: str) -> float:
        """
        Base 주문 마진 계산.

        Args:
            capital: 총 자본
            side: "long" 또는 "short"

        Returns:
            Base 마진 금액
        """
        allocation = self.get_allocation(side)
        return capital * allocation * self.base_margin

    def calculate_dca_levels(
        self,
        base_price: float,
        side: str,
        capital: float,
    ) -> List[DCALevel]:
        """
        DCA 레벨 계산 (ga_engine.py 로직 포팅).

        Long: 가격이 base_price * (1 - cumulative_deviation) 아래로 떨어지면 트리거
        Short: 가격이 base_price * (1 + cumulative_deviation) 위로 오르면 트리거

        Args:
            base_price: 최초 진입가
            side: "long" 또는 "short"
            capital: 총 자본

        Returns:
            DCA 레벨 리스트
        """
        params = self.get_side_params(side)

        price_deviation = params["price_deviation"]
        dev_multiplier = params["dev_multiplier"]
        vol_multiplier = params["vol_multiplier"]
        max_dca = int(params["max_dca"])

        allocation = self.get_allocation(side)
        base_dca_margin = capital * allocation * self.dca_margin

        levels: List[DCALevel] = []

        # 기하급수적 deviation 계산
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
        Take Profit 가격 계산.

        Args:
            avg_price: 평균 진입가
            side: "long" 또는 "short"

        Returns:
            TP 가격
        """
        params = self.get_side_params(side)
        tp_ratio = params["take_profit"]

        if side == "long":
            return avg_price * (1.0 + tp_ratio)
        else:
            return avg_price * (1.0 - tp_ratio)

    def calculate_sl_price(self, base_price: float, side: str) -> float:
        """
        Stop Loss 가격 계산 (base_price 기준).

        Args:
            base_price: 최초 진입가
            side: "long" 또는 "short"

        Returns:
            SL 가격
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
        마진으로 살 수 있는 포지션 수량 계산.

        Args:
            margin: 투입 마진
            price: 현재 가격
            side: "long" 또는 "short"

        Returns:
            포지션 수량 (contracts)
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
        DCA 후 새로운 평균가 계산.

        Args:
            current_amount: 현재 보유 수량
            current_cost: 현재 투입 마진
            add_amount: 추가 수량
            add_margin: 추가 마진
            side: "long" 또는 "short"

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
        신규 진입 가능 여부 (쿨다운 체크).

        Args:
            state: 현재 포지션 상태
            cooldown_hours: SL 후 대기 시간

        Returns:
            True면 진입 가능
        """
        # 이미 포지션 있으면 불가
        if state.active:
            return False

        # SL 이력이 없으면 바로 진입
        if state.last_sl_time is None:
            return True

        # 쿨다운 체크
        now = datetime.now(timezone.utc)
        elapsed = (now - state.last_sl_time).total_seconds() / 3600
        return elapsed >= cooldown_hours

    def check_tp_hit(self, current_price: float, state: PositionState) -> bool:
        """
        TP 도달 여부 확인.

        Args:
            current_price: 현재 가격
            state: 포지션 상태

        Returns:
            True면 TP 도달
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
        SL 도달 여부 확인.

        Args:
            current_price: 현재 가격
            state: 포지션 상태

        Returns:
            True면 SL 도달
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
        DCA 트리거 확인.

        Args:
            current_price: 현재 가격
            state: 포지션 상태

        Returns:
            트리거된 DCA 레벨 또는 None
        """
        if not state.active:
            return None

        # 아직 체결 안 된 DCA 중 첫 번째만 확인
        for dca in state.dca_orders:
            if dca.order_id is not None:
                # 이미 주문 걸려 있으면 스킵 (체결은 거래소에서 처리)
                continue

            if state.side == "long" and current_price <= dca.trigger_price:
                return dca
            elif state.side == "short" and current_price >= dca.trigger_price:
                return dca

        return None

    def estimate_pnl(
        self,
        exit_price: float,
        state: PositionState,
        fee_rate: float = 0.0005,
    ) -> float:
        """
        예상 PnL 계산.

        Args:
            exit_price: 청산 가격
            state: 포지션 상태
            fee_rate: 수수료율

        Returns:
            예상 PnL (마진 기준)
        """
        if not state.active or state.amount <= 0:
            return 0.0

        leverage = self.get_leverage(state.side)
        notional = state.amount * exit_price

        if state.side == "long":
            pnl = notional - (state.cost * leverage)
        else:
            pnl = (state.cost * leverage) - notional

        fee = notional * fee_rate
        return state.cost + pnl - fee
