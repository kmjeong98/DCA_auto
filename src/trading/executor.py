"""주문 체결 및 잔고 관리."""

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
from src.trading.margin_manager import MarginManager
from src.trading.price_feed import PriceFeed, OrderUpdateFeed
from src.trading.state_manager import StateManager, TradeLogger
from src.trading.strategy import DCAStrategy, DCALevel, PositionState


class SymbolTrader:
    """심볼별 트레이딩 관리자."""

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
    ) -> None:
        self.symbol = symbol
        self.api = api_client
        self.capital = capital
        self.weight = weight
        self.margin_manager = margin_manager
        self.config_loader = config_loader
        self._params_safe_name = safe_name
        self.cooldown_hours = cooldown_hours

        # 전략 초기화
        self.strategy = DCAStrategy(params)

        # 포지션 상태
        self.long_state = PositionState(side="long")
        self.short_state = PositionState(side="short")

        # 로깅
        safe_symbol = symbol.replace("/", "_")
        self.logger = setup_logger(f"trader_{safe_symbol}", f"logs/trader_{safe_symbol}.log")
        self.trade_logger = TradeLogger()

        # 상태 관리
        self.state_manager = StateManager()

        # 현재 가격
        self._current_price: float = 0.0
        self._lock = threading.Lock()

        # 잔고 스냅샷 (PnL 계산용)
        self._last_equity: float = 0.0

        # active_params 디렉토리
        self._active_params_dir = Path("data/active_params")

        # 초기화 완료 플래그
        self._initialized = False

        # 제거 대기 플래그 (config에서 삭제된 종목)
        self._marked_for_removal = False

    def initialize(self) -> None:
        """거래소 설정 및 초기 포지션 동기화."""
        self.logger.info(f"Initializing {self.symbol}...")

        # 정전 복구: active_params 파일이 있으면 그것을 우선 사용
        active_params = self._load_active_params()
        if active_params:
            self.strategy = DCAStrategy(active_params)
            self.logger.info("Loaded params from active_params (recovery)")
        else:
            # 최초 시작: 현재 params를 active_params에 저장
            self._save_active_params(self.strategy.params)
            self.logger.info("Saved initial params to active_params")

        # Cross 마진 모드 설정
        try:
            self.api.set_margin_mode(self.symbol, "cross")
        except Exception as e:
            self.logger.warning(f"Margin mode setting: {e}")

        # 레버리지 설정
        lev = self.strategy.leverage
        try:
            self.api.set_leverage(self.symbol, lev)
        except Exception as e:
            self.logger.warning(f"Leverage setting: {e}")

        # 저장된 상태 로드
        saved_state = self.state_manager.load_state(self.symbol)
        if saved_state:
            self.long_state = saved_state["long"]
            self.short_state = saved_state["short"]
            self.logger.info("Loaded saved state")

        # 거래소 포지션과 동기화
        self._sync_with_exchange()

        # 잔고 스냅샷 초기화 (PnL 계산 기준점)
        try:
            self._last_equity = self.api.get_account_equity()
        except Exception as e:
            self.logger.warning(f"Initial equity snapshot failed: {e}")

        self._initialized = True
        self.logger.info(
            f"Initialized {self.symbol} - Capital: ${self.capital:.2f}, "
            f"Leverage: {lev}x"
        )

    def _sync_with_exchange(self) -> None:
        """거래소 포지션과 로컬 상태 동기화."""
        try:
            positions = self.api.get_positions(self.symbol)

            # 거래소 포지션 확인
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

            # 거래소에 포지션이 없는데 로컬 state가 active이면 리셋
            if exchange_long_amt == 0.0 and self.long_state.active:
                self.logger.warning("Long state was active but no exchange position — resetting")
                sl_time = self.long_state.last_sl_time
                self.long_state.reset()
                self.long_state.last_sl_time = sl_time  # 쿨다운 유지

            if exchange_short_amt == 0.0 and self.short_state.active:
                self.logger.warning("Short state was active but no exchange position — resetting")
                sl_time = self.short_state.last_sl_time
                self.short_state.reset()
                self.short_state.last_sl_time = sl_time  # 쿨다운 유지

            self.logger.info(
                f"Synced - Long: {self.long_state.amount:.4f}, "
                f"Short: {self.short_state.amount:.4f}"
            )

        except Exception as e:
            self.logger.error(f"Sync error: {e}")

    def _save_state(self) -> None:
        """현재 상태 저장."""
        try:
            self.state_manager.save_state(
                self.symbol,
                self.long_state,
                self.short_state,
                {"capital": self.capital, "current_price": self._current_price},
            )
        except Exception as e:
            self.logger.error(f"Save state error: {e}")

    def _try_update_margin(self) -> None:
        """양쪽 모두 비활성일 때 마진 업데이트 시도."""
        if self.long_state.active or self.short_state.active:
            return  # 한쪽이라도 활성이면 업데이트 안 함

        try:
            new_balance = self.api.get_account_equity()
            self.capital = self.margin_manager.try_update(
                self.symbol, self.weight, self.capital, new_balance
            )
        except Exception as e:
            self.logger.error(f"Margin update error: {e}")

        # 파라미터 업데이트도 같이 시도
        self._try_update_params()

    def _try_update_params(self) -> None:
        """양쪽 비활성 시 파라미터 변경 감지 및 교체."""
        try:
            new_params = self.config_loader.load(self._params_safe_name)
            if new_params == self.strategy.params:
                return  # 동일하면 무시

            self.strategy = DCAStrategy(new_params)
            self._save_active_params(new_params)

            # 레버리지 재설정 (변경 가능)
            lev = self.strategy.leverage
            try:
                self.api.set_leverage(self.symbol, lev)
            except Exception:
                pass

            self.logger.info(
                f"Params updated for {self.symbol} (leverage: {lev}x)"
            )
        except Exception as e:
            self.logger.error(f"Params update error: {e}")

    def _save_active_params(self, params: Dict[str, Any]) -> None:
        """현재 사용 중인 파라미터를 active_params에 저장."""
        try:
            self._active_params_dir.mkdir(parents=True, exist_ok=True)
            file_path = self._active_params_dir / f"{self._params_safe_name}.json"
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Save active params error: {e}")

    def _load_active_params(self) -> Optional[Dict[str, Any]]:
        """active_params에서 파라미터 로드 (정전 복구용)."""
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
        """시장가로 Base 포지션 진입."""
        # 제거 대기 종목은 신규 진입 안 함
        if self._marked_for_removal:
            return False

        state = self.long_state if side == "long" else self.short_state

        if state.active:
            return False

        # 쿨다운 체크
        if not self.strategy.should_enter(state, self.cooldown_hours):
            return False

        try:
            # 마진 계산
            margin = self.strategy.calculate_base_margin(self.capital, side)
            leverage = self.strategy.get_leverage(side)

            # 현재가 조회
            price = self._current_price
            if price <= 0:
                price = self.api.get_mark_price(self.symbol)

            # 수량 계산
            notional = margin * leverage
            amount = notional / price
            amount = self.api.round_amount(self.symbol, amount)

            min_amount = self.api.get_min_order_amount(self.symbol)
            if amount < min_amount:
                self.logger.warning(f"Amount {amount} < min {min_amount}")
                return False

            # 시장가 주문
            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "buy" if side == "long" else "sell"

            order = self.api.place_market_order(
                self.symbol, order_side, amount, position_side,
            )

            fill_price = float(order.get("average", 0))
            filled_amount = float(order.get("filled", 0))

            # Fallback: 응답에 체결 정보가 없으면 mark price / 주문 수량 사용
            if fill_price <= 0:
                fill_price = self.api.get_mark_price(self.symbol)
                self.logger.warning(f"No fill price in response, using mark price: {fill_price}")
            if filled_amount <= 0:
                filled_amount = amount
                self.logger.warning(f"No filled amount in response, using order amount: {filled_amount}")

            # 상태 업데이트
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

            # DCA 주문 배치
            self._place_dca_orders(side)

            # SL 주문 배치
            self._place_sl_order(side)

            # TP 주문 배치
            self._place_tp_order(side)

            # 잔고 스냅샷 갱신 (진입 후 기준점)
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
        """DCA 지정가 주문 배치."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.base_price <= 0:
            return

        # DCA 레벨 계산
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

    def _place_sl_order(self, side: str) -> None:
        """SL 주문 배치 (STOP_MARKET — mark price 트리거)."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.amount <= 0:
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

            self.logger.info(
                f"SL {side.upper()} placed - Price: {sl_price:.2f}, "
                f"Amount: {state.amount:.4f}"
            )

        except Exception as e:
            self.logger.error(f"SL order error: {e}")

    def _place_tp_order(self, side: str) -> None:
        """TP 주문 배치 (LIMIT — 정확한 가격에 체결)."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.amount <= 0:
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

            self.logger.info(
                f"TP {side.upper()} placed - Price: {tp_price:.2f}, "
                f"Amount: {state.amount:.4f}"
            )

        except Exception as e:
            self.logger.error(f"TP order error: {e}")

    def _cancel_side_orders(self, side: str) -> None:
        """한 방향의 모든 주문 취소."""
        state = self.long_state if side == "long" else self.short_state

        # DCA 주문 취소
        for dca in state.dca_orders:
            if dca.order_id:
                try:
                    self.api.cancel_order(self.symbol, dca.order_id)
                except Exception:
                    pass
        state.dca_orders = []

        # SL 취소
        if state.sl_order_id:
            try:
                self.api.cancel_order(self.symbol, state.sl_order_id)
            except Exception:
                pass
            state.sl_order_id = None

        # TP 취소
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
        """이전 포지션의 미체결 주문 취소 (TP는 이미 체결이므로 스킵)."""
        for dca in old_dca_orders:
            if dca.order_id:
                try:
                    self.api.cancel_order(self.symbol, dca.order_id)
                except Exception:
                    pass
        if old_sl_order_id:
            try:
                self.api.cancel_order(self.symbol, old_sl_order_id)
            except Exception:
                pass

    def _try_update_margin_with_equity(self, equity: float) -> None:
        """equity를 외부에서 받아 마진 업데이트 (REST 중복 호출 방지)."""
        if self.long_state.active or self.short_state.active:
            return
        if equity <= 0:
            return
        try:
            self.capital = self.margin_manager.try_update(
                self.symbol, self.weight, self.capital, equity
            )
        except Exception as e:
            self.logger.error(f"Margin update error: {e}")
        self._try_update_params()

    def _handle_tp(self, side: str) -> None:
        """TP 체결 처리 — 재진입 우선, 정리 후처리."""
        state = self.long_state if side == "long" else self.short_state

        # ── ① 즉시 재진입 (최우선) ──
        # old order ID 임시 보관 후 reset → 바로 시장가 진입
        old_amount = state.amount
        old_dca_orders = state.dca_orders
        old_sl_order_id = state.sl_order_id
        state.reset()

        self._enter_position(side)

        # ── ② 이전 주문 정리 (시장가 전송 후) ──
        self._cancel_old_orders(old_dca_orders, old_sl_order_id)

        # ── ③ PnL 계산 (1회만 호출) ──
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

        # ── ④ 마진 업데이트 (equity 재사용, 중복 호출 제거) ──
        self._try_update_margin_with_equity(new_equity)

        self._save_state()

    def _handle_sl(self, side: str) -> None:
        """SL 체결 처리."""
        state = self.long_state if side == "long" else self.short_state

        # SL은 이미 체결 → old DCA + TP만 취소 (SL 스킵)
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

        # 잔고 기반 PnL 계산 (1회만)
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

        # 상태 리셋 (쿨다운 유지)
        state.reset()
        state.last_sl_time = datetime.now(timezone.utc)

        # 마진 업데이트 (equity 재사용)
        self._try_update_margin_with_equity(new_equity)

        self._save_state()

    def _check_and_handle_dca_fills(self, side: str) -> None:
        """DCA 체결 확인 및 처리 (폴링 백업용)."""
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
        """DCA 체결 처리 — avg 재계산, SL/TP 재배치."""
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

        # SL 취소 → 재배치 (같은 base_price, 증가된 amount)
        if state.sl_order_id:
            try:
                self.api.cancel_order(self.symbol, state.sl_order_id)
            except Exception:
                pass
            state.sl_order_id = None
        self._place_sl_order(side)

        # TP 취소 → 재배치 (새 avg_price, 증가된 amount)
        if state.tp_order_id:
            try:
                self.api.cancel_order(self.symbol, state.tp_order_id)
            except Exception:
                pass
            state.tp_order_id = None
        self._place_tp_order(side)

    def on_order_filled(self, order_id: str, data: Dict[str, Any]) -> None:
        """User Data Stream에서 주문 체결 감지."""
        with self._lock:
            for side in ["long", "short"]:
                state = self.long_state if side == "long" else self.short_state

                # TP 체결 확인
                if state.tp_order_id == order_id:
                    self._handle_tp(side)
                    return

                # SL 체결 확인
                if state.sl_order_id == order_id:
                    self._handle_sl(side)
                    return

                # DCA 체결 확인
                for dca in list(state.dca_orders):
                    if dca.order_id == order_id:
                        state.dca_orders.remove(dca)
                        self._process_dca_fill(side, dca)
                        self._save_state()
                        return

    def on_price_update(self, price: float) -> None:
        """가격 업데이트 콜백."""
        with self._lock:
            self._current_price = price

        if not self._initialized:
            return

        # Long 포지션 처리
        if self.long_state.active:
            if self.strategy.check_tp_hit(price, self.long_state):
                self._handle_tp("long")
            elif self.strategy.check_sl_hit(price, self.long_state):
                self._handle_sl("long")
        else:
            if self.strategy.should_enter(self.long_state, self.cooldown_hours):
                self._enter_position("long")

        # Short 포지션 처리
        if self.short_state.active:
            if self.strategy.check_tp_hit(price, self.short_state):
                self._handle_tp("short")
            elif self.strategy.check_sl_hit(price, self.short_state):
                self._handle_sl("short")
        else:
            if self.strategy.should_enter(self.short_state, self.cooldown_hours):
                self._enter_position("short")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 요약."""
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
        """종료 처리."""
        self._save_state()
        self.logger.info(f"Shutdown {self.symbol}")


class TradingExecutor:
    """메인 트레이딩 실행기."""

    def __init__(
        self,
        config: TradingConfig,
        testnet: bool = True,
        config_path: str = "config.json",
    ) -> None:
        self.config = config
        self.testnet = testnet

        self.logger = setup_logger("executor", "logs/executor.log")
        self.config_loader = ConfigLoader()

        # API 클라이언트
        self.api = APIClient(testnet=testnet)

        # 마진 관리자
        self.margin_manager = MarginManager()

        # 심볼 목록
        self.symbols = config.get_symbol_names()

        if not self.symbols:
            raise ValueError("No symbols to trade")

        # 심볼별 트레이더
        self.traders: Dict[str, SymbolTrader] = {}

        # WebSocket 피드
        self.price_feed: Optional[PriceFeed] = None
        self.order_feed: Optional[OrderUpdateFeed] = None
        self._listen_key: Optional[str] = None
        self._listen_key_timer: Optional[threading.Timer] = None

        # 실행 상태
        self._running = False
        self._shutdown_event = threading.Event()

        # config.json 핫 리로드
        self._config_path: str = config_path
        self._config_mtime: float = 0.0
        self._pending_removals: Set[str] = set()  # 제거 대기 종목 (symbol 형식)

    @staticmethod
    def _raw_to_symbol(raw_symbol: str) -> str:
        """BTCUSDT → BTC/USDT 변환."""
        # 일반적인 USDT 쌍 처리
        if raw_symbol.endswith("USDT"):
            base = raw_symbol[:-4]
            return f"{base}/USDT"
        return raw_symbol

    def _on_price_update(self, symbol: str, price: float) -> None:
        """가격 업데이트 콜백 (PriceFeed에서 호출)."""
        if symbol in self.traders:
            self.traders[symbol].on_price_update(price)

    def _on_order_update(self, data: Dict[str, Any]) -> None:
        """주문 업데이트 콜백 (OrderUpdateFeed에서 호출)."""
        raw_symbol = data.get("symbol", "")
        symbol = self._raw_to_symbol(raw_symbol)
        status = data.get("status", "")
        order_id = str(data.get("order_id", ""))

        if symbol in self.traders and status == "FILLED":
            self.logger.info(f"Order filled: {symbol} #{order_id}")
            self.traders[symbol].on_order_filled(order_id, data)

    def _on_position_update(self, data: Dict[str, Any]) -> None:
        """포지션 업데이트 콜백 (로깅용)."""
        self.logger.debug(f"Position update: {data}")

    # ---- config.json 핫 리로드 ----

    def _check_config_update(self) -> None:
        """config.json 변경 감지 및 처리 (mtime 비교)."""
        try:
            current_mtime = os.path.getmtime(self._config_path)
        except OSError:
            return  # 파일 접근 불가 시 무시

        if current_mtime == self._config_mtime:
            return  # 변경 없음

        self._config_mtime = current_mtime
        self.logger.info("config.json change detected, reloading...")

        try:
            new_config = TradingConfig.load(self._config_path)
        except Exception as e:
            self.logger.error(f"Config reload failed: {e}")
            return

        # 현재 vs 새 심볼 비교
        old_symbols = set(self.config.symbols.keys())  # safe_name: "BTC_USDT"
        new_symbols = set(new_config.symbols.keys())

        added = new_symbols - old_symbols
        removed = old_symbols - new_symbols
        common = old_symbols & new_symbols

        # cooldown_hours 변경 반영
        if new_config.cooldown_hours != self.config.cooldown_hours:
            self.logger.info(
                f"Cooldown changed: {self.config.cooldown_hours}h → "
                f"{new_config.cooldown_hours}h"
            )
            for trader in self.traders.values():
                trader.cooldown_hours = new_config.cooldown_hours

        # weight 변경 감지
        for safe_name in common:
            old_weight = self.config.symbols[safe_name].weight
            new_weight = new_config.symbols[safe_name].weight
            if old_weight != new_weight:
                symbol = new_config.symbols[safe_name].symbol
                if symbol in self.traders:
                    self.traders[symbol].weight = new_weight
                    self.logger.info(
                        f"{symbol} weight changed: {old_weight:.4f} → {new_weight:.4f}"
                    )

        # 종목 추가
        need_reconnect = False
        for safe_name in added:
            sym_cfg = new_config.symbols[safe_name]
            self.logger.info(f"Adding new symbol: {sym_cfg.symbol}")
            if self._add_trader(safe_name, sym_cfg):
                need_reconnect = True

        # 종목 제거 — 즉시 종료 아닌 제거 대기 등록
        for safe_name in removed:
            symbol = self.config.symbols[safe_name].symbol
            if symbol in self.traders:
                self._mark_trader_for_removal(symbol)

        # config 객체 교체
        self.config = new_config

        # PriceFeed 재연결 (심볼 추가 시만)
        if need_reconnect:
            self._reconnect_price_feed()

    def _add_trader(self, safe_name: str, sym_cfg: Any) -> bool:
        """런타임에 새 트레이더 추가. 성공 시 True 반환."""
        symbol = sym_cfg.symbol

        if symbol in self.traders:
            self.logger.warning(f"{symbol} already exists, skipping add")
            return False

        try:
            # 파라미터 로드
            params = self.config_loader.load(safe_name)

            # 마진 초기화
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
            )
            trader.initialize()
            self.traders[symbol] = trader

            self.logger.info(f"Trader added: {symbol} (${capital:.2f})")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add trader {symbol}: {e}")
            return False

    def _mark_trader_for_removal(self, symbol: str) -> None:
        """종목을 제거 대기로 등록 (신규 진입 차단)."""
        if symbol not in self.traders:
            return

        self._pending_removals.add(symbol)
        self.traders[symbol]._marked_for_removal = True
        self.logger.info(
            f"{symbol} marked for removal (waiting for positions to close)"
        )

    def _remove_trader(self, symbol: str) -> None:
        """트레이더 완전 제거 (양쪽 비활성 확인 후 호출)."""
        trader = self.traders.get(symbol)
        if not trader:
            return

        # state 저장 + 종료
        trader.shutdown()

        # 미체결 주문 전부 취소
        try:
            self.api.cancel_all_orders(symbol)
        except Exception as e:
            self.logger.error(f"Cancel orders error for {symbol}: {e}")

        del self.traders[symbol]
        self._pending_removals.discard(symbol)

        self.logger.info(f"Trader removed: {symbol}")

        # PriceFeed 재연결 (구독 심볼 변경)
        self._reconnect_price_feed()

    def _process_pending_removals(self) -> None:
        """제거 대기 종목 중 양쪽 비활성인 것을 실제 제거."""
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
        """현재 traders 기준으로 PriceFeed 재연결."""
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
        """Listen Key 30분마다 갱신."""
        if not self._running or not self._listen_key:
            return
        try:
            self.api.renew_listen_key(self._listen_key)
            self.logger.debug("Listen key renewed")
        except Exception as e:
            self.logger.error(f"Listen key renewal failed: {e}")
            # 새 키 생성 시도
            try:
                self._listen_key = self.api.new_listen_key()
                self.logger.info("New listen key created")
            except Exception as e2:
                self.logger.error(f"New listen key failed: {e2}")

        # 다음 갱신 예약 (30분)
        if self._running:
            self._listen_key_timer = threading.Timer(
                30 * 60, self._start_listen_key_renewal
            )
            self._listen_key_timer.daemon = True
            self._listen_key_timer.start()

    def _setup_signal_handlers(self) -> None:
        """시그널 핸들러 설정."""
        def handle_shutdown(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

    def run(self) -> None:
        """메인 트레이딩 루프 시작."""
        self.logger.info("=" * 50)
        self.logger.info("Starting Trading Executor")
        self.logger.info(f"Testnet: {self.testnet}")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info("=" * 50)

        self._running = True
        self._setup_signal_handlers()

        try:
            # Hedge Mode 설정
            self.api.set_position_mode(hedge_mode=True)

            # Binance에서 총 잔고 조회
            total_balance = self.api.get_account_equity()
            self.logger.info(f"Total wallet balance: ${total_balance:.2f}")

            # 심볼별 자본 배분 표시
            for safe_name, sym_cfg in self.config.symbols.items():
                cap = total_balance * sym_cfg.weight
                self.logger.info(
                    f"  {sym_cfg.symbol}: weight={sym_cfg.weight:.4f}, "
                    f"capital=${cap:.2f}"
                )

            # 트레이더 초기화
            for safe_name, sym_cfg in self.config.symbols.items():
                symbol = sym_cfg.symbol
                try:
                    # 파라미터 로드
                    params = self.config_loader.load(safe_name)

                    # 마진 로드 또는 초기화
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
                    )
                    trader.initialize()
                    self.traders[symbol] = trader

                    self.logger.info(f"Trader initialized: {symbol} (${capital:.2f})")

                except Exception as e:
                    self.logger.error(f"Failed to init {symbol}: {e}")

            if not self.traders:
                raise RuntimeError("No traders initialized")

            # 가격 피드 시작
            self.price_feed = PriceFeed(
                symbols=list(self.traders.keys()),
                on_price_update=self._on_price_update,
                testnet=self.testnet,
            )
            self.price_feed.start()

            # User Data Stream 시작
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

            # config.json 초기 mtime 저장
            try:
                self._config_mtime = os.path.getmtime(self._config_path)
            except OSError:
                self._config_mtime = 0.0

            # 메인 루프 (상태 모니터링 + 폴링 백업)
            poll_counter = 0
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=60)

                if not self._shutdown_event.is_set():
                    poll_counter += 1
                    self._log_status()

                    # 제거 대기 종목 처리 (매 60초)
                    if self._pending_removals:
                        self._process_pending_removals()

                    # 5분마다: config 변경 감지 + DCA 폴링 백업
                    if poll_counter % 5 == 0:
                        self._check_config_update()

                        for trader in self.traders.values():
                            trader._check_and_handle_dca_fills("long")
                            trader._check_and_handle_dca_fills("short")

        except Exception as e:
            self.logger.error(f"Executor error: {e}")
            raise

        finally:
            self.shutdown()

    def _log_status(self) -> None:
        """현재 상태 로깅."""
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
        """안전한 종료."""
        self.logger.info("Shutting down...")
        self._running = False

        # Listen Key 갱신 타이머 종료
        if self._listen_key_timer:
            self._listen_key_timer.cancel()

        # User Data Stream 종료
        if self.order_feed:
            self.order_feed.stop()

        # Listen Key 삭제
        if self._listen_key:
            try:
                self.api.close_listen_key(self._listen_key)
            except Exception:
                pass

        # 가격 피드 종료
        if self.price_feed:
            self.price_feed.stop()

        # 트레이더 종료
        for trader in self.traders.values():
            trader.shutdown()

        self.logger.info("Shutdown complete")
