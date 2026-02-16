"""주문 체결 및 잔고 관리."""

import os
import signal
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.common.api_client import APIClient
from src.common.config_loader import ConfigLoader
from src.common.logger import setup_logger
from src.trading.price_feed import PriceFeed
from src.trading.state_manager import StateManager, TradeLogger
from src.trading.strategy import DCAStrategy, DCALevel, PositionState

load_dotenv()


class SymbolTrader:
    """심볼별 트레이딩 관리자."""

    def __init__(
        self,
        symbol: str,
        params: Dict[str, Any],
        api_client: APIClient,
        capital: float,
        cooldown_hours: int = 6,
        fee_rate: float = 0.0005,
    ) -> None:
        """
        SymbolTrader 초기화.

        Args:
            symbol: 거래 심볼 (예: "BTC/USDT")
            params: 최적화된 파라미터
            api_client: API 클라이언트
            capital: 이 심볼에 할당된 자본
            cooldown_hours: SL 후 쿨다운 시간
            fee_rate: 수수료율
        """
        self.symbol = symbol
        self.api = api_client
        self.capital = capital
        self.cooldown_hours = cooldown_hours
        self.fee_rate = fee_rate

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

        # 초기화 완료 플래그
        self._initialized = False

    def initialize(self) -> None:
        """거래소 설정 및 초기 포지션 동기화."""
        self.logger.info(f"Initializing {self.symbol}...")

        # 레버리지 설정
        long_lev = self.strategy.get_leverage("long")
        short_lev = self.strategy.get_leverage("short")

        try:
            self.api.set_margin_mode(self.symbol, "isolated")
        except Exception as e:
            self.logger.warning(f"Margin mode setting: {e}")

        try:
            self.api.set_leverage(self.symbol, long_lev)
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

        self._initialized = True
        self.logger.info(f"Initialized {self.symbol}")

    def _sync_with_exchange(self) -> None:
        """거래소 포지션과 로컬 상태 동기화."""
        try:
            positions = self.api.get_positions(self.symbol)

            for pos in positions:
                side = pos.get("side", "").lower()
                amount = float(pos.get("contracts", 0))
                entry_price = float(pos.get("entryPrice", 0))

                if side == "long" and amount > 0:
                    self.long_state.active = True
                    self.long_state.amount = amount
                    if entry_price > 0:
                        self.long_state.avg_price = entry_price
                        if self.long_state.base_price == 0:
                            self.long_state.base_price = entry_price

                elif side == "short" and amount > 0:
                    self.short_state.active = True
                    self.short_state.amount = amount
                    if entry_price > 0:
                        self.short_state.avg_price = entry_price
                        if self.short_state.base_price == 0:
                            self.short_state.base_price = entry_price

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

    def _enter_position(self, side: str) -> bool:
        """
        시장가로 Base 포지션 진입.

        Args:
            side: "long" 또는 "short"

        Returns:
            성공 여부
        """
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
                self.symbol,
                order_side,
                amount,
                position_side,
            )

            fill_price = float(order.get("average", price))
            filled_amount = float(order.get("filled", amount))

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
                    self.symbol,
                    order_side,
                    amount,
                    price,
                    position_side,
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
        """SL 주문 배치."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.amount <= 0:
            return

        try:
            sl_price = self.strategy.calculate_sl_price(state.base_price, side)
            sl_price = self.api.round_price(self.symbol, sl_price)

            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "sell" if side == "long" else "buy"

            order = self.api.place_stop_loss(
                self.symbol,
                order_side,
                state.amount,
                sl_price,
                position_side,
            )

            state.sl_order_id = str(order.get("id"))

            self.logger.info(
                f"SL {side.upper()} placed - Price: {sl_price:.2f}, "
                f"Amount: {state.amount:.4f}"
            )

        except Exception as e:
            self.logger.error(f"SL order error: {e}")

    def _place_tp_order(self, side: str) -> None:
        """TP 주문 배치."""
        state = self.long_state if side == "long" else self.short_state

        if not state.active or state.amount <= 0:
            return

        try:
            tp_price = self.strategy.calculate_tp_price(state.avg_price, side)
            tp_price = self.api.round_price(self.symbol, tp_price)

            position_side = "LONG" if side == "long" else "SHORT"
            order_side = "sell" if side == "long" else "buy"

            order = self.api.place_take_profit(
                self.symbol,
                order_side,
                state.amount,
                tp_price,
                position_side,
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

    def _handle_tp(self, side: str) -> None:
        """TP 체결 처리."""
        state = self.long_state if side == "long" else self.short_state

        pnl = self.strategy.estimate_pnl(self._current_price, state, self.fee_rate)

        self.logger.info(
            f"TP HIT {side.upper()} - Price: {self._current_price:.2f}, "
            f"PnL: {pnl:.2f}"
        )
        self.trade_logger.log_tp(
            self.symbol, side, self._current_price, state.amount, pnl
        )

        # 주문 취소 및 상태 리셋
        self._cancel_side_orders(side)
        state.reset()

        # 즉시 재진입
        self._enter_position(side)

    def _handle_sl(self, side: str) -> None:
        """SL 체결 처리."""
        state = self.long_state if side == "long" else self.short_state

        pnl = self.strategy.estimate_pnl(self._current_price, state, self.fee_rate)

        self.logger.info(
            f"SL HIT {side.upper()} - Price: {self._current_price:.2f}, "
            f"PnL: {pnl:.2f}"
        )
        self.trade_logger.log_sl(
            self.symbol, side, self._current_price, state.amount, pnl
        )

        # 주문 취소 및 상태 리셋
        self._cancel_side_orders(side)
        state.last_sl_time = datetime.now(timezone.utc)
        state.reset()
        state.last_sl_time = datetime.now(timezone.utc)  # reset 후 다시 설정

        self._save_state()

    def _check_and_handle_dca_fills(self, side: str) -> None:
        """DCA 체결 확인 및 처리."""
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
                    # 주문이 없음 = 체결됨
                    filled_dcas.append(dca)
                else:
                    remaining_dcas.append(dca)

            for dca in filled_dcas:
                # 상태 업데이트
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
                    f"Price: {dca.trigger_price:.2f}, New Avg: {new_avg:.2f}"
                )
                self.trade_logger.log_dca(
                    self.symbol, side, dca.level,
                    dca.trigger_price, add_amount, dca.margin, new_avg
                )

                # SL/TP 업데이트 (avg 변경으로 인해)
                if state.sl_order_id:
                    try:
                        self.api.cancel_order(self.symbol, state.sl_order_id)
                    except Exception:
                        pass
                    state.sl_order_id = None

                # 새 SL 배치 (수량 증가)
                self._place_sl_order(side)

            state.dca_orders = remaining_dcas
            self._save_state()

        except Exception as e:
            self.logger.error(f"DCA check error: {e}")

    def on_price_update(self, price: float) -> None:
        """
        가격 업데이트 콜백.

        Args:
            price: 새로운 마크 가격
        """
        with self._lock:
            self._current_price = price

        if not self._initialized:
            return

        # Long 포지션 처리
        if self.long_state.active:
            # TP 체크
            if self.strategy.check_tp_hit(price, self.long_state):
                self._handle_tp("long")
            # SL 체크 (거래소 주문으로 처리되지만 백업용)
            elif self.strategy.check_sl_hit(price, self.long_state):
                self._handle_sl("long")
            # DCA 체결 확인
            else:
                self._check_and_handle_dca_fills("long")
        else:
            # 신규 진입 시도
            if self.strategy.should_enter(self.long_state, self.cooldown_hours):
                self._enter_position("long")

        # Short 포지션 처리
        if self.short_state.active:
            if self.strategy.check_tp_hit(price, self.short_state):
                self._handle_tp("short")
            elif self.strategy.check_sl_hit(price, self.short_state):
                self._handle_sl("short")
            else:
                self._check_and_handle_dca_fills("short")
        else:
            if self.strategy.should_enter(self.short_state, self.cooldown_hours):
                self._enter_position("short")

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 요약."""
        return {
            "symbol": self.symbol,
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
        symbols: Optional[List[str]] = None,
        testnet: bool = True,
        capital: float = 1000.0,
        cooldown_hours: int = 6,
    ) -> None:
        """
        TradingExecutor 초기화.

        Args:
            symbols: 거래할 심볼 리스트 (None이면 params/ 폴더에서 로드)
            testnet: Testnet 사용 여부
            capital: 총 자본
            cooldown_hours: SL 후 쿨다운 시간
        """
        self.testnet = testnet
        self.capital = capital
        self.cooldown_hours = cooldown_hours

        self.logger = setup_logger("executor", "logs/executor.log")
        self.config_loader = ConfigLoader()

        # API 클라이언트
        self.api = APIClient(testnet=testnet)

        # 심볼 결정
        if symbols:
            self.symbols = symbols
        else:
            # params/ 폴더에서 심볼 목록 가져오기
            self.symbols = self._discover_symbols()

        if not self.symbols:
            raise ValueError("No symbols to trade")

        # 심볼별 트레이더
        self.traders: Dict[str, SymbolTrader] = {}

        # WebSocket 가격 피드
        self.price_feed: Optional[PriceFeed] = None

        # 실행 상태
        self._running = False
        self._shutdown_event = threading.Event()

    def _discover_symbols(self) -> List[str]:
        """params/ 폴더에서 심볼 목록 발견."""
        from pathlib import Path

        params_dir = Path("params")
        if not params_dir.exists():
            return []

        symbols = []
        for f in params_dir.glob("*.json"):
            # BTC_USDT.json -> BTC/USDT
            symbol = f.stem.replace("_", "/")
            symbols.append(symbol)

        return symbols

    def _on_price_update(self, symbol: str, price: float) -> None:
        """가격 업데이트 콜백 (PriceFeed에서 호출)."""
        if symbol in self.traders:
            self.traders[symbol].on_price_update(price)

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
        self.logger.info(f"Capital: ${self.capital}")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info("=" * 50)

        self._running = True
        self._setup_signal_handlers()

        try:
            # Hedge Mode 설정
            self.api.set_position_mode(hedge_mode=True)

            # 잔고 확인
            balance = self.api.get_balance()
            self.logger.info(f"Available balance: ${balance:.2f}")

            if balance < self.capital * 0.1:
                self.logger.warning(f"Low balance: ${balance:.2f}")

            # 심볼별 자본 분배
            capital_per_symbol = self.capital / len(self.symbols)

            # 트레이더 초기화
            for symbol in self.symbols:
                try:
                    # 파라미터 로드
                    safe_symbol = symbol.replace("/", "_")
                    params = self.config_loader.load(safe_symbol)

                    trader = SymbolTrader(
                        symbol=symbol,
                        params=params,
                        api_client=self.api,
                        capital=capital_per_symbol,
                        cooldown_hours=self.cooldown_hours,
                    )
                    trader.initialize()
                    self.traders[symbol] = trader

                    self.logger.info(f"Trader initialized: {symbol}")

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

            self.logger.info("Trading started")

            # 메인 루프 (상태 모니터링)
            while not self._shutdown_event.is_set():
                self._shutdown_event.wait(timeout=60)

                # 주기적 상태 로깅
                if not self._shutdown_event.is_set():
                    self._log_status()

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
                f"{symbol} - Price: {status['current_price']:.2f} | "
                f"Long: {status['long']['amount']:.4f} @ {status['long']['avg_price']:.2f} "
                f"(DCA: {status['long']['dca_count']}) | "
                f"Short: {status['short']['amount']:.4f} @ {status['short']['avg_price']:.2f} "
                f"(DCA: {status['short']['dca_count']})"
            )

    def shutdown(self) -> None:
        """안전한 종료."""
        self.logger.info("Shutting down...")
        self._running = False

        # 가격 피드 종료
        if self.price_feed:
            self.price_feed.stop()

        # 트레이더 종료
        for trader in self.traders.values():
            trader.shutdown()

        self.logger.info("Shutdown complete")
