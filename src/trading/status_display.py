"""터미널 상태 디스플레이 (In-Place Update)."""

import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Box drawing characters
_TL = "╔"
_TR = "╗"
_BL = "╚"
_BR = "╝"
_H = "═"
_V = "║"
_ML = "╠"
_MR = "╣"

# ANSI 컬러
_GREEN = "\x1b[32m"
_RED = "\x1b[31m"
_YELLOW = "\x1b[33m"
_DIM = "\x1b[90m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class SymbolSnapshot:
    """디스플레이용 심볼 스냅샷.

    executor(SymbolTrader) 또는 state 파일에서 생성 가능.
    """

    def __init__(
        self,
        symbol: str,
        capital: float,
        current_price: float,
        long_active: bool,
        long_amount: float,
        long_avg_price: float,
        long_dca_count: int,
        long_max_dca: int,
        long_tp_price: float,
        long_last_sl_time: Optional[datetime],
        short_active: bool,
        short_amount: float,
        short_avg_price: float,
        short_dca_count: int,
        short_max_dca: int,
        short_tp_price: float,
        short_last_sl_time: Optional[datetime],
        cooldown_hours: int = 6,
    ) -> None:
        self.symbol = symbol
        self.capital = capital
        self.current_price = current_price

        self.long_active = long_active
        self.long_amount = long_amount
        self.long_avg_price = long_avg_price
        self.long_dca_count = long_dca_count
        self.long_max_dca = long_max_dca
        self.long_tp_price = long_tp_price
        self.long_last_sl_time = long_last_sl_time

        self.short_active = short_active
        self.short_amount = short_amount
        self.short_avg_price = short_avg_price
        self.short_dca_count = short_dca_count
        self.short_max_dca = short_max_dca
        self.short_tp_price = short_tp_price
        self.short_last_sl_time = short_last_sl_time

        self.cooldown_hours = cooldown_hours

    @classmethod
    def from_trader(cls, trader: Any) -> "SymbolSnapshot":
        """SymbolTrader 객체에서 생성."""
        strategy = trader.strategy
        long_tp = 0.0
        if trader.long_state.active and trader.long_state.avg_price > 0:
            try:
                long_tp = strategy.calculate_tp_price(
                    trader.long_state.avg_price, "long"
                )
            except Exception:
                pass

        short_tp = 0.0
        if trader.short_state.active and trader.short_state.avg_price > 0:
            try:
                short_tp = strategy.calculate_tp_price(
                    trader.short_state.avg_price, "short"
                )
            except Exception:
                pass

        return cls(
            symbol=trader.symbol,
            capital=trader.capital,
            current_price=trader._current_price,
            long_active=trader.long_state.active,
            long_amount=trader.long_state.amount,
            long_avg_price=trader.long_state.avg_price,
            long_dca_count=trader.long_state.dca_count,
            long_max_dca=int(strategy.long_params.get("max_dca", 0)),
            long_tp_price=long_tp,
            long_last_sl_time=trader.long_state.last_sl_time,
            short_active=trader.short_state.active,
            short_amount=trader.short_state.amount,
            short_avg_price=trader.short_state.avg_price,
            short_dca_count=trader.short_state.dca_count,
            short_max_dca=int(strategy.short_params.get("max_dca", 0)),
            short_tp_price=short_tp,
            short_last_sl_time=trader.short_state.last_sl_time,
            cooldown_hours=trader.cooldown_hours,
        )

    @classmethod
    def from_state_files(
        cls,
        symbol: str,
        state_data: Dict[str, Any],
        params_data: Dict[str, Any],
        margin_data: Dict[str, Any],
        cooldown_hours: int = 6,
    ) -> "SymbolSnapshot":
        """state/params/margin 파일 데이터에서 생성."""
        long_data = state_data.get("long", {})
        short_data = state_data.get("short", {})
        extra = state_data.get("extra", {})

        # 파라미터에서 max_dca, TP ratio 추출
        parameters = params_data.get("parameters", {})
        long_params = parameters.get("long", {})
        short_params = parameters.get("short", {})

        long_max_dca = int(long_params.get("max_dca", 0))
        short_max_dca = int(short_params.get("max_dca", 0))

        # TP 가격 계산
        long_avg = float(long_data.get("avg_price", 0))
        long_tp = 0.0
        if long_data.get("active") and long_avg > 0:
            tp_ratio = long_params.get("take_profit", 0)
            long_tp = long_avg * (1.0 + tp_ratio)

        short_avg = float(short_data.get("avg_price", 0))
        short_tp = 0.0
        if short_data.get("active") and short_avg > 0:
            tp_ratio = short_params.get("take_profit", 0)
            short_tp = short_avg * (1.0 - tp_ratio)

        # last_sl_time 파싱
        long_sl_time = None
        if long_data.get("last_sl_time"):
            try:
                long_sl_time = datetime.fromisoformat(long_data["last_sl_time"])
            except Exception:
                pass

        short_sl_time = None
        if short_data.get("last_sl_time"):
            try:
                short_sl_time = datetime.fromisoformat(short_data["last_sl_time"])
            except Exception:
                pass

        return cls(
            symbol=symbol,
            capital=float(margin_data.get("capital", extra.get("capital", 0))),
            current_price=float(extra.get("current_price", 0)),
            long_active=bool(long_data.get("active", False)),
            long_amount=float(long_data.get("amount", 0)),
            long_avg_price=long_avg,
            long_dca_count=int(long_data.get("dca_count", 0)),
            long_max_dca=long_max_dca,
            long_tp_price=long_tp,
            long_last_sl_time=long_sl_time,
            short_active=bool(short_data.get("active", False)),
            short_amount=float(short_data.get("amount", 0)),
            short_avg_price=short_avg,
            short_dca_count=int(short_data.get("dca_count", 0)),
            short_max_dca=short_max_dca,
            short_tp_price=short_tp,
            short_last_sl_time=short_sl_time,
            cooldown_hours=cooldown_hours,
        )


class StatusDisplay:
    """ANSI escape 기반 터미널 상태 표시.

    - TTY인 경우에만 화면 클리어 + 재그리기
    - PM2 등 non-TTY 환경에서는 아무것도 출력하지 않음
    """

    WIDTH = 62  # 내부 폭 (border 제외)

    def __init__(self, force_tty: bool = False) -> None:
        self._is_tty = force_tty or sys.stdout.isatty()
        self._start_time = time.time()

    def _format_uptime(self) -> str:
        """경과 시간 포맷 (HH:MM:SS)."""
        elapsed = int(time.time() - self._start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _hline(self, left: str, right: str) -> str:
        """수평선 생성."""
        return f"{left}{_H * (self.WIDTH + 2)}{right}"

    def _row(self, text: str) -> str:
        """테이블 행 생성 (패딩 포함)."""
        display_len = self._visible_len(text)
        pad = self.WIDTH - display_len
        if pad < 0:
            pad = 0
        return f"{_V} {text}{' ' * pad} {_V}"

    @staticmethod
    def _fmt_price(value: float, sig: int = 5) -> str:
        """유효숫자 sig자리 이상으로 가격 포맷 (콤마 포함)."""
        if value == 0:
            return "0." + "0" * (sig - 1)
        import math
        magnitude = math.floor(math.log10(abs(value)))
        decimals = max(sig - 1 - magnitude, 0)
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _visible_len(text: str) -> int:
        """ANSI escape 코드를 제외한 표시 길이."""
        clean = _ANSI_RE.sub("", text)
        length = 0
        for ch in clean:
            if "\u1100" <= ch <= "\uffdc" or "\uffe0" <= ch <= "\uffe6":
                length += 2
            else:
                length += 1
        return length

    def _format_side(
        self,
        side: str,
        active: bool,
        amount: float,
        avg_price: float,
        dca_count: int,
        max_dca: int,
        tp_price: float,
        last_sl_time: Optional[datetime],
        cooldown_hours: int,
    ) -> str:
        """한 방향(Long/Short) 상태 문자열."""
        label = "LONG " if side == "long" else "SHORT"
        arrow = "▲" if side == "long" else "▼"

        if not active:
            # 비활성: 쿨다운 확인
            if last_sl_time:
                now = datetime.now(timezone.utc)
                elapsed_h = (now - last_sl_time).total_seconds() / 3600
                remaining = cooldown_hours - elapsed_h
                if remaining > 0:
                    rm = int(remaining * 60)
                    rh, rm2 = divmod(rm, 60)
                    return (
                        f"  {label} {arrow}  "
                        f"{_YELLOW}── 대기 (쿨다운 {rh}:{rm2:02d}) ──{_RESET}"
                    )

            return f"  {label} {arrow}  {_DIM}── 대기 중 ──{_RESET}"

        # 활성 포지션
        color = _GREEN if side == "long" else _RED

        tp_str = ""
        if tp_price > 0:
            tp_str = f"TP {self._fmt_price(tp_price)}"

        line = (
            f"  {label} {arrow}  "
            f"{color}{amount:.4f} @ {self._fmt_price(avg_price)}{_RESET}  "
            f"DCA {dca_count}/{max_dca}"
        )
        if tp_str:
            line += f"  {tp_str}"

        return line

    def update(
        self,
        snapshots: List[SymbolSnapshot],
        testnet: bool,
        account_equity: Optional[float] = None,
    ) -> None:
        """화면을 클리어하고 상태를 다시 그린다.

        Args:
            snapshots: 심볼별 스냅샷 리스트
            testnet: 테스트넷 여부
            account_equity: Binance 실제 계좌 잔고 (None이면 표시 안 함)
        """
        if not self._is_tty:
            return

        uptime = self._format_uptime()
        now_str = datetime.now().strftime("%H:%M:%S")
        network = f"{_YELLOW}TESTNET{_RESET}" if testnet else f"{_RED}MAINNET{_RESET}"

        lines = []

        # ── 헤더 ──
        lines.append(self._hline(_TL, _TR))
        header = f"DCA Trading Bot       {network}     Uptime: {uptime}"
        lines.append(self._row(header))
        lines.append(self._hline(_ML, _MR))

        # ── 심볼별 상태 ──
        active_count = 0
        total_positions = 0

        for snap in snapshots:
            # 심볼 헤더
            price_str = f"${self._fmt_price(snap.current_price)}" if snap.current_price > 0 else "$---"
            cap_str = f"Capital: ${snap.capital:,.2f}"
            sym_line = f"{_BOLD}{snap.symbol}{_RESET}   {price_str}"
            visible_sym = self._visible_len(sym_line)
            visible_cap = self._visible_len(cap_str)
            gap = self.WIDTH - visible_sym - visible_cap
            if gap < 1:
                gap = 1
            lines.append(self._row(f"{sym_line}{' ' * gap}{cap_str}"))

            # Long
            lines.append(self._row(self._format_side(
                "long",
                snap.long_active,
                snap.long_amount,
                snap.long_avg_price,
                snap.long_dca_count,
                snap.long_max_dca,
                snap.long_tp_price,
                snap.long_last_sl_time,
                snap.cooldown_hours,
            )))
            if snap.long_active:
                active_count += 1
            total_positions += 1

            # Short
            lines.append(self._row(self._format_side(
                "short",
                snap.short_active,
                snap.short_amount,
                snap.short_avg_price,
                snap.short_dca_count,
                snap.short_max_dca,
                snap.short_tp_price,
                snap.short_last_sl_time,
                snap.cooldown_hours,
            )))
            if snap.short_active:
                active_count += 1
            total_positions += 1

            # 빈 줄 (심볼 구분)
            lines.append(self._row(""))

        # ── 푸터 ──
        lines.append(self._hline(_ML, _MR))
        equity_str = f"${account_equity:,.2f}" if account_equity is not None else "$---"
        footer = (
            f"Equity: {equity_str}  |  "
            f"Active: {active_count}/{total_positions}  |  "
            f"{now_str}"
        )
        lines.append(self._row(footer))
        lines.append(self._hline(_BL, _BR))

        # ── 화면 클리어 + 출력 ──
        output = "\033[H\033[J" + "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
