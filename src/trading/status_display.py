"""터미널 상태 디스플레이 (In-Place Update)."""

import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict


# Box drawing characters
_TL = "╔"
_TR = "╗"
_BL = "╚"
_BR = "╝"
_H = "═"
_V = "║"
_ML = "╠"
_MR = "╣"


class StatusDisplay:
    """ANSI escape 기반 터미널 상태 표시.

    - TTY인 경우에만 화면 클리어 + 재그리기
    - PM2 등 non-TTY 환경에서는 아무것도 출력하지 않음
    """

    WIDTH = 62  # 내부 폭 (border 제외)

    def __init__(self) -> None:
        self._is_tty = sys.stdout.isatty()
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
        # ANSI escape 제거 후 실제 표시 길이 계산
        display_len = self._visible_len(text)
        pad = self.WIDTH - display_len
        if pad < 0:
            pad = 0
        return f"{_V} {text}{' ' * pad} {_V}"

    @staticmethod
    def _visible_len(text: str) -> int:
        """ANSI escape 코드를 제외한 표시 길이."""
        import re
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        clean = ansi_re.sub("", text)
        # 유니코드 전각 문자(한글 등) 처리
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
        info: Dict[str, Any],
        strategy: Any,
        cooldown_hours: int,
    ) -> str:
        """한 방향(Long/Short) 상태 문자열."""
        if not info["active"]:
            # 비활성: 쿨다운 확인
            last_sl = info.get("last_sl_time")
            if last_sl:
                now = datetime.now(timezone.utc)
                elapsed_h = (now - last_sl).total_seconds() / 3600
                remaining = cooldown_hours - elapsed_h
                if remaining > 0:
                    rm = int(remaining * 60)
                    rh, rm2 = divmod(rm, 60)
                    label = "LONG " if side == "long" else "SHORT"
                    arrow = "▲" if side == "long" else "▼"
                    return f"  {label} {arrow}  \x1b[33m── 대기 (쿨다운 {rh}:{rm2:02d}) ──\x1b[0m"

            label = "LONG " if side == "long" else "SHORT"
            arrow = "▲" if side == "long" else "▼"
            return f"  {label} {arrow}  \x1b[90m── 대기 중 ──\x1b[0m"

        # 활성 포지션
        label = "LONG " if side == "long" else "SHORT"
        arrow = "▲" if side == "long" else "▼"
        color = "\x1b[32m" if side == "long" else "\x1b[31m"
        reset = "\x1b[0m"

        amount = info["amount"]
        avg_price = info["avg_price"]
        dca_count = info["dca_count"]
        max_dca = info.get("max_dca", "?")

        # TP 가격
        tp_str = ""
        if avg_price > 0 and strategy:
            try:
                tp_price = strategy.calculate_tp_price(avg_price, side)
                tp_str = f"TP {tp_price:,.1f}"
            except Exception:
                tp_str = ""

        line = (
            f"  {label} {arrow}  "
            f"{color}{amount:.4f} @ {avg_price:,.2f}{reset}  "
            f"DCA {dca_count}/{max_dca}"
        )
        if tp_str:
            line += f"  {tp_str}"

        return line

    def update(
        self,
        traders: Dict[str, Any],
        testnet: bool,
    ) -> None:
        """화면을 클리어하고 상태를 다시 그린다."""
        if not self._is_tty:
            return

        uptime = self._format_uptime()
        now_str = datetime.now().strftime("%H:%M:%S")
        network = "\x1b[33mTESTNET\x1b[0m" if testnet else "\x1b[31mMAINNET\x1b[0m"

        lines = []

        # ── 헤더 ──
        lines.append(self._hline(_TL, _TR))
        header = f"DCA Trading Bot       {network}     Uptime: {uptime}"
        lines.append(self._row(header))
        lines.append(self._hline(_ML, _MR))

        # ── 심볼별 상태 ──
        total_capital = 0.0
        active_count = 0
        total_positions = 0

        for symbol, trader in traders.items():
            status = trader.get_status()
            capital = status["capital"]
            price = status["current_price"]
            total_capital += capital
            strategy = trader.strategy

            # 심볼 헤더
            price_str = f"${price:,.2f}" if price > 0 else "$---"
            cap_str = f"Capital: ${capital:,.2f}"
            sym_line = f"\x1b[1m{symbol}\x1b[0m   {price_str}"
            # 오른쪽 정렬을 위해 패딩 계산
            visible_sym = self._visible_len(sym_line)
            visible_cap = self._visible_len(cap_str)
            gap = self.WIDTH - visible_sym - visible_cap
            if gap < 1:
                gap = 1
            lines.append(self._row(f"{sym_line}{' ' * gap}{cap_str}"))

            # Long
            long_info = status["long"]
            long_info["last_sl_time"] = trader.long_state.last_sl_time
            max_dca_long = int(trader.strategy.long_params.get("max_dca", 0))
            long_info["max_dca"] = max_dca_long
            lines.append(self._row(self._format_side(
                "long", long_info, strategy, trader.cooldown_hours
            )))
            if long_info["active"]:
                active_count += 1
            total_positions += 1

            # Short
            short_info = status["short"]
            short_info["last_sl_time"] = trader.short_state.last_sl_time
            max_dca_short = int(trader.strategy.short_params.get("max_dca", 0))
            short_info["max_dca"] = max_dca_short
            lines.append(self._row(self._format_side(
                "short", short_info, strategy, trader.cooldown_hours
            )))
            if short_info["active"]:
                active_count += 1
            total_positions += 1

            # 빈 줄 (심볼 구분)
            lines.append(self._row(""))

        # ── 푸터 ──
        lines.append(self._hline(_ML, _MR))
        footer = (
            f"Capital: ${total_capital:,.2f}  |  "
            f"Active: {active_count}/{total_positions}  |  "
            f"{now_str}"
        )
        lines.append(self._row(footer))
        lines.append(self._hline(_BL, _BR))

        # ── 화면 클리어 + 출력 ──
        output = "\033[H\033[J" + "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
