"""Telegram daily report — sends a summary of bot status every day.

Reuses data-loading functions from main_web_monitor.py.

Usage:
  python main_daily_report.py            # send report once and exit
  python main_daily_report.py --test     # dry-run: print message to stdout
  python main_daily_report.py --listen   # long-poll for /report command

Env vars required in config/.env:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone

import requests as http_req
from dotenv import load_dotenv

load_dotenv("config/.env")

from main_web_monitor import (
    _calc_pnl_from_snapshots,
    _fetch_wallet_info,
    _group_positions,
    _load_optimization_info,
    _load_trade_events,
    build_snapshots,
)
from src.common.api_client import APIClient

CONFIG_PATH = "config/config.json"


# ── Telegram ─────────────────────────────────────────────────

def send_telegram(message: str) -> bool:
    """Send an HTML message via Telegram Bot API. Returns True on success."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    resp = http_req.post(url, json={
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"ERROR: Telegram API returned {resp.status_code}: {resp.text}", file=sys.stderr)
        return False
    return True


# ── Helpers ──────────────────────────────────────────────────

def _fmt_usd(v: float) -> str:
    """Format USD value with sign and commas."""
    if v >= 0:
        return f"+${v:,.2f}"
    return f"-${abs(v):,.2f}"


def _fmt_pct(pnl: float, base: float) -> str:
    """Format PnL as percentage of base."""
    if base <= 0:
        return ""
    pct = pnl / base * 100
    return f"({pct:+.2f}%)"


def _fmt_duration(start: str, end: str) -> str:
    """Human-readable duration between two ISO timestamps."""
    try:
        t0 = datetime.fromisoformat(start)
        t1 = datetime.fromisoformat(end)
        delta = t1 - t0
        total_m = int(delta.total_seconds() / 60)
        if total_m < 60:
            return f"{total_m}m"
        h = total_m // 60
        if h < 24:
            return f"{h}h{total_m % 60}m"
        d = h // 24
        return f"{d}d{h % 24}h"
    except Exception:
        return "?"


def _fmt_kst(ts: str) -> str:
    """Format ISO timestamp as KST date+time string."""
    try:
        t = datetime.fromisoformat(ts)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        kst = t + timedelta(hours=9)
        return kst.strftime("%d %H:%M:%S")
    except Exception:
        return ""


def _decay_weight(age_days: float, half_life: float = 7.0) -> float:
    """Exponential decay weight. half_life=7 → 7-day-old trade reflected at 50%."""
    return 2.0 ** (-age_days / half_life)


def _short_sym(symbol: str) -> str:
    """Shorten symbol name: ETH/USDT→ETH, ETH/USDC→ETH(C)."""
    base, quote = symbol.split("/") if "/" in symbol else (symbol, "")
    if quote == "USDC":
        return f"{base}(C)"
    return base


def _fmt_price(v: float) -> str:
    """Format price: large values with 2dp, small with more precision."""
    if v >= 100:
        return f"{v:.2f}"
    elif v >= 1:
        return f"{v:.3f}"
    else:
        return f"{v:.5f}"


# ── Report builder (unified) ────────────────────────────────

def build_report() -> str:
    """Build the unified report message in compact HTML format."""
    from pathlib import Path
    import json

    lines = []
    kst_now = datetime.now(timezone.utc) + timedelta(hours=9)
    lines.append(f"<b>DCA Report</b>  {kst_now.strftime('%m/%d %H:%M')} KST")
    lines.append("")

    # ── Wallet + Performance ──────────────────────────────
    testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
    api = APIClient(testnet=testnet)
    wallet = _fetch_wallet_info(api)
    equity = wallet.get("total_equity")

    if equity is not None:
        lines.append(f"<b>${equity:,.2f}</b>")
    else:
        lines.append("Equity: N/A")

    bnb_usd = wallet.get("bnb_value_usd")
    if bnb_usd is not None:
        lines.append(f"BNB ${bnb_usd:.2f}")

    pnl_data = _calc_pnl_from_snapshots(equity)
    pnl_24h = pnl_data.get("pnl_24h")
    pnl_7d = pnl_data.get("pnl_7d")
    pnl_monthly = pnl_data.get("pnl_monthly")

    if pnl_24h is not None:
        base_24h = equity - pnl_24h if equity else 0
        lines.append(f"24h {_fmt_usd(pnl_24h)} {_fmt_pct(pnl_24h, base_24h)}")
    if pnl_7d is not None:
        base_7d = equity - pnl_7d if equity else 0
        lines.append(f"7d  {_fmt_usd(pnl_7d)} {_fmt_pct(pnl_7d, base_7d)}")
    if pnl_monthly is not None:
        base_mo = equity - pnl_monthly if equity else 0
        lines.append(f"30d {_fmt_usd(pnl_monthly)} {_fmt_pct(pnl_monthly, base_mo)}")
    lines.append("")

    # ── Status summary ────────────────────────────────────
    cfg_raw = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    symbols_cfg = cfg_raw.get("symbols", {})
    snapshots, _err, est_mpr = build_snapshots(CONFIG_PATH)

    pos_pnl = wallet.get("position_pnl", {})

    total_sides = len(snapshots) * 2
    active_count = 0
    cooldown_count = 0
    for s in snapshots:
        for side in ("long", "short"):
            if getattr(s, f"{side}_active", False):
                active_count += 1
            sl_time = getattr(s, f"{side}_last_sl_time", None)
            if sl_time is not None:
                remaining = s.cooldown_hours - (datetime.now(timezone.utc) - sl_time).total_seconds() / 3600
                if remaining > 0:
                    cooldown_count += 1

    status_parts = [f"Position {active_count}/{total_sides}"]
    if cooldown_count:
        status_parts.append(f"{cooldown_count} cooldown")
    lines.append(" | ".join(status_parts))

    opt_info = _load_optimization_info(symbols_cfg)
    pending_set = {o["symbol"] for o in opt_info if o.get("status") == "pending_update"}

    # ── Position table ────────────────────────────────────
    rows = []
    for s in snapshots:
        binance_sym = s.symbol.replace("/", "")
        name = _short_sym(s.symbol)

        for side in ("long", "short"):
            active = getattr(s, f"{side}_active", False)
            tag = "L" if side == "long" else "S"

            if active:
                dca_cnt = getattr(s, f"{side}_dca_count", 0)
                max_dca = getattr(s, f"{side}_max_dca", 0)
                tp = getattr(s, f"{side}_tp_price", 0)
                sl = getattr(s, f"{side}_sl_price", 0)
                price = s.current_price
                tp_pct = (tp - price) / price * 100 if price > 0 else 0
                sl_pct = (sl - price) / price * 100 if price > 0 else 0
                pnl = pos_pnl.get((binance_sym, side))
                pnl_str = f"{_fmt_usd(pnl)}" if pnl is not None else ""
                rows.append(
                    f" {name:<7s}{tag} {dca_cnt}/{max_dca}"
                    f" {tp_pct:>+6.1f}%"
                    f" {sl_pct:>+6.1f}%"
                    f" {pnl_str:>7s}"
                )
            else:
                sl_time = getattr(s, f"{side}_last_sl_time", None)
                status = "idle"
                if sl_time is not None:
                    remaining = s.cooldown_hours - (datetime.now(timezone.utc) - sl_time).total_seconds() / 3600
                    if remaining > 0:
                        rh = int(remaining)
                        rm = int((remaining - rh) * 60)
                        status = f"cd {rh}h{rm:02d}m"
                rows.append(f" {name:<7s}{tag} {status}")

    if rows:
        lines.append("")
        header = (
            f" {'':7s}{'':1s} {'DCA':3s}"
            f" {'TP':>7s}"
            f" {'SL':>7s}"
            f" {'PnL':>7s}"
        )
        lines.append("<pre>" + header + "\n" + "\n".join(rows) + "</pre>")

    # ── Capital table ─────────────────────────────────────
    cap_rows = []
    for s in snapshots:
        name = _short_sym(s.symbol)
        pending = " *" if s.symbol in pending_set else ""
        cap_rows.append(f" {name:<7s}${s.capital:>7,.0f}{pending}")
    if cap_rows:
        lines.append("")
        cap_label = "<b>Capital</b>"
        if pending_set:
            cap_label += "  (* param pending)"
        lines.append(cap_label)
        lines.append("<pre>" + "\n".join(cap_rows) + "</pre>")

    # ── Capital drift ─────────────────────────────────────
    if equity:
        drift_lines = []
        for s in snapshots:
            safe_name = s.symbol.replace("/", "_")
            sym_cfg = symbols_cfg.get(safe_name, {})
            weight = float(sym_cfg.get("weight", 0)) if isinstance(sym_cfg, dict) else 0
            target = equity * weight
            if target <= 0 or s.capital <= 0:
                continue

            if target <= s.capital:
                continue

            diff_pct = (target - s.capital) / s.capital * 100
            if diff_pct <= 10:
                continue

            name = _short_sym(s.symbol)
            drift_lines.append(f"  {name:<7s} ${s.capital:.0f} → ${target:.0f}")
        if drift_lines:
            lines.append("")
            lines.append("<b>Drift</b>")
            lines.append("<pre>" + "\n".join(drift_lines) + "</pre>")

    # ── Decay-weighted PnL (half-life 7d, 180d window) ─────
    events = _load_trade_events(symbols_cfg, max_months=7)
    positions = _group_positions(events, limit=2000)

    now_utc = datetime.now(timezone.utc)
    cutoff = (now_utc - timedelta(days=180)).isoformat()
    recent = [p for p in positions if (p.get("end_time") or "") >= cutoff]

    lines.append("")
    if recent:
        wins = sum(1 for p in recent if p.get("pnl", 0) > 0)
        losses = len(recent) - wins

        # Apply exponential decay weight to each position's PnL
        for p in recent:
            try:
                end_dt = datetime.fromisoformat(p["end_time"])
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                age = (now_utc - end_dt).total_seconds() / 86400
            except Exception:
                age = 180.0
            p["eff_pnl"] = p.get("pnl", 0) * _decay_weight(age)

        total_eff = sum(p["eff_pnl"] for p in recent)

        lines.append("<b>PnL (½=7d)</b>")
        lines.append(f"{len(recent)} ({wins}W/{losses}L)")
        lines.append(f"Net <b>{_fmt_usd(total_eff)}</b>")

        # Aggregate per symbol
        from collections import defaultdict
        sym_agg = defaultdict(lambda: {"eff": 0.0, "count": 0, "unreal": 0.0})
        for p in recent:
            sym = p.get("symbol", "?")
            sym_agg[sym]["count"] += 1
            sym_agg[sym]["eff"] += p["eff_pnl"]

        tbl = []
        for sym in sorted(sym_agg.keys()):
            a = sym_agg[sym]
            name = _short_sym(sym)
            tbl.append(f" {name:<6s}{a['count']:>2d} {a['eff']:>+8.1f}")

        header = f" {'':6s}{'#':>2s} {'Net':>8s}"
        lines.append("<pre>" + header + "\n" + "\n".join(tbl) + "</pre>")
    else:
        lines.append("<b>PnL (½=7d)</b>  0")

    # ── Recent 10 closed positions ────────────────────────
    last10 = positions[:10]
    if last10:
        lines.append("")
        lines.append("<b>Recent Closes</b>")
        rows = []
        for p in last10:
            name = _short_sym(p.get("symbol", "?"))
            side = "L" if p.get("side") == "long" else "S"
            exit_t = p.get("exit_type", "?")
            pnl = p.get("pnl", 0)
            pnl_str = f"{pnl:+.1f}"
            dur = _fmt_duration(p.get("start_time", ""), p.get("end_time", ""))
            closed = _fmt_kst(p.get("end_time", ""))
            rows.append(f" {name:<7s} {side} {exit_t} {pnl_str:>5s} {dur:>6s}   {closed}")
        header = f" {'':7s} {'':1s} {'':2s} {'PnL':^5s} {'Dur':>6s}   {'Closed'}"
        lines.append("<pre>" + header + "\n" + "\n".join(rows) + "</pre>")

    # ── Est. MPR + MDD ───────────────────────────────────
    if est_mpr and est_mpr > 0:
        total_capital = sum(s.capital for s in snapshots)
        est_mo = total_capital * est_mpr / 100
        est_pct = est_mo / equity * 100 if equity else est_mpr

        opt_info = _load_optimization_info(symbols_cfg)
        weighted_mdd = 0.0
        total_w = 0.0
        for info in opt_info:
            safe = info["symbol"].replace("/", "_")
            w = float(symbols_cfg.get(safe, {}).get("weight", 0)) if isinstance(symbols_cfg.get(safe), dict) else 0
            mdd = info.get("active_mdd")
            if mdd is not None and w > 0:
                weighted_mdd += mdd * w
                total_w += w
        est_mdd = weighted_mdd if weighted_mdd > 0 else None

        mpr_line = f"\nEst. MPR {est_pct:+.1f}% ${est_mo:,.0f}/mo"
        if est_mdd is not None:
            mpr_line += f"  MDD {est_mdd:.1f}%"
        lines.append(mpr_line)

    return "\n".join(lines)


# ── Scheduled hours (KST) ────────────────────────────────────
SCHEDULE_HOURS_KST = {0, 9, 14, 19}


def _send_report(reason: str) -> None:
    """Build and send report."""
    print(f"[{datetime.now().isoformat()}] sending report ({reason})")
    try:
        report = build_report()
    except Exception as e:
        report = f"<b>DCA Report</b>\n\nERROR:\n<code>{e}</code>"
    send_telegram(report)


# ── Main loop: scheduled + /report command ───────────────────

def main():
    if "--test" in sys.argv:
        try:
            msg = build_report()
        except Exception as e:
            msg = f"<b>DCA Report</b>\n\nERROR:\n<code>{e}</code>"
        import re
        plain = re.sub(r"</?(?:b|i|code|pre)>", "", msg)
        print(plain)
        print("\n--- (dry run, not sent) ---")
        return

    if "--send" in sys.argv:
        _send_report("manual --send")
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set", file=sys.stderr)
        sys.exit(1)

    base_url = f"https://api.telegram.org/bot{token}"
    offset = 0
    last_scheduled_hour = -1
    print("Report service started (schedule + /report listener)")

    while True:
        try:
            # ── Check schedule ──
            kst_now = datetime.now(timezone.utc) + timedelta(hours=9)
            h = kst_now.hour
            if h in SCHEDULE_HOURS_KST and h != last_scheduled_hour and kst_now.minute < 5:
                last_scheduled_hour = h
                _send_report(f"scheduled {h:02d}:00 KST")

            # ── Poll for /report command ──
            resp = http_req.get(
                f"{base_url}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=60,
            )
            if resp.status_code != 200:
                print(f"getUpdates error: {resp.status_code}", file=sys.stderr)
                time.sleep(5)
                continue

            updates = resp.json().get("result", [])
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})
                text = msg.get("text", "")
                from_chat = str(msg.get("chat", {}).get("id", ""))

                cmd = text.strip()
                if cmd == "/report" and from_chat == chat_id:
                    _send_report("/report command")

        except http_req.exceptions.Timeout:
            continue
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            traceback.print_exc()
            time.sleep(10)


if __name__ == "__main__":
    main()
