"""Telegram daily report — sends a summary of bot status every day.

Reuses data-loading functions from main_web_monitor.py.

Usage:
  python main_daily_report.py          # send report once and exit
  python main_daily_report.py --test   # dry-run: print message to stdout

Env vars required in config/.env:
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
"""

import os
import sys
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


# ── Report builder ───────────────────────────────────────────

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
        return kst.strftime("%m/%d %H:%M:%S")
    except Exception:
        return ""


def build_report() -> str:
    """Build the daily report message in HTML format."""
    lines = ["<b>DCA Bot Daily Report</b>"]
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"<i>{now_str}</i>")
    lines.append("")

    # ── Wallet ────────────────────────────────────────────
    testnet = os.getenv("USE_TESTNET", "true").lower() == "true"
    api = APIClient(testnet=testnet)
    wallet = _fetch_wallet_info(api)

    equity = wallet.get("total_equity")
    lines.append("<b>Wallet</b>")
    if equity is not None:
        lines.append(f"  Total Equity: <b>${equity:,.2f}</b>")
        usdt = wallet.get("usdt_balance") or 0
        usdc = wallet.get("usdc_balance") or 0
        if usdc > 0.01:
            lines.append(f"  USDT: ${usdt:,.2f} | USDC: ${usdc:,.2f}")
        else:
            lines.append(f"  USDT: ${usdt:,.2f}")
        bnb_usd = wallet.get("bnb_value_usd") or 0
        if bnb_usd > 0.01:
            lines.append(f"  BNB: ${bnb_usd:,.2f}")
        upnl = wallet.get("unrealized_pnl") or 0
        if abs(upnl) > 0.01:
            lines.append(f"  Unrealized PnL: {_fmt_usd(upnl)}")
    else:
        lines.append("  (unavailable)")
    lines.append("")

    # ── Performance ───────────────────────────────────────
    pnl_data = _calc_pnl_from_snapshots(equity)
    pnl_24h = pnl_data.get("pnl_24h")
    pnl_monthly = pnl_data.get("pnl_monthly")

    lines.append("<b>Performance</b>")
    if pnl_24h is not None:
        base_24h = equity - pnl_24h if equity else 0
        lines.append(f"  24h PnL: {_fmt_usd(pnl_24h)} {_fmt_pct(pnl_24h, base_24h)}")
    else:
        lines.append("  24h PnL: N/A")
    if pnl_monthly is not None:
        base_mo = equity - pnl_monthly if equity else 0
        lines.append(f"  Monthly PnL: {_fmt_usd(pnl_monthly)} {_fmt_pct(pnl_monthly, base_mo)}")
    else:
        lines.append("  Monthly PnL: N/A")
    lines.append("")

    # ── Status ────────────────────────────────────────────
    from pathlib import Path
    import json
    cfg_raw = json.loads(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    symbols_cfg = cfg_raw.get("symbols", {})

    snapshots, _err, _est_mpr = build_snapshots(CONFIG_PATH)

    total_sides = len(snapshots) * 2  # each symbol has long + short
    active_count = 0
    cooldown_count = 0
    for s in snapshots:
        for side in ("long", "short"):
            if getattr(s, f"{side}_active", False):
                active_count += 1
            if getattr(s, f"{side}_last_sl_time", None) is not None:
                from datetime import timezone as tz
                now = datetime.now(tz.utc)
                sl_time = getattr(s, f"{side}_last_sl_time")
                remaining = s.cooldown_hours - (now - sl_time).total_seconds() / 3600
                if remaining > 0:
                    cooldown_count += 1

    lines.append("<b>Status</b>")
    lines.append(f"  Positions: {active_count}/{total_sides} active, {cooldown_count} cooldown")

    # Parameter updates (pending_update = new params not yet deployed)
    opt_info = _load_optimization_info(symbols_cfg)
    pending_params = [o["symbol"] for o in opt_info if o.get("status") == "pending_update"]
    if pending_params:
        lines.append(f"  Pending params: {', '.join(pending_params)}")

    # Capital changes — compare margin file capital vs config weight * equity
    if equity:
        total_weight = sum(
            float(v.get("weight", 0)) if isinstance(v, dict) else 0
            for v in symbols_cfg.values()
        )
        capital_diffs = []
        for s in snapshots:
            safe_name = s.symbol.replace("/", "_")
            sym_cfg = symbols_cfg.get(safe_name, {})
            weight = float(sym_cfg.get("weight", 0)) if isinstance(sym_cfg, dict) else 0
            expected = equity * weight / total_weight if total_weight > 0 else 0
            if expected > 0 and s.capital > 0:
                diff_pct = abs(s.capital - expected) / expected * 100
                if diff_pct > 10:  # >10% difference
                    capital_diffs.append(s.symbol)
        if capital_diffs:
            lines.append(f"  Capital drift: {', '.join(capital_diffs)}")

    lines.append("")

    # ── Recent Trades (24h) ───────────────────────────────

    events = _load_trade_events(symbols_cfg, max_months=2)
    positions = _group_positions(events, limit=200)

    # Filter to last 24h
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    recent = [p for p in positions if (p.get("end_time") or "") >= cutoff]

    lines.append(f"<b>Recent Closed Positions (24h): {len(recent)}</b>")
    if recent:
        wins = sum(1 for p in recent if p.get("pnl", 0) > 0)
        losses = len(recent) - wins
        total_pnl = sum(p.get("pnl", 0) for p in recent)

        for p in recent:
            sym = p.get("symbol", "?")
            side = p.get("side", "?")[0].upper()  # L or S
            exit_t = p.get("exit_type", "?")
            pnl = p.get("pnl", 0)
            ts = _fmt_kst(p.get("end_time", ""))
            lines.append(f"  {sym} {side}/{exit_t}  {_fmt_usd(pnl)}  {ts}")

        lines.append(f"  WR: {wins}W/{losses}L | Total: {_fmt_usd(total_pnl)}")
    else:
        lines.append("  No closed positions in last 24h")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────

def main():
    dry_run = "--test" in sys.argv

    try:
        msg = build_report()
    except Exception as e:
        msg = f"<b>DCA Bot Daily Report</b>\n\nERROR building report:\n<code>{e}</code>"

    if dry_run:
        # Replace HTML tags for terminal readability
        plain = msg.replace("<b>", "").replace("</b>", "")
        plain = plain.replace("<i>", "").replace("</i>", "")
        plain = plain.replace("<code>", "").replace("</code>", "")
        print(plain)
        print("\n--- (dry run, not sent) ---")
    else:
        ok = send_telegram(msg)
        if ok:
            print("Report sent successfully")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
