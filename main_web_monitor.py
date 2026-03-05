"""Web-based monitoring dashboard — check real-time status in browser.

When the bot (main_trading.py) is running in the background via PM2,
run this in a separate terminal to auto-open the browser dashboard.

Usage:
  python main_web_monitor.py                    # localhost:8080
  python main_web_monitor.py --port 3000        # custom port
  python main_web_monitor.py --mainnet          # MAINNET mode
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv("config/.env")

from src.common.api_client import APIClient
from src.trading.status_display import SymbolSnapshot


# ── Data loading utilities ────────────────────────────────────

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file. Returns None on failure."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


SNAPSHOT_DIR = Path("data/balance_snapshots")
SNAPSHOT_INTERVAL = 600  # 10 minutes


def _append_snapshot(value: float) -> None:
    """Append a balance snapshot record to the monthly JSONL file."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    filename = f"balance_{now.strftime('%Y%m')}.jsonl"
    record = json.dumps({"t": now.isoformat(), "v": round(value, 4)})
    with (SNAPSHOT_DIR / filename).open("a", encoding="utf-8") as f:
        f.write(record + "\n")


def _load_snapshots(since: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """Load balance snapshots from JSONL files.

    If since is given, only returns records with timestamp >= since.
    Reads current month and previous month files to cover 24h boundary.
    """
    results: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    months_to_check = {now.strftime("%Y%m")}
    if since:
        months_to_check.add(since.strftime("%Y%m"))

    for ym in sorted(months_to_check):
        fpath = SNAPSHOT_DIR / f"balance_{ym}.jsonl"
        if not fpath.exists():
            continue
        try:
            with fpath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        ts = datetime.fromisoformat(rec["t"])
                        if since and ts < since:
                            continue
                        results.append(rec)
                    except Exception:
                        continue
        except Exception:
            continue
    return results


def _calc_pnl_from_snapshots(current: Optional[float]) -> Dict[str, Optional[float]]:
    """Calculate 24h and monthly PnL from balance snapshots."""
    result: Dict[str, Optional[float]] = {"pnl_24h": None, "pnl_monthly": None}
    if current is None:
        return result

    now = datetime.now(timezone.utc)

    # 24h PnL: find the oldest snapshot within 24h window
    since_24h = now - timedelta(hours=24)
    snapshots_24h = _load_snapshots(since=since_24h)
    if snapshots_24h:
        oldest_val = snapshots_24h[0]["v"]
        result["pnl_24h"] = current - oldest_val

    # Monthly PnL ($) + Estimated monthly return (%)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    snapshots_month = _load_snapshots(since=month_start)
    if snapshots_month:
        first_val = snapshots_month[0]["v"]
        result["pnl_monthly"] = current - first_val

    return result


# ── Trade History ─────────────────────────────────────────────

_cached_trade_history: List[Dict] = []
_last_trade_load: float = 0.0

TRADE_LOG_DIR = Path("data/logs/trades")


def _load_trade_events(symbols_cfg: Dict, max_months: int = 2) -> List[Dict]:
    """Load recent trade events from JSONL files."""
    events: List[Dict] = []
    if not TRADE_LOG_DIR.exists():
        return events

    for safe_name in symbols_cfg.keys():
        files = sorted(TRADE_LOG_DIR.glob(f"{safe_name}_*.jsonl"), reverse=True)
        for fpath in files[:max_months]:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            events.append(json.loads(line))
            except Exception:
                pass

    events.sort(key=lambda e: e.get("timestamp", ""))
    return events


def _group_positions(events: List[Dict], limit: int = 200) -> List[Dict]:
    """Group trade events into closed positions (ENTRY→DCA(s)→TP/SL).

    PnL is calculated from entry/DCA avg_price vs close price,
    since the balance-based pnl field in JSONL is unreliable.
    """
    # Per (symbol, side) state machine
    open_pos: Dict[tuple, Dict] = {}
    pending_entry: Dict[tuple, Dict] = {}
    completed: List[Dict] = []

    for ev in events:
        sym = ev.get("symbol", "")
        side = ev.get("side", "")
        event = ev.get("event", "")
        key = (sym, side)

        if event == "ENTRY":
            if key in open_pos:
                # Position already open — buffer this entry for after TP/SL
                pending_entry[key] = ev
            else:
                price = float(ev.get("price", 0))
                amount = float(ev.get("amount", 0))
                open_pos[key] = {
                    "symbol": sym,
                    "side": side,
                    "start_time": ev.get("timestamp"),
                    "dca_count": 0,
                    "total_cost": price * amount,
                    "total_amount": amount,
                }

        elif event == "DCA":
            if key in open_pos:
                pos = open_pos[key]
                pos["dca_count"] += 1
                price = float(ev.get("price", 0))
                dca_amount = float(ev.get("amount", 0))
                pos["total_cost"] += price * dca_amount
                pos["total_amount"] += dca_amount

        elif event in ("TP", "SL"):
            if key in open_pos:
                pos = open_pos.pop(key)
                close_price = float(ev.get("price", 0))
                pos["end_time"] = ev.get("timestamp")
                pos["exit_type"] = event

                # Calculate PnL from avg_price vs close_price
                total_amt = pos.get("total_amount", 0)
                avg_price = (pos["total_cost"] / total_amt) if total_amt > 0 else 0
                if side == "long":
                    pos["pnl"] = (close_price - avg_price) * total_amt
                else:
                    pos["pnl"] = (avg_price - close_price) * total_amt
                # Remove internal tracking fields
                pos.pop("total_cost", None)
                pos.pop("total_amount", None)
                completed.append(pos)

                # Start new position from pending entry if exists
                if key in pending_entry:
                    pe = pending_entry.pop(key)
                    pe_price = float(pe.get("price", 0))
                    pe_amount = float(pe.get("amount", 0))
                    open_pos[key] = {
                        "symbol": sym,
                        "side": side,
                        "start_time": pe.get("timestamp"),
                        "dca_count": 0,
                        "total_cost": pe_price * pe_amount,
                        "total_amount": pe_amount,
                    }

    # Sort by end_time descending (most recent first), limit
    completed.sort(key=lambda p: p.get("end_time", ""), reverse=True)
    return completed[:limit]


# ── Optimization Info ─────────────────────────────────────────

_cached_opt_info: List[Dict] = []
_last_opt_load: float = 0.0

OPT_DECISIONS_PATH = Path("data/logs/optimization_decisions.jsonl")


def _load_optimization_info(symbols_cfg: Dict) -> List[Dict]:
    """Load optimization info by comparing params vs active_params."""
    # Load latest decisions from log
    latest_decisions: Dict[str, Dict] = {}
    if OPT_DECISIONS_PATH.exists():
        try:
            with open(OPT_DECISIONS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        sym = rec.get("symbol", "")
                        latest_decisions[sym] = rec
        except Exception:
            pass

    result = []
    for safe_name in symbols_cfg.keys():
        symbol = safe_name.replace("_", "/")
        # USDC pairs share optimization results with USDT counterpart
        params_key = safe_name
        if not (Path("data/params") / f"{safe_name}.json").exists():
            usdt_name = re.sub(r"_(?!USDT$)[A-Z]+$", "_USDT", safe_name)
            if usdt_name != safe_name:
                params_key = usdt_name
        params_data = _load_json(Path("data/params") / f"{params_key}.json")
        active_data = _load_json(Path("data/active_params") / f"{safe_name}.json")

        info: Dict[str, Any] = {"symbol": symbol}

        # Pending (latest optimization result)
        if params_data:
            meta = params_data.get("meta", {})
            perf = params_data.get("performance", {})
            info["pending_created"] = meta.get("created_at")
            info["pending_generation"] = meta.get("generation")
            info["pending_mpr"] = perf.get("mpr")
            info["pending_mdd"] = perf.get("mdd")
            info["pending_sharpe"] = perf.get("sharpe")
            info["pending_fitness"] = perf.get("fitness")
            info["pending_params"] = params_data.get("parameters")

        # Active (currently deployed)
        if active_data:
            meta = active_data.get("meta", {})
            perf = active_data.get("performance", {})
            info["active_created"] = meta.get("created_at")
            info["active_generation"] = meta.get("generation")
            info["active_mpr"] = perf.get("mpr")
            info["active_mdd"] = perf.get("mdd")
            info["active_sharpe"] = perf.get("sharpe")
            info["active_fitness"] = perf.get("fitness")
            info["active_params"] = active_data.get("parameters")

        # Status: compare actual parameter values, not just timestamps
        pc = info.get("pending_created", "")
        ac = info.get("active_created", "")
        pp = info.get("pending_params")
        ap = info.get("active_params")
        if pc and ac and pc > ac and pp and ap and pp != ap:
            info["status"] = "pending_update"
        else:
            info["status"] = "up_to_date"

        # Last updated = most recent created_at among params/active_params
        info["last_updated"] = ac or pc or None

        # Latest decision from log (fallback to USDT counterpart for USDC pairs)
        decision = latest_decisions.get(symbol)
        if not decision:
            usdt_sym = re.sub(r"/(?!USDT$)[A-Z]+$", "/USDT", symbol)
            if usdt_sym != symbol:
                decision = latest_decisions.get(usdt_sym)
        if decision:
            info["last_decision"] = decision.get("decision")
            info["last_decision_time"] = decision.get("timestamp")

        result.append(info)

    return result


def build_snapshots(config_path: str) -> tuple:
    """Build snapshot list from config, state, params, and margin files.

    Returns:
        (snapshots, error_msg, est_mpr)
    """
    cfg = _load_json(Path(config_path))
    if cfg is None:
        return [], f"Config not found: {config_path}", None

    cooldown_hours = int(cfg.get("cooldown_hours", 6))
    symbols_cfg = cfg.get("symbols", {})

    snapshots: List[SymbolSnapshot] = []
    weighted_mpr_sum = 0.0
    total_weight = 0.0

    for safe_name, sym_val in symbols_cfg.items():
        symbol = safe_name.replace("_", "/")
        weight = float(sym_val.get("weight", 0)) if isinstance(sym_val, dict) else 0

        state_path = Path("data/state") / f"{safe_name}_state.json"
        state_data = _load_json(state_path) or {}

        params_data = _load_json(Path("data/active_params") / f"{safe_name}.json")
        if params_data is None:
            params_data = _load_json(Path("data/params") / f"{safe_name}.json")
        if params_data is None:
            params_data = {}

        # Accumulate weighted MPR from params performance
        perf = params_data.get("performance", {})
        mpr = perf.get("mpr")
        if mpr is not None and weight > 0:
            weighted_mpr_sum += float(mpr) * weight
            total_weight += weight

        margin_path = Path("data/margins") / f"{safe_name}_margin.json"
        margin_data = _load_json(margin_path) or {}

        snap = SymbolSnapshot.from_state_files(
            symbol=symbol,
            state_data=state_data,
            params_data=params_data,
            margin_data=margin_data,
            cooldown_hours=cooldown_hours,
        )
        snapshots.append(snap)

    est_mpr = weighted_mpr_sum / total_weight if total_weight > 0 else None

    return snapshots, None, est_mpr


# ── Shared data ───────────────────────────────────────────────

_data_lock = threading.Lock()
_shared_data: Dict[str, Any] = {
    "snapshots": [],
    "equity": None,
    "testnet": True,
    "uptime": "00:00:00",
    "updated_at": "",
    "error": None,
}


def _fmt_price(value: float, sig: int = 5) -> str:
    """Format price with at least sig significant figures (with commas)."""
    if value == 0:
        return "0." + "0" * (sig - 1)
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(sig - 1 - magnitude, 0)
    return f"{value:,.{decimals}f}"


def _snapshot_to_dict(snap: SymbolSnapshot) -> Dict[str, Any]:
    """Serialize SymbolSnapshot to JSON-serializable dict."""

    def _sl_remaining(last_sl_time: Optional[datetime], cooldown_hours: int) -> Optional[str]:
        if not last_sl_time:
            return None
        now = datetime.now(timezone.utc)
        elapsed_h = (now - last_sl_time).total_seconds() / 3600
        remaining = cooldown_hours - elapsed_h
        if remaining <= 0:
            return None
        rm = int(remaining * 60)
        rh, rm2 = divmod(rm, 60)
        return f"{rh}:{rm2:02d}"

    return {
        "symbol": snap.symbol,
        "capital": snap.capital,
        "current_price": snap.current_price,
        "current_price_fmt": _fmt_price(snap.current_price) if snap.current_price > 0 else "---",
        "long": {
            "active": snap.long_active,
            "amount": snap.long_amount,
            "avg_price": snap.long_avg_price,
            "avg_price_fmt": _fmt_price(snap.long_avg_price) if snap.long_avg_price > 0 else "---",
            "dca_count": snap.long_dca_count,
            "max_dca": snap.long_max_dca,
            "tp_price": snap.long_tp_price,
            "tp_price_fmt": _fmt_price(snap.long_tp_price) if snap.long_tp_price > 0 else "",
            "sl_price": snap.long_sl_price,
            "sl_price_fmt": _fmt_price(snap.long_sl_price) if snap.long_sl_price > 0 else "",
            "dca_prices": snap.long_dca_prices,
            "cooldown": _sl_remaining(snap.long_last_sl_time, snap.cooldown_hours),
        },
        "short": {
            "active": snap.short_active,
            "amount": snap.short_amount,
            "avg_price": snap.short_avg_price,
            "avg_price_fmt": _fmt_price(snap.short_avg_price) if snap.short_avg_price > 0 else "---",
            "dca_count": snap.short_dca_count,
            "max_dca": snap.short_max_dca,
            "tp_price": snap.short_tp_price,
            "tp_price_fmt": _fmt_price(snap.short_tp_price) if snap.short_tp_price > 0 else "",
            "sl_price": snap.short_sl_price,
            "sl_price_fmt": _fmt_price(snap.short_sl_price) if snap.short_sl_price > 0 else "",
            "dca_prices": snap.short_dca_prices,
            "cooldown": _sl_remaining(snap.short_last_sl_time, snap.cooldown_hours),
        },
        "pending_retries": snap.pending_retries,
        "params_date": snap.params_date,
        "mpr": snap.mpr,
        "mdd": snap.mdd,
        "sharpe": snap.sharpe,
        "est_monthly": snap.capital * snap.mpr / 100.0 if snap.mpr > 0 else 0.0,
    }


def _fetch_wallet_info(api: APIClient) -> Dict[str, Any]:
    """Fetch Futures wallet info: USDT, BNB balances + unrealized PnL.

    Uses totalWalletBalance (excluding unrealized PnL) as the base wallet value,
    and fetches per-position unrealized PnL from /fapi/v2/positionRisk.
    """
    info: Dict[str, Any] = {
        "usdt_balance": None,
        "usdc_balance": None,
        "stable_balance": None,  # USDT + USDC
        "bnb_balance": None,
        "bnb_value_usd": None,
        "unrealized_pnl": None,
        "wallet_balance": None,
        "total_equity": None,
        "position_pnl": {},  # {(symbol, side): pnl}
    }
    try:
        account = api.client.account()
        info["wallet_balance"] = float(account.get("totalWalletBalance", 0))
        info["margin_balance"] = float(account.get("totalMarginBalance", 0))
        info["unrealized_pnl"] = float(account.get("totalUnrealizedProfit", 0))

        # Per-asset balances from account assets array
        for a in account.get("assets", []):
            asset = a.get("asset")
            if asset == "USDT":
                info["usdt_balance"] = float(a.get("walletBalance", 0))
            elif asset == "USDC":
                info["usdc_balance"] = float(a.get("walletBalance", 0))
            elif asset == "BNB":
                info["bnb_balance"] = float(a.get("walletBalance", 0))
    except Exception:
        pass

    # Stable = USDT + USDC
    usdt = info["usdt_balance"] or 0
    usdc = info["usdc_balance"] or 0
    info["stable_balance"] = usdt + usdc

    # Per-position unrealized PnL from positionRisk
    try:
        positions = api.get_positions()
        for p in positions:
            sym = p["symbol"]  # e.g. "ETHUSDT"
            side = p["side"]   # "long" or "short"
            info["position_pnl"][(sym, side)] = p["unrealizedPnl"]
    except Exception:
        pass

    if info["bnb_balance"] is not None:
        try:
            bnb_price = api.get_mark_price("BNBUSDT")
            info["bnb_value_usd"] = info["bnb_balance"] * bnb_price
        except Exception:
            pass

    # Total Equity = (USDT + USDC) + BNB(USD) + Unrealized PnL
    stable = info["stable_balance"] or 0
    bnb_usd = info["bnb_value_usd"] or 0
    upnl = info["unrealized_pnl"] or 0
    info["total_equity"] = stable + bnb_usd + upnl

    return info


# ── Data collection thread ────────────────────────────────────

def _data_loop(
    config_path: str,
    api: APIClient,
    testnet: bool,
    interval: int,
    start_time: float,
) -> None:
    """Periodically collect data in background."""
    global _shared_data

    last_snapshot_time = 0.0  # force immediate first snapshot

    while True:
        # Uptime
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        uptime = f"{h:02d}:{m:02d}:{s:02d}"

        snapshots, error, est_mpr = build_snapshots(config_path)

        wallet = {"usdt_balance": None, "bnb_balance": None, "bnb_value_usd": None,
                  "unrealized_pnl": None, "wallet_balance": None, "total_equity": None,
                  "position_pnl": {}}
        equity = None
        if not error:
            try:
                wallet = _fetch_wallet_info(api)
                equity = wallet["total_equity"]
            except Exception:
                pass

        # Periodic balance snapshot (every 10 min)
        now_ts = time.time()
        if equity is not None and (now_ts - last_snapshot_time) >= SNAPSHOT_INTERVAL:
            try:
                _append_snapshot(equity)
                last_snapshot_time = now_ts
            except Exception:
                pass

        # Calculate 24h / monthly PnL from snapshots
        pnl_data = _calc_pnl_from_snapshots(equity)

        # Trade history + optimization info (cached, not every cycle)
        global _cached_trade_history, _last_trade_load
        global _cached_opt_info, _last_opt_load
        now_ts_cache = time.time()
        cfg_raw = _load_json(Path(config_path))
        symbols_cfg = cfg_raw.get("symbols", {}) if cfg_raw else {}
        if now_ts_cache - _last_trade_load >= 30:
            _cached_trade_history = _group_positions(_load_trade_events(symbols_cfg))
            _last_trade_load = now_ts_cache
        if now_ts_cache - _last_opt_load >= 60:
            _cached_opt_info = _load_optimization_info(symbols_cfg)
            _last_opt_load = now_ts_cache

        # Inject per-position unrealized PnL into snapshot dicts
        pos_pnl = wallet.get("position_pnl", {})
        snap_dicts = []
        for sn in snapshots:
            d = _snapshot_to_dict(sn)
            binance_sym = sn.symbol.replace("/", "")  # "ETH/USDT" -> "ETHUSDT"
            d["long"]["pnl"] = pos_pnl.get((binance_sym, "long"))
            d["short"]["pnl"] = pos_pnl.get((binance_sym, "short"))
            snap_dicts.append(d)

        now_str = datetime.now().strftime("%H:%M:%S")

        active_count = 0
        total_positions = 0
        total_capital = 0.0
        est_monthly_total = 0.0
        for sn in snapshots:
            total_capital += sn.capital
            if sn.mpr and sn.mpr > 0:
                est_monthly_total += sn.capital * sn.mpr / 100.0
            if sn.long_active:
                active_count += 1
            total_positions += 1
            if sn.short_active:
                active_count += 1
            total_positions += 1

        def _fmt_usd(v):
            return f"${v:,.2f}" if v is not None else "$---"

        def _fmt_pnl(v):
            if v is None:
                return "$---"
            sign = "+" if v >= 0 else ""
            return f"{sign}${v:,.2f}"

        def _fmt_bnb(v):
            return f"{v:.4f}" if v is not None else "---"

        with _data_lock:
            _shared_data = {
                "snapshots": snap_dicts,
                "equity": equity,
                "equity_fmt": _fmt_usd(equity),
                "total_capital": total_capital,
                "total_capital_fmt": _fmt_usd(total_capital),
                "testnet": testnet,
                "uptime": uptime,
                "updated_at": now_str,
                "active_count": active_count,
                "total_positions": total_positions,
                "error": error,
                "pnl_24h": pnl_data["pnl_24h"],
                "pnl_24h_fmt": _fmt_pnl(pnl_data["pnl_24h"]),
                "pnl_monthly": pnl_data["pnl_monthly"],
                "pnl_monthly_fmt": _fmt_pnl(pnl_data["pnl_monthly"]),
                "est_mpr": est_mpr,
                "est_mpr_fmt": f"{est_mpr:+.1f}%" if est_mpr is not None else None,
                "est_monthly_usd": est_monthly_total,
                "est_monthly_usd_fmt": f"${est_monthly_total:,.0f}/mo" if est_monthly_total > 0 else None,
                "wallet": {
                    "stable_balance": wallet["stable_balance"],
                    "stable_balance_fmt": _fmt_usd(wallet["stable_balance"]),
                    "bnb_balance": wallet["bnb_balance"],
                    "bnb_balance_fmt": _fmt_bnb(wallet["bnb_balance"]),
                    "bnb_value_usd": wallet["bnb_value_usd"],
                    "bnb_value_usd_fmt": _fmt_usd(wallet["bnb_value_usd"]),
                    "unrealized_pnl": wallet["unrealized_pnl"],
                    "unrealized_pnl_fmt": _fmt_pnl(wallet["unrealized_pnl"]),
                    "wallet_balance": wallet["wallet_balance"],
                    "wallet_balance_fmt": _fmt_usd(wallet["wallet_balance"]),
                    "total_equity": wallet["total_equity"],
                    "total_equity_fmt": _fmt_usd(wallet["total_equity"]),
                },
                "trade_history": _cached_trade_history,
                "optimization_info": _cached_opt_info,
            }

        time.sleep(interval)


# ── HTML page ─────────────────────────────────────────────────

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DCA Trading Bot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace;font-size:16px;padding:24px}
.container{max-width:1800px;margin:0 auto}

.header{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px 24px;margin-bottom:18px}
.header-top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.title{font-size:22px;font-weight:700;color:#e6edf3}
.badge{font-size:14px;font-weight:600;padding:3px 10px;border-radius:4px}
.badge-testnet{background:#1f2d1f;color:#3fb950;border:1px solid #238636}
.badge-mainnet{background:#1f2d1f;color:#3fb950;border:1px solid #238636}

.wallet-bar{display:grid;grid-template-columns:repeat(8,1fr);gap:16px;margin-top:14px;padding-top:14px;border-top:1px solid #30363d}
.wallet-item{display:flex;flex-direction:column;gap:2px}
.wallet-label{font-size:13px;color:#8b949e}
.wallet-value{font-size:18px;font-weight:700;color:#e6edf3}
.wallet-value.pnl-pos{color:#3fb950}
.wallet-value.pnl-neg{color:#f85149}
.wallet-sub{font-size:12px;color:#6e7681}

.meta-bar{display:flex;gap:24px;margin-top:12px;font-size:14px;color:#8b949e}
.meta-bar span{display:flex;align-items:center;gap:4px}

.footer-bar{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 24px;margin-top:18px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font-size:14px;color:#8b949e}

.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px}

.section-header{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 24px;margin:18px 0 14px;display:flex;justify-content:space-between;align-items:center}
.section-title{font-size:18px;font-weight:700;color:#e6edf3}
.section-sub{font-size:14px;color:#8b949e}

.trade-table-wrap{max-height:400px;overflow-y:auto;border:1px solid #30363d;border-radius:8px;margin-bottom:18px}
.trade-table{width:100%;border-collapse:collapse;font-size:14px}
.trade-table thead{position:sticky;top:0;background:#1c2128;z-index:1}
.trade-table th{padding:10px 14px;text-align:left;color:#8b949e;font-weight:600;border-bottom:1px solid #30363d;font-size:13px}
.trade-table td{padding:8px 14px;border-bottom:1px solid #21262d;color:#c9d1d9}
.trade-table tr:hover{background:#1c2128}

.opt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:14px;margin-bottom:18px}
.opt-card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.opt-card-header{padding:12px 18px;border-bottom:1px solid #21262d;background:#1c2128;display:flex;justify-content:space-between;align-items:center;cursor:pointer}
.opt-card-symbol{font-size:16px;font-weight:700;color:#e6edf3}
.opt-status{font-size:12px;padding:2px 8px;border-radius:4px;font-weight:600}
.opt-status-updated{background:#1f2d1f;color:#3fb950;border:1px solid #238636}
.opt-status-kept{background:#2d2200;color:#d29922;border:1px solid #6e5a00}
.opt-status-uptodate{background:#161b22;color:#8b949e;border:1px solid #30363d}
.opt-card-body{padding:10px 18px;font-size:13px}
.opt-row{display:flex;justify-content:space-between;padding:3px 0;color:#8b949e}
.opt-row .val{color:#c9d1d9;font-weight:600}
.opt-detail{display:none;padding:10px 18px;border-top:1px solid #21262d;font-size:12px;color:#8b949e}
.opt-detail.open{display:block}
.opt-detail table{width:100%;border-collapse:collapse}
.opt-detail th,.opt-detail td{padding:3px 8px;text-align:left}
.opt-detail th{color:#8b949e;font-weight:600}
.opt-detail td{color:#c9d1d9}

.card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden;display:flex}
.card-main{flex:1;min-width:0;display:flex;flex-direction:column}
.card-header{display:flex;justify-content:space-between;align-items:flex-start;padding:10px 18px;border-bottom:1px solid #21262d;background:#1c2128}
.card-symbol{font-size:18px;font-weight:700;color:#e6edf3;display:block}
.card-price{font-size:14px;color:#8b949e;display:block;margin-top:2px}
.card-price .val{color:#e6edf3}
.card-capital{font-size:14px;color:#8b949e}

.card-body{padding:4px 0;flex:1}
.side-row{display:flex;align-items:center;padding:10px 18px;border-bottom:1px solid #21262d;gap:12px}
.side-row:last-child{border-bottom:none}
.side-label{font-weight:700;font-size:15px;width:80px;flex-shrink:0;display:flex;flex-direction:column;gap:2px}
.side-long .side-label{color:#8b949e}
.side-short .side-label{color:#8b949e}
.side-pnl{font-size:13px;font-weight:700}
.side-pnl.pnl-pos{color:#3fb950}
.side-pnl.pnl-neg{color:#f85149}
.side-detail{font-size:15px;color:#c9d1d9;flex:1}
.side-detail .dim{color:#8b949e}
.side-detail .tag{font-size:13px;color:#e6edf3;font-weight:600;margin-left:8px}
.side-waiting{color:#8b949e;font-style:italic}
.cooldown{color:#d29922}

.params-row{display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;padding:8px 18px;border-top:1px solid #21262d;background:#1c2128}
.params-row .p-item{font-size:13px;color:#8b949e}
.params-row .p-val{color:#c9d1d9;font-weight:600}
.params-row .p-est{color:#3fb950;font-weight:700}

.card-vbar{width:130px;border-left:1px solid #21262d;background:#0d1117;padding:12px 0;position:relative}
.card-vbar .pv-area{position:relative;width:100%;height:100%}
.card-vbar .pv-track{position:absolute;top:0;bottom:0;left:50%;width:2px;transform:translateX(-50%);background:#30363d;border-radius:1px}
.card-vbar .pv-bar-wrap{position:absolute;transform:translateY(-50%);left:50%;margin-left:-10px}
.card-vbar .pv-bar{width:20px;height:2px;border-radius:1px}
.card-vbar .pv-cur .pv-bar{width:24px;height:3px;margin-left:-2px;box-shadow:0 0 4px #58a6ff}
.card-vbar .pv-dca .pv-bar{width:12px;height:2px;margin-left:4px;opacity:0.5}
.card-vbar .pv-price-wrap{position:absolute;transform:translateY(-50%);right:50%;margin-right:14px;text-align:right;white-space:nowrap}
.card-vbar .pv-price{font-size:9px;opacity:0.8;line-height:1}
.card-vbar .pv-cur .pv-price{opacity:1;font-size:10px;font-weight:700}
.card-vbar .pv-lbl-wrap{position:absolute;transform:translateY(-50%);left:50%;margin-left:14px;white-space:nowrap}
.card-vbar .pv-lbl{font-size:10px;line-height:1;font-weight:600}
.card-vbar .pv-cur .pv-lbl{font-weight:700;font-size:11px}
.card-vbar .pv-dca .pv-lbl{font-size:9px;opacity:0.5}

.error-box{background:#2d1f1f;border:1px solid #da3633;border-radius:8px;padding:16px;color:#f85149;margin-bottom:16px;font-size:16px}

.pending-row{padding:8px 18px;background:#1a1500;border-top:1px solid #3d2e00;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pending-icon{color:#d29922;font-size:16px;animation:spin 2s linear infinite}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.pending-tag{font-size:13px;color:#d29922;background:#2d2200;border:1px solid #6e5a00;border-radius:4px;padding:2px 8px}
</style>
</head>
<body>
<div class="container" id="app">
  <!-- Top-level flex: Left (header+cards) + Right (position history) -->
  <div style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap">
    <div id="left-col" style="flex:1;min-width:400px">
      <div class="header">
        <div class="header-top">
          <span class="title">DCA Trading Bot</span>
          <span class="badge" id="network-badge">---</span>
        </div>
        <div class="wallet-bar">
          <div class="wallet-item">
            <span class="wallet-label">Total Equity</span>
            <span class="wallet-value" id="wallet-equity">$---</span>
          </div>
          <div class="wallet-item">
            <span class="wallet-label">USDT+USDC</span>
            <span class="wallet-value" id="wallet-stable">$---</span>
          </div>
          <div class="wallet-item">
            <span class="wallet-label">BNB</span>
            <span class="wallet-value" id="wallet-bnb">$---</span>
          </div>
          <div class="wallet-item">
            <span class="wallet-label">Unrealized PnL</span>
            <span class="wallet-value" id="wallet-pnl">$---</span>
          </div>
          <div class="wallet-item">
            <span class="wallet-label">24h PnL</span>
            <span class="wallet-value" id="pnl-24h">$---</span>
          </div>
          <div class="wallet-item">
            <span class="wallet-label">Monthly PnL</span>
            <span class="wallet-value" id="pnl-monthly">$---</span>
          </div>
          <div class="wallet-item">
            <span class="wallet-label">Est. Monthly</span>
            <span class="wallet-value" id="est-monthly-ret">---</span>
          </div>
          <div class="wallet-item">
            <span class="wallet-label">Allocated</span>
            <span class="wallet-value" id="total-capital">$---</span>
          </div>
        </div>
        <div class="meta-bar">
          <span>Uptime: <b id="uptime">--:--:--</b></span>
          <span>Active: <b id="active">-/-</b></span>
        </div>
      </div>
      <div id="error-area"></div>
      <div class="cards-grid" id="cards"></div>
    </div>
    <div style="flex:0 0 580px;min-width:480px">
      <div class="section-header" style="margin-top:0">
        <span class="section-title">Position History</span>
        <span class="section-sub" id="trade-summary">---</span>
      </div>
      <div class="trade-table-wrap" id="trade-table-wrap" style="overflow-y:auto">
        <table class="trade-table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Side/Exit</th>
              <th>PnL</th>
              <th>DCAs</th>
              <th>Duration</th>
              <th>Closed</th>
            </tr>
          </thead>
          <tbody id="trade-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Optimization Status Section -->
  <div class="section-header">
    <span class="section-title">Optimization Status</span>
    <span class="section-sub" id="opt-summary">---</span>
  </div>
  <div class="opt-grid" id="opt-grid"></div>

  <div class="footer-bar">
    <span id="updated">--:--:--</span>
    <span>5s auto-refresh</span>
  </div>
</div>

<script>
function fmtSidePnl(v) {
  if (v == null) return '';
  const sign = v >= 0 ? '+' : '';
  const cls = v >= 0 ? 'pnl-pos' : 'pnl-neg';
  return `<span class="side-pnl ${cls}">${sign}$${v.toFixed(2)}</span>`;
}

function renderLabel(name, side) {
  let pnl = side.active && side.pnl != null ? fmtSidePnl(side.pnl) : '';
  return `<span>${name}</span>${pnl}`;
}

function renderPriceBar(snap) {
  const points = [];
  const cur = snap.current_price;
  if (!cur || cur <= 0) return '';

  const LC = '#3fb950', SC = '#f85149', NC = '#58a6ff';
  const L = snap.long, S = snap.short;
  if (L.active) {
    if (L.sl_price > 0) points.push({price: L.sl_price, label: 'L-SL', color: LC});
    if (L.avg_price > 0) points.push({price: L.avg_price, label: 'L-Avg', color: LC});
    if (L.tp_price > 0) points.push({price: L.tp_price, label: 'L-TP', color: LC});
    for (const dp of (L.dca_prices || [])) {
      points.push({price: dp, label: 'L-DCA', color: LC, isDca: true});
    }
  }
  if (S.active) {
    if (S.tp_price > 0) points.push({price: S.tp_price, label: 'S-TP', color: SC});
    if (S.avg_price > 0) points.push({price: S.avg_price, label: 'S-Avg', color: SC});
    if (S.sl_price > 0) points.push({price: S.sl_price, label: 'S-SL', color: SC});
    for (const dp of (S.dca_prices || [])) {
      points.push({price: dp, label: 'S-DCA', color: SC, isDca: true});
    }
  }
  if (points.length === 0) return '';

  points.push({price: cur, label: 'Now', color: NC, isCur: true});

  const allPrices = points.map(p => p.price);
  const mn = Math.min(...allPrices);
  const mx = Math.max(...allPrices);
  const range = mx - mn;
  if (range <= 0) return '';

  const toTop = (p) => (1 - (p - mn) / range) * 100;

  const fmtP = (p) => {
    if (p >= 1000) return p.toFixed(1);
    if (p >= 1) return p.toFixed(2);
    return p.toPrecision(4);
  };

  // Sort by price descending (top to bottom)
  points.sort((a, b) => b.price - a.price);

  // Bars at real positions, labels with overlap resolution
  const items = points.map(pt => ({...pt, barTop: toTop(pt.price), lblTop: toTop(pt.price)}));

  const minGap = 8;
  for (let pass = 0; pass < 5; pass++) {
    for (let i = 1; i < items.length; i++) {
      const gap = items[i].lblTop - items[i-1].lblTop;
      if (gap < minGap) {
        const push = (minGap - gap) / 2;
        items[i-1].lblTop -= push;
        items[i].lblTop += push;
      }
    }
    for (const it of items) {
      it.lblTop = Math.max(0, Math.min(100, it.lblTop));
    }
  }

  let markers = '';
  for (const pt of items) {
    const barCls = pt.isCur ? 'pv-bar-wrap pv-cur' : pt.isDca ? 'pv-bar-wrap pv-dca' : 'pv-bar-wrap';
    markers += `<div class="${barCls}" style="top:${pt.barTop.toFixed(1)}%">` +
      `<div class="pv-bar" style="background:${pt.color}"></div></div>`;
    // Price on left
    const pCls = pt.isCur ? 'pv-price-wrap pv-cur' : 'pv-price-wrap';
    markers += `<div class="${pCls}" style="top:${pt.lblTop.toFixed(1)}%">` +
      `<div class="pv-price" style="color:${pt.color}">${fmtP(pt.price)}</div></div>`;
    // Label on right
    const lCls = pt.isCur ? 'pv-lbl-wrap pv-cur' : pt.isDca ? 'pv-lbl-wrap pv-dca' : 'pv-lbl-wrap';
    markers += `<div class="${lCls}" style="top:${pt.lblTop.toFixed(1)}%">` +
      `<div class="pv-lbl" style="color:${pt.color}">${pt.label}</div></div>`;
  }
  return `<div class="card-vbar"><div class="pv-area"><div class="pv-track"></div>${markers}</div></div>`;
}

function renderSide(side) {
  if (side.active) {
    return `<span class="tag">DCA ${side.dca_count}/${side.max_dca}</span>`;
  }
  if (side.cooldown) {
    return `<span class="cooldown">Cooldown ${side.cooldown}</span>`;
  }
  return `<span class="side-waiting">Waiting</span>`;
}

function render(d) {
  // Badge
  const badge = document.getElementById('network-badge');
  if (d.testnet) {
    badge.textContent = 'TESTNET';
    badge.className = 'badge badge-testnet';
  } else {
    badge.textContent = 'ONLINE';
    badge.className = 'badge badge-mainnet';
  }

  // Wallet
  const w = d.wallet || {};
  document.getElementById('wallet-equity').textContent = w.total_equity_fmt || '$---';
  document.getElementById('wallet-stable').textContent = w.stable_balance_fmt || '$---';
  document.getElementById('wallet-bnb').textContent = w.bnb_value_usd_fmt || '$---';

  const pnlEl = document.getElementById('wallet-pnl');
  pnlEl.textContent = w.unrealized_pnl_fmt || '$---';
  pnlEl.className = 'wallet-value' + (w.unrealized_pnl != null ? (w.unrealized_pnl >= 0 ? ' pnl-pos' : ' pnl-neg') : '');

  const pnl24hEl = document.getElementById('pnl-24h');
  pnl24hEl.textContent = d.pnl_24h_fmt || '$---';
  pnl24hEl.className = 'wallet-value' + (d.pnl_24h != null ? (d.pnl_24h >= 0 ? ' pnl-pos' : ' pnl-neg') : '');

  const pnlMonthEl = document.getElementById('pnl-monthly');
  pnlMonthEl.textContent = d.pnl_monthly_fmt || '$---';
  pnlMonthEl.className = 'wallet-value' + (d.pnl_monthly != null ? (d.pnl_monthly >= 0 ? ' pnl-pos' : ' pnl-neg') : '');

  const estMonthEl = document.getElementById('est-monthly-ret');
  estMonthEl.textContent = d.est_monthly_usd_fmt || '---';
  estMonthEl.className = 'wallet-value' + (d.est_mpr != null ? (d.est_mpr >= 0 ? ' pnl-pos' : ' pnl-neg') : '');

  document.getElementById('total-capital').textContent = d.total_capital_fmt;
  document.getElementById('uptime').textContent = d.uptime;
  document.getElementById('active').textContent = d.active_count + '/' + d.total_positions;
  document.getElementById('updated').textContent = d.updated_at;

  // Error
  const errArea = document.getElementById('error-area');
  errArea.innerHTML = d.error ? `<div class="error-box">${d.error}</div>` : '';

  // Cards
  const container = document.getElementById('cards');
  let html = '';
  for (const snap of d.snapshots) {
    const capFmt = snap.capital.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
    html += `<div class="card">
      <div class="card-main">
        <div class="card-header">
          <div>
            <span class="card-symbol">${snap.symbol}</span>
            <span class="card-price"> $<span class="val">${snap.current_price_fmt}</span></span>
          </div>
          <span class="card-capital">$${capFmt}</span>
        </div>
        <div class="card-body">
          <div class="side-row side-long">
            <span class="side-label">${renderLabel('LONG ▲', snap.long)}</span>
            <span class="side-detail">${renderSide(snap.long)}</span>
          </div>
          <div class="side-row side-short">
            <span class="side-label">${renderLabel('SHORT ▼', snap.short)}</span>
            <span class="side-detail">${renderSide(snap.short)}</span>
          </div>
        </div>
        <div class="params-row">
          <span class="p-item">MPR <span class="p-val">${snap.mpr ? snap.mpr.toFixed(1)+'%' : '--'}</span></span>
          <span class="p-item">MDD <span class="p-val">${snap.mdd ? snap.mdd.toFixed(1)+'%' : '--'}</span></span>
          <span class="p-item">SR <span class="p-val">${snap.sharpe ? snap.sharpe.toFixed(2) : '--'}</span></span>
          <span class="p-item">Est <span class="p-est">${snap.est_monthly > 0 ? '$'+snap.est_monthly.toFixed(0)+'/mo' : '--'}</span></span>
        </div>
      </div>
      ${renderPriceBar(snap)}
      ${snap.pending_retries && snap.pending_retries.length > 0 ? `<div class="pending-row"><span class="pending-icon">⟳</span>${snap.pending_retries.map(r => `<span class="pending-tag">${r}</span>`).join('')}</div>` : ''}
    </div>`;
  }
  container.innerHTML = html;

  // ── Trade History ──
  renderTradeHistory(d.trade_history || []);

  // ── Optimization Status ──
  renderOptimization(d.optimization_info || []);

  // Sync trade table height to left column
  requestAnimationFrame(() => {
    const left = document.getElementById('left-col');
    const wrap = document.getElementById('trade-table-wrap');
    if (left && wrap) {
      const headerH = wrap.previousElementSibling ? wrap.previousElementSibling.offsetHeight : 0;
      wrap.style.maxHeight = (left.offsetHeight - headerH) + 'px';
    }
  });
}

function fmtDuration(start, end) {
  if (!start || !end) return '---';
  const ms = new Date(end) - new Date(start);
  if (isNaN(ms) || ms < 0) return '---';
  const totalMin = Math.floor(ms / 60000);
  if (totalMin < 60) return totalMin + 'm';
  const h = Math.floor(totalMin / 60);
  const m = totalMin % 60;
  if (h < 24) return h + 'h ' + m + 'm';
  const days = Math.floor(h / 24);
  const rh = h % 24;
  return days + 'd ' + rh + 'h';
}

function fmtTimeAgo(ts) {
  if (!ts) return '---';
  const dt = new Date(ts);
  if (isNaN(dt.getTime())) return '---';
  const mm = String(dt.getMonth()+1).padStart(2,'0');
  const dd = String(dt.getDate()).padStart(2,'0');
  const hh = String(dt.getHours()).padStart(2,'0');
  const mi = String(dt.getMinutes()).padStart(2,'0');
  const ss = String(dt.getSeconds()).padStart(2,'0');
  return `${mm}/${dd} ${hh}:${mi}:${ss}`;
}

function renderTradeHistory(trades) {
  const tbody = document.getElementById('trade-tbody');
  const summary = document.getElementById('trade-summary');
  if (!trades.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#8b949e;padding:20px">No closed positions</td></tr>';
    summary.textContent = '---';
    return;
  }

  let totalPnl = 0, wins = 0, losses = 0;
  trades.forEach(t => {
    totalPnl += t.pnl || 0;
    if ((t.pnl || 0) >= 0) wins++; else losses++;
  });
  const pnlClass = totalPnl >= 0 ? 'pnl-pos' : 'pnl-neg';
  const pnlSign = totalPnl >= 0 ? '+' : '';
  summary.innerHTML = `${trades.length} trades | W${wins}/L${losses} | <span class="${pnlClass}">${pnlSign}$${totalPnl.toFixed(2)}</span>`;

  let rows = '';
  for (const t of trades) {
    const pnl = t.pnl || 0;
    const pc = pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
    const ps = pnl >= 0 ? '+' : '';
    const exitCls = t.exit_type === 'TP' ? 'pnl-pos' : 'pnl-neg';
    const sideCls = t.side === 'long' ? 'pnl-pos' : 'pnl-neg';
    const sideStr = (t.side||'').toUpperCase();
    const exitStr = t.exit_type || '---';
    const sideColor = t.side==='long' ? '#3fb950' : '#f85149';
    const exitColor = t.exit_type==='TP' ? '#3fb950' : '#f85149';
    rows += `<tr>
      <td>${t.symbol || '---'}</td>
      <td><span style="color:${sideColor}">${sideStr}</span>/<span style="color:${exitColor}">${exitStr}</span></td>
      <td><span class="${pc}">${ps}$${pnl.toFixed(2)}</span></td>
      <td>${t.dca_count || 0}</td>
      <td>${fmtDuration(t.start_time, t.end_time)}</td>
      <td>${fmtTimeAgo(t.end_time)}</td>
    </tr>`;
  }
  tbody.innerHTML = rows;
}

function renderOptimization(items) {
  const grid = document.getElementById('opt-grid');
  const summary = document.getElementById('opt-summary');
  if (!items.length) {
    grid.innerHTML = '<div style="color:#8b949e;padding:20px">No optimization data</div>';
    summary.textContent = '---';
    return;
  }

  const pending = items.filter(i => i.status === 'pending_update').length;
  const upToDate = items.filter(i => i.status === 'up_to_date').length;
  summary.textContent = `${upToDate} up to date` + (pending > 0 ? ` | ${pending} pending` : '');

  // Remember which details are open before re-render
  const openIds = new Set();
  grid.querySelectorAll('.opt-detail.open').forEach(el => openIds.add(el.id));

  let html = '';
  for (const item of items) {
    const statusLabel = item.status === 'pending_update' ? 'Pending' : 'Up to date';
    const statusCls = item.status === 'pending_update' ? 'opt-status-kept' : 'opt-status-uptodate';
    const decLabel = item.last_decision ? item.last_decision : '';
    const decTime = item.last_decision_time ? fmtTimeAgo(item.last_decision_time) : '';
    const decBadge = decLabel ? `<span class="opt-status ${decLabel==='updated'?'opt-status-updated':'opt-status-kept'}">${decLabel}</span>` : '';

    const isPending = item.status === 'pending_update';

    // Show active metrics, and pending metrics side-by-side if pending
    const aMpr = item.active_mpr, aMdd = item.active_mdd, aSharpe = item.active_sharpe, aFit = item.active_fitness;
    const pMpr = item.pending_mpr, pMdd = item.pending_mdd, pSharpe = item.pending_sharpe, pFit = item.pending_fitness;
    const mpr = aMpr != null ? aMpr : pMpr;
    const mdd = aMdd != null ? aMdd : pMdd;
    const sharpe = aSharpe != null ? aSharpe : pSharpe;
    const fitness = aFit != null ? aFit : pFit;

    function fmtMetric(active, pending, fmt) {
      const aStr = active != null ? fmt(active) : '--';
      if (!isPending || pending == null || active == null) return aStr;
      const pStr = fmt(pending);
      if (aStr === pStr) return aStr;
      return `${aStr} → <span style="color:#d29922">${pStr}</span>`;
    }
    const mprStr = fmtMetric(aMpr, pMpr, v => v.toFixed(1)+'%');
    const mddStr = fmtMetric(aMdd, pMdd, v => v.toFixed(1)+'%');
    const sharpeStr = fmtMetric(aSharpe, pSharpe, v => v.toFixed(3));
    const fitnessStr = fmtMetric(aFit, pFit, v => v.toFixed(4));

    // Parameters: active vs pending
    const aParams = item.active_params || {};
    const pParams = item.pending_params || {};
    const params = aParams.long ? aParams : pParams;
    const longP = params.long || {};
    const shortP = params.short || {};

    const uid = item.symbol.replace(/[^a-zA-Z0-9]/g, '_');

    // Build param table: show active, and if pending differs, show arrow
    function paramCell(aP, pP, key, fmt) {
      const aVal = aP[key], pVal = pP[key];
      const aStr = aVal != null ? fmt(aVal) : '--';
      if (!isPending) return aStr;
      const pPar = (item.pending_params || {})[aP === longP ? 'long' : 'short'] || {};
      const newVal = pPar[key];
      if (newVal == null || aVal == null) return aStr;
      const pStr = fmt(newVal);
      if (aStr === pStr) return aStr;
      return `${aStr} → <span style="color:#d29922">${pStr}</span>`;
    }
    const f4 = v => v.toFixed(4), f3 = v => v.toFixed(3), fi = v => v;
    const pLong = (item.pending_params || {}).long || {};
    const pShort = (item.pending_params || {}).short || {};

    function pc(side, key, fmt) {
      const aP = side === 'long' ? longP : shortP;
      const pP = side === 'long' ? pLong : pShort;
      const aVal = aP[key], pVal = pP[key];
      const aStr = aVal != null ? fmt(aVal) : '--';
      if (!isPending || pVal == null || aVal == null) return aStr;
      const pStr = fmt(pVal);
      if (aStr === pStr) return aStr;
      return `${aStr} → <span style="color:#d29922">${pStr}</span>`;
    }

    html += `<div class="opt-card">
      <div class="opt-card-header" onclick="document.getElementById('opt-detail-${uid}').classList.toggle('open')">
        <span class="opt-card-symbol">${item.symbol}</span>
        <span style="display:flex;gap:6px;align-items:center">
          ${decBadge}
          <span class="opt-status ${statusCls}">${statusLabel}</span>
        </span>
      </div>
      <div class="opt-card-body">
        <div class="opt-row"><span>MPR</span><span class="val">${mprStr}</span></div>
        <div class="opt-row"><span>MDD</span><span class="val">${mddStr}</span></div>
        <div class="opt-row"><span>Sharpe</span><span class="val">${sharpeStr}</span></div>
        <div class="opt-row"><span>Fitness</span><span class="val">${fitnessStr}</span></div>
        <div class="opt-row"><span>Last Updated</span><span class="val">${item.last_updated ? fmtTimeAgo(item.last_updated) : '--'}</span></div>
        ${isPending && item.pending_created ? `<div class="opt-row"><span>Pending Since</span><span class="val" style="color:#d29922">${fmtTimeAgo(item.pending_created)}</span></div>` : ''}
      </div>
      <div class="opt-detail" id="opt-detail-${uid}">
        <table>
          <tr><th></th><th>Long</th><th>Short</th></tr>
          <tr><td>price_dev</td><td>${pc('long','price_deviation',f4)}</td><td>${pc('short','price_deviation',f4)}</td></tr>
          <tr><td>tp</td><td>${pc('long','take_profit',f4)}</td><td>${pc('short','take_profit',f4)}</td></tr>
          <tr><td>max_dca</td><td>${pc('long','max_dca',fi)}</td><td>${pc('short','max_dca',fi)}</td></tr>
          <tr><td>dev_mult</td><td>${pc('long','dev_multiplier',f3)}</td><td>${pc('short','dev_multiplier',f3)}</td></tr>
          <tr><td>vol_mult</td><td>${pc('long','vol_multiplier',f3)}</td><td>${pc('short','vol_multiplier',f3)}</td></tr>
          <tr><td>sl</td><td>${pc('long','stop_loss',f4)}</td><td>${pc('short','stop_loss',f4)}</td></tr>
        </table>
      </div>
    </div>`;
  }
  grid.innerHTML = html;

  // Restore open state
  openIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.add('open');
  });
}

async function fetchData() {
  try {
    const resp = await fetch('/api/data');
    const data = await resp.json();
    render(data);
  } catch (e) {
    console.error('Fetch error:', e);
  }
}

fetchData();
setInterval(fetchData, 5000);
</script>
</body>
</html>
"""


# ── HTTP handler ──────────────────────────────────────────────

class MonitorHandler(BaseHTTPRequestHandler):
    """GET / → HTML, GET /api/data → JSON."""

    def do_GET(self) -> None:
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/data":
            self._serve_json()
        else:
            self.send_error(404)

    def _serve_html(self) -> None:
        body = _HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_json(self) -> None:
        with _data_lock:
            payload = json.dumps(_shared_data, ensure_ascii=False)
        body = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress HTTP logs (keep terminal clean)."""
        pass


# ── Main ──────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCA Trading Bot — Web Monitor")
    parser.add_argument("--config", type=str, default="config/config.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--interval", type=int, default=5, help="Data refresh interval (seconds)")
    parser.add_argument("--testnet", action="store_true", help="Force TESTNET mode")
    parser.add_argument("--mainnet", action="store_true", help="Force MAINNET mode")
    parser.add_argument("--no-open", action="store_true", help="Disable auto browser open")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mainnet:
        testnet = False
    elif args.testnet:
        testnet = True
    else:
        testnet = os.getenv("USE_TESTNET", "true").lower() == "true"

    api = APIClient(testnet=testnet)
    start_time = time.time()

    # Data collection thread
    t = threading.Thread(
        target=_data_loop,
        args=(args.config, api, testnet, args.interval, start_time),
        daemon=True,
    )
    t.start()

    # HTTP server
    server = HTTPServer(("0.0.0.0", args.port), MonitorHandler)

    def shutdown(signum, frame):
        print("\nShutting down...")
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    url = f"http://localhost:{args.port}"
    print(f"Web monitor: {url}")

    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    server.serve_forever()


if __name__ == "__main__":
    main()
