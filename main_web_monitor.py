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
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
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


def build_snapshots(config_path: str) -> tuple:
    """Build snapshot list from config, state, params, and margin files.

    Returns:
        (snapshots, error_msg)
    """
    cfg = _load_json(Path(config_path))
    if cfg is None:
        return [], f"Config not found: {config_path}"

    cooldown_hours = int(cfg.get("cooldown_hours", 6))
    symbols_cfg = cfg.get("symbols", {})

    snapshots: List[SymbolSnapshot] = []

    for safe_name, sym_val in symbols_cfg.items():
        symbol = safe_name.replace("_", "/")

        state_path = Path("data/state") / f"{safe_name}_state.json"
        state_data = _load_json(state_path) or {}

        params_data = _load_json(Path("data/active_params") / f"{safe_name}.json")
        if params_data is None:
            params_data = _load_json(Path("data/params") / f"{safe_name}.json")
        if params_data is None:
            params_data = {}

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

    return snapshots, None


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
    """Fetch Futures wallet info: USDT, BNB balances + unrealized PnL."""
    info: Dict[str, Any] = {
        "usdt_balance": None,
        "bnb_balance": None,
        "bnb_value_usd": None,
        "unrealized_pnl": None,
        "total_wallet_value": None,
    }
    try:
        account = api.client.account()
        info["unrealized_pnl"] = float(account.get("totalUnrealizedProfit", 0))
        total_margin = float(account.get("totalMarginBalance", 0))
        info["total_wallet_value"] = total_margin
    except Exception:
        pass

    try:
        balances = api.client.balance()
        for b in balances:
            if b["asset"] == "USDT":
                info["usdt_balance"] = float(b["balance"])
            elif b["asset"] == "BNB":
                info["bnb_balance"] = float(b["balance"])
    except Exception:
        pass

    if info["bnb_balance"] is not None:
        try:
            bnb_price = api.get_mark_price("BNBUSDT")
            info["bnb_value_usd"] = info["bnb_balance"] * bnb_price
        except Exception:
            pass

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

    while True:
        # Uptime
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        uptime = f"{h:02d}:{m:02d}:{s:02d}"

        snapshots, error = build_snapshots(config_path)

        wallet = {"usdt_balance": None, "bnb_balance": None, "bnb_value_usd": None,
                  "unrealized_pnl": None, "total_wallet_value": None}
        equity = None
        if not error:
            try:
                wallet = _fetch_wallet_info(api)
                equity = wallet["total_wallet_value"]
            except Exception:
                pass

        snap_dicts = [_snapshot_to_dict(sn) for sn in snapshots]

        now_str = datetime.now().strftime("%H:%M:%S")

        active_count = 0
        total_positions = 0
        total_capital = 0.0
        for sn in snapshots:
            total_capital += sn.capital
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
                "wallet": {
                    "usdt_balance": wallet["usdt_balance"],
                    "usdt_balance_fmt": _fmt_usd(wallet["usdt_balance"]),
                    "bnb_balance": wallet["bnb_balance"],
                    "bnb_balance_fmt": _fmt_bnb(wallet["bnb_balance"]),
                    "bnb_value_usd": wallet["bnb_value_usd"],
                    "bnb_value_usd_fmt": _fmt_usd(wallet["bnb_value_usd"]),
                    "unrealized_pnl": wallet["unrealized_pnl"],
                    "unrealized_pnl_fmt": _fmt_pnl(wallet["unrealized_pnl"]),
                    "total_wallet_value": wallet["total_wallet_value"],
                    "total_wallet_value_fmt": _fmt_usd(wallet["total_wallet_value"]),
                },
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
.container{max-width:1200px;margin:0 auto}

.header{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:18px 24px;margin-bottom:18px}
.header-top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.title{font-size:22px;font-weight:700;color:#e6edf3}
.badge{font-size:14px;font-weight:600;padding:3px 10px;border-radius:4px}
.badge-testnet{background:#1f2d1f;color:#3fb950;border:1px solid #238636}
.badge-mainnet{background:#2d1f1f;color:#f85149;border:1px solid #da3633}

.wallet-bar{display:grid;grid-template-columns:repeat(5,1fr);gap:16px;margin-top:14px;padding-top:14px;border-top:1px solid #30363d}
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

.card{background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}
.card-header{display:flex;justify-content:space-between;align-items:center;padding:12px 18px;border-bottom:1px solid #21262d;background:#1c2128}
.card-symbol{font-size:18px;font-weight:700;color:#e6edf3}
.card-price{font-size:16px;color:#8b949e}
.card-price .val{color:#e6edf3}
.card-capital{font-size:14px;color:#8b949e}

.card-body{padding:4px 0}
.side-row{display:flex;align-items:center;padding:10px 18px;border-bottom:1px solid #21262d;gap:12px}
.side-row:last-child{border-bottom:none}
.side-label{font-weight:700;font-size:15px;width:80px;flex-shrink:0}
.side-long .side-label{color:#3fb950}
.side-short .side-label{color:#f85149}
.side-detail{font-size:15px;color:#c9d1d9;flex:1}
.side-detail .dim{color:#8b949e}
.side-detail .tag{font-size:13px;color:#8b949e;margin-left:8px}
.side-waiting{color:#8b949e;font-style:italic}
.cooldown{color:#d29922}

.params-row{display:flex;gap:12px;padding:8px 18px;border-top:1px solid #21262d;background:#1c2128;flex-wrap:wrap}
.params-row .p-item{font-size:13px;color:#8b949e}
.params-row .p-val{color:#c9d1d9;font-weight:600}
.params-row .p-est{color:#3fb950;font-weight:700}

.error-box{background:#2d1f1f;border:1px solid #da3633;border-radius:8px;padding:16px;color:#f85149;margin-bottom:16px;font-size:16px}

.pending-row{padding:8px 18px;background:#1a1500;border-top:1px solid #3d2e00;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.pending-icon{color:#d29922;font-size:16px;animation:spin 2s linear infinite}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.pending-tag{font-size:13px;color:#d29922;background:#2d2200;border:1px solid #6e5a00;border-radius:4px;padding:2px 8px}
</style>
</head>
<body>
<div class="container" id="app">
  <div class="header">
    <div class="header-top">
      <span class="title">DCA Trading Bot</span>
      <span class="badge" id="network-badge">---</span>
    </div>
    <div class="wallet-bar">
      <div class="wallet-item">
        <span class="wallet-label">Total Value</span>
        <span class="wallet-value" id="wallet-total">$---</span>
      </div>
      <div class="wallet-item">
        <span class="wallet-label">USDT</span>
        <span class="wallet-value" id="wallet-usdt">$---</span>
      </div>
      <div class="wallet-item">
        <span class="wallet-label">BNB</span>
        <span class="wallet-value" id="wallet-bnb">---</span>
        <span class="wallet-sub" id="wallet-bnb-usd">$---</span>
      </div>
      <div class="wallet-item">
        <span class="wallet-label">Unrealized PnL</span>
        <span class="wallet-value" id="wallet-pnl">$---</span>
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

  <div class="footer-bar">
    <span id="updated">--:--:--</span>
    <span>5s auto-refresh</span>
  </div>
</div>

<script>
function renderSide(side) {
  if (side.active) {
    let tp = side.tp_price_fmt ? `<span class="tag">TP ${side.tp_price_fmt}</span>` : '';
    return `<span>${side.amount.toFixed(4)} @ ${side.avg_price_fmt}</span>` +
           `<span class="tag">DCA ${side.dca_count}/${side.max_dca}</span>${tp}`;
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
    badge.textContent = 'MAINNET';
    badge.className = 'badge badge-mainnet';
  }

  // Wallet
  const w = d.wallet || {};
  document.getElementById('wallet-total').textContent = w.total_wallet_value_fmt || '$---';
  document.getElementById('wallet-usdt').textContent = w.usdt_balance_fmt || '$---';
  document.getElementById('wallet-bnb').textContent = w.bnb_balance_fmt || '---';
  document.getElementById('wallet-bnb-usd').textContent = w.bnb_value_usd_fmt ? '≈ ' + w.bnb_value_usd_fmt : '';

  const pnlEl = document.getElementById('wallet-pnl');
  pnlEl.textContent = w.unrealized_pnl_fmt || '$---';
  pnlEl.className = 'wallet-value' + (w.unrealized_pnl != null ? (w.unrealized_pnl >= 0 ? ' pnl-pos' : ' pnl-neg') : '');

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
      <div class="card-header">
        <div>
          <span class="card-symbol">${snap.symbol}</span>
          <span class="card-price"> $<span class="val">${snap.current_price_fmt}</span></span>
        </div>
        <span class="card-capital">$${capFmt}</span>
      </div>
      <div class="card-body">
        <div class="side-row side-long">
          <span class="side-label">LONG ▲</span>
          <span class="side-detail">${renderSide(snap.long)}</span>
        </div>
        <div class="side-row side-short">
          <span class="side-label">SHORT ▼</span>
          <span class="side-detail">${renderSide(snap.short)}</span>
        </div>
      </div>
      <div class="params-row">
        <span class="p-item">MPR <span class="p-val">${snap.mpr ? snap.mpr.toFixed(1)+'%' : '--'}</span></span>
        <span class="p-item">MDD <span class="p-val">${snap.mdd ? snap.mdd.toFixed(1)+'%' : '--'}</span></span>
        <span class="p-item">SR <span class="p-val">${snap.sharpe ? snap.sharpe.toFixed(2) : '--'}</span></span>
        <span class="p-item">Est <span class="p-est">${snap.est_monthly > 0 ? '$'+snap.est_monthly.toFixed(0)+'/mo' : '--'}</span></span>
      </div>
      ${snap.pending_retries && snap.pending_retries.length > 0 ? `<div class="pending-row"><span class="pending-icon">⟳</span>${snap.pending_retries.map(r => `<span class="pending-tag">${r}</span>`).join('')}</div>` : ''}
    </div>`;
  }
  container.innerHTML = html;
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
