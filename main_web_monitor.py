"""웹 기반 모니터링 대시보드 — 브라우저에서 실시간 상태 확인.

봇(main_trading.py)이 PM2로 백그라운드 실행 중일 때
별도 터미널에서 실행하면 브라우저가 자동으로 열린다.

사용법:
  python main_web_monitor.py                    # localhost:8080
  python main_web_monitor.py --port 3000        # 포트 변경
  python main_web_monitor.py --mainnet          # MAINNET 모드
"""

import argparse
import json
import math
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv("config/.env")

from main_monitor import build_snapshots
from src.common.api_client import APIClient
from src.trading.status_display import SymbolSnapshot


# ── 공유 데이터 ──────────────────────────────────────────────

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
    """유효숫자 sig자리 이상으로 가격 포맷 (콤마 포함)."""
    if value == 0:
        return "0." + "0" * (sig - 1)
    magnitude = math.floor(math.log10(abs(value)))
    decimals = max(sig - 1 - magnitude, 0)
    return f"{value:,.{decimals}f}"


def _snapshot_to_dict(snap: SymbolSnapshot) -> Dict[str, Any]:
    """SymbolSnapshot → JSON 직렬화 가능 dict."""

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
    }


# ── 데이터 수집 스레드 ───────────────────────────────────────

def _data_loop(
    config_path: str,
    api: APIClient,
    testnet: bool,
    interval: int,
    start_time: float,
) -> None:
    """백그라운드에서 주기적으로 데이터 수집."""
    global _shared_data

    while True:
        # Uptime
        elapsed = int(time.time() - start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        uptime = f"{h:02d}:{m:02d}:{s:02d}"

        snapshots, error = build_snapshots(config_path)

        equity = None
        if not error:
            try:
                equity = api.get_account_equity()
            except Exception:
                pass

        snap_dicts = [_snapshot_to_dict(sn) for sn in snapshots]

        now_str = datetime.now().strftime("%H:%M:%S")

        active_count = 0
        total_positions = 0
        for sn in snapshots:
            if sn.long_active:
                active_count += 1
            total_positions += 1
            if sn.short_active:
                active_count += 1
            total_positions += 1

        with _data_lock:
            _shared_data = {
                "snapshots": snap_dicts,
                "equity": equity,
                "equity_fmt": f"${equity:,.2f}" if equity is not None else "$---",
                "testnet": testnet,
                "uptime": uptime,
                "updated_at": now_str,
                "active_count": active_count,
                "total_positions": total_positions,
                "error": error,
            }

        time.sleep(interval)


# ── HTML 페이지 ──────────────────────────────────────────────

_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DCA Trading Bot</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:'JetBrains Mono','Fira Code','SF Mono',Consolas,monospace;font-size:14px;padding:20px}
.container{max-width:720px;margin:0 auto}

.header{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;margin-bottom:16px}
.header-top{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px}
.title{font-size:18px;font-weight:700;color:#e6edf3}
.badge{font-size:12px;font-weight:600;padding:2px 8px;border-radius:4px}
.badge-testnet{background:#1f2d1f;color:#3fb950;border:1px solid #238636}
.badge-mainnet{background:#2d1f1f;color:#f85149;border:1px solid #da3633}
.header-meta{display:flex;gap:20px;margin-top:8px;font-size:13px;color:#8b949e}
.header-meta span{display:flex;align-items:center;gap:4px}

.footer-bar{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 20px;margin-top:16px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font-size:13px;color:#8b949e}

.card{background:#161b22;border:1px solid #30363d;border-radius:8px;margin-bottom:12px;overflow:hidden}
.card-header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid #21262d;background:#1c2128}
.card-symbol{font-size:15px;font-weight:700;color:#e6edf3}
.card-price{font-size:14px;color:#8b949e}
.card-price .val{color:#e6edf3}
.card-capital{font-size:13px;color:#8b949e}

.side-row{display:flex;align-items:center;padding:10px 16px;border-bottom:1px solid #21262d;gap:12px}
.side-row:last-child{border-bottom:none}
.side-label{font-weight:700;font-size:13px;width:70px;flex-shrink:0}
.side-long .side-label{color:#3fb950}
.side-short .side-label{color:#f85149}
.side-detail{font-size:13px;color:#c9d1d9;flex:1}
.side-detail .dim{color:#8b949e}
.side-detail .tag{font-size:12px;color:#8b949e;margin-left:8px}
.side-waiting{color:#8b949e;font-style:italic}
.cooldown{color:#d29922}

.error-box{background:#2d1f1f;border:1px solid #da3633;border-radius:8px;padding:16px;color:#f85149;margin-bottom:16px}

.dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:6px}
.dot-green{background:#3fb950}
.dot-red{background:#f85149}
</style>
</head>
<body>
<div class="container" id="app">
  <div class="header">
    <div class="header-top">
      <span class="title">DCA Trading Bot</span>
      <span class="badge" id="network-badge">---</span>
    </div>
    <div class="header-meta">
      <span>Uptime: <b id="uptime">--:--:--</b></span>
      <span>Equity: <b id="equity">$---</b></span>
      <span>Active: <b id="active">-/-</b></span>
    </div>
  </div>

  <div id="error-area"></div>
  <div id="cards"></div>

  <div class="footer-bar">
    <span id="updated">--:--:--</span>
    <span>5s auto-refresh</span>
  </div>
</div>

<script>
function renderSide(side, label, arrow) {
  if (side.active) {
    let tp = side.tp_price_fmt ? `<span class="tag">TP ${side.tp_price_fmt}</span>` : '';
    return `<span>${side.amount.toFixed(4)} @ ${side.avg_price_fmt}</span>` +
           `<span class="tag">DCA ${side.dca_count}/${side.max_dca}</span>${tp}`;
  }
  if (side.cooldown) {
    return `<span class="cooldown">-- 대기 (쿨다운 ${side.cooldown}) --</span>`;
  }
  return `<span class="side-waiting">-- 대기 중 --</span>`;
}

function render(d) {
  // Header
  const badge = document.getElementById('network-badge');
  if (d.testnet) {
    badge.textContent = 'TESTNET';
    badge.className = 'badge badge-testnet';
  } else {
    badge.textContent = 'MAINNET';
    badge.className = 'badge badge-mainnet';
  }
  document.getElementById('uptime').textContent = d.uptime;
  document.getElementById('equity').textContent = d.equity_fmt;
  document.getElementById('active').textContent = d.active_count + '/' + d.total_positions;
  document.getElementById('updated').textContent = d.updated_at;

  // Error
  const errArea = document.getElementById('error-area');
  if (d.error) {
    errArea.innerHTML = `<div class="error-box">${d.error}</div>`;
  } else {
    errArea.innerHTML = '';
  }

  // Cards
  const container = document.getElementById('cards');
  let html = '';
  for (const snap of d.snapshots) {
    html += `<div class="card">
      <div class="card-header">
        <div>
          <span class="card-symbol">${snap.symbol}</span>
          <span class="card-price"> &nbsp; $<span class="val">${snap.current_price_fmt}</span></span>
        </div>
        <span class="card-capital">Capital: $${snap.capital.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2})}</span>
      </div>
      <div class="side-row side-long">
        <span class="side-label">LONG ▲</span>
        <span class="side-detail">${renderSide(snap.long, 'LONG', '▲')}</span>
      </div>
      <div class="side-row side-short">
        <span class="side-label">SHORT ▼</span>
        <span class="side-detail">${renderSide(snap.short, 'SHORT', '▼')}</span>
      </div>
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


# ── HTTP 핸들러 ──────────────────────────────────────────────

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
        """HTTP 로그 억제 (터미널 깔끔하게)."""
        pass


# ── 메인 ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCA Trading Bot — 웹 모니터")
    parser.add_argument("--config", type=str, default="config/config.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--interval", type=int, default=5, help="데이터 갱신 간격 (초)")
    parser.add_argument("--testnet", action="store_true", help="TESTNET 모드 강제")
    parser.add_argument("--mainnet", action="store_true", help="MAINNET 모드 강제")
    parser.add_argument("--no-open", action="store_true", help="브라우저 자동 열기 비활성화")
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

    # 데이터 수집 스레드
    t = threading.Thread(
        target=_data_loop,
        args=(args.config, api, testnet, args.interval, start_time),
        daemon=True,
    )
    t.start()

    # HTTP 서버
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
