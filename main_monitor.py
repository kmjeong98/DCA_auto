"""독립 모니터링 스크립트 — state 파일 + Binance API로 터미널 상태 표시.

봇(main_trading.py)이 PM2로 백그라운드 실행 중일 때
별도 터미널에서 실행하여 상태를 실시간 확인한다.

사용법:
  python main_monitor.py                    # 기본: config/config.json, 5초 간격
  python main_monitor.py --interval 10      # 10초 간격
  python main_monitor.py --config config/my_config.json
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv("config/.env")

from src.common.api_client import APIClient
from src.trading.status_display import StatusDisplay, SymbolSnapshot


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """JSON 파일 로드. 실패 시 None."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_snapshots(
    config_path: str,
) -> tuple:
    """config, state, params, margin 파일에서 스냅샷 리스트 생성.

    Returns:
        (snapshots, error_msg)
    """
    cfg = load_json(Path(config_path))
    if cfg is None:
        return [], f"Config not found: {config_path}"

    cooldown_hours = int(cfg.get("cooldown_hours", 6))
    symbols_cfg = cfg.get("symbols", {})

    snapshots: List[SymbolSnapshot] = []

    for safe_name, sym_val in symbols_cfg.items():
        symbol = safe_name.replace("_", "/")

        # state 파일
        state_path = Path("data/state") / f"{safe_name}_state.json"
        state_data = load_json(state_path) or {}

        # params 파일 (active_params 우선, 없으면 params)
        params_data = load_json(Path("data/active_params") / f"{safe_name}.json")
        if params_data is None:
            params_data = load_json(Path("data/params") / f"{safe_name}.json")
        if params_data is None:
            params_data = {}

        # margin 파일
        margin_path = Path("data/margins") / f"{safe_name}_margin.json"
        margin_data = load_json(margin_path) or {}

        snap = SymbolSnapshot.from_state_files(
            symbol=symbol,
            state_data=state_data,
            params_data=params_data,
            margin_data=margin_data,
            cooldown_hours=cooldown_hours,
        )
        snapshots.append(snap)

    return snapshots, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCA Trading Bot — 독립 모니터",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="트레이딩 설정 파일 경로 (기본값: config/config.json)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="갱신 간격 초 (기본값: 5)",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="TESTNET 모드 강제",
    )
    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="MAINNET 모드 강제",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mainnet:
        testnet = False
    elif args.testnet:
        testnet = True
    else:
        testnet = os.getenv("USE_TESTNET", "true").lower() == "true"

    # Binance API 클라이언트 (실제 잔고 조회용)
    api = APIClient(testnet=testnet)

    display = StatusDisplay(force_tty=True)

    # Ctrl+C 시 깔끔한 종료
    running = True

    def handle_signal(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while running:
        snapshots, error = build_snapshots(args.config)

        if error:
            sys.stdout.write(f"\033[H\033[J{error}\n")
            sys.stdout.flush()
        else:
            # 실제 계좌 잔고 조회
            try:
                equity = api.get_account_equity()
            except Exception:
                equity = None

            display.update(snapshots, testnet, account_equity=equity)

        # interval 동안 대기 (0.5초 단위로 체크하여 빠른 종료 지원)
        waited = 0.0
        while waited < args.interval and running:
            time.sleep(0.5)
            waited += 0.5

    # 종료 시 화면 정리
    sys.stdout.write("\033[H\033[JMonitor stopped.\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
