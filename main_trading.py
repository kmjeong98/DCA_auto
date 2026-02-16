"""24/7 실전 매매 메인 프로세스 (PM2 구동)."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance Futures 양방향 DCA 트레이딩 봇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Testnet에서 BTC/USDT 거래
  python main_trading.py --symbols BTC_USDT --testnet

  # Mainnet에서 여러 심볼 거래
  python main_trading.py --symbols BTC_USDT ETH_USDT --capital 5000

  # params/ 폴더의 모든 심볼 거래
  python main_trading.py --capital 10000
        """,
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="거래할 심볼 (예: BTC_USDT ETH_USDT). 지정하지 않으면 params/ 폴더에서 자동 로드",
    )

    parser.add_argument(
        "--testnet",
        action="store_true",
        default=None,
        help="Binance Testnet 사용 (기본값: .env의 USE_TESTNET)",
    )

    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Binance Mainnet 사용",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="트레이딩 자본 (기본값: .env의 INITIAL_CAPITAL 또는 1000)",
    )

    parser.add_argument(
        "--cooldown",
        type=int,
        default=None,
        help="SL 후 쿨다운 시간 (시간 단위, 기본값: 6)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 주문 없이 시뮬레이션 모드 (미구현)",
    )

    return parser.parse_args()


def validate_environment() -> bool:
    """환경 변수 검증."""
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or api_key == "your_api_key_here":
        print("Error: BINANCE_API_KEY가 설정되지 않았습니다.")
        print("  1. .env.example 파일을 .env로 복사하세요")
        print("  2. Binance API 키를 설정하세요")
        return False

    if not api_secret or api_secret == "your_api_secret_here":
        print("Error: BINANCE_API_SECRET가 설정되지 않았습니다.")
        return False

    return True


def validate_params(symbols: list) -> bool:
    """심볼 파라미터 파일 존재 여부 확인."""
    params_dir = Path("params")

    if not params_dir.exists():
        print(f"Error: params/ 디렉토리가 없습니다.")
        return False

    missing = []
    for symbol in symbols:
        param_file = params_dir / f"{symbol}.json"
        if not param_file.exists():
            missing.append(symbol)

    if missing:
        print(f"Error: 파라미터 파일이 없습니다: {missing}")
        print("  main_optimize.py를 실행하여 파라미터를 생성하세요")
        return False

    return True


def main() -> None:
    args = parse_args()

    # 환경 검증
    if not validate_environment():
        sys.exit(1)

    # Testnet/Mainnet 결정
    if args.mainnet:
        testnet = False
    elif args.testnet:
        testnet = True
    else:
        testnet = os.getenv("USE_TESTNET", "true").lower() == "true"

    # 자본 결정
    capital = args.capital
    if capital is None:
        capital = float(os.getenv("INITIAL_CAPITAL", "1000"))

    # 쿨다운 결정
    cooldown = args.cooldown
    if cooldown is None:
        cooldown = int(os.getenv("COOLDOWN_HOURS", "6"))

    # 심볼 결정
    symbols = args.symbols
    if symbols is None:
        # params/ 폴더에서 자동 발견
        params_dir = Path("params")
        if params_dir.exists():
            symbols = [f.stem for f in params_dir.glob("*.json")]

    if not symbols:
        print("Error: 거래할 심볼이 없습니다.")
        print("  --symbols BTC_USDT 옵션을 사용하거나")
        print("  params/ 폴더에 파라미터 파일을 생성하세요")
        sys.exit(1)

    # 파라미터 파일 검증
    if not validate_params(symbols):
        sys.exit(1)

    # 확인 메시지
    print("=" * 60)
    print("Binance Futures DCA Trading Bot")
    print("=" * 60)
    print(f"  Network:  {'TESTNET' if testnet else 'MAINNET'}")
    print(f"  Capital:  ${capital:,.2f}")
    print(f"  Cooldown: {cooldown} hours")
    print(f"  Symbols:  {', '.join(s.replace('_', '/') for s in symbols)}")
    print("=" * 60)

    if not testnet:
        print("\n⚠️  WARNING: MAINNET 모드입니다. 실제 자금이 사용됩니다!")
        confirm = input("계속하시겠습니까? (yes/no): ")
        if confirm.lower() != "yes":
            print("취소됨")
            sys.exit(0)

    # 심볼 포맷 변환 (BTC_USDT -> BTC/USDT)
    formatted_symbols = [s.replace("_", "/") for s in symbols]

    # Executor 실행
    from src.trading.executor import TradingExecutor

    executor = TradingExecutor(
        symbols=formatted_symbols,
        testnet=testnet,
        capital=capital,
        cooldown_hours=cooldown,
    )
    executor.run()


if __name__ == "__main__":
    main()

