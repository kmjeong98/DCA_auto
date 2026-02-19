"""24/7 실전 매매 메인 프로세스 (PM2 구동)."""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance Futures 양방향 DCA 트레이딩 봇",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # Testnet에서 config.json 기본 설정으로 거래
  python main_trading.py --testnet

  # Mainnet에서 커스텀 설정 파일 사용
  python main_trading.py --config my_config.json

  # Mainnet 실행
  python main_trading.py --mainnet
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="트레이딩 설정 파일 경로 (기본값: config.json)",
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

    # TradingConfig 로드
    from src.common.trading_config import TradingConfig

    try:
        config = TradingConfig.load(args.config)
        config.validate()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Binance에서 총 잔고 조회 (정보 표시용)
    from src.common.api_client import APIClient

    try:
        api = APIClient(testnet=testnet)
        total_balance = api.get_account_equity()
    except Exception as e:
        print(f"Error: Binance 잔고 조회 실패 - {e}")
        sys.exit(1)

    # weight 합계 계산
    total_weight = sum(cfg.weight for cfg in config.symbols.values())

    # 확인 메시지
    print("=" * 60)
    print("Binance Futures DCA Trading Bot")
    print("=" * 60)
    print(f"  Network:      {'TESTNET' if testnet else 'MAINNET'}")
    print(f"  Config:       {args.config}")
    print(f"  Balance:      ${total_balance:,.2f}")
    print(f"  Cooldown:     {config.cooldown_hours} hours")
    print(f"  Total Weight: {total_weight:.4f}")
    print(f"  Symbols:")
    for safe_name, sym_cfg in config.symbols.items():
        allocated = total_balance * sym_cfg.weight
        print(
            f"    {sym_cfg.symbol:>10s}  "
            f"weight={sym_cfg.weight:.4f}  "
            f"capital=${allocated:,.2f}"
        )
    print("=" * 60)

    if not testnet:
        print("\nWARNING: MAINNET 모드입니다. 실제 자금이 사용됩니다!")
        confirm = input("계속하시겠습니까? (yes/no): ")
        if confirm.lower() != "yes":
            print("취소됨")
            sys.exit(0)

    # Executor 실행
    from src.trading.executor import TradingExecutor

    executor = TradingExecutor(
        config=config,
        testnet=testnet,
        config_path=args.config,
    )
    executor.run()


if __name__ == "__main__":
    main()
