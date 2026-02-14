"""24/7 실전 매매 메인 프로세스 (PM2 구동)."""

from src.trading.executor import TradingExecutor


def main() -> None:
    executor = TradingExecutor()
    executor.run()


if __name__ == "__main__":
    main()
