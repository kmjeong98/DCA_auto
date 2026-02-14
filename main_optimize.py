"""GA 최적화 실행 스크립트 (주기적 구동)."""

from src.optimization.ga_engine import GAEngine

# ========================================
# 최적화할 종목 목록 (여기서 수정)
# ========================================
TICKERS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "ZEC/USDT",
    "SUI/USDT",
]


def main() -> None:
    engine = GAEngine()
    engine.run(tickers=TICKERS)


if __name__ == "__main__":
    main()
