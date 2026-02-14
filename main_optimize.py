"""GA 최적화 실행 스크립트 (주기적 구동)."""

from src.optimization.ga_engine import GAEngine


def main() -> None:
    engine = GAEngine()
    engine.run()


if __name__ == "__main__":
    main()
