"""24/7 live trading main process (PM2-driven)."""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv("config/.env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binance Futures bidirectional DCA trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trade on Testnet with default settings
  python main_trading.py --testnet

  # Use custom config file on Mainnet
  python main_trading.py --config config/my_config.json

  # Run on Mainnet
  python main_trading.py --mainnet
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Trading config file path (default: config/config.json)",
    )

    parser.add_argument(
        "--testnet",
        action="store_true",
        default=None,
        help="Use Binance Testnet (default: USE_TESTNET from .env)",
    )

    parser.add_argument(
        "--mainnet",
        action="store_true",
        help="Use Binance Mainnet",
    )

    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip Mainnet confirmation prompt (for non-interactive environments like PM2)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulation mode without real orders (not implemented)",
    )

    return parser.parse_args()


def validate_environment() -> bool:
    """Validate environment variables."""
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")

    if not api_key or api_key == "your_api_key_here":
        print("Error: BINANCE_API_KEY is not set.")
        print("  1. Copy .env.example to .env")
        print("  2. Set your Binance API key")
        return False

    if not api_secret or api_secret == "your_api_secret_here":
        print("Error: BINANCE_API_SECRET is not set.")
        return False

    return True


def main() -> None:
    args = parse_args()

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    # Determine Testnet/Mainnet
    if args.mainnet:
        testnet = False
    elif args.testnet:
        testnet = True
    else:
        testnet = os.getenv("USE_TESTNET", "true").lower() == "true"

    # Load TradingConfig
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

    # Fetch total balance from Binance (for display)
    from src.common.api_client import APIClient

    try:
        api = APIClient(testnet=testnet)
        total_balance = api.get_account_equity()
    except Exception as e:
        print(f"Error: Failed to fetch Binance balance - {e}")
        sys.exit(1)

    # Calculate total weight
    total_weight = sum(cfg.weight for cfg in config.symbols.values())

    # Confirmation message
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

    if not testnet and not args.confirm:
        print("\nWARNING: Running in MAINNET mode. Real funds will be used!")
        confirm = input("Do you want to continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Cancelled")
            sys.exit(0)

    # Run executor
    from src.trading.executor import TradingExecutor

    executor = TradingExecutor(
        config=config,
        testnet=testnet,
        config_path=args.config,
    )
    executor.run()


if __name__ == "__main__":
    main()
