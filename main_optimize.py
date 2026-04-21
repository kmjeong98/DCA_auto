"""GA optimization runner script (periodic execution)."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("config/.env")

from src.optimization.ga_engine import GAEngine, SimulationConfig, GAConfig

CONFIG_PATH = "config/optimize_config.json"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load optimize_config.json."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Run: cp config/optimize_config.example.json config/optimize_config.json"
        )
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _backtest_30d(engine: GAEngine, ticker: str):
    """Run 30-day backtest for a ticker using its saved params. Returns stats or None."""
    from datetime import datetime, timedelta, timezone
    from src.common.data_manager import DataManager

    safe_name = ticker.replace("/", "_")
    params_path = Path("data/params") / f"{safe_name}.json"
    if not params_path.exists():
        return None

    try:
        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        genome = engine._params_to_genome(params)
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)
        df = DataManager.fetch_data(
            ticker,
            timeframe=engine.sim_config.timeframe,
            start_date_str=start.strftime("%Y-%m-%d"),
            end_date_str=end.strftime("%Y-%m-%d"),
        )
        if df is None or len(df) == 0:
            return None

        data = engine._df_to_arrays(df, engine.sim_config.timeframe)
        return engine._evaluate_genome(genome, data)
    except Exception:
        return None


def _send_completion_telegram(tickers: list, engine: GAEngine) -> None:
    """Send optimization completion summary via Telegram using decision log + 30d backtest."""
    from main_daily_report import send_telegram

    log_path = Path("data/logs/optimization_decisions.jsonl")
    if not log_path.exists():
        send_telegram("<b>Optimize Done</b>\nNo decision log found.")
        return

    # Read latest decision for each ticker
    decisions = {}
    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        try:
            rec = json.loads(line)
            decisions[rec["symbol"]] = rec
        except Exception:
            continue

    updated = 0
    rows = []
    for ticker in tickers:
        d = decisions.get(ticker)
        if not d:
            continue
        sym = ticker.split("/")[0]
        dec = d["decision"]
        if dec in ("new", "updated"):
            updated += 1
            tag = "NEW" if dec == "new" else "UPD"
        else:
            tag = "KEPT"

        rows.append(f" {sym:<5s}[{tag}]")
        if "old_mpr" in d:
            rows.append(f"  old  {d['old_mpr']:>5.1f}%  {d['old_mdd']:>5.1f}%  {d['old_sharpe']:.2f}")
        rows.append(f"  new  {d['new_mpr']:>5.1f}%  {d['new_mdd']:>5.1f}%  {d['new_sharpe']:.2f}")

        # 30d backtest
        bt = _backtest_30d(engine, ticker)
        if bt:
            rows.append(f"  30d  {bt['mpr']:>5.1f}%  {bt['mdd']:>5.1f}%  {bt['sharpe']:.2f}")

    lines = [f"<b>Optimize Done</b>  {updated}/{len(tickers)} updated"]
    if rows:
        header = f"       {'MPR':>6s}  {'MDD':>6s}  {'S':>4s}"
        lines.append("<pre>" + header + "\n" + "\n".join(rows) + "</pre>")

    send_telegram("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="GA optimization for DCA trading bot")
    parser.add_argument(
        "--tickers",
        nargs="+",
        metavar="TICKER",
        help="Specific tickers to optimize (e.g. BTC/USDT ETH/USDT). "
             "Defaults to all tickers in optimize_config.json.",
    )
    args = parser.parse_args()

    config = load_config()

    sim_config = SimulationConfig.from_dict(config.get("simulation", {}))
    ga_config = GAConfig.from_dict(config.get("ga", {}))

    if args.tickers:
        tickers = args.tickers
    else:
        tickers = config.get("tickers", [])

    if not tickers:
        raise ValueError("No tickers specified. Use --tickers or check config/optimize_config.json.")

    engine = GAEngine(sim_config=sim_config, ga_config=ga_config)
    engine.run(tickers=tickers)

    try:
        _send_completion_telegram(tickers, engine)
    except Exception as e:
        print(f"Telegram notification failed: {e}")


if __name__ == "__main__":
    main()
