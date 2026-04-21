"""
Reconcile local trade logs (TP/SL) with actual Binance exchange data.

Fetches real trade history from the exchange, matches each local TP/SL event
to the corresponding exchange trades by timestamp+symbol+side, and overwrites
the local amount/pnl with the actual values.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("config/.env")

from binance.um_futures import UMFutures

API = UMFutures(
    key=os.getenv("BINANCE_API_KEY"),
    secret=os.getenv("BINANCE_API_SECRET"),
    base_url="https://fapi.binance.com",
)

TRADE_DIR = Path("data/logs/trades")

# All traded symbols (from config)
SYMBOLS = [
    "ETH/USDT", "ETH/USDC", "DOGE/USDT", "SUI/USDT",
    "SOL/USDT", "SOL/USDC", "ZEC/USDT", "ADA/USDT", "LINK/USDT",
]


def to_binance(symbol: str) -> str:
    return symbol.replace("/", "")


def fetch_all_trades(binance_symbol: str, start_time: int) -> list:
    """Fetch all trades from start_time to now using 7-day windows."""
    all_trades = []
    now_ms = int(time.time() * 1000)
    window_ms = 7 * 24 * 60 * 60 * 1000  # 7 days
    seen_ids = set()

    cursor = start_time
    while cursor < now_ms:
        end = min(cursor + window_ms, now_ms)
        params = {
            "symbol": binance_symbol,
            "startTime": cursor,
            "endTime": end,
            "limit": 1000,
        }

        trades = API.get_account_trades(**params)
        for t in trades:
            if t["id"] not in seen_ids:
                seen_ids.add(t["id"])
                all_trades.append(t)

        # If we got 1000, there may be more in this window — paginate by fromId
        while len(trades) == 1000:
            last_id = trades[-1]["id"]
            trades = API.get_account_trades(
                symbol=binance_symbol, fromId=last_id, limit=1000
            )
            trades = [t for t in trades if t["id"] != last_id and t["time"] <= end]
            for t in trades:
                if t["id"] not in seen_ids:
                    seen_ids.add(t["id"])
                    all_trades.append(t)
            time.sleep(0.05)

        cursor = end + 1
        time.sleep(0.1)

    all_trades.sort(key=lambda t: t["time"])
    return all_trades


def group_trades_by_order(trades: list) -> dict:
    """Group trades by orderId, summing qty and realizedPnl."""
    orders = defaultdict(lambda: {"qty": 0.0, "pnl": 0.0, "trades": []})
    for t in trades:
        oid = t["orderId"]
        orders[oid]["qty"] += float(t["qty"])
        orders[oid]["pnl"] += float(t["realizedPnl"])
        orders[oid]["trades"].append(t)

    # Attach metadata from first trade in each order
    result = {}
    for oid, data in orders.items():
        first = data["trades"][0]
        result[oid] = {
            "orderId": oid,
            "symbol": first["symbol"],
            "side": first["side"],
            "positionSide": first["positionSide"],
            "qty": data["qty"],
            "pnl": data["pnl"],
            "time": first["time"],
            "price": float(first["price"]),
        }
    return result


def load_local_logs(symbol: str) -> list:
    """Load all JSONL log entries for a symbol, with file path info."""
    safe = symbol.replace("/", "_")
    entries = []
    for path in sorted(TRADE_DIR.glob(f"{safe}_*.jsonl")):
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                entry["_file"] = str(path)
                entry["_line"] = i
                entries.append(entry)
    return entries


def match_logs_to_exchange(tp_sl_logs: list, exchange_orders: dict):
    """Match local TP/SL log entries to exchange orders.

    Strategy: for each log entry, find the closest exchange order by time
    that matches side and hasn't been claimed yet. Uses multi-pass with
    increasing time tolerance (60s, 5min, 30min, 2h, no limit).
    """
    # Build candidate list per (positionSide, orderSide)
    candidates = {}
    for oid, order in exchange_orders.items():
        key = (order["positionSide"], order["side"])
        candidates.setdefault(key, []).append(order)

    # Sort candidates by time
    for key in candidates:
        candidates[key].sort(key=lambda o: o["time"])

    claimed = set()  # order IDs already matched
    results = [None] * len(tp_sl_logs)

    # Multi-pass with increasing time tolerance
    tolerances = [60_000, 300_000, 1_800_000, 7_200_000, float("inf")]

    for tol in tolerances:
        for i, entry in enumerate(tp_sl_logs):
            if results[i] is not None:
                continue

            log_ts = datetime.fromisoformat(entry["timestamp"])
            log_ts_ms = int(log_ts.timestamp() * 1000)

            if entry["side"] == "long":
                key = ("LONG", "SELL")
            else:
                key = ("SHORT", "BUY")

            best = None
            best_diff = float("inf")

            for order in candidates.get(key, []):
                if order["orderId"] in claimed:
                    continue
                diff = abs(order["time"] - log_ts_ms)
                if diff < best_diff and diff < tol:
                    best_diff = diff
                    best = order

            if best is not None:
                results[i] = best
                claimed.add(best["orderId"])

    return results


def main():
    # Earliest log entry across all files
    earliest_ts = None
    for path in TRADE_DIR.glob("*.jsonl"):
        with open(path) as f:
            first = f.readline().strip()
            if first:
                entry = json.loads(first)
                ts = datetime.fromisoformat(entry["timestamp"])
                if earliest_ts is None or ts < earliest_ts:
                    earliest_ts = ts

    if earliest_ts is None:
        print("No trade logs found.")
        return

    start_ms = int(earliest_ts.timestamp() * 1000)
    print(f"Fetching trades from {earliest_ts.strftime('%Y-%m-%d')}...\n")

    # Phase 1: Fetch all exchange trades
    all_exchange_orders = {}  # symbol -> {orderId -> order_data}

    for symbol in SYMBOLS:
        bsym = to_binance(symbol)
        print(f"Fetching {symbol}...", end=" ", flush=True)
        trades = fetch_all_trades(bsym, start_ms)
        orders = group_trades_by_order(trades)
        all_exchange_orders[symbol] = orders
        print(f"{len(trades)} trades, {len(orders)} orders")
        time.sleep(0.2)

    # Phase 2: Match and compare
    print("\n=== Comparing TP/SL logs with exchange data ===\n")

    mismatches = []  # (log_entry, exchange_order, file, line)

    for symbol in SYMBOLS:
        logs = load_local_logs(symbol)
        exchange_orders = all_exchange_orders[symbol]

        tp_sl_logs = [e for e in logs if e["event"] in ("TP", "SL")]
        matches = match_logs_to_exchange(tp_sl_logs, exchange_orders)

        for entry, match in zip(tp_sl_logs, matches):
            if match is None:
                print(f"  NO MATCH: {symbol} {entry['event']} {entry['side']} @ {entry['timestamp']}")
                continue

            local_amt = entry["amount"]
            exchange_amt = match["qty"]
            local_pnl = entry["pnl"]
            exchange_pnl = match["pnl"]

            amt_diff = abs(local_amt - exchange_amt)
            pnl_diff = abs(local_pnl - exchange_pnl)

            if amt_diff > 1e-9 or pnl_diff > 1e-9:
                mismatches.append({
                    "entry": entry,
                    "exchange": match,
                    "file": entry["_file"],
                    "line": entry["_line"],
                })
                print(
                    f"  MISMATCH: {symbol} {entry['event']} {entry['side']} "
                    f"@ {entry['timestamp'][:19]}"
                )
                print(f"    amount: local={local_amt} -> exchange={exchange_amt} (diff={amt_diff})")
                print(f"    pnl:    local={local_pnl} -> exchange={exchange_pnl} (diff={pnl_diff})")

    print(f"\nTotal mismatches: {len(mismatches)}")

    if not mismatches:
        print("All logs match exchange data!")
        return

    # Phase 3: Fix the logs
    print("\n=== Fixing trade logs ===\n")

    # Group fixes by file
    fixes_by_file = defaultdict(list)
    for m in mismatches:
        fixes_by_file[m["file"]].append(m)

    for filepath, fixes in fixes_by_file.items():
        # Read all lines
        with open(filepath) as f:
            lines = f.readlines()

        fix_lines = {fix["line"]: fix for fix in fixes}

        new_lines = []
        fixed_count = 0
        for i, line in enumerate(lines):
            if i in fix_lines:
                fix = fix_lines[i]
                entry = json.loads(line.strip())
                old_amt = entry["amount"]
                old_pnl = entry["pnl"]
                entry["amount"] = fix["exchange"]["qty"]
                entry["pnl"] = fix["exchange"]["pnl"]
                new_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
                fixed_count += 1
            else:
                new_lines.append(line)

        with open(filepath, "w") as f:
            f.writelines(new_lines)

        print(f"  Fixed {fixed_count} entries in {Path(filepath).name}")

    print(f"\nDone! Fixed {len(mismatches)} entries total.")


if __name__ == "__main__":
    main()
