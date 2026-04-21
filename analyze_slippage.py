"""Analyze actual slippage from trader logs + trade event logs.

Strategy:
1. SL slippage: Parse "SL X placed - Price: Y" from trader logs,
   find the last SL placement before each SL fill event, compare prices.
2. Market order slippage: Find TP→re-entry pairs (within 5s) and compare
   TP fill price (≈mark price) with re-entry fill price.
"""

import os
import re
import json
import glob
from datetime import datetime, timedelta, timezone
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_trade_events():
    """Load all trade events from JSONL logs."""
    trade_dir = os.path.join(BASE_DIR, "data", "logs", "trades")
    events = []
    for f in sorted(glob.glob(os.path.join(trade_dir, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    events.sort(key=lambda x: x["timestamp"])
    return events


def parse_sl_placements_from_trader_logs():
    """Parse SL placement entries from per-symbol trader logs.

    Returns dict: symbol -> list of {time, side, price}
    """
    log_dir = os.path.join(BASE_DIR, "data", "logs")
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) - INFO - "
        r"SL (LONG|SHORT) placed - Price: ([\d.]+)"
    )

    placements = defaultdict(list)

    for f in glob.glob(os.path.join(log_dir, "trader_*.log")):
        # Extract symbol from filename: trader_ETH_USDT.log -> ETH/USDT
        basename = os.path.basename(f)
        sym_part = basename.replace("trader_", "").replace(".log", "")
        symbol = sym_part.replace("_", "/")

        with open(f) as fh:
            for line in fh:
                m = pattern.search(line)
                if m:
                    time_str = m.group(1).replace(",", ".")
                    side = m.group(2).lower()
                    price = float(m.group(3))

                    # Parse time
                    dt = datetime.strptime(time_str.split(".")[0], "%Y-%m-%d %H:%M:%S")

                    placements[symbol].append({
                        "time": dt,
                        "side": side,
                        "price": price,
                    })

    # Sort each symbol's placements by time
    for sym in placements:
        placements[sym].sort(key=lambda x: x["time"])

    return placements


def find_sl_order_price(placements, symbol, side, sl_fill_time):
    """Find the last SL placement price before the SL fill time."""
    if symbol not in placements:
        return None

    # Parse fill time
    fill_dt = datetime.fromisoformat(sl_fill_time.replace("+00:00", ""))

    best = None
    for p in placements[symbol]:
        if p["side"] == side and p["time"] < fill_dt:
            best = p
        elif p["time"] >= fill_dt:
            break

    return best


def analyze_sl_slippage(events, placements):
    """Analyze SL slippage by matching placement prices with fill prices."""
    sl_events = [e for e in events if e["event"] == "SL"]
    results = []

    for sl in sl_events:
        symbol = sl["symbol"]
        side = sl["side"]
        fill_price = sl["price"]

        placement = find_sl_order_price(placements, symbol, side, sl["timestamp"])
        if placement is None:
            continue

        order_price = placement["price"]

        # Slippage: how much worse is the fill vs the order price
        if side == "long":
            # Long SL = SELL. Order at X, filled at Y. Slippage = (X - Y) / X
            slip_pct = (order_price - fill_price) / order_price * 100
        else:
            # Short SL = BUY. Order at X, filled at Y. Slippage = (Y - X) / X
            slip_pct = (fill_price - order_price) / order_price * 100

        results.append({
            "time": sl["timestamp"][:19],
            "symbol": symbol,
            "side": side,
            "order_price": order_price,
            "fill_price": fill_price,
            "slip_pct": slip_pct,
        })

    return results


def analyze_reentry_slippage(events):
    """Analyze market order slippage from TP→re-entry pairs."""
    entry_events = [e for e in events if e["event"] == "ENTRY"]
    tp_events = [e for e in events if e["event"] == "TP"]

    results = []
    for entry in entry_events:
        entry_dt = datetime.fromisoformat(entry["timestamp"].replace("+00:00", ""))
        for tp in tp_events:
            tp_dt = datetime.fromisoformat(tp["timestamp"].replace("+00:00", ""))
            diff = (entry_dt - tp_dt).total_seconds()
            if 0 < diff < 5 and entry["symbol"] == tp["symbol"]:
                tp_price = tp["price"]
                entry_price = entry["price"]

                if entry["side"] == "long":
                    slip = (entry_price - tp_price) / tp_price * 100
                else:
                    slip = (tp_price - entry_price) / tp_price * 100

                results.append({
                    "time": entry["timestamp"][:19],
                    "symbol": entry["symbol"],
                    "side": entry["side"],
                    "ref_price": tp_price,
                    "fill_price": entry_price,
                    "slip_pct": slip,
                })
                break

    return results


def print_report(sl_results, reentry_results):
    print("=" * 80)
    print("SLIPPAGE ANALYSIS REPORT")
    print("=" * 80)

    # --- SL ---
    print(f"\n{'='*80}")
    print(f"STOP-LOSS SLIPPAGE (order price vs fill price)  —  {len(sl_results)} fills")
    print(f"{'='*80}")

    if sl_results:
        print(f"{'Time':<22s} {'Symbol':<12s} {'Side':<6s} {'Order':>10s} {'Fill':>10s} {'Slip%':>8s}")
        print("-" * 72)
        for e in sl_results:
            print(f"{e['time']:<22s} {e['symbol']:<12s} {e['side']:<6s} "
                  f"{e['order_price']:>10.4f} {e['fill_price']:>10.4f} {e['slip_pct']:>7.4f}%")

        slips = [e["slip_pct"] for e in sl_results]
        slips_sorted = sorted(slips)
        n = len(slips)
        print(f"\n  Count:   {n}")
        print(f"  Average: {sum(slips)/n:.4f}%")
        print(f"  Median:  {slips_sorted[n//2]:.4f}%")
        print(f"  P75:     {slips_sorted[int(n*0.75)]:.4f}%")
        print(f"  P90:     {slips_sorted[int(n*0.90)]:.4f}%")
        print(f"  Max:     {max(slips):.4f}%")
        print(f"  Min:     {min(slips):.4f}%")

        # Per-symbol breakdown
        by_symbol = defaultdict(list)
        for e in sl_results:
            by_symbol[e["symbol"]].append(e["slip_pct"])
        print(f"\n  Per-symbol average:")
        for sym in sorted(by_symbol):
            vals = by_symbol[sym]
            print(f"    {sym:<12s}: {sum(vals)/len(vals):>7.4f}%  (n={len(vals)})")
    else:
        print("  No SL data matched.")

    # --- Re-entry ---
    print(f"\n{'='*80}")
    print(f"MARKET ORDER SLIPPAGE (TP→re-entry)  —  {len(reentry_results)} pairs")
    print(f"{'='*80}")

    if reentry_results:
        print(f"{'Time':<22s} {'Symbol':<12s} {'Side':<6s} {'Ref':>10s} {'Fill':>10s} {'Slip%':>8s}")
        print("-" * 72)
        for e in reentry_results:
            print(f"{e['time']:<22s} {e['symbol']:<12s} {e['side']:<6s} "
                  f"{e['ref_price']:>10.4f} {e['fill_price']:>10.4f} {e['slip_pct']:>7.4f}%")

        slips = [e["slip_pct"] for e in reentry_results]
        n = len(slips)
        print(f"\n  Count:   {n}")
        print(f"  Average: {sum(slips)/n:.4f}%")
        print(f"  Median:  {sorted(slips)[n//2]:.4f}%")
        print(f"  Max:     {max(slips):.4f}%")
        print(f"  Min:     {min(slips):.4f}%")

    # --- Summary ---
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")
    if sl_results:
        sl_slips = [e["slip_pct"] for e in sl_results]
        avg_sl = sum(sl_slips) / len(sl_slips) / 100
        p90_sl = sorted(sl_slips)[int(len(sl_slips) * 0.90)] / 100
        print(f"  SL avg  slippage: {avg_sl*100:.4f}% = {avg_sl:.5f}")
        print(f"  SL P90  slippage: {p90_sl*100:.4f}% = {p90_sl:.5f}")

    if reentry_results:
        re_slips = [e["slip_pct"] for e in reentry_results]
        avg_re = sum(re_slips) / len(re_slips) / 100
        print(f"  Entry avg slippage: {avg_re*100:.4f}% = {avg_re:.5f}")

    print(f"\n  Current slip_rate: 0.001 (0.1%)")
    all_market_slips = []
    if sl_results:
        all_market_slips.extend([e["slip_pct"] for e in sl_results])
    if reentry_results:
        all_market_slips.extend([e["slip_pct"] for e in reentry_results])

    if all_market_slips:
        overall_avg = sum(all_market_slips) / len(all_market_slips) / 100
        if overall_avg < 0.001:
            print(f"  → Current 0.1% is conservative (actual avg: {overall_avg*100:.4f}%)")
        else:
            print(f"  → Consider adjusting slip_rate to ~{overall_avg:.4f} ({overall_avg*100:.3f}%)")


def main():
    print("Loading trade events...", end=" ", flush=True)
    events = load_trade_events()
    n_sl = sum(1 for e in events if e["event"] == "SL")
    n_entry = sum(1 for e in events if e["event"] == "ENTRY")
    print(f"{len(events)} events ({n_entry} entries, {n_sl} SLs)")

    print("Parsing trader logs for SL placements...", end=" ", flush=True)
    placements = parse_sl_placements_from_trader_logs()
    total_placements = sum(len(v) for v in placements.values())
    print(f"{total_placements} SL placements across {len(placements)} symbols")

    sl_results = analyze_sl_slippage(events, placements)
    reentry_results = analyze_reentry_slippage(events)

    print_report(sl_results, reentry_results)


if __name__ == "__main__":
    main()
