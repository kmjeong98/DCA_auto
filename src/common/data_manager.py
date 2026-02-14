# data_manager.py
# Reusable CCXT OHLCV cache manager (incremental head/tail download)
# - Stores per (symbol,timeframe) under ./ohlcv_cache/
# - Keeps Timestamp as int64(ms) for fast load/merge
# - Uses Feather if pyarrow is available, otherwise pickle fallback

from __future__ import annotations

import os
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

import ccxt
import pandas as pd


__all__ = ["DataManager"]


# Keep this local so the module is self-contained.
TIMEFRAME_TO_MIN = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240,
    "1d": 1440,
}


def timeframe_minutes(tf: str) -> int:
    if tf not in TIMEFRAME_TO_MIN:
        raise ValueError(f"Unsupported timeframe: {tf}")
    return TIMEFRAME_TO_MIN[tf]


class DataManager:
    # Default exchange instance (override if needed)
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})

    # Cache root (override per environment if you want)
    @staticmethod
    def _project_root() -> Path:
        """Find project root by walking up to common marker files/dirs."""
        start = Path(__file__).resolve()
        markers = {".git", "pyproject.toml", "requirements.txt", "setup.cfg", "setup.py"}
        for parent in [start.parent, *start.parents]:
            for marker in markers:
                if (parent / marker).exists():
                    return parent
        return start.parent

    CACHE_ROOT = str(_project_root.__func__() / "data" / "ohlcv_cache")

    _CACHE_EXT = None
    _PANDAS_READ = None
    _PANDAS_WRITE = None

    # Binance often supports 1500; fallback to 1000 automatically on error
    _MAX_LIMIT = 1500

    @staticmethod
    def _init_cache_format():
        """Prefer Feather (pyarrow) when available, otherwise pickle."""
        if DataManager._CACHE_EXT is not None:
            return

        os.makedirs(DataManager.CACHE_ROOT, exist_ok=True)

        try:
            import pyarrow  # noqa: F401
            DataManager._CACHE_EXT = "feather"
            DataManager._PANDAS_READ = pd.read_feather
            DataManager._PANDAS_WRITE = lambda df, path: df.to_feather(path)
        except Exception:
            DataManager._CACHE_EXT = "pkl"
            DataManager._PANDAS_READ = pd.read_pickle
            DataManager._PANDAS_WRITE = lambda df, path: df.to_pickle(path, protocol=4)

    @staticmethod
    def _sanitize_symbol(symbol: str) -> str:
        return symbol.replace("/", "_").replace(":", "_")

    @staticmethod
    def _cache_paths(symbol: str, timeframe: str) -> Tuple[str, str]:
        DataManager._init_cache_format()
        base = f"{DataManager._sanitize_symbol(symbol)}__{timeframe}"
        data_path = os.path.join(DataManager.CACHE_ROOT, f"{base}.{DataManager._CACHE_EXT}")
        meta_path = os.path.join(DataManager.CACHE_ROOT, f"{base}.meta.json")
        return data_path, meta_path

    @staticmethod
    def _load_meta(meta_path: str) -> Optional[dict]:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _save_meta(meta_path: str, meta: dict) -> None:
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        os.replace(tmp, meta_path)

    @staticmethod
    def _tf_ms(timeframe: str) -> int:
        return int(timeframe_minutes(timeframe) * 60_000)

    @staticmethod
    def _fetch_ohlcv_range(symbol: str, timeframe: str, since_ms: int, until_ms: int) -> Optional[pd.DataFrame]:
        """
        Fetch [since_ms, until_ms] with pagination.
        Returns DF columns:
          Timestamp(int64 ms), Open/High/Low/Close/V(float32)
        """
        if since_ms > until_ms:
            return None

        tf_ms = DataManager._tf_ms(timeframe)
        now_ms = int(time.time() * 1000)
        until = min(int(until_ms), now_ms)

        all_rows = []
        since = int(since_ms)

        limit_try = DataManager._MAX_LIMIT
        retry = 0

        while True:
            if since > until:
                break

            try:
                ohlcv = DataManager.exchange.fetch_ohlcv(symbol, timeframe, since, limit_try)
                retry = 0
            except Exception as e:
                # fallback limit to 1000 if exchange rejects 1500
                if limit_try > 1000:
                    limit_try = 1000
                    continue

                retry += 1
                if retry >= 8:
                    raise RuntimeError(f"fetch_ohlcv failed too many times: {symbol} {timeframe} err={e}")
                time.sleep(0.8 * retry)
                continue

            if not ohlcv:
                break

            for row in ohlcv:
                if row[0] > until:
                    break
                all_rows.append(row)

            last_ts = ohlcv[-1][0]
            if last_ts >= until:
                break

            since = last_ts + tf_ms  # next page

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "V"])
        df["Timestamp"] = df["Timestamp"].astype("int64")
        for col in ["Open", "High", "Low", "Close", "V"]:
            df[col] = df[col].astype("float32")

        df = df.drop_duplicates("Timestamp").sort_values("Timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def fetch_data(
        symbol: str,
        timeframe: str = "15m",
        years: int = 4,
        start_date_str: Optional[str] = None,
        end_date_str: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Cache-first load with incremental extension (head/tail).
        - If cache missing: downloads full requested range and stores.
        - If cache exists but doesn't cover requested range: downloads only missing head/tail and merges.
        - Returns DF filtered to the requested range.

        start_date_str/end_date_str: "YYYY-MM-DD" in UTC.
        If start_date_str is None: uses now - years.
        If end_date_str is None: uses now.
        """
        data_path, meta_path = DataManager._cache_paths(symbol, timeframe)
        tf_ms = DataManager._tf_ms(timeframe)

        now_ms = int(time.time() * 1000)

        if start_date_str:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        else:
            start_dt = datetime.utcnow() - timedelta(days=365 * years)

        if end_date_str:
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            # include the whole day end (23:59:59.999)
            end_ms = int((end_dt + timedelta(days=1)).timestamp() * 1000) - 1
        else:
            end_ms = now_ms

        start_ms = int(start_dt.timestamp() * 1000)

        if force_refresh:
            for p in (data_path, meta_path):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        cache_exists = os.path.exists(data_path)
        meta = DataManager._load_meta(meta_path) if cache_exists else None

        # Cache miss -> full download
        if not cache_exists:
            print(f" 🔄 [CACHE MISS] Downloading: {symbol} ({timeframe})")
            full = DataManager._fetch_ohlcv_range(symbol, timeframe, start_ms, end_ms)
            if full is None:
                return None

            DataManager._PANDAS_WRITE(full, data_path)
            DataManager._save_meta(meta_path, {
                "symbol": symbol,
                "timeframe": timeframe,
                "min_ts": int(full["Timestamp"].min()),
                "max_ts": int(full["Timestamp"].max()),
                "rows": int(len(full)),
                "updated_ms": now_ms,
            })
            return full[(full["Timestamp"] >= start_ms) & (full["Timestamp"] <= end_ms)].reset_index(drop=True)

        # Recover meta if missing/broken
        have_min = meta.get("min_ts") if meta else None
        have_max = meta.get("max_ts") if meta else None

        if have_min is None or have_max is None:
            print(f" ⚠️ [CACHE META RECOVER] {symbol} ({timeframe})")
            df0 = DataManager._PANDAS_READ(data_path)
            have_min = int(df0["Timestamp"].min())
            have_max = int(df0["Timestamp"].max())
            DataManager._save_meta(meta_path, {
                "symbol": symbol,
                "timeframe": timeframe,
                "min_ts": have_min,
                "max_ts": have_max,
                "rows": int(len(df0)),
                "updated_ms": now_ms,
            })

        pre_df = None
        post_df = None

        if start_ms < have_min:
            print(f" 🔄 [CACHE EXTEND HEAD] {symbol} ({timeframe})")
            pre_df = DataManager._fetch_ohlcv_range(symbol, timeframe, start_ms, have_min - tf_ms)

        if end_ms > have_max + tf_ms:
            print(f" 🔄 [CACHE EXTEND TAIL] {symbol} ({timeframe})")
            post_df = DataManager._fetch_ohlcv_range(symbol, timeframe, have_max + tf_ms, end_ms)

        # Update cache only if needed
        if pre_df is not None or post_df is not None:
            base_df = DataManager._PANDAS_READ(data_path)
            frames = [base_df]
            if pre_df is not None:
                frames.append(pre_df)
            if post_df is not None:
                frames.append(post_df)

            merged = pd.concat(frames, ignore_index=True)
            merged = merged.drop_duplicates("Timestamp").sort_values("Timestamp").reset_index(drop=True)

            DataManager._PANDAS_WRITE(merged, data_path)
            DataManager._save_meta(meta_path, {
                "symbol": symbol,
                "timeframe": timeframe,
                "min_ts": int(merged["Timestamp"].min()),
                "max_ts": int(merged["Timestamp"].max()),
                "rows": int(len(merged)),
                "updated_ms": now_ms,
            })

        df = DataManager._PANDAS_READ(data_path)
        df = df[(df["Timestamp"] >= start_ms) & (df["Timestamp"] <= end_ms)].reset_index(drop=True)
        return df
