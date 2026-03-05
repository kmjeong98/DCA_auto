"""Load and watch JSON files in the params/ folder."""

from pathlib import Path
import json
import re
from typing import Any, Dict, Optional


def _usdt_fallback(safe_name: str) -> Optional[str]:
    """Return USDT equivalent of a non-USDT safe name, or None.

    Example: 'ETH_USDC' -> 'ETH_USDT', 'ETH_USDT' -> None
    """
    m = re.match(r"^(.+)_(?!USDT$)([A-Z]+)$", safe_name)
    if m:
        return f"{m.group(1)}_USDT"
    return None


class ConfigLoader:
    def __init__(self, params_dir: str = "data/params") -> None:
        self.params_dir = Path(params_dir)

    def load(self, symbol: str) -> Dict[str, Any]:
        """Load params for symbol, falling back to USDT version if not found."""
        file_path = self.params_dir / f"{symbol}.json"
        if not file_path.exists():
            fallback = _usdt_fallback(symbol)
            if fallback:
                fallback_path = self.params_dir / f"{fallback}.json"
                if fallback_path.exists():
                    with fallback_path.open("r", encoding="utf-8") as f:
                        return json.load(f)
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
