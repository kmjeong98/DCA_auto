"""Load and watch JSON files in the params/ folder."""

from pathlib import Path
import json
from typing import Any, Dict


class ConfigLoader:
    def __init__(self, params_dir: str = "data/params") -> None:
        self.params_dir = Path(params_dir)

    def load(self, symbol: str) -> Dict[str, Any]:
        file_path = self.params_dir / f"{symbol}.json"
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
