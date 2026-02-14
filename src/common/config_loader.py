"""params/ 폴더 내 JSON 로드 및 감시."""

from pathlib import Path
import json
from typing import Any, Dict


class ConfigLoader:
    def __init__(self, params_dir: str = "params") -> None:
        self.params_dir = Path(params_dir)

    def load(self, symbol: str) -> Dict[str, Any]:
        file_path = self.params_dir / f"{symbol}.json"
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
