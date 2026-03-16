from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any

def append_jsonl(path: str | Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")