"""Load OpenAI API key from env or project openai_api.txt (never log the value)."""
from __future__ import annotations

import os
from pathlib import Path

from experiment.config import ROOT

_KEY_FILE = ROOT / "openai_api.txt"


def ensure_openai_api_key() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    if _KEY_FILE.exists():
        key = _KEY_FILE.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return True
    return False
