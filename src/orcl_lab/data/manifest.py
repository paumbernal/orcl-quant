"""Data provenance manifest.

Every dataset the pipeline produces is recorded in ``data/manifest.json`` with its source, the
UTC timestamp at which it was retrieved, the date range it covers and a content hash. Each entry
also carries a ``kind`` so historical data, consensus forecasts, assumptions and model outputs
can never be confused with one another:

    HISTORICAL   reported / traded data (SEC filings, exchange prices, FRED rates)
    CONSENSUS    third-party analyst forecasts captured as a dated snapshot
    ASSUMPTION   scenario inputs chosen by the analyst (see config.yaml)
    MODEL        outputs produced by this repository's models
    MANUAL       hand-curated overrides supplied by the user (source URL required)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

KINDS = {"HISTORICAL", "CONSENSUS", "ASSUMPTION", "MODEL", "MANUAL"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))

    def record(
        self,
        name: str,
        *,
        kind: str,
        source: str,
        file: Path | None = None,
        df: pd.DataFrame | None = None,
        notes: str = "",
        retrieved_at: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        entry: dict[str, Any] = {
            "kind": kind,
            "source": source,
            "retrieved_at_utc": retrieved_at or utc_now(),
            "notes": notes,
        }
        if file is not None:
            entry["file"] = str(Path(file).as_posix())
            if Path(file).exists():
                entry["sha256"] = file_sha256(Path(file))
        if df is not None and len(df):
            entry["rows"] = int(len(df))
            idx = df.index
            if isinstance(idx, pd.DatetimeIndex):
                entry["first_date"] = str(idx.min().date())
                entry["last_date"] = str(idx.max().date())
        entry.update(extra)
        self.entries[name] = entry
        return entry

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, name: str) -> dict[str, Any] | None:
        return self.entries.get(name)

    def source_line(self, *names: str) -> str:
        """Human-readable provenance string for chart footers."""
        parts = []
        for n in names:
            e = self.entries.get(n)
            if not e:
                continue
            when = e.get("retrieved_at_utc", "")[:10]
            parts.append(f"{e['source']} (retrieved {when})")
        return "; ".join(parts)
