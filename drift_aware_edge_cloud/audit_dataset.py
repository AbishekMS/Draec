"""Dataset audit tool -- inspect delimiter-separated data files by CONTENT.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Why this exists
---------------
A new dataset drop has to be identified from what is inside the files, never from
their names, extensions or sizes. Two releases of the same corpus can carry
identical filenames, and a truncated or re-encoded download can carry a plausible
size. So this reads every file: shape, column names, timestamp range, sampling
interval, and for label files the class distribution and timestamp integrity.

No filename is written into this script. Directories come from argv and files are
discovered by globbing, so the tool is dataset- and version-agnostic and cannot
smuggle a file-role assignment into Python -- roles are config's business.

It only reads. Nothing here writes, moves or deletes a data file.

Run:
    PYTHONIOENCODING=utf-8 python audit_dataset.py <dir> [<dir> ...]
    PYTHONIOENCODING=utf-8 python audit_dataset.py --json <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

DATA_SUFFIXES = (".csv", ".txt")
# A column is treated as the time axis if its name matches, case-insensitively.
# Discovered, not assumed: the report states which name was found.
TIME_CANDIDATES = ("timestamp", "time", "datetime", "date")
# A column is a candidate label if its name matches AND it has few distinct
# values. Both conditions are reported separately so the caller can judge.
LABEL_CANDIDATES = ("label", "attack", "anomaly", "class", "target", "y")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff_delimiter(path: Path) -> str:
    """Guess the delimiter from the header line by counting candidates."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.readline()
    counts = {d: head.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else ","


def audit_file(path: Path) -> dict:
    rec: dict = {
        "file": path.name,
        "dir": str(path.parent),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    delim = sniff_delimiter(path)
    rec["delimiter"] = {",": "comma", ";": "semicolon", "\t": "tab",
                        "|": "pipe"}[delim]
    try:
        df = pd.read_csv(path, sep=delim, low_memory=False)
    except Exception as exc:                                # pragma: no cover
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec

    rec["rows"] = int(len(df))
    rec["n_columns"] = int(df.shape[1])
    cols = [str(c) for c in df.columns]
    rec["columns"] = cols
    rec["columns_head"] = cols[:8]
    rec["columns_tail"] = cols[-8:]
    rec["column_hash"] = hashlib.sha256(
        "|".join(cols).encode("utf-8")).hexdigest()[:16]
    rec["missing_cells_total"] = int(df.isna().sum().sum())

    # --- time axis -----------------------------------------------------------
    tcol = next((c for c in cols if c.strip().lower() in TIME_CANDIDATES), None)
    rec["time_column"] = tcol
    if tcol is not None:
        ts = pd.to_datetime(df[tcol], errors="coerce")
        rec["time_unparseable"] = int(ts.isna().sum())
        if ts.notna().any():
            rec["first_timestamp"] = str(ts.min())
            rec["last_timestamp"] = str(ts.max())
            rec["duration_hours"] = round(
                (ts.max() - ts.min()).total_seconds() / 3600.0, 4)
            rec["monotonic_increasing"] = bool(ts.is_monotonic_increasing)
            rec["duplicate_timestamps"] = int(ts.duplicated().sum())
            deltas = ts.sort_values().diff().dropna().dt.total_seconds()
            if len(deltas):
                vc = deltas.value_counts().head(4)
                rec["interval_seconds_mode"] = float(vc.index[0])
                rec["interval_distribution"] = {
                    f"{float(k)}s": int(v) for k, v in vc.items()}
                rec["interval_min_seconds"] = float(deltas.min())
                rec["interval_max_seconds"] = float(deltas.max())
                rec["gaps_over_mode"] = int((deltas > vc.index[0]).sum())

    # --- low-cardinality columns (label candidates) --------------------------
    lowcard = {}
    for c in cols:
        if c == tcol:
            continue
        nun = int(df[c].nunique(dropna=True))
        if nun <= 12:
            vals = sorted(df[c].dropna().unique().tolist(),
                          key=lambda v: str(v))
            counts = df[c].value_counts(dropna=False).to_dict()
            lowcard[c] = {
                "n_unique": nun,
                "values": [str(v) for v in vals],
                "distribution": {str(k): int(v) for k, v in counts.items()},
                "missing": int(df[c].isna().sum()),
                "name_matches_label": any(
                    t in c.strip().lower() for t in LABEL_CANDIDATES),
            }
    rec["low_cardinality_columns"] = lowcard
    rec["label_candidates"] = [c for c, v in lowcard.items()
                              if v["name_matches_label"]]
    rec["zero_variance_columns"] = int(sum(
        1 for c in cols if c != tcol and df[c].nunique(dropna=True) <= 1))
    return rec


def render(rec: dict) -> list[str]:
    if "error" in rec:
        return [f"  {rec['file']}: UNREADABLE -- {rec['error']}"]
    out = [
        f"  {rec['file']}",
        f"    sha256      {rec['sha256']}",
        f"    size        {rec['size_bytes']:,} bytes, {rec['delimiter']}"
        f"-separated",
        f"    shape       {rec['rows']:,} rows x {rec['n_columns']} columns"
        f"   (column-set hash {rec['column_hash']})",
        f"    columns[:8] {rec['columns_head']}",
        f"    columns[-8:] {rec['columns_tail']}",
        f"    missing     {rec['missing_cells_total']:,} cells; "
        f"{rec['zero_variance_columns']} zero-variance columns",
    ]
    if rec.get("time_column"):
        out += [
            f"    time column '{rec['time_column']}' "
            f"({rec.get('time_unparseable', 0)} unparseable)",
            f"    range       {rec.get('first_timestamp')} -> "
            f"{rec.get('last_timestamp')}"
            f"   ({rec.get('duration_hours')} h)",
            f"    sampling    mode {rec.get('interval_seconds_mode')}s, "
            f"min {rec.get('interval_min_seconds')}s, "
            f"max {rec.get('interval_max_seconds')}s, "
            f"{rec.get('gaps_over_mode')} gaps beyond mode",
            f"    integrity   monotonic={rec.get('monotonic_increasing')}, "
            f"duplicate timestamps={rec.get('duplicate_timestamps')}",
        ]
    else:
        out.append("    time column NONE FOUND")
    # Only label-*candidate* columns are printed in full. HAI has dozens of
    # low-cardinality actuator channels; dumping every distribution would bury
    # the one thing this audit is trying to establish. Full detail is in --json.
    out.append(f"    low-cardinality columns: "
               f"{len(rec['low_cardinality_columns'])} "
               f"(<=12 distinct values, excluding the time axis)")
    if rec["label_candidates"]:
        out.append(f"    label candidates by NAME: {rec['label_candidates']}")
        for c in rec["label_candidates"]:
            v = rec["low_cardinality_columns"][c]
            out.append(f"      '{c}': {v['n_unique']} unique {v['values']}, "
                       f"missing={v['missing']}")
            out.append(f"          distribution {v['distribution']}")
    else:
        out.append("    label candidates by NAME: none")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", help="directories to audit")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args(argv)

    records: list[dict] = []
    for d in args.dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"not a directory: {p}", file=sys.stderr)
            return 2
        files = sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in DATA_SUFFIXES)
        if not args.json:
            print("=" * 78)
            print(f"{p}  --  {len(files)} data file(s)")
            print("=" * 78)
        for f in files:
            rec = audit_file(f)
            records.append(rec)
            if not args.json:
                for ln in render(rec):
                    print(ln)
                print()

    if args.json:
        print(json.dumps(records, indent=1, default=str))
        return 0

    # --- duplicate detection across everything audited ----------------------
    print("=" * 78)
    print("CONTENT-IDENTICAL FILES (same SHA-256)")
    print("=" * 78)
    by_hash: dict[str, list[str]] = {}
    for r in records:
        by_hash.setdefault(r["sha256"], []).append(
            f"{Path(r['dir']).name}/{r['file']}")
    dupes = {k: v for k, v in by_hash.items() if len(v) > 1}
    if not dupes:
        print("  none -- every audited file is content-distinct")
    for dg, names in dupes.items():
        print(f"  {dg[:16]}...  {names}")

    print()
    print("=" * 78)
    print("COLUMN-SET GROUPS (same schema)")
    print("=" * 78)
    by_cols: dict[str, list[str]] = {}
    for r in records:
        if "column_hash" in r:
            by_cols.setdefault(r["column_hash"], []).append(
                f"{Path(r['dir']).name}/{r['file']} ({r['n_columns']} cols)")
    for ch, names in sorted(by_cols.items()):
        print(f"  {ch}  {names}")

    print()
    print("=" * 78)
    print("CHRONOLOGY (by first timestamp)")
    print("=" * 78)
    timed = [r for r in records if r.get("first_timestamp")]
    for r in sorted(timed, key=lambda r: r["first_timestamp"]):
        print(f"  {r['first_timestamp']} -> {r['last_timestamp']}  "
              f"{Path(r['dir']).name}/{r['file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
