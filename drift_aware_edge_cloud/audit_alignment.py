"""Label-alignment audit -- verify a data stream against its label sidecar.

Standalone and re-runnable. Root-level, not under src/, so it cannot be mistaken
for a component of the system under test.

Why this exists
---------------
Matching row counts and matching min/max timestamps do NOT establish alignment.
Two files can agree on both and still be misaligned in the middle -- a dropped
second in one and a duplicated second in the other cancel out in the summary and
silently shift every label by one row. Row i of the labels must be shown to
describe row i of the data, elementwise, or the evaluation is measuring noise.

So this compares the full timestamp vectors position by position, and separately
reports the label file's own timestamp resolution -- because a label file stamped
to the minute cannot be joined to a 1 Hz stream on timestamp at all, even when
its row count is exactly right.

Pairs come from argv. No filename is written into this script.

It only reads.

Run:
    PYTHONIOENCODING=utf-8 python audit_alignment.py <data.csv> <labels.csv> [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TIME_CANDIDATES = ("timestamp", "time", "datetime", "date")


def time_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in TIME_CANDIDATES:
            return str(c)
    raise SystemExit(f"no timestamp column among {list(df.columns)[:6]}")


def resolution(ts: pd.Series) -> str:
    """Finest unit at which the stamps actually vary."""
    for unit, attr in (("second", "second"), ("minute", "minute"),
                       ("hour", "hour")):
        if getattr(ts.dt, attr).nunique() > 1:
            # varies at this unit; check whether anything finer is constant
            if unit == "second" and (ts.dt.second.nunique() > 1):
                return "second (1 s)"
        # fall through
    if ts.dt.second.nunique() > 1:
        return "second (1 s)"
    if ts.dt.minute.nunique() > 1:
        return "minute (60 s) -- SECONDS ARE CONSTANT"
    return "coarser than minute"


def audit_pair(data_path: Path, label_path: Path) -> bool:
    print("=" * 78)
    print(f"{data_path.name}  <->  {label_path.name}")
    print("=" * 78)

    d = pd.read_csv(data_path, low_memory=False)
    l = pd.read_csv(label_path, low_memory=False)
    dt, lt = time_col(d), time_col(l)
    dts = pd.to_datetime(d[dt], format="mixed", dayfirst=False)
    lts = pd.to_datetime(l[lt], format="mixed", dayfirst=False)

    checks: list[tuple[str, bool, str]] = []

    checks.append(("row counts equal", len(d) == len(l),
                   f"data {len(d):,} vs labels {len(l):,}"))

    # Raw string form of the first stamps, so a format difference is visible
    # rather than hidden behind pandas' parse.
    print(f"  data   first raw stamp: {d[dt].iloc[0]!r}   "
          f"last: {d[dt].iloc[-1]!r}")
    print(f"  labels first raw stamp: {l[lt].iloc[0]!r}   "
          f"last: {l[lt].iloc[-1]!r}")
    print(f"  data   timestamp resolution: {resolution(dts)}")
    print(f"  labels timestamp resolution: {resolution(lts)}")
    print(f"  data   distinct stamps: {dts.nunique():,} / {len(dts):,}")
    print(f"  labels distinct stamps: {lts.nunique():,} / {len(lts):,}")

    checks.append(("label stamps are unique (joinable key)",
                   int(lts.duplicated().sum()) == 0,
                   f"{int(lts.duplicated().sum()):,} duplicate label stamps"))

    if len(d) == len(l):
        eq = (dts.reset_index(drop=True) == lts.reset_index(drop=True))
        n_bad = int((~eq).sum())
        checks.append(("timestamps match ELEMENTWISE, row i to row i",
                       n_bad == 0,
                       "every row aligned" if n_bad == 0
                       else f"{n_bad:,} of {len(d):,} rows disagree; first at "
                            f"index {int((~eq).idxmax())}: data "
                            f"{dts.iloc[int((~eq).idxmax())]} vs labels "
                            f"{lts.iloc[int((~eq).idxmax())]}"))
        # Set-equality is a weaker but independent statement: same instants,
        # possibly reordered. Reported so a positional failure can be diagnosed.
        checks.append(("timestamp SETS identical (order-insensitive)",
                       set(dts) == set(lts),
                       f"data-only {len(set(dts) - set(lts)):,}, "
                       f"labels-only {len(set(lts) - set(dts)):,}"))

    lab = [c for c in l.columns if str(c) != lt]
    for c in lab:
        vc = l[c].value_counts(dropna=False).sort_index()
        total = int(len(l))
        dist = {str(k): f"{int(v):,} ({100.0 * v / total:.4f}%)"
                for k, v in vc.items()}
        print(f"  label column '{c}': dtype={l[c].dtype}, "
              f"{l[c].nunique(dropna=True)} distinct, "
              f"{int(l[c].isna().sum())} missing")
        print(f"      class distribution {dist}")
        checks.append((f"'{c}' has no missing labels",
                       int(l[c].isna().sum()) == 0,
                       f"{int(l[c].isna().sum())} NaN"))
        checks.append((f"'{c}' has >= 2 classes (usable for classification)",
                       int(l[c].nunique(dropna=True)) >= 2,
                       f"classes {sorted(l[c].dropna().unique().tolist())}"))
        # Contiguity of the positive class: attacks in ICS data come in runs.
        # A single scattered positive is a red flag worth seeing.
        pos = (l[c] == l[c].max()).to_numpy()
        runs = int((pos[1:] & ~pos[:-1]).sum()) + int(pos[0])
        print(f"      positive-class runs (contiguous attack episodes): {runs}")

    print("-" * 78)
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"         {detail}")
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print(f"  {n_pass}/{len(checks)} alignment checks passed")
    print()
    return n_pass == len(checks)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) % 2:
        print(__doc__)
        return 2
    ok = True
    for i in range(0, len(argv), 2):
        ok &= audit_pair(Path(argv[i]), Path(argv[i + 1]))
    print("=" * 78)
    print(f"OVERALL: {'all pairs aligned' if ok else 'AT LEAST ONE PAIR FAILED'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
