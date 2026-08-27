"""Generate the current project state by reading the repository.

Why this exists
---------------
This project is developed across sessions and, deliberately, across different
models and APIs. A hand-written status document is the obvious way to hand over
context and the wrong one: it is correct on the day it is written and silently
wrong afterwards, and a confidently stale handover is worse than none. The rest
of this repository refuses to assert numbers it has not measured; a status file
that asserts "Phase 1 complete, 230 tests" from memory would break that rule at
the one place a newcomer trusts most.

So this script MEASURES the state instead:

  * module implementation status  <- the `Status :` header in every source file
  * blocking decisions            <- null / "unresolved" keys in the resolved config
  * open findings                 <- PROVENANCE.json entries awaiting a user decision
  * leakage guards                <- their live configured values
  * raw-data integrity            <- size (always) and SHA-256 (with --hash)
  * test and harness inventory    <- files on disk, functions parsed with ast

and renders it as markdown. Regenerating it is one command, so it cannot drift.

What it does NOT do
-------------------
It runs no test, trains nothing, and reports no accuracy, latency or
Edge/Cloud/Hybrid decision. It cannot tell you that the suite passes -- only that
the suite exists and how many test functions are declared in it. The
authoritative pass counts come from actually running the five commands listed in
`AGENT_CONTEXT.md`, and this script prints them as commands to run, never as
results.

Staleness
---------
Every generated report embeds a fingerprint over the content of the code, config
and provenance files it read. `--check` recomputes it and exits non-zero when the
committed `PROJECT_STATE.md` no longer describes the tree, which makes the
staleness detectable instead of merely regrettable.

Usage
-----
    python state_report.py                 # print to stdout
    python state_report.py --write         # also write PROJECT_STATE.md
    python state_report.py --check         # exit 1 if PROJECT_STATE.md is stale
    python state_report.py --hash          # verify raw SHA-256 too (slow, ~360 MB)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import config as cfgmod          # noqa: E402

OUT_NAME = "PROJECT_STATE.md"
PROVENANCE = "data/raw/PROVENANCE.json"

# The surface whose content defines "the state of the project". The raw data is
# excluded on purpose: it is hundreds of megabytes and already has its own
# recorded checksums, which this script verifies separately under --hash.
FINGERPRINT_GLOBS = (
    "config/*.yaml",
    "src/**/*.py",
    "adaptation/**/*.py",
    "tests/*.py",
    "experiments/*.py",
    "verify_*.py",
    "state_report.py",
    "main.py",
    PROVENANCE,
)

# Config keys whose value is itself a research decision. Read, never defaulted.
GUARDS = (
    "dataset.concatenate_files",
    "dataset.allow_acausal_baseline",
    "dataset.baseline_source",
    "streaming.shuffle",
    "preprocessing.normalization.adaptation",
    "preprocessing.normalization.forbid_global_fit",
    "drift.injection_target",
    "drift.modify_labels",
)
UNRESOLVED_MARKERS = ("dataset.task", "dataset.target_column",
                      "dataset.label_column", "dataset.label_file",
                      "dataset.positive_class")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def _surface() -> list[Path]:
    seen: dict[str, Path] = {}
    for pattern in FINGERPRINT_GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file():
                seen[_rel(p)] = p
    return [seen[k] for k in sorted(seen)]


def _fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(_rel(p).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _header_fields(text: str) -> dict[str, str]:
    """Parse the `Key : value` block from a module docstring header."""
    out: dict[str, str] = {}
    for line in text.splitlines()[:14]:
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        if key in ("Module", "Phase", "Status"):
            out[key] = val.strip()
    return out


def _test_functions(path: Path) -> tuple[int, int]:
    """(declared test functions, of which marked slow). Parsed, not executed."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return (0, 0)
    total = slow = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        total += 1
        if any("slow" in ast.dump(d) for d in node.decorator_list):
            slow += 1
    return (total, slow)


def _dir_inventory(path: Path) -> list[str]:
    if not path.is_dir():
        return []
    return sorted(p.name for p in path.iterdir() if p.name != ".gitkeep")


def _first_sentence(text: str, limit: int = 110) -> str:
    """First sentence, without amputating a dotted config key.

    Splitting on "." alone turns "dataset.baseline_source: train1_only" into
    "dataset", which is worse than no summary at all. A sentence end here is a
    period followed by whitespace.
    """
    text = " ".join(str(text).split())
    for i in range(len(text) - 1):
        if text[i] == "." and text[i + 1] == " ":
            text = text[:i]
            break
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


# -----------------------------------------------------------------------------
# sections
# -----------------------------------------------------------------------------


def sec_intro(fp: str, n_files: int) -> list[str]:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return [
        "# PROJECT_STATE — generated, do not edit by hand",
        "",
        f"Generated `{stamp}` by `state_report.py` from {n_files} code, config",
        f"and provenance files. Surface fingerprint **`{fp}`**.",
        "",
        "Regenerate after any change:",
        "",
        "```bash",
        "cd drift_aware_edge_cloud && ../.venv/Scripts/python.exe state_report.py --write",
        "```",
        "",
        "Everything below was read off the tree just now. Nothing here is a",
        "remembered value, and nothing here is an experimental result — this",
        "script runs no test and no model. For pass counts, run the commands in",
        "[Verification](#verification) yourself.",
        "",
    ]


def sec_environment() -> list[str]:
    lines = ["## Environment", "",
             f"Interpreter running this report: **{sys.version.split()[0]}** "
             f"(`{sys.executable}`)", ""]
    try:
        from importlib import metadata
    except ImportError:                                   # pragma: no cover
        return lines
    want = ("numpy", "pandas", "scipy", "scikit-learn", "PyYAML", "xgboost",
            "river", "simpy", "matplotlib", "pytest")
    rows = []
    for name in want:
        try:
            rows.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            rows.append((name, "**NOT INSTALLED**"))
    lines += ["| Package | Installed |", "|---|---|"]
    lines += [f"| {n} | {v} |" for n, v in rows]
    lines.append("")
    return lines


def sec_raw(prov: dict, do_hash: bool) -> list[str]:
    lines = ["## Raw data", "",
             "Roles come from `config/default.yaml -> dataset.files`; the file",
             "names are never written in Python.", "",
             "| File | Recorded role | Size on disk | Matches record | Modified flag |",
             "|---|---|---|---|---|"]
    problems = []
    for entry in prov.get("supplied_files", []):
        path = ROOT / "data/raw" / entry["file"]
        if not path.exists():
            lines.append(f"| `{entry['file']}` | {entry.get('role','?')} | "
                         f"**MISSING** | no | — |")
            problems.append(f"{entry['file']} is missing")
            continue
        size = path.stat().st_size
        ok = size == entry.get("size_bytes")
        if not ok:
            problems.append(f"{entry['file']} size {size} != recorded "
                            f"{entry.get('size_bytes')}")
        lines.append(f"| `{entry['file']}` | {entry.get('role','?')} | "
                     f"{size:,} | {'yes' if ok else '**NO**'} | "
                     f"{entry.get('modified')} |")
    if do_hash:
        lines += ["", "SHA-256 verified this run:", ""]
        for entry in prov.get("supplied_files", []):
            path = ROOT / "data/raw" / entry["file"]
            if not path.exists():
                continue
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            ok = h.hexdigest() == entry["sha256"]
            if not ok:
                problems.append(f"{entry['file']} SHA-256 MISMATCH")
            lines.append(f"- `{entry['file']}` — "
                         f"{'byte-identical' if ok else '**MODIFIED**'}")
    else:
        lines += ["", "_SHA-256 not checked in this run; pass `--hash` "
                      "(or run the `slow` pytest marker) to verify._"]
    lines.append("")
    if problems:
        lines += ["> **RAW DATA PROBLEM — every measurement above it is void:**", ""]
        lines += [f"> - {p}" for p in problems] + [""]
    return lines


def sec_modules() -> list[str]:
    by_phase: dict[str, list[tuple[str, str]]] = {}
    for p in sorted(list((ROOT / "src").rglob("*.py"))
                    + list((ROOT / "adaptation").rglob("*.py"))):
        if p.name == "__init__.py":
            continue
        fields = _header_fields(p.read_text(encoding="utf-8"))
        phase = fields.get("Phase", "unlabelled")
        status = fields.get("Status", "**no Status header**")
        by_phase.setdefault(phase, []).append((_rel(p), status))

    done = sum(1 for v in by_phase.values() for _, s in v if s == "IMPLEMENTED")
    total = sum(len(v) for v in by_phase.values())
    lines = ["## Implementation status", "",
             f"Read from the `Status :` header of {total} source modules: "
             f"**{done} IMPLEMENTED**, **{total - done} not**.", ""]

    def phase_order(label: str) -> tuple[int, str]:
        """Sort by phase NUMBER, not lexically -- 'Phase 1 / Step 3' must not
        land after 'Phase 10' just because it is a longer string."""
        digits = "".join(ch for ch in label.split("/")[0] if ch.isdigit())
        return (int(digits) if digits else 999, label)

    for phase in sorted(by_phase, key=phase_order):
        mods = by_phase[phase]
        n_impl = sum(1 for _, s in mods if s == "IMPLEMENTED")
        mark = "complete" if n_impl == len(mods) else \
               ("not started" if n_impl == 0 else f"{n_impl}/{len(mods)}")
        lines.append(f"- **{phase}** — {mark}")
        for rel, status in mods:
            flag = "x" if status == "IMPLEMENTED" else " "
            lines.append(f"  - [{flag}] `{rel}` — {status}")
    lines.append("")
    return lines


def sec_blocking(cfg: dict, prov: dict) -> list[str]:
    lines = ["## Blocking decisions — reserved to the user", ""]

    unresolved = []
    for key in UNRESOLVED_MARKERS:
        val = cfgmod.get(cfg, key, None)
        if val is None or val == "unresolved":
            unresolved.append((key, "null" if val is None else str(val)))
    if unresolved:
        lines += ["Configuration keys deliberately left unset rather than "
                  "guessed:", "", "| Key | Value |", "|---|---|"]
        lines += [f"| `{k}` | `{v}` |" for k, v in unresolved]
        lines.append("")

    dec = prov.get("ACTIVE_DATASET_DECISION", {})
    q = dec.get("blocking_open_question")
    if q:
        lines += [f"### {q.get('id','open question')} — `{q.get('config_key')}`",
                  "",
                  q.get("why_blocking", ""), "",
                  "Options on record, with their consequences:", ""]
        for opt in q.get("options", []):
            lines.append(f"- **`{opt.get('value')}`** — target: "
                         f"{opt.get('target','?')}. "
                         f"{opt.get('consequence','')}")
            for extra in ("risk", "requires", "status", "leakage_note"):
                if opt.get(extra):
                    lines.append(f"  - _{extra}_: {opt[extra]}")
        lines += ["", f"_{q.get('integrity_rule_applied','')}_", ""]

    # A finding is "open" if it names something it is waiting for, OR if its
    # severity is BLOCKER. `finding_no_label_column` has no `awaiting` field --
    # what it waits on is the separate blocking question above -- but filing it
    # under "settled" would read as though the label problem had gone away.
    def is_open(v: dict) -> bool:
        return bool(v.get("awaiting")) or v.get("severity") == "BLOCKER"

    findings = [(k, v) for k, v in prov.items()
                if k.startswith("finding_") and isinstance(v, dict)]
    open_findings = [(k, v) for k, v in findings if is_open(v)]
    if open_findings:
        lines += ["### Open findings", "",
                  "Each was measured and deliberately **not** repaired, because "
                  "repairing it would be a scientific choice.", ""]
        for key, v in open_findings:
            awaiting = v.get("awaiting") or "the blocking question above"
            lines += [f"- **`{key}`** ({v.get('severity')}) — "
                      f"{v.get('claim','')}",
                      f"  - action taken: {v.get('action_taken','')}",
                      f"  - awaiting: {awaiting}"]
        lines.append("")

    other = [(k, v) for k, v in findings if not is_open(v)]
    if other:
        lines += ["### Findings already settled or accepted as measured fact",
                  "", "| Finding | Severity | Resolution |", "|---|---|---|"]
        lines += [f"| `{k}` | {v.get('severity')} | "
                  f"{_first_sentence(v.get('action_taken', ''))} |"
                  for k, v in other]
        lines.append("")
    return lines


def sec_guards(cfg: dict, names: tuple[str, ...]) -> list[str]:
    lines = ["## Leakage and integrity guards — live configured values", "",
             "| Key | Value |", "|---|---|"]
    for key in GUARDS:
        lines.append(f"| `{key}` | `{cfgmod.get(cfg, key, '<absent>')}` |")
    lines += ["",
              "Ground-truth consumers, from `ground_truth`:", ""]
    for which in ("allowed_consumers", "forbidden_consumers"):
        val = cfgmod.get(cfg, f"ground_truth.{which}", None)
        lines.append(f"- **{which}**: {val}")
    lines += ["", f"Configurations present: "
                  f"{', '.join('`' + n + '`' for n in names)}", ""]
    for name in names:
        try:
            fp = cfgmod.fingerprint(cfgmod.load(name))
        except Exception as exc:                       # pragma: no cover
            lines.append(f"- `{name}` — **FAILED TO RESOLVE: {exc}**")
            continue
        lines.append(f"- `{name}` — resolves and validates, fingerprint `{fp}`")
    lines.append("")
    return lines


def sec_verification() -> list[str]:
    lines = ["## Verification", "",
             "This script did **not** run any of these. Run them; quote the "
             "counts they print.", "",
             "```bash",
             "cd drift_aware_edge_cloud",
             "export PYTHONIOENCODING=utf-8"]
    harnesses = sorted((ROOT).glob("verify_step*.py"))
    for h in harnesses:
        lines.append(f"../.venv/Scripts/python.exe {h.name}")
    lines += ["../.venv/Scripts/python.exe -m pytest -q", "```", ""]
    if (ROOT / "verify_handover.py").exists():
        lines += ["Handover artifacts have their own harness, run at session "
                  "end rather than per edit (it checks this file for "
                  "staleness, so it fails until it is regenerated):", "",
                  "```bash",
                  "../.venv/Scripts/python.exe verify_handover.py",
                  "```", ""]
    lines += ["Test suite on disk, parsed with `ast`:", "",
              "| Module | Test functions | Marked slow |", "|---|---|---|"]
    grand = grand_slow = 0
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        n, s = _test_functions(p)
        grand += n
        grand_slow += s
        lines.append(f"| `{_rel(p)}` | {n} | {s} |")
    lines += [f"| **total declared** | **{grand}** | **{grand_slow}** |", "",
              "Declared functions are not the collected count: `pytest` expands "
              "parametrised cases, so the number it reports is higher. The "
              "collected/passed count is only knowable by running it.", ""]
    return lines


def sec_outputs() -> list[str]:
    lines = ["## Generated output present on disk", "",
             "Regenerable, not authoritative. `data/synthetic/` must contain "
             "only ground-truth sidecars — it is an output directory that "
             "nothing reads back as input.", "",
             "| Directory | Entries |", "|---|---|"]
    for d in ("results", "plots", "data/synthetic", "data/processed"):
        inv = _dir_inventory(ROOT / d)
        shown = ", ".join(f"`{n}`" for n in inv[:6])
        if len(inv) > 6:
            shown += f", … (+{len(inv) - 6} more)"
        lines.append(f"| `{d}/` | {len(inv)}{' — ' + shown if inv else ''} |")
    lines.append("")
    return lines


def sec_pointers() -> list[str]:
    return [
        "## Where the rest of the context lives", "",
        "| File | Holds |", "|---|---|",
        "| `AGENT_CONTEXT.md` | The contract: what you must not do, how to "
        "work, how to verify. **Read first.** |",
        "| `DECISIONS.md` | Append-only log of decisions, errors and fixes, "
        "with the reasoning. Explains *why* the tree looks like this. |",
        "| `README.md` | The research design: question, hypothesis, causal "
        "chain, dataset diagnostics, scope. |",
        "| `data/raw/PROVENANCE.json` | Machine-readable dataset provenance, "
        "checksums, every measured diagnostic and finding. |",
        "| `config/*.yaml` | Every parameter and assumption. Nothing is "
        "hard-coded in Python. |",
        "| `tests/test_integrity.py` | The prohibitions as executable tests — "
        "the only form of context that cannot be forgotten. |",
        "",
    ]


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def build(do_hash: bool) -> tuple[str, str]:
    surface = _surface()
    fp = _fingerprint(surface)
    prov = json.loads((ROOT / PROVENANCE).read_text(encoding="utf-8"))
    names = cfgmod.available(ROOT / "config")
    cfg = cfgmod.load("default", config_dir=ROOT / "config")

    lines: list[str] = []
    lines += sec_intro(fp, len(surface))
    lines += sec_pointers()
    lines += sec_modules()
    lines += sec_blocking(cfg, prov)
    lines += sec_guards(cfg, names)
    lines += sec_raw(prov, do_hash)
    lines += sec_verification()
    lines += sec_outputs()
    lines += sec_environment()
    lines += ["---", "",
              f"_End of generated report. Fingerprint `{fp}`; "
              f"`state_report.py --check` fails if the tree has moved since._",
              ""]
    return "\n".join(lines), fp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help=f"write {OUT_NAME} as well as printing")
    ap.add_argument("--check", action="store_true",
                    help=f"exit 1 if {OUT_NAME} is missing or stale")
    ap.add_argument("--hash", action="store_true",
                    help="verify raw-file SHA-256 (reads ~360 MB)")
    args = ap.parse_args(argv)

    if args.check:
        target = ROOT / OUT_NAME
        current = _fingerprint(_surface())
        if not target.exists():
            print(f"STALE: {OUT_NAME} does not exist "
                  f"(current fingerprint {current})")
            return 1
        text = target.read_text(encoding="utf-8")
        if f"`{current}`" not in text:
            print(f"STALE: {OUT_NAME} does not carry the current fingerprint "
                  f"{current}. Regenerate with --write.")
            return 1
        print(f"CURRENT: {OUT_NAME} matches fingerprint {current}")
        return 0

    report, fp = build(args.hash)
    print(report)
    if args.write:
        (ROOT / OUT_NAME).write_text(report, encoding="utf-8")
        print(f"\n[written] {OUT_NAME} — fingerprint {fp}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
