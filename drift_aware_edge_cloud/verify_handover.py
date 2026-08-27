"""Handover verification harness -- the cross-model context artifacts.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Why this harness exists
-----------------------
`AGENT_CONTEXT.md` asserts that executable tests are the only form of project
context that cannot be forgotten. Leaving the handover mechanism itself unchecked
would contradict that in the one place a newcomer trusts most. The specific
failure this guards against is not a missing file -- it is a handover document
that is confidently wrong: a `PROJECT_STATE.md` describing a tree that has moved,
or two documents quoting different verification counts.

It is deliberately NOT part of the pytest suite. Regenerating `PROJECT_STATE.md`
is a handover ritual, not a per-edit obligation, and wiring staleness into
`pytest` would leave the suite red during ordinary development -- destroying the
signal that "the suite is green" currently carries. This runs at session end,
alongside the four step harnesses.

What it does NOT check
----------------------
It does not run the step harnesses or the test suite, and it cannot tell you the
project is correct. It checks that the handover artifacts are present, mutually
consistent, and current.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe drift_aware_edge_cloud/verify_handover.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CONTEXT = ROOT / "AGENT_CONTEXT.md"
DECISIONS = ROOT / "DECISIONS.md"
STATE = ROOT / "PROJECT_STATE.md"
GENERATOR = ROOT / "state_report.py"
PROVENANCE = ROOT / "data/raw/PROVENANCE.json"
README = ROOT / "README.md"

# Matched on the phrase, not the full sentence, so the marker can say "above" or
# "below" without this harness caring which convention the log settles on.
APPEND_SENTINEL = "Append new entries"

results: list[tuple[str, bool, str]] = []
_details: list[str] = []


def note(msg: str) -> None:
    _details.append(msg)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


ctx = read(CONTEXT)
dec = read(DECISIONS)
state = read(STATE)
readme = read(README)


# =============================================================================
# 1. the four artifacts exist and are not placeholders
# =============================================================================
present = {p.name: (p.exists(), len(read(p))) for p in
           (CONTEXT, DECISIONS, STATE, GENERATOR)}
for name, (exists, size) in present.items():
    note(f"  {name}: {'present' if exists else 'MISSING'}, {size:,} bytes")
check(
    "1. all four handover artifacts exist with real content",
    all(exists and size > 1000 for exists, size in present.values()),
    "AGENT_CONTEXT.md (contract), DECISIONS.md (reasoning), "
    "PROJECT_STATE.md (generated status), state_report.py (the generator)",
)


# =============================================================================
# 2. PROJECT_STATE.md still describes this tree
# =============================================================================
proc = subprocess.run([sys.executable, str(GENERATOR), "--check"],
                      cwd=str(ROOT), capture_output=True, text=True)
first = (proc.stdout or proc.stderr).strip().splitlines()
check(
    "2. PROJECT_STATE.md is current, not stale",
    proc.returncode == 0,
    (first[0] if first else "no output")
    + ("" if proc.returncode == 0
       else "  -- regenerate: python state_report.py --write"),
)


# =============================================================================
# 3. PROJECT_STATE.md declares itself generated
# =============================================================================
check(
    "3. PROJECT_STATE.md warns that it is generated, and names its generator",
    "generated, do not edit by hand" in state and "state_report.py" in state,
    "a hand-edited generated file is the staleness failure wearing a disguise",
)


# =============================================================================
# 4. the contract names every standing prohibition
# =============================================================================
# One probe per prohibition recorded in the project's integrity rules. Matching
# on topic keywords, not on wording, so the contract can be rephrased freely.
TOPICS = {
    "fabricated data": r"fabricat\w+ data|replacement dataset",
    "fabricated labels": r"fabricat\w+ label|no label column|label column",
    "fabricated results": r"fabricat\w+ result|hard-code a metric",
    "leakage / future information": r"future information|leakage",
    "ground-truth isolation": r"[Gg]round truth is evaluation-only|"
                              r"ground.truth.*evaluation",
    "computed orchestration decision": r"forced orchestration|argmin|WDS",
    "silent assumptions": r"ASSUMPTION \[A",
    "findings not silently repaired": r"silent repairs|awaiting",
    "config-not-code file roles": r"live in configuration|not in Python",
    "architecture unchanged": r"[Dd]o not redesign",
    "no tuning on test results": r"tune the proposed method",
    "discrete channels": r"discrete actuator",
    "venv untouched": r"[Dd]o not recreate or modify",
}
missing = [k for k, pat in TOPICS.items() if not re.search(pat, ctx)]
check(
    "4. AGENT_CONTEXT.md names all 13 standing prohibitions",
    not missing,
    f"all {len(TOPICS)} topics present" if not missing
    else f"MISSING from the contract: {missing}",
)


# =============================================================================
# 5. the contract carries a runnable bootstrap
# =============================================================================
needed = ["verify_step2.py", "verify_step3.py", "verify_step4.py",
          "verify_step5.py", "pytest", "PYTHONIOENCODING",
          "../.venv/Scripts/python.exe"]
absent = [n for n in needed if n not in ctx]
check(
    "5. AGENT_CONTEXT.md gives the exact commands, interpreter and encoding",
    not absent,
    "a fresh model can bootstrap without guessing" if not absent
    else f"MISSING: {absent}",
)


# =============================================================================
# 6. no two documents quote different verification counts
# =============================================================================
# The point of failure this catches: README updated after a step, contract not,
# and the next model reproduces the wrong target and thinks the tree is broken.
COUNT_RE = re.compile(r"verify_step(\d)\D{0,40}?(\d+)\s*/\s*(\d+)")


def counts(text: str) -> dict[str, str]:
    return {f"step{m.group(1)}": f"{m.group(2)}/{m.group(3)}"
            for m in COUNT_RE.finditer(text)}


c_ctx, c_readme = counts(ctx), counts(readme)
shared = sorted(set(c_ctx) & set(c_readme))
disagree = [(k, c_ctx[k], c_readme[k]) for k in shared if c_ctx[k] != c_readme[k]]
pytest_ctx = re.findall(r"(\d+)\s+passed", ctx)
pytest_readme = re.findall(r"(\d+)\s+passed", readme)
pytest_bad = bool(pytest_ctx) and bool(pytest_readme) and \
    set(pytest_ctx) != set(pytest_readme)
note(f"  contract counts: {c_ctx}, pytest {pytest_ctx}")
note(f"  README counts:   {c_readme}, pytest {pytest_readme}")
check(
    "6. AGENT_CONTEXT.md and README.md quote identical verification counts",
    not disagree and not pytest_bad and len(shared) == 4,
    f"{len(shared)}/4 harnesses cross-checked, all agreeing"
    if not disagree and not pytest_bad
    else f"DISAGREEMENT {disagree}, pytest {pytest_ctx} vs {pytest_readme}",
)


# =============================================================================
# 7. the contract states counts as targets to reproduce, not as evidence
# =============================================================================
check(
    "7. AGENT_CONTEXT.md forbids quoting its own counts without re-running",
    bool(re.search(r"reproduce.{0,200}not as evidence|"
                   r"[Nn]ever quote them without having run", ctx, re.S)),
    "recorded numbers in a handover file are a target, never a result",
)


# =============================================================================
# 8. DECISIONS.md is shaped as an append-only log with unique ids
# =============================================================================
ids = re.findall(r"^##\s+([DE]-\d{3})\s", dec, re.M)
dupes = sorted({i for i in ids if ids.count(i) > 1})
check(
    "8. DECISIONS.md carries the append-only sentinel and unique entry ids",
    APPEND_SENTINEL in dec and bool(ids) and not dupes,
    f"{len(ids)} entries ({len(set(ids))} unique), sentinel present"
    if not dupes and APPEND_SENTINEL in dec
    else f"duplicate ids {dupes}; sentinel "
         f"{'present' if APPEND_SENTINEL in dec else 'MISSING'}",
)


# =============================================================================
# 9. every entry gives its reasoning, not just its outcome
# =============================================================================
# An outcome without a rationale is exactly what a fresh model cannot recover by
# reading code -- so a bare entry defeats the file's only purpose.
bodies = re.split(r"^##\s+[DE]-\d{3}\s", dec, flags=re.M)[1:]
thin = [ids[i] for i, b in enumerate(bodies)
        if not re.search(r"\*\*Why|\*\*Root cause|\*\*Lesson|\*\*Context|"
                         r"\*\*Measured|Reported because|deliberately",
                         b, re.I)]
check(
    "9. every DECISIONS.md entry records reasoning, not only an outcome",
    not thin,
    f"all {len(bodies)} entries carry a Why / Root cause / Context / Lesson"
    if not thin else f"entries with no rationale: {thin}",
)


# =============================================================================
# 10. no open blocker is missing from the log
# =============================================================================
prov = json.loads(read(PROVENANCE)) if PROVENANCE.exists() else {}
open_keys = [k for k, v in prov.items()
             if k.startswith("finding_") and isinstance(v, dict)
             and (v.get("awaiting") or v.get("severity") == "BLOCKER")]
# The log may refer to a finding by key or by the subject it names, so accept
# either -- the test is whether a reader of the log learns the blocker exists,
# not whether a particular string was pasted in.
SUBJECTS = {
    "finding_no_label_column": r"no label|label column",
    "finding_near_collinear_pairs": r"collinear|0\.999|r > 0",
    "finding_continuous_command_channels": r"command/demand|actuator_policy",
    "finding_degenerate_outlier_bounds": r"IQR|zero-width",
}
unlogged = [k for k in open_keys
            if k not in dec
            and not re.search(SUBJECTS.get(k, k), dec, re.I)]
note(f"  open findings in provenance: {open_keys}")
check(
    "10. every open finding / blocker appears in DECISIONS.md",
    not unlogged and bool(open_keys),
    f"{len(open_keys)} open items, all logged" if not unlogged
    else f"NOT logged: {unlogged}",
)


# =============================================================================
# 11. the blocked task is named as reserved to the user, in both places
# =============================================================================
blocking = (prov.get("ACTIVE_DATASET_DECISION", {})
                .get("blocking_open_question", {}))
opts = {o.get("value") for o in blocking.get("options", [])}
in_ctx = all(o in ctx for o in opts) if opts else False
reserved = bool(re.search(r"reserved to the user|scientific decision", ctx))
check(
    "11. AGENT_CONTEXT.md names all target options and calls the choice the "
    "user's",
    in_ctx and reserved and bool(opts),
    f"options {sorted(opts)} all named; the choice is marked reserved"
    if in_ctx and reserved else
    f"options in provenance {sorted(opts)}; named in contract: {in_ctx}; "
    f"marked reserved: {reserved}",
)


# =============================================================================
# 12. the handover artifacts are discoverable from the generated state
# =============================================================================
check(
    "12. PROJECT_STATE.md points back at the contract and the log",
    "AGENT_CONTEXT.md" in state and "DECISIONS.md" in state
    and "AGENT_CONTEXT.md" in readme and "DECISIONS.md" in readme,
    "a newcomer landing on README or PROJECT_STATE finds the contract",
)


# =============================================================================
# Report
# =============================================================================
print("=" * 78)
print("HANDOVER VERIFICATION -- cross-model context artifacts")
print("=" * 78)
for ln in _details:
    print(ln)
print("-" * 78)
n_pass = sum(1 for _, ok, _ in results if ok)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for ln in str(detail).splitlines():
            print(f"       {ln}")
print("-" * 78)
print(f"{n_pass}/{len(results)} checks passed")
print("=" * 78)
raise SystemExit(0 if n_pass == len(results) else 1)
