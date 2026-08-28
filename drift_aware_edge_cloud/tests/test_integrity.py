"""Cross-cutting research-integrity checks.

These tests do not exercise a component. They assert the properties that make the
eventual results trustworthy, and each one corresponds to a rule recorded in
`data/raw/PROVENANCE.json -> integrity_rules`:

  raw_immutability        -> the SHA-256 tests
  no_fabricated_data      -> no synthetic input, data/synthetic is output-only
  causality               -> no fit path reaches the inference stream
  ground_truth_isolation  -> the quarantine tests
  no_results_asserted     -> no hard-coded performance number anywhere in src/

A test suite that only checks that code works, while the code is quietly
leaking, is worse than none.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from src.data import generator, loader, preprocessing as pp

PROVENANCE = "data/raw/PROVENANCE.json"
PHASE_1_MODULES = {
    "src/data/loader.py", "src/data/generator.py", "src/data/stream.py",
    "src/data/preprocessing.py", "src/utils/config.py", "src/utils/seed.py",
    "src/utils/logger.py",
}
PHASE_2_MODULES = {
    "src/models/__init__.py", "src/models/base.py", "src/models/cloud_model.py",
    "src/models/edge_model.py", "src/models/trainer.py",
}
IMPLEMENTED = PHASE_1_MODULES | PHASE_2_MODULES
GT_FIELDS = ("scenario", "drift_start_index", "drift_end_index",
             "affected_features", "drift_magnitude", "random_seed")


@pytest.fixture(scope="session")
def provenance(project_root) -> dict:
    return json.loads((project_root / PROVENANCE).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def sources(project_root) -> dict[str, str]:
    return {
        str(p.relative_to(project_root)).replace("\\", "/"):
            p.read_text(encoding="utf-8")
        for p in sorted((project_root / "src").rglob("*.py"))
    }


# -----------------------------------------------------------------------------
# raw_immutability
# -----------------------------------------------------------------------------


def test_provenance_records_every_supplied_file(provenance, project_root):
    recorded = {e["file"] for e in provenance["supplied_files"]}
    if "superseded_swat_record" in provenance:
        swat_f = provenance["superseded_swat_record"].get("file", {}).get("file")
        if swat_f:
            recorded.add(swat_f)
    on_disk = {p.name for p in (project_root / "data/raw").iterdir() if p.is_file() and p.suffix in {".txt", ".csv"}}
    assert on_disk == recorded, f"unrecorded raw file(s): {on_disk ^ recorded}"


@pytest.mark.slow
def test_raw_files_are_byte_identical_to_the_recorded_checksums(provenance,
                                                                project_root):
    """If a raw file was ever modified in place, every measurement above it is
    void. Hashing 360 MB is cheap next to that."""
    for entry in provenance["supplied_files"]:
        path = project_root / "data/raw" / entry["file"]
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        assert h.hexdigest() == entry["sha256"], f"{entry['file']} was modified"
        assert path.stat().st_size == entry["size_bytes"]
        assert entry["modified"] is False


def test_the_repository_never_opens_a_raw_file_for_writing(sources):
    for rel, text in sources.items():
        for m in re.finditer(r"""open\([^)]*['"](w|a|wb|ab)['"]""", text):
            assert "data/raw" not in text[max(0, m.start() - 200):m.start()], rel
        assert "data/raw" not in text or "to_csv" not in text, rel


def test_synthetic_directory_is_an_output_not_an_input(sources, cfg):
    """No config or module may READ from data/synthetic; drift derivatives are
    written there, never loaded back as if they were a recording."""
    for key, spec in cfg["dataset"]["files"].items():
        path = str(spec["path"]).replace("\\", "/")
        assert path.startswith("data/raw/"), f"{key} reads from {path}"
        assert "synthetic" not in path and "processed" not in path, key
    for rel, text in sources.items():
        for m in re.finditer(r"read_csv\(|read_table\(|load_file\(", text):
            window = text[max(0, m.start() - 300):m.start()]
            assert "synthetic" not in window, f"{rel} reads from data/synthetic"


# -----------------------------------------------------------------------------
# no_fabricated_data / no_fabricated_labels
# -----------------------------------------------------------------------------


def test_no_module_fabricates_a_label(sources, cfg, provenance):
    """The label must be READ, never manufactured."""
    ds = cfg["dataset"]
    task = ds["task"]
    if task == "supervised_classification":
        assert ds["target_column"] == "Target"
        assert ds["training_labels_available"] is True
    else:
        assert ds["task"] == "labels_from_hai_labels"
        assert ds["target_column"] is None
        label_file = ds["label_file"]
        assert label_file, "a resolved label task must name the file it reads"
        recorded = {e["file"]: e for e in provenance["supplied_files"]}
        name = label_file.rsplit("/", 1)[-1]
        assert name in recorded, f"{name} has no provenance record"
        entry = recorded[name]
        assert entry["byte_identical_to_source"] is True
        assert entry["modified"] is False
        assert entry["alignment"] == "elementwise_verified", (
            "a label file whose alignment is merely plausible would score the "
            "detector against rows it was never shown to describe"
        )
        stream_key = next(k for k, v in ds["files"].items()
                          if v["role"] == "inference_stream")
        assert entry["aligns_with"].rsplit("/", 1)[-1] == \
            ds["files"][stream_key]["path"].rsplit("/", 1)[-1]
        assert ds["label_usage"] == "evaluation_only"
        assert {"models", "drift_detectors"} <= set(ds["label_forbidden_consumers"])
        assert ds["training_labels_available"] is False

    banned = re.compile(r"""\b(label|target|y)\s*=\s*(np\.)?(random|zeros|ones)""")
    for rel, text in sources.items():
        assert not banned.search(text), rel


def test_resolving_a_target_never_guesses(cfg, profile):
    """Resolution is either grounded in a declared target/label or it raises."""
    target = loader.resolve_target(cfg, profile)
    expected = cfg["dataset"].get("target_column") or cfg["dataset"].get("label_column")
    assert target == expected

    unresolved = copy.deepcopy(cfg)
    unresolved["dataset"]["task"] = "unresolved"
    with pytest.raises(loader.UnresolvedTaskError):
        loader.resolve_target(unresolved, profile)

    if cfg["dataset"]["task"] == "labels_from_hai_labels":
        half = copy.deepcopy(cfg)
        half["dataset"]["label_file"] = None
        with pytest.raises(loader.ConfigError, match="label_file"):
            loader.resolve_target(half, profile)
    elif cfg["dataset"]["task"] == "supervised_classification":
        half = copy.deepcopy(cfg)
        half["dataset"]["target_column"] = None
        with pytest.raises(loader.ConfigError, match="target_column"):
            loader.resolve_target(half, profile)


def test_drift_is_injected_only_into_the_inference_stream(cfg_sudden, cfg_gradual,
                                                          cfg_stress):
    for c in (cfg_sudden, cfg_gradual, cfg_stress):
        assert c["drift"]["injection_target"] == "inference_stream_only"
        assert c["drift"]["modify_labels"] is False


# -----------------------------------------------------------------------------
# causality
# -----------------------------------------------------------------------------


def _code_lines(text: str) -> list[tuple[int, str]]:
    """Executable lines only, with string literals and comments blanked out.

    A docstring that NAMES `scaler.fit(all_data)` as the thing not to do is
    documentation; a call to it is an offence. A scanner that cannot tell them
    apart punishes the codebase for explaining itself.
    """
    import io as _io
    import tokenize

    lines = text.splitlines()
    blanked = list(lines)
    literal = {tokenize.STRING, tokenize.COMMENT}
    # Python 3.12+ tokenizes f-strings as FSTRING_START/MIDDLE/END rather than
    # STRING, so a prose f-string would otherwise slip through as code.
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END"):
        tt = getattr(tokenize, name, None)
        if tt is not None:
            literal.add(tt)
    try:
        toks = list(tokenize.generate_tokens(_io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError):  # pragma: no cover
        return list(enumerate(lines, 1))
    for tok in toks:
        if tok.type not in literal:
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        for r in range(r1, r2 + 1):
            i = r - 1
            if i >= len(blanked):
                continue
            line = blanked[i]
            lo = c1 if r == r1 else 0
            hi = c2 if r == r2 else len(line)
            blanked[i] = line[:lo] + " " * max(0, hi - lo) + line[hi:]
    return list(enumerate(blanked, 1))


def test_the_source_scanner_itself_can_still_see_an_offence():
    """A scanner that blanks too much passes everything. Plant a real offence
    alongside prose mentions of the same constructs and require it to separate
    them -- otherwise the leakage sweep above proves nothing."""
    planted = (
        'def f(x):\n'
        '    """Never call scaler.fit(all_data) -- prose, not code."""\n'
        '    # shuffle=True in a comment is also prose\n'
        '    msg = f"and shuffle=True inside an f-string is prose too"\n'
        '    return x.shift(-1)\n'
    )
    pattern = re.compile(r"shuffle\s*=\s*True|\.fit\(\s*all_data|\.shift\(\s*-\d")
    hits = [i for i, ln in _code_lines(planted) if pattern.search(ln)]
    assert hits == [5], f"scanner flagged lines {hits}, expected only line 5"


def test_no_source_line_fits_on_a_concatenation_or_on_everything(sources):
    banned = [
        re.compile(r"\.fit\(\s*(all_data|full|everything)"),
        re.compile(r"\.fit\(\s*pd\.concat"),
        re.compile(r"\.fit\(\s*np\.(concatenate|vstack)"),
        re.compile(r"train_test_split"),
        re.compile(r"shuffle\s*=\s*True"),
        re.compile(r"\.sample\(\s*frac"),
        re.compile(r"\bbfill\b|\bbackfill\b"),
        re.compile(r"\.interpolate\([^)]*limit_direction\s*="),
        re.compile(r"\.shift\(\s*-\d"),
    ]
    offences = []
    for rel, text in sources.items():
        for i, line in _code_lines(text):
            if any(p.search(line) for p in banned):
                offences.append(f"{rel}:{i}: {line.strip()}")
    assert not offences, "acausal or leaking constructs:\n" + "\n".join(offences)


def test_the_configured_baseline_excludes_the_acausal_file(cfg, provenance):
    """The filename is resolved from configuration, not written here.

    File roles are config's business; naming the train2 file literally in a test
    is the same hard-coding the rest of the suite forbids, and it would go on
    passing against a stale name after the config had moved on.
    """
    keys = loader.resolve_baseline_keys(cfg)
    assert "train2" not in keys
    name = Path(cfg["dataset"]["files"]["train2"]["path"]).name
    entry = next(e for e in provenance["supplied_files"] if e["file"] == name)
    assert "NOT usable as baseline" in entry["usage_restriction"]


def test_both_fit_entry_points_guard_the_role(cfg, infer, profile):
    """Two places fit; a single unguarded one would be enough to leak."""
    with pytest.raises(loader.CausalityError):
        loader.profile_baseline(cfg, [infer])
    with pytest.raises(loader.CausalityError):
        pp.fit(cfg, [infer], profile)


def test_leakage_guards_cannot_be_switched_off_by_configuration(cfg):
    assert cfg["preprocessing"]["normalization"]["forbid_global_fit"] is True
    assert cfg["dataset"]["allow_acausal_baseline"] is False
    assert cfg["dataset"]["concatenate_files"] is False
    assert cfg["streaming"]["shuffle"] is False
    assert cfg["preprocessing"]["normalization"]["adaptation"] == \
        "frozen_after_baseline"


# -----------------------------------------------------------------------------
# ground_truth_isolation
# -----------------------------------------------------------------------------


def test_config_names_the_forbidden_consumers(cfg):
    forbidden = cfg["ground_truth"]["forbidden_consumers"]
    assert forbidden
    lowered = " ".join(str(f).lower() for f in forbidden)
    for who in ("model", "detector", "reliability", "controller"):
        assert who in lowered, f"{who} is not listed as a forbidden consumer"


def test_ground_truth_travels_as_a_separate_return_value(injected):
    ds, gt = injected
    assert isinstance(gt, generator.GroundTruth)
    for f in GT_FIELDS:
        assert not hasattr(ds, f)
        assert f not in ds.frame.columns


def test_ground_truth_does_not_reach_the_feature_matrix(features, injected):
    _, gt = injected
    for f in GT_FIELDS:
        assert not any(f in n for n in features.names)
    for c in gt.affected_features:
        assert not any(n.startswith("drift") for n in features.names)


def test_only_the_generator_and_evaluation_helpers_touch_ground_truth(sources):
    """A module that imports GroundTruth has been handed the answer key. The
    permitted set is: the generator that creates it, and the one preprocessing
    diagnostic that takes drift_start_index as an explicit argument."""
    allowed = {"src/data/generator.py"}
    for rel, text in sources.items():
        if rel in allowed:
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {a.name for a in node.names}
                assert "GroundTruth" not in names, f"{rel} imports GroundTruth"
            if isinstance(node, ast.Attribute) and node.attr == "GroundTruth":
                pytest.fail(f"{rel} references generator.GroundTruth")


def test_the_one_function_taking_an_onset_declares_it_as_evaluation_only():
    doc = pp.measure_drift_absorption.__doc__ or ""
    assert "ONLY" in doc and "never an input" in doc
    params = pp.measure_drift_absorption.__code__.co_varnames[:5]
    assert "drift_start_index" in params, \
        "the onset must be an explicit argument, not read from a sidecar"


def test_no_module_reads_the_ground_truth_sidecar(sources, cfg):
    sidecar = Path(cfg["ground_truth"]["sidecar_path"]).name
    for rel, text in sources.items():
        if rel == "src/data/generator.py":
            continue           # writes it
        assert sidecar not in text, f"{rel} mentions the sidecar"
    assert "read_text" not in sources["src/data/generator.py"] or \
        "write_text" in sources["src/data/generator.py"]


def test_the_event_log_offers_no_way_to_read_a_run_back(sources):
    from src.utils import logger as logmod
    for forbidden in ("read", "load", "rows", "readlines"):
        assert not hasattr(logmod.EventLog, forbidden)
    assert "WRITE-ONLY" in (logmod.__doc__ or "")


# -----------------------------------------------------------------------------
# no_results_asserted
# -----------------------------------------------------------------------------


def test_no_module_hard_codes_a_performance_number(sources):
    banned = re.compile(
        r"""\b(accuracy|f1|precision|recall|auc|rmse|mae|latency|score)\s*=\s*"""
        r"""[-+]?\d*\.?\d+""", re.IGNORECASE)
    offences = [f"{rel}:{i}: {ln.strip()}"
                for rel, text in sources.items()
                for i, ln in _code_lines(text)
                if banned.search(ln)]
    assert not offences, "hard-coded result value(s):\n" + "\n".join(offences)


def test_no_module_forces_an_orchestration_decision(sources):
    """The decision must be argmin of the WDS. A literal 'return "cloud"' beside
    a drift test would decide the study's finding in advance."""
    banned = re.compile(r"""return\s+['"](edge|cloud|hybrid)['"]""",
                        re.IGNORECASE)
    for rel, text in sources.items():
        for i, ln in enumerate(text.splitlines(), 1):
            assert not banned.search(ln.split("#", 1)[0]), f"{rel}:{i}"


def test_provenance_still_asserts_that_no_experiment_has_run(provenance):
    rules = provenance["integrity_rules"]
    assert "No experiment has been run" in rules["no_results_asserted"]
    assert "never an input" in rules["no_fabricated_data"]


def test_open_findings_are_recorded_rather_than_quietly_repaired(provenance):
    for key in ("finding_degenerate_outlier_bounds",
                "finding_pre_existing_regime_shift"):
        assert key in provenance, key
        entry = provenance[key]
        for required in ("severity", "claim", "measured", "consequence",
                         "action_taken"):
            assert required in entry, f"{key} lacks {required}"
    deg = provenance["finding_degenerate_outlier_bounds"]
    assert deg["severity"] == "OPEN_DECISION"
    assert deg["action_taken"].startswith("NONE"), \
        "an open finding must not have been silently repaired"
    assert deg["awaiting"] == "user decision"
    shift = provenance["finding_pre_existing_regime_shift"]
    assert shift["severity"] == "MEASURED_FACT"
    assert "leakage" in shift["relevance"], \
        "re-centering test1 on its own statistics would be leakage"


# -----------------------------------------------------------------------------
# Phase discipline: nothing beyond Phase 1 is implemented
# -----------------------------------------------------------------------------


def test_exactly_the_phase_1_modules_are_implemented(sources):
    implemented = {rel for rel, text in sources.items()
                   if "Status   : IMPLEMENTED" in text}
    assert implemented == IMPLEMENTED, (
        f"unexpectedly implemented: {sorted(implemented - IMPLEMENTED)}; "
        f"expected but missing: {sorted(IMPLEMENTED - implemented)}")


def test_later_phase_modules_are_declared_stubs_with_no_logic(sources):
    for rel, text in sources.items():
        if rel in IMPLEMENTED or rel.endswith("__init__.py"):
            continue
        assert "Status   : NOT IMPLEMENTED" in text, rel
        body = [n for n in ast.parse(text).body
                if not isinstance(n, (ast.Expr, ast.Import, ast.ImportFrom))]
        assert not body, f"{rel} contains logic but is marked NOT IMPLEMENTED"


def test_no_phase_1_module_imports_a_later_phase_dependency(sources):
    later = {"river", "xgboost", "simpy", "sklearn"}
    for rel in PHASE_1_MODULES:
        tree = ast.parse(sources[rel])
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                mod = node.names[0].name.split(".")[0]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split(".")[0]
            assert mod not in later, f"{rel} imports {mod} in Phase 1"


def test_configs_park_later_phase_settings_instead_of_deciding_them(cfg):
    reserved = cfg["reserved_for_later_phases"]
    assert reserved, "later-phase settings must be visible, not invented later"
    text = json.dumps(reserved).lower()
    for who in ("wds", "adwin", "lri"):
        assert who in text, f"{who} has no reserved placeholder"
