"""The schemas, the invariants and the documents agree on the record contract."""
from __future__ import annotations

import copy
import json
import re
import sys
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "src"
ROOT = CORE.parent
DOCS = ROOT / "docs"
# The design documents are private, so a public checkout skips checks against them.
HAS_DOCS = (DOCS / "ARCHITECTURE.md").exists()
sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jsonschema

import isolation  # noqa: F401
from repohone import session_validation
from repohone.record_invariants import check_record

SCHEMA_DIR = CORE / "repohone" / "schemas"
SESSION = jsonschema.Draft202012Validator(
    json.loads((SCHEMA_DIR / "session-record.v1.schema.json").read_text()))
T = "2026-09-20T12:00:00Z"


def snap(cls, ordinal):
    return {"ref": f"refs/repohone/snapshots/rh_abcd1234/{cls.replace('_', '-')}/{ordinal}",
            "class": cls, "ordinal": ordinal, "commit": "a" * 40, "tree": "b" * 40,
            "head": "c" * 40, "branch": "main", "taken_at": T, "path": ".",
            "submodules": []}


def turn(index, logical_turn_id):
    return {"index": index, "logical_turn_id": logical_turn_id, "cwd": "/r",
            "prompt_events": [{"ordinal": 1, "at": T, "snapshot": snap("prompt", 1),
                               "content": {"text": "do the thing", "sha256": "d" * 64,
                                           "chars": 12, "redactions": 0}}],
            "stop_events": [{"ordinal": 1, "at": T, "kind": "stop", "error": None,
                             "snapshot": snap("stop", 1), "last_assistant_message": None,
                             "effort": "medium", "model": "claude-opus-5",
                             "host_continuation": False}],
            "workspace_changed_before_turn": False, "completion": "stop",
            "completion_source": "observed", "completed_at": T,
            "permission_mode": "default", "model": "claude-opus-5"}


BASE = {
    "schema": "repohone.session/v1", "schema_version": 1, "contract_version": "1.4",
    "session_id": "rh_abcd1234", "host": {"name": "claude-code", "session_id": "h-1"},
    "lifecycle": {"state": "observed", "started_at": T, "ended_at": None,
                  "end_reason": None, "reconciled_at": None,
                  "starts": [{"at": T, "source": "startup", "cwd": "/r"}]},
    "capture": {"adapter": "claude-code", "adapter_version": "1.0.0",
                "content_policy": "redacted", "cwds": ["/r"], "errors": []},
    "agent": {"name": "claude-code", "version": "2.1.267"},
    "models": [{"model": "claude-opus-5", "from_turn": 1, "observed_at": T,
                "source": "transcript"}],
    "developer": {"id": "dev_" + "e" * 16},
    "repository": {"root": "/r", "id": "repo_" + "f" * 16,
                   "checkout_id": "11111111-2222-3333-4444-555555555555",
                   "start_head": "c" * 40, "start_branch": "main",
                   "branches_observed": ["main"]},
    "turns": [turn(1, "p-1")],
    "session_end_snapshots": [snap("session_end", 1)],
    "transcript": {"source_path": "/t.jsonl", "stored_path": None, "stored_sha256": None,
                   "lines": 10, "redactions": 0, "observed_models": ["claude-opus-5"],
                   "observed_agent_versions": ["2.1.267"]},
    "labels": {"developer_label": None, "labeled_at": None},
    "reconciliation": {"outcome": "pending", "pull_request": None,
                       "agent_final_snapshot": None, "accepted_snapshot": None,
                       "related_session_ids": []},
    "extensions": {},
}


def _to_stop_failure(r):
    r["turns"][0]["stop_events"][0].update(
        kind="stop_failure", error="rate_limit", snapshot=snap("stop_failure", 1))
    r["turns"][0]["completion"] = "stop_failure"
    r["reconciliation"]["agent_final_snapshot"] = {
        "snapshot": snap("stop_failure", 1), "resolution": "provisional"}


# name -> (mutation of BASE, the rule it breaks)
CONTRADICTIONS = {
    "observed but outcome merged": (
        lambda r: r["reconciliation"].update(outcome="merged"),
        "ARCH §19: terminal outcomes are reconciliation's result"),
    "reconciled but outcome pending": (
        lambda r: r["lifecycle"].update(state="reconciled", reconciled_at=T),
        "ARCH §19: a reconciled session has a resolved outcome"),
    "reconciled without reconciled_at": (
        lambda r: (r["lifecycle"].update(state="reconciled"),
                   r["reconciliation"].update(outcome="merged")),
        "reconciled without a time it happened"),
    "observed with reconciled_at": (
        lambda r: r["lifecycle"].update(reconciled_at=T),
        "reconciliation time on an unreconciled record"),
    "observed with accepted_snapshot": (
        lambda r: r["reconciliation"].update(
            accepted_snapshot={"tree": "b" * 40, "commit": "a" * 40}),
        "ARCH §19: the accepted tree is reconciliation output"),
    "agent-final on a stop_failure-only session": (
        _to_stop_failure,
        "StopFailure turns are infrastructure errors, never agent-final"),
    "session_end ordinal disagrees with its ref": (
        lambda r: r["session_end_snapshots"][0].update(
            ref="refs/repohone/snapshots/rh_abcd1234/session-end/99"),
        "ARCH §17: the ref encodes the allocated ordinal"),
    "prompt ordinal disagrees with its ref": (
        lambda r: r["turns"][0]["prompt_events"][0]["snapshot"].update(
            ref="refs/repohone/snapshots/rh_abcd1234/prompt/42"),
        "ARCH §17: the ref encodes the allocated ordinal"),
    "turn model set while models[] is empty": (
        lambda r: r.update(models=[]),
        "unobserved means empty models and null turns[].model"),
    "two turns share one logical_turn_id": (
        lambda r: r["turns"].append(turn(2, "p-1")),
        "ARCH §14: logical_turn_id identifies one user turn"),
}


def verdicts(rec):
    return list(SESSION.iter_errors(rec)), check_record(rec)


class ContradictionsAreRejected(unittest.TestCase):
    """Each record breaks the architecture; the schema or the invariants must say so."""

    def test_the_base_record_passes_both_layers(self):
        # Otherwise every rejection below could be the base's fault.
        self.assertEqual(([], []), verdicts(BASE))

    def test_every_contradiction_is_caught_by_a_layer(self):
        for name, (mutate, why) in CONTRADICTIONS.items():
            with self.subTest(name):
                rec = copy.deepcopy(BASE)
                mutate(rec)
                schema_errors, violations = verdicts(rec)
                self.assertTrue(schema_errors or violations, f"accepted: {why}")


class SchemasAreWellFormed(unittest.TestCase):

    def test_every_schema_is_valid_json_schema(self):
        for kind, schema in session_validation.SCHEMAS.items():
            with self.subTest(kind):
                jsonschema.Draft202012Validator.check_schema(schema)

    def test_every_ref_resolves(self):
        for kind, schema in session_validation.SCHEMAS.items():
            with self.subTest(kind):
                refs = set(re.findall(r'"\$ref":\s*"#/\$defs/(\w+)"', json.dumps(schema)))
                self.assertEqual(set(), refs - set(schema.get("$defs", {})))


class TheDocumentsAgreeOnTheContract(unittest.TestCase):
    """Regression: the runtime reached 1.18 while three places still said 1.16,
    including an operator-facing error message."""

    def runtime_version(self):
        text = (CORE / "repohone" / "__init__.py").read_text()
        return re.search(r'CONTRACT_VERSION = "([\d.]+)"', text).group(1)

    def architecture(self):
        return (DOCS / "ARCHITECTURE.md").read_text()

    @unittest.skipUnless(HAS_DOCS, "design documents are not in this checkout")
    def test_the_frozen_status_line_matches_the_runtime(self):
        header = self.architecture().split("\n## ")[0]
        self.assertIn(f"contract {self.runtime_version()}.", header)

    @unittest.skipUnless(HAS_DOCS, "design documents are not in this checkout")
    def test_the_implementation_plan_matches_the_runtime(self):
        plan = (DOCS / "IMPLEMENTATION_PLAN.md").read_text()
        self.assertIn(f"frozen at contract {self.runtime_version()}.", plan)

    @unittest.skipUnless(HAS_DOCS, "design documents are not in this checkout")
    def test_the_history_records_the_running_contract(self):
        history = self.architecture().split("### Contract history")[1]
        self.assertRegex(history.split("```")[1],
                         rf"(?m)^{re.escape(self.runtime_version())}\s")

    def test_no_supported_range_is_written_out_by_hand(self):
        """The newest supported contract is data; prose copies of it go stale."""
        source = (CORE / "repohone" / "artifacts.py").read_text()
        body = source.split("SUPPORTED_CONTRACTS", 2)[-1]
        self.assertNotIn("contracts end at 1.", body)


if __name__ == "__main__":
    unittest.main()
