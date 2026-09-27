"""Phase 3 tests: bounded retrospective diagnosis.

The engine is stubbed so every terminal state is reachable deterministically and
no test reaches a model provider. Each test names the section it enforces.

Run:  python3 -m unittest test_phase3
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "src"
ROOT = CORE.parent
sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from repohone import (
    CONTRACT_VERSION,
    artifacts,
    bootstrap,
    cli,
    diagnosis,
    evidence,
    fingerprint,
    identity,
    paths,
    profile,
    reasoning,
    record,
    selection,
    state,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import canonical_occurrence, ensure_session

SCHEMA_PATH = CORE / "repohone" / "schemas" / "diagnosis.v1.schema.json"
HOOK = CORE / "repohone_hook.py"

try:
    import jsonschema
    VALIDATOR = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
except ImportError:
    VALIDATOR = None


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                          text=True, check=True).stdout.strip()


GOOD_REPLY = {
    "outcome": "SUCCESS",
    "root_cause": {"class": "MISSING_CONTEXT", "summary": "The convention was never written down."},
    "confidence": "medium",
    "required_property": "stats functions must return typed results",
    "corrective_turn": 2,
    "fingerprint": {"area": "reporting-layer", "behavior": "untyped-return-values"},
    "evidence_for": ["the developer restated a rule that appears nowhere in the repo"],
    "evidence_against": ["only one correction was observed"],
    "risks": ["over-fitting a single correction"],
}


class FakeEngine:
    name = "fake"

    def __init__(self, reply=None, raises=None):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply or GOOD_REPLY)
        self.raises = raises
        self.prompt = None
        self.calls = 0

    def run(self, prompt):
        self.calls += 1
        self.prompt = prompt
        if self.raises:
            raise self.raises
        return reasoning.Reply(text=self.reply, model="fake-model", isolated=True)


class Fixture(unittest.TestCase):
    """A real repository with one corrected two-turn session already captured."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-dx-"))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.data = self.tmp / "data"
        os.environ["REPOHONE_DATA_DIR"] = str(self.data)
        git(["init", "-q"], self.repo)
        git(["config", "user.email", "d@e.com"], self.repo)
        git(["config", "user.name", "D"], self.repo)
        (self.repo / "src").mkdir()
        (self.repo / "src" / "stats.py").write_text("def count():\n    return 1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "init"], self.repo)
        (self.repo / ".repohone").mkdir()
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        self.checkout = identity.checkout_id(self.repo)
        state.initialize(self.checkout)
        self.session = identity.session_id("claude-code", "s-dx")

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fire(self, name, **kw):
        payload = {"hook_event_name": name, "session_id": "s-dx", "cwd": str(self.repo),
                   "transcript_path": "/tmp/x"}
        payload.update(kw)
        return subprocess.run([sys.executable, str(HOOK), "hook"], input=json.dumps(payload),
                              cwd=str(self.repo), capture_output=True, text=True,
                              env=dict(os.environ))

    def capture_corrected_session(self):
        self.fire("UserPromptSubmit", prompt="add count_users to src/stats.py",
                  prompt_id="t1")
        (self.repo / "src" / "stats.py").write_text("def count_users():\n    return (1,)\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="Added count_users.")
        self.fire("UserPromptSubmit", prompt="No. It must return a typed dataclass.",
                  prompt_id="t2")
        (self.repo / "src" / "stats.py").write_text(
            "from dataclasses import dataclass\n\n\n@dataclass\nclass C:\n    n: int\n")
        self.fire("Stop", prompt_id="t2", last_assistant_message="Fixed.")

    def diagnose(self, engine, rule=None):
        return diagnosis.run(self.repo, self.checkout, self.session, engine,
                             rule_statement=rule)

    def assert_conformant(self, rec):
        if VALIDATOR is not None:
            errors = sorted(VALIDATOR.iter_errors(rec), key=lambda e: e.path)
            self.assertEqual([], ["/".join(map(str, e.path)) + ": " + e.message
                                  for e in errors])


class PassiveCaptureMakesNoModelCalls(Fixture):
    """ARCHITECTURE §1.1 — capture is automatic, analysis is explicit.

    Instrumented at the **process boundary**: a fake `claude` executable is put
    on PATH and logs every invocation. Patching the engine class in this process
    would not observe the hook subprocesses, so it could not fail.
    """

    def setUp(self):
        super().setUp()
        self.calls = self.tmp / "model-calls.log"
        fake_bin = self.tmp / "bin"
        fake_bin.mkdir()
        fake = fake_bin / "claude"
        fake.write_text(
            "#!/bin/sh\n"
            f'echo "$@" >> "{self.calls}"\n'
            'echo \'{"result":"{}","is_error":false}\'\n')
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")

    def fire(self, name, **kw):
        payload = {"hook_event_name": name, "session_id": "s-dx",
                   "cwd": str(self.repo), "transcript_path": "/tmp/x"}
        payload.update(kw)
        return subprocess.run([sys.executable, str(HOOK), "hook"],
                              input=json.dumps(payload), cwd=str(self.repo),
                              capture_output=True, text=True, env=self.env)

    def model_calls(self):
        return self.calls.read_text().splitlines() if self.calls.is_file() else []

    def test_a_whole_session_invokes_no_model(self):
        self.fire("SessionStart", source="startup")
        self.fire("UserPromptSubmit", prompt="implement feature X", prompt_id="t1")
        (self.repo / "src" / "stats.py").write_text("x = 1\n")
        self.fire("PreToolUse", prompt_id="t1", tool_name="Read", tool_use_id="tu1",
                  tool_input={"file_path": str(self.repo / "src/stats.py")})
        self.fire("PostToolUse", prompt_id="t1", tool_name="Read", tool_use_id="tu1",
                  tool_response={"stdout": "x"}, duration_ms=3)
        self.fire("Stop", prompt_id="t1", last_assistant_message="done")
        self.fire("SessionEnd", reason="clear")
        self.assertEqual([], self.model_calls(), "passive capture reached a model")

    def test_capture_still_produced_the_evidence(self):
        """The silence must come from not analysing, not from not capturing."""
        self.fire("UserPromptSubmit", prompt="implement feature X", prompt_id="t1")
        (self.repo / "src" / "stats.py").write_text("x = 2\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="done")
        self.fire("SessionEnd", reason="clear")
        rec = record.load(self.checkout, self.session)
        self.assertEqual(1, len(rec["turns"]))
        self.assertTrue(rec["session_end_snapshots"])
        self.assertEqual([], self.model_calls())

    def test_no_fingerprint_is_created_without_an_analysis(self):
        """Semantic work must not happen merely because data exists."""
        self.capture_corrected_session()
        self.fire("SessionEnd", reason="clear")
        self.assertEqual([], fingerprint.known(self.checkout, None))
        self.assertEqual([], diagnosis.load_all(self.checkout))
        self.assertEqual([], self.model_calls())

    def test_only_an_explicit_analysis_reaches_a_model(self):
        self.capture_corrected_session()
        self.assertEqual([], self.model_calls())
        subprocess.run([sys.executable, str(HOOK), "diagnose", self.session,
                        "--path", str(self.repo), "--yes"],
                       capture_output=True, text=True, env=self.env)
        self.assertTrue(self.model_calls(),
                        "an explicit analysis did not reach the model")


class EgressPolicy(unittest.TestCase):
    """ARCH §11 — effective policy is the most restrictive of project and local."""

    def test_local_may_only_tighten(self):
        self.assertEqual("disabled", reasoning.effective_policy("interactive", "disabled"))
        self.assertEqual("disabled", reasoning.effective_policy("disabled", "interactive"))

    def test_absent_local_leaves_the_project_policy(self):
        self.assertEqual("interactive", reasoning.effective_policy("interactive", None))

    def test_a_developer_can_disable_reasoning_entirely(self):
        self.assertEqual("disabled", reasoning.effective_policy("interactive", "disabled"))

    def test_automatic_is_not_in_the_beta_contract(self):
        """ARCH §11 — reasoning runs only on explicit intent, so there is no
        policy value meaning 'reason without being asked'."""
        self.assertNotIn("automatic", reasoning._RESTRICTIVENESS)
        from repohone import profile as prof
        self.assertNotIn("automatic", prof.EGRESS)


class ADamagedRestrictionIsStillARestriction(unittest.TestCase):
    """Regression: malformed or unknown local settings returned None, which
    `effective_policy` reads as "the developer set nothing" — so a damaged
    persistent restriction silently reverted to whatever the project allows."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-egress-"))
        os.environ["REPOHONE_DATA_DIR"] = str(self.tmp)
        self.settings = self.tmp / "settings.json"

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_an_absent_file_is_not_a_restriction(self):
        policy, problem = reasoning.local_policy()
        self.assertIsNone(policy)
        self.assertIsNone(problem)
        self.assertEqual("interactive",
                         reasoning.effective_policy("interactive", policy))

    def test_a_valid_restriction_is_obeyed(self):
        self.settings.write_text('{"reasoning_egress": "disabled"}')
        policy, problem = reasoning.local_policy()
        self.assertEqual("disabled", policy)
        self.assertIsNone(problem)

    def test_a_damaged_restriction_fails_closed_and_says_why(self):
        for body in ('{"reasoning_egress": ',
                     '{"reasoning_egress": "automatic"}',
                     '{"reasoning_egress": 7}',
                     '["reasoning_egress"]',
                     'null'):
            with self.subTest(body=body):
                self.settings.write_text(body)
                policy, problem = reasoning.local_policy()
                self.assertEqual(reasoning.DISABLED, policy)
                self.assertTrue(problem)
                self.assertEqual("disabled",
                                 reasoning.effective_policy("interactive", policy))

    def _clear(self):
        """Remove whatever is at the settings path, whatever it is."""
        mode = os.lstat(self.settings).st_mode
        if stat.S_ISDIR(mode):
            self.settings.rmdir()
            return
        if not stat.S_ISLNK(mode):
            os.chmod(self.settings, 0o644)
        self.settings.unlink()

    def _policy_within(self, seconds=5):
        """A FIFO blocks whoever opens it, so a regression here would hang the
        suite instead of failing it."""
        outcome = {}
        worker = threading.Thread(
            target=lambda: outcome.update(result=reasoning.local_policy()), daemon=True)
        worker.start()
        worker.join(seconds)
        self.assertFalse(worker.is_alive(), "reading local settings blocked")
        return outcome["result"]

    def test_every_filesystem_object_that_is_not_a_readable_file_fails_closed(self):
        """Regression: `is_file()` answered False for a directory, so a damaged
        settings path read as "no restriction" before any read was attempted."""
        elsewhere = self.tmp / "elsewhere.json"
        elsewhere.write_text('{"reasoning_egress": "interactive"}')
        folder = self.tmp / "folder"
        folder.mkdir()
        damaged = {
            "directory": lambda: self.settings.mkdir(),
            "dangling link": lambda: self.settings.symlink_to(self.tmp / "gone.json"),
            "link to a directory": lambda: self.settings.symlink_to(folder),
            "fifo": lambda: os.mkfifo(self.settings),
            "unreadable": lambda: (self.settings.write_text('{"reasoning_egress": "disabled"}'),
                                   os.chmod(self.settings, 0)),
        }
        for label, make in damaged.items():
            with self.subTest(case=label):
                make()
                try:
                    policy, problem = self._policy_within()
                finally:
                    self._clear()
                self.assertEqual(reasoning.DISABLED, policy)
                self.assertTrue(problem)

    def test_only_genuine_absence_and_a_readable_file_are_trusted(self):
        self.assertEqual((None, None), self._policy_within())
        target = self.tmp / "real.json"
        target.write_text('{"reasoning_egress": "disabled"}')
        self.settings.symlink_to(target)
        self.assertEqual(("disabled", None), self._policy_within(),
                         "a link to a valid file is a valid setting")

    def test_the_retired_policy_value_is_refused_here_too(self):
        """§11 retired `automatic`; an invalid profile suspends capture, and an
        invalid local setting must not be the one place it still passes."""
        self.settings.write_text('{"reasoning_egress": "automatic"}')
        policy, problem = reasoning.local_policy()
        self.assertEqual(reasoning.DISABLED, policy)
        self.assertIn("automatic", problem)

    def test_doctor_reports_it_rather_than_a_policy(self):
        repo = self.tmp / "repo"
        repo.mkdir()
        git(["init", "-q"], repo)
        git(["config", "user.email", "d@e.com"], repo)
        git(["config", "user.name", "D"], repo)
        (repo / "a.txt").write_text("x\n")
        git(["add", "-A"], repo)
        git(["commit", "-qm", "init"], repo)
        (repo / ".repohone").mkdir()
        (repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(repo, profile.load(repo))
        self.settings.write_text('{"reasoning_egress": "automatic"}')
        from repohone import doctor
        rows = {name: (status, detail) for name, status, detail in doctor.run(repo)}
        status, detail = rows["reasoning egress"]
        self.assertEqual(doctor.FAIL, status)
        self.assertIn("automatic", detail)


class PolicyGate(Fixture):
    """Reasoning is an egress surface: nothing leaves without an approved policy."""

    def _cli(self, *args):
        return subprocess.run([sys.executable, str(HOOK), "diagnose", self.session,
                               "--path", str(self.repo), *args],
                              capture_output=True, text=True, env=dict(os.environ))

    def test_disabled_sends_nothing_and_writes_no_record(self):
        (self.repo / ".repohone" / "profile.yaml").write_text(
            "profile_version: 1\nreasoning_egress: disabled\n")
        proc = self._cli()
        self.assertEqual(3, proc.returncode)
        self.assertIn("disabled by policy", proc.stderr)
        self.assertEqual([], diagnosis.load_all(self.checkout))

    def test_interactive_requires_explicit_approval(self):
        proc = self._cli()
        self.assertEqual(3, proc.returncode)
        self.assertIn("--yes", proc.stderr)
        self.assertEqual([], diagnosis.load_all(self.checkout))

    def test_an_uninitialized_repository_is_never_diagnosed(self):
        (self.repo / ".repohone" / "profile.yaml").unlink()
        proc = self._cli("--yes")
        self.assertEqual(2, proc.returncode)


class Budget(Fixture):
    """ARCH §12 — 8,000 bytes and 8 items, whichever is reached first."""

    def test_item_count_is_capped(self):
        self.capture_corrected_session()
        rec = record.load(self.checkout, self.session)
        selection = evidence.select(rec, self.repo, max_items=2)
        self.assertEqual(2, len(selection.items))
        self.assertGreater(selection.skipped, 0)

    def test_token_budget_is_capped(self):
        self.capture_corrected_session()
        rec = record.load(self.checkout, self.session)
        selection = evidence.select(rec, self.repo, max_bytes=40)
        self.assertLessEqual(selection.bytes, 40)

    def test_the_budget_is_recorded_with_the_diagnosis(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine())
        self.assertEqual(evidence.MAX_BYTES, result["budget"]["max_tokens"])
        self.assertEqual(evidence.MAX_ITEMS, result["budget"]["max_items"])
        self.assertLessEqual(result["budget"]["items_sent"], evidence.MAX_ITEMS)
        self.assertLessEqual(result["budget"]["estimated_tokens"], evidence.MAX_BYTES)

    def test_secrets_never_reach_the_payload(self):
        self.fire("UserPromptSubmit", prompt="deploy with AKIAIOSFODNN7EXAMPLE",
                  prompt_id="t1")
        (self.repo / "src" / "stats.py").write_text("x = 2\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="done")
        engine = FakeEngine()
        self.diagnose(engine)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", engine.prompt)

    def test_absolute_paths_never_reach_the_payload(self):
        self.capture_corrected_session()
        engine = FakeEngine()
        self.diagnose(engine)
        self.assertNotIn(str(self.repo), engine.prompt)


class HistoricalInstructions(Fixture):
    """Audit B6: instruction files were read from the working tree, so a rule
    written AFTER a session was presented as context the agent had — turning a
    MISSING_CONTEXT failure into an apparent AGENT_REASONING_ERROR."""

    def _kinds(self):
        rec = record.load(self.checkout, self.session)
        return {item.kind for item in evidence.select(rec, self.repo).items}

    def test_an_instruction_added_after_the_session_is_not_evidence(self):
        self.capture_corrected_session()
        self.assertNotIn("project_instruction", self._kinds())
        (self.repo / "AGENTS.md").write_text("- You must never use print().\n")
        self.assertNotIn("project_instruction", self._kinds(),
                         "a later instruction was presented as session context")

    def test_an_instruction_present_at_session_time_is_evidence(self):
        (self.repo / "CLAUDE.md").write_text("- Handlers must not import persistence.\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "rules"], self.repo)
        self.capture_corrected_session()
        self.assertIn("project_instruction", self._kinds())

    def test_the_reference_names_the_tree_it_came_from(self):
        (self.repo / "CLAUDE.md").write_text("- Always add a test.\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "rules"], self.repo)
        self.capture_corrected_session()
        rec = record.load(self.checkout, self.session)
        refs = [i.ref for i in evidence.select(rec, self.repo).items
                if i.kind == "project_instruction"]
        self.assertTrue(refs)
        self.assertIn("CLAUDE.md@", refs[0])

    def test_a_session_with_no_snapshot_offers_no_instructions(self):
        """Absent evidence is honest; the working tree is not a substitute."""
        rec = {"turns": [{"index": 1, "logical_turn_id": "t1",
                          "workspace_changed_before_turn": None,
                          "prompt_events": [{"ordinal": 1, "content": {"text": "go"},
                                             "snapshot": None}],
                          "stop_events": []}],
               "labels": {"developer_label": None}, "extensions": {}}
        (self.repo / "CLAUDE.md").write_text("- Always add a test.\n")
        kinds = {item.kind for item in evidence.select(rec, self.repo).items}
        self.assertNotIn("project_instruction", kinds)


class CorrectiveTurn(Fixture):
    """Audit B7: the correction boundary must be identified, not inferred."""

    def test_the_prompt_asks_which_turn_the_correction_was_in(self):
        self.capture_corrected_session()
        engine = FakeEngine()
        self.diagnose(engine)
        self.assertIn("corrective_turn", engine.prompt)
        self.assertIn('"corrective_turn":2', engine.prompt)
        self.assertNotIn('"corrective_turn":"1-based', engine.prompt)
        self.assertIn("request is not a correction", engine.prompt)

    def test_an_identified_turn_is_recorded(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine(dict(GOOD_REPLY, corrective_turn=2)))
        self.assertEqual(2, result["corrective_turn"])
        self.assert_conformant(result)

    def test_no_correction_is_a_valid_answer(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine(dict(GOOD_REPLY, corrective_turn=None)))
        self.assertEqual("SUCCESS", result["outcome"])
        self.assertIsNone(result["corrective_turn"])
        self.assert_conformant(result)

    def test_a_nonsense_turn_is_invalid_output(self):
        self.capture_corrected_session()
        for bad in ("two", 0, -1, True):
            result = self.diagnose(FakeEngine(dict(GOOD_REPLY, corrective_turn=bad)))
            self.assertEqual("INVALID_OUTPUT", result["outcome"], repr(bad))

    def test_a_failed_diagnosis_names_no_turn(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine("prose"))
        self.assertIsNone(result["corrective_turn"])
        self.assert_conformant(result)


class InsufficientEvidence(Fixture):
    """ARCH §12 — the answer is INSUFFICIENT_EVIDENCE, never context expansion."""

    def test_a_session_with_no_behaviour_is_not_sent_to_the_model(self):
        self.fire("UserPromptSubmit", prompt="do the thing", prompt_id="t1")
        engine = FakeEngine()
        result = self.diagnose(engine)
        self.assertEqual("INSUFFICIENT_EVIDENCE", result["outcome"])
        self.assertEqual(0, engine.calls, "evidence was sent despite being insufficient")

    def test_insufficient_evidence_leaves_no_partial_diagnosis(self):
        self.fire("UserPromptSubmit", prompt="do the thing", prompt_id="t1")
        result = self.diagnose(FakeEngine())
        self.assertIsNone(result["root_cause"])
        self.assertIsNone(result["required_property"])
        self.assertIsNone(result["confidence"])
        self.assert_conformant(result)


class TerminalStates(Fixture):
    """ARCH §12.3 — exactly one state, and partial output is never a proposal."""

    def setUp(self):
        super().setUp()
        self.capture_corrected_session()

    def test_success(self):
        result = self.diagnose(FakeEngine())
        self.assertEqual("SUCCESS", result["outcome"])
        self.assertEqual("MISSING_CONTEXT", result["root_cause"]["class"])
        self.assertEqual("medium", result["confidence"])
        self.assert_conformant(result)

    def test_model_unavailable_preserves_evidence(self):
        result = self.diagnose(FakeEngine(raises=reasoning.Unavailable("no engine")))
        self.assertEqual("MODEL_UNAVAILABLE", result["outcome"])
        self.assertTrue(result["evidence_refs"], "evidence refs were lost")
        self.assert_conformant(result)

    def test_interrupted_is_not_a_result(self):
        result = self.diagnose(FakeEngine(raises=reasoning.Interrupted("timeout")))
        self.assertEqual("INTERRUPTED", result["outcome"])
        self.assertIsNone(result["required_property"])
        self.assert_conformant(result)

    def test_no_actionable_improvement_is_a_valid_answer(self):
        reply = dict(GOOD_REPLY, outcome="NO_ACTIONABLE_PROJECT_IMPROVEMENT",
                     root_cause=None, confidence=None, required_property=None,
                     fingerprint=None)
        result = self.diagnose(FakeEngine(reply))
        self.assertEqual("NO_ACTIONABLE_PROJECT_IMPROVEMENT", result["outcome"])
        self.assert_conformant(result)

    def test_prose_instead_of_json_is_invalid_output(self):
        result = self.diagnose(FakeEngine("I think the handlers are fine."))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])
        self.assert_conformant(result)

    def test_fenced_json_is_accepted(self):
        result = self.diagnose(FakeEngine("```json\n" + json.dumps(GOOD_REPLY) + "\n```"))
        self.assertEqual("SUCCESS", result["outcome"])

    def test_a_class_outside_the_taxonomy_is_invalid(self):
        reply = dict(GOOD_REPLY, root_cause={"class": "VIBES", "summary": "x"})
        result = self.diagnose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])
        self.assertIn("taxonomy", result["failure"]["reason"])

    def test_missing_contradicting_evidence_is_invalid(self):
        """LEARNING_PLAN §11 — the question is load-bearing."""
        reply = dict(GOOD_REPLY)
        del reply["evidence_against"]
        result = self.diagnose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])
        self.assertIn("evidence_against", result["failure"]["reason"])

    def test_an_empty_required_property_is_invalid(self):
        result = self.diagnose(FakeEngine(dict(GOOD_REPLY, required_property="  ")))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])

    def test_a_missing_confidence_is_never_inferred(self):
        reply = dict(GOOD_REPLY)
        del reply["confidence"]
        result = self.diagnose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])

    def test_every_failure_still_writes_a_record(self):
        for engine in (FakeEngine("prose"), FakeEngine(raises=reasoning.Unavailable("x")),
                       FakeEngine(raises=reasoning.Interrupted("y"))):
            self.diagnose(engine)
        self.assertEqual(3, len(diagnosis.load_all(self.checkout)))


class NoMechanism(Fixture):
    """Phase 3 diagnoses a property. Discovery is Phase 4, selection is Phase 5."""

    def setUp(self):
        super().setUp()
        self.capture_corrected_session()

    def test_a_property_naming_a_tool_is_rejected(self):
        for bad in ("add an import-linter rule",
                    "write a unit test for stats functions",
                    "configure eslint to ban raw tuples",
                    "document the convention in CLAUDE.md"):
            result = self.diagnose(FakeEngine(dict(GOOD_REPLY, required_property=bad)))
            self.assertEqual("INVALID_OUTPUT", result["outcome"], bad)
            self.assertIn("mechanism", result["failure"]["reason"])

    def test_a_genuine_property_is_accepted(self):
        for good in ("stats functions must return typed results",
                     "handlers must not import persistence directly",
                     "error paths must be verified before merge"):
            result = self.diagnose(FakeEngine(dict(GOOD_REPLY, required_property=good)))
            self.assertEqual("SUCCESS", result["outcome"], good)

    def test_tools_the_blocklist_used_to_miss_are_caught(self):
        """Regression: the guard held nine common tool names, so eight of nine
        properties that plainly prescribed a mechanism went through."""
        for bad in ("the build must run Biome on every commit",
                    "tsc --noEmit must pass before merge",
                    "every handler must be covered by a Playwright spec",
                    "the repository must enable Dependabot",
                    "contributors must run cargo clippy",
                    "a CODEOWNERS entry must exist for api/",
                    "the Makefile must have a verify target",
                    "add a pre-commit hook for formatting",
                    "configure the linter to forbid direct db imports"):
            self.assertIsNotNone(diagnosis.names_a_mechanism(bad), bad)

    def test_a_property_may_mention_a_mechanism_it_does_not_prescribe(self):
        """Regression: a bare substring match discarded whole diagnoses whose
        property was sound. A kind counts only next to a prescriptive verb."""
        for good in ("code review comments must be resolved before merge",
                     "test suite coverage of error paths must not regress",
                     "adding an endpoint must not change existing response shapes",
                     "database migrations must be reversible",
                     "errors returned to callers must not leak internal identifiers"):
            self.assertIsNone(diagnosis.names_a_mechanism(good), good)

    def test_prescribing_the_same_kind_is_still_refused(self):
        for bad in ("add a test suite for the error paths",
                    "introduce code review for api/ changes",
                    "require a ci check on every merge",
                    "use a git hook to block commits",
                    "the project should adopt a linter",
                    "every handler needs a new integration test",
                    "writing unit tests for handlers is required",
                    "set up a git hook that blocks the commit"):
            self.assertIsNotNone(diagnosis.names_a_mechanism(bad), bad)

    def test_what_the_agent_must_know_about_a_mechanism_is_a_property(self):
        """Regression: a live diagnosis about running this repository's tests was
        discarded for mentioning the test suite; a problem about tests cannot
        be described without the word."""
        for good in ("how to run the project's test suite must be discoverable "
                     "without guessing",
                     "agents must know which command runs the test suite and must "
                     "read its result before reporting done",
                     "the agent must use the existing test suite rather than guess "
                     "a runner",
                     "a failing ci check must be reported, not retried silently"):
            self.assertIsNone(diagnosis.names_a_mechanism(good), good)
        prop = "how to run the project's test suite must be discoverable without guessing"
        result = self.diagnose(FakeEngine(dict(GOOD_REPLY, required_property=prop)))
        self.assertEqual("SUCCESS", result["outcome"])

    def test_the_prompt_forbids_naming_a_mechanism(self):
        engine = FakeEngine()
        self.diagnose(engine)
        self.assertIn("must NOT name or suggest a mechanism", engine.prompt)

    def test_the_prompt_asks_every_pass_one_question(self):
        engine = FakeEngine()
        self.diagnose(engine)
        for question in ("What objectively happened",
                         "What did the developer change or correct",
                         "What property is the project missing",
                         "What evidence supports this? What contradicts it",
                         "Is any project improvement warranted at all"):
            self.assertIn(question, engine.prompt)


class ToolEvidence(Fixture):
    """Phase 2's outcomes feed Phase 3: a check that produced no result is the
    strongest available signal of a verification gap."""

    def test_a_failed_tool_call_is_offered_as_evidence(self):
        self.fire("UserPromptSubmit", prompt="run the tests", prompt_id="t1")
        self.fire("PreToolUse", prompt_id="t1", tool_name="Bash", tool_use_id="tu1",
                  tool_input={"command": "pytest -q"})
        (self.repo / "src" / "stats.py").write_text("x = 3\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="done")
        engine = FakeEngine()
        result = self.diagnose(engine)
        kinds = {r["kind"] for r in result["evidence_refs"]}
        self.assertIn("tool_failure", kinds)
        self.assertIn("pytest -q", engine.prompt)

    def test_a_successful_tool_call_is_not_offered_as_a_failure(self):
        self.fire("UserPromptSubmit", prompt="run the tests", prompt_id="t1")
        self.fire("PreToolUse", prompt_id="t1", tool_name="Bash", tool_use_id="tu1",
                  tool_input={"command": "pytest -q"})
        self.fire("PostToolUse", prompt_id="t1", tool_name="Bash", tool_use_id="tu1",
                  tool_response={"stdout": "ok", "stderr": ""}, duration_ms=5)
        (self.repo / "src" / "stats.py").write_text("x = 4\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="done")
        result = self.diagnose(FakeEngine())
        kinds = {r["kind"] for r in result["evidence_refs"]}
        self.assertNotIn("tool_failure", kinds)


class ReadAndIgnoredIsNotMissing(Fixture):
    """MISSING_CONTEXT and AGENT_REASONING_ERROR leave identical trees and call
    for opposite improvements; what the agent read is what tells them apart.

    Regression: tool events reached the model only for calls with no result,
    and only the first 800 characters of each instruction file were sent. An
    agent that read CLAUDE.md and broke its rule on line 25 looked like one that
    was never told."""

    RULE = "- HTTP handlers must never import the db module; go through services/."

    def long_instructions(self, rule_line):
        body = ("# Orders service\n\n## Style\n\n"
                + "".join(f"- Style rule {i}: keep functions under 40 lines and "
                          f"name them clearly.\n" for i in range(14))
                + "\n## Layering\n\n" + rule_line + "\n")
        (self.repo / "CLAUDE.md").write_text(body)
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "instructions"], self.repo)
        return body

    def read_then_corrected(self, correction):
        self.fire("UserPromptSubmit", prompt="add a recent-orders endpoint", prompt_id="t1")
        claude_md = str(self.repo / "CLAUDE.md")
        self.fire("PreToolUse", prompt_id="t1", tool_name="Read", tool_use_id="r1",
                  tool_input={"file_path": claude_md})
        self.fire("PostToolUse", prompt_id="t1", tool_name="Read", tool_use_id="r1",
                  tool_input={"file_path": claude_md}, tool_response={"type": "text"})
        self.fire("PreToolUse", prompt_id="t1", tool_name="Bash", tool_use_id="b1",
                  tool_input={"command": "make check"})
        (self.repo / "src" / "stats.py").write_text("import db\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="added")
        self.fire("UserPromptSubmit", prompt=correction, prompt_id="t2")
        (self.repo / "src" / "stats.py").write_text("from services import orders\n")
        self.fire("Stop", prompt_id="t2", last_assistant_message="fixed")

    def test_the_model_is_told_what_the_agent_read_and_ran(self):
        self.long_instructions(self.RULE)
        self.read_then_corrected("No - handlers never import db; use services.")
        engine = FakeEngine()
        rec = diagnosis.run(self.repo, self.checkout, self.session, engine,
                            corrective_turn=2)
        self.assertIn("turn 1: 2 calls (Read 1, Bash 1), 1 failed: "
                      "Bash make check (no result); Read CLAUDE.md", engine.prompt)
        self.assertIn("consulted", {r["kind"] for r in rec["evidence_refs"]})
        self.assert_conformant(rec)

    def test_the_instruction_lines_that_bear_on_the_correction_are_sent(self):
        body = self.long_instructions(self.RULE)
        self.assertGreater(body.index(self.RULE), evidence.INSTRUCTION_CHARS)
        self.read_then_corrected("No - handlers never import db; use services.")
        engine = FakeEngine()
        diagnosis.run(self.repo, self.checkout, self.session, engine, corrective_turn=2)
        self.assertIn(self.RULE, engine.prompt)
        self.assertNotIn("Style rule 3:", engine.prompt)

    def test_with_nothing_in_common_the_head_is_sent(self):
        self.long_instructions(self.RULE)
        self.read_then_corrected("Wrong. Try again.")
        engine = FakeEngine()
        diagnosis.run(self.repo, self.checkout, self.session, engine, corrective_turn=2)
        self.assertIn("Style rule 0:", engine.prompt)
        self.assertNotIn("excerpt:", engine.prompt)

    def test_a_turn_with_many_calls_stays_bounded(self):
        self.fire("UserPromptSubmit", prompt="explore everything", prompt_id="t1")
        for i in range(40):
            self.fire("PreToolUse", prompt_id="t1", tool_name="Read", tool_use_id=f"r{i}",
                      tool_input={"file_path": str(self.repo / f"src/module_{i}.py")})
        (self.repo / "src" / "stats.py").write_text("x = 9\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="explored")
        rec = record.load(self.checkout, self.session)
        consulted = evidence._consulted(rec, rec["turns"], self.repo)
        self.assertLessEqual(len(consulted), evidence.CONSULTED_TURN_CHARS + len(" … +99 more"))
        self.assertRegex(consulted, r" … \+\d+ more$")

    def test_every_call_is_counted_when_the_list_is_cut(self):
        """Regression: a turn of twenty attempts reached the model as three
        commands, and the diagnosis called the evidence insufficient."""
        self.fire("UserPromptSubmit", prompt="run the tests", prompt_id="t1")
        for i in range(30):
            self.fire("PreToolUse", prompt_id="t1", tool_name="Read", tool_use_id=f"r{i}",
                      tool_input={"file_path": str(self.repo / f"src/module_{i}.py")})
            self.fire("PostToolUse", prompt_id="t1", tool_name="Read", tool_use_id=f"r{i}",
                      tool_input={}, tool_response={"type": "text"})
        for i in range(5):
            self.fire("PreToolUse", prompt_id="t1", tool_name="Bash", tool_use_id=f"b{i}",
                      tool_input={"command": f"pytest attempt-{i}"})
        (self.repo / "src" / "stats.py").write_text("x = 9\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="gave up")
        rec = record.load(self.checkout, self.session)
        consulted = evidence._consulted(rec, rec["turns"], self.repo)
        self.assertTrue(consulted.startswith("turn 1: 35 calls (Read 30, Bash 5), 5 failed: "
                                             "Bash pytest attempt-0 (no result)"), consulted)

    def test_a_long_command_is_scrubbed_before_it_is_cut(self):
        """Cutting first left half a path the scrubber no longer recognised."""
        deep = self.repo / ("d" * 60) / "tests"
        self.fire("UserPromptSubmit", prompt="run the tests", prompt_id="t1")
        self.fire("PreToolUse", prompt_id="t1", tool_name="Bash", tool_use_id="b1",
                  tool_input={"command": f"cd {self.repo} && ls {deep}"})
        (self.repo / "src" / "stats.py").write_text("x = 9\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="listed")
        rec = record.load(self.checkout, self.session)
        rec["extensions"]["tool_events"]["turns"]["t1"]["events"][0]["detail"]["command"] = (
            f"cd {self.repo} && ls {deep}")
        consulted = evidence._consulted(rec, rec["turns"], self.repo)
        self.assertIn("Bash cd <repo> && ls <repo>/", consulted)
        self.assertNotIn(str(self.tmp.resolve())[:10], consulted)
        self.assertNotIn(str(self.tmp)[:10], consulted)


class ConsentCanSeeWhatItApproves(Fixture):
    """Regression: under `interactive`, --yes approved a send nobody could see."""

    def args(self, **kw):
        values = dict(path=str(self.repo), session_id=self.session, candidate=None,
                      rule=None, turn=2, yes=False, show=False, model=None, timeout=5)
        values.update(kw)
        return argparse.Namespace(**values)

    def test_show_prints_the_exact_prompt_and_sends_nothing(self):
        self.capture_corrected_session()
        engine = FakeEngine()
        expected = diagnosis.prepare(self.repo, self.checkout, self.session, None, 2).prompt
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.cmd_diagnose(self.args(show=True))
        self.assertEqual(0, code, err.getvalue())
        self.assertEqual(expected, out.getvalue())
        self.assertIn("Nothing was sent", err.getvalue())
        self.assertEqual([], diagnosis.load_all(self.checkout))
        diagnosis.run(self.repo, self.checkout, self.session, engine, corrective_turn=2)
        self.assertEqual(expected, engine.prompt, "the preview is not what gets sent")

    def test_the_consent_message_offers_the_preview(self):
        self.capture_corrected_session()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.cmd_diagnose(self.args())
        self.assertEqual(3, code)
        self.assertIn("--show", err.getvalue())


class EvidenceDiversity(Fixture):
    """Regression: candidates were taken in kind order, so a session with eight
    prompts filled every slot and starved the evidence of agent behaviour. That
    surfaced as a clean INSUFFICIENT_EVIDENCE rather than the ranking bug it was.
    """

    def _session(self, turns: int, correction_at: int) -> dict:
        built = []
        for i in range(1, turns + 1):
            text = "No! Stats must return dataclasses." if i == correction_at else f"request {i}"
            built.append({
                "index": i, "logical_turn_id": f"t{i}",
                "workspace_changed_before_turn": i == correction_at,
                "prompt_events": [{"ordinal": 1, "content": {"text": text}, "snapshot": None}],
                "stop_events": [{"ordinal": 1, "snapshot": None,
                                 "last_assistant_message": {"text": f"done {i}"}}]})
        return {"turns": built, "labels": {"developer_label": None},
                "extensions": {"tool_events": {"turns": {f"t{turns}": {"count": 1, "events": [
                    {"seq": 1, "tool": "Bash", "detail": {"command": "pytest -q"},
                     "outcome_observed": False}]}}}}}

    def test_a_long_session_is_still_diagnosable(self):
        for turns in (2, 8, 12, 20, 40):
            selection = evidence.select(self._session(turns, turns), self.repo)
            self.assertTrue(evidence.is_sufficient(selection),
                            f"{turns}-turn session became undiagnosable")

    def test_every_kind_present_gets_a_slot_before_any_kind_repeats(self):
        selection = evidence.select(self._session(20, 20), self.repo)
        kinds = [i.kind for i in selection.items]
        self.assertIn("tool_failure", kinds)
        self.assertIn("assistant_message", kinds)
        self.assertIn("turn_prompt", kinds)

    def test_the_correction_survives_a_long_session(self):
        selection = evidence.select(self._session(20, 20), self.repo)
        self.assertTrue(any("dataclasses" in i.text for i in selection.items))

    def test_recent_turns_are_preferred(self):
        selection = evidence.select(self._session(20, 20), self.repo)
        refs = [i.ref for i in selection.items if i.kind == "turn_prompt"]
        self.assertIn("turn/20/prompt/1", refs)
        self.assertNotIn("turn/1/prompt/1", refs)

    def test_a_diff_is_only_built_when_it_is_chosen(self):
        """Building a diff costs a git subprocess, so unchosen turns must not
        pay for one."""
        self.capture_corrected_session()
        rec = record.load(self.checkout, self.session)
        built = []
        original = evidence._turn_diff

        def counted(root, a, b):
            built.append(b)
            return original(root, a, b)

        evidence._turn_diff = counted
        try:
            selection = evidence.select(rec, self.repo, max_items=1)
        finally:
            evidence._turn_diff = original
        chosen = [i for i in selection.items if i.kind == "diff"]
        self.assertLessEqual(len(built), len(chosen) + 1,
                             "diffs were built for turns that were never sent")


class TheBudgetSaysWhatItCounts(Fixture):
    """Regression: the ceiling was enforced in bytes, named in tokens, and
    printed as tokens, so a reader expected roughly four times what was sent."""

    def test_the_ceiling_is_bytes(self):
        self.assertEqual(len("é" * 100), 100)
        self.assertEqual(200, evidence.measure_bytes("é" * 100))
        self.assertEqual(reasoning.MAX_INPUT_BYTES, evidence.MAX_BYTES)

    def test_the_reported_budget_is_the_enforced_one(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine())
        budget = result["budget"]
        self.assertEqual(evidence.MAX_BYTES, budget["max_tokens"])
        self.assertLessEqual(budget["estimated_tokens"], budget["max_tokens"])

    def test_the_engine_refuses_a_payload_over_the_ceiling(self):
        engine = reasoning.ClaudeCliEngine()
        with self.assertRaises(reasoning.Unavailable) as caught:
            engine.run("x" * (reasoning.MAX_INPUT_BYTES + 1))
        self.assertIn("bytes", str(caught.exception))

    def test_nothing_still_calls_the_budget_tokens(self):
        """The unit is the whole finding; a leftover name reintroduces it."""
        for module in (evidence, reasoning, diagnosis):
            leftovers = [n for n in dir(module)
                         if "TOKEN" in n.upper() and not n.startswith("__")]
            self.assertEqual([], leftovers, module.__name__)


class TheCorrectionIsNotAlwaysLast(Fixture):
    """Regression: selection ranked turns by recency, so a correction more than
    two turns from the end was dropped entirely — including when the developer
    had named it. `EvidenceDiversity` missed it because its fixture sets
    `workspace_changed_before_turn` on the correction; real capture sets that
    only when the developer edits files by hand between turns, never for a
    conversational correction.
    """

    def _session(self, turns: int, correction_at: int) -> dict:
        built = []
        for i in range(1, turns + 1):
            text = ("No! Stats must return dataclasses." if i == correction_at
                    else f"request {i}")
            built.append({
                "index": i, "logical_turn_id": f"t{i}",
                "workspace_changed_before_turn": False,
                "prompt_events": [{"ordinal": 1, "content": {"text": text},
                                   "snapshot": None}],
                "stop_events": [{"ordinal": 1, "snapshot": None,
                                 "last_assistant_message": {"text": f"done {i}"}}]})
        return {"turns": built, "labels": {"developer_label": None},
                "extensions": {}}

    def test_a_named_correction_reaches_the_model_from_anywhere(self):
        for turns, at in ((5, 2), (12, 3), (20, 2), (40, 7)):
            selection = evidence.select(self._session(turns, at), self.repo,
                                        corrective_turn=at)
            refs = [i.ref for i in selection.items]
            self.assertTrue(any(f"turn/{at}/" in r for r in refs),
                            f"turn {at} of {turns} was not sent: {refs}")
            self.assertTrue(any("dataclasses" in i.text for i in selection.items),
                            f"the correction text was dropped ({turns} turns)")

    def test_the_turn_that_was_corrected_is_sent_too(self):
        """Authentic validation needs the problematic state, not only its repair."""
        selection = evidence.select(self._session(20, 6), self.repo,
                                    corrective_turn=6)
        refs = [i.ref for i in selection.items]
        self.assertTrue(any("turn/5/" in r for r in refs), refs)

    def test_a_named_turn_is_labelled_a_correction_not_a_prompt(self):
        selection = evidence.select(self._session(9, 4), self.repo,
                                    corrective_turn=4)
        named = [i for i in selection.items if "turn/4/prompt" in i.ref]
        self.assertEqual(["correction"], [i.kind for i in named])

    def test_without_a_named_turn_an_early_correction_is_still_missed(self):
        """The known limit this phase does not close: unlabelled sessions fall
        back to recency, so `--turn` is what makes an early correction visible."""
        selection = evidence.select(self._session(20, 2), self.repo)
        self.assertFalse(any("turn/2/" in i.ref for i in selection.items))

    def test_diagnosis_passes_the_boundary_into_selection(self):
        self.capture_corrected_session()
        engine = FakeEngine()
        diagnosis.run(self.repo, self.checkout, self.session, engine,
                      corrective_turn=1)
        self.assertIn("[1] correction (turn/1/prompt/1)", engine.prompt)

    def test_a_boundary_naming_no_turn_is_refused(self):
        self.capture_corrected_session()
        with self.assertRaises(ValueError):
            diagnosis.run(self.repo, self.checkout, self.session, FakeEngine(),
                          corrective_turn=99)


class ABoundaryOutlivesAFailedJob(Fixture):
    """Regression: a developer-supplied corrective turn was written to every
    record, but the schema forbids it on a non-SUCCESS outcome — so marking a
    rule with `--turn` and then meeting an unreachable model crashed on save."""

    def test_an_unavailable_model_with_a_named_turn_still_saves(self):
        self.capture_corrected_session()
        result = diagnosis.run(
            self.repo, self.checkout, self.session,
            FakeEngine(raises=reasoning.Unavailable("no model")),
            corrective_turn=1)
        self.assertEqual(reasoning.MODEL_UNAVAILABLE, result["outcome"])
        self.assertIsNone(result["corrective_turn"])
        if VALIDATOR:
            VALIDATOR.validate(result)

    def test_a_successful_diagnosis_keeps_the_developers_turn(self):
        self.capture_corrected_session()
        result = diagnosis.run(self.repo, self.checkout, self.session,
                               FakeEngine(), corrective_turn=1)
        self.assertEqual(reasoning.SUCCESS, result["outcome"])
        self.assertEqual(1, result["corrective_turn"])


class PayloadIsNotOnTheCommandLine(unittest.TestCase):
    """ARCH §12 — a command line is readable by every user on the machine, which
    would undo the whole minimization pipeline.

    Measured at the process boundary rather than read out of the source: the
    property is where the bytes actually go, not how the call is spelled."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-argv-"))
        binaries = self.tmp / "bin"
        binaries.mkdir()
        (binaries / "claude").write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "2.1.267 (Claude Code)"; exit 0; fi\n'
            f'printf "%s\\n" "$@" > {self.tmp}/argv.txt\n'
            f"cat > {self.tmp}/stdin.txt\n"
            'echo \'{"result": "{}", "modelUsage": {"m": 1}}\'\n')
        (binaries / "claude").chmod(0o755)
        self._path = os.environ["PATH"]
        os.environ["PATH"] = f"{binaries}:{self._path}"
        os.environ["REPOHONE_DATA_DIR"] = str(self.tmp / "data")

    def tearDown(self):
        os.environ["PATH"] = self._path
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_prompt_never_reaches_argv(self):
        secret = "a-very-distinctive-payload-marker"
        reasoning.ClaudeCliEngine(timeout=20).run(f"diagnose this: {secret}")
        argv = (self.tmp / "argv.txt").read_text()
        self.assertNotIn(secret, argv, "the prompt was visible in the process table")
        self.assertIn(secret, (self.tmp / "stdin.txt").read_text())

    def test_the_cli_uses_fail_closed_isolation_controls(self):
        reasoning.ClaudeCliEngine(timeout=20).run("small prompt")
        argv = (self.tmp / "argv.txt").read_text().splitlines()
        for flag in ("--safe-mode", "--restricted", "--strict-mcp-config",
                     "--permission-prompts", "--no-session-persistence",
                     "--disable-slash-commands", "--no-chrome"):
            self.assertIn(flag, argv)
        self.assertNotIn("--disallowedTools", argv)
        tools = argv.index("--tools")
        self.assertEqual("", argv[tools + 1], "the tool set was not empty")

    def test_a_permission_denial_invalidates_the_reply(self):
        binary = self.tmp / "bin" / "claude"
        binary.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "2.1.267 (Claude Code)"; exit 0; fi\n'
            "cat >/dev/null\n"
            "echo '{\"result\": \"{}\", \"permission_denials\": [\"tool\"]}'\n")
        binary.chmod(0o755)
        with self.assertRaises(reasoning.Unavailable):
            reasoning.ClaudeCliEngine(timeout=20).run("small prompt")

    def test_non_ascii_input_cannot_bypass_the_hard_budget(self):
        with self.assertRaises(reasoning.Unavailable):
            reasoning.ClaudeCliEngine(timeout=20).run(
                "🙂" * (reasoning.MAX_INPUT_BYTES // 4 + 1))
        self.assertFalse((self.tmp / "argv.txt").exists(),
                         "the provider was invoked for an over-budget prompt")

    def test_an_old_cli_fails_before_any_model_request(self):
        binary = self.tmp / "bin" / "claude"
        invoked = self.tmp / "model-invoked"
        binary.write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "2.1.266 (Claude Code)"; exit 0; fi\n'
            f"touch {invoked}\n")
        binary.chmod(0o755)
        with self.assertRaises(reasoning.Unavailable):
            reasoning.ClaudeCliEngine(timeout=20).run("small prompt")
        self.assertFalse(invoked.exists())


class ToolIsolation(unittest.TestCase):
    """ARCH §12.1 — isolation covers future and user-provided capabilities."""

    def test_isolation_is_an_allow_none_policy_not_a_tool_name_list(self):
        self.assertIn("--safe-mode", reasoning.ISOLATION_ARGS)
        self.assertIn("--strict-mcp-config", reasoning.ISOLATION_ARGS)
        position = reasoning.ISOLATION_ARGS.index("--tools")
        self.assertEqual("", reasoning.ISOLATION_ARGS[position + 1])


class NonSuccessKeepsItsReasoning(Fixture):
    """'Do nothing' is the answer that prevents manufactured proposals, so the
    argument for it must survive (LEARNING_PLAN §11)."""

    def setUp(self):
        super().setUp()
        self.capture_corrected_session()

    def test_no_actionable_improvement_keeps_its_evidence(self):
        reply = dict(GOOD_REPLY, outcome="NO_ACTIONABLE_PROJECT_IMPROVEMENT",
                     root_cause=None, confidence=None, required_property=None,
                     fingerprint=None,
                     evidence_for=["the change was a new requirement"],
                     evidence_against=["the agent had no way to know"],
                     risks=["treating scope growth as a defect"])
        result = self.diagnose(FakeEngine(reply))
        self.assertEqual(["the change was a new requirement"], result["evidence_for"])
        self.assertEqual(["the agent had no way to know"], result["evidence_against"])
        self.assertEqual(["treating scope growth as a defect"], result["risks"])
        self.assert_conformant(result)

    def test_it_still_leaves_no_partial_diagnosis(self):
        reply = dict(GOOD_REPLY, outcome="NO_ACTIONABLE_PROJECT_IMPROVEMENT",
                     root_cause=None, confidence=None, required_property=None,
                     fingerprint=None)
        result = self.diagnose(FakeEngine(reply))
        self.assertIsNone(result["root_cause"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["required_property"])


class DiagnosisIdentity(Fixture):
    """Regression: an id derived from session plus millisecond collided, and a
    collision silently overwrote the earlier record."""

    def test_two_diagnoses_of_one_session_do_not_collide(self):
        self.capture_corrected_session()
        first = self.diagnose(FakeEngine())
        second = self.diagnose(FakeEngine())
        self.assertNotEqual(first["diagnosis_id"], second["diagnosis_id"])
        self.assertEqual(2, len(diagnosis.load_all(self.checkout)))

    def test_ids_are_unique_within_one_millisecond(self):
        made = {diagnosis.new_id("rh_abcd1234", "2026-09-20T12:00:00.123Z")
                for _ in range(200)}
        self.assertEqual(200, len(made))


class UnsupportedSessionVersion(Fixture):
    """ARCH §22 — a reader refuses unknown semantics; the CLI reports it."""

    def test_the_cli_refuses_without_a_traceback(self):
        from repohone import paths
        directory = paths.records_dir(self.checkout)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rh_future1234.json").write_text(json.dumps(
            {"schema": "repohone.session/v1", "schema_version": 1,
             "contract_version": "9.9", "turns": []}))
        proc = subprocess.run(
            [sys.executable, str(HOOK), "diagnose", "rh_future1234",
             "--path", str(self.repo), "--yes"],
            capture_output=True, text=True, env=dict(os.environ))
        self.assertEqual(1, proc.returncode)
        self.assertIn("cannot diagnose", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)


class RuleStatement(Fixture):
    """ARCH §23 — an explicit rule statement alone is sufficient input."""

    def test_a_rule_statement_is_ranked_first(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine(), rule="all stats functions return dataclasses")
        self.assertEqual("rule_statement", result["evidence_refs"][0]["kind"])


class EngineIsolation(Fixture):
    """ARCH §12.1 — reasoning must not run inside the developer's context or repo."""

    def test_the_cli_engine_runs_outside_the_repository(self):
        engine = reasoning.ClaudeCliEngine()
        cwd = engine._isolated_cwd().resolve()
        self.assertFalse(str(cwd).startswith(str(self.repo.resolve())))

    def test_the_cli_engine_disables_every_tool(self):
        position = reasoning.ISOLATION_ARGS.index("--tools")
        self.assertEqual("", reasoning.ISOLATION_ARGS[position + 1])
        self.assertIn("--safe-mode", reasoning.ISOLATION_ARGS)


class Fingerprints(unittest.TestCase):
    """LEARNING_PLAN §9 — which recurring problem, not what kind of problem."""

    def test_wording_differences_normalise_away(self):
        self.assertEqual(fingerprint.normalize("handler-direct-database-access"),
                         fingerprint.normalize("Handler direct database access"))
        self.assertEqual(fingerprint.normalize("handler_direct_database_access"),
                         fingerprint.normalize("handler-direct-database-access"))

    def test_unrelated_problems_do_not_match(self):
        existing = [fingerprint.Fingerprint("fp_1", "MISSING_GUARDRAIL",
                                            "persistence-boundary",
                                            "handler-direct-database-access")]
        self.assertIsNone(fingerprint.match(existing, "MISSING_GUARDRAIL",
                                            "input-validation", "missing-length-check"))

    def test_a_reworded_label_joins_the_existing_series(self):
        existing = [fingerprint.Fingerprint("fp_1", "MISSING_GUARDRAIL",
                                            "persistence boundary",
                                            "handler direct database access")]
        hit = fingerprint.match(existing, "MISSING_GUARDRAIL",
                                "persistence-boundary", "handler-direct-database-access")
        self.assertIsNotNone(hit)
        self.assertEqual("fp_1", hit.fingerprint_id)

    def test_a_different_class_is_a_different_problem(self):
        existing = [fingerprint.Fingerprint("fp_1", "MISSING_GUARDRAIL",
                                            "persistence-boundary", "handler-direct-db")]
        self.assertIsNone(fingerprint.match(existing, "TOOLING_FRICTION",
                                            "persistence-boundary", "handler-direct-db"))


class MatchingPrecision(unittest.TestCase):
    """Regression: averaging area and behavior let a shared area carry a behavior
    that meant something else, merging four distinct problems into one series.
    An over-merged series invents a pattern that does not exist."""

    EXISTING = [None]  # replaced in setUp

    def setUp(self):
        self.existing = [fingerprint.Fingerprint(
            "fp_1", "MISSING_GUARDRAIL", "input-validation", "missing-length-check")]

    def _match(self, area, behavior, cls="MISSING_GUARDRAIL"):
        return fingerprint.match(self.existing, cls, area, behavior)

    def test_distinct_behaviours_in_one_area_stay_distinct(self):
        for behavior in ("missing-null-check", "missing-range-check",
                         "missing-encoding-check", "missing-type-check"):
            self.assertIsNone(self._match("input-validation", behavior),
                              f"{behavior} was merged into missing-length-check")

    def test_the_same_behaviour_still_matches(self):
        self.assertIsNotNone(self._match("input-validation", "missing-length-check"))

    def test_an_abbreviation_is_the_same_token_as_its_expansion(self):
        existing = [fingerprint.Fingerprint("fp_2", "MISSING_GUARDRAIL",
                                            "persistence boundary",
                                            "handler direct database access")]
        for behavior in ("handler-direct-db-access", "handler-direct-database-access",
                         "handlers-direct-database-access"):
            self.assertIsNotNone(
                fingerprint.match(existing, "MISSING_GUARDRAIL",
                                  "persistence-boundary", behavior), behavior)

    def test_unrelated_tokens_are_not_equivalent(self):
        for a, b in (("length", "null"), ("range", "random"), ("import", "export"),
                     ("null", "number")):
            self.assertFalse(fingerprint._equivalent(a, b), f"{a} ~ {b}")

    def test_morphological_variants_are_equivalent(self):
        for a, b in (("validate", "validation"), ("check", "checks"),
                     ("handler", "handlers"), ("db", "database")):
            self.assertTrue(fingerprint._equivalent(a, b), f"{a} !~ {b}")


class NegationIsMeaning(unittest.TestCase):
    """Regression: 'not' was a stop word, so a rule and its exact negation
    normalised to the same fingerprint."""

    def test_a_rule_and_its_negation_are_different(self):
        for positive, negative in (
                ("handler-must-import-persistence", "handler-must-not-import-persistence"),
                ("returns-typed-result", "returns-not-typed-result"),
                ("is-validated", "is-not-validated")):
            self.assertNotEqual(fingerprint.normalize(positive),
                                fingerprint.normalize(negative))

    def test_a_negated_behaviour_does_not_match_its_opposite(self):
        existing = [fingerprint.Fingerprint("fp_1", "MISSING_GUARDRAIL",
                                            "persistence", "handler-imports-database")]
        self.assertIsNone(fingerprint.match(
            existing, "MISSING_GUARDRAIL", "persistence", "handler-not-imports-database"))


class ProvenanceIsATable(unittest.TestCase):
    """The verdict depends only on which record kinds are present, so every
    combination is enumerated, in every order — not a list of examples."""

    TARGET = identity.repository_id("git@github.com:Acme/App.git")
    OTHER = identity.repository_id("git@github.com:Other/Thing.git")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["REPOHONE_DATA_DIR"] = self.tmp

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def expected(kinds):
        """The specification, written once: True wins; then anything unreadable
        or unidentified makes it undecidable; only unanimous 'other' excludes."""
        if "target" in kinds:
            return True, False
        if "invalid" in kinds:
            return None, True
        if "null" in kinds or not kinds:
            return None, False
        return False, False

    def _checkout(self, name, order):
        """Records whose sorted filenames follow `order` exactly."""
        hosts = {identity.session_id("fixture", f"{name}-{i}"): f"{name}-{i}"
                 for i in range(len(order))}
        for session_id, kind in zip(sorted(hosts), order, strict=True):
            if kind == "invalid":
                target = paths.records_dir(name) / f"{session_id}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{ not json")
                continue
            repository = {"target": self.TARGET, "other": self.OTHER, "null": None}[kind]
            ensure_session(name, hosts[session_id], repository, "dev")

    def test_every_combination_in_every_order(self):
        kinds = ("target", "other", "null", "invalid")
        cases = 0
        for size in range(1, len(kinds) + 1):
            for chosen in itertools.combinations(kinds, size):
                for order in itertools.permutations(chosen):
                    name = "c" + "".join(k[0] for k in order)
                    self._checkout(name, order)
                    verdict, problems = fingerprint._is_clone_of(name, self.TARGET)
                    want_verdict, want_problems = self.expected(set(order))
                    with self.subTest(order=order):
                        self.assertIs(want_verdict, verdict)
                        self.assertEqual(want_problems, bool(problems))
                    cases += 1
        self.assertEqual(64, cases)

    def test_a_checkout_with_no_records_is_undecidable(self):
        paths.records_dir("empty").mkdir(parents=True)
        self.assertEqual((None, []), fingerprint._is_clone_of("empty", self.TARGET))
        self.assertEqual((None, []), fingerprint._is_clone_of("absent", self.TARGET))

    def test_a_records_directory_that_cannot_be_listed_is_reported(self):
        records = paths.records_dir("locked")
        records.mkdir(parents=True)
        os.chmod(records, 0)
        try:
            verdict, problems = fingerprint._is_clone_of("locked", self.TARGET)
        finally:
            os.chmod(records, 0o755)
        self.assertIsNone(verdict)
        self.assertTrue(problems)

    def test_a_possibly_related_checkout_carries_its_corruption_into_the_scan(self):
        """The reviewer's case end to end: null beside other is undecidable, so
        the sibling's corrupt diagnosis must make recurrence incomplete."""
        home = ensure_session("home", "h1", self.TARGET, "dev")
        canonical_occurrence("home", self.TARGET, "MISSING_CONTEXT", "reporting",
                             "untyped-return", home, developer_id="dev")
        self._checkout("sibling", ("other", "null"))
        broken = paths.checkout_dir("sibling") / "diagnoses"
        broken.mkdir(parents=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")
        self.assertTrue(fingerprint.known("home", self.TARGET).problems)


class OnlyTheDiagnosisFilesCount(unittest.TestCase):
    """A sibling's diagnosis file against every state of its state database.

    The files are the whole record: a deleted diagnosis stops counting, one that
    cannot be read makes the scan incomplete, and the database never matters.
    """

    TARGET = identity.repository_id("git@github.com:Acme/App.git")

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["REPOHONE_DATA_DIR"] = self.tmp
        self.locked = []

    def tearDown(self):
        for path in self.locked:
            os.chmod(path, 0o755)
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _occurrence(self, checkout, host):
        session = ensure_session(checkout, host, self.TARGET, "dev")
        canonical_occurrence(checkout, self.TARGET, "MISSING_CONTEXT", "reporting",
                             "untyped-return", session, developer_id="dev")

    def _derived(self, state_name, db):
        if state_name == "present":
            return
        db.unlink()
        if state_name == "corrupt":
            db.write_bytes(b"this is not a database" * 64)
        elif state_name == "inaccessible":
            db.write_bytes(b"")
            os.chmod(db, 0)
            self.locked.append(db)
        elif state_name == "wrong-type":
            db.mkdir()

    def _canonical(self, state_name, checkout):
        directory = paths.checkout_dir(checkout) / "diagnoses"
        artifact = next(directory.glob("dx_*.json"))
        if state_name == "missing":
            artifact.unlink()
        elif state_name == "corrupt":
            artifact.write_text("{ not json")
        elif state_name == "unreadable":
            os.chmod(directory, 0)
            self.locked.append(directory)

    # canonical -> (complete, sibling counted), whatever state the database is in
    EXPECTED = {"present": (True, True), "missing": (True, False),
                "corrupt": (False, False), "unreadable": (False, False)}
    DATABASE = ("present", "missing", "corrupt", "inaccessible", "wrong-type")

    def test_every_canonical_and_database_state(self):
        for canonical, (complete, counted) in self.EXPECTED.items():
            for derived in self.DATABASE:
                with self.subTest(canonical=canonical, database=derived):
                    self.tearDown()
                    self.setUp()
                    self._occurrence("home", "h1")
                    self._occurrence("sibling", "s1")
                    self._canonical(canonical, "sibling")
                    self._derived(derived, paths.state_db("sibling"))

                    scan = fingerprint.known("home", self.TARGET)
                    self.assertEqual(complete, not scan.problems, list(scan.problems))
                    sessions = scan[0].sessions if len(scan) else 0
                    self.assertEqual(2 if counted else 1, sessions)

    def test_reading_recurrence_writes_no_database(self):
        self._occurrence("home", "h1")
        self._occurrence("sibling", "s1")
        paths.state_db("sibling").unlink()
        mark = fingerprint.known("home", self.TARGET)[0]
        fingerprint.occurrences_for("home", mark.fingerprint_id)
        self.assertFalse(paths.state_db("sibling").exists())

    def test_occurrences_for_reads_the_same_table(self):
        """The two recurrence queries read the same files, so the table above
        holds for Phase 5's query too, not only Phase 3's."""
        self._occurrence("home", "h1")
        self._occurrence("sibling", "s1")
        mark = fingerprint.known("home", self.TARGET)[0]
        paths.state_db("sibling").unlink()
        self.assertEqual(2, len(fingerprint.occurrences_for("home", mark.fingerprint_id)))

    def test_an_unreadable_own_diagnosis_is_not_an_unknown_pattern(self):
        """Regression: with its own diagnosis corrupt and its index lost,
        `occurrences_for` found no row and returned complete and empty — and
        `propose` would have validated against nothing while claiming coverage."""
        self._occurrence("home", "h1")
        mark = fingerprint.known("home", self.TARGET)[0]
        next((paths.checkout_dir("home") / "diagnoses").glob("dx_*.json")) \
            .write_text("{ not json")
        paths.state_db("home").unlink()
        scan = fingerprint.occurrences_for("home", mark.fingerprint_id)
        with self.assertRaises(fingerprint.IncompleteHistory):
            scan.require_complete()


class RecurrenceScope(unittest.TestCase):
    """ARCH §15 — telemetry is checkout-scoped, but analysis may group by
    repository identity. One developer's clones and worktrees are one project."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["REPOHONE_DATA_DIR"] = self.tmp
        self.repo_id = identity.repository_id("git@github.com:Acme/App.git")

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seen(self, checkout, developer, session, repository_id="DEFAULT"):
        repository_id = self.repo_id if repository_id == "DEFAULT" else repository_id
        session_id = ensure_session(checkout, session, repository_id, developer)
        canonical_occurrence(
            checkout, repository_id, "MISSING_CONTEXT", "reporting",
            "untyped-return", session_id, developer_id=developer)
        return session_id

    def test_recurrence_aggregates_across_local_checkouts(self):
        self._seen("alice", "dev_alice", "rh_a1")
        self._seen("bob", "dev_bob", "rh_b1")
        self._seen("alice", "dev_alice", "rh_a2")
        for checkout in ("alice", "bob"):
            mark = fingerprint.known(checkout, self.repo_id)[0]
            self.assertEqual(3, mark.sessions)
            self.assertEqual(2, mark.checkouts)
            self.assertEqual(2, mark.developers,
                             "developer diversity is what separates a habit from a pattern")

    def test_another_repository_is_a_different_series(self):
        self._seen("alice", "dev_alice", "rh_a1")
        other = identity.repository_id("git@github.com:Acme/Other.git")
        session = ensure_session("alice", "other", other, "dev_alice")
        canonical_occurrence("alice", other, "MISSING_CONTEXT", "x", "y",
                             session, developer_id="dev_alice")
        self.assertEqual(["reporting / untyped-return"],
                         [m.label() for m in fingerprint.known("alice", self.repo_id)])
        self.assertEqual(["x / y"], [m.label() for m in fingerprint.known("alice", other)])

    def test_two_classes_with_the_same_labels_are_two_patterns(self):
        """Regression: `match` treats a different root-cause class as a
        different problem, but the id hashed only repository, area and
        behavior — so the second diagnosis contradicted the first index row
        and every later scan in the checkout came back incomplete."""
        first = ensure_session("alice", "rh_a1", self.repo_id, "dev_alice")
        second = ensure_session("alice", "rh_a2", self.repo_id, "dev_alice")
        a = canonical_occurrence("alice", self.repo_id, "MISSING_CONTEXT",
                                 "reporting", "untyped-return", first,
                                 developer_id="dev_alice")[0]
        b = canonical_occurrence("alice", self.repo_id, "MISSING_GUARDRAIL",
                                 "reporting", "untyped-return", second,
                                 developer_id="dev_alice")[0]
        self.assertNotEqual(a.fingerprint_id, b.fingerprint_id)
        scan = fingerprint.known("alice", self.repo_id)
        self.assertEqual([], list(scan.problems))
        self.assertEqual({"MISSING_CONTEXT", "MISSING_GUARDRAIL"},
                         {mark.cls for mark in scan})

    def test_the_id_covers_every_field_matching_distinguishes(self):
        base = ("MISSING_CONTEXT", "reporting", "untyped-return")
        ids = {fingerprint.new_id(self.repo_id, *base)}
        for changed in (("BAD_GUARDRAIL", "reporting", "untyped-return"),
                        ("MISSING_CONTEXT", "persistence", "untyped-return"),
                        ("MISSING_CONTEXT", "reporting", "raw-sql-in-handler")):
            ids.add(fingerprint.new_id(self.repo_id, *changed))
        self.assertEqual(4, len(ids), "two distinguishable patterns share an id")

    def test_an_invalid_sibling_record_cannot_prove_it_is_unrelated(self):
        """Regression: provenance was read as plain JSON, so a parseable but
        schema-invalid record could claim another repository and have its
        checkout silently dropped from a scan that still reported complete."""
        self._seen("home", "dev_alice", "rh_h1")
        self._seen("laptop", "dev_alice", "rh_l1")
        self.assertEqual(2, fingerprint.known("home", self.repo_id)[0].sessions)

        target = next(paths.records_dir("laptop").glob("rh_*.json"))
        forged = json.loads(target.read_text())
        forged["repository"]["id"] = identity.repository_id(
            "git@github.com:Other/Thing.git")
        forged.pop("schema_version", None)
        target.write_text(json.dumps(forged))

        scan = fingerprint.known("home", self.repo_id)
        self.assertTrue(scan.problems, "a forged sibling was silently excluded")
        with self.assertRaises(fingerprint.IncompleteHistory):
            scan.require_complete()

    def test_one_unreadable_record_outweighs_a_valid_one_pointing_elsewhere(self):
        """A checkout is one clone, so a record naming another repository
        normally settles it — but not while a sibling record cannot be read at
        all. Undecidable wins, because exclusion is the unrecoverable answer."""
        self._seen("home", "dev_alice", "rh_h1")
        other = identity.repository_id("git@github.com:Other/Thing.git")
        ensure_session("laptop", "rh_l1", other, "dev_x")
        unreadable = paths.records_dir("laptop") / "rh_00000000000000ff.json"
        unreadable.write_text("{ not json")

        verdict, problems = fingerprint._is_clone_of("laptop", self.repo_id)
        self.assertIsNone(verdict, "an unreadable record was treated as settled")
        self.assertTrue(problems)
        self.assertTrue(fingerprint.known("home", self.repo_id).problems)

    def test_a_valid_sibling_of_another_repository_is_still_excluded(self):
        """Scoping must not collapse into "include everything"."""
        self._seen("home", "dev_alice", "rh_h1")
        other = identity.repository_id("git@github.com:Other/Thing.git")
        ensure_session("stranger", "rh_s1", other, "dev_x")
        broken = Path(self.tmp) / "checkouts" / "stranger" / "diagnoses"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")
        self.assertEqual([], list(fingerprint.known("home", self.repo_id).problems))

    def test_every_recurrence_query_scopes_the_same_way(self):
        """Regression: `known` filtered by repository and `occurrences_for` did
        not, so unrelated corruption blocked Phase 5 but not Phase 3."""
        self._seen("mine", "dev_alice", "rh_m1")
        mark = fingerprint.known("mine", self.repo_id)[0]
        other = identity.repository_id("git@github.com:Other/Client.git")
        ensure_session("stranger", "rh_s1", other, "dev_x")
        broken = Path(self.tmp) / "checkouts" / "stranger" / "diagnoses"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")

        self.assertEqual([], list(fingerprint.known("mine", self.repo_id).problems))
        self.assertEqual([], list(fingerprint.occurrences_for(
            "mine", mark.fingerprint_id).problems))
        self.assertEqual(1, len(fingerprint.sessions_for("mine", mark.fingerprint_id)))

    def test_an_unrelated_project_cannot_block_this_one(self):
        """Regression: every checkout on the machine was read before the
        repository filter, so one corrupt artifact in an unrelated project made
        every diagnosis here INSUFFICIENT_EVIDENCE."""
        self._seen("alice", "dev_alice", "rh_a1")
        other = identity.repository_id("git@github.com:Other/Client.git")
        ensure_session("stranger", "s1", other, "dev_x")
        broken = Path(self.tmp) / "checkouts" / "stranger" / "diagnoses"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")

        scan = fingerprint.known("alice", self.repo_id)
        self.assertEqual([], list(scan.problems))
        self.assertEqual(1, scan[0].sessions)

    def test_an_unrelated_projects_identifier_never_enters_this_record(self):
        self._seen("alice", "dev_alice", "rh_a1")
        ensure_session("stranger", "s1",
                       identity.repository_id("git@github.com:Other/Client.git"), "dev_x")
        broken = Path(self.tmp) / "checkouts" / "stranger" / "diagnoses"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")
        self.assertNotIn("stranger",
                         " ".join(fingerprint.known("alice", self.repo_id).problems))

    def test_a_sibling_checkout_of_this_repository_still_fails_closed(self):
        """Scoping the scan must not become a way to ignore real corruption."""
        self._seen("alice", "dev_alice", "rh_a1")
        self._seen("bob", "dev_bob", "rh_b1")
        broken = Path(self.tmp) / "checkouts" / "bob" / "diagnoses"
        (broken / "dx_1111111111111111.json").write_text("{ not json")
        self.assertTrue(fingerprint.known("alice", self.repo_id).problems)

    def test_a_checkout_with_no_records_is_not_assumed_unrelated(self):
        self._seen("alice", "dev_alice", "rh_a1")
        stranger = Path(self.tmp) / "checkouts" / "stranger"
        (stranger / "diagnoses").mkdir(parents=True, exist_ok=True)
        (stranger / "diagnoses" / "dx_0000000000000000.json").write_text("{ not json")
        state.initialize("stranger")
        self.assertTrue(fingerprint.known("alice", self.repo_id).problems)

    def test_a_checkout_with_no_remote_is_not_assumed_unrelated(self):
        """A local-only clone carries no repository id, so it cannot be ruled
        out — only a *different* id is grounds for skipping a checkout."""
        self._seen("alice", "dev_alice", "rh_a1")
        ensure_session("stranger", "s1", None, "dev_x")
        broken = Path(self.tmp) / "checkouts" / "stranger" / "diagnoses"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")
        self.assertTrue(fingerprint.known("alice", self.repo_id).problems)

    def test_a_repository_without_a_remote_sees_only_its_own(self):
        """Regression: the predicate degenerated to 'match everything'."""
        self._seen("alice", "dev_alice", "rh_a1")
        self.assertEqual([], fingerprint.known("alice", None))

    def test_two_repositories_without_remotes_never_share_a_series(self):
        for checkout, session in (("local-a", "rh_a"), ("local-b", "rh_b")):
            self._seen(checkout, "dev", session, repository_id=None)
        mark = fingerprint.known("local-a", None)[0]
        self.assertEqual(1, mark.sessions)
        self.assertEqual(1, mark.checkouts)
        expected = identity.session_id("fixture", "rh_a")
        self.assertEqual([("local-a", expected)],
                         fingerprint.occurrences_for("local-a", mark.fingerprint_id))

    def test_a_checkout_from_an_older_core_does_not_crash(self):
        """Regression: known() queried tables an older database never had."""
        from repohone import paths
        db = paths.state_db("old")
        db.parent.mkdir(parents=True, exist_ok=True)
        older = [part for part in state.SCHEMA.split(";") if "fingerprint" not in part]
        with state.connect(db) as conn:
            conn.executescript(";".join(older))
            conn.executemany("INSERT INTO meta(key, value) VALUES(?, ?)", [
                ("schema_version", str(state.SCHEMA_VERSION)),
                ("contract_version", state.CONTRACT_VERSION),
                ("checkout_id", "old")])
        self.assertEqual([], fingerprint.known("old", self.repo_id))

    def test_concurrent_equivalent_labels_join_one_series(self):
        state.initialize("shared")
        sessions = {
            name: ensure_session("shared", name, self.repo_id, "dev")
            for name in ("one", "two")
        }
        original = fingerprint.known
        first_inside = threading.Event()
        release_first = threading.Event()
        calls_lock = threading.Lock()
        calls = 0

        def slowed(checkout, repository):
            nonlocal calls
            result = original(checkout, repository)
            with calls_lock:
                calls += 1
                first = calls == 1
            if first:
                first_inside.set()
                release_first.wait(2)
            return result

        results, errors = [], []

        def write(name, behavior):
            try:
                results.append(canonical_occurrence(
                    "shared", self.repo_id, "MISSING_GUARDRAIL", "persistence",
                    behavior, sessions[name], developer_id="dev"))
            except Exception as exc:
                errors.append(exc)

        fingerprint.known = slowed
        try:
            first = threading.Thread(
                target=write, args=("one", "direct db access"))
            second = threading.Thread(
                target=write, args=("two", "direct database access"))
            first.start()
            self.assertTrue(first_inside.wait(2))
            second.start()
            time.sleep(0.05)
            release_first.set()
            first.join(3)
            second.join(3)
        finally:
            release_first.set()
            fingerprint.known = original

        self.assertEqual([], errors)
        self.assertEqual(2, len(results))
        self.assertEqual(1, sum(created for _, created in results))
        marks = fingerprint.known("shared", self.repo_id)
        self.assertEqual(1, len(marks))
        self.assertEqual(2, marks[0].sessions)

    def test_tables_left_by_an_older_core_are_ignored(self):
        """Before 1.23 an index copied every diagnosis into state.db, and a row
        left there by a deleted diagnosis blocked every proposal."""
        from repohone import paths
        self._seen("legacy", "dev", "one")
        with state.connect(paths.state_db("legacy")) as conn:
            conn.execute("CREATE TABLE fingerprint_occurrences (fingerprint_id TEXT, "
                         "session_id TEXT, diagnosis_id TEXT, developer_id TEXT, at TEXT)")
            conn.execute("INSERT INTO fingerprint_occurrences VALUES(?,?,?,?,?)",
                         ("fp_deadbeefdeadbeef", "rh_gone", "dx_deadbeefdeadbeef", None, "t"))
            state.validate_state(conn, "legacy")

        scan = fingerprint.known("legacy", self.repo_id)
        self.assertTrue(scan.complete, list(scan.problems))
        self.assertEqual([1], [mark.sessions for mark in scan])

    def test_a_damaged_sibling_database_loses_no_history(self):
        """M6's case: a sibling's state.db holds no diagnoses, so damaging it
        can no longer make history incomplete."""
        from repohone import paths
        self._seen("healthy", "dev", "rh_one")
        broken = paths.state_db("unreadable")
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"not a sqlite database")

        scan = fingerprint.known("healthy", self.repo_id)

        self.assertEqual(1, len(scan))
        self.assertTrue(scan.complete, list(scan.problems))
        second = ensure_session("healthy", "two", self.repo_id, "dev")
        projected, _created = canonical_occurrence(
            "healthy", self.repo_id, "MISSING_CONTEXT", "reporting",
            "untyped-return", second, developer_id="dev")
        self.assertTrue(projected.complete)

    def test_a_published_diagnosis_that_differs_from_its_occurrence_is_refused(self):
        from repohone import artifacts
        session = ensure_session("pub", "one", self.repo_id, "dev")
        real = diagnosis.save

        def altered(checkout, rec):
            return real(checkout, dict(rec, fingerprint=dict(rec["fingerprint"],
                                                            area="somewhere else")))
        diagnosis.save = altered
        try:
            with self.assertRaises(artifacts.MalformedArtifact):
                canonical_occurrence("pub", self.repo_id, "MISSING_CONTEXT", "reporting",
                                     "untyped-return", session, developer_id="dev")
        finally:
            diagnosis.save = real


class Recurrence(Fixture):
    """Recurrence is what lets a diagnosis answer 'is this a project-level
    pattern?' — without it every hypothesis rests on one session."""

    def setUp(self):
        super().setUp()
        self.capture_corrected_session()

    def test_a_first_diagnosis_reports_a_single_session(self):
        result = self.diagnose(FakeEngine())
        self.assertEqual(1, result["recurrence"]["sessions"])
        self.assertEqual(0, result["recurrence"]["known_fingerprints"])
        self.assertTrue(result["fingerprint"]["created"])
        self.assert_conformant(result)

    def test_a_failed_artifact_publish_never_commits_an_occurrence(self):
        real = diagnosis.save

        def fail(*args, **kwargs):
            raise OSError("simulated artifact failure")

        diagnosis.save = fail
        try:
            with self.assertRaises(OSError):
                self.diagnose(FakeEngine())
        finally:
            diagnosis.save = real
        self.assertEqual([], fingerprint.known(self.checkout, self._repository_id()))

    def test_a_deleted_diagnosis_stops_counting_and_blocks_nothing(self):
        """Regression: its index row outlived a deleted diagnosis, so every later
        proposal in the repository was refused, citing the file that was gone."""
        from repohone import paths
        repo = self._repository_id()
        first = ensure_session("tomb", "one", repo, "dev")
        canonical_occurrence("tomb", repo, "MISSING_CONTEXT", "reporting",
                             "untyped-return", first, developer_id="dev")
        second = ensure_session("tomb", "two", repo, "dev")
        kept, _created = canonical_occurrence(
            "tomb", repo, "MISSING_GUARDRAIL", "layering", "handler-imports-db",
            second, developer_id="dev")
        doomed = [p for p in (paths.checkout_dir("tomb") / "diagnoses").glob("dx_*.json")
                  if "untyped-return" in p.read_text()]
        doomed[0].unlink()

        scan = fingerprint.known("tomb", repo)
        self.assertTrue(scan.complete, list(scan.problems))
        self.assertEqual([kept.fingerprint_id], [mark.fingerprint_id for mark in scan])
        occurrences = fingerprint.occurrences_for("tomb", kept.fingerprint_id)
        self.assertEqual([("tomb", second)], occurrences.require_complete())
        self.assertEqual((1, []), fingerprint.diagnosis_status("tomb"))

    def test_the_same_problem_in_another_session_recurs(self):
        self.diagnose(FakeEngine())
        second = diagnosis.run(self.repo, self.checkout, self.session, FakeEngine())
        self.assertFalse(second["fingerprint"]["created"], "a near-duplicate was created")
        self.assertEqual(first_id(self.checkout), second["fingerprint"]["fingerprint_id"])

    def test_recurrence_counts_distinct_sessions_not_diagnoses(self):
        self.diagnose(FakeEngine())
        again = self.diagnose(FakeEngine())
        self.assertEqual(1, again["recurrence"]["sessions"],
                         "re-diagnosing one session inflated recurrence")

    def test_a_reworded_fingerprint_does_not_fork_the_series(self):
        self.diagnose(FakeEngine())
        reworded = dict(GOOD_REPLY, fingerprint={"area": "reporting layer",
                                                 "behavior": "untyped return values"})
        second = self.diagnose(FakeEngine(reworded))
        self.assertFalse(second["fingerprint"]["created"])
        self.assertEqual(1, len(fingerprint.known(self.checkout, self._repository_id())))

    def test_a_genuinely_different_problem_creates_a_new_pattern(self):
        self.diagnose(FakeEngine())
        other = dict(GOOD_REPLY, fingerprint={"area": "input-validation",
                                              "behavior": "missing-length-check"})
        second = self.diagnose(FakeEngine(other))
        self.assertTrue(second["fingerprint"]["created"])
        self.assertEqual(2, len(fingerprint.known(self.checkout, self._repository_id())))

    def test_known_patterns_are_shown_to_the_model(self):
        """§9 — an existing fingerprint must be preferred to a new one."""
        self.diagnose(FakeEngine())
        engine = FakeEngine()
        self.diagnose(engine)
        self.assertIn("reporting-layer", engine.prompt)
        self.assertIn("KNOWN FINGERPRINTS", engine.prompt)

    def test_the_prompt_warns_that_one_observation_is_weak(self):
        engine = FakeEngine()
        self.diagnose(engine)
        self.assertIn("One observation is weak grounds", engine.prompt)

    def test_the_fingerprint_block_is_charged_to_the_budget(self):
        """ARCH §12 — it is model input, so it is not free."""
        self.diagnose(FakeEngine())
        second = self.diagnose(FakeEngine())
        self.assertGreater(second["budget"]["estimated_tokens"],
                           sum(r["estimated_tokens"] for r in second["evidence_refs"]))
        self.assertLessEqual(second["budget"]["estimated_tokens"],
                             second["budget"]["max_tokens"])

    def test_a_failed_diagnosis_records_no_pattern(self):
        result = self.diagnose(FakeEngine("prose, not json"))
        self.assertIsNone(result["fingerprint"])
        scan = fingerprint.known(self.checkout, self._repository_id())
        self.assertEqual([], scan)
        # A failed run has no fingerprint; read as damage it would leave every
        # later recurrence incomplete, and Phase 5 refuses incomplete history.
        self.assertTrue(scan.complete, list(scan.problems))
        self.assertIs(True, self.diagnose(FakeEngine())["recurrence"]["complete"])
        self.assert_conformant(result)

    def test_a_success_without_a_fingerprint_is_invalid(self):
        reply = dict(GOOD_REPLY)
        del reply["fingerprint"]
        result = self.diagnose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])
        self.assertIn("fingerprint", result["failure"]["reason"])

    def test_incomplete_cross_checkout_history_is_a_floor_not_a_refusal(self):
        """Regression: one unreadable sibling refused every diagnosis in the
        repository. Missing history can only undercount; the record says so."""
        from repohone import artifacts, paths
        repository_id = identity.repository_id("git@github.com:Acme/App.git")
        rec = record.load(self.checkout, self.session)
        rec["repository"]["id"] = repository_id
        artifacts.atomic_write(record.record_path(self.checkout, self.session), rec)
        broken = paths.checkout_dir("unreadable-sibling") / "diagnoses"
        broken.mkdir(parents=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")
        engine = FakeEngine()

        result = self.diagnose(engine)

        self.assertEqual(reasoning.SUCCESS, result["outcome"])
        self.assertEqual(1, engine.calls)
        self.assertIs(False, result["recurrence"]["complete"])
        self.assert_conformant(result)
        mark = result["fingerprint"]["fingerprint_id"]
        with self.assertRaises(fingerprint.IncompleteHistory):
            fingerprint.occurrences_for(self.checkout, mark).require_complete()

    def test_complete_history_says_so(self):
        result = self.diagnose(FakeEngine())
        self.assertIs(True, result["recurrence"]["complete"])

    def test_a_damaged_state_database_loses_no_recurrence(self):
        """The database holds no diagnoses, so damaging it loses none."""
        from repohone import paths
        paths.state_db(self.checkout).write_bytes(b"not a sqlite database")
        result = self.diagnose(FakeEngine())
        self.assertEqual(reasoning.SUCCESS, result["outcome"])
        self.assertIs(True, result["recurrence"]["complete"])
        self.assertEqual([result["diagnosis_id"]],
                         [d["diagnosis_id"] for d in diagnosis.load_all(self.checkout)])

    def _repository_id(self):
        from repohone import identity as ident
        return ident.repository_id(ident.origin_url(self.repo))

    def test_doctor_names_an_unreadable_diagnosis(self):
        """The one thing that still makes recurrence incomplete, and deleting the
        file is now safe."""
        from repohone import doctor, paths
        self.diagnose(FakeEngine())
        broken = paths.checkout_dir(self.checkout) / "diagnoses" / "dx_0000000000000000.json"
        broken.write_text("{ not json")
        status, detail = {name: (status, detail)
                          for name, status, detail in doctor.run(self.repo)}["diagnoses"]
        self.assertEqual("fail", status)
        self.assertIn("dx_0000000000000000.json", detail)
        broken.unlink()
        statuses = {name: status for name, status, _ in doctor.run(self.repo)}
        self.assertEqual("ok", statuses["diagnoses"])


def first_id(checkout):
    marks = fingerprint.known(checkout, None)
    return marks[0].fingerprint_id if marks else None


class RecordShape(Fixture):
    """ARCH §22 — analysis output is a separate record carrying its versions."""

    def test_versions_are_present_and_not_guessed(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine())
        self.assertEqual("repohone.diagnosis/v1", result["schema"])
        self.assertEqual(1, result["schema_version"])
        self.assertEqual(CONTRACT_VERSION, result["contract_version"])

    def test_the_diagnosis_is_not_written_into_the_session_record(self):
        self.capture_corrected_session()
        self.diagnose(FakeEngine())
        session = record.load(self.checkout, self.session)
        self.assertNotIn("diagnosis", json.dumps(session))
        self.assertNotIn("root_cause", json.dumps(session))

    def test_the_engine_records_whether_it_was_isolated(self):
        self.capture_corrected_session()
        result = self.diagnose(FakeEngine())
        self.assertTrue(result["engine"]["isolated"])
        self.assertEqual("fake", result["engine"]["name"])



class TheWholePromptIsBudgeted(Fixture):
    """Regression: only the evidence was counted, so ~750 tokens of instructions
    went to the provider outside the ceiling the profile promises."""

    def a_wordy_session(self):
        """Evidence large enough that the template's cost decides whether the
        prompt fits: with a small session everything fits either way."""
        filler = "context " * 900
        self.fire("UserPromptSubmit", prompt=f"add count_users. {filler}",
                  prompt_id="t1")
        (self.repo / "src" / "stats.py").write_text("def count_users():\n    return (1,)\n")
        self.fire("Stop", prompt_id="t1",
                  last_assistant_message=f"Added count_users. {filler}")
        self.fire("UserPromptSubmit", prompt=f"No, a typed dataclass. {filler}",
                  prompt_id="t2")
        self.fire("Stop", prompt_id="t2", last_assistant_message=f"Fixed. {filler}")

    def test_what_is_actually_sent_stays_within_the_ceiling(self):
        self.a_wordy_session()
        engine = FakeEngine()
        self.diagnose(engine)
        self.assertIsNotNone(engine.prompt)
        sent = evidence.measure_bytes(engine.prompt)
        self.assertGreater(sent, evidence.MAX_BYTES // 2, "fixture was too small")
        self.assertLessEqual(sent, evidence.MAX_BYTES,
                             "the prompt sent to the provider exceeded the ceiling")

    def test_the_recorded_estimate_covers_the_whole_prompt(self):
        self.a_wordy_session()
        engine = FakeEngine()
        rec = self.diagnose(engine)
        self.assertGreaterEqual(rec["budget"]["estimated_tokens"],
                                evidence.measure_bytes(engine.prompt),
                                "the record under-reported what was sent")
        self.assertLessEqual(rec["budget"]["estimated_tokens"], evidence.MAX_BYTES)

    def test_the_trim_bounds_the_prompt_when_the_estimate_is_short(self):
        items = [evidence.Item("turn_prompt", f"r{i}", "word " * 300)
                 for i in range(8)]
        selection = evidence.Selection(items=list(items))
        ceiling = diagnosis.static_prompt_bytes() + 1700
        prompt, tokens = diagnosis.fit_to_budget(selection, [], ceiling=ceiling)
        self.assertLessEqual(tokens, ceiling)
        self.assertLess(len(selection.items), len(items))
        self.assertGreater(selection.skipped, 0)

    def test_the_trim_keeps_at_least_one_item(self):
        selection = evidence.Selection(
            items=[evidence.Item("turn_prompt", "r", "word " * 2000)])
        diagnosis.fit_to_budget(selection, [], ceiling=10)
        self.assertEqual(1, len(selection.items))

    def test_the_static_template_costs_something(self):
        self.assertGreater(diagnosis.static_prompt_bytes(), 100)

    def framing(self):
        """The whole prompt around one empty item: template, the item's header,
        and the line saying no fingerprints are known."""
        return evidence.measure_bytes(diagnosis.build_prompt(
            evidence.Selection(items=[evidence.Item("turn_prompt", "r", "")]), []))

    def test_a_budget_filling_selection_still_fits_the_ceiling(self):
        """Exactly: evidence filling what the framing leaves makes the ceiling.
        Budgeting against the template alone left out the item's header."""
        item = evidence.Item("turn_prompt", "r", "w" * (evidence.MAX_BYTES - self.framing()))
        whole = diagnosis.build_prompt(evidence.Selection(items=[item]), [])
        self.assertEqual(evidence.MAX_BYTES, evidence.measure_bytes(whole))

    def test_the_instructions_leave_evidence_at_least_half_the_budget(self):
        """Every sentence added to the instructions is evidence the model no
        longer sees; a live diagnosis already came back short of evidence."""
        self.assertGreaterEqual(evidence.MAX_BYTES - self.framing(), evidence.MAX_BYTES // 2)

    def test_the_template_is_not_counted_twice(self):
        first = diagnosis.static_prompt_bytes()
        self.assertEqual(first, diagnosis.static_prompt_bytes())

    def test_known_fingerprint_summary_cannot_poison_the_next_prompt(self):
        known = [fingerprint.Fingerprint(
            f"fp_{i:016x}", "MISSING_CONTEXT", "a" * 200, "b" * 200,
            sessions=1) for i in range(12)]
        summary = fingerprint.summarise(known)
        self.assertLessEqual(len(summary.encode("utf-8")),
                             fingerprint.SUMMARY_MAX_BYTES)
        selection = evidence.Selection(
            items=[evidence.Item("correction", "turn/1", "correct this")])
        _prompt, tokens = diagnosis.fit_to_budget(selection, known)
        self.assertLessEqual(tokens, evidence.MAX_BYTES)


class InterruptionAtTheProcessBoundary(Fixture):
    """ARCH §12. Exercised against a real subprocess, not a stub engine: the
    contract that matters is what happens to the process that is mid-egress."""

    def slow_claude(self, seconds=30):
        binaries = self.tmp / "bin"
        binaries.mkdir(exist_ok=True)
        # A grandchild, like the real CLI has. Killing only the direct child
        # leaves it running — and still talking to the provider.
        (binaries / "claude").write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "2.1.267 (Claude Code)"; exit 0; fi\n'
            f"sh -c 'sleep {seconds}' &\n"
            f"echo $! > {self.tmp}/grandchild.pid\n"
            f"sleep {seconds}\n")
        (binaries / "claude").chmod(0o755)
        self._path = os.environ["PATH"]
        os.environ["PATH"] = f"{binaries}:{self._path}"
        self.addCleanup(lambda: os.environ.__setitem__("PATH", self._path))

    def grandchild_alive(self):
        pid_file = self.tmp / "grandchild.pid"
        if not pid_file.is_file():
            return False
        try:
            os.kill(int(pid_file.read_text().strip()), 0)
        except (OSError, ValueError):
            return False
        return True

    def test_a_cut_short_run_is_recorded_as_interrupted(self):
        self.capture_corrected_session()
        self.slow_claude()
        rec = self.diagnose(reasoning.ClaudeCliEngine(timeout=1.0))
        self.assertEqual(reasoning.INTERRUPTED, rec["outcome"])
        self.assertIn("exceeded", rec["failure"]["reason"])
        self.assert_conformant(rec)

    def test_an_interrupted_run_stores_no_partial_diagnosis(self):
        self.capture_corrected_session()
        self.slow_claude()
        rec = self.diagnose(reasoning.ClaudeCliEngine(timeout=1.0))
        for field in ("root_cause", "confidence", "required_property", "fingerprint"):
            self.assertIsNone(rec[field], f"{field} survived an interruption")
        self.assertEqual(0, rec["recurrence"]["sessions"],
                         "an interrupted run counted as an occurrence")

    def test_interruption_leaves_no_process_still_talking_to_the_provider(self):
        """Regression: the CLI's own child outlived the kill, so egress carried
        on after RepoHone had reported the run stopped."""
        self.capture_corrected_session()
        self.slow_claude()
        self.diagnose(reasoning.ClaudeCliEngine(timeout=1.0))
        time.sleep(0.5)
        self.assertFalse(self.grandchild_alive(),
                         "a subprocess survived the interruption")

    def test_the_evidence_selected_before_the_call_is_still_recorded(self):
        self.capture_corrected_session()
        self.slow_claude()
        rec = self.diagnose(reasoning.ClaudeCliEngine(timeout=1.0))
        self.assertTrue(rec["evidence_refs"],
                        "what was sent is not recoverable from the record")


class ConfidenceMustBeEarned(unittest.TestCase):
    """Regression: SUCCESS with high confidence and three empty lists was accepted,
    which reads as decisive downstream where it drives a real change."""

    BASE = {"outcome": "SUCCESS",
            "root_cause": {"class": "MISSING_CONTEXT", "summary": "no rule was written"},
            "required_property": "output must go through the logger",
            "fingerprint": {"area": "logging", "behavior": "print used"},
            "corrective_turn": 2}

    def interpret(self, **kw):
        return diagnosis.interpret({**self.BASE, **kw})

    def test_a_success_with_no_supporting_evidence_is_rejected(self):
        out = self.interpret(confidence="high", evidence_for=[],
                             evidence_against=[], risks=[])
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("assertion, not a finding", out["failure"])

    def test_whitespace_is_not_evidence(self):
        out = self.interpret(confidence="low", evidence_for=["   ", ""],
                             evidence_against=[], risks=[])
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])

    def test_fingerprint_fields_cannot_exceed_the_persisted_schema(self):
        self.assertEqual(200, diagnosis.FINGERPRINT_MAX_CHARS)
        out = self.interpret(
            confidence="low", evidence_for=["turn 1"], evidence_against=[], risks=[],
            fingerprint={"area": "a" * 201, "behavior": "valid"})
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("200", out["failure"])

    def test_high_confidence_that_weighed_nothing_is_downgraded(self):
        out = self.interpret(confidence="high", evidence_for=["turn 1 shows it"],
                             evidence_against=[], risks=[])
        self.assertEqual(reasoning.SUCCESS, out["outcome"])
        self.assertEqual("medium", out["body"]["confidence"])
        self.assertEqual({"from": "high", "to": "medium"},
                         {k: out["adjustment"][k] for k in ("from", "to")})

    def test_high_confidence_survives_a_named_risk(self):
        out = self.interpret(confidence="high", evidence_for=["turn 1"],
                             evidence_against=[], risks=["may be noisy"])
        self.assertEqual("high", out["body"]["confidence"])
        self.assertIsNone(out["adjustment"])

    def test_high_confidence_survives_named_counter_evidence(self):
        out = self.interpret(confidence="high", evidence_for=["turn 1"],
                             evidence_against=["turn 3 contradicts"], risks=[])
        self.assertEqual("high", out["body"]["confidence"])
        self.assertIsNone(out["adjustment"])

    def test_the_adjustment_is_recorded_not_applied_silently(self):
        out = self.interpret(confidence="high", evidence_for=["turn 1"],
                             evidence_against=[], risks=[])
        built = diagnosis.build_record(
            "rh_0123456789abcdef", "c", reasoning.SUCCESS, evidence.Selection(),
            "fake", None, True, body=out["body"], adjustment=out["adjustment"],
            mark={"fingerprint_id": "fp_0123456789abcdef", "area": "logging",
                  "behavior": "print used", "created": True})
        self.assertEqual("medium", built["confidence"])
        self.assertIsNotNone(built["confidence_adjusted"])
        self.assertIn("counter-evidence", built["confidence_adjusted"]["reason"])
        if VALIDATOR is not None:
            self.assertEqual([], [e.message for e in VALIDATOR.iter_errors(built)])


class TheModelThatAnswered(unittest.TestCase):
    """Regression: every live diagnosis recorded Haiku, a background call Claude
    Code lists first in `modelUsage`, although Opus wrote the answer."""

    USAGE = {"claude-haiku-4-5-20251001": {"outputTokens": 38},
             "claude-opus-5": {"outputTokens": 912}}

    def test_the_model_that_wrote_the_most_is_recorded(self):
        self.assertEqual("claude-opus-5", reasoning.answering_model(self.USAGE))

    def test_nothing_reported_names_no_model(self):
        self.assertIsNone(reasoning.answering_model({}))
        self.assertIsNone(reasoning.answering_model(None))

    def test_the_cli_envelope_is_read_that_way(self):
        tmp = Path(tempfile.mkdtemp(prefix="repohone-model-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "bin").mkdir()
        (tmp / "bin" / "claude").write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "2.1.267 (Claude Code)"; exit 0; fi\n'
            "cat > /dev/null\n"
            f"echo '{json.dumps({'result': '{}', 'modelUsage': self.USAGE})}'\n")
        (tmp / "bin" / "claude").chmod(0o755)
        path, data = os.environ["PATH"], os.environ.get("REPOHONE_DATA_DIR")
        os.environ["PATH"] = f"{tmp / 'bin'}:{path}"
        os.environ["REPOHONE_DATA_DIR"] = str(tmp / "data")
        try:
            reply = reasoning.ClaudeCliEngine(timeout=20).run("small prompt")
        finally:
            os.environ["PATH"] = path
            os.environ.pop("REPOHONE_DATA_DIR", None)
            if data is not None:
                os.environ["REPOHONE_DATA_DIR"] = data
        self.assertEqual("claude-opus-5", reply.model)


class ADiagnosisNamesItsRepository(Fixture):
    """Recurrence reads each diagnosis alone; it used to load a whole session
    record, up to 1.3 MB, for two short ids (Recommendation 7)."""

    def setUp(self):
        super().setUp()
        git(["remote", "add", "origin", "git@github.com:Acme/App.git"], self.repo)

    def test_the_ids_are_copied_from_the_session(self):
        self.capture_corrected_session()
        rec = self.diagnose(FakeEngine())
        session = record.load(self.checkout, self.session)
        self.assertIsNotNone(rec["repository_id"])
        self.assertEqual(session["repository"]["id"], rec["repository_id"])
        self.assertEqual(session["developer"]["id"], rec["developer_id"])
        self.assert_conformant(rec)

    def test_recurrence_no_longer_needs_the_session_record(self):
        self.capture_corrected_session()
        rec = self.diagnose(FakeEngine())
        record.record_path(self.checkout, self.session).unlink()
        marks = fingerprint.known(self.checkout, rec["repository_id"])
        self.assertEqual((), marks.problems)
        self.assertEqual(1, marks[0].sessions)

    def test_a_diagnosis_from_before_1_24_still_reads_its_session(self):
        self.capture_corrected_session()
        rec = self.diagnose(FakeEngine())
        path = diagnosis.diagnoses_dir(self.checkout) / f"{rec['diagnosis_id']}.json"
        older = json.loads(path.read_text())
        for key in ("repository_id", "developer_id", "history"):
            older.pop(key)
        older["recurrence"].pop("historical")
        older["contract_version"] = "1.23"
        path.write_text(json.dumps(older))
        self.assertEqual(1, fingerprint.known(self.checkout, rec["repository_id"])[0].sessions)
        record.record_path(self.checkout, self.session).unlink()
        self.assertTrue(fingerprint.known(self.checkout, rec["repository_id"]).problems)

    def test_a_current_diagnosis_without_them_is_invalid(self):
        self.capture_corrected_session()
        rec = self.diagnose(FakeEngine())
        for key in ("repository_id", "developer_id", "history"):
            broken = {k: v for k, v in rec.items() if k != key}
            with self.assertRaises(artifacts.MalformedArtifact, msg=key):
                artifacts.validate(broken, "diagnosis")


class AnEarlierCommitCanBeTheSameProblem(Fixture):
    """Phase 6 candidates join analysis when a captured session shows the same
    problem: the diagnosis is offered matching reverts and fix-ups, and the model
    says which, if any, it is."""

    CORRECTION = "No - handlers must not query the db directly."

    def revert_in_history(self, subject="handlers query db directly"):
        (self.repo / "src" / "handler.py").write_text("import db\n")
        git(["add", "src/handler.py"], self.repo)
        git(["commit", "-qm", subject], self.repo)
        git(["revert", "--no-edit", "HEAD"], self.repo)
        return git(["rev-parse", "--short", "HEAD"], self.repo)

    def mine(self):
        found, errors, unavailable = bootstrap.collect(self.repo)
        mined = bootstrap.build_record(self.checkout, None, found, record.now(),
                                       errors, unavailable)
        bootstrap.save(self.checkout, mined)
        return next(c for c in mined["candidates"] if c["source"] == "revert")

    def capture_db_correction(self):
        self.fire("UserPromptSubmit", prompt="add a recent-orders handler", prompt_id="t1")
        (self.repo / "src" / "orders.py").write_text("import db\n")
        self.fire("Stop", prompt_id="t1", last_assistant_message="Added.")
        self.fire("UserPromptSubmit", prompt=self.CORRECTION, prompt_id="t2")
        (self.repo / "src" / "orders.py").write_text("from services import orders\n")
        self.fire("Stop", prompt_id="t2", last_assistant_message="Fixed.")

    def run_with(self, reply=None):
        engine = FakeEngine(reply)
        rec = diagnosis.run(self.repo, self.checkout, self.session, engine,
                            corrective_turn=2)
        return engine, rec

    def test_a_matching_revert_the_session_started_from_is_offered(self):
        sha = self.revert_in_history()
        found = self.mine()
        self.capture_db_correction()
        engine, rec = self.run_with()
        self.assertIn(f"{found['id']}) ---\n{sha} (revert): reverted: handlers query db "
                      f"directly", engine.prompt)
        self.assertEqual([], rec["history"], "offered is not linked")

    def test_a_link_is_recorded_and_counted(self):
        sha = self.revert_in_history()
        found = self.mine()
        self.capture_db_correction()
        _, rec = self.run_with(dict(GOOD_REPLY, same_as_history=[found["id"]]))
        self.assertEqual(reasoning.SUCCESS, rec["outcome"])
        self.assertEqual([{"candidate_id": found["id"], "source": "revert", "commit": sha,
                           "statement": "reverted: handlers query db directly"}],
                         rec["history"])
        self.assertEqual(1, rec["recurrence"]["historical"])
        self.assertEqual(1, fingerprint.known(self.checkout, None)[0].historical)
        self.assert_conformant(rec)

    def test_the_history_reaches_selection_and_the_pattern_list(self):
        self.revert_in_history()
        found = self.mine()
        self.capture_db_correction()
        _, rec = self.run_with(dict(GOOD_REPLY, same_as_history=[found["id"]]))
        self.assertIn("earlier_reverted_or_fixed_commits=1", selection.build_prompt(rec, None))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli.cmd_patterns(argparse.Namespace(path=str(self.repo)))
        self.assertIn("+ 1 earlier commit", out.getvalue())

    def test_a_revert_made_after_the_session_started_is_not_offered(self):
        """It may be this very correction, and would be counted twice."""
        self.capture_db_correction()
        self.revert_in_history()
        self.mine()
        engine, _ = self.run_with()
        self.assertNotIn("(revert)", engine.prompt)

    def test_an_unrelated_revert_is_not_offered(self):
        self.revert_in_history("bump the logo size")
        self.mine()
        self.capture_db_correction()
        engine, _ = self.run_with()
        self.assertNotIn("(revert)", engine.prompt)

    def test_a_link_to_something_never_sent_is_invalid(self):
        self.capture_db_correction()
        _, rec = self.run_with(dict(GOOD_REPLY, same_as_history=["cand_000000000000"]))
        self.assertEqual(reasoning.INVALID_OUTPUT, rec["outcome"])
        self.assertEqual([], rec["history"])
        self.assert_conformant(rec)


class TheTeamChoosesTheAnalysisModel(Fixture):
    """The profile names the model for the whole team, a developer's own setting
    or `--model` overrides it, and Opus is the default (decided 2026-09-27).
    Measured at the process boundary: the flag must reach Claude Code."""

    def setUp(self):
        super().setUp()
        self.capture_corrected_session()
        binaries = self.tmp / "bin"
        binaries.mkdir()
        (binaries / "claude").write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "2.1.267 (Claude Code)"; exit 0; fi\n'
            f'printf "%s\\n" "$@" > {self.tmp}/argv.txt\n'
            "cat > /dev/null\n"
            "echo '{\"result\": \"{\\\"outcome\\\": \\\"INSUFFICIENT_EVIDENCE\\\"}\", "
            "\"modelUsage\": {\"m\": {}}}'\n")
        (binaries / "claude").chmod(0o755)
        self._path = os.environ["PATH"]
        os.environ["PATH"] = f"{binaries}:{self._path}"

    def tearDown(self):
        os.environ["PATH"] = self._path
        super().tearDown()

    def set_profile_model(self, model):
        path = self.repo / ".repohone" / "profile.yaml"
        path.write_text(path.read_text().replace("reasoning_model: opus",
                                                 f"reasoning_model: {model}"))

    def requested(self, **flags):
        args = argparse.Namespace(path=str(self.repo), session_id=self.session, candidate=None,
                                  rule=None, turn=2, yes=True, show=False, model=None,
                                  timeout=20)
        for key, value in flags.items():
            setattr(args, key, value)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(0, cli.cmd_diagnose(args))
        argv = (self.tmp / "argv.txt").read_text().splitlines()
        return argv[argv.index("--model") + 1]

    def test_new_profiles_name_opus_and_a_profile_without_one_defaults_to_it(self):
        self.assertIn("reasoning_model: opus", profile.template())
        path = self.repo / ".repohone" / "profile.yaml"
        path.write_text(path.read_text().replace("reasoning_model: opus\n", ""))
        self.assertIsNone(profile.load(self.repo).reasoning_model)
        self.assertEqual("opus", self.requested())

    def test_the_profile_chooses_for_the_team(self):
        self.set_profile_model("sonnet")
        self.assertEqual("sonnet", self.requested())

    def test_a_developer_setting_overrides_the_profile(self):
        self.set_profile_model("sonnet")
        (self.data / "settings.json").write_text(json.dumps({"reasoning_model": "haiku"}))
        self.assertEqual("haiku", self.requested())

    def test_one_command_overrides_everything(self):
        (self.data / "settings.json").write_text(json.dumps({"reasoning_model": "haiku"}))
        self.assertEqual("claude-opus-5-5", self.requested(model="claude-opus-5-5"))

    def test_a_damaged_setting_stops_analysis_instead_of_guessing(self):
        (self.data / "settings.json").write_text(json.dumps({"reasoning_model": "opus; rm"}))
        args = argparse.Namespace(path=str(self.repo), session_id=self.session, candidate=None,
                                  rule=None, turn=2, yes=True, show=False, model=None, timeout=5)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(2, cli.cmd_diagnose(args))
        self.assertIn("reasoning_model", err.getvalue())
        self.assertFalse((self.tmp / "argv.txt").exists(), "a model was called anyway")

    def test_the_preview_names_the_model_and_why(self):
        self.set_profile_model("sonnet")
        args = argparse.Namespace(path=str(self.repo), session_id=self.session, candidate=None,
                                  rule=None, turn=2, yes=False, show=True, model=None, timeout=5)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            cli.cmd_diagnose(args)
        self.assertIn("sent to sonnet (the project profile)", err.getvalue())

    def test_changing_the_model_pauses_no_one(self):
        """It says which model reads the evidence, not what is captured or sent."""
        before = profile.load(self.repo)
        self.set_profile_model("sonnet")
        after = profile.load(self.repo)
        self.assertEqual(before.digest, after.digest)
        self.assertIs(True, profile.accepted(self.repo, after))

    def test_a_malformed_model_makes_the_profile_invalid(self):
        self.set_profile_model("'opus sonnet'")
        loaded = profile.load(self.repo)
        self.assertEqual(profile.INVALID, loaded.state)
        self.assertIn("reasoning_model", "; ".join(loaded.problems))


if __name__ == "__main__":
    unittest.main()
