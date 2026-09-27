"""Phase 5 tests: mechanism selection, authentic-state validation, application.

The engine is stubbed so every terminal state is reachable without reaching a
model provider. Each test names the section it enforces.

Run:  python3 -m unittest test_phase5
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "src"
ROOT = CORE.parent
sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from repohone import (
    artifacts,
    cli,
    doctor,
    identity,
    probe,
    profile,
    proposal,
    reasoning,
    record,
    selection,
    state,
    validation,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import canonical_occurrence, ensure_session

SCHEMA_PATH = CORE / "repohone" / "schemas" / "proposal.v1.schema.json"
HOOK = CORE / "repohone_hook.py"
from repohone import CONTRACT_VERSION as CONTRACT

try:
    import jsonschema
    VALIDATOR = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
except ImportError:
    VALIDATOR = None


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                          text=True, check=True).stdout.strip()


MECHANISM = {
    "id": "mech_0123456789ab", "kind": "lint", "name": "ruff",
    "tier": "existing-project",
    # Passes while the rule is unselected; detects only once the change adds it.
    "argv": ["sh", "-c",
             "! grep -qs T20 pyproject.toml ruff.toml || ! grep -rq 'print(' src"],
    "invocation": "ruff check .", "config_path": "pyproject.toml",
    "config_excerpt": None, "deterministic": True, "evidence": ["configured"],
    "carries": ["static-rule"], "enforced_in": ["ci"],
    "cost": {"probed": False, "duration_ms": None, "exit_code": None, "note": None},
}

INVENTORY = {"schema": "repohone.mechanisms/v1", "mechanisms": [MECHANISM]}

# Advice an agent reads: nothing to execute, so it is graded, never run.
ADVICE = {**MECHANISM, "id": "mech_ba9876543210", "kind": "agent-instruction",
          "name": "CLAUDE.md", "argv": None, "invocation": "read by the agent",
          "deterministic": False, "enforced_in": []}
ADVICE_INVENTORY = {"schema": "repohone.mechanisms/v1", "mechanisms": [ADVICE]}

GOOD_REPLY = {
    "outcome": "SUCCESS",
    "mechanism_id": MECHANISM["id"],
    "why_this": "ruff already runs in CI",
    "why_existing_did_not_help": "ruff ran but no selected rule could flag print()",
    "alternatives_considered": ["a git hook: not enforced anywhere",
                                "doing nothing: the pattern recurred three times"],
    "change": {"summary": "enable the print rule",
               "files": [{"path": "pyproject.toml", "action": "modify",
                          "edits": [{"find": "select = ['E']",
                                     "replace": "select = ['E','T20']"}]}]},
    "expected_effect": "print() fails lint",
    "friction": ["fires at lint time, not while typing"],
    "adds_model_egress": False,
}

DIAGNOSIS = {
    "diagnosis_id": "dx_0123456789abcdef", "session_id": "rh_0123456789abcdef",
    "outcome": "SUCCESS", "confidence": "high",
    "root_cause": {"class": "MISSING_CONTEXT", "summary": "the rule was unenforced"},
    "required_property": "output in src must go through the logger",
    "corrective_turn": 2,
    "fingerprint": {"fingerprint_id": "fp_0123456789abcdef", "area": "output-handling",
                    "behavior": "print-instead-of-logger", "created": False},
    "recurrence": {"sessions": 3, "checkouts": 1, "developers": 1,
                   "known_fingerprints": 1, "scope": "local-checkouts"},
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
        return reasoning.Reply(text=self.reply, model="fake", isolated=True)


class Fixture(unittest.TestCase):
    """A repository whose two captured states differ in exactly the way the
    candidate mechanism is supposed to discriminate."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-p5-"))
        self.repo = self.tmp / "repo"
        (self.repo / "src").mkdir(parents=True)
        os.environ["REPOHONE_DATA_DIR"] = str(self.tmp / "data")
        git(["init", "-q"], self.repo)
        git(["config", "user.email", "d@e.com"], self.repo)
        git(["config", "user.name", "D"], self.repo)
        (self.repo / "pyproject.toml").write_text("[tool.ruff.lint]\nselect = ['E']\n")
        (self.repo / "src" / "app.py").write_text("def run():\n    return 1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "init"], self.repo)
        self.checkout = identity.checkout_id(self.repo)
        state.initialize(self.checkout)
        self.bad_tree, self.good_tree = self._two_states()

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tree(self):
        git(["add", "-A"], self.repo)
        return git(["write-tree"], self.repo)

    def _two_states(self):
        (self.repo / "src" / "app.py").write_text('def run():\n    print("x")\n')
        bad = self._tree()
        (self.repo / "src" / "app.py").write_text(
            "from src.log import log\n\n\ndef run():\n    log.info('x')\n")
        good = self._tree()
        git(["commit", "-qm", "states"], self.repo)
        return bad, good

    def session(self):
        return {"turns": [
            {"index": 1, "logical_turn_id": "t1",
             "prompt_events": [{"ordinal": 1, "snapshot": None}],
             "stop_events": [{"ordinal": 1, "kind": "stop",
                              "snapshot": {"tree": self.bad_tree, "ref": "r/stop/1"}}]},
            {"index": 2, "logical_turn_id": "t2",
             "prompt_events": [{"ordinal": 1, "snapshot":
                                {"tree": self.bad_tree, "ref": "r/prompt/1"}}],
             "stop_events": [{"ordinal": 1, "kind": "stop",
                              "snapshot": {"tree": self.good_tree, "ref": "r/stop/2"}}]},
        ], "session_end_snapshots": []}

    def pair(self, session=None, boundary=2):
        """Turn 2 is the identified correction in this fixture's session."""
        return validation.authentic_pair(session or self.session(), boundary)

    def propose(self, engine=None, session=None, inventory=INVENTORY):
        return proposal.propose(self.repo, self.checkout, DIAGNOSIS, inventory,
                                session if session is not None else self.session(),
                                engine or FakeEngine(), timeout=30)

    def approved(self, changes, proposal_id="prop_d1ec7000"):
        """An approved proposal for exactly these files, without going through
        selection: these tests exercise how a change lands, not how it is chosen."""
        digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        entries = []
        for path, contents in changes.items():
            target = self.repo / path
            existing = target.read_text() if target.is_file() else None
            entries.append({"path": path,
                            "action": "modify" if existing is not None else "create",
                            "bytes": len(contents), "sha256": digest(contents),
                            "baseline_sha256": digest(existing) if existing is not None
                            else None,
                            "baseline_mode": (stat.S_IMODE(target.lstat().st_mode)
                                              if existing is not None else None)})
        rec = {"schema": "repohone.proposal/v1", "schema_version": 1,
               "contract_version": CONTRACT, "proposal_id": proposal_id,
               "diagnosis_id": "dx_deadbeef", "session_id": "rh_deadbeef",
               "checkout_id": self.checkout, "created_at": "2026-01-01T00:00:00Z",
               "state": "approved", "outcome": "SUCCESS", "problem": "x",
               "required_property": "x", "fingerprint": None,
               "mechanism": {"id": MECHANISM["id"], "kind": MECHANISM["kind"],
                             "name": MECHANISM["name"], "tier": MECHANISM["tier"],
                             "why_this": "application fixture",
                             "why_existing_did_not_help": "the rule was absent",
                             "alternatives_considered": ["do nothing"],
                             "deterministic": True, "cost": None},
               "change": {"summary": "x", "files": entries},
               "expected_effect": "x", "friction": [],
               "privacy": {"adds_model_egress": False, "note": None,
                           "egress": None},
               "validation": {"performed": True, "attributable": True,
                              "rejects_problematic": True, "accepts_repaired": True,
                              "pre_correction": None, "post_correction": None,
                              "occurrences": [], "working_tree_unchanged": True,
                              "estimate": None, "notes": []},
               "measurement": {"signal": "x", "target_fingerprint": None,
                               "baseline_sessions": 0},
               "rollback": {"files": [], "instructions": "x"},
               "repohone_runtime_required": False, "failure": None}
        proposal.save(self.checkout, rec)
        proposal.stage(self.checkout, proposal_id, dict(changes))
        return rec

    def apply_it(self, rec):
        return proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                              proposal.staged(self.checkout, rec["proposal_id"]))

    def assert_conformant(self, rec):
        if VALIDATOR is not None:
            errors = sorted(VALIDATOR.iter_errors(rec), key=lambda e: e.path)
            self.assertEqual([], ["/".join(map(str, e.path)) + ": " + e.message
                                  for e in errors])


class AuthenticPair(Fixture):
    """LEARNING_PLAN §14 — prefer authentic captured states over fixtures, with
    a ladder because the hardest correction shapes never produce a Stop."""

    def test_the_preferred_pair_is_stop_to_stop(self):
        pre, post, notes = self.pair()
        self.assertEqual("stop", pre.source)
        self.assertEqual("stop", post.source)
        self.assertEqual(self.bad_tree, pre.tree)
        self.assertEqual(self.good_tree, post.tree)

    def test_a_type_ahead_correction_falls_back_to_its_prompt(self):
        """The agent never stopped, so there is no Stop before the correction."""
        session = self.session()
        session["turns"][0]["stop_events"] = []
        pre, post, notes = self.pair(session)
        self.assertEqual("prompt", pre.source)
        self.assertTrue(any("prompt's own snapshot" in note for note in notes))

    def test_an_interrupted_correction_falls_back_to_session_end(self):
        session = self.session()
        session["turns"][1]["stop_events"] = []
        session["session_end_snapshots"] = [{"tree": self.good_tree, "ref": "r/se/1"}]
        pre, post, notes = self.pair(session)
        self.assertEqual("session_end", post.source)

    def test_a_session_with_no_usable_states_says_so(self):
        pre, post, notes = validation.authentic_pair({"turns": []}, 2)
        self.assertIsNone(pre)
        self.assertIsNone(post)
        self.assertTrue(notes)

    def test_identical_trees_cannot_discriminate(self):
        session = self.session()
        session["turns"][1]["stop_events"][0]["snapshot"]["tree"] = self.bad_tree
        pre, post, notes = self.pair(session)
        self.assertTrue(any("identical" in note for note in notes))


class CorrectionBoundary(Fixture):
    """Audit B7: adjacency is not a correction. A feature turn followed by an
    unrelated request looks identical to a mistake and its repair."""

    def test_no_boundary_means_validation_is_unavailable(self):
        pre, post, notes = validation.authentic_pair(self.session(), None)
        self.assertIsNone(pre)
        self.assertIsNone(post)
        self.assertIn("no identified correction boundary", notes[0])

    def test_unrelated_turns_are_never_called_a_correction(self):
        result = validation.validate(self.repo, self.checkout, MECHANISM["argv"], {},
                                     *validation.authentic_pair(self.session(), None)[:2],
                                     timeout=30)
        self.assertFalse(result.performed)
        self.assertFalse(result.passed)

    def test_a_first_turn_has_no_predecessor(self):
        pre, post, notes = validation.authentic_pair(self.session(), 1)
        self.assertIsNone(pre)
        self.assertIn("no predecessor", notes[0])

    def test_an_unknown_turn_is_refused(self):
        pre, post, notes = validation.authentic_pair(self.session(), 99)
        self.assertIsNone(pre)

    def test_a_proposal_without_a_boundary_is_validation_rejected(self):
        diagnosis = json.loads(json.dumps(DIAGNOSIS))
        diagnosis["corrective_turn"] = None
        result = proposal.propose(self.repo, self.checkout, diagnosis, INVENTORY,
                                  self.session(), FakeEngine(), timeout=30)
        self.assertEqual("VALIDATION_REJECTED", result["outcome"])
        self.assertIn("no identified correction boundary",
                      " ".join(result["validation"]["notes"]))


class Validation(Fixture):
    """The candidate must reject the state the agent produced and accept the
    state the developer accepted."""

    def test_a_discriminating_mechanism_passes(self):
        pre, post, _ = self.pair()
        overlay = {"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"}
        result = validation.validate(self.repo, self.checkout, MECHANISM["argv"],
                                     overlay, pre, post, timeout=30)
        self.assertTrue(result.attributable)
        self.assertTrue(result.performed)
        self.assertTrue(result.rejects_problematic)
        self.assertTrue(result.accepts_repaired)
        self.assertTrue(result.passed)

    def test_a_mechanism_that_misses_the_mistake_fails(self):
        pre, post, _ = self.pair()
        result = validation.validate(self.repo, self.checkout, ["true"], {},
                                     pre, post, timeout=30)
        self.assertTrue(result.attributable)
        self.assertFalse(result.rejects_problematic)
        self.assertFalse(result.passed)
        self.assertTrue(any("would not have caught" in n for n in result.notes))

    def test_a_mechanism_that_also_flags_correct_work_fails(self):
        """Passes without the change, then fires on both states with it."""
        pre, post, _ = self.pair()
        overlay = {"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"}
        result = validation.validate(
            self.repo, self.checkout,
            ["sh", "-c", "! grep -q T20 pyproject.toml"], overlay,
            pre, post, timeout=30)
        self.assertTrue(result.attributable)
        self.assertFalse(result.accepts_repaired)
        self.assertFalse(result.passed)
        self.assertTrue(any("correct work" in n for n in result.notes))

    def test_validation_never_touches_the_working_tree(self):
        """ARCH §12.2 — and the proposed change is applied only in scratch."""
        before = probe.tree_fingerprint(self.repo)
        pre, post, _ = self.pair()
        overlay = {"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"}
        result = validation.validate(self.repo, self.checkout, MECHANISM["argv"],
                                     overlay, pre, post, timeout=30)
        self.assertTrue(result.working_tree_unchanged)
        self.assertEqual(before, probe.tree_fingerprint(self.repo))
        self.assertIn("select = ['E']", (self.repo / "pyproject.toml").read_text())

    def test_an_overlay_cannot_escape_the_checkout(self):
        pre, post, _ = self.pair()
        with self.assertRaises(ValueError):
            probe.run(self.repo, self.checkout, pre.tree, ["true"],
                      overlay={"../escaped.txt": "x"}, timeout=10)


class ExistingFilesChangeByEdits(Fixture):
    """The model sees a few hundred characters of a mechanism's configuration and
    none of its scripts.

    Regression: a `modify` carried complete new contents, so a proposal to add
    one line to CLAUDE.md deleted 34 others, and a rewritten check dropped an
    existing rule; both passed validation, which tests only the new property."""

    INSTRUCTIONS = ("# Orders\n\n## Security\n\n- Never log customer email addresses.\n"
                    + "".join(f"- Style rule {i}.\n" for i in range(30)))

    def interpret(self, files):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = files
        return selection.interpret(reply, INVENTORY, proposal._existing_paths(self.repo))

    def setUp(self):
        super().setUp()
        (self.repo / "CLAUDE.md").write_text(self.INSTRUCTIONS)

    def test_restating_an_existing_file_is_refused(self):
        out = self.interpret([{"path": "CLAUDE.md", "action": "modify",
                               "contents": "# Orders\n\n- Handlers never import db.\n"}])
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("give edits", out["failure"])

    def test_an_append_keeps_every_existing_line(self):
        out = self.interpret([{"path": "CLAUDE.md", "action": "modify",
                               "edits": [{"append": "- Handlers never import db."}]}])
        self.assertEqual(reasoning.SUCCESS, out["outcome"], out.get("failure"))
        self.assertEqual(self.INSTRUCTIONS + "- Handlers never import db.\n",
                         out["files"][0]["contents"])

    def test_a_replacement_changes_only_the_text_it_names(self):
        out = self.interpret([{"path": "CLAUDE.md", "action": "modify",
                               "edits": [{"find": "- Style rule 7.",
                                          "replace": "- Style rule 7, enforced."}]}])
        self.assertEqual(reasoning.SUCCESS, out["outcome"], out.get("failure"))
        self.assertEqual(self.INSTRUCTIONS.replace("- Style rule 7.", "- Style rule 7, enforced."),
                         out["files"][0]["contents"])

    def test_an_anchor_that_is_absent_or_repeated_is_refused(self):
        for find in ("- Style rule 99.", "- Style rule 1"):
            with self.subTest(find=find):
                out = self.interpret([{"path": "CLAUDE.md", "action": "modify",
                                       "edits": [{"find": find, "replace": "x"}]}])
                self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
                self.assertIn("exactly once", out["failure"])

    def test_edits_that_change_nothing_are_refused(self):
        out = self.interpret([{"path": "CLAUDE.md", "action": "modify",
                               "edits": [{"find": "# Orders", "replace": "# Orders"}]}])
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("change nothing", out["failure"])

    def test_a_malformed_edit_is_refused(self):
        for edit in ({"find": "", "replace": "x"}, {"append": "  "},
                     {"find": "# Orders", "replace": "x", "append": "y"}, "append"):
            with self.subTest(edit=edit):
                out = self.interpret([{"path": "CLAUDE.md", "action": "modify",
                                       "edits": [edit]}])
                self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])

    def test_the_approval_patch_names_what_a_change_removes(self):
        self.approved({"CLAUDE.md": self.INSTRUCTIONS.replace(
            "- Never log customer email addresses.\n", "")})
        rec = proposal.load(self.checkout, "prop_d1ec7000")
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            cli._print_patch(self.repo, self.checkout, rec)
        self.assertIn("REMOVES 1 existing line(s) from CLAUDE.md", printed.getvalue())


class TheProposalSaysWhatHappened(Fixture):
    """Regression: the printout led with the mechanism. What happened, the
    property it lacked and the evidence were in the record and never printed."""

    def dx(self):
        dx = json.loads(json.dumps(DIAGNOSIS))
        dx.update(evidence_for=["turn 2 replaced print() with the logger"],
                  evidence_against=["seen in one developer's sessions"])
        return dx

    def test_propose_prints_the_diagnosis_before_the_change(self):
        rec = self.propose()
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            cli._print_proposal(rec, self.dx())
        text = printed.getvalue()
        for expected in ("problem    the rule was unenforced",
                         "cause      MISSING_CONTEXT (high confidence)",
                         "property   output in src must go through the logger",
                         "seen in 3 sessions",
                         "for        turn 2 replaced print() with the logger",
                         "against    seen in one developer's sessions"):
            self.assertIn(expected, text)
        self.assertLess(text.index("problem"), text.index("mechanism"))

    def test_apply_repeats_it_at_the_moment_of_approval(self):
        self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            code = cli.cmd_apply(argparse.Namespace(path=str(self.repo), yes=False,
                                                    proposal_id="prop_d1ec7000"))
        self.assertEqual(3, code)
        self.assertIn("property   x", printed.getvalue())


class SelectionCanBeShownWithoutSending(Fixture):
    def test_show_prints_the_request_and_sends_nothing(self):
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        dx = json.loads(json.dumps(DIAGNOSIS))
        dx.update(checkout_id=self.checkout, schema="repohone.diagnosis/v1",
                  schema_version=1, contract_version=CONTRACT)
        from repohone import diagnosis as diagnosis_module
        original = diagnosis_module.load_all
        diagnosis_module.load_all = lambda checkout: [dx]
        self.addCleanup(setattr, diagnosis_module, "load_all", original)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.cmd_propose(argparse.Namespace(
                path=str(self.repo), yes=False, show=True, diagnosis_id=dx["diagnosis_id"],
                model=None, timeout=30, validate_timeout=30))
        self.assertEqual(0, code, err.getvalue())
        self.assertIn("RepoHone mechanism selection", out.getvalue())
        self.assertIn("Nothing was sent", err.getvalue())
        self.assertEqual([], proposal.load_all(self.checkout))


class ACommandThatCannotStart(Fixture):
    """A snapshot has no ignored files, so a check that lives in node_modules or a
    virtualenv exits 127 in the detached checkout.

    Regression: validation reported that such a check "already flags the
    problematic state", blaming the change for a missing dependency."""

    MISSING = ["sh", "-c", "node_modules/.bin/lint src"]
    GATED = ["sh", "-c", "if [ -f marker ]; then exit 127; fi; "
             "if [ -f rule.enabled ] && grep -q print src/app.py; then exit 1; fi"]

    def test_a_missing_command_is_unrunnable_not_attributed(self):
        pre, post, _ = self.pair()
        result = validation.validate(self.repo, self.checkout, self.MISSING, {},
                                     pre, post, timeout=30)
        self.assertFalse(result.performed)
        self.assertIsNone(result.attributable)
        self.assertFalse(result.passed)
        self.assertTrue(any("could not run in the detached checkout (exit 127" in n
                            for n in result.notes), result.notes)

    def test_a_failing_baseline_is_reported_in_the_command_s_own_words(self):
        pre, post, _ = self.pair()
        result = validation.validate(
            self.repo, self.checkout,
            ["sh", "-c", "echo 'No module named requests' >&2; exit 1"], {},
            pre, post, timeout=30)
        self.assertFalse(result.attributable)
        self.assertTrue(any("already fails on the problematic state" in n
                            and "No module named requests" in n
                            for n in result.notes), result.notes)

    def test_a_new_runner_that_cannot_start_is_not_credited(self):
        pre, post, _ = self.pair()
        result = validation.validate(self.repo, self.checkout, self.MISSING,
                                     {"Makefile": "check:\n\ttrue\n"}, pre, post,
                                     timeout=30, creates_runner=True)
        self.assertFalse(result.passed)
        self.assertIsNone(result.rejects_problematic)
        self.assertTrue(any("could not run" in n for n in result.notes))

    def test_an_occurrence_that_cannot_start_is_unevaluated(self):
        (self.repo / "marker").write_text("x\n")
        (self.repo / "src" / "app.py").write_text('def run():\n    print("x")\n')
        other_bad = self._tree()
        (self.repo / "src" / "app.py").write_text("def run():\n    return 2\n")
        other_good = self._tree()
        (self.repo / "marker").unlink()
        git(["reset", "-q", "--hard"], self.repo)
        pre, post, _ = self.pair()
        result = validation.validate(
            self.repo, self.checkout, self.GATED, {"rule.enabled": "on\n"}, pre, post,
            timeout=30, others=[("elsewhere", validation.State("stop", other_bad, None),
                                 validation.State("stop", other_good, None))])
        self.assertIsNone(result.occurrences[0].caught)
        self.assertIsNone(result.occurrences[0].false_positive)
        self.assertTrue(result.passed, result.notes)


class Attribution(Fixture):
    """Regression: validation credited the change for a detection it had not
    added. The mechanism is now run WITHOUT the change first."""

    def test_a_pre_existing_failure_is_not_credited_to_the_change(self):
        pre, post, _ = self.pair()
        # Fails on the problematic state for a reason the change has nothing to
        # do with, paired with a change that does nothing.
        result = validation.validate(
            self.repo, self.checkout,
            ["sh", "-c", "! grep -rq 'print(' src"],
            {"unrelated.txt": "a change that does nothing\n"},
            pre, post, timeout=30)
        self.assertFalse(result.attributable)
        self.assertFalse(result.passed)
        self.assertTrue(any("cannot show that the change is what catches the mistake" in n
                            for n in result.notes))

    def test_a_change_that_enables_the_detection_is_attributable(self):
        pre, post, _ = self.pair()
        overlay = {"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"}
        result = validation.validate(self.repo, self.checkout, MECHANISM["argv"],
                                     overlay, pre, post, timeout=30)
        self.assertTrue(result.attributable)
        self.assertTrue(result.passed)

    def test_a_change_that_detects_nothing_is_rejected_end_to_end(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"][0]["edits"] = [{"append": "# a change that enables nothing"}]
        result = self.propose(FakeEngine(reply))
        self.assertEqual("VALIDATION_REJECTED", result["outcome"])
        self.assertTrue(result["validation"]["attributable"],
                        "the baseline passed, so the change was given its chance")
        self.assertFalse(result["validation"]["rejects_problematic"])
        self.assert_conformant(result)

    def test_attribution_travels_in_the_record(self):
        rec = self.propose()
        self.assertTrue(rec["validation"]["attributable"])


class EveryOccurrence(Fixture):
    """Regression: recurrence justified acting, then validation used one session."""

    def _main_occurrence(self, repository_id=None):
        session_id = ensure_session(self.checkout, "main", repository_id)
        self._main_session = session_id
        return canonical_occurrence(
            self.checkout, repository_id, "MISSING_CONTEXT", "output-handling",
            "print-instead-of-logger", session_id, corrective_turn=2)

    def _register(self, host_session_id, bad_tree, good_tree, with_diagnosis=True,
                  owner=None, repository_id=None):
        from repohone import identity
        owner = owner or self.checkout
        session_id = identity.session_id("claude-code", host_session_id)
        state.initialize(owner)
        rec = record.new_record(session_id, "claude-code", host_session_id, "claude-code",
                                "0.4.0", "redacted",
                                {"root": str(self.repo), "id": repository_id,
                                 "checkout_id": owner, "start_head": None,
                                 "start_branch": None, "branches_observed": []},
                                None, "claude-code", None, record.now(), str(self.repo))
        when = record.now()

        def snapshot(cls, ordinal, tree):
            segment = cls.replace("_", "-")
            return {"ref": (f"refs/repohone/snapshots/{session_id}/"
                            f"{segment}/{ordinal}"),
                    "class": cls, "ordinal": ordinal, "commit": tree,
                    "tree": tree, "head": None, "branch": None,
                    "taken_at": when, "path": None, "submodules": []}

        def turn(index, prompt_tree, stop_tree):
            value = record.new_turn(index, f"t{index}", None, None, None)
            value["prompt_events"].append({
                "ordinal": 1, "at": when,
                "content": {"text": "work", "sha256": "a" * 64,
                            "chars": 4, "redactions": 0},
                "snapshot": snapshot("prompt", 1, prompt_tree)})
            value["stop_events"].append({
                "ordinal": 1, "at": when, "kind": "stop", "error": None,
                "snapshot": snapshot("stop", 1, stop_tree),
                "last_assistant_message": None, "effort": None, "model": None,
                "host_continuation": None})
            value.update({"completion": "stop", "completion_source": "observed",
                          "completed_at": when})
            return value

        rec["turns"] = [turn(1, bad_tree, bad_tree), turn(2, bad_tree, good_tree)]
        with record.open_record(owner, session_id, lambda: rec):
            pass
        canonical_occurrence(
            owner, repository_id, "MISSING_CONTEXT", "output-handling",
            "print-instead-of-logger", session_id,
            corrective_turn=2 if with_diagnosis else None)
        return session_id

    def test_a_sibling_checkouts_accepted_state_can_reject_the_candidate(self):
        from repohone import fingerprint, identity
        repository_id = identity.repository_id("git@github.com:Acme/App.git")
        self._main_occurrence(repository_id)
        sibling = self._register("sibling-session", self.good_tree, self.bad_tree,
                                 owner="sibling-checkout", repository_id=repository_id)
        mark = fingerprint.known(self.checkout, repository_id)[0]
        diagnosis = json.loads(json.dumps(DIAGNOSIS))
        diagnosis["fingerprint"]["fingerprint_id"] = mark.fingerprint_id
        diagnosis["session_id"] = self._main_session
        result = proposal.propose(self.repo, self.checkout, diagnosis, INVENTORY,
                                  self.session(), FakeEngine(), timeout=30)
        self.assertEqual(reasoning.VALIDATION_REJECTED, result["outcome"])
        self.assertEqual([sibling],
                         [o["session_id"] for o in result["validation"]["occurrences"]])
        self.assertEqual("sibling-checkout",
                         result["validation"]["occurrences"][0]["checkout_id"])
        self.assertTrue(result["validation"]["occurrences"][0]["false_positive"])
        if VALIDATOR is not None:
            self.assertEqual([], list(VALIDATOR.iter_errors(result)))

    def test_an_occurrence_without_a_boundary_is_unevaluable_not_guessed(self):
        from repohone import fingerprint
        self._main_occurrence()
        other_session = self._register("no-boundary", self.bad_tree, self.good_tree,
                                       with_diagnosis=False)
        mark = fingerprint.known(self.checkout, None)[0]
        diagnosis = json.loads(json.dumps(DIAGNOSIS))
        diagnosis["fingerprint"]["fingerprint_id"] = mark.fingerprint_id
        diagnosis["session_id"] = self._main_session
        result = proposal.propose(self.repo, self.checkout, diagnosis, INVENTORY,
                                  self.session(), FakeEngine(), timeout=30)
        other = [o for o in result["validation"]["occurrences"]
                 if o["session_id"] == other_session]
        self.assertTrue(other)
        self.assertIsNone(other[0]["caught"], "an unidentified boundary was guessed")
        # Regression: an unevaluable occurrence failed the whole proposal, so a
        # pattern became unproposable the moment it recurred.
        self.assertEqual(reasoning.SUCCESS, result["outcome"],
                         result["validation"]["notes"])
        self.assertTrue(any("could not be evaluated" in n
                            for n in result["validation"]["notes"]))

    def test_other_occurrences_are_validated_too(self):
        from repohone import fingerprint
        self._main_occurrence()
        other_session = self._register("other-occurrence", self.bad_tree, self.good_tree)
        mark = fingerprint.known(self.checkout, None)[0]
        diagnosis = json.loads(json.dumps(DIAGNOSIS))
        diagnosis["fingerprint"]["fingerprint_id"] = mark.fingerprint_id
        diagnosis["session_id"] = self._main_session
        result = proposal.propose(self.repo, self.checkout, diagnosis, INVENTORY,
                                  self.session(), FakeEngine(), timeout=30)
        occurrences = result["validation"]["occurrences"]
        self.assertEqual([other_session],
                         [o["session_id"] for o in occurrences])
        self.assertTrue(occurrences[0]["caught"])
        self.assertFalse(occurrences[0]["false_positive"])
        self.assertTrue(any("other captured occurrence" in n
                            for n in result["validation"]["notes"]))

    def test_an_unreadable_sibling_stops_proposal_validation(self):
        from repohone import paths
        repository_id = identity.repository_id("git@github.com:Acme/App.git")
        mark, _ = self._main_occurrence(repository_id)
        broken = paths.checkout_dir("unreadable-proposal-sibling") / "diagnoses"
        broken.mkdir(parents=True)
        (broken / "dx_0000000000000000.json").write_text("{ not json")
        diagnosed = json.loads(json.dumps(DIAGNOSIS))
        diagnosed["fingerprint"]["fingerprint_id"] = mark.fingerprint_id
        diagnosed["session_id"] = self._main_session

        engine = FakeEngine()
        result = proposal.propose(self.repo, self.checkout, diagnosed, INVENTORY,
                                  self.session(), engine, timeout=30)

        self.assertEqual(reasoning.INSUFFICIENT_EVIDENCE, result["outcome"])
        self.assertIn("recurrence history is incomplete",
                      result["failure"]["reason"])
        self.assertEqual(0, engine.calls)
        self.assertIsNone(proposal.staged(self.checkout, result["proposal_id"]))

    def test_a_false_positive_on_accepted_work_fails_the_candidate(self):
        pre, post, _ = self.pair()
        overlay = {"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"}
        others = [("rh_bbbbbbbbbbbbbbbb", pre, post)]
        result = validation.validate(
            self.repo, self.checkout,
            ["sh", "-c", "! grep -qs T20 pyproject.toml ruff.toml"],
            overlay, pre, post, timeout=30, others=others)
        self.assertTrue(result.false_positives)
        self.assertFalse(result.passed)

    def test_coverage_is_reported(self):
        pre, post, _ = self.pair()
        overlay = {"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"}
        others = [("rh_cccccccccccccccc", pre, post)]
        result = validation.validate(self.repo, self.checkout, MECHANISM["argv"],
                                     overlay, pre, post, timeout=30, others=others)
        self.assertEqual((1, 1), result.coverage)


class ApplySafety(Fixture):
    """Regression: applying silently overwrote an edit made after validation."""

    def test_applying_refuses_when_the_target_changed(self):
        rec = self.propose()
        (self.repo / "pyproject.toml").write_text("# edited after validation\n")
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        with self.assertRaises(proposal.Conflict) as caught:
            proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        self.assertIn("changed since this proposal was validated", str(caught.exception))
        self.assertEqual("# edited after validation\n",
                         (self.repo / "pyproject.toml").read_text())

    def test_the_baseline_hash_is_recorded(self):
        rec = self.propose()
        entry = rec["change"]["files"][0]
        self.assertIsNotNone(entry["baseline_sha256"])

    def test_a_created_file_records_no_baseline(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = [{"path": "ruff.toml", "action": "create",
                                     "contents": "select = ['T20']\n"}]
        rec = self.propose(FakeEngine(reply))
        self.assertIsNone(rec["change"]["files"][0]["baseline_sha256"])


class AtomicApply(Fixture):
    """Regression: a bad second file was written after the first had landed, and
    the record still said `candidate` — so rollback refused a changed tree."""

    def setUp(self):
        super().setUp()
        for name in "abc":
            (self.repo / f"{name}.txt").write_text(f"{name.upper()} base\n")

    def two_file_proposal(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = [
            {"path": "pyproject.toml", "action": "modify",
             "edits": [{"find": "select = ['E']", "replace": "select = ['E', 'T20']"}]},
            {"path": "ruff.toml", "action": "create", "contents": "line-length = 100\n"}]
        rec = self.propose(FakeEngine(reply))
        self.assertEqual(2, len(rec["change"]["files"]))
        return rec

    def test_a_corrupt_staged_file_changes_nothing(self):
        rec = self.two_file_proposal()
        before = (self.repo / "pyproject.toml").read_text()
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        contents["ruff.toml"] = "not what was reviewed\n"
        with self.assertRaises(proposal.Conflict) as caught:
            proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        self.assertIn("nothing was changed", str(caught.exception))
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())
        self.assertFalse((self.repo / "ruff.toml").exists())
        self.assertEqual([], proposal.load(self.checkout,
                                           rec["proposal_id"])["rollback"]["files"])

    def test_a_failed_write_restores_what_already_landed(self):
        rec = self.two_file_proposal()
        before = (self.repo / "pyproject.toml").read_text()
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        with _failing_write(self.repo, "ruff.toml"):
            with self.assertRaises(proposal.Conflict) as caught:
                proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        self.assertIn("the repository is as it was", str(caught.exception))
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())
        self.assertEqual("candidate",
                         proposal.load(self.checkout, rec["proposal_id"])["state"])

    def test_an_unrecoverable_failure_records_exactly_what_landed(self):
        rec = self.two_file_proposal()
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        with _failing_write(self.repo, "ruff.toml", also_restore_of="pyproject.toml"):
            with self.assertRaises(proposal.Conflict):
                proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        stored = proposal.load(self.checkout, rec["proposal_id"])
        self.assertEqual("partially_applied", stored["state"])
        self.assertEqual(["pyproject.toml"],
                         [f["path"] for f in stored["rollback"]["files"]])
        self.assertIn("T20", (self.repo / "pyproject.toml").read_text())
        self.assert_conformant(stored)

    def test_only_files_still_changed_are_listed_as_pending(self):
        """Three files: A and B land, C fails, B cannot be restored. A came back,
        so listing it would make a later rollback reject its own restored text."""
        rec = self.approved({"a.txt": "A new\n", "b.txt": "B new\n",
                             "c.txt": "C new\n"}, proposal_id="prop_70000003")
        with _failing_write(self.repo, "c.txt", also_restore_of="b.txt"):
            with self.assertRaises(proposal.Conflict):
                self.apply_it(rec)
        stored = proposal.load(self.checkout, "prop_70000003")
        self.assertEqual(["b.txt"],
                         [f["path"] for f in stored["rollback"]["files"]])
        self.assertEqual("A base\n", (self.repo / "a.txt").read_text())
        proposal.rollback(self.repo, self.checkout, "prop_70000003")
        self.assertEqual("B base\n", (self.repo / "b.txt").read_text())

    def test_rollback_recovers_a_partially_applied_proposal(self):
        rec = self.two_file_proposal()
        before = (self.repo / "pyproject.toml").read_text()
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        with _failing_write(self.repo, "ruff.toml", also_restore_of="pyproject.toml"):
            with self.assertRaises(proposal.Conflict):
                proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())

    def test_the_cli_reports_a_conflict_instead_of_raising(self):
        rec = self.two_file_proposal()
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        (self.repo / "pyproject.toml").write_text("# edited after validation\n")
        args = argparse.Namespace(path=str(self.repo), proposal_id=rec["proposal_id"],
                                  yes=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.cmd_apply(args)
        self.assertEqual(4, code)
        self.assertIn("changed since this proposal was validated", err.getvalue())


@contextlib.contextmanager
def _failing_write(root, fails, also_restore_of=None):
    """Injects an OSError on one target write, and optionally on the restore of
    another, to reach the double fault where recovery itself cannot complete."""
    real = proposal._replace_inside
    seen = {}

    def flaky(repo_root, relative, text, **kw):
        name = os.path.basename(relative)
        if name == fails:
            raise OSError("injected write failure")
        if name == also_restore_of:
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                raise OSError("injected restore failure")
        return real(repo_root, relative, text, **kw)

    proposal._replace_inside = flaky
    try:
        yield
    finally:
        proposal._replace_inside = real


class ApplyStaysInsideTheRepository(Fixture):
    """Regression: a lexical `..` check passed a symlink, and apply wrote through
    it to a file outside the repository."""

    def test_a_symlinked_target_is_refused(self):
        rec = self.propose()
        outside = self.tmp / "OUTSIDE.txt"
        outside.write_text((self.repo / "pyproject.toml").read_text())
        (self.repo / "pyproject.toml").unlink()
        (self.repo / "pyproject.toml").symlink_to(outside)
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        with self.assertRaises(proposal.Conflict) as caught:
            proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        self.assertIn("symbolic link", str(caught.exception))
        self.assertEqual("[tool.ruff.lint]\nselect = ['E']\n", outside.read_text())

    def test_a_symlinked_parent_directory_is_refused(self):
        outside = self.tmp / "outside-dir"
        outside.mkdir()
        (self.repo / "conf").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(proposal.Conflict) as caught:
            proposal._write_inside(self.repo, "conf/ruff.toml", "x\n")
        self.assertIn("symbolic link", str(caught.exception))
        self.assertEqual([], list(outside.iterdir()))

    def test_a_path_escaping_the_repository_is_refused(self):
        with self.assertRaises(proposal.Conflict):
            proposal.safe_target(self.repo, "../escaped.txt")

    def test_safe_target_accepts_an_ordinary_path(self):
        target = proposal.safe_target(self.repo, "src/app.py")
        self.assertEqual((self.repo / "src" / "app.py").resolve(), target.resolve())

    def test_a_parent_swap_cannot_create_an_external_directory(self):
        (self.repo / "conf" / "sub").mkdir(parents=True)
        rec = self.approved({"conf/sub/new.txt": "new\n"},
                            proposal_id="prop_a1e00001")
        outside = self.tmp / "outside-parent"
        outside.mkdir()
        moved = self.tmp / "moved-conf"
        real = proposal.safe_target
        calls = 0

        def swap_before_write(root, relative):
            nonlocal calls
            result = real(root, relative)
            calls += 1
            if calls == 2:
                shutil.move(str(self.repo / "conf"), str(moved))
                (self.repo / "conf").symlink_to(outside, target_is_directory=True)
            return result

        proposal.safe_target = swap_before_write
        try:
            with self.assertRaises(proposal.Conflict):
                self.apply_it(rec)
        finally:
            proposal.safe_target = real

        self.assertEqual([], list(outside.iterdir()),
                         "apply created a directory outside the repository")

    def test_a_future_proposal_version_cannot_change_the_project(self):
        target = self.repo / "future.txt"
        rec = self.approved({"future.txt": "landed\n"},
                            proposal_id="prop_f0700001")
        rec["schema_version"] = 99
        artifacts.atomic_write(proposal.path_for(self.checkout, rec["proposal_id"]), rec)

        with self.assertRaises(record.UnsupportedVersion):
            self.apply_it(rec)

        self.assertFalse(target.exists())


class ApplyResistsASwapAtTheWriteBoundary(Fixture):
    """Regression: the path was checked and then opened by name, so a link
    substituted in between was followed out of the repository."""

    def swap_at_the_window(self, rec, kind):
        """Replaces the target the instant after its baseline is read — the exact
        window between the check and the write."""
        real = proposal._read_inside
        outside_dir = self.tmp / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside = outside_dir / "pyproject.toml"
        outside.write_text((self.repo / "pyproject.toml").read_text())

        def once(root, relative):
            out = real(root, relative)
            if relative == "pyproject.toml":
                proposal._read_inside = real
                (self.repo / "pyproject.toml").unlink()
                if kind == "file":
                    (self.repo / "pyproject.toml").symlink_to(outside)
            return out

        proposal._read_inside = once
        try:
            proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                           proposal.staged(self.checkout, rec["proposal_id"]))
        finally:
            proposal._read_inside = real
        return outside

    def test_a_file_swapped_to_a_link_is_refused(self):
        rec = self.propose()
        with self.assertRaises(proposal.Conflict) as caught:
            self.swap_at_the_window(rec, "file")
        self.assertIn("symbolic link", str(caught.exception))

    def test_the_outside_file_is_untouched(self):
        rec = self.propose()
        outside = None
        try:
            outside = self.swap_at_the_window(rec, "file")
        except proposal.Conflict:
            outside = self.tmp / "outside" / "pyproject.toml"
        self.assertEqual("[tool.ruff.lint]\nselect = ['E']\n", outside.read_text())

    def test_success_is_never_recorded_for_a_swapped_write(self):
        rec = self.propose()
        with self.assertRaises(proposal.Conflict):
            self.swap_at_the_window(rec, "file")
        stored = proposal.load(self.checkout, rec["proposal_id"])
        self.assertNotIn(stored["state"], ("observing", "applied"))

    def test_a_symlinked_parent_is_refused_at_the_open(self):
        outside = self.tmp / "outside-dir"
        outside.mkdir()
        (self.repo / "conf").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(proposal.Conflict) as caught:
            proposal._write_inside(self.repo, "conf/x.toml", "hello\n")
        self.assertIn("symbolic link", str(caught.exception))
        self.assertEqual([], list(outside.iterdir()))

    def test_writing_through_a_link_is_refused(self):
        outside = self.tmp / "target.txt"
        outside.write_text("original\n")
        (self.repo / "link.txt").symlink_to(outside)
        with self.assertRaises(proposal.Conflict) as caught:
            proposal._write_inside(self.repo, "link.txt", "overwritten\n")
        self.assertIn("symbolic link", str(caught.exception))
        self.assertEqual("original\n", outside.read_text())

    def test_reading_through_a_link_is_refused(self):
        outside = self.tmp / "secret.txt"
        outside.write_text("private\n")
        (self.repo / "link.txt").symlink_to(outside)
        with self.assertRaises(proposal.Conflict):
            proposal._read_inside(self.repo, "link.txt")

    def test_an_ordinary_file_still_round_trips(self):
        proposal._write_inside(self.repo, "src/new.py", "x = 1\n")
        self.assertEqual("x = 1\n", proposal._read_inside(self.repo, "src/new.py"))
        self.assertIsNone(proposal._read_inside(self.repo, "src/absent.py"))


class WritesTouchOnlyTheRepositoryEntry(Fixture):
    """Regression: writes truncated the inode, so every other name for it — a
    hard link outside the repository included — changed with it. And rollback
    used plain path operations, so it followed a link straight out."""

    def a_proposal(self):
        return self.propose()

    def test_a_hard_linked_file_outside_is_not_rewritten(self):
        rec = self.a_proposal()
        outside = self.tmp / "shared.toml"
        (self.repo / "pyproject.toml").rename(outside)
        os.link(outside, self.repo / "pyproject.toml")
        before = outside.read_text()
        proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                       proposal.staged(self.checkout, rec["proposal_id"]))
        self.assertIn("T20", (self.repo / "pyproject.toml").read_text())
        self.assertEqual(before, outside.read_text(),
                         "the change reached a hard link outside the repository")

    def test_rollback_refuses_to_follow_a_link_out(self):
        rec = self.a_proposal()
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        outside = self.tmp / "escaped.toml"
        outside.write_text((self.repo / "pyproject.toml").read_text())
        (self.repo / "pyproject.toml").unlink()
        (self.repo / "pyproject.toml").symlink_to(outside)
        applied = outside.read_text()
        with self.assertRaises(proposal.Conflict):
            proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertEqual(applied, outside.read_text())

    def test_rollbacks_check_does_not_read_through_a_link(self):
        """Distinguishes the two guards: an unconfined pre-check would read the
        outside file and complain that the content changed, rather than that the
        path is a link at all."""
        rec = self.a_proposal()
        proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                       proposal.staged(self.checkout, rec["proposal_id"]))
        outside = self.tmp / "elsewhere.toml"
        outside.write_text("something else entirely\n")
        (self.repo / "pyproject.toml").unlink()
        (self.repo / "pyproject.toml").symlink_to(outside)
        with self.assertRaises(proposal.Conflict) as caught:
            proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertIn("symbolic link", str(caught.exception))
        self.assertEqual("something else entirely\n", outside.read_text())

    def test_rollbacks_restore_refuses_a_link_swapped_in_after_the_check(self):
        """Isolates the write guard: the link appears after the pre-check has
        already passed."""
        rec = self.a_proposal()
        proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                       proposal.staged(self.checkout, rec["proposal_id"]))
        outside = self.tmp / "late.toml"
        outside.write_text((self.repo / "pyproject.toml").read_text())
        real = proposal._read_inside

        def swap_after_check(root, relative):
            out = real(root, relative)
            if relative == "pyproject.toml":
                proposal._read_inside = real
                (self.repo / "pyproject.toml").unlink()
                (self.repo / "pyproject.toml").symlink_to(outside)
            return out

        proposal._read_inside = swap_after_check
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        finally:
            proposal._read_inside = real
        self.assertIn("symbolic link", str(caught.exception))
        self.assertIn("T20", outside.read_text(),
                      "the backup was written through the link")

    def test_rollback_still_works_normally(self):
        rec = self.a_proposal()
        before = (self.repo / "pyproject.toml").read_text()
        proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                       proposal.staged(self.checkout, rec["proposal_id"]))
        proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())

    def test_rollback_refuses_when_the_applied_file_was_edited(self):
        rec = self.a_proposal()
        proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                       proposal.staged(self.checkout, rec["proposal_id"]))
        (self.repo / "pyproject.toml").write_text("someone edited this\n")
        with self.assertRaises(proposal.Conflict):
            proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertEqual("someone edited this\n",
                         (self.repo / "pyproject.toml").read_text())


class ApplyUnderAConcurrentWriter(Fixture):
    """ARCHITECTURE §27.1. Apply assumes a quiescent checkout; where it cannot
    prevent a concurrent change it must notice and say so, never report a clean
    apply it did not achieve."""

    def a_proposal(self):
        return self.approved({"pyproject.toml":
                              "[tool.ruff.lint]\nselect = ['E', 'T20']\n"})

    def test_an_edit_before_the_commit_is_caught(self):
        rec = self.a_proposal()
        real = os.fchmod

        def edit_then(fd, mode):
            os.fchmod = real
            (self.repo / "pyproject.toml").write_text("SOMEONE ELSE\n")
            return real(fd, mode)

        os.fchmod = edit_then
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                self.apply_it(rec)
        finally:
            os.fchmod = real
        self.assertIn("changed", str(caught.exception))
        self.assertEqual("SOMEONE ELSE\n", (self.repo / "pyproject.toml").read_text())

    def test_a_parent_moved_out_of_the_tree_is_caught(self):
        rec = self.approved({"src/app.py": "REPOHONE WROTE THIS\n"})
        moved = self.tmp / "moved-away"
        real = os.fchmod

        def move_then(fd, mode):
            os.fchmod = real
            shutil.move(str(self.repo / "src"), str(moved))
            return real(fd, mode)

        os.fchmod = move_then
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                self.apply_it(rec)
        finally:
            os.fchmod = real
        self.assertIn("moved out of the repository", str(caught.exception))
        self.assertNotIn("REPOHONE WROTE THIS", (moved / "app.py").read_text())

    def test_a_relocation_inside_the_rename_is_still_caught(self):
        """The last resort: the move lands after every pre-check, so only
        re-reading the landed file from the root notices it."""
        rec = self.approved({"src/app.py": "REPOHONE WROTE THIS\n"})
        moved = self.tmp / "gone-mid-rename"
        real = os.rename

        def move_then_rename(src, dst, **kw):
            os.rename = real
            shutil.move(str(self.repo / "src"), str(moved))
            return real(src, dst, **kw)

        os.rename = move_then_rename
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                self.apply_it(rec)
        finally:
            os.rename = real
        self.assertIn("moved", str(caught.exception))
        self.assertFalse((self.repo / "src" / "app.py").exists())

    def test_a_disturbed_tree_is_never_a_clean_apply(self):
        """An unrelated file changing during the apply is reported, and what
        landed is still recorded so it can be undone."""
        rec = self.a_proposal()
        real = proposal._replace_inside

        def meddle(root, relative, text, **kw):
            out = real(root, relative, text, **kw)
            (self.repo / "src" / "other.py").write_text("touched by someone\n")
            return out

        proposal._replace_inside = meddle
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                self.apply_it(rec)
        finally:
            proposal._replace_inside = real
        self.assertIn("working tree changed", str(caught.exception))
        stored = proposal.load(self.checkout, rec["proposal_id"])
        self.assertEqual("partially_applied", stored["state"])
        self.assertTrue(stored["rollback"]["files"], "what landed was not recorded")

    def test_an_undisturbed_apply_is_clean(self):
        rec = self.a_proposal()
        self.assertEqual("observing", self.apply_it(rec)["state"])


class InterruptionLeavesARecoverableRecord(Fixture):
    """ARCHITECTURE §27.1. A process killed mid-apply leaves files changed; the
    record has to be able to say which, or the change cannot be undone."""

    def two_files(self):
        for name in "ab":
            (self.repo / f"{name}.txt").write_text(f"{name} base\n")
        return self.approved({"a.txt": "a new\n", "b.txt": "b new\n"},
                             proposal_id="prop_001c1110")

    def die_after(self, rec, path):
        """Forks so the child really dies, as a killed process would."""
        child = os.fork()
        if child == 0:
            real = proposal._replace_inside

            def die(root, relative, text, **kw):
                out = real(root, relative, text, **kw)
                if relative == path:
                    os._exit(9)
                return out

            proposal._replace_inside = die
            try:
                self.apply_it(rec)
            except BaseException:
                pass
            os._exit(0)
        os.waitpid(child, 0)

    def test_the_record_marks_the_attempt_before_touching_anything(self):
        rec = self.two_files()
        self.die_after(rec, "a.txt")
        stored = proposal.load(self.checkout, "prop_001c1110")
        self.assertEqual("applying", stored["state"])
        self.assertEqual({"a.txt", "b.txt"},
                         {f["path"] for f in stored["rollback"]["files"]})

    def test_reconciliation_finds_exactly_what_landed(self):
        rec = self.two_files()
        self.die_after(rec, "a.txt")
        settled = proposal.reconcile(self.repo, self.checkout, "prop_001c1110")
        self.assertEqual("partially_applied", settled["state"])
        self.assertEqual(["a.txt"], [f["path"] for f in settled["rollback"]["files"]])

    def test_the_interrupted_change_can_be_undone(self):
        rec = self.two_files()
        self.die_after(rec, "a.txt")
        proposal.rollback(self.repo, self.checkout, "prop_001c1110")
        self.assertEqual("a base\n", (self.repo / "a.txt").read_text())
        self.assertEqual("b base\n", (self.repo / "b.txt").read_text())

    def test_dying_before_any_write_reconciles_to_untouched(self):
        rec = self.two_files()
        child = os.fork()
        if child == 0:
            proposal._replace_inside = lambda *a, **k: os._exit(9)
            try:
                self.apply_it(rec)
            except BaseException:
                pass
            os._exit(0)
        os.waitpid(child, 0)
        settled = proposal.reconcile(self.repo, self.checkout, "prop_001c1110")
        self.assertEqual("approved", settled["state"])
        self.assertEqual("a base\n", (self.repo / "a.txt").read_text())

    def test_an_unfinished_staging_file_is_swept(self):
        """Killed between staging and the rename, `.rh-<pid>-name` is left in the
        project: an unapproved file the record called 'nothing changed'."""
        rec = self.two_files()
        child = os.fork()
        if child == 0:
            try:
                os.rename = lambda *a, **k: os._exit(9)
                self.apply_it(rec)
            except BaseException:
                pass
            finally:
                os._exit(0)
        os.waitpid(child, 0)
        stages = [p.name for p in self.repo.iterdir() if p.name.startswith(".rh-")]
        self.assertTrue(stages, "fixture did not leave a staging file")
        settled = proposal.reconcile(self.repo, self.checkout, "prop_001c1110")
        self.assertEqual([], [p.name for p in self.repo.iterdir()
                              if p.name.startswith(".rh-")])
        self.assertIn("staging file", settled["rollback"]["instructions"])

    def test_a_rollback_killed_after_its_last_restore_is_not_called_untouched(self):
        rec = self.two_files()
        self.apply_it(rec)
        child = os.fork()
        if child == 0:
            try:
                real = proposal.save

                def die_on_final(checkout_id, record_):
                    if record_["state"] == "rejected":
                        os._exit(9)
                    return real(checkout_id, record_)

                proposal.save = die_on_final
                proposal.rollback(self.repo, self.checkout, "prop_001c1110")
            except BaseException:
                pass
            finally:
                os._exit(0)
        os.waitpid(child, 0)
        settled = proposal.reconcile(self.repo, self.checkout, "prop_001c1110")
        self.assertEqual("rejected", settled["state"])
        self.assertIn("already restored", settled["rollback"]["instructions"])

    def test_an_uncertain_interrupted_rollback_is_described_as_undoing(self):
        rec = self.two_files()
        self.apply_it(rec)
        stored = proposal.load(self.checkout, "prop_001c1110")
        stored["state"] = "rolling_back"
        proposal.save(self.checkout, stored)
        (self.repo / "a.txt").write_text("changed by someone else\n")
        settled = proposal.reconcile(self.repo, self.checkout, "prop_001c1110")
        self.assertEqual("unresolved", settled["state"])
        self.assertIn("interrupted while undoing",
                      settled["rollback"]["instructions"])

    def test_a_record_is_never_left_half_written(self):
        rec = self.two_files()
        self.die_after(rec, "a.txt")
        raw = proposal.path_for(self.checkout, "prop_001c1110").read_text()
        self.assertEqual("prop_001c1110", json.loads(raw)["proposal_id"])


class EveryUncertainLandingIsUnresolved(Fixture):
    """Regression: only `Escaped` was treated as unresolved. A post-rename walk
    that raised an ordinary Conflict — the moved parent replaced by a symlink —
    was recovered as if nothing had escaped, and the record said `approved`."""

    def move_and_symlink(self, rec, moved):
        real = os.rename

        def swap(src, dst, **kw):
            if "b.txt" in str(dst):
                os.rename = real
                shutil.move(str(self.repo / "conf"), str(moved))
                (self.repo / "conf").symlink_to(moved, target_is_directory=True)
            return real(src, dst, **kw)

        os.rename = swap
        try:
            with self.assertRaises(proposal.Escaped):
                self.apply_it(rec)
        finally:
            os.rename = real

    def test_a_symlinked_parent_after_the_rename_is_unresolved(self):
        (self.repo / "conf").mkdir(exist_ok=True)
        (self.repo / "conf" / "b.txt").write_text("b base\n")
        (self.repo / "a.txt").write_text("a base\n")
        rec = self.approved({"a.txt": "a new\n", "conf/b.txt": "b new\n"},
                            proposal_id="prop_5aa00001")
        self.move_and_symlink(rec, self.tmp / "swapped-out")
        stored = proposal.load(self.checkout, "prop_5aa00001")
        self.assertEqual("unresolved", stored["state"])
        self.assertIn("conf/b.txt", stored.get("unresolved_paths") or [])
        self.assertNotIn("as it was", stored["rollback"]["instructions"])

    def test_the_backup_survives_that_landing(self):
        (self.repo / "conf").mkdir(exist_ok=True)
        (self.repo / "conf" / "b.txt").write_text("b base\n")
        (self.repo / "a.txt").write_text("a base\n")
        rec = self.approved({"a.txt": "a new\n", "conf/b.txt": "b new\n"},
                            proposal_id="prop_5aa00002")
        self.move_and_symlink(rec, self.tmp / "swapped-out2")
        backup = proposal.backup_dir(self.checkout, "prop_5aa00002")
        self.assertEqual("b base\n", (backup / "conf" / "b.txt").read_text())


class RecoveryTouchesOnlyItsOwnStaging(Fixture):
    """Regression: the sweep unlinked every `.rh-*-<name>` sibling, and deleted a
    developer's own `.rh-my-notes-a.txt`."""

    def killed_mid_apply(self):
        (self.repo / "a.txt").write_text("a base\n")
        (self.repo / "b.txt").write_text("b base\n")
        rec = self.approved({"a.txt": "a new\n", "b.txt": "b new\n"},
                            proposal_id="prop_5ee00001")
        child = os.fork()
        if child == 0:
            try:
                os.rename = lambda *a, **k: os._exit(9)
                self.apply_it(rec)
            except BaseException:
                pass
            finally:
                os._exit(0)
        os.waitpid(child, 0)
        return rec

    def test_an_unrelated_dot_rh_file_is_left_alone(self):
        mine = self.repo / ".rh-my-notes-a.txt"
        mine.write_text("my own notes\n")
        self.killed_mid_apply()
        proposal.reconcile(self.repo, self.checkout, "prop_5ee00001")
        self.assertTrue(mine.exists(), "recovery deleted a developer's own file")
        self.assertEqual("my own notes\n", mine.read_text())

    def test_its_own_staging_file_is_still_removed(self):
        self.killed_mid_apply()
        stored = proposal.load(self.checkout, "prop_5ee00001")
        token = stored["stage_token"]
        staged = self.repo / proposal.stage_name(token, "a.txt")
        self.assertTrue(staged.exists(), "fixture left no staging file")
        proposal.reconcile(self.repo, self.checkout, "prop_5ee00001")
        self.assertFalse(staged.exists())

    def test_the_token_is_recorded_before_any_write(self):
        self.killed_mid_apply()
        self.assertTrue(proposal.load(self.checkout, "prop_5ee00001")["stage_token"])


class ApplyNoticesWhatDisappears(Fixture):
    """Regression: only newly appearing paths counted, so deleting an unrelated
    file during apply produced a clean `observing`."""

    def test_deleting_an_unrelated_untracked_file_is_a_disturbance(self):
        """An untracked file vanishes from the status listing entirely when it is
        deleted, so only comparing both directions notices it."""
        spare = self.repo / "spare.txt"
        spare.write_text("keep me\n")
        rec = self.approved({"pyproject.toml":
                             "[tool.ruff.lint]\nselect = ['E', 'T20']\n"})
        real = proposal._replace_inside

        def meddle(root, relative, text, **kw):
            out = real(root, relative, text, **kw)
            spare.unlink()
            return out

        proposal._replace_inside = meddle
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                self.apply_it(rec)
        finally:
            proposal._replace_inside = real
        self.assertIn("working tree changed", str(caught.exception))


class ValidationSeesIgnoredArtifacts(Fixture):
    """Regression: the comparison used Git status, which omits ignored paths —
    exactly where build output and caches land."""

    def test_an_ignored_artifact_written_by_the_run_is_detected(self):
        (self.repo / ".gitignore").write_text("build/\n")
        (self.repo / "build").mkdir(exist_ok=True)
        (self.repo / "build" / "cache.txt").write_text("c1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "ignore"], self.repo)
        before = probe.tree_fingerprint(self.repo, deep=True)
        (self.repo / "build" / "cache.txt").write_text("written by a run\n")
        self.assertNotEqual(before, probe.tree_fingerprint(self.repo, deep=True),
                            "an ignored artifact changed without being noticed")

    def _with_ignored_build(self):
        (self.repo / ".gitignore").write_text("build/\n")
        (self.repo / "build").mkdir(exist_ok=True)
        (self.repo / "build" / "cache.txt").write_text("c1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "ignore"], self.repo)
        return self.repo / "build" / "cache.txt"

    def test_an_ignored_artifact_of_the_same_size_is_still_detected(self):
        """Ignored paths trade content hashing for size and mtime; a same-size
        rewrite is the case that trade could have lost."""
        artifact = self._with_ignored_build()
        before = probe.tree_fingerprint(self.repo, deep=True)
        artifact.write_text("c2\n")
        self.assertEqual(len("c1\n"), len(artifact.read_text()))
        self.assertNotEqual(before, probe.tree_fingerprint(self.repo, deep=True))

    def test_an_ignored_path_is_not_read(self):
        self._with_ignored_build()
        digests = probe.tree_fingerprint(self.repo, deep=True)[2]
        ignored = [v for k, v in digests.items() if k.startswith("build")]
        self.assertTrue(ignored, digests)
        for value in ignored:
            self.assertTrue(value.startswith(("stat:", "directory:", "link:")), value)

    def test_a_dirty_tracked_file_is_still_compared_by_content(self):
        """Only ignored paths weaken to stat. The file has to be dirty to be
        fingerprinted at all — a clean one never appears in `git status`."""
        self._with_ignored_build()
        tracked = self.repo / "src" / "app.py"
        tracked.write_text("edited by the developer\n")
        before = probe.tree_fingerprint(self.repo, deep=True)
        self.assertIn("src/app.py", before[2])
        os.utime(tracked, (0, 0))
        self.assertEqual(before, probe.tree_fingerprint(self.repo, deep=True))

    def test_a_run_that_writes_an_ignored_artifact_cannot_pass(self):
        (self.repo / ".gitignore").write_text("build/\n")
        (self.repo / "build").mkdir(exist_ok=True)
        (self.repo / "build" / "cache.txt").write_text("c1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "ignore"], self.repo)
        artifact = self.repo / "build" / "cache.txt"
        argv = ["sh", "-c", f"printf 'touched\\n' > {artifact}; "
                            f"! grep -qs MARK marker.txt || ! grep -rq 'print(' src"]
        result = validation.validate(self.repo, self.checkout, argv,
                                     {"marker.txt": "MARK\n"},
                                     *self.pair()[:2], timeout=30)
        self.assertEqual("c1\n", artifact.read_text())
        self.assertFalse(result.passed)


class ValidationConfinementAndFilenames(Fixture):
    """Project commands cannot modify the live checkout, including odd names."""

    def test_quoted_git_filename_is_hashed_as_the_actual_file(self):
        live = self.repo / 'note\nprivate.txt'
        live.write_text('old\n')
        before = probe.tree_fingerprint(self.repo, deep=True)
        live.write_text('new\n')
        self.assertNotEqual(before, probe.tree_fingerprint(self.repo, deep=True))
        self.assertIn('note\nprivate.txt',
                      probe.tree_fingerprint(self.repo, deep=True)[2])

    def test_an_already_dirty_file_is_compared_by_content(self):
        live = self.repo / 'ordinary.txt'
        live.write_text('old\n')
        before = probe.tree_fingerprint(self.repo, deep=True)
        live.write_text('new\n')
        self.assertNotEqual(before, probe.tree_fingerprint(self.repo, deep=True))

    def test_index_flags_cannot_hide_a_tracked_edit(self):
        tracked = self.repo / "src" / "app.py"
        for flag, undo in (("--assume-unchanged", "--no-assume-unchanged"),
                           ("--skip-worktree", "--no-skip-worktree")):
            git(["update-index", flag, "src/app.py"], self.repo)
            before = probe.tree_fingerprint(self.repo, deep=True)
            tracked.write_text(f"hidden by {flag}\n")
            self.assertNotEqual(before, probe.tree_fingerprint(self.repo, deep=True))
            git(["update-index", undo, "src/app.py"], self.repo)
            git(["checkout", "--", "src/app.py"], self.repo)

    def test_an_absolute_live_write_cannot_escape_the_probe(self):
        live = self.repo / 'note\nprivate.txt'
        live.write_text('old\n')
        command = [sys.executable, '-c',
                   'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text("new")',
                   str(live)]
        result = probe.run(self.repo, self.checkout, self.good_tree,
                           command, timeout=30)
        self.assertEqual('old\n', live.read_text())
        self.assertTrue(not result.ran or result.exit_code != 0)

    def test_a_caught_live_write_denial_is_still_a_probe_failure(self):
        live = self.repo / "must-not-change.txt"
        live.write_text("old\n")
        command = [sys.executable, "-c",
                   ("import pathlib,sys\n"
                    "try:\n pathlib.Path(sys.argv[1]).write_text('new')\n"
                    "except OSError:\n pass\n"
                    "raise SystemExit(0)"), str(live)]
        result = probe.run(self.repo, self.checkout, self.good_tree,
                           command, timeout=30)
        self.assertEqual("old\n", live.read_text())
        self.assertFalse(result.ran)
        self.assertIn("confinement refused", result.note)

    def test_a_parent_cannot_hide_its_childs_live_write_denial(self):
        live = self.repo / "child-must-not-change.txt"
        live.write_text("old\n")
        child = ("import pathlib,sys; "
                 "pathlib.Path(sys.argv[1]).write_text('new')")
        parent = ("import subprocess,sys; "
                  "subprocess.run([sys.executable, '-c', sys.argv[1], "
                  "sys.argv[2]], capture_output=True); raise SystemExit(0)")
        command = [sys.executable, "-c", parent, child, str(live)]
        result = probe.run(self.repo, self.checkout, self.good_tree,
                           command, timeout=30)
        self.assertEqual("old\n", live.read_text())
        self.assertFalse(result.ran)
        self.assertIn("confinement refused", result.note)

    def test_an_unavailable_confinement_backend_refuses_execution(self):
        original = probe._confined_argv
        probe._confined_argv = lambda scratch, argv, protected_repo=None, \
            denial_marker=None: None
        try:
            result = probe.run(self.repo, self.checkout, self.good_tree,
                               ['sh', '-c', 'exit 0'], timeout=30)
        finally:
            probe._confined_argv = original
        self.assertFalse(result.ran)
        self.assertIn('confinement is unavailable', result.note)


class UnresolvedStaysUnresolved(Fixture):
    """Regression: rolling back the files it *could* undo marked the proposal
    `rejected`, deleted the backups, and said the repository was as it was —
    while the escaped write sat outside it."""

    def escaped_two_file_apply(self):
        (self.repo / "conf").mkdir(exist_ok=True)
        (self.repo / "conf" / "b.txt").write_text("b base\n")
        (self.repo / "a.txt").write_text("a base\n")
        rec = self.approved({"a.txt": "a new\n", "conf/b.txt": "b new\n"},
                            proposal_id="prop_e5ca0002")
        moved = self.tmp / "carried-off"
        real = os.rename

        def move_before_second(src, dst, **kw):
            if "b.txt" in str(dst):
                os.rename = real
                shutil.move(str(self.repo / "conf"), str(moved))
            return real(src, dst, **kw)

        os.rename = move_before_second
        try:
            with self.assertRaises(proposal.Escaped):
                self.apply_it(rec)
        finally:
            os.rename = real
        return rec, moved

    def test_rolling_back_the_rest_does_not_finalise_it(self):
        rec, _moved = self.escaped_two_file_apply()
        after = proposal.rollback(self.repo, self.checkout, "prop_e5ca0002")
        self.assertEqual("unresolved", after["state"])
        self.assertIn("conf/b.txt", after["rollback"]["instructions"])
        self.assertNotIn("as it was", after["rollback"]["instructions"])

    def test_the_backups_are_kept_while_anything_is_unresolved(self):
        rec, _moved = self.escaped_two_file_apply()
        proposal.rollback(self.repo, self.checkout, "prop_e5ca0002")
        backup = proposal.backup_dir(self.checkout, "prop_e5ca0002")
        self.assertTrue(backup.exists(), "the only copy of the original was deleted")
        self.assertEqual("b base\n", (backup / "conf" / "b.txt").read_text())

    def test_what_could_be_undone_still_is(self):
        rec, _moved = self.escaped_two_file_apply()
        proposal.rollback(self.repo, self.checkout, "prop_e5ca0002")
        self.assertEqual("a base\n", (self.repo / "a.txt").read_text())

    def test_the_cli_does_not_report_success_for_an_unresolved_rollback(self):
        """Regression: it printed 'the repository is as it was' and exited 0
        while the proposed text sat outside the repository."""
        rec, _moved = self.escaped_two_file_apply()
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.cmd_rollback(argparse.Namespace(
                path=str(self.repo), proposal_id="prop_e5ca0002", yes=True))
        self.assertEqual(4, code)
        self.assertNotIn("as it was", out.getvalue())
        self.assertIn("NOT fully undone", err.getvalue())
        self.assertIn("conf/b.txt", err.getvalue())

    def test_the_unresolved_path_is_named_in_the_record(self):
        rec, _moved = self.escaped_two_file_apply()
        stored = proposal.load(self.checkout, "prop_e5ca0002")
        self.assertIn("conf/b.txt", stored.get("unresolved_paths") or [])


class OneWriterPerProposal(Fixture):
    """Regression: two processes both journalled, one landed the change, and the
    loser's stale record overwrote the winner's — leaving a changed file that
    rollback refused as 'nothing was applied'."""

    def race(self, rec, workers=2):
        """The first worker stalls inside the preflight, before it journals, so
        the others still read `approved` and proceed. Without a lock every worker
        journals from the same stale record and the last to finish overwrites the
        one that actually landed the change."""
        kids = []
        for index in range(workers):
            pid = os.fork()
            if pid == 0:
                try:
                    if index == 0:
                        # Stalls after the preflight passed and before the
                        # journal is written — the only window where a second
                        # worker can still read `approved` and proceed.
                        real = probe.tree_fingerprint

                        def stall(repo, deep=False, real=real):
                            time.sleep(1.0)
                            probe.tree_fingerprint = real
                            return real(repo, deep=deep)

                        probe.tree_fingerprint = stall
                    else:
                        time.sleep(0.3)
                    self.apply_it(rec)
                except BaseException:
                    pass
                finally:
                    os._exit(0)
            kids.append(pid)
        for pid in kids:
            os.waitpid(pid, 0)

    def test_the_surviving_record_can_undo_what_landed(self):
        (self.repo / "f.txt").write_text("base\n")
        rec = self.approved({"f.txt": "new\n"}, proposal_id="prop_aace0002")
        self.race(rec)
        self.assertEqual("new\n", (self.repo / "f.txt").read_text())
        stored = proposal.load(self.checkout, "prop_aace0002")
        self.assertEqual(["f.txt"], [f["path"] for f in stored["rollback"]["files"]])
        proposal.rollback(self.repo, self.checkout, "prop_aace0002")
        self.assertEqual("base\n", (self.repo / "f.txt").read_text())

    def test_the_record_is_never_left_saying_nothing_happened(self):
        (self.repo / "f.txt").write_text("base\n")
        rec = self.approved({"f.txt": "new\n"}, proposal_id="prop_aace0003")
        self.race(rec, workers=3)
        stored = proposal.load(self.checkout, "prop_aace0003")
        changed = (self.repo / "f.txt").read_text() == "new\n"
        if changed:
            self.assertTrue(stored["rollback"]["files"],
                            f"file changed but {stored['state']} records nothing")


class ValidationMayNotTouchTheCheckout(Fixture):
    """ARCHITECTURE §12.2. Project commands cannot write to the live tree."""

    def test_a_mechanism_that_works_still_fails_if_it_dirties_the_checkout(self):
        """Everything else about this run is good: it flags the bad state, accepts
        the repaired one, and the change is what does it. It still must not pass."""
        live = self.repo / "scratch.log"
        live.write_text("one\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "log"], self.repo)
        live.write_text("two\n")
        # Discriminates only once the change adds marker.txt, exactly like the
        # ruff fixture — and writes to the live checkout on the way past.
        argv = ["sh", "-c", f"printf 'touched\\n' >> {live}; "
                            f"! grep -qs MARK marker.txt || ! grep -rq 'print(' src"]
        result = validation.validate(self.repo, self.checkout, argv,
                                     {"marker.txt": "MARK\n"},
                                     *self.pair()[:2], timeout=30)
        self.assertEqual("two\n", live.read_text())
        self.assertFalse(result.passed,
                         "a run that attempted to change the checkout was accepted")
        self.assertTrue(any("confinement refused" in n for n in result.notes))

    def test_a_run_that_rewrites_a_live_file_cannot_pass(self):
        live = self.repo / "live.txt"
        live.write_text("v1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "live"], self.repo)
        live.write_text("v2 already dirty\n")
        result = validation.validate(
            self.repo, self.checkout,
            ["sh", "-c", f"printf 'rewritten\\n' > {live}"], {},
            *self.pair()[:2], timeout=30)
        self.assertEqual("v2 already dirty\n", live.read_text())
        self.assertFalse(result.passed)
        self.assertTrue(any("confinement refused" in n for n in result.notes))

    def test_an_already_dirty_file_is_compared_by_content(self):
        dirty = self.repo / "notes.txt"
        dirty.write_text("one\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "notes"], self.repo)
        dirty.write_text("two\n")
        before = probe.tree_fingerprint(self.repo, deep=True)
        dirty.write_text("three\n")
        self.assertNotEqual(before, probe.tree_fingerprint(self.repo, deep=True))

    def test_a_file_inside_an_untracked_directory_is_hashed(self):
        (self.repo / "scratchdir").mkdir()
        (self.repo / "scratchdir" / "deep.txt").write_text("a\n")
        digests = probe.tree_fingerprint(self.repo, deep=True)[2]
        self.assertIn("scratchdir/deep.txt", digests)
        self.assertNotIn("directory", digests.values())


class AnEscapedWriteIsUnresolved(Fixture):
    """Regression: a write that landed outside reported the repository as it was
    and left the record `approved`, naming nothing."""

    def test_a_relocated_parent_is_recorded_as_unresolved(self):
        rec = self.approved({"src/app.py": "REPOHONE WROTE THIS\n"},
                            proposal_id="prop_e5ca0001")
        moved = self.tmp / "escaped-dir"
        real = os.rename

        def move_then(src, dst, **kw):
            os.rename = real
            shutil.move(str(self.repo / "src"), str(moved))
            return real(src, dst, **kw)

        os.rename = move_then
        try:
            with self.assertRaises(proposal.Escaped) as caught:
                self.apply_it(rec)
        finally:
            os.rename = real
        self.assertIn("outside the repository", str(caught.exception))
        stored = proposal.load(self.checkout, "prop_e5ca0001")
        self.assertEqual("unresolved", stored["state"])
        self.assertIn("src/app.py", stored["rollback"]["instructions"])
        self.assertNotIn("as it was", stored["rollback"]["instructions"])


class ModeIsPartOfWhatWasApproved(Fixture):
    """Regression: only content was compared, so a chmod between validation and
    approval landed a check that no longer behaved as it was shown to."""

    def an_executable_proposal(self):
        script = self.repo / "gate.sh"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        rec = self.approved({"gate.sh": "#!/bin/sh\nexit 1\n"},
                            proposal_id="prop_0d0e0001")
        rec["change"]["files"][0]["baseline_mode"] = 0o755
        proposal.save(self.checkout, rec)
        return rec, script

    def test_a_chmod_after_validation_refuses_the_apply(self):
        rec, script = self.an_executable_proposal()
        script.chmod(0o644)
        with self.assertRaises(proposal.Conflict) as caught:
            self.apply_it(rec)
        self.assertIn("0o755", str(caught.exception))
        self.assertEqual("#!/bin/sh\nexit 0\n", script.read_text())

    def test_an_unchanged_mode_applies_normally(self):
        rec, script = self.an_executable_proposal()
        self.assertEqual("observing", self.apply_it(rec)["state"])
        self.assertEqual(0o755, stat.S_IMODE(script.stat().st_mode))

    def test_the_baseline_mode_is_recorded_when_proposing(self):
        script = self.repo / "tool.sh"
        script.write_text("#!/bin/sh\n")
        script.chmod(0o755)
        self.assertEqual(0o755, selection._existing_mode(
            proposal._existing_paths(self.repo), "tool.sh"))


class ARewrittenDirtyFileIsNoticed(Fixture):
    """Regression: the fingerprint compared which paths were dirty, and a file
    already modified stays on the same status line when modified again."""

    def test_rewriting_an_already_dirty_file_is_detected(self):
        (self.repo / "notes.txt").write_text("v1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "notes"], self.repo)
        (self.repo / "notes.txt").write_text("v2\n")
        rec = self.approved({"pyproject.toml":
                             "[tool.ruff.lint]\nselect = ['E', 'T20']\n"})
        real = proposal._replace_inside

        def meddle(root, relative, text, **kw):
            out = real(root, relative, text, **kw)
            (self.repo / "notes.txt").write_text("v3 rewritten mid-apply\n")
            return out

        proposal._replace_inside = meddle
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                self.apply_it(rec)
        finally:
            proposal._replace_inside = real
        self.assertIn("working tree changed", str(caught.exception))
        self.assertIn("notes.txt", str(caught.exception))


class ModeSurvivesApplyAndRollback(Fixture):
    """Regression: the replacement was created 0644, so an executable script
    stopped being executable — validation approved behaviour that then did not
    land."""

    def an_executable_proposal(self):
        script = self.repo / "scripts" / "check.sh"
        script.parent.mkdir(exist_ok=True)
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "script"], self.repo)
        return self.approved({"scripts/check.sh": "#!/bin/sh\nexit 1\n"}), script

    def test_the_executable_bit_survives_apply(self):
        rec, script = self.an_executable_proposal()
        self.assertTrue(os.access(script, os.X_OK))
        self.apply_it(rec)
        self.assertTrue(os.access(script, os.X_OK),
                        "the applied file is no longer executable")
        self.assertEqual(0o755, stat.S_IMODE(script.stat().st_mode))

    def test_the_mode_is_restored_by_rollback(self):
        rec, script = self.an_executable_proposal()
        self.apply_it(rec)
        script.chmod(0o700)
        proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertEqual(0o755, stat.S_IMODE(script.stat().st_mode))

    def test_the_mode_is_recorded_for_undoing(self):
        rec, _script = self.an_executable_proposal()
        self.assertEqual(0o755,
                         self.apply_it(rec)["rollback"]["files"][0]["previous_mode"])


class InterruptedRollbackStaysRetryable(Fixture):
    """Regression: an OSError escaped, the state still said `observing`, both
    files were still listed, and a retry conflicted on the one already restored."""

    def two_file_proposal(self):
        rec = self.approved({
            "pyproject.toml": "[tool.ruff.lint]\nselect = ['E', 'T20']\n",
            "ruff.toml": "line-length = 100\n"})
        self.apply_it(rec)
        return rec

    def rollback_failing_on(self, rec, path):
        real = proposal._replace_inside
        unlink = proposal._unlink_inside

        def fail(root, relative, *a, **kw):
            if relative == path:
                raise OSError("injected restore failure")
            return real(root, relative, *a, **kw)

        def fail_unlink(root, relative, **kw):
            if relative == path:
                raise OSError("injected restore failure")
            return unlink(root, relative, **kw)

        proposal._replace_inside = fail
        proposal._unlink_inside = fail_unlink
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        finally:
            proposal._replace_inside = real
            proposal._unlink_inside = unlink
        return caught.exception

    def test_an_os_error_becomes_a_reported_partial_rollback(self):
        rec = self.two_file_proposal()
        problem = self.rollback_failing_on(rec, "ruff.toml")
        self.assertIn("could not be restored", str(problem))
        stored = proposal.load(self.checkout, rec["proposal_id"])
        self.assertEqual("partially_applied", stored["state"])

    def test_only_the_unrestored_file_stays_pending(self):
        rec = self.two_file_proposal()
        self.rollback_failing_on(rec, "ruff.toml")
        stored = proposal.load(self.checkout, rec["proposal_id"])
        self.assertEqual(["ruff.toml"],
                         [f["path"] for f in stored["rollback"]["files"]])

    def test_a_retry_finishes_the_job(self):
        rec = self.two_file_proposal()
        before = "[tool.ruff.lint]\nselect = ['E']\n"
        self.rollback_failing_on(rec, "ruff.toml")
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())
        proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertFalse((self.repo / "ruff.toml").exists())
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())


class BinaryContentIsNotAbsence(Fixture):
    """Regression: an undecodable file read as `None`, the same as a missing one,
    so a `create` proposal considered its baseline unchanged and replaced it."""

    def a_create_proposal(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = [{"path": "ruff.toml", "action": "create",
                                     "contents": "select = ['T20']\n"}]
        rec = self.propose(FakeEngine(reply))
        self.assertIsNone(rec["change"]["files"][0]["baseline_sha256"])
        return rec

    def test_a_binary_file_created_after_validation_survives(self):
        rec = self.a_create_proposal()
        (self.repo / "ruff.toml").write_bytes(b"\xff\xfe\x00binary")
        with self.assertRaises(proposal.Conflict) as caught:
            proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                           proposal.staged(self.checkout, rec["proposal_id"]))
        self.assertIn("not text", str(caught.exception))
        self.assertEqual(b"\xff\xfe\x00binary", (self.repo / "ruff.toml").read_bytes())

    def test_a_genuinely_absent_file_is_still_created(self):
        rec = self.a_create_proposal()
        self.assertFalse((self.repo / "ruff.toml").exists())
        proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                       proposal.staged(self.checkout, rec["proposal_id"]))
        self.assertIn("T20", (self.repo / "ruff.toml").read_text())

    def test_reading_binary_content_is_a_conflict_not_none(self):
        (self.repo / "blob.bin").write_bytes(b"\x00\xff\xfe")
        with self.assertRaises(proposal.Conflict):
            proposal._read_inside(self.repo, "blob.bin")
        self.assertIsNone(proposal._read_inside(self.repo, "nothing-here.txt"))


class RecoveryDoesNotOverwriteALaterEdit(Fixture):
    """Regression: when a later file failed, recovery restored an earlier one
    over a third party's edit and reported the tree as it was."""

    def two_file_proposal(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = [
            {"path": "pyproject.toml", "action": "modify",
             "edits": [{"find": "select = ['E']", "replace": "select = ['E', 'T20']"}]},
            {"path": "ruff.toml", "action": "create", "contents": "line-length = 100\n"}]
        return self.propose(FakeEngine(reply))

    def test_an_edit_to_an_already_written_file_is_kept(self):
        rec = self.two_file_proposal()
        real = proposal._replace_inside

        def meddle(root, relative, text, **kw):
            if relative == "ruff.toml":
                # someone edits the file RepoHone already wrote, then this fails
                (self.repo / "pyproject.toml").write_text("THEIR EDIT\n")
                raise OSError("injected write failure")
            return real(root, relative, text, **kw)

        proposal._replace_inside = meddle
        try:
            with self.assertRaises(proposal.Conflict) as caught:
                proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                               proposal.staged(self.checkout, rec["proposal_id"]))
        finally:
            proposal._replace_inside = real
        self.assertEqual("THEIR EDIT\n", (self.repo / "pyproject.toml").read_text())
        self.assertNotIn("the repository is as it was", str(caught.exception))
        self.assertEqual("partially_applied",
                         proposal.load(self.checkout, rec["proposal_id"])["state"])


class ApplyPreservesAnInterveningEdit(Fixture):
    """Regression: baselines were checked once up front, so an edit landing while
    an earlier file was written was overwritten and reported as success."""

    def two_file_proposal(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = [
            {"path": "pyproject.toml", "action": "modify",
             "edits": [{"find": "select = ['E']", "replace": "select = ['E', 'T20']"}]},
            {"path": "ruff.toml", "action": "create", "contents": "line-length = 100\n"}]
        return self.propose(FakeEngine(reply))

    def apply_with_interference(self, rec):
        """Another process edits the second file while the first is written."""
        real = proposal._replace_inside

        def meddle(root, relative, text, **kw):
            real(root, relative, text, **kw)
            if relative == "pyproject.toml":
                (self.repo / "ruff.toml").write_text("SOMEONE ELSE WAS HERE\n")

        proposal._replace_inside = meddle
        try:
            return proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                                  proposal.staged(self.checkout, rec["proposal_id"]))
        finally:
            proposal._replace_inside = real

    def test_the_intervening_edit_survives(self):
        rec = self.two_file_proposal()
        with self.assertRaises(proposal.Conflict) as caught:
            self.apply_with_interference(rec)
        self.assertIn("changed while this proposal was being applied",
                      str(caught.exception))
        self.assertEqual("SOMEONE ELSE WAS HERE\n",
                         (self.repo / "ruff.toml").read_text())

    def test_the_earlier_write_is_rolled_back(self):
        rec = self.two_file_proposal()
        before = (self.repo / "pyproject.toml").read_text()
        with self.assertRaises(proposal.Conflict):
            self.apply_with_interference(rec)
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())

    def test_the_proposal_is_not_marked_applied(self):
        rec = self.two_file_proposal()
        with self.assertRaises(proposal.Conflict):
            self.apply_with_interference(rec)
        stored = proposal.load(self.checkout, rec["proposal_id"])
        self.assertNotIn(stored["state"], ("observing", "applied"))
        self.assertEqual([], stored["rollback"]["files"])


class TheMaintainerSeesThePatch(Fixture):
    """Regression: approval showed filenames and byte counts, which is not
    something a maintainer can consent to."""

    def patch_for(self, rec):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli._print_patch(self.repo, self.checkout, rec)
        return out.getvalue()

    def test_the_diff_shows_the_lines_being_added(self):
        rec = self.propose()
        text = self.patch_for(rec)
        self.assertIn("+++ b/pyproject.toml", text)
        self.assertIn("T20", text)
        self.assertIn("-select = ['E']", text)

    def test_a_created_file_shows_its_whole_contents(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = [{"path": "ruff.toml", "action": "create",
                                     "contents": "select = ['T20']\n"}]
        text = self.patch_for(self.propose(FakeEngine(reply)))
        self.assertIn("+++ b/ruff.toml", text)
        self.assertIn("+select = ['T20']", text)

    def test_a_long_patch_is_truncated_with_a_pointer(self):
        rec = self.propose()
        big = "".join(f"line {i}\n" for i in range(500))
        proposal.stage(self.checkout, rec["proposal_id"], {"pyproject.toml": big})
        text = self.patch_for(rec)
        self.assertIn("patch truncated", text)
        self.assertIn("staged.json", text)

    def test_apply_without_yes_prints_the_patch_and_does_not_change_anything(self):
        rec = self.propose()
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        before = (self.repo / "pyproject.toml").read_text()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.cmd_apply(argparse.Namespace(
                path=str(self.repo), proposal_id=rec["proposal_id"], yes=False))
        self.assertEqual(3, code)
        self.assertIn("T20", out.getvalue())
        self.assertIn("Re-run with --yes", out.getvalue())
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())

    def test_a_probabilistic_mechanism_is_disclosed(self):
        rec = self.propose()
        rec["mechanism"]["deterministic"] = False
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli._print_proposal(rec)
        self.assertIn("JUDGEMENT", out.getvalue())
        self.assertIn("judges rather than decides", out.getvalue())

    def test_a_deterministic_mechanism_carries_no_judgement_warning(self):
        rec = self.propose()
        self.assertTrue(rec["mechanism"]["deterministic"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli._print_proposal(rec)
        self.assertNotIn("JUDGEMENT", out.getvalue())


class Selection(Fixture):
    """Pass 2 chooses from what the repository already provides."""

    def test_a_mechanism_outside_the_inventory_is_rejected(self):
        reply = dict(GOOD_REPLY, mechanism_id="mech_ffffffffffff")
        result = self.propose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])
        self.assertIn("not one this repository provides", result["failure"]["reason"])

    def test_the_selection_call_obeys_the_shared_egress_budget(self):
        inventory = {"mechanisms": [
            {**INVENTORY["mechanisms"][0], "id": f"mech_{i:012x}",
             "name": "n" * 500, "invocation": "x" * 1000,
             "config_excerpt": "config " * 1000}
            for i in range(30)]}
        prompt = selection.build_prompt(DIAGNOSIS, inventory)
        self.assertLessEqual(reasoning.measure_bytes(prompt),
                             reasoning.MAX_INPUT_BYTES)
        self.assertLessEqual(prompt.count("- id="), reasoning.MAX_INPUT_ITEMS)

    def test_the_only_relevant_mechanism_survives_the_item_bound(self):
        kinds = ["architecture", "build", "ci", "format", "lint", "script",
                 "test", "vcs-hook"]
        inventory = {"mechanisms": [
            {"id": f"mech_{index}", "kind": kind, "name": f"existing {kind}",
             "tier": "existing-project", "invocation": f"run-{kind}",
             "enforced_in": [], "carries": [f"{kind}-constraint"]}
            for index, kind in enumerate(kinds)] + [{
                "id": "mech_types", "kind": "types", "name": "mypy",
                "tier": "existing-project", "invocation": "mypy .",
                "enforced_in": ["ci"], "carries": ["type-constraint"]}]}
        diagnosis = dict(DIAGNOSIS,
                         required_property="function results must have type constraints")

        prompt = selection.build_prompt(diagnosis, inventory)

        self.assertIn("id=mech_types", prompt)
        self.assertEqual(reasoning.MAX_INPUT_ITEMS, prompt.count("- id="))

    def test_duplicate_proposal_paths_are_invalid_output(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"].append(dict(reply["change"]["files"][0]))
        result = self.propose(FakeEngine(reply))
        self.assertEqual(reasoning.INVALID_OUTPUT, result["outcome"])
        self.assertIn("duplicate path", result["failure"]["reason"])

    def test_case_and_unicode_path_aliases_are_duplicates(self):
        files = [
            {"path": "Rule.txt", "action": "modify", "edits": [{"append": "one"}]},
            {"path": "rule.txt", "action": "modify", "edits": [{"append": "two"}]},
        ]
        _cleaned, problem = selection._clean_files(files, _AllPaths())
        self.assertIn("duplicate path", problem)

    def test_noncanonical_path_and_action_mismatch_are_rejected(self):
        _cleaned, problem = selection._clean_files([
            {"path": "./rule.txt", "action": "modify", "edits": [{"append": "one"}]}],
            _AllPaths())
        self.assertIn("normalized", problem)
        _cleaned, problem = selection._clean_files([
            {"path": "rule.txt", "action": "create", "contents": "one\n"}],
            _AllPaths())
        self.assertIn("already exists", problem)

    def test_interactive_consent_describes_the_possible_judge_call(self):
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.cmd_propose(argparse.Namespace(
                path=str(self.repo), yes=False, diagnosis_id="dx_missing",
                model=None, timeout=30, validate_timeout=30))
        self.assertEqual(3, code)
        self.assertIn("second call", err.getvalue())
        self.assertIn("cross-session summaries", err.getvalue())

    def test_the_missing_why_question_is_rejected(self):
        """LEARNING_PLAN §11 — load-bearing."""
        reply = dict(GOOD_REPLY)
        del reply["why_existing_did_not_help"]
        result = self.propose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])
        self.assertIn("why_existing_did_not_help", result["failure"]["reason"])

    def test_no_mechanism_available_is_a_valid_answer(self):
        reply = dict(GOOD_REPLY, outcome="NO_MECHANISM_AVAILABLE",
                     mechanism_id=None, change=None)
        result = self.propose(FakeEngine(reply))
        self.assertEqual("NO_MECHANISM_AVAILABLE", result["outcome"])
        self.assertEqual("rejected", result["state"])
        self.assert_conformant(result)

    def test_an_empty_inventory_still_reaches_rung_three(self):
        """A project with no tooling is the case the plan's third rung exists for;
        it used to short-circuit to NO_MECHANISM_AVAILABLE without asking."""
        engine = FakeEngine()
        self.propose(engine, inventory={"mechanisms": []})
        self.assertIsNotNone(engine.prompt, "the model was never asked")
        self.assertIn("new project-owned check", engine.prompt)

    def test_an_empty_inventory_is_described_honestly(self):
        self.assertIn("no test, lint, type or check mechanism",
                      selection.describe_inventory({"mechanisms": []}))

    def test_the_prompt_shows_what_is_enforced(self):
        engine = FakeEngine()
        self.propose(engine)
        self.assertIn("enforced in ci", engine.prompt)
        self.assertIn("mech_0123456789ab", engine.prompt)

    def test_the_prompt_asks_every_pass_two_question(self):
        engine = FakeEngine()
        self.propose(engine)
        for question in ("best carries that property", "simpler option",
                         "didn't an existing mechanism already help",
                         "friction will this add", "make no proposal"):
            self.assertIn(question, engine.prompt)


class ADormantCheckIsNotAnImprovement(Fixture):
    """Regression: an existing script with `enforced_in: []` could be modified
    and validated directly, producing a SUCCESS for a check nothing runs. The
    rule applied only to brand-new mechanisms."""

    def dormant(self, **over):
        base = {"id": "mech_dormant00", "kind": "lint", "name": "scripts/check.sh",
                "tier": "existing-project", "argv": ["sh", "scripts/check.sh"],
                "invocation": "sh scripts/check.sh", "config_path": None,
                "deterministic": True, "enforced_in": [], "cost": None}
        base.update(over)
        return base

    def reply_for(self, mechanism, files=None):
        return {"outcome": "SUCCESS", "mechanism_id": mechanism["id"],
                "why_this": "x", "why_existing_did_not_help": "it never checked this",
                "alternatives_considered": ["doing nothing"],
                "change": {"summary": "x", "files": files or [
                    {"path": "scripts/check.sh", "action": "modify",
                     "edits": [{"append": "exit 1"}]}]},
                "expected_effect": "x", "friction": [], "adds_model_egress": False}

    def interpret(self, mechanism, files=None):
        return selection.interpret(self.reply_for(mechanism, files),
                                   {"mechanisms": [mechanism]}, _AllPaths())

    def test_a_check_nothing_runs_is_refused(self):
        out = self.interpret(self.dormant())
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("scripts/check.sh", out["failure"])

    def test_the_same_check_is_accepted_once_something_runs_it(self):
        self.assertEqual(reasoning.SUCCESS,
                         self.interpret(self.dormant(enforced_in=["ci"]))["outcome"])

    def test_wiring_it_in_makes_it_acceptable(self):
        out = self.interpret(self.dormant(argv=["make", "check"]), files=[
            {"path": "scripts/check.sh", "action": "modify",
             "edits": [{"append": "exit 1"}]},
            {"path": "Makefile", "action": "modify",
             "edits": [{"append": "check:\n\tsh scripts/check.sh"}]}])
        self.assertEqual(reasoning.SUCCESS, out["outcome"])

    def test_advice_is_exempt_because_nothing_runs_it_by_design(self):
        out = self.interpret(self.dormant(deterministic=False, argv=None,
                                          kind="agent-instruction",
                                          name="CLAUDE.md"),
                             files=[{"path": "CLAUDE.md", "action": "modify",
                                     "edits": [{"append": "- Never use print()."}]}])
        self.assertEqual(reasoning.SUCCESS, out["outcome"])


class ARunnerTheChangeCreates(Fixture):
    """Regression: a proposed `make verify` was rejected because its baseline run
    failed — the target does not exist until the change adds it. That blocked the
    plan's third rung for exactly the projects it exists for."""

    def test_a_missing_baseline_command_is_not_a_failed_baseline(self):
        result = validation.validate(
            self.repo, self.checkout, ["make", "verify"],
            {"Makefile": "verify:\n\t@! grep -rq 'print(' src\n"},
            *self.pair()[:2], timeout=30, creates_runner=True)
        self.assertTrue(result.performed, result.notes)
        self.assertTrue(any("does not exist until this change" in n
                            for n in result.notes))

    def test_without_that_signal_a_missing_target_reads_as_already_catching_it(self):
        """The old behaviour, kept as a test so the difference is visible: a
        target that does not exist exits non-zero and is indistinguishable from
        a mechanism that already flags the problem."""
        result = validation.validate(
            self.repo, self.checkout, ["make", "verify"],
            {"Makefile": "verify:\n\t@! grep -rq 'print(' src\n"},
            *self.pair()[:2], timeout=30)
        self.assertFalse(result.passed)
        self.assertTrue(any("already fails on the problematic state" in n
                            for n in result.notes))

    def test_the_signal_is_only_for_a_runner_the_change_adds(self):
        self.assertTrue(proposal._creates_its_runner(
            {"tier": "project-owned-custom"}, [{"path": "Makefile"}], self.repo))
        (self.repo / "Makefile").write_text("build:\n\t@echo hi\n")
        self.assertFalse(proposal._creates_its_runner(
            {"tier": "project-owned-custom"}, [{"path": "Makefile"}], self.repo))
        self.assertFalse(proposal._creates_its_runner(
            {"tier": "existing-project"}, [{"path": "Makefile"}], self.repo))


class _AllPaths:
    def __contains__(self, key):
        return True

    def text(self, path):
        return "base\n"

    def sha256(self, path):
        return "0" * 64

    def mode(self, path):
        return 0o644


class TheCorrectionIsWhereTheRecordSaysItIs(Fixture):
    """Regression: probabilistic grading always showed turns 1 and 2. A session
    corrected at turn 5 was presented to the judge as if turn 2 corrected turn 1
    — two unrelated requests offered as a mistake and its repair."""

    def five_turn_record(self):
        turns = []
        for index in range(1, 6):
            turns.append({
                "index": index, "logical_turn_id": f"t{index}",
                "prompt_events": [{"ordinal": 1, "snapshot": None,
                                   "content": {"text": f"request {index}"}}],
                "stop_events": [{"ordinal": 1, "kind": "stop", "snapshot": None,
                                 "last_assistant_message": {"text": f"reply {index}"}}],
            })
        return {"turns": turns}

    def test_the_named_turn_is_the_one_shown(self):
        summary = proposal._summarise_session("rh_x", self.five_turn_record(),
                                              corrective_turn=5)
        self.assertEqual("request 4", summary["asked"])
        self.assertEqual("reply 4", summary["produced"])
        self.assertEqual("request 5", summary["corrected"])

    def test_turns_one_and_two_are_not_assumed(self):
        summary = proposal._summarise_session("rh_x", self.five_turn_record(),
                                              corrective_turn=5)
        self.assertNotEqual("request 1", summary["asked"])

    def test_a_session_with_no_identified_turn_is_not_graded(self):
        self.assertIsNone(proposal._summarise_session("rh_x",
                                                      self.five_turn_record()))

    def test_a_correction_at_turn_one_has_no_predecessor(self):
        self.assertIsNone(proposal._summarise_session("rh_x",
                                                      self.five_turn_record(),
                                                      corrective_turn=1))

    def test_judge_examples_are_redacted_and_path_scrubbed_again(self):
        rec = self.five_turn_record()
        rec["turns"][3]["prompt_events"][0]["content"]["text"] = (
            f"password=hunter2xyz read {self.repo}/private.txt")
        summary = proposal._summarise_session(
            "rh_x", rec, corrective_turn=5, root=self.repo, owner=self.checkout)
        self.assertIn("[REDACTED]", summary["asked"])
        self.assertIn("<repo>/private.txt", summary["asked"])
        self.assertNotIn(str(self.repo), summary["asked"])
        self.assertTrue(summary["id"].startswith(self.checkout + "/"))


class EgressMustBeDisclosedInFull(Fixture):
    """ARCHITECTURE §13 — a maintainer cannot consent to a boolean."""

    def reply_with(self, egress):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["adds_model_egress"] = True
        if egress is not None:
            reply["egress"] = egress
        return reply

    FULL = {"what_leaves": "the diff under review", "provider": "Anthropic",
            "when": "on every pull request", "frequency": "~30 a week",
            "cost": "about $4 a week on the team account"}

    def test_a_boolean_alone_is_refused(self):
        out = selection.interpret(self.reply_with(None), INVENTORY, _AllPaths())
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("egress disclosure object", out["failure"])

    def test_each_missing_field_is_named(self):
        partial = {k: v for k, v in self.FULL.items() if k not in ("cost", "frequency")}
        out = selection.interpret(self.reply_with(partial), INVENTORY, _AllPaths())
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("frequency", out["failure"])
        self.assertIn("cost", out["failure"])

    def test_a_complete_disclosure_is_accepted(self):
        out = selection.interpret(self.reply_with(self.FULL), INVENTORY,
                                  proposal._existing_paths(self.repo))
        self.assertEqual(reasoning.SUCCESS, out["outcome"])

    def test_a_mechanism_with_no_egress_needs_no_disclosure(self):
        out = selection.interpret(json.loads(json.dumps(GOOD_REPLY)),
                                  INVENTORY, proposal._existing_paths(self.repo))
        self.assertEqual(reasoning.SUCCESS, out["outcome"])

    def test_malformed_disclosure_is_a_persisted_invalid_result(self):
        reply = self.reply_with('not an object')
        result = self.propose(engine=FakeEngine(reply))
        self.assertEqual(reasoning.INVALID_OUTPUT, result['outcome'])
        self.assertEqual(1, len(proposal.load_all(self.checkout)))


class ProbabilisticMechanisms(Fixture):
    """Plan, phase 5: 'test against labelled historical examples, estimate false
    positives'. Advice an agent reads cannot be executed, so §14's ladder does
    not apply and the result must not be dressed up as if it did."""

    def sessions(self, n_pos=3, n_neg=2):
        pos = [{"id": f"rh_pos{i}", "asked": "add startup output",
                "produced": 'added print("x")', "corrected": "no, use the logger"}
               for i in range(n_pos)]
        neg = [{"id": f"rh_neg{i}", "asked": "rename a variable",
                "produced": "renamed it", "corrected": None}
               for i in range(n_neg)]
        return pos, neg

    def judge(self, caught=(0, 1, 2), misfired=(0,)):
        pos, neg = self.sessions()
        verdicts = [{"id": f"rh_pos{i}", "would_have_caught": i in caught,
                     "would_have_misfired": False, "why": "x"} for i in range(3)]
        verdicts += [{"id": f"rh_neg{i}", "would_have_caught": False,
                      "would_have_misfired": i in misfired, "why": "x"}
                     for i in range(2)]
        return FakeEngine({"verdicts": verdicts})

    def estimate(self, engine=None, **kw):
        pos, neg = self.sessions()
        return validation.estimate_probabilistic(
            "never use print()", pos, neg, engine or self.judge(**kw))

    def test_it_reports_what_it_would_have_caught(self):
        out = self.estimate()
        self.assertTrue(out["performed"])
        self.assertEqual(3, out["caught"])
        self.assertEqual(3, out["of_occurrences"])

    def test_a_rule_that_misfires_on_clean_work_does_not_pass(self):
        """Same bar as the executable path: it must fire on nothing that was
        already correct. An estimate is weaker evidence, not a lower bar."""
        for misfired, expected in ((0, True), (1, False), (2, False)):
            estimate = {"performed": True, "caught": 1, "of_occurrences": 1,
                        "misfired": misfired, "of_clean_sessions": 2}
            result = validation.Result(True, rejects_problematic=True,
                                       estimate=estimate)
            self.assertEqual(expected, result.passed, f"misfired={misfired}")

    def test_catching_nothing_does_not_pass(self):
        estimate = {"performed": True, "caught": 0, "of_occurrences": 3,
                    "misfired": 0, "of_clean_sessions": 2}
        self.assertFalse(validation.Result(True, estimate=estimate).passed)

    def test_it_reports_misfires_but_never_a_rate(self):
        """The controls are sessions with a *different* diagnosis, not verified
        clean work; a rate over them would read as a measurement it is not."""
        out = self.estimate(misfired=(0, 1))
        self.assertEqual(2, out["misfired"])
        self.assertEqual(2, out["of_clean_sessions"])
        self.assertIsNone(out["false_positive_rate"])
        self.assertIn("proxy", out["basis"])

    def test_it_never_claims_to_have_run_the_mechanism(self):
        self.assertIn("not executed", self.estimate()["basis"])

    def test_it_names_the_provider_that_judged(self):
        self.assertEqual("fake", self.estimate()["provider"])

    def test_a_small_sample_says_so(self):
        self.assertIn("too few", self.estimate()["note"])

    def test_no_labelled_sessions_is_not_a_silent_zero(self):
        out = validation.estimate_probabilistic("x", [], [], self.judge())
        self.assertFalse(out["performed"])
        self.assertIn("no labelled sessions", out["note"])

    def test_an_unavailable_judge_does_not_pass(self):
        engine = FakeEngine(raises=reasoning.Unavailable("no provider"))
        out = self.estimate(engine=engine)
        self.assertFalse(out["performed"])
        self.assertIn("unavailable", out["note"])

    def test_a_partial_judge_reply_does_not_pass(self):
        engine = FakeEngine({"verdicts": [
            {"id": "rh_pos0", "would_have_caught": True,
             "would_have_misfired": False, "why": "x"}]})
        out = self.estimate(engine=engine)
        self.assertFalse(out["performed"])
        self.assertIn("partial", out["note"])

    def test_duplicate_judge_ids_do_not_pass(self):
        verdict = {"id": "rh_pos0", "would_have_caught": True,
                   "would_have_misfired": False, "why": "x"}
        out = self.estimate(engine=FakeEngine({"verdicts": [verdict] * 5}))
        self.assertFalse(out["performed"])
        self.assertIn("duplicated", out["note"])

    def test_judge_call_has_eight_items_total_and_stays_in_budget(self):
        pos, neg = self.sessions(20, 20)
        chosen = validation._balanced_examples(pos, neg)
        verdicts = [{"id": entry["id"], "would_have_caught": label.startswith("THE"),
                     "would_have_misfired": False, "why": "x"}
                    for label, entry in chosen]
        engine = FakeEngine({"verdicts": verdicts})
        out = validation.estimate_probabilistic("rule " * 1000, pos, neg, engine)
        self.assertTrue(out["performed"])
        self.assertEqual(8, engine.prompt.count("--- "))
        self.assertLessEqual(reasoning.measure_bytes(engine.prompt),
                             reasoning.MAX_INPUT_BYTES)
    def test_apply_without_yes_reports_the_rejection_instead_of_crashing(self):
        rec = proposal.build_record(
            DIAGNOSIS, self.checkout, reasoning.INVALID_OUTPUT, record.now(),
            "prop_ae1ec7ed", failure="bad model output")
        proposal.save(self.checkout, rec)
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.cmd_apply(argparse.Namespace(
                path=str(self.repo), proposal_id=rec["proposal_id"], yes=False))
        self.assertEqual(1, code)
        self.assertIn("nothing can be applied", err.getvalue())

    def test_a_probabilistic_proposal_is_graded_not_executed(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["mechanism_id"] = ADVICE["id"]
        reply["change"]["files"] = [{"path": "CLAUDE.md", "action": "create",
                                     "contents": "- Never use print().\n"}]
        rec = self.propose(FakeEngine(reply), inventory=ADVICE_INVENTORY)
        self.assertIsNotNone(rec["validation"]["estimate"])
        self.assertIn("not a check that runs",
                      " ".join(rec["validation"]["notes"]))
        self.assert_conformant(rec)

    def test_the_display_says_estimated_not_validated(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["mechanism_id"] = ADVICE["id"]
        reply["change"]["files"] = [{"path": "CLAUDE.md", "action": "create",
                                     "contents": "- Never use print().\n"}]
        rec = self.propose(FakeEngine(reply), inventory=ADVICE_INVENTORY)
        rec["outcome"] = reasoning.SUCCESS
        rec["mechanism"] = rec["mechanism"] or {
            "name": "CLAUDE.md", "kind": "agent-instruction",
            "tier": "existing-project", "why_this": "x",
            "why_existing_did_not_help": "x", "alternatives_considered": [],
            "deterministic": False, "cost": None, "id": "mech_00000000"}
        rec["change"] = rec["change"] or {"summary": "x", "files": []}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cli._print_proposal(rec)
        self.assertIn("ESTIMATED", out.getvalue())
        self.assertNotIn("rejects the state the agent produced", out.getvalue())


class RungThree(Fixture):
    """The plan's ladder ends in `add a project-owned custom mechanism`. The
    prompt used to forbid it, so a project whose tooling could not carry the
    property dead-ended at NO_MECHANISM_AVAILABLE."""

    def reply(self, **mechanism):
        base = {"name": "scripts/check-logging.sh", "kind": "lint",
                "invocation": "make check-logging",
                "argv": ["make", "check-logging"]}
        base.update(mechanism)
        return {"outcome": "NEW_MECHANISM", "mechanism_id": None,
                "new_mechanism": base,
                "why_this": "nothing here inspects Python source at all",
                "why_existing_did_not_help": "the only check greps whitespace",
                "alternatives_considered": ["doing nothing: it recurs"],
                "change": {"summary": "add a logging check",
                           "files": [{"path": "scripts/check-logging.sh",
                                      "action": "create",
                                      "contents": "#!/bin/sh\nexit 0\n"},
                                     {"path": "Makefile", "action": "create",
                                      "contents": "check:\n\tsh scripts/check-logging.sh\n"}]},
                "expected_effect": "the check fails on a stray print()",
                "friction": ["one more command to run"],
                "adds_model_egress": False}

    def interpret(self, reply):
        return selection.interpret(reply, {"mechanisms": []}, _NoPaths())

    def test_a_new_project_owned_check_is_accepted(self):
        out = self.interpret(self.reply())
        self.assertEqual(reasoning.SUCCESS, out["outcome"])
        self.assertEqual("project-owned-custom", out["mechanism"]["tier"])
        self.assertEqual(["make", "check-logging"], out["mechanism"]["argv"])

    def test_it_is_marked_as_never_having_run_here(self):
        mechanism = self.interpret(self.reply())["mechanism"]
        self.assertFalse(mechanism["cost"]["probed"])
        self.assertIn("never run in this project", mechanism["cost"]["note"])

    def test_a_mechanism_that_cannot_be_run_is_refused(self):
        """An unvalidatable mechanism is worse than none: §14 cannot show it
        catches anything, so approving it would be approving a claim."""
        out = self.interpret(self.reply(argv=[]))
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("cannot be run", out["failure"])

    def test_a_new_mechanism_may_not_depend_on_repohone(self):
        out = self.interpret(self.reply(name="repohone-check",
                                        invocation="repohone check",
                                        argv=["repohone", "check"]))
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("survive its removal", out["failure"])

    def test_an_unknown_kind_is_refused(self):
        self.assertEqual(reasoning.INVALID_OUTPUT,
                         self.interpret(self.reply(kind="vibes"))["outcome"])

    def test_it_must_still_explain_why_what_existed_failed(self):
        reply = self.reply()
        reply["why_existing_did_not_help"] = "  "
        self.assertEqual(reasoning.INVALID_OUTPUT, self.interpret(reply)["outcome"])

    def test_a_command_that_runs_the_new_script_directly_is_refused(self):
        """§14 executes this argv. Declaring the script itself proves the script
        works, not that anything in the project invokes it."""
        reply = self.reply(argv=["sh", "scripts/check-logging.sh"])
        out = selection.interpret(reply, {"mechanisms": []}, _NoPaths())
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("runs scripts/check-logging.sh directly", out["failure"])

    def test_a_runner_that_only_echoes_the_check_fails_validation(self):
        """The pre-filter lets it through; running `make check` on the state the
        agent produced is what rejects it, because an echo exits zero there."""
        (self.repo / "Makefile").write_text("check:\n\t@echo scripts/check.sh\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "makefile"], self.repo)
        result = validation.validate(
            self.repo, self.checkout, ["make", "check"],
            {"scripts/check.sh": "#!/bin/sh\nexit 1\n"},
            *self.pair()[:2], timeout=30)
        self.assertTrue(result.performed)
        self.assertFalse(result.rejects_problematic,
                         "an echo-only runner was treated as catching the mistake")
        self.assertFalse(result.passed)

    def test_a_runner_that_calls_the_check_is_accepted(self):
        reply = self.reply(argv=["make", "check-logging"])
        reply["change"]["files"] = [
            {"path": "scripts/check-logging.sh", "action": "create",
             "contents": "#!/bin/sh\nexit 0\n"},
            {"path": "Makefile", "action": "create",
             "contents": "check-logging:\n\tsh scripts/check-logging.sh\n"}]
        self.assertEqual(reasoning.SUCCESS,
                         selection.interpret(reply, {"mechanisms": []},
                                             _NoPaths())["outcome"])

    def test_a_change_that_touches_no_runner_is_refused(self):
        reply = self.reply()
        reply["change"]["files"] = [f for f in reply["change"]["files"]
                                    if f["path"] != "Makefile"]
        out = selection.interpret(reply, {"mechanisms": []}, _NoPaths())
        self.assertEqual(reasoning.INVALID_OUTPUT, out["outcome"])
        self.assertIn("wires it into nothing", out["failure"])

    def test_a_new_mechanism_is_still_validated_before_it_is_proposed(self):
        """It reaches §14 like any other candidate: a check that does not catch
        the mistake is rejected however new it is."""
        rec = self.propose(FakeEngine(self.reply()), inventory={"mechanisms": []})
        self.assertEqual("VALIDATION_REJECTED", rec["outcome"])
        self.assert_conformant(rec)


class _NoPaths:
    def __contains__(self, key):
        return False

    def sha256(self, path):
        return None


class NoRepoHoneDependence(Fixture):
    """ARCH §25 — an accepted improvement must survive RepoHone's removal."""

    def test_a_change_referencing_repohone_is_rejected(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"][0]["edits"] = [{"append": "command = 'repohone check'"}]
        result = self.propose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])
        self.assertIn("depend on RepoHone", result["failure"]["reason"])

    def test_a_mention_the_file_already_had_is_not_a_new_dependence(self):
        (self.repo / "pyproject.toml").write_text(
            "# telemetry by RepoHone, installed per developer\n"
            "[tool.ruff.lint]\nselect = ['E']\n")
        out = selection.interpret(json.loads(json.dumps(GOOD_REPLY)), INVENTORY,
                                  proposal._existing_paths(self.repo))
        self.assertEqual(reasoning.SUCCESS, out["outcome"], out.get("failure"))

    def test_a_path_escaping_the_repository_is_rejected(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"][0]["path"] = "../outside.toml"
        result = self.propose(FakeEngine(reply))
        self.assertEqual("INVALID_OUTPUT", result["outcome"])

    def test_every_proposal_declares_no_runtime_dependence(self):
        for engine in (FakeEngine(), FakeEngine("prose")):
            self.assertFalse(self.propose(engine)["repohone_runtime_required"])


class TerminalStates(Fixture):
    def test_success_is_a_candidate_with_validation(self):
        result = self.propose()
        self.assertEqual("SUCCESS", result["outcome"])
        self.assertEqual("candidate", result["state"])
        self.assertTrue(result["validation"]["rejects_problematic"])
        self.assertTrue(result["validation"]["accepts_repaired"])
        self.assert_conformant(result)

    def _rejected(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        session = self.session()
        session["turns"][0]["stop_events"][0]["snapshot"]["tree"] = self.good_tree
        return self.propose(FakeEngine(reply), session=session)

    def test_a_non_discriminating_candidate_is_validation_rejected(self):
        result = self._rejected()
        self.assertEqual("VALIDATION_REJECTED", result["outcome"])
        self.assertEqual("rejected", result["state"])
        self.assert_conformant(result)

    def test_a_rejected_candidate_still_says_what_was_tested(self):
        """Regression: §12.3 keeps a rejected candidate as failed-analysis
        evidence, but the record nulled mechanism, property and change — so it
        could not say what had been disproved, and nothing stopped the next run
        proposing it again. The previous test asserted the nulls."""
        result = self._rejected()
        self.assertTrue(result["required_property"])
        self.assertTrue(result["mechanism"])
        self.assertTrue(result["mechanism"]["name"])
        self.assertTrue(result["change"])
        self.assertTrue(all(f["sha256"] for f in result["change"]["files"]))
        self.assert_conformant(result)

    def test_a_rejected_candidate_keeps_no_staged_contents(self):
        """Identity, not payload: hashes are enough to recognise the candidate
        again, and a rejected change must not sit on disk ready to apply."""
        result = self._rejected()
        blob = json.dumps(result)
        self.assertNotIn("contents", blob)
        self.assertIsNone(proposal.staged(self.checkout, result["proposal_id"]))

    def test_a_rejected_candidate_cannot_be_applied(self):
        result = self._rejected()
        with self.assertRaises(proposal.NotApplicable):
            proposal.apply(self.repo, self.checkout, result["proposal_id"], {})

    def test_an_outcome_with_no_candidate_keeps_nothing(self):
        """Only a candidate that was actually built is retained."""
        result = proposal.propose(self.repo, self.checkout,
                                  dict(DIAGNOSIS, outcome="INSUFFICIENT_EVIDENCE"),
                                  INVENTORY, self.session(), FakeEngine())
        self.assertIsNone(result["mechanism"])
        self.assertIsNone(result["change"])
        self.assertIsNone(result["required_property"])

    def test_a_non_success_diagnosis_produces_nothing(self):
        result = proposal.propose(self.repo, self.checkout,
                                  dict(DIAGNOSIS, outcome="INSUFFICIENT_EVIDENCE"),
                                  INVENTORY, self.session(), FakeEngine())
        self.assertEqual("INSUFFICIENT_EVIDENCE", result["outcome"])
        self.assertEqual("rejected", result["state"])

    def test_model_unavailable_is_recorded(self):
        result = self.propose(FakeEngine(raises=reasoning.Unavailable("no engine")))
        self.assertEqual("MODEL_UNAVAILABLE", result["outcome"])
        self.assert_conformant(result)

    def test_prose_is_invalid_output(self):
        self.assertEqual("INVALID_OUTPUT", self.propose(FakeEngine("no json"))["outcome"])

    def test_every_outcome_writes_a_record(self):
        for engine in (FakeEngine(), FakeEngine("prose"),
                       FakeEngine(raises=reasoning.Interrupted("t"))):
            self.propose(engine)
        self.assertEqual(3, len(proposal.load_all(self.checkout)))


class ApplyAndRollback(Fixture):
    """ARCH §27 — how to undo it is recorded, not promised."""

    def _applied(self):
        rec = self.propose()
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        return proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)

    def test_applying_starts_the_experiment(self):
        """ARCH §26 — an applied improvement is under observation, not finished."""
        original = (self.repo / "pyproject.toml").read_text()
        rec = self._applied()
        self.assertEqual("observing", rec["state"])
        self.assertIn("not yet measured", rec["measurement"]["signal"])
        self.assertIn("T20", (self.repo / "pyproject.toml").read_text())
        backup = proposal.backup_dir(self.checkout, rec["proposal_id"]) / "pyproject.toml"
        self.assertEqual(original, backup.read_text())

    def test_rollback_restores_the_original(self):
        original = (self.repo / "pyproject.toml").read_text()
        rec = self._applied()
        proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertEqual(original, (self.repo / "pyproject.toml").read_text())

    def test_rollback_refuses_to_discard_a_later_edit(self):
        rec = self._applied()
        (self.repo / "pyproject.toml").write_text("# edited by hand\n")
        with self.assertRaises(proposal.Conflict):
            proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertEqual("# edited by hand\n",
                         (self.repo / "pyproject.toml").read_text())

    def test_a_rejected_proposal_cannot_be_applied(self):
        rec = self.propose(FakeEngine("prose"))
        with self.assertRaises(proposal.NotApplicable):
            proposal.apply(self.repo, self.checkout, rec["proposal_id"], {})

    def test_applying_contents_that_do_not_match_the_review_is_refused(self):
        rec = self.propose()
        with self.assertRaises(proposal.Conflict):
            proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                           {"pyproject.toml": "something else entirely\n"})

    def test_a_created_file_is_removed_on_rollback(self):
        reply = json.loads(json.dumps(GOOD_REPLY))
        reply["change"]["files"] = [{"path": "ruff.toml", "action": "create",
                                     "contents": "select = ['T20']\n"}]
        rec = self.propose(FakeEngine(reply))
        contents = proposal.staged(self.checkout, rec["proposal_id"])
        proposal.apply(self.repo, self.checkout, rec["proposal_id"], contents)
        self.assertTrue((self.repo / "ruff.toml").is_file())
        proposal.rollback(self.repo, self.checkout, rec["proposal_id"])
        self.assertFalse((self.repo / "ruff.toml").is_file())


class ProposalPublicationIsRecoverable(Fixture):
    def test_a_staging_failure_never_publishes_success(self):
        real = proposal.stage

        def fail(*args, **kwargs):
            raise OSError("simulated staging failure")

        proposal.stage = fail
        try:
            with self.assertRaises(OSError):
                self.propose()
        finally:
            proposal.stage = real
        self.assertEqual([], proposal.load_all(self.checkout))

    def test_a_record_failure_removes_the_prepared_payload(self):
        real = proposal.save

        def fail(*args, **kwargs):
            raise OSError("simulated record failure")

        proposal.save = fail
        try:
            with self.assertRaises(OSError):
                self.propose()
        finally:
            proposal.save = real
        self.assertEqual([], proposal.load_all(self.checkout))
        self.assertEqual([], list(proposal.directory(self.checkout).glob("*/staged.json")))

    def test_doctor_rejects_an_applicable_proposal_without_its_payload(self):
        rec = self.approved({"new.txt": "new\n"}, proposal_id="prop_0badcafe")
        proposal.staged_path(self.checkout, rec["proposal_id"]).unlink()
        problems = doctor._unsupported_records(self.checkout)
        self.assertTrue(any("has no staged contents" in item for item in problems))


class HistoricalProposalMigration(Fixture):
    def test_every_supported_additive_proposal_shape_migrates_conservatively(self):
        """Derived from the runtime, so a contract bump cannot drop a version."""
        def version(value):
            return tuple(int(part) for part in value.split("."))

        contracts = sorted((c for c in artifacts.SUPPORTED_CONTRACTS
                            if version(c) >= (1, 7)), key=version)
        self.assertIn(CONTRACT, contracts)
        self.assertEqual(len(contracts), len(set(contracts)))
        for contract in contracts:
            with self.subTest(contract=contract):
                proposal_id = "prop_" + contract.replace(".", "") + "000000"
                rec = self.approved({"new.txt": "new\n"}, proposal_id=proposal_id)
                rec["contract_version"] = contract
                version = tuple(int(part) for part in contract.split("."))
                if version < (1, 8):
                    rec["validation"].pop("estimate", None)
                if version < (1, 15):
                    rec["privacy"].pop("egress", None)
                if version < (1, 13):
                    for entry in rec["change"]["files"]:
                        entry.pop("baseline_mode", None)
                artifacts.atomic_write(proposal.path_for(self.checkout, proposal_id), rec)

                loaded = proposal.load(self.checkout, proposal_id)
                self.assertIsNone(loaded["validation"]["estimate"])
                self.assertIsNone(loaded["privacy"]["egress"])
                self.assertIsNone(loaded["change"]["files"][0]["baseline_mode"])

    def test_a_readable_old_candidate_without_mode_evidence_must_be_regenerated(self):
        rec = self.approved(
            {"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"},
            proposal_id="prop_1120cafe")
        rec["contract_version"] = "1.12"
        rec["change"]["files"][0].pop("baseline_mode")
        artifacts.atomic_write(proposal.path_for(self.checkout, rec["proposal_id"]), rec)
        loaded = proposal.load(self.checkout, rec["proposal_id"])
        self.assertIsNone(loaded["change"]["files"][0]["baseline_mode"])
        with self.assertRaises(proposal.NotApplicable) as caught:
            proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                           proposal.staged(self.checkout, rec["proposal_id"]))
        self.assertIn("mode-aware approval", str(caught.exception))
        loaded["change"]["files"][0]["baseline_mode"] = 0o644
        self.assertIn("mode-aware approval", proposal.application_blocker(loaded))

    def test_a_readable_old_candidate_cannot_bypass_egress_disclosure(self):
        rec = self.approved({"new.txt": "new\n"}, proposal_id="prop_1140cafe")
        rec["contract_version"] = "1.14"
        rec["privacy"] = {"adds_model_egress": True, "note": "historical"}
        artifacts.atomic_write(proposal.path_for(self.checkout, rec["proposal_id"]), rec)
        loaded = proposal.load(self.checkout, rec["proposal_id"])
        self.assertIsNone(loaded["privacy"]["egress"])
        with self.assertRaises(proposal.NotApplicable) as caught:
            proposal.apply(self.repo, self.checkout, rec["proposal_id"],
                           proposal.staged(self.checkout, rec["proposal_id"]))
        self.assertIn("egress disclosure", str(caught.exception))
        loaded["privacy"]["egress"] = {
            "provider": "claimed", "model": "claimed", "data": ["claimed"],
            "cost": "claimed"}
        self.assertIn("egress disclosure", proposal.application_blocker(loaded))


class ProposalShape(Fixture):
    """LEARNING_PLAN §13 — every proposal states its evidence, risk and rollback."""

    def test_a_successful_proposal_states_everything_required(self):
        rec = self.propose()
        self.assertTrue(rec["mechanism"]["why_existing_did_not_help"])
        self.assertTrue(rec["mechanism"]["alternatives_considered"])
        self.assertTrue(rec["expected_effect"])
        self.assertTrue(rec["friction"])
        self.assertIn("adds_model_egress", rec["privacy"])
        self.assertEqual("fp_0123456789abcdef", rec["measurement"]["target_fingerprint"])
        self.assertEqual(3, rec["measurement"]["baseline_sessions"])

    def test_the_target_fingerprint_travels_with_the_proposal(self):
        rec = self.propose()
        self.assertEqual("print-instead-of-logger", rec["fingerprint"]["behavior"])
        self.assertEqual(3, rec["fingerprint"]["sessions"])

    def test_versions_are_recorded(self):
        rec = self.propose()
        self.assertEqual("repohone.proposal/v1", rec["schema"])
        self.assertEqual(CONTRACT, rec["contract_version"])


if __name__ == "__main__":
    unittest.main()
