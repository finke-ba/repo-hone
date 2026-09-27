"""Phase 5b — RepoHone from the chat.

The approval that counts is the host's prompt, raised by RepoHone's own
PreToolUse hook; a gated command run from the agent's shell without it is
refused. What RepoHone prints there reaches the model provider."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

CORE = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from helpers import canonical_occurrence, ensure_session
from repohone import (
    announcer,
    approvals,
    cli,
    doctor,
    gate,
    identity,
    profile,
    proposal,
    rule,
    targets,
)
from repohone.adapters import claude
from test_phase5 import Fixture as ProposalFixture

HOOK = CORE / "repohone_hook.py"
LAUNCHER = "/Users/dev/.claude/skills/repohone/bin/repohone"
HOST = "host-session-1"


def skills():
    return claude.skill_sources()


class TheGateDecides(unittest.TestCase):
    """Each gated action asks; everything else passes; `hook` from a shell is refused."""

    ASKED = [
        ["diagnose", "rh_1", "--yes"], ["diagnose", "--candidate", "rule_1", "--yes"],
        ["propose", "dx_1", "--yes"], ["apply", "prop_1", "--yes"],
        ["rollback", "prop_1", "--yes"], ["init"], ["init", "--force"],
        ["deinit", "--yes"], ["deinit", "--purge", "--yes"],
        ["mechanisms", "--probe", "--yes"], ["mechanisms", "--probe", "--only", "ruff"],
        ["mechanisms", "--probe", "--only=ruff"], ["bootstrap", "--include-remote"],
        ["uninstall", "--yes"], ["uninstall", "--purge", "--yes"], ["install"],
    ]
    PASSED = [
        ["status"], ["status", "--json"], ["doctor"], ["list"], ["rules", "--json"],
        ["rule", "we always use the logger"], ["diagnoses"], ["patterns"], ["proposals"],
        ["diagnose", "--show"], ["diagnose", "--candidate", "rule_1", "--show"],
        ["propose", "--show"], ["apply"], ["apply", "prop_1"], ["rollback"],
        ["deinit"], ["deinit", "--dry-run"], ["deinit", "--dry-run", "--yes"],
        ["mechanisms"], ["mechanisms", "--probe"], ["mechanisms", "--cached"],
        ["bootstrap"], ["bootstrap", "--all"], ["uninstall", "--dry-run", "--yes"],
        ["show", "rh_1"], ["--version"],
    ]

    NEVER_GATED = {"status", "doctor", "list", "rules", "rule", "diagnoses", "patterns",
                   "proposals", "show"}

    def test_every_command_is_gated_or_declared_safe(self):
        """A new command fails here until someone decides whether it needs approval."""
        commands = set(cli.build_parser()._subparsers._group_actions[0].choices)
        gated = {argv[0] for argv in self.ASKED} | {"hook"}
        self.assertEqual(set(), gated & self.NEVER_GATED)
        self.assertEqual(commands, gated | self.NEVER_GATED)
        for command in self.NEVER_GATED:
            self.assertIsNone(gate.decide([command, "x", "--yes"]), command)

    def test_every_gated_action_asks(self):
        for argv in self.ASKED:
            decision = gate.decide(argv)
            self.assertEqual(gate.ASK, decision and decision[0], argv)

    def test_everything_else_passes(self):
        for argv in self.PASSED:
            self.assertIsNone(gate.decide(argv), argv)

    def test_hook_events_are_never_taken_from_a_shell(self):
        for command in ("repohone hook < x.json", "echo {} | python3 -m repohone.hook",
                        f"{LAUNCHER} hook", "repohone-hook"):
            decision, asked = gate.for_command(command)
            self.assertEqual(gate.DENY, decision[0], command)
            self.assertEqual([], asked)

    def test_every_spelling_of_repohone_is_recognised(self):
        for program in (LAUNCHER, "~/.claude/skills/repohone/bin/repohone",
                        "$HOME/.claude/skills/repohone/bin/repohone", "repohone",
                        "'/Users/a dev/.claude/skills/repohone/bin/repohone'",
                        "python3 -m repohone", "python3.12 -m repohone.cli",
                        "uv run --with jsonschema python -m repohone.cli", "uvx repohone",
                        "env REPOHONE_DATA_DIR=/x repohone", "nohup repohone"):
            decision, asked = gate.for_command(f"{program} apply prop_1 --yes")
            self.assertEqual(gate.ASK, decision and decision[0], program)
            self.assertEqual([["apply", "prop_1", "--yes"]], asked, program)

    def test_shell_around_the_command_does_not_hide_it(self):
        for command in (f"cd /r && {LAUNCHER} apply prop_1 --yes",
                        f"X=1 {LAUNCHER} apply prop_1 --yes 2>&1 | tail -5",
                        f"{LAUNCHER} apply prop_1 --yes > out.txt; echo done",
                        f"({LAUNCHER} apply prop_1 --yes)",
                        f"true\n{LAUNCHER} apply prop_1 --yes"):
            decision, asked = gate.for_command(command)
            self.assertEqual(gate.ASK, decision and decision[0], command)
            self.assertEqual([["apply", "prop_1", "--yes"]], asked, command)

    def test_what_it_cannot_follow_is_not_asked(self):
        """Nothing is recorded for it, so a gated command inside refuses by itself."""
        for command in ('eval "$(echo repohone apply p --yes)"', "sh -c 'repohone init'",
                        "bash -lc 'repohone init'", "echo `repohone init`",
                        "xargs repohone apply < ids", "python3 -c 'import repohone'",
                        "python3 -c 'from repohone.record_invariants import check_record'",
                        "python3 - <<'EOF'\nopen('core/repohone/reasoning.py')\nEOF",
                        "source repohone.sh", "repohone apply 'unbalanced --yes"):
            self.assertEqual((None, []), gate.for_command(command), command)

    def test_a_readable_line_in_a_heredoc_is_still_asked(self):
        decision, asked = gate.for_command("cat <<EOF | sh\nrepohone init\nEOF")
        self.assertEqual(gate.ASK, decision[0])
        self.assertEqual([["init"]], asked)

    def test_code_that_can_record_an_approval_is_asked(self):
        for command in ("python3 -c \"from repohone.cli import main; main(['hook'])\"",
                        "sh -c 'echo {} | repohone hook'", "echo `repohone-hook`",
                        "python3 -c 'from repohone import approvals'",
                        "python3 - <<'EOF'\nimport repohone.approvals\nEOF"):
            self.assertEqual(((gate.ASK, gate.MINTS), []), gate.for_command(command), command)

    def test_both_reasons_are_given_when_both_apply(self):
        decision, asked = gate.for_command("repohone apply p --yes; sh -c 'repohone hook'")
        self.assertIn(gate.decide(["apply", "p", "--yes"])[1], decision[1])
        self.assertIn(gate.MINTS, decision[1])
        self.assertEqual([["apply", "p", "--yes"]], asked)

    def test_a_mere_mention_passes(self):
        for command in ("cat .repohone/profile.yaml", "grep -rn repohone src",
                        "git log --grep repohone", "ls ~/.claude/skills/repohone",
                        "echo repohone apply --yes", "cat core/repohone/approvals.py",
                        "grep -n hook core/repohone/cli.py"):
            self.assertEqual((None, []), gate.for_command(command), command)

    def test_the_hook_output_is_the_documented_shape(self):
        decision, _ = gate.for_command("repohone init")
        self.assertEqual({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "ask",
            "permissionDecisionReason": decision[1]}}, json.loads(gate.hook_output(decision)))

    def test_no_reason_carries_a_path_or_a_foreign_character(self):
        """The reason reaches the model provider too."""
        for argv in self.ASKED + [["apply", "/Users/dev/secret;rm", "--yes"]]:
            reason = gate.decide(argv)[1]
            self.assertNotIn("/Users", reason, argv)
            self.assertNotIn("secret;rm", reason, argv)

    def test_abbreviations_cannot_slip_past(self):
        """argparse would read `--ye` as `--yes`; the gate reads what was written."""
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["apply", "prop_1", "--ye"])


class Approvals(unittest.TestCase):

    def setUp(self):
        tmp = Path(os.environ.get("TMPDIR", "/tmp"))
        self.data = tmp / f"rh-approvals-{os.getpid()}-{id(self)}"
        self.patch = mock.patch.dict(os.environ, {"REPOHONE_DATA_DIR": str(self.data)})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        import shutil
        shutil.rmtree(self.data, ignore_errors=True)

    def test_one_approval_is_used_once(self):
        approvals.mint(HOST, ["apply", "p", "--yes"])
        self.assertTrue(approvals.consume(HOST, ["apply", "p", "--yes"]))
        self.assertFalse(approvals.consume(HOST, ["apply", "p", "--yes"]))

    def test_it_is_bound_to_the_session_and_the_exact_command(self):
        approvals.mint(HOST, ["apply", "p", "--yes"])
        self.assertFalse(approvals.consume("another-session", ["apply", "p", "--yes"]))
        self.assertFalse(approvals.consume(HOST, ["apply", "q", "--yes"]))
        self.assertTrue(approvals.consume(HOST, ["apply", "p", "--yes"]))

    def test_two_racing_processes_cannot_both_use_it(self):
        approvals.mint(HOST, ["init"])
        barrier, won = threading.Barrier(8), []

        def race():
            barrier.wait()
            won.append(approvals.consume(HOST, ["init"]))
        threads = [threading.Thread(target=race) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(1, won.count(True))

    def test_an_hour_old_approval_is_void(self):
        approvals.mint(HOST, ["init"])
        later = time.time() + approvals.EXPIRY_S + 1
        with mock.patch.object(approvals.time, "time", return_value=later):
            self.assertFalse(approvals.consume(HOST, ["init"]))

    def test_voiding_clears_the_session(self):
        approvals.mint(HOST, ["init"])
        approvals.void(HOST)
        self.assertFalse(approvals.consume(HOST, ["init"]))


class Chat(ProposalFixture):
    """A repository with RepoHone on, a fake `claude` that logs any model call, and
    helpers to fire hooks and run commands as the agent's shell would."""

    def setUp(self):
        super().setUp()
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.calls = self.tmp / "model-calls.log"
        fake = self.bin / "claude"
        fake.write_text(f'#!/bin/sh\necho "$@" >> {self.calls}\necho "{{}}"\n')
        fake.chmod(0o755)
        self.env = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}")
        self.env.pop("CLAUDE_CODE_SESSION_ID", None)

    def fire(self, name, host=HOST, **kw):
        payload = {"hook_event_name": name, "session_id": host, "cwd": str(self.repo)}
        payload.update(kw)
        return subprocess.run([sys.executable, str(HOOK), "hook"], input=json.dumps(payload),
                              cwd=str(self.repo), capture_output=True, text=True,
                              env=self.env)

    def bash(self, command, host=HOST):
        """The agent's Bash call, as the host shows it to PreToolUse."""
        return self.fire("PreToolUse", host, prompt_id="p9", tool_name="Bash",
                         tool_use_id="toolu_1", tool_input={"command": command})

    def cli(self, argv, agent=True, host=HOST):
        env = {"CLAUDE_CODE_SESSION_ID": host} if agent else {}
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env), \
                mock.patch.dict(os.environ, {"PATH": self.env["PATH"]}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if not agent:
                os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def captured_session(self, host=HOST, prompt="add a feature", close=True,
                         correction="no, we always use the logger here"):
        self.fire("UserPromptSubmit", host, prompt_id="p1", prompt=prompt)
        self.fire("Stop", host, prompt_id="p1", last_assistant_message="done")
        self.fire("UserPromptSubmit", host, prompt_id="p2", prompt=correction)
        if close:
            self.fire("Stop", host, prompt_id="p2", last_assistant_message="fixed")
        return identity.session_id(claude.NAME, host)

    def model_calls(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0


class TheCliRefusesWhatTheGateDidNotSee(Chat):

    def test_apply_without_an_approval_changes_nothing(self):
        self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        before = (self.repo / "pyproject.toml").read_text()
        code, out, err = self.cli(["apply", "prop_d1ec7000", "--yes", "--path", str(self.repo)])
        self.assertEqual(3, code)
        self.assertIn("/repohone:apply", err)
        self.assertEqual(before, (self.repo / "pyproject.toml").read_text())
        self.assertEqual("approved", proposal.load(self.checkout, "prop_d1ec7000")["state"])

    def test_the_gate_asks_and_the_approved_command_runs(self):
        self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        argv = ["apply", "prop_d1ec7000", "--yes", "--path", str(self.repo)]
        answer = self.bash(f"{LAUNCHER} {shlex.join(argv)}")
        decision = json.loads(answer.stdout)["hookSpecificOutput"]
        self.assertEqual("ask", decision["permissionDecision"])
        code, out, err = self.cli(argv)
        self.assertEqual(0, code, err)
        self.assertIn("T20", (self.repo / "pyproject.toml").read_text())

    def test_an_approval_does_not_carry_to_another_command(self):
        self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        self.bash(f"{LAUNCHER} apply prop_d1ec7000 --yes --path {self.repo}")
        code, _, _ = self.cli(["apply", "prop_d1ec7000", "--yes"])
        self.assertEqual(3, code, "a different argv used the approval")

    def test_a_new_prompt_voids_an_unused_approval(self):
        """After a denial nothing consumed it; the next request starts clean."""
        self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        argv = ["apply", "prop_d1ec7000", "--yes", "--path", str(self.repo)]
        self.bash(f"repohone {shlex.join(argv)}")
        self.fire("UserPromptSubmit", prompt_id="p10", prompt="never mind")
        self.assertEqual(3, self.cli(argv)[0])

    def test_session_end_voids_them_too(self):
        argv = ["init", "--path", str(self.repo)]
        self.bash(f"repohone {shlex.join(argv)}")
        self.fire("SessionEnd", reason="exit")
        self.assertEqual(3, self.cli(argv)[0])

    def test_diagnosis_without_an_approval_reaches_no_model(self):
        session = self.captured_session()
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        code, _, err = self.cli(["diagnose", "--candidate", candidate["candidate_id"], "--yes",
                                 "--path", str(self.repo)])
        self.assertEqual(3, code, err)
        self.assertEqual(0, self.model_calls())

    def test_init_without_an_approval_records_no_consent(self):
        profile.withdraw(self.repo)
        code, _, _ = self.cli(["init", "--path", str(self.repo)])
        self.assertEqual(3, code)
        self.assertIsNone(profile.accepted(self.repo, profile.load(self.repo)))

    def test_the_developers_terminal_needs_no_approval(self):
        self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        code, _, err = self.cli(["apply", "prop_d1ec7000", "--yes", "--path", str(self.repo)],
                                agent=False)
        self.assertEqual(0, code, err)

    def test_the_gate_runs_whatever_the_consent_state(self):
        """`init` is approved exactly when capture is off."""
        profile.withdraw(self.repo)
        answer = self.bash("repohone init")
        self.assertEqual("ask", json.loads(answer.stdout)["hookSpecificOutput"]
                         ["permissionDecision"])
        (self.repo / ".repohone" / "profile.yaml").unlink()
        answer = self.bash("repohone init")
        self.assertIn('"ask"', answer.stdout)

    def test_an_ordinary_command_leaves_nothing_behind(self):
        answer = self.bash("ls -la && git status")
        self.assertEqual("", answer.stdout)
        self.assertFalse(approvals.root().exists())

    def test_the_gate_reaches_no_model(self):
        self.bash("repohone diagnose rh_x --yes")
        self.bash("repohone propose dx_x --yes")
        self.assertEqual(0, self.model_calls())


class DefaultTargets(Chat):

    def test_a_rule_marked_in_this_session_is_the_default(self):
        session = self.captured_session()
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        pick = targets.for_diagnosis(self.checkout, claude.NAME, HOST)
        self.assertEqual((candidate["candidate_id"], True), (pick.chosen, pick.candidate))

    def test_a_rule_whose_turn_is_in_progress_waits(self):
        session = self.captured_session(close=False)
        rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        pick = targets.for_diagnosis(self.checkout, claude.NAME, HOST)
        self.assertIsNone(pick.chosen)
        self.assertIn("in progress", pick.why)

    def test_without_a_rule_it_is_this_session(self):
        session = self.captured_session()
        pick = targets.for_diagnosis(self.checkout, claude.NAME, HOST)
        self.assertEqual((session, False), (pick.chosen, pick.candidate))

    def test_two_waiting_rules_are_listed_not_chosen(self):
        session = self.captured_session()
        rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        rule.mark(self.repo, self.checkout, session, "never print", 2)
        pick = targets.for_diagnosis(self.checkout, claude.NAME, HOST)
        self.assertIsNone(pick.chosen)
        self.assertEqual(2, len(pick.choices))

    def test_another_sessions_rule_is_not_this_sessions_default(self):
        other = self.captured_session(host="host-other")
        rule.mark(self.repo, self.checkout, other, "always the logger", 2)
        mine = self.captured_session()
        self.assertEqual(mine, targets.for_diagnosis(self.checkout, claude.NAME, HOST).chosen)

    def test_a_session_not_captured_here_is_never_a_default(self):
        pick = targets.for_diagnosis(self.checkout, claude.NAME, "never-captured-here")
        self.assertIsNone(pick.chosen)

    def test_propose_takes_the_only_diagnosis_without_a_proposal(self):
        session = ensure_session(self.checkout, "fixture-1")
        canonical_occurrence(self.checkout, None, "MISSING_CONTEXT", "logging",
                             "print-instead-of-logger", session)
        pick = targets.for_proposal(self.checkout)
        self.assertIsNotNone(pick.chosen)
        session2 = ensure_session(self.checkout, "fixture-2")
        canonical_occurrence(self.checkout, None, "MISSING_CONTEXT", "http",
                             "missing-timeout", session2)
        self.assertIsNone(targets.for_proposal(self.checkout).chosen)

    def test_apply_takes_the_only_validated_proposal(self):
        self.approved({"pyproject.toml": "x = 1\n"})
        self.assertEqual("prop_d1ec7000", targets.for_apply(self.checkout).chosen)
        self.approved({"pyproject.toml": "x = 2\n"}, proposal_id="prop_d1ec7001")
        pick = targets.for_apply(self.checkout)
        self.assertIsNone(pick.chosen)
        self.assertEqual({"prop_d1ec7000", "prop_d1ec7001"}, set(pick.choices))

    def test_rollback_takes_the_most_recently_applied(self):
        first = self.approved({"pyproject.toml": "x = 1\n"})
        self.apply_it(first)
        time.sleep(1.1)
        second = self.approved({"src/other.py": "y = 1\n"}, proposal_id="prop_d1ec7001")
        self.apply_it(second)
        pick = targets.for_rollback(self.checkout)
        self.assertEqual("prop_d1ec7001", pick.chosen)
        self.assertEqual(["prop_d1ec7000"], pick.choices)

    def test_a_preview_says_which_it_picked(self):
        self.approved({"pyproject.toml": "x = 1\n"})
        code, out, err = self.cli(["apply", "--path", str(self.repo)], agent=False)
        self.assertEqual(3, code)
        self.assertIn("target: prop_d1ec7000", err)
        self.assertIn("apply prop_d1ec7000 --yes", out)

    def test_yes_without_a_named_target_is_refused(self):
        self.approved({"pyproject.toml": "x = 1\n"})
        for argv in (["apply", "--yes"], ["rollback", "--yes"], ["propose", "--yes"],
                     ["diagnose", "--yes"]):
            code, _, err = self.cli(argv + ["--path", str(self.repo)], agent=False)
            self.assertEqual(2, code, argv)
            self.assertIn("needs its target named", err, argv)
        self.assertEqual("approved", proposal.load(self.checkout, "prop_d1ec7000")["state"])


class RollbackPreviews(Chat):

    def test_without_yes_it_changes_nothing(self):
        rec = self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        applied = self.apply_it(rec)["state"]
        code, out, _ = self.cli(["rollback", "prop_d1ec7000", "--path", str(self.repo)],
                                agent=False)
        self.assertEqual(3, code)
        self.assertIn("restore pyproject.toml", out)
        self.assertIn("T20", (self.repo / "pyproject.toml").read_text())
        self.assertEqual(applied, proposal.load(self.checkout, "prop_d1ec7000")["state"])

    def test_it_says_when_a_file_changed_since(self):
        rec = self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        self.apply_it(rec)
        (self.repo / "pyproject.toml").write_text("edited later\n")
        code, out, _ = self.cli(["rollback", "--path", str(self.repo)], agent=False)
        self.assertIn("REFUSED", out)

    def test_with_yes_it_rolls_back(self):
        rec = self.approved({"pyproject.toml": "[tool.ruff.lint]\nselect = ['E','T20']\n"})
        self.apply_it(rec)
        code, out, _ = self.cli(["rollback", "prop_d1ec7000", "--yes", "--path",
                                 str(self.repo)], agent=False)
        self.assertEqual(0, code)
        self.assertNotIn("T20", (self.repo / "pyproject.toml").read_text())


class StructuredOutput(Chat):
    """Skills find things through these fields; rewording the text cannot break them."""

    FIELDS = {
        "status": {"core_version", "contract_version", "repository", "state", "capture",
                   "checkout_id", "sessions", "problems"},
        "list": {"session_id", "turns", "state", "started_at", "unreadable"},
        "rules": {"candidate_id", "state", "session_id", "corrective_turn", "statement",
                  "diagnosis_id", "created_at"},
        "diagnoses": {"diagnosis_id", "outcome", "session_id", "created_at", "candidate_id",
                      "area", "behavior", "sessions", "required_property", "proposals"},
        "proposals": {"proposal_id", "diagnosis_id", "state", "outcome", "mechanism",
                      "required_property", "files", "created_at", "applied_at"},
    }

    def load(self, command):
        code, out, err = self.cli([command, "--json", "--path", str(self.repo)], agent=False)
        self.assertIn(code, (0, 1), err)
        return json.loads(out)

    def test_each_command_has_its_pinned_fields(self):
        session = self.captured_session()
        rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        canonical_occurrence(self.checkout, None, "MISSING_CONTEXT", "logging",
                             "print-instead-of-logger", ensure_session(self.checkout, "f-1"))
        self.approved({"pyproject.toml": "x = 1\n"})
        self.assertEqual(self.FIELDS["status"], set(self.load("status")))
        for command, key in (("list", "sessions"), ("rules", "rules"),
                             ("diagnoses", "diagnoses"), ("proposals", "proposals")):
            rows = self.load(command)[key]
            self.assertTrue(rows, command)
            self.assertEqual(self.FIELDS[command], set(rows[0]), command)

    def test_doctor_json_is_what_section_31_promises(self):
        doc = self.load("doctor")
        self.assertIn(doc["worst"], (doctor.OK, doctor.WARN, doctor.FAIL))
        self.assertEqual({"check", "status", "detail"}, set(doc["checks"][0]))

    def test_a_diagnosis_links_its_rule_and_proposals(self):
        session = self.captured_session()
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        dx_session = ensure_session(self.checkout, "f-1")
        canonical_occurrence(self.checkout, None, "MISSING_CONTEXT", "logging",
                             "print-instead-of-logger", dx_session, diagnosis_id="dx_feed0001")
        rule.link_diagnosis(self.checkout, candidate["candidate_id"], "dx_feed0001", True)
        rec = self.approved({"pyproject.toml": "x = 1\n"})
        rec["diagnosis_id"] = "dx_feed0001"
        proposal.save(self.checkout, rec)
        row = next(r for r in self.load("diagnoses")["diagnoses"]
                   if r["diagnosis_id"] == "dx_feed0001")
        self.assertEqual(candidate["candidate_id"], row["candidate_id"])
        self.assertEqual(["prop_d1ec7000"], row["proposals"])


class WhatReachesTheChat(Chat):
    """Output in the agent's shell is egress: no absolute path, no session content."""

    MARKER = "MARKER-7Q-DO-NOT-SEND"

    def forms(self):
        forms = {str(self.repo), str(self.repo.resolve()), os.environ["REPOHONE_DATA_DIR"]}
        return {f for f in forms} | {f.replace("/private", "", 1) for f in forms}

    def test_no_read_only_command_prints_a_path_or_a_prompt(self):
        session = self.captured_session(correction=f"no, {self.MARKER}, always use the logger")
        rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        self.approved({"pyproject.toml": "x = 1\n"})
        commands = [["status"], ["status", "--json"], ["doctor"], ["doctor", "--json"],
                    ["list"], ["list", "--json"], ["rules"], ["rules", "--json"],
                    ["diagnoses", "--json"], ["proposals", "--json"], ["patterns"],
                    ["deinit", "--dry-run"], ["apply"], ["bootstrap"], ["mechanisms"]]
        for argv in commands:
            code, out, err = self.cli(argv + ["--path", str(self.repo)])
            text = out + err
            for form in self.forms():
                self.assertNotIn(form, text, argv)
            self.assertNotIn(self.MARKER, text, argv)

    def test_the_terminal_still_sees_them(self):
        """Proves the check above is not vacuous."""
        code, out, _ = self.cli(["status", "--path", str(self.repo)], agent=False)
        self.assertIn(str(self.repo), out)

    def test_show_stays_in_the_terminal(self):
        session = self.captured_session(correction=f"no, {self.MARKER}, always use the logger")
        code, out, err = self.cli(["show", session, "--path", str(self.repo)])
        self.assertEqual(2, code)
        self.assertNotIn(self.MARKER, out + err)
        code, out, _ = self.cli(["show", session, "--path", str(self.repo)], agent=False)
        self.assertIn(self.MARKER, out)

    def test_a_preview_in_the_chat_summarises_and_sends_nothing(self):
        session = self.captured_session(correction=f"no, {self.MARKER}, always use the logger")
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger", 2)
        argv = ["diagnose", "--show", "--candidate", candidate["candidate_id"],
                "--path", str(self.repo)]
        code, out, err = self.cli(argv)
        self.assertEqual(0, code, err)
        self.assertIn("evidence items", out)
        self.assertIn("in a terminal", out)
        self.assertNotIn(self.MARKER, out + err)
        self.assertEqual(0, self.model_calls())
        code, out, _ = self.cli(argv, agent=False)
        self.assertIn(self.MARKER, out)


class SkillsAgreeWithTheGate(unittest.TestCase):

    def commands(self, text):
        """Every RepoHone command a skill names in code, placeholders and all."""
        found = []
        for span in re.findall(r"`([^`]*\{repohone\}[^`]*)`", text):
            for part in span.split("{repohone}")[1:]:
                found.append(shlex.split(part.split("|")[0].strip()))
        return found

    def test_each_command_is_read_only_or_asked(self):
        for name, text in skills().items():
            for argv in self.commands(text):
                decision = gate.decide(argv)
                self.assertIn(decision and decision[0], (None, gate.ASK), (name, argv))
                if "--yes" in argv:
                    self.assertEqual(gate.ASK, decision[0], (name, argv))

    def test_a_bang_line_or_allowed_tool_is_never_gated(self):
        """Neither passes through the gate: `!` never reaches PreToolUse (verified)."""
        for name, text in skills().items():
            bangs = re.findall(r"!`\{repohone\} ([^`]*)`", text)
            allowed = re.findall(r"Bash\(\{repohone\} ([^)]*)\)",
                                 claude.frontmatter(text).get("allowed-tools", ""))
            for command in bangs + [a.replace(":*", "") for a in allowed]:
                self.assertIsNone(gate.decide(shlex.split(command)), (name, command))

    def test_only_two_skills_can_be_started_by_the_agent(self):
        startable = {n for n, t in skills().items()
                     if claude.frontmatter(t).get("disable-model-invocation") != "true"}
        self.assertEqual(set(claude.AGENT_STARTABLE), startable)
        spent = sum(len(claude.frontmatter(skills()[n])["description"].encode()) for n in startable)
        self.assertLess(spent, 600, "every session pays for these descriptions")

    def test_every_command_but_three_has_a_chat_command(self):
        commands = set(cli.build_parser()._subparsers._group_actions[0].choices)
        slash = {n for n in skills() if n not in claude.AGENT_STARTABLE}
        reachable = {next((c for c, s in cli.SLASH.items() if s == n), n) for n in slash}
        self.assertEqual(commands - {"install", "show", "hook", "rule"}, reachable)

    def test_asking_in_words_runs_what_the_slash_commands_run(self):
        def gated(text):
            return {(argv[0], frozenset(w for w in argv if w.startswith("--")))
                    for argv in self.commands(text) if gate.decide(argv)}
        slash = set().union(*(gated(t) for n, t in skills().items()
                              if n not in claude.AGENT_STARTABLE))
        self.assertLessEqual(gated(skills()["ask"]), slash)

    def test_every_chat_command_named_anywhere_exists(self):
        texts = list(skills().values()) + [announcer.UNINITIALIZED_TEXT,
                                           announcer.INVALID_TEXT,
                                           announcer.NOT_ACCEPTED_TEXT,
                                           announcer.CHANGED_TEXT]
        for text in texts:
            for name in re.findall(r"/repohone:([a-z-]+)", text):
                self.assertIn(name, skills(), name)

    def test_apply_and_rollback_forbid_doing_it_by_hand(self):
        for name in ("apply", "rollback", "ask"):
            self.assertRegex(skills()[name], r"(?i)never (make|restore|edit)|not by editing|"
                                             r"never by editing", name)

    def test_the_gated_skills_leave_the_approval_to_the_host(self):
        for name, text in skills().items():
            if any(gate.decide(argv) for argv in self.commands(text)):
                self.assertIn("Do not ask in the chat before", text, name)
                self.assertIn("never retry", text, name)


class TheInstalledPlugin(Chat):

    def install(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(0, cli.cmd_install(argparse.Namespace(path=None)))

    def test_every_skill_is_installed_with_the_launcher_filled_in(self):
        self.install()
        installed = claude.installed_skills()
        self.assertEqual(set(skills()), set(installed))
        for name, text in installed.items():
            self.assertNotIn("{repohone}", text, name)
        self.assertIn(str(claude.plugin_dir() / "bin" / "repohone"), installed["status"])

    def test_a_skill_an_older_core_shipped_is_removed(self):
        self.install()
        stale = claude.plugin_dir() / "skills" / "retired"
        stale.mkdir()
        (stale / "SKILL.md").write_text("old")
        self.install()
        self.assertFalse(stale.exists())

    def test_doctor_counts_the_chat_commands(self):
        self.install()
        row = next(r for r in doctor.run(str(self.repo)) if r[0] == "chat commands")
        self.assertEqual(doctor.OK, row[1])
        self.assertIn("2 skills the agent may start", row[2])

    def test_uninstall_from_the_chat_completes_through_its_own_launcher(self):
        """It deletes the plugin its launcher runs from."""
        self.install()
        hooks = claude.plugin_hooks()
        launcher = claude.plugin_dir() / "bin" / "repohone"
        env = {k: v for k, v in self.env.items() if k != "PYTHONPATH"}
        done = subprocess.run([str(launcher), "uninstall", "--yes"], cwd=str(self.repo),
                              capture_output=True, text=True, env=env)
        self.assertEqual(0, done.returncode, done.stderr)
        self.assertFalse(claude.plugin_dir().exists())
        entry = hooks[0]
        after = subprocess.run([entry["command"], *entry["args"]], cwd=str(self.repo),
                               input=json.dumps({"hook_event_name": "Stop",
                                                 "session_id": HOST}),
                               capture_output=True, text=True, env=env)
        self.assertEqual((0, ""), (after.returncode, after.stdout))


if __name__ == "__main__":
    unittest.main()
