"""Phase 2 — the explicit "this is a project rule" path.

Local only. The developer names the session; nothing about it is inferred, and
marking a candidate never reaches a model.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

CORE = Path(__file__).resolve().parent.parent / "src"
ROOT = CORE.parent
sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from repohone import cli, identity, profile, rule
from repohone.adapters import claude

SCHEMA_PATH = CORE / "repohone" / "schemas" / "rule-candidate.v1.schema.json"
HOOK = CORE / "repohone_hook.py"

import jsonschema

VALIDATOR = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                          text=True, check=True).stdout.strip()


class Fixture(unittest.TestCase):
    """A repository with two captured sessions, so cross-association is testable."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-p2-"))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.data = self.tmp / "data"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        git(["init", "-q"], self.repo)
        git(["config", "user.email", "d@e.com"], self.repo)
        git(["config", "user.name", "D"], self.repo)
        (self.repo / "app.py").write_text('def run():\n    print("x")\n')
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "init"], self.repo)
        (self.repo / ".repohone").mkdir()
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        os.environ["REPOHONE_DATA_DIR"] = str(self.data)

        # A `claude` that logs every call: Phase 2 must never reach it.
        self.calls = self.tmp / "model-calls.log"
        fake = self.bin / "claude"
        fake.write_text(f'#!/bin/sh\necho "$@" >> {self.calls}\necho "{{}}"\n')
        fake.chmod(0o755)
        self.env = dict(os.environ, REPOHONE_DATA_DIR=str(self.data),
                        PATH=f"{self.bin}:{os.environ['PATH']}")
        self.checkout = identity.checkout_id(self.repo)

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fire(self, name, session, **kw):
        payload = {"hook_event_name": name, "session_id": session,
                   "cwd": str(self.repo)}
        payload.update(kw)
        subprocess.run([sys.executable, str(HOOK), "hook"], input=json.dumps(payload),
                       cwd=str(self.repo), capture_output=True, text=True,
                       env=self.env)

    def a_session(self, host_id, subject):
        self.fire("UserPromptSubmit", host_id, prompt_id="p1",
                  prompt=f"{subject}: add a feature")
        self.fire("Stop", host_id, prompt_id="p1",
                  last_assistant_message=f"did it in {subject}")
        self.fire("UserPromptSubmit", host_id, prompt_id="p2",
                  prompt=f"{subject}: no, always use the logger")
        self.fire("Stop", host_id, prompt_id="p2",
                  last_assistant_message=f"fixed {subject}")
        return identity.session_id("claude-code", host_id)

    def open_turn(self, host_id, prompt="no, we always use the logger here"):
        """A session whose second turn is in progress, as when the agent runs a
        command while answering the developer's correction."""
        self.fire("UserPromptSubmit", host_id, prompt_id="p1", prompt="add a feature")
        self.fire("Stop", host_id, prompt_id="p1", last_assistant_message="done")
        self.fire("UserPromptSubmit", host_id, prompt_id="p2", prompt=prompt)
        return identity.session_id("claude-code", host_id)

    def rule_from_agent(self, statement, host_id=None, turn=None):
        """`repohone rule` as the agent's shell runs it: the host session is in
        the environment, and nothing else identifies it."""
        env = {"CLAUDE_CODE_SESSION_ID": host_id} if host_id else {}
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(os.environ, env), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.cmd_rule(argparse.Namespace(path=str(self.repo), statement=statement,
                                                   session=None, turn=turn))
        return code, out.getvalue(), err.getvalue()

    def model_calls(self):
        return len(self.calls.read_text().splitlines()) if self.calls.exists() else 0


class MarkingARule(Fixture):

    def test_a_candidate_is_bound_to_the_session_that_was_named(self):
        alpha = self.a_session("s-alpha", "alpha")
        beta = self.a_session("s-beta", "beta")
        self.assertNotEqual(alpha, beta)
        candidate = rule.mark(self.repo, self.checkout, alpha, "always use the logger")
        self.assertEqual(alpha, candidate["session_id"])
        self.assertEqual(self.checkout, candidate["checkout_id"])
        self.assertEqual([candidate["candidate_id"]],
                         [c["candidate_id"] for c in rule.load_all(self.checkout)])

    def test_two_concurrent_sessions_cannot_cross_associate(self):
        alpha = self.a_session("s-alpha", "alpha")
        beta = self.a_session("s-beta", "beta")
        a = rule.mark(self.repo, self.checkout, alpha, "alpha's rule")
        b = rule.mark(self.repo, self.checkout, beta, "beta's rule")
        self.assertNotEqual(a["session_id"], b["session_id"])
        self.assertNotEqual(a["candidate_id"], b["candidate_id"])
        by_session = {c["session_id"]: c["statement"] for c in rule.load_all(self.checkout)}
        self.assertEqual({alpha: "alpha's rule", beta: "beta's rule"}, by_session)

    def test_a_session_from_another_checkout_does_not_resolve(self):
        self.a_session("s-alpha", "alpha")
        other = identity.session_id("claude-code", "belongs-elsewhere")
        with self.assertRaises(rule.UnknownSession):
            rule.mark(self.repo, self.checkout, other, "a rule")

    def test_missing_ids_fail_rather_than_defaulting(self):
        session = self.a_session("s-alpha", "alpha")
        for checkout, sid, statement in ((("", session, "r")), (self.checkout, "", "r"),
                                         (self.checkout, session, "  ")):
            with self.assertRaises(ValueError):
                rule.mark(self.repo, checkout, sid, statement)

    def test_a_turn_outside_the_session_is_refused(self):
        session = self.a_session("s-alpha", "alpha")
        with self.assertRaises(ValueError):
            rule.mark(self.repo, self.checkout, session, "a rule", corrective_turn=99)
        marked = rule.mark(self.repo, self.checkout, session, "a rule",
                           corrective_turn=2)
        self.assertEqual(2, marked["corrective_turn"])

    def test_an_unnamed_turn_stays_unidentified(self):
        session = self.a_session("s-alpha", "alpha")
        self.assertIsNone(
            rule.mark(self.repo, self.checkout, session, "a rule")["corrective_turn"])

    def test_the_candidate_conforms_to_its_schema(self):
        session = self.a_session("s-alpha", "alpha")
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger",
                              corrective_turn=2)
        errors = sorted(VALIDATOR.iter_errors(candidate), key=lambda e: e.path)
        self.assertEqual([], ["/".join(map(str, e.path)) + ": " + e.message
                              for e in errors])

    def test_evidence_is_recorded_as_refs_not_copies(self):
        session = self.a_session("s-alpha", "alpha")
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger")
        self.assertTrue(candidate["evidence_refs"])
        blob = json.dumps(candidate)
        self.assertNotIn("alpha: add a feature", blob,
                         "the candidate copied prompt text instead of pointing at it")

    def test_the_named_turn_anchors_the_evidence_that_is_recorded(self):
        """Regression: the turn was validated and stored, then ignored by
        selection — so a candidate marked against turn 1 recorded turn 2's
        evidence and Phase 3 reasoned from the wrong correction."""
        session = self.a_session("s-alpha", "alpha")
        first = rule.mark(self.repo, self.checkout, session, "always the logger",
                          corrective_turn=1)
        refs = [r["ref"] for r in first["evidence_refs"]]
        self.assertEqual(["correction"],
                         [r["kind"] for r in first["evidence_refs"]
                          if r["ref"] == "turn/1/prompt/1"], refs)

    def test_marking_reaches_no_model(self):
        session = self.a_session("s-alpha", "alpha")
        rule.mark(self.repo, self.checkout, session, "always the logger",
                  corrective_turn=2)
        self.assertEqual(0, self.model_calls(),
                         "marking a candidate called a model")


class ConsentSemantics(Fixture):
    """Option B in the plan: marking consents to analysing *this* candidate."""

    def test_marking_grants_analysis_of_this_candidate_only(self):
        session = self.a_session("s-alpha", "alpha")
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger")
        self.assertEqual({"analysis": True, "scope": "this candidate only",
                          "authorizes_apply": False}, candidate["consent"])

    def test_marking_never_authorizes_applying(self):
        session = self.a_session("s-alpha", "alpha")
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger")
        self.assertFalse(candidate["consent"]["authorizes_apply"])
        self.assertEqual(rule.MARKED, candidate["state"])
        self.assertIsNone(candidate["diagnosis_id"])

    def test_a_failed_diagnosis_leaves_the_candidate_marked(self):
        session = self.a_session("s-alpha", "alpha")
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger")
        rule.link_diagnosis(self.checkout, candidate["candidate_id"], "dx_1", False)
        again = rule.load(self.checkout, candidate["candidate_id"])
        self.assertEqual(rule.MARKED, again["state"])
        self.assertEqual("dx_1", again["diagnosis_id"])

    def test_a_successful_diagnosis_marks_it_analysed(self):
        session = self.a_session("s-alpha", "alpha")
        candidate = rule.mark(self.repo, self.checkout, session, "always the logger")
        rule.link_diagnosis(self.checkout, candidate["candidate_id"], "dx_2", True)
        self.assertEqual(rule.ANALYSED,
                         rule.load(self.checkout, candidate["candidate_id"])["state"])


class SessionIsNeverInferred(Fixture):
    """ARCHITECTURE §23: `session_id` is mandatory.

    Regression: the tool took the sole live session. A caller whose own session
    was never captured was handed somebody else's, and its rule was filed
    against evidence it had nothing to do with."""

    def test_a_missing_id_is_refused_even_with_one_live_session(self):
        self.a_session("s-alpha", "alpha")
        for blank in (None, "", "   "):
            with self.assertRaises(rule.UnknownSession):
                rule.require_session(blank)

    def test_a_stated_id_is_returned_as_given(self):
        self.assertEqual("rh_abc", rule.require_session("  rh_abc  "))

    def test_an_uncaptured_callers_rule_is_not_filed_against_another_session(self):
        """A is recorded, B's events were refused, and B runs the command. There
        is one live session and it is not B's."""
        self.open_turn("s-alpha")
        self.fire("UserPromptSubmit", "", prompt_id="q1", prompt="session B work")
        code, _, err = self.rule_from_agent("a rule", host_id="s-bravo")
        self.assertEqual((1, True), (code, "not been captured" in err))
        self.assertEqual([], rule.load_all(self.checkout))


class TheHostNamesTheSessionAndCaptureTheTurn(Fixture):
    """Claude Code sets CLAUDE_CODE_SESSION_ID for every command the agent runs,
    matching the hooks' session_id and updated on /clear.

    Regression (B1): an MCP server keeps the id it was spawned with, so a rule
    was filed against the previous session after /clear, resume or a fork."""

    def test_the_session_comes_from_the_host_and_the_turn_from_the_record(self):
        alpha = self.open_turn("s-alpha")
        self.a_session("s-beta", "beta")
        code, out, _ = self.rule_from_agent("we always use the logger", host_id="s-alpha")
        self.assertEqual(0, code)
        [candidate] = rule.load_all(self.checkout)
        self.assertEqual((alpha, 2), (candidate["session_id"], candidate["corrective_turn"]))

    def test_without_a_host_session_nothing_is_recorded(self):
        self.open_turn("s-alpha")
        code, _, err = self.rule_from_agent("a rule")
        self.assertEqual((2, True), (code, "--session" in err))
        self.assertEqual([], rule.load_all(self.checkout))

    def test_a_turn_that_has_closed_is_not_guessed(self):
        self.a_session("s-alpha", "alpha")
        code, _, err = self.rule_from_agent("a rule", host_id="s-alpha")
        self.assertEqual((1, True), (code, "no turn in progress" in err))
        self.assertEqual([], rule.load_all(self.checkout))

    def test_after_clear_the_new_session_is_the_callers(self):
        self.open_turn("s-alpha")
        self.fire("SessionEnd", "s-alpha", reason="clear")
        self.fire("SessionStart", "s-gamma", source="clear")
        self.fire("UserPromptSubmit", "s-gamma", prompt_id="g1", prompt="we never print")
        self.assertEqual(1, self.rule_from_agent("a rule", host_id="s-alpha")[0],
                         "the ended session took a rule")
        self.assertEqual(0, self.rule_from_agent("we never print", host_id="s-gamma")[0])
        [candidate] = rule.load_all(self.checkout)
        self.assertEqual((identity.session_id("claude-code", "s-gamma"), 1),
                         (candidate["session_id"], candidate["corrective_turn"]))

    def test_a_named_turn_must_be_one_the_session_has_reached(self):
        self.open_turn("s-alpha")
        self.assertEqual(2, self.rule_from_agent("a rule", host_id="s-alpha", turn=3)[0])
        self.assertEqual(0, self.rule_from_agent("a rule", host_id="s-alpha", turn=1)[0])
        self.assertEqual([1], [c["corrective_turn"] for c in rule.load_all(self.checkout)])

    def test_the_output_carries_no_other_sessions_content(self):
        """1.10: what the agent reads back is egress into its context."""
        beta = self.a_session("s-beta", "beta")
        rule.mark(self.repo, self.checkout, beta, "beta's private convention")
        self.open_turn("s-alpha")
        _, out, err = self.rule_from_agent("alpha's rule", host_id="s-alpha")
        self.assertNotIn("beta", out + err)

    def test_a_resumed_session_takes_rules_again(self):
        """Regression: `ended_at` survives a resume, and a check on it refused
        every rule in a resumed session."""
        self.open_turn("s-alpha")
        self.fire("SessionEnd", "s-alpha", reason="prompt_input_exit")
        self.fire("SessionStart", "s-alpha", source="resume")
        self.fire("UserPromptSubmit", "s-alpha", prompt_id="p3", prompt="we never print")
        self.assertEqual(0, self.rule_from_agent("we never print", host_id="s-alpha")[0])
        self.assertEqual([3], [c["corrective_turn"] for c in rule.load_all(self.checkout)])

    def test_two_open_turns_are_not_guessed_between(self):
        both = {"turns": [{"index": 1, "completion": "pending"},
                          {"index": 2, "completion": "pending"}]}
        earlier = {"turns": [{"index": 1, "completion": "pending"},
                             {"index": 2, "completion": "stop"}]}
        self.assertEqual((None, None), (rule.open_turn(both), rule.open_turn(earlier)))


class TheSkill(Fixture):
    """The skill is what makes the tool reachable inside a session."""

    def skill_text(self):
        return (CORE / "repohone" / "skills" / "repohone-rule" / "SKILL.md").read_text()

    def test_the_skill_ships_with_core(self):
        self.assertIn("name: repohone-rule", self.skill_text())

    def test_it_names_the_command_it_teaches(self):
        self.assertIn('{repohone} rule "', self.skill_text())

    def test_it_is_chosen_even_when_the_agent_also_remembers_the_rule(self):
        """The live agent saved a stated rule to Claude Code's memory instead."""
        description = next(line for line in self.skill_text().splitlines()
                           if line.startswith("description:"))
        self.assertIn("memory", description)
        self.assertLessEqual(len(description), 1536, "Claude Code truncates it there")

    def test_it_says_marking_is_not_consent_to_change_anything(self):
        text = self.skill_text()
        self.assertIn("not** consent to change the project", text)
        self.assertIn("When not to run it", text)

    def test_it_tells_the_agent_not_to_choose_the_session_or_turn(self):
        text = self.skill_text()
        self.assertIn("so you pass neither", text)
        self.assertIn("Do not retry", text)

    def test_it_names_only_commands_and_options_the_cli_has(self):
        """The skill once sent agents to a `list_sessions` tool removed in 1.10."""
        commands = cli.build_parser()._subparsers._group_actions[0].choices
        named = re.findall(r"\{repohone\} ([a-z]+)((?: --[a-z-]+)*)", self.skill_text())
        self.assertTrue(named)
        for command, options in named:
            self.assertIn(command, commands)
            known = {o for a in commands[command]._actions for o in a.option_strings}
            self.assertEqual(set(), set(options.split()) - known)

    def test_install_places_the_skill_in_the_plugin_not_the_repository(self):
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            cli.cmd_install(argparse.Namespace(path=None))
        self.assertTrue((claude.plugin_dir() / "skills" / claude.SKILL / "SKILL.md").is_file())
        self.assertFalse((self.repo / ".mcp.json").exists())
        self.assertFalse((self.repo / ".claude").exists())

    def test_deinit_removes_only_what_an_older_install_left(self):
        (self.repo / ".mcp.json").write_text(json.dumps({"mcpServers": {
            "other": {"command": "node", "args": ["x.js"]},
            "repohone": {"command": sys.executable, "args": ["-m", "repohone.mcp"]}}}))
        copy = self.repo / ".claude" / "skills" / "repohone-rule"
        copy.mkdir(parents=True)
        (copy / "SKILL.md").write_text("old")
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            cli.cmd_deinit(argparse.Namespace(
                path=str(self.repo), purge=False, yes=True, dry_run=False))
        servers = json.loads((self.repo / ".mcp.json").read_text())["mcpServers"]
        self.assertEqual({"other"}, set(servers))
        self.assertFalse(copy.exists())


class RuleCommands(Fixture):

    def args(self, **kw):
        base = dict(path=str(self.repo), session=None, statement=None, turn=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_the_rule_command_exists_and_marks_a_candidate(self):
        session = self.a_session("s-alpha", "alpha")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.cmd_rule(self.args(session=session,
                                          statement="always the logger", turn=2))
        self.assertEqual(0, code)
        self.assertIn("marked against session", out.getvalue())
        self.assertEqual(1, len(rule.load_all(self.checkout)))

    def test_the_rule_command_reports_an_unknown_session(self):
        self.a_session("s-alpha", "alpha")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = cli.cmd_rule(self.args(session="rh_0000000000000000",
                                          statement="x"))
        self.assertEqual(1, code)
        self.assertIn("no session", err.getvalue())

    def test_the_parser_accepts_rule_and_rules(self):
        parser = cli.build_parser()
        marked = parser.parse_args(["rule", "a statement", "--session", "rh_x", "--turn", "3"])
        self.assertEqual(("a statement", "rh_x", 3),
                         (marked.statement, marked.session, marked.turn))
        self.assertIsNone(parser.parse_args(["rule", "a statement"]).session)
        self.assertIs(cli.cmd_rules, parser.parse_args(["rules"]).func)

    def test_diagnose_accepts_a_candidate_instead_of_a_session(self):
        parser = cli.build_parser()
        args = parser.parse_args(["diagnose", "--candidate", "rule_abc"])
        self.assertIsNone(args.session_id)
        self.assertEqual("rule_abc", args.candidate)

    def test_listing_candidates_shows_what_was_marked(self):
        session = self.a_session("s-alpha", "alpha")
        rule.mark(self.repo, self.checkout, session, "always the logger")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(0, cli.cmd_rules(self.args()))
        self.assertIn("always the logger", out.getvalue())
        self.assertIn(session, out.getvalue())


if __name__ == "__main__":
    unittest.main()
