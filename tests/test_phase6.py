"""Phase 6 tests: historical bootstrap.

Candidates only. A convention is never adopted because it is frequent —
frequency is evidence, not consent. Nothing here runs a model.

Run:  python3 -m unittest test_phase6
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from repohone import bootstrap, diagnosis, evidence, identity, profile

HOOK = CORE / "repohone_hook.py"


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                          text=True, check=True).stdout.strip()


class Repo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-p6-"))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        os.environ["REPOHONE_DATA_DIR"] = str(self.tmp / "data")
        git(["init", "-q"], self.repo)
        git(["config", "user.email", "d@e.com"], self.repo)
        git(["config", "user.name", "D"], self.repo)
        self.write("a.py", "x = 1\n")
        self.commit("init")

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, relative, text):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    _seq = 0

    def commit(self, subject, body="x"):
        Repo._seq += 1
        self.write(f"f{Repo._seq}.txt", f"{body}{Repo._seq}")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", subject], self.repo)

    def collect(self, inventory=None):
        return bootstrap.collect(self.repo, inventory)

    def statements(self, source=None):
        candidates, _, _ = self.collect()
        return [c.statement for c in candidates
                if source is None or c.source == source]


class InstructionFiles(Repo):
    """Rules the project already wrote down."""

    def test_normative_lines_become_candidates(self):
        self.write("CONVENTIONS.md",
                   "# Conventions\n\n"
                   "- Never call print() in src/; use the logger.\n"
                   "- Handlers must not import persistence directly.\n")
        found = self.statements("instruction-file")
        self.assertEqual(2, len(found))
        self.assertTrue(any("Never call print()" in s for s in found))

    def test_descriptive_prose_is_not_a_rule(self):
        self.write("CLAUDE.md",
                   "# Project\n\nThis document describes how the project is "
                   "organised.\nThe build lives in the Makefile.\n")
        self.assertEqual([], self.statements("instruction-file"))

    def test_headings_and_code_fences_are_skipped(self):
        self.write("AGENTS.md",
                   "# You must always read this\n```\nyou must not run this\n```\n"
                   "- Always add a test for a new public function.\n")
        self.assertEqual(["Always add a test for a new public function."],
                         self.statements("instruction-file"))

    def test_secrets_are_redacted_before_storage(self):
        self.write("CLAUDE.md",
                   "- You must always deploy with AKIAIOSFODNN7EXAMPLE as the key.\n")
        found = self.statements("instruction-file")
        self.assertTrue(found)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", found[0])

    def test_absolute_paths_are_scrubbed(self):
        self.write("CLAUDE.md", f"- You must never edit {self.repo}/src by hand.\n")
        found = self.statements("instruction-file")
        self.assertTrue(found)
        self.assertNotIn(str(self.repo), found[0])


class History(Repo):
    """What the project has already corrected, in its own words."""

    def test_reverts_are_candidates(self):
        self.commit('Revert "add experimental cache layer"')
        found = self.statements("revert")
        self.assertEqual(1, len(found))
        self.assertIn("add experimental cache layer", found[0])

    def test_fixups_are_candidates(self):
        self.commit("fixup! add report emitter")
        self.commit("fix typo in worker docstring")
        self.assertEqual(2, len(self.statements("fixup")))

    def test_ordinary_commits_are_not_candidates(self):
        self.commit("add the worker module")
        self.commit("implement the report emitter")
        self.assertEqual([], self.statements("revert"))
        self.assertEqual([], self.statements("fixup"))

    def test_repeated_statements_are_counted_not_duplicated(self):
        self.commit("fixup! add report emitter")
        self.commit("fixup! add report emitter")
        candidates, _, _ = self.collect()
        repeated = [c for c in candidates if c.statement == "fixup! add report emitter"]
        self.assertEqual(1, len(repeated))
        self.assertEqual(2, repeated[0].occurrences)

    def test_a_fixup_that_says_nothing_is_not_a_candidate(self):
        """Regression: "fix typo (x6)" and "oops (x3)" took the top two slots."""
        for subject in ("fix typo", "typo", "oops", "Oops, fix typo again"):
            self.commit(subject)
        self.assertEqual([], self.statements("fixup"))

    def test_a_repository_with_no_history_yields_nothing(self):
        self.assertEqual([], self.statements())


class CustomChecks(Repo):
    """A check the project wrote itself encodes a rule someone cared about."""

    def test_project_owned_checks_become_candidates(self):
        inventory = {"mechanisms": [
            {"id": "mech_a", "name": "scripts/check_imports.sh", "kind": "script",
             "tier": "project-owned-custom", "evidence": ["it is executable"],
             "enforced_in": []},
            {"id": "mech_b", "name": "ruff", "kind": "lint",
             "tier": "existing-project", "evidence": ["configured"],
             "enforced_in": ["ci"]}]}
        found = self.statements("custom-check")
        self.assertEqual([], found)
        candidates, _, _ = self.collect(inventory)
        checks = [c for c in candidates if c.source == "custom-check"]
        self.assertEqual(1, len(checks), "only project-owned checks are candidates")
        self.assertIn("nothing runs it", checks[0].statement)


class NothingIsAdopted(Repo):
    """The whole point: frequency is evidence, not consent."""

    def setUp(self):
        super().setUp()
        self.write("CONVENTIONS.md", "- Never call print(); use the logger.\n")
        self.commit("fix typo")

    def test_the_record_says_nothing_was_adopted(self):
        candidates, errors, unavailable = self.collect()
        record_ = bootstrap.build_record("co", None, candidates, "2026-01-01T00:00:00Z",
                                         errors, unavailable)
        self.assertFalse(record_["adopted"])
        self.assertIn("never adopted", record_["note"])

    def test_unavailable_sources_are_named_not_omitted(self):
        _, _, unavailable = self.collect()
        self.assertTrue(any("pull-request review comments" in u for u in unavailable),
                        unavailable)
        self.assertTrue(any("CI failure history" in u for u in unavailable),
                        unavailable)

    def test_no_fingerprint_or_proposal_is_created(self):
        from repohone import fingerprint, proposal
        self.collect()
        checkout = identity.checkout_id(self.repo)
        self.assertEqual([], fingerprint.known(checkout, None))
        self.assertEqual([], proposal.load_all(checkout))

    def test_a_failing_collector_is_reported(self):
        original = bootstrap._reverts
        bootstrap._reverts = lambda root: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            candidates, errors, _ = self.collect()
        finally:
            bootstrap._reverts = original
        self.assertTrue(errors)
        self.assertIn("boom", errors[0])
        self.assertTrue(candidates, "one broken collector must not empty the rest")


PR_COMMENTS = """[[
  {"path": "src/db.py", "body": "You must never open a raw connection here - use the pool."},
  {"path": "src/api.py", "body": "nit: spacing"},
  {"path": "src/api.py", "body": "Handlers must not import persistence directly. token=ghp_AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIII"}
]]"""  # noqa: E501

CI_RUNS = ('[{"workflowName":"lint","conclusion":"failure"},'
           '{"workflowName":"typecheck","conclusion":"failure"},'
           '{"workflowName":"lint","conclusion":"failure"}]')


class RemoteSources(Repo):
    """Phase 6's two named-unavailable sources. Opt-in: opening a repository is
    consent to read it, not consent to call GitHub on the developer's behalf."""

    def fake_gh(self, pr=PR_COMMENTS, runs=CI_RUNS, works=True):
        binaries = self.tmp / "bin"
        binaries.mkdir(exist_ok=True)
        script = binaries / "gh"
        body = "#!/bin/sh\n" + ("" if works else "exit 1\n")
        if works:
            body += (f'case "$*" in\n'
                     f'  *pulls/comments*) cat <<\'J\'\n{pr}\nJ\n;;\n'
                     f'  *"run list"*) cat <<\'J\'\n{runs}\nJ\n;;\n'
                     f'  *) exit 1 ;;\nesac\n')
        script.write_text(body)
        script.chmod(0o755)
        self._path = os.environ["PATH"]
        os.environ["PATH"] = f"{binaries}:{self._path}"
        self.addCleanup(lambda: os.environ.__setitem__("PATH", self._path))

    def test_remote_sources_are_not_read_unless_asked(self):
        self.fake_gh()
        candidates, _, unavailable = bootstrap.collect(self.repo)
        self.assertEqual([], [c for c in candidates
                              if c.source in ("pr-review", "ci-failure")])
        self.assertTrue(any("not requested" in u for u in unavailable))

    def test_review_comments_become_candidates_when_asked(self):
        self.fake_gh()
        candidates, _, _ = bootstrap.collect(self.repo, include_remote=True)
        statements = [c.statement for c in candidates if c.source == "pr-review"]
        self.assertEqual(2, len(statements))
        self.assertTrue(any("raw connection" in s for s in statements))

    def test_chatter_is_not_a_rule(self):
        self.fake_gh()
        candidates, _, _ = bootstrap.collect(self.repo, include_remote=True)
        self.assertNotIn("nit: spacing",
                         [c.statement for c in candidates])

    def test_a_secret_in_a_review_comment_is_redacted(self):
        self.fake_gh()
        candidates, _, _ = bootstrap.collect(self.repo, include_remote=True)
        blob = " ".join(c.statement for c in candidates)
        self.assertNotIn("ghp_AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHHIIII", blob)
        self.assertIn("[REDACTED]", blob)

    def test_a_repeatedly_failing_check_is_counted_not_duplicated(self):
        self.fake_gh()
        candidates, _, _ = bootstrap.collect(self.repo, include_remote=True)
        lint = [c for c in candidates
                if c.source == "ci-failure" and "lint" in c.statement]
        self.assertEqual(1, len(lint))
        self.assertEqual(2, lint[0].occurrences)

    def test_an_unavailable_gh_is_a_stated_limit_not_an_empty_result(self):
        self.fake_gh(works=False)
        candidates, errors, unavailable = bootstrap.collect(self.repo,
                                                            include_remote=True)
        self.assertEqual([], [c for c in candidates
                              if c.source in ("pr-review", "ci-failure")])
        self.assertEqual(2, len([u for u in unavailable if "gh unavailable" in u]))
        self.assertEqual([], errors, "an absent source is not a collection error")

    def test_malformed_gh_output_does_not_raise(self):
        self.fake_gh(pr="not json at all", runs="{}")
        candidates, errors, unavailable = bootstrap.collect(self.repo,
                                                            include_remote=True)
        self.assertEqual([], [c for c in candidates
                              if c.source in ("pr-review", "ci-failure")])
        self.assertTrue(unavailable)


class ABoundedReviewSet(Repo):
    """Regression: bootstrap returned up to 25 per source, which is a backlog to
    triage rather than a set a maintainer will actually review."""

    def many_rules(self, count=30):
        lines = [f"- Rule {i}: you must never use approach-{i} here."
                 for i in range(count)]
        self.write("CLAUDE.md", "\n".join(lines) + "\n")
        self.commit("rules")

    def record(self):
        candidates, errors, unavailable = self.collect()
        return bootstrap.build_record("c", None, candidates, "2026-01-01T00:00:00Z",
                                      errors, unavailable)

    def test_a_large_history_yields_a_short_review_set(self):
        self.many_rules(30)
        for i in range(12):
            self.commit(f'Revert "change {i}"')
        rec = self.record()
        self.assertGreater(len(rec["candidates"]), 20, "fixture was too small")
        self.assertLessEqual(len(rec["review_set"]), 5)
        self.assertGreaterEqual(len(rec["review_set"]), 3)

    def test_the_full_set_is_still_recorded(self):
        self.many_rules(30)
        rec = self.record()
        self.assertGreater(len(rec["candidates"]), len(rec["review_set"]),
                           "the review set replaced the record instead of ranking it")

    def test_the_review_set_names_real_candidates(self):
        self.many_rules(10)
        rec = self.record()
        known = {c["id"] for c in rec["candidates"]}
        self.assertTrue(set(rec["review_set"]) <= known)

    def test_a_repeated_correction_outranks_a_written_rule(self):
        self.write("CLAUDE.md", "- You must always run the tests.\n")
        self.commit("rules")
        for _ in range(3):
            self.commit('Revert "drop the null check"')
        rec = self.record()
        first = next(c for c in rec["candidates"] if c["id"] == rec["review_set"][0])
        self.assertEqual("revert", first["source"])
        self.assertEqual(3, first["occurrences"])

    def test_source_decides_when_two_candidates_recur_equally(self):
        """Occurrences dominate, so the source ranking only shows up in a tie."""
        self.write("CLAUDE.md", "- You must always run the tests.\n")
        self.commit("rules")
        self.commit('Revert "drop the null check"')
        rec = self.record()
        ranked = [c for c in rec["candidates"] if c["id"] in rec["review_set"]]
        tied = [c["source"] for c in ranked if c["occurrences"] == 1]
        self.assertEqual(1, tied.count("revert"))
        self.assertLess(tied.index("revert"), tied.index("instruction-file"),
                        "a rule already written down was ranked above a correction")

    def test_no_source_fills_the_set_by_repetition(self):
        self.write("CLAUDE.md", "- You must always run the tests.\n"
                                "- Never log customer email addresses.\n")
        self.commit("rules")
        for i in range(4):
            self.commit(f'Revert "change {i}"')
        self.commit("fixup! add report emitter")
        rec = self.record()
        sources = [next(c["source"] for c in rec["candidates"] if c["id"] == cid)
                   for cid in rec["review_set"]]
        self.assertEqual(["revert", "fixup", "instruction-file", "revert",
                          "instruction-file"], sources)

    def test_the_ranking_basis_is_stated(self):
        self.many_rules(6)
        self.assertIn("repeated", self.record()["ranking_basis"])


class Cli(Repo):
    def _cli(self, *args):
        return subprocess.run([sys.executable, str(HOOK), *args],
                              cwd=str(self.repo), capture_output=True, text=True,
                              env=dict(os.environ))

    def test_an_uninitialized_repository_is_not_mined(self):
        proc = self._cli("bootstrap", "--path", str(self.repo))
        self.assertEqual(2, proc.returncode)

    def test_the_output_states_that_nothing_is_adopted(self):
        self.write(".repohone/profile.yaml", profile.template())
        self.write("CONVENTIONS.md", "- Never call print(); use the logger.\n")
        proc = self._cli("bootstrap", "--path", str(self.repo))
        self.assertEqual(0, proc.returncode)
        self.assertIn("Candidates only", proc.stdout)
        self.assertIn("nothing here is adopted", proc.stdout)

    def test_a_candidate_a_session_showed_again_says_so(self):
        self.commit("handlers query db directly")
        git(["revert", "--no-edit", "HEAD"], self.repo)
        self.write(".repohone/profile.yaml", profile.template())
        self.assertEqual(0, self._cli("bootstrap", "--path", str(self.repo)).returncode)
        checkout = identity.checkout_id(self.repo)
        found = next(c for c in bootstrap.load(checkout)["candidates"]
                     if c["source"] == "revert")
        body = {"root_cause": {"class": "MISSING_CONTEXT", "summary": "s"},
                "confidence": "medium", "required_property": "p", "corrective_turn": 2,
                "fingerprint": {"area": "a", "behavior": "b"}, "evidence_for": ["e"],
                "evidence_against": [], "risks": []}
        linked = diagnosis.build_record(
            "rh_00000000000000aa", checkout, "SUCCESS", evidence.Selection(), "fake",
            None, True, body=body,
            mark={"fingerprint_id": "fp_0000000000000000", "area": "a", "behavior": "b",
                  "created": True},
            history=[{"candidate_id": found["id"], "source": "revert",
                      "commit": found["evidence"][0], "statement": found["statement"]}])
        diagnosis.save(checkout, linked)
        proc = self._cli("bootstrap", "--path", str(self.repo))
        self.assertIn(f"seen again in a session: {linked['diagnosis_id']}", proc.stdout)


if __name__ == "__main__":
    unittest.main()
