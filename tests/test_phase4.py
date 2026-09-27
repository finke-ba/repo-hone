"""Phase 4 tests: native mechanism discovery.

Discovery answers *what does this repository already provide?* — never *what
should we add*. Every test names the section it enforces.

Run:  python3 -m unittest test_phase4
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from repohone import identity, mechanisms, probe, profile

HOOK = CORE / "repohone_hook.py"
SCHEMA_PATH = CORE / "repohone" / "schemas" / "mechanisms.v1.schema.json"

try:
    import jsonschema
    VALIDATOR = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
except ImportError:
    VALIDATOR = None


def git(args, cwd):
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                          text=True, check=True).stdout.strip()


class Repo(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-mech-"))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.data = self.tmp / "data"
        os.environ["REPOHONE_DATA_DIR"] = str(self.data)
        git(["init", "-q"], self.repo)
        git(["config", "user.email", "d@e.com"], self.repo)
        git(["config", "user.name", "D"], self.repo)
        self.write("a.py", "x = 1\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "init"], self.repo)

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write(self, relative, text):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def found(self):
        found, _ = mechanisms.discover(self.repo)
        return found

    def errors(self):
        _, errors = mechanisms.discover(self.repo)
        return errors

    def kinds(self):
        return {m.kind for m in self.found()}

    def named(self, name):
        return [m for m in self.found() if m.name == name]


class ADirectoryNameIsNotARunner(Repo):
    """Regression: any `tests/` directory produced "python tests" as an
    existing-project mechanism — in a Jest repository too — and a project
    configured by pytest.ini was reported as running unittest."""

    def test_a_node_project_with_tests_gets_no_python_runner(self):
        self.write("tests/app.test.js", "test('x', () => {});\n")
        self.write("package.json", json.dumps({"scripts": {"test": "jest"}}))
        self.assertEqual(["npm run test"], [m.name for m in self.found() if m.kind == "test"])

    def test_pytest_style_tests_without_configuration_are_not_guessed(self):
        self.write("tests/test_app.py", "def test_x():\n    assert True\n")
        self.assertEqual([], [m for m in self.found() if m.kind == "test"])

    def test_pytest_is_found_through_each_of_its_own_files(self):
        for name, text in (("pytest.ini", "[pytest]\naddopts = -q\n"),
                           ("tox.ini", "[tox]\nenvlist = py3\n\n[pytest]\naddopts = -q\n"),
                           ("conftest.py", "import pytest\n"),
                           ("tests/conftest.py", "import pytest\n")):
            with self.subTest(name=name):
                target = self.write(name, text)
                found = self.named("pytest")
                self.assertEqual(1, len(found))
                self.assertEqual("existing-project", found[0].tier)
                self.assertIn(name, found[0].evidence[0])
                target.unlink()

    def test_a_tox_file_without_a_pytest_section_is_not_pytest(self):
        self.write("tox.ini", "[tox]\nenvlist = py3\n")
        self.assertEqual([], self.named("pytest"))

    def test_unittest_is_found_through_what_the_tests_import(self):
        self.write("tests/test_app.py", "import unittest\n\nclass T(unittest.TestCase):\n"
                                        "    def test_x(self):\n        pass\n")
        found = self.named("python unittest")
        self.assertEqual(1, len(found))
        self.assertIn("imports unittest", found[0].evidence[0])


class Detection(Repo):
    """Discovery is observation: each mechanism carries the evidence it exists."""

    def test_nothing_is_discovered_in_an_empty_repository(self):
        self.assertEqual([], self.found())

    def test_python_tooling_from_pyproject(self):
        self.write("pyproject.toml",
                   "[tool.pytest.ini_options]\n[tool.ruff]\n[tool.mypy]\n"
                   "[tool.importlinter]\nroot_package='src'\n")
        found = {m.kind: m for m in self.found()}
        self.assertIn(mechanisms.TEST, found)
        self.assertIn(mechanisms.LINT, found)
        self.assertIn(mechanisms.TYPES, found)
        self.assertIn(mechanisms.ARCHITECTURE, found)
        self.assertEqual(["lint-imports"], found[mechanisms.ARCHITECTURE].argv)

    def test_node_scripts_and_dev_dependencies(self):
        self.write("package.json", json.dumps({
            "scripts": {"test": "jest", "lint": "eslint ."},
            "devDependencies": {"typescript": "^5", "prettier": "^3"}}))
        names = {m.name for m in self.found()}
        self.assertIn("npm run test", names)
        self.assertIn("npm run lint", names)
        self.assertIn("typescript", names)
        self.assertIn("prettier", names)

    def test_makefile_targets(self):
        self.write("Makefile", "test:\n\tpytest\nlint:\n\truff check .\nverify: lint test\n")
        names = {m.name for m in self.found()}
        self.assertIn("make test", names)
        self.assertIn("make lint", names)
        self.assertIn("make verify", names)

    def test_rust_and_go(self):
        self.write("Cargo.toml", "[package]\nname='x'\n")
        self.write("go.mod", "module x\n")
        names = {m.name for m in self.found()}
        self.assertIn("cargo test", names)
        self.assertIn("go test", names)

    def test_ci_configuration(self):
        self.write(".github/workflows/ci.yml", "name: ci\n")
        self.assertIn(mechanisms.CI, self.kinds())

    def test_vcs_hooks(self):
        self.write(".pre-commit-config.yaml", "repos: []\n")
        self.assertIn(mechanisms.VCS_HOOK, self.kinds())

    def test_agent_configuration_is_a_project_mechanism(self):
        """ARCH §3 prefers a native agent instruction over RepoHone runtime."""
        self.write("CLAUDE.md", "# Conventions\n")
        self.write(".claude/settings.json", json.dumps({"hooks": {"Stop": []}}))
        self.write(".claude/skills/reviewer/skill.json", "{}")
        kinds = self.kinds()
        self.assertIn(mechanisms.AGENT_INSTRUCTION, kinds)
        self.assertIn(mechanisms.AGENT_HOOK, kinds)
        self.assertIn(mechanisms.AGENT_SKILL, kinds)

    def test_project_owned_check_scripts(self):
        self.write("scripts/check_arch.sh", "#!/bin/sh\n")
        found = self.named("scripts/check_arch.sh")
        self.assertTrue(found)
        self.assertEqual("project-owned-custom", found[0].tier)

    def test_every_mechanism_states_its_evidence(self):
        self.write("Makefile", "test:\n\tpytest\n")
        self.write("CLAUDE.md", "x\n")
        self.write(".github/workflows/ci.yml", "name: ci\n")
        for mechanism in self.found():
            self.assertTrue(mechanism.evidence, f"{mechanism.name} claims no evidence")

    def test_malformed_configuration_does_not_break_discovery(self):
        self.write("package.json", "{not json")
        self.write("Makefile", "test:\n\tpytest\n")
        self.assertIn("make test", {m.name for m in self.found()})


class NeverAShellString(Repo):
    """Regression: a filename is repository-controlled content, and it was being
    concatenated into a `shell=True` command line."""

    def test_a_hostile_filename_is_not_discovered(self):
        path = self.write("scripts/check;touch PWNED_MARKER.sh", "#!/bin/sh\n")
        path.chmod(0o755)
        self.write("scripts/check_arch.sh", "#!/bin/sh\n").chmod(0o755)
        names = [m.name for m in self.found()]
        self.assertEqual(["scripts/check_arch.sh"], names)
        self.assertFalse(any(";" in n for n in names))

    def test_every_invocation_is_an_argv_list(self):
        self.write("Makefile", "test:\n\tpytest\n")
        self.write("pyproject.toml", "[tool.ruff]\n")
        self.write("scripts/check_arch.sh", "#!/bin/sh\n").chmod(0o755)
        for mechanism in self.found():
            if mechanism.argv is not None:
                self.assertIsInstance(mechanism.argv, list)
                self.assertTrue(all(isinstance(part, str) for part in mechanism.argv))

    def test_the_probe_refuses_a_shell_string(self):
        with self.assertRaises(TypeError):
            probe.run(self.repo, "co", "HEAD", "./scripts/check; rm -rf x")

    def test_the_probe_refuses_an_empty_argv(self):
        with self.assertRaises(TypeError):
            probe.run(self.repo, "co", "HEAD", [])


class NoPhantomMechanisms(Repo):
    """Regression: substring matching reported tools the project does not use —
    including from a comment saying the tool had been removed."""

    def test_a_comment_is_not_evidence(self):
        self.write("pyproject.toml",
                   '[project]\nname = "x"\n'
                   '# we removed black last year, do not reintroduce it\n')
        self.assertEqual([], [m for m in self.found() if m.name == "black"])

    def test_a_similarly_named_dependency_is_not_the_tool(self):
        self.write("pyproject.toml",
                   '[project]\nname = "x"\ndependencies = ["pytest-benchmark"]\n')
        self.assertEqual([], [m for m in self.found() if m.name == "pytest"])

    def test_configured_outranks_merely_declared(self):
        self.write("pyproject.toml",
                   '[tool.ruff]\nline-length = 100\n\n'
                   '[project]\nname = "x"\ndependencies = ["pytest"]\n')
        by_name = {m.name: m for m in self.found()}
        self.assertEqual("existing-project", by_name["ruff"].tier)
        self.assertEqual("existing-ecosystem", by_name["pytest"].tier)
        self.assertIn("configures [tool.ruff]", by_name["ruff"].evidence[0])

    def test_a_data_file_is_not_a_project_check(self):
        self.write("scripts/test_data.json", "{}")
        self.write("scripts/validate_notes.md", "# notes")
        self.assertEqual([], self.found())

    def test_an_executable_check_still_counts(self):
        self.write("scripts/check_real.sh", "#!/bin/sh\n").chmod(0o755)
        self.assertEqual(["scripts/check_real.sh"], [m.name for m in self.found()])

    def test_configuration_is_captured_so_a_proposal_can_fit_it(self):
        self.write("pyproject.toml", "[tool.ruff]\nline-length = 100\nselect = ['E']\n")
        ruff = [m for m in self.found() if m.name == "ruff"][0]
        self.assertIn("line-length", ruff.config_excerpt)


class DetectorFailuresAreReported(Repo):
    """Regression: a detector bug was indistinguishable from 'no such mechanism'."""

    def test_a_failing_detector_is_recorded(self):
        def broken(root):
            raise RuntimeError("detector bug")

        original = mechanisms.DETECTORS
        mechanisms.DETECTORS = (broken,)
        try:
            found, errors = mechanisms.discover(self.repo)
        finally:
            mechanisms.DETECTORS = original
        self.assertEqual([], found)
        self.assertTrue(errors)
        self.assertIn("detector bug", errors[0])

    def test_errors_travel_with_the_inventory(self):
        record = mechanisms.build_record("co", None, [], "2026-01-01T00:00:00Z",
                                         errors=["_make: RuntimeError: boom"])
        self.assertEqual(["_make: RuntimeError: boom"], record["discovery_errors"])


class EnforcementLinkage(Repo):
    """LEARNING_PLAN §11 pass 2 asks why an existing mechanism did not already
    help. A linter nobody runs must not look like one CI gates on."""

    def setUp(self):
        super().setUp()
        self.write("pyproject.toml",
                   "[tool.ruff]\n[tool.mypy]\n[tool.pytest.ini_options]\n")
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n"
                   "      - run: pytest -q\n      - run: ruff check .\n")

    def test_tools_wired_into_ci_are_marked(self):
        by_name = {m.name: m for m in self.found()}
        self.assertIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_a_commented_command_is_not_ci_enforcement(self):
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n"
                   "      # run: ruff check .\n      - run: pytest -q\n")
        by_name = {m.name: m for m in self.found()}
        self.assertNotIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_block_commands_are_still_recognised(self):
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n"
                   "      - run: |\n          pytest -q\n          ruff check .\n")
        by_name = {m.name: m for m in self.found()}
        self.assertIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_run_keys_in_environment_or_action_inputs_are_not_steps(self):
        self.write(".github/workflows/ci.yml",
                   "name: ci\nenv:\n  run: ruff check .\njobs:\n  t:\n"
                   "    steps:\n      - uses: example/action@v1\n"
                   "        with:\n          run: ruff check .\n"
                   "      - run: pytest -q\n")
        by_name = {m.name: m for m in self.found()}
        self.assertNotIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_printing_a_tool_name_is_not_enforcement(self):
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n"
                   "      - run: echo 'ruff check .'\n"
                   "      - run: pytest -q\n")
        by_name = {m.name: m for m in self.found()}
        self.assertNotIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_folded_yaml_preserves_one_shell_command(self):
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n"
                   "      - run: >\n          echo\n          ruff check .\n"
                   "      - run: pytest -q\n")
        by_name = {m.name: m for m in self.found()}
        self.assertNotIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_heredoc_contents_are_not_commands(self):
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n"
                   "      - run: |\n          cat <<'EOF'\n"
                   "          ruff check .\n          EOF\n"
                   "      - run: pytest -q\n")
        by_name = {m.name: m for m in self.found()}
        self.assertNotIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_continued_echo_arguments_are_not_commands(self):
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n"
                   "      - run: |\n          echo \\\n"
                   "            ruff check .\n"
                   "      - run: pytest -q\n")
        by_name = {m.name: m for m in self.found()}
        self.assertNotIn("ci", by_name["ruff"].enforced_in)
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_a_configured_but_unenforced_tool_is_distinguishable(self):
        by_name = {m.name: m for m in self.found()}
        self.assertEqual([], by_name["mypy"].enforced_in,
                         "mypy is configured but never run; the record must say so")

    def test_pre_commit_enforcement_is_separate_from_ci(self):
        self.write(".pre-commit-config.yaml",
                   "repos:\n  - hooks:\n      - id: mypy\n")
        by_name = {m.name: m for m in self.found()}
        self.assertIn("vcs-hook", by_name["mypy"].enforced_in)
        self.assertNotIn("ci", by_name["mypy"].enforced_in)

    def test_a_commented_pre_commit_hook_is_not_enforcement(self):
        self.write(".pre-commit-config.yaml", "# - id: mypy\n")
        by_name = {m.name: m for m in self.found()}
        self.assertNotIn("vcs-hook", by_name["mypy"].enforced_in)

    def test_enforcement_follows_one_level_of_indirection(self):
        """CI commonly runs `make verify`, which runs the tools. A gated linter
        must not read as unenforced."""
        self.write("Makefile", "test:\n\tpytest -q\nlint:\n\truff check .\nverify: lint test\n")
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n      - run: make verify\n")
        by_name = {m.name: m for m in self.found()}
        self.assertIn("ci", by_name["make verify"].enforced_in)
        self.assertIn("ci", by_name["ruff"].enforced_in, "ruff is gated via make verify")
        self.assertIn("ci", by_name["pytest"].enforced_in)

    def test_a_tool_no_target_runs_stays_unenforced(self):
        self.write("Makefile", "lint:\n\truff check .\nverify: lint\n")
        self.write(".github/workflows/ci.yml",
                   "name: ci\njobs:\n  t:\n    steps:\n      - run: make verify\n")
        by_name = {m.name: m for m in self.found()}
        self.assertEqual([], by_name["mypy"].enforced_in,
                         "mypy is configured but nothing runs it")

    def test_enforcement_travels_in_the_record(self):
        record = mechanisms.build_record("co", None, self.found(),
                                         "2026-01-01T00:00:00Z")
        by_name = {m["name"]: m for m in record["mechanisms"]}
        self.assertEqual(["ci"], by_name["ruff"]["enforced_in"])
        self.assertEqual([], by_name["mypy"]["enforced_in"])


class ProbeSelection(Repo):
    """Probing executes the project's own code, so running everything is not a
    default."""

    def _cli(self, *args):
        return subprocess.run([sys.executable, str(HOOK), "mechanisms",
                               "--path", str(self.repo), *args],
                              cwd=str(self.repo), capture_output=True, text=True,
                              env=dict(os.environ))

    def setUp(self):
        super().setUp()
        self.write(".repohone/profile.yaml", profile.template())
        self.write("Makefile", "test:\n\ttrue\nlint:\n\ttrue\n")

    def test_probing_everything_needs_confirmation(self):
        proc = self._cli("--probe")
        self.assertEqual(3, proc.returncode)
        self.assertIn("would execute", proc.stderr)
        self.assertIn("--yes", proc.stderr)

    def test_a_single_mechanism_can_be_probed(self):
        proc = self._cli("--probe", "--only", "make test")
        self.assertEqual(0, proc.returncode)
        record = mechanisms.load(identity.peek_checkout_id(self.repo))
        probed = [m for m in record["mechanisms"] if m["cost"]["probed"]]
        self.assertEqual(["make test"], [m["name"] for m in probed])

    def test_an_unmatched_selection_is_an_error(self):
        proc = self._cli("--probe", "--only", "nonexistent")
        self.assertEqual(3, proc.returncode)
        self.assertIn("nothing matches", proc.stderr)


class Preference(Repo):
    """ARCH §3 — existing project, then ecosystem, then project-owned."""

    def _unsorted(self):
        """Deliberately worst-case order, so ranking cannot pass by accident."""
        return [
            mechanisms.Mechanism(mechanisms.SCRIPT, "scripts/verify.sh",
                                 "project-owned-custom", "./scripts/verify.sh", ["e"]),
            mechanisms.Mechanism(mechanisms.AGENT_INSTRUCTION, "CLAUDE.md",
                                 "existing-project", None, ["e"]),
            mechanisms.Mechanism(mechanisms.TEST, "cargo test",
                                 "existing-ecosystem", "cargo test", ["e"]),
            mechanisms.Mechanism(mechanisms.TEST, "make test",
                                 "existing-project", "make test", ["e"]),
        ]

    def test_tiers_are_ranked(self):
        tiers = [m.tier for m in mechanisms.ranked(self._unsorted())]
        self.assertEqual(["existing-project", "existing-project",
                          "existing-ecosystem", "project-owned-custom"], tiers)

    def test_deterministic_mechanisms_rank_above_probabilistic(self):
        order = [m.name for m in mechanisms.ranked(self._unsorted())]
        self.assertLess(order.index("make test"), order.index("CLAUDE.md"))

    def test_discovery_output_is_ranked_too(self):
        self.write("Makefile", "test:\n\tpytest\n")
        self.write("CLAUDE.md", "x\n")
        self.write("scripts/verify.sh", "#!/bin/sh\n")
        tiers = [m.tier for m in mechanisms.ranked(self.found())]
        self.assertEqual(sorted(tiers, key=lambda t: mechanisms.TIER_ORDER[t]), tiers)

    def test_repohone_is_never_a_discovered_mechanism(self):
        """ARCH §25 — RepoHone does not own normal enforcement."""
        self.write("Makefile", "test:\n\tpytest\n")
        self.write(".claude/settings.json", json.dumps({"hooks": {"Stop": []}}))
        record = mechanisms.build_record("co", None, self.found(), "2026-01-01T00:00:00Z")
        blob = json.dumps(record).lower()
        self.assertNotIn("repohone-runtime", blob)
        for mechanism in record["mechanisms"]:
            self.assertIn(mechanism["tier"], mechanisms.TIER_ORDER)

    def test_kinds_map_to_improvement_classes(self):
        """Phase 5 matches these against the required property."""
        self.write("pyproject.toml", "[tool.pytest.ini_options]\n[tool.importlinter]\n")
        by_kind = {m.kind: m.as_record() for m in self.found()}
        self.assertIn("deterministic-test", by_kind[mechanisms.TEST]["carries"])
        self.assertIn("architecture-constraint",
                      by_kind[mechanisms.ARCHITECTURE]["carries"])


class RecordShape(Repo):
    def test_the_inventory_conforms_to_its_schema(self):
        self.write("Makefile", "test:\n\tpytest\n")
        self.write("CLAUDE.md", "x\n")
        record = mechanisms.build_record("co-1", "repo_abc", self.found(),
                                         "2026-01-01T00:00:00Z")
        if VALIDATOR is not None:
            errors = sorted(VALIDATOR.iter_errors(record), key=lambda e: e.path)
            self.assertEqual([], ["/".join(map(str, e.path)) + ": " + e.message
                                  for e in errors])

    def test_versions_are_recorded(self):
        record = mechanisms.build_record("co-1", None, [], "2026-01-01T00:00:00Z")
        self.assertEqual("repohone.mechanisms/v1", record["schema"])
        self.assertEqual(1, record["schema_version"])

    def test_an_unprobed_inventory_says_so(self):
        self.write("Makefile", "test:\n\tpytest\n")
        record = mechanisms.build_record("co-1", None, self.found(),
                                         "2026-01-01T00:00:00Z")
        self.assertFalse(record["probe"]["ran"])
        self.assertTrue(all(not m["cost"]["probed"] for m in record["mechanisms"]))


class Probing(Repo):
    """ARCH §12.2 — project tooling never runs against the live working tree."""

    def setUp(self):
        super().setUp()
        self.checkout = identity.checkout_id(self.repo)

    def test_artifacts_never_reach_the_working_tree(self):
        self.write("Makefile", 'build:\n\techo art > BUILD_ARTIFACT.txt\n')
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "make"], self.repo)
        result = probe.run(self.repo, self.checkout, "HEAD", ["make", "build"], timeout=20)
        self.assertTrue(result.ran)
        self.assertEqual(0, result.exit_code)
        self.assertFalse((self.repo / "BUILD_ARTIFACT.txt").exists(),
                         "a probe wrote into the developer's working tree")

    def test_the_working_tree_is_verified_unchanged(self):
        self.write("Makefile", "noop:\n\ttrue\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "make"], self.repo)
        before = probe.tree_fingerprint(self.repo)
        probe.run(self.repo, self.checkout, "HEAD", ["make", "noop"], timeout=20)
        self.assertEqual(before, probe.tree_fingerprint(self.repo))

    def test_a_failing_mechanism_is_measured_not_hidden(self):
        self.write("Makefile", "fail:\n\texit 3\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "make"], self.repo)
        result = probe.run(self.repo, self.checkout, "HEAD", ["make", "fail"], timeout=20)
        self.assertTrue(result.ran)
        self.assertNotEqual(0, result.exit_code)
        self.assertIsNotNone(result.duration_ms)

    def test_a_hanging_mechanism_is_bounded(self):
        result = probe.run(self.repo, self.checkout, "HEAD", ["sleep", "30"], timeout=2)
        self.assertTrue(result.ran)
        self.assertIsNone(result.exit_code)
        self.assertIn("exceeded", result.note)

    def _bounded_child(self, parent_sleeps, independent=False):
        pid_file = self.tmp / (
            "independent.pid" if independent else
            "timeout.pid" if parent_sleeps else "normal.pid")
        child = "import time; time.sleep(30)"
        session = ", start_new_session=True" if independent else ""
        parent = (
            "import pathlib,subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable, '-c', sys.argv[1]], "
            "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
            f"stderr=subprocess.DEVNULL{session}); "
            "pathlib.Path(sys.argv[2]).write_text(str(child.pid)); "
            f"time.sleep({30 if parent_sleeps else 0})")
        _returncode, _stdout, _stderr, timed_out = probe._run_bounded(
            [sys.executable, "-c", parent, child, str(pid_file)],
            self.tmp, dict(os.environ), 0.5 if parent_sleeps else 10)
        self.assertTrue(pid_file.is_file(), "fixture did not start a child")
        pid = int(pid_file.read_text())
        time.sleep(0.2)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        else:
            alive = True
            os.kill(pid, 9)
        self.assertFalse(alive, "a validation descendant survived the bounded run")
        return timed_out

    def test_a_daemonised_child_is_killed_after_normal_exit(self):
        self.assertFalse(self._bounded_child(parent_sleeps=False))

    def test_a_daemonised_child_is_killed_after_timeout(self):
        self.assertTrue(self._bounded_child(parent_sleeps=True))

    def test_an_independently_managed_worker_is_killed(self):
        self.assertFalse(self._bounded_child(parent_sleeps=False, independent=True))

    def test_tool_state_is_redirected_into_scratch(self):
        result = probe.run(self.repo, self.checkout, "HEAD",
                           ["sh", "-c",
                            'test "$HOME" != "%s" && test -n "$REPOHONE_PROBE"'
                            % os.path.expanduser("~")], timeout=20)
        self.assertEqual(0, result.exit_code, "HOME was not redirected")

    def test_a_probe_run_from_a_terminal_does_not_inherit_it(self):
        """Regression: macOS's sandbox refused the inherited tty, so every
        probe failed when the developer ran RepoHone from a terminal."""
        import pty
        master, slave = pty.openpty()
        saved = os.dup(0)
        os.dup2(slave, 0)
        try:
            result = probe.run(self.repo, self.checkout, "HEAD",
                               ["sh", "-c", "test ! -t 0"], timeout=20)
        finally:
            os.dup2(saved, 0)
            for fd in (saved, slave, master):
                os.close(fd)
        self.assertEqual(0, result.exit_code, result.note)

    def test_the_probe_sees_the_snapshot_not_the_live_tree(self):
        self.write("tracked.py", "committed = True\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "tracked"], self.repo)
        (self.repo / "uncommitted.py").write_text("x = 1\n")
        result = probe.run(self.repo, self.checkout, "HEAD",
                           ["sh", "-c",
                            "test -f tracked.py && test ! -f uncommitted.py"], timeout=20)
        self.assertEqual(0, result.exit_code)


class OnlyWhenAsked(Repo):
    """ARCH §1.1 — discovery is part of an explicit analysis."""

    def _cli(self, *args):
        return subprocess.run([sys.executable, str(HOOK), *args],
                              cwd=str(self.repo), capture_output=True, text=True,
                              env=dict(os.environ))

    def test_an_uninitialized_repository_is_not_surveyed(self):
        proc = self._cli("mechanisms", "--path", str(self.repo))
        self.assertEqual(2, proc.returncode)
        self.assertIsNone(mechanisms.load(identity.peek_checkout_id(self.repo) or "x"))

    def test_probing_is_opt_in(self):
        self.write(".repohone/profile.yaml", profile.template())
        self.write("Makefile", "test:\n\techo ran > /dev/null\n")
        proc = self._cli("mechanisms", "--path", str(self.repo))
        self.assertEqual(0, proc.returncode)
        self.assertIn("not probed", proc.stdout)
        record = mechanisms.load(identity.peek_checkout_id(self.repo))
        self.assertFalse(record["probe"]["ran"])

    def test_capture_does_not_trigger_discovery(self):
        """A normal session must produce no inventory."""
        self.write(".repohone/profile.yaml", profile.template())
        profile.accept(self.repo, profile.load(self.repo))
        self.write("Makefile", "test:\n\tpytest\n")
        for name, extra in (("SessionStart", {"source": "startup"}),
                            ("UserPromptSubmit", {"prompt": "go", "prompt_id": "t1"}),
                            ("Stop", {"prompt_id": "t1"}),
                            ("SessionEnd", {"reason": "clear"})):
            payload = dict(extra, hook_event_name=name, session_id="s-1",
                           cwd=str(self.repo), transcript_path="/tmp/x")
            subprocess.run([sys.executable, str(HOOK), "hook"], input=json.dumps(payload),
                           cwd=str(self.repo), capture_output=True, text=True,
                           env=dict(os.environ))
        checkout = identity.peek_checkout_id(self.repo)
        self.assertIsNone(mechanisms.load(checkout),
                          "capture produced a mechanism inventory on its own")

    def test_a_cached_inventory_can_be_shown_without_rediscovery(self):
        self.write(".repohone/profile.yaml", profile.template())
        self.write("Makefile", "test:\n\tpytest\n")
        self._cli("mechanisms", "--path", str(self.repo))
        (self.repo / "Makefile").unlink()
        proc = self._cli("mechanisms", "--path", str(self.repo), "--cached")
        self.assertIn("make test", proc.stdout)


if __name__ == "__main__":
    unittest.main()
