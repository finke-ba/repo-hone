"""Cross-cutting boundary tests that enumerate every equivalent implementation."""
from __future__ import annotations

import ast
import inspect
import json
import os
import shutil
import tempfile
import textwrap
import unittest
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "src"
import sys

sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from repohone import (
    CONTRACT_VERSION,
    artifacts,
    bootstrap,
    diagnosis,
    doctor,
    mechanisms,
    paths,
    proposal,
    rule,
    session_validation,
)


class EveryArtifactLoaderIsStrict(unittest.TestCase):
    def test_runtime_validator_understands_every_normative_schema_keyword(self):
        self.assertEqual([], session_validation.unsupported_keywords())

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-artifacts-"))
        self.old_data = os.environ.get("REPOHONE_DATA_DIR")
        os.environ["REPOHONE_DATA_DIR"] = str(self.tmp)
        self.checkout = "checkout-test"

    def tearDown(self):
        if self.old_data is None:
            os.environ.pop("REPOHONE_DATA_DIR", None)
        else:
            os.environ["REPOHONE_DATA_DIR"] = self.old_data
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cases(self):
        return [
            ("rule", rule.candidates_dir(self.checkout) / "rule_deadbeefdeadbeef.json",
             lambda: rule.load(self.checkout, "rule_deadbeefdeadbeef")),
            ("diagnosis", diagnosis.diagnoses_dir(self.checkout) / "dx_test.json",
             lambda: diagnosis.load_all(self.checkout)),
            ("mechanisms", mechanisms.path_for(self.checkout),
             lambda: mechanisms.load(self.checkout)),
            ("proposal", proposal.path_for(self.checkout, "prop_deadbeef"),
             lambda: proposal.load(self.checkout, "prop_deadbeef")),
            ("bootstrap", bootstrap.path_for(self.checkout),
             lambda: bootstrap.load(self.checkout)),
        ]

    def test_a_directory_or_dangling_link_is_damaged_not_missing(self):
        """Regression: `load` returned None — "missing" — for anything that was
        not a regular file, so `load_all` silently dropped the member its own
        docstring promised never to drop."""
        for kind, target, read in self.cases():
            for shape in ("directory", "dangling link"):
                with self.subTest(kind=kind, shape=shape):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if shape == "directory":
                        target.mkdir()
                    else:
                        target.symlink_to(self.tmp / "gone.json")
                    try:
                        with self.assertRaises(artifacts.MalformedArtifact):
                            read()
                    finally:
                        if target.is_symlink():
                            target.unlink()
                        else:
                            target.rmdir()

    def write(self, target: Path, kind: str, **changes):
        target.parent.mkdir(parents=True, exist_ok=True)
        value = {"schema": artifacts.REGISTRY[kind][0], "schema_version": 1,
                 "contract_version": CONTRACT_VERSION}
        value.update(changes)
        target.write_text(json.dumps(value), encoding="utf-8")

    def test_every_loader_refuses_a_future_shape(self):
        for kind, target, load in self.cases():
            with self.subTest(kind=kind):
                self.write(target, kind, schema_version=99)
                with self.assertRaises(artifacts.UnsupportedVersion):
                    load()

    def test_every_loader_refuses_a_future_contract(self):
        for kind, target, load in self.cases():
            with self.subTest(kind=kind):
                self.write(target, kind, contract_version="99.0")
                with self.assertRaises(artifacts.UnsupportedVersion):
                    load()

    def test_every_loader_reports_malformed_json(self):
        for kind, target, load in self.cases():
            with self.subTest(kind=kind):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("{malformed", encoding="utf-8")
                with self.assertRaises(artifacts.MalformedArtifact):
                    load()

    def test_schema_identity_cannot_be_borrowed_from_another_artifact(self):
        for kind, target, load in self.cases():
            with self.subTest(kind=kind):
                self.write(target, kind, schema="repohone.session/v1")
                with self.assertRaises(artifacts.MalformedArtifact):
                    load()

    def test_header_types_fail_closed_instead_of_crashing(self):
        for kind in artifacts.REGISTRY:
            base = {"schema": artifacts.REGISTRY[kind][0], "schema_version": 1,
                    "contract_version": CONTRACT_VERSION}
            for field, value in (("schema_version", True), ("schema_version", []),
                                 ("contract_version", []), ("contract_version", None)):
                with self.subTest(kind=kind, field=field, value=value):
                    candidate = dict(base, **{field: value})
                    with self.assertRaises(artifacts.ArtifactError):
                        artifacts.validate(candidate, kind)

    def test_a_valid_header_is_not_a_valid_artifact(self):
        """Every kind must cross its full normative schema boundary."""
        for kind in artifacts.REGISTRY:
            with self.subTest(kind=kind):
                header = {"schema": artifacts.REGISTRY[kind][0],
                          "schema_version": 1,
                          "contract_version": CONTRACT_VERSION}
                with self.assertRaises(artifacts.MalformedArtifact):
                    artifacts.validate(header, kind)

    def test_every_contextual_artifact_is_bound_to_checkout_and_filename(self):
        for kind in artifacts.REGISTRY:
            value = {"repository": {"checkout_id": "source"}} if kind == "session" else {
                "checkout_id": "source"}
            id_field = artifacts.ID_FIELDS.get(kind)
            if id_field:
                value[id_field] = "artifact-id"
            with self.subTest(kind=kind, boundary="checkout"):
                with self.assertRaises(artifacts.MalformedArtifact):
                    artifacts.validate_context(
                        value, kind, expected_checkout_id="target")
            if id_field:
                with self.subTest(kind=kind, boundary="filename"):
                    with self.assertRaises(artifacts.MalformedArtifact):
                        artifacts.validate_context(
                            value, kind, expected_artifact_id="filename-id")

    def test_a_malformed_session_is_reported_without_validator_exceptions(self):
        value = {"schema": "repohone.session/v1", "schema_version": 1,
                 "contract_version": CONTRACT_VERSION, "turns": 1}
        problems = session_validation.validate_session(value)
        self.assertTrue(problems)
        self.assertTrue(any("turns" in problem for problem in problems))

    def test_a_schema_valid_proposal_cannot_be_moved_to_another_checkout(self):
        source, target = "checkout-source", "checkout-target"
        diagnosis_record = {"diagnosis_id": "dx_12345678",
                            "session_id": "rh_12345678", "fingerprint": None,
                            "recurrence": {}, "root_cause": None,
                            "required_property": None}
        built = proposal.build_record(
            diagnosis_record, source, "INSUFFICIENT_EVIDENCE",
            "2026-01-01T00:00:00Z", "prop_12345678", failure="insufficient")
        destination = proposal.path_for(target, built["proposal_id"])
        artifacts.atomic_write(destination, built)
        with self.assertRaises(artifacts.MalformedArtifact):
            proposal.load(target, built["proposal_id"])

    def test_every_semantic_writer_uses_the_shared_atomic_boundary(self):
        for module in (rule, diagnosis, mechanisms, proposal, bootstrap):
            source = inspect.getsource(module.save)
            with self.subTest(module=module.__name__):
                self.assertIn("artifacts.save", source)
                self.assertNotIn("write_text", source)
                self.assertNotIn("json.dump", source)
        self.assertIn("artifacts.atomic_write", inspect.getsource(proposal.stage))

    def test_an_invalid_id_is_rejected_before_it_becomes_a_path(self):
        with self.assertRaises(artifacts.MalformedArtifact):
            proposal.load(self.checkout, "../../outside")
        with self.assertRaises(artifacts.MalformedArtifact):
            rule.load(self.checkout, "../outside")
        self.assertFalse((self.tmp / "outside").exists())

    def test_staged_proposal_contents_have_a_strict_shape(self):
        with self.assertRaises(artifacts.MalformedArtifact):
            proposal.stage(self.checkout, "prop_deadbeef", {"x": 1})

    def test_project_json_backups_are_not_interpreted_as_artifacts(self):
        backup = (proposal.backup_dir(self.checkout, "prop_deadbeef") /
                  "package.json")
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text("historical bytes need not remain valid JSON")
        self.assertEqual([], doctor._unsupported_records(self.checkout))


class AbsentIsNotDamaged(unittest.TestCase):
    """`Path.is_file()`, `is_dir()`, `exists()` and `glob()` answer "absent"
    when the truth is "unreadable". Each place that trusted one of them turned
    damaged evidence or settings into none; these helpers are the replacement."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-present-"))

    def tearDown(self):
        for path in self.tmp.rglob("*"):
            try:
                os.chmod(path, 0o755, follow_symlinks=False)
            except (OSError, NotImplementedError):
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_only_nothing_at_all_is_absent(self):
        regular = self.tmp / "file"
        regular.write_text("x")
        folder = self.tmp / "folder"
        folder.mkdir()
        dangling = self.tmp / "dangling"
        dangling.symlink_to(self.tmp / "gone")
        self.assertIsNone(paths.read_regular(self.tmp / "absent"))
        self.assertIsNone(paths.entries(self.tmp / "absent"))
        self.assertFalse(paths.present(self.tmp / "absent"))
        self.assertEqual("x", paths.read_regular(regular))
        self.assertEqual([], paths.entries(folder))
        for damaged in (folder, dangling):
            with self.subTest(path=damaged.name), self.assertRaises(OSError):
                paths.read_regular(damaged)
        for damaged in (regular, dangling):
            with self.subTest(path=damaged.name), self.assertRaises(OSError):
                paths.entries(damaged)
        self.assertTrue(paths.present(dangling), "a dangling link is something")

    def test_an_unanswerable_question_raises(self):
        locked = self.tmp / "locked"
        (locked / "inside").mkdir(parents=True)
        os.chmod(locked, 0)
        with self.assertRaises(OSError):
            paths.present(locked / "inside")
        with self.assertRaises(OSError):
            paths.entries(locked)


class NoEvidenceReadTrustsAnExistencePredicate(unittest.TestCase):
    """Pathlib predicates answer "absent" when the truth is "unreadable". Evidence
    and privacy reads use `paths` instead; an exception is added here, with why."""

    MODULES = ("artifacts", "diagnosis", "doctor", "fingerprint", "profile",
               "proposal", "reasoning", "record", "rule", "state")
    PREDICATES = {"is_file", "is_dir", "exists", "glob", "rglob"}
    ALLOWED = {
        # Finds a directory to try a write in; the write itself is the test.
        ("doctor", "_probe_write"),
        # Host configuration, where "missing" is itself reported as a failure.
        ("doctor", "_integration"),
        ("doctor", "_unrunnable"),
    }

    def test_the_idiom_is_gone_from_evidence_and_privacy_reads(self):
        found = []
        for module in self.MODULES:
            source = (CORE / "repohone" / f"{module}.py").read_text()
            for function in ast.walk(ast.parse(source)):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for node in ast.walk(function):
                    if (isinstance(node, ast.Call)
                            and isinstance(node.func, ast.Attribute)
                            and node.func.attr in self.PREDICATES
                            and (module, function.name) not in self.ALLOWED):
                        found.append(f"{module}.{function.name}: .{node.func.attr}() "
                                     f"at line {node.lineno}")
        self.assertEqual([], sorted(set(found)))


class ProjectMutationHasOneBoundary(unittest.TestCase):
    def test_apply_has_no_direct_path_mutation(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(proposal._apply_locked)))
        forbidden = {"mkdir", "write_text", "write_bytes", "unlink", "rename",
                     "replace", "rmtree", "move"}
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in forbidden]
        self.assertEqual([], calls,
                         "project mutation must go through the descriptor-confined primitive")
        names = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("_replace_inside", names)

if __name__ == "__main__":
    unittest.main()


class EveryTestModuleImportsFromEitherRoot(unittest.TestCase):
    """Regression: two modules imported `core.tests.helpers`, so discovery from
    the repository root — the documented command — failed to load 360 tests."""

    def test_discovery_from_the_root_imports_every_module(self):
        import subprocess
        names = sorted(p.stem for p in Path(__file__).resolve().parent.glob("test_*.py"))
        probe = ("import sys; sys.path.insert(0, 'tests')\n"
                 "for name in sys.argv[1:]:\n    __import__(name)\n")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run([sys.executable, "-c", probe, *names], cwd=str(CORE.parent),
                              capture_output=True, text=True, env=env)
        self.assertEqual(0, proc.returncode, proc.stderr[-800:])


class NoTestReachesTheRealClaudeConfiguration(unittest.TestCase):
    """Regression: install tests wrote a plugin and an `enabledPlugins` entry
    into the developer's own ~/.claude."""

    def test_every_test_module_isolates_it_before_importing_repohone(self):
        import re
        for path in sorted(Path(__file__).resolve().parent.glob("test_*.py")):
            lines = path.read_text().splitlines()
            isolated = next((i for i, line in enumerate(lines)
                             if line.startswith("import isolation")), None)
            first = next(i for i, line in enumerate(lines)
                         if re.match(r"(from|import) repohone", line))
            with self.subTest(module=path.name):
                self.assertIsNotNone(isolated)
                self.assertLess(isolated, first)

    def test_the_configuration_in_use_is_not_the_developers(self):
        from repohone.adapters import claude
        self.assertNotEqual((Path.home() / ".claude").resolve(), claude.config_dir().resolve())
        self.assertTrue(str(claude.config_dir()).startswith(tempfile.gettempdir()))

    def test_no_session_id_leaks_in_from_the_agent_running_the_tests(self):
        self.assertNotIn("CLAUDE_CODE_SESSION_ID", os.environ)


class AnInterruptedWriteIsNotAMember(unittest.TestCase):
    """A writer killed between staging and rename leaves `.rh-*.json` behind.

    Regression: listings read it as an artifact, so `repohone list` aborted and
    every diagnosis in the repository ended INSUFFICIENT_EVIDENCE."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-staging-"))
        self.old_data = os.environ.get("REPOHONE_DATA_DIR")
        os.environ["REPOHONE_DATA_DIR"] = str(self.tmp)
        self.checkout = "checkout-staging"

    def tearDown(self):
        if self.old_data is None:
            os.environ.pop("REPOHONE_DATA_DIR", None)
        else:
            os.environ["REPOHONE_DATA_DIR"] = self.old_data
        shutil.rmtree(self.tmp, ignore_errors=True)

    def leave(self, directory, age_s=0):
        directory.mkdir(parents=True, exist_ok=True)
        leftover = directory / f"{artifacts.STAGING_PREFIX}k1ll3d.json"
        leftover.write_text('{"schema": "repohone.sess')
        if age_s:
            os.utime(leftover, (leftover.stat().st_atime - age_s,) * 2)
        return leftover

    def test_every_listing_skips_a_staging_file(self):
        from repohone import fingerprint, record
        listings = [
            (paths.records_dir(self.checkout), lambda: record.list_sessions(self.checkout)),
            (diagnosis.diagnoses_dir(self.checkout), lambda: diagnosis.load_all(self.checkout)),
            (rule.candidates_dir(self.checkout), lambda: rule.load_all(self.checkout)),
            (proposal.directory(self.checkout), lambda: proposal.load_all(self.checkout)),
        ]
        for directory, listing in listings:
            with self.subTest(directory=directory.name):
                self.leave(directory)
                self.assertEqual([], listing())
        self.assertEqual((), fingerprint.known(self.checkout, None).problems)
        verdict, problems = fingerprint._is_clone_of(self.checkout, "repo_x")
        self.assertEqual([], problems)

    def test_any_other_stray_file_is_still_refused(self):
        directory = diagnosis.diagnoses_dir(self.checkout)
        directory.mkdir(parents=True)
        (directory / "dx_0123456789abcdef.json").write_text('{"schema": "repohone.diag')
        with self.assertRaises(artifacts.MalformedArtifact):
            diagnosis.load_all(self.checkout)

    def test_doctor_names_a_settled_leftover_without_calling_it_corrupt(self):
        self.leave(paths.records_dir(self.checkout), age_s=3600)
        self.leave(diagnosis.diagnoses_dir(self.checkout))            # still being written
        self.assertEqual([], doctor._unsupported_records(self.checkout))
        self.assertEqual([f"records/{artifacts.STAGING_PREFIX}k1ll3d.json"],
                         doctor._interrupted_writes(self.checkout))
