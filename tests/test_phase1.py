"""Phase 1 implementation tests: installation, consent, identity, storage, capture.

Every test names the ARCHITECTURE section it enforces. Records produced here are
validated against the normative schema and record_invariants.py, so an
implementation that drifts from the frozen contract fails here.

Run:  pip install jsonschema
      python3 -m unittest discover -s core/tests -t core
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

CORE = Path(__file__).resolve().parent.parent / "src"
ROOT = CORE.parent
sys.path.insert(0, str(CORE))

sys.path.insert(0, str(Path(__file__).resolve().parent))
import isolation  # noqa: F401
from repohone import (
    CONTRACT_VERSION,
    CORE_VERSION,
    capture,
    cli,
    deadline,
    doctor,
    identity,
    paths,
    profile,
    record,
    snapshot,
    state,
)
from repohone.adapters import claude
from repohone.record_invariants import check_record

HOOK = CORE / "repohone_hook.py"
SCHEMA_PATH = CORE / "repohone" / "schemas" / "session-record.v1.schema.json"

try:
    import jsonschema
    VALIDATOR = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
except ImportError:
    VALIDATOR = None


def git(args, cwd, check=True):
    return subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                          text=True, check=check).stdout.strip()


def git_ignored(repo, relative) -> int:
    return subprocess.run(["git", "check-ignore", "-q", relative], cwd=str(repo)).returncode


class Fixture(unittest.TestCase):
    """A real git repository plus an isolated RepoHone data directory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="repohone-test-"))
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        self.data = self.tmp / "data"
        git(["init", "-q"], self.repo)
        git(["config", "user.email", "dev@example.com"], self.repo)
        git(["config", "user.name", "Dev"], self.repo)
        git(["remote", "add", "origin", "git@github.com:Acme/App.git"], self.repo)
        (self.repo / "calc.py").write_text("def mean(xs):\n    return sum(xs)/len(xs)\n")
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "init"], self.repo)
        self.config = os.environ["CLAUDE_CONFIG_DIR"]
        os.environ["CLAUDE_CONFIG_DIR"] = str(self.tmp / "claude")
        self.env = dict(os.environ, REPOHONE_DATA_DIR=str(self.data))
        os.environ["REPOHONE_DATA_DIR"] = str(self.data)

    def tearDown(self):
        os.environ.pop("REPOHONE_DATA_DIR", None)
        os.environ["CLAUDE_CONFIG_DIR"] = self.config
        shutil.rmtree(self.tmp, ignore_errors=True)

    def init_repo(self):
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(self.repo, profile.load(self.repo))

    def fire(self, name, session="s-1", **kw):
        payload = {"hook_event_name": name, "session_id": session,
                   "cwd": str(self.repo), "transcript_path": f"/tmp/{session}.jsonl"}
        payload.update(kw)
        return subprocess.run([sys.executable, str(HOOK), "hook"],
                              input=json.dumps(payload), cwd=str(self.repo),
                              capture_output=True, text=True, env=self.env)

    def the_record(self, session="s-1"):
        checkout = identity.peek_checkout_id(self.repo)
        self.assertIsNotNone(checkout, "no checkout id assigned")
        sid = identity.session_id("claude-code", session)
        rec = record.load(checkout, sid)
        self.assertIsNotNone(rec, f"no record for {sid}")
        return rec

    def assert_conformant(self, rec):
        if VALIDATOR is not None:
            errors = sorted(VALIDATOR.iter_errors(rec), key=lambda e: e.path)
            self.assertEqual(
                [], ["/".join(map(str, e.path)) + ": " + e.message for e in errors])
        self.assertEqual([], check_record(rec))

    def install(self, source=None):
        """The machine-level install; returns the plugin's hooks file."""
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            if source is None:
                self.assertEqual(0, cli.cmd_install(argparse.Namespace(path=None)))
            else:
                claude.write_plugin(sys.executable, source, "test")
        return claude.plugin_dir() / "hooks" / "hooks.json"

    def rewrite_plugin_hooks(self, change):
        path = claude.plugin_dir() / "hooks" / "hooks.json"
        doc = json.loads(path.read_text())
        change(doc["hooks"])
        path.write_text(json.dumps(doc))


class PrivacyBoundary(Fixture):
    """ARCH §5, §6 — global install does not authorize observation."""

    def test_uninitialized_repository_captures_nothing(self):
        self.fire("SessionStart", source="startup")
        self.fire("UserPromptSubmit", prompt="confidential client work", prompt_id="t1")
        self.assertFalse(self.data.exists(), "telemetry written without initialization")
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))

    def test_uninitialized_announcer_offers_initialization(self):
        out = self.fire("SessionStart", source="startup").stdout
        self.assertIn("not initialized", out)
        self.assertNotIn(str(self.repo), out, "announcer leaked an absolute path")

    def test_invalid_profile_suspends_capture(self):
        (self.repo / ".repohone").mkdir()
        (self.repo / ".repohone" / "profile.yaml").write_text("profile_version: 99\n")
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))
        self.assertFalse(self.data.exists(), "an unreadable profile is not permission")

    def test_every_ambiguous_profile_suspends_capture_without_side_effects(self):
        (self.repo / ".repohone").mkdir()
        target = self.repo / ".repohone" / "profile.yaml"
        cases = {
            "boolean-version": "profile_version: true\ncontent_policy: hash_only\n",
            "duplicate-version": "profile_version: 1\nprofile_version: 1\n",
            "duplicate-privacy": (
                "profile_version: 1\ncontent_policy: hash_only\n"
                "content_policy: redacted\n"),
        }
        for index, (name, text) in enumerate(cases.items(), 1):
            with self.subTest(name=name):
                target.write_text(text)
                result = self.fire("UserPromptSubmit", session=f"s-{index}",
                                   prompt="private review text", prompt_id="t1")
                self.assertEqual(0, result.returncode)
                self.assertEqual(profile.INVALID, profile.load(self.repo).state)
                self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))
                self.assertFalse(self.data.exists(),
                                 "ambiguous configuration was treated as consent")

    def test_invalid_profile_announces_a_diagnostic(self):
        (self.repo / ".repohone").mkdir()
        (self.repo / ".repohone" / "profile.yaml").write_text("profile_version: 99\n")
        self.assertIn("doctor", self.fire("SessionStart", source="startup").stdout)

    def test_where_capture_is_on_the_agent_is_told_only_to_record_rules(self):
        """The live agent saved a stated rule to its own memory, and RepoHone
        never saw it."""
        self.init_repo()
        out = json.loads(self.fire("SessionStart", source="startup").stdout)
        text = out["hookSpecificOutput"]["additionalContext"]
        self.assertIn("repohone-rule skill", text)
        self.assertIn("memory", text)
        self.assertNotIn(str(self.repo), text)
        self.assertNotIn(str(Path.home()), text)


class FailureContract(Fixture):
    """ARCH §9.2 — capture never disrupts a coding session."""

    def test_every_event_exits_zero_and_prints_nothing(self):
        self.init_repo()
        for name, kw in (("UserPromptSubmit", {"prompt": "x", "prompt_id": "t1"}),
                         ("Stop", {"prompt_id": "t1"}),
                         ("StopFailure", {"prompt_id": "t1", "error": {"type": "overloaded"}}),
                         ("SessionEnd", {"reason": "clear"})):
            proc = self.fire(name, **kw)
            self.assertEqual(0, proc.returncode, f"{name}: {proc.stderr}")
            self.assertEqual("", proc.stdout, f"{name} wrote to stdout")

    def test_malformed_input_is_survivable(self):
        self.init_repo()
        proc = subprocess.run([sys.executable, str(HOOK), "hook"], input="{not json",
                              cwd=str(self.repo), capture_output=True, text=True, env=self.env)
        self.assertEqual(0, proc.returncode)
        self.assertEqual("", proc.stdout)

    def test_outside_a_git_repository_nothing_happens(self):
        loose = self.tmp / "loose"
        loose.mkdir()
        proc = subprocess.run([sys.executable, str(HOOK), "hook"],
                              input=json.dumps({"hook_event_name": "UserPromptSubmit",
                                                "session_id": "s", "cwd": str(loose),
                                                "prompt": "x", "prompt_id": "t1"}),
                              cwd=str(loose), capture_output=True, text=True, env=self.env)
        self.assertEqual(0, proc.returncode)
        self.assertEqual("", proc.stdout)

    def test_a_failed_snapshot_keeps_the_event(self):
        """ARCH §16.1 — absent evidence is honest; losing the prompt is not."""
        self.init_repo()
        original = snapshot.stable_tree

        def unstable(*a, **kw):
            raise snapshot.UnstableTree("forced")

        snapshot.stable_tree = unstable
        try:
            from repohone import capture
            capture.handle(claude.to_event(
                {"hook_event_name": "UserPromptSubmit", "session_id": "s-1",
                 "cwd": str(self.repo), "prompt": "keep me", "prompt_id": "t1"}))
        finally:
            snapshot.stable_tree = original
        rec = self.the_record()
        event = rec["turns"][0]["prompt_events"][0]
        self.assertIsNone(event["snapshot"])
        self.assertEqual("keep me", event["content"]["text"])
        self.assertTrue(rec["capture"]["errors"])
        self.assert_conformant(rec)


class Identity(unittest.TestCase):
    """ARCH §14, §15 — identity is derived, never looked up."""

    def test_session_id_is_order_independent_and_idempotent(self):
        a = identity.session_id("claude-code", "abc")
        b = identity.session_id("claude-code", "abc")
        self.assertEqual(a, b)
        self.assertNotEqual(a, identity.session_id("codex", "abc"))
        self.assertRegex(a, r"^rh_[0-9a-f]{8,}$")

    def test_ssh_and_https_remotes_give_one_repository_id(self):
        pairs = [("git@github.com:Owner/Repo.git", "https://github.com/Owner/Repo.git"),
                 ("ssh://git@gitlab.com:22/g/p.git", "https://gitlab.com/g/p")]
        for ssh, https in pairs:
            self.assertEqual(identity.repository_id(ssh), identity.repository_id(https))

    def test_a_repository_without_a_remote_has_no_repository_id(self):
        self.assertIsNone(identity.repository_id(None))
        self.assertIsNone(identity.repository_id("not a url"))


class UnidentifiedEventsAreNeverAttributed(Fixture):
    """ARCH §14 + the record schema: logical_turn_id is required and non-empty.

    An event that cannot be placed is recorded as a capture error. Silently
    dropping it would make a record with a missing turn look complete."""

    def test_session_id_is_none_when_the_host_supplies_none(self):
        for blank in (None, "", "   "):
            self.assertIsNone(identity.session_id("claude-code", blank))
        self.assertIsNotNone(identity.session_id("claude-code", "s-1"))

    def test_a_session_without_a_host_id_captures_nothing(self):
        self.init_repo()
        self.fire("UserPromptSubmit", session="", prompt="orphan work", prompt_id="t1")
        self.fire("Stop", session="", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        records = list((self.data / "checkouts" / checkout / "records").glob("*.json")) \
            if checkout and (self.data / "checkouts" / checkout / "records").exists() else []
        self.assertEqual([], records, "an id-less session produced a record")
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))

    def test_two_id_less_sessions_do_not_merge_into_one(self):
        """Hashing "" is stable, so both would land in the same record and their
        prompts would read as type-ahead within a single turn."""
        self.init_repo()
        self.fire("UserPromptSubmit", session="", prompt="session A work", prompt_id="a1")
        self.fire("UserPromptSubmit", session="", prompt="session B work", prompt_id="b1")
        checkout = identity.peek_checkout_id(self.repo)
        store = self.data / "checkouts" / checkout / "records" if checkout else None
        found = list(store.glob("*.json")) if store and store.exists() else []
        for path in found:
            rec = json.loads(path.read_text())
            texts = [e["content"].get("text")
                     for turn in rec["turns"] for e in turn["prompt_events"]]
            self.assertNotIn("session B work", texts,
                             "two sessions were merged into one record")

    def test_a_prompt_without_a_turn_id_is_refused_and_recorded(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="identified", prompt_id="t1")
        self.fire("UserPromptSubmit", prompt="orphan", prompt_id=None)
        rec = self.the_record()
        self.assertEqual(1, len(rec["turns"]))
        texts = [e["content"].get("text") for e in rec["turns"][0]["prompt_events"]]
        self.assertEqual(["identified"], texts)
        self.assertTrue(any("logical turn id" in e["error"]
                            for e in rec["capture"]["errors"]),
                        "an unplaceable prompt was dropped without a trace")
        self.assert_conformant(rec)

    def test_a_stop_for_an_unknown_turn_does_not_complete_the_latest_turn(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="real work", prompt_id="t1")
        self.fire("Stop", prompt_id="does-not-exist", last_assistant_message="done")
        rec = self.the_record()
        turn = rec["turns"][0]
        self.assertEqual("pending", turn["completion"])
        self.assertIsNone(turn["completion_source"])
        self.assertEqual([], turn["stop_events"])
        self.assertTrue(any("does-not-exist" in e["error"]
                            for e in rec["capture"]["errors"]))
        self.assert_conformant(rec)

    def test_a_matching_stop_still_completes_its_own_turn(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="real work", prompt_id="t1")
        self.fire("Stop", prompt_id="t1", last_assistant_message="done")
        turn = self.the_record()["turns"][0]
        self.assertEqual("stop", turn["completion"])
        self.assertEqual("observed", turn["completion_source"])


class CheckoutScope(Fixture):
    """ARCH §15 — telemetry is checkout-scoped."""

    def test_checkout_id_is_stable_and_uncommitted(self):
        first = identity.checkout_id(self.repo)
        self.assertEqual(first, identity.checkout_id(self.repo))
        self.assertEqual("", git(["status", "--porcelain"], self.repo),
                         "checkout id must not appear in the working tree")

    def test_a_worktree_is_its_own_checkout(self):
        other = self.tmp / "wt"
        git(["worktree", "add", "-q", str(other), "-b", "feature"], self.repo)
        self.assertNotEqual(identity.checkout_id(self.repo), identity.checkout_id(other))

    def test_peek_never_creates_the_marker(self):
        self.assertIsNone(identity.peek_checkout_id(self.repo))
        self.assertIsNone(identity.peek_checkout_id(self.repo))


class AtomicCheckoutIdentity(Fixture):
    """Audit B1: concurrent first use minted several ids, splitting one checkout
    across several telemetry stores."""

    def test_concurrent_first_use_yields_one_id(self):
        import concurrent.futures as futures
        ids, errors = set(), 0
        with futures.ThreadPoolExecutor(32) as pool:
            for future in [pool.submit(identity.checkout_id, self.repo)
                           for _ in range(32)]:
                try:
                    ids.add(future.result())
                except Exception:
                    errors += 1
        self.assertEqual(0, errors)
        self.assertEqual(1, len(ids), f"{len(ids)} checkout ids were minted")
        marker = (self.repo / ".git" / "repohone" / "checkout-id").read_text().strip()
        self.assertEqual({marker}, ids)

    def test_no_staging_files_are_left_behind(self):
        identity.checkout_id(self.repo)
        stray = [p.name for p in (self.repo / ".git" / "repohone").iterdir()
                 if p.name not in ("checkout-id", identity.HOME_FILE)]
        self.assertEqual([], stray)


class CheckoutScopedRefs(Fixture):
    """Audit B2: worktrees share one ref store, so refs must be checkout-owned."""

    def _worktree(self):
        other = self.tmp / "wt"
        git(["worktree", "add", "-q", str(other), "-b", "feature"], self.repo)
        (other / ".repohone").mkdir(exist_ok=True)
        (other / ".repohone" / "profile.yaml").write_text(profile.template())
        profile.accept(other, profile.load(other))
        return other

    def _capture_in(self, path, session):
        for name, extra in (("UserPromptSubmit", {"prompt": "go", "prompt_id": f"{session}-t1"}),
                            ("Stop", {"prompt_id": f"{session}-t1"})):
            payload = dict(extra, hook_event_name=name, session_id=session,
                           cwd=str(path), transcript_path="/tmp/x")
            subprocess.run([sys.executable, str(HOOK), "hook"], input=json.dumps(payload),
                           cwd=str(path), capture_output=True, text=True, env=self.env)

    def test_refs_are_namespaced_by_checkout(self):
        self.init_repo()
        self._capture_in(self.repo, "s-a")
        checkout = identity.peek_checkout_id(self.repo)
        refs = git(["for-each-ref", "--format=%(refname)", "refs/repohone/"], self.repo)
        self.assertTrue(refs)
        for ref in refs.splitlines():
            self.assertTrue(ref.startswith(snapshot.checkout_namespace(checkout)), ref)

    def test_purging_one_worktree_spares_the_other(self):
        self.init_repo()
        other = self._worktree()
        self._capture_in(self.repo, "s-a")
        self._capture_in(other, "s-b")
        mine = snapshot.checkout_namespace(identity.peek_checkout_id(self.repo))
        theirs = snapshot.checkout_namespace(identity.peek_checkout_id(other))
        self.assertNotEqual(mine, theirs)
        before = git(["for-each-ref", "--format=%(refname)", "refs/repohone/"],
                     other).splitlines()
        self.assertTrue([r for r in before if r.startswith(theirs)])

        subprocess.run([sys.executable, str(HOOK), "deinit", "--purge", "--yes",
                        "--path", str(self.repo)], capture_output=True, text=True,
                       env=self.env)
        after = git(["for-each-ref", "--format=%(refname)", "refs/repohone/"],
                    other).splitlines()
        self.assertEqual([r for r in before if r.startswith(theirs)],
                         [r for r in after if r.startswith(theirs)],
                         "purging one worktree destroyed the other's evidence")
        self.assertEqual([], [r for r in after if r.startswith(mine)],
                         "the purged checkout's own refs survived")

    def test_two_worktrees_reusing_a_session_id_do_not_collide(self):
        self.init_repo()
        other = self._worktree()
        (other / "calc.py").write_text("# different content\n")
        self._capture_in(self.repo, "shared")
        self._capture_in(other, "shared")
        for path in (self.repo, other):
            checkout = identity.peek_checkout_id(path)
            session = identity.session_id("claude-code", "shared")
            rec = record.load(checkout, session)
            events = [e for t in rec["turns"]
                      for e in t["prompt_events"] + t["stop_events"]]
            self.assertTrue(all(e["snapshot"] for e in events),
                            f"{path.name} lost a snapshot to a ref collision")
            self.assertEqual([], rec["capture"]["errors"])

    def test_doctor_fails_when_referenced_evidence_is_gone(self):
        self.init_repo()
        self._capture_in(self.repo, "s-a")
        refs = git(["for-each-ref", "--format=%(refname)", "refs/repohone/"],
                   self.repo).splitlines()
        git(["update-ref", "-d", refs[0]], self.repo)
        rows = doctor.run(self.repo)
        refs_row = [r for r in rows if r[0] == "snapshot refs"][0]
        self.assertEqual(doctor.FAIL, refs_row[1])
        self.assertIn("no longer resolve", refs_row[2])


class SameSessionOrdering(Fixture):
    """Audit B4: a later prompt whose snapshot finished first became
    prompt_events[0] — the immutable turn baseline (§18)."""

    def test_a_later_prompt_cannot_overtake_the_baseline(self):
        import threading
        self.init_repo()
        from repohone import capture as capture_mod
        from repohone.adapters import claude as adapter
        original = snapshot.capture

        def slow_first(*args, **kwargs):
            if args[4] == 1:
                time.sleep(0.5)
            return original(*args, **kwargs)

        def fire(text):
            capture_mod.handle(adapter.to_event(
                {"hook_event_name": "UserPromptSubmit", "session_id": "s-order",
                 "cwd": str(self.repo), "prompt": text, "prompt_id": "t1"}))

        snapshot.capture = slow_first
        try:
            first = threading.Thread(target=fire, args=("original",))
            first.start()
            time.sleep(0.1)
            second = threading.Thread(target=fire, args=("type-ahead",))
            second.start()
            first.join()
            second.join()
        finally:
            snapshot.capture = original

        rec = self.the_record("s-order")
        events = rec["turns"][0]["prompt_events"]
        self.assertEqual([1, 2], [e["ordinal"] for e in events])
        self.assertEqual("original", events[0]["content"]["text"],
                         "a later prompt became the turn baseline")
        self.assertEqual([], check_record(rec))


class InstallPreservesProjectHooks(Fixture):
    """Audit B9: install replaced an existing project hook. RepoHone observes;
    it must never remove what the project already runs (§25)."""

    def _install(self):
        return subprocess.run([sys.executable, str(HOOK), "install",
                               "--path", str(self.repo)],
                              capture_output=True, text=True, env=self.env)

    def _hooks(self, name="settings.local.json"):
        return json.loads((self.repo / ".claude" / name).read_text())["hooks"]

    def test_an_existing_hook_survives(self):
        for name in ("settings.json", "settings.local.json"):
            (self.repo / ".claude").mkdir(exist_ok=True)
            (self.repo / ".claude" / name).write_text(json.dumps(
                {"hooks": {"Stop": [{"hooks": [{"type": "command",
                                                "command": "existing-check"}]}]}}))
        self._install()
        subprocess.run([sys.executable, str(HOOK), "init", "--path", str(self.repo)],
                       capture_output=True, text=True, env=self.env)
        for name in ("settings.json", "settings.local.json"):
            commands = [h["command"] for g in self._hooks(name)["Stop"] for h in g["hooks"]]
            self.assertEqual(["existing-check"], commands)

    def test_installing_twice_changes_nothing(self):
        self._install()
        plugin = claude.plugin_dir()
        before = {p.relative_to(plugin): p.read_bytes() for p in plugin.rglob("*") if p.is_file()}
        settings = (claude.config_dir() / "settings.json").read_text()
        self._install()
        after = {p.relative_to(plugin): p.read_bytes() for p in plugin.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(settings, (claude.config_dir() / "settings.json").read_text())


class ADamagedProfileIsNotAbsent(Fixture):
    """Regression: `path.exists()` read an unreadable profile as no profile,
    so a damaged configuration was reported as "uninitialized" and the
    announcer offered to initialize over it."""

    def test_each_damaged_shape_is_invalid_and_captures_nothing(self):
        target = self.repo / ".repohone" / "profile.yaml"
        target.parent.mkdir()
        for shape in ("directory", "dangling link"):
            with self.subTest(shape=shape):
                if shape == "directory":
                    target.mkdir()
                else:
                    target.symlink_to(self.tmp / "nowhere.yaml")
                loaded = profile.load(self.repo)
                self.assertEqual(profile.INVALID, loaded.state)
                self.assertFalse(loaded.capture_enabled)
                if target.is_symlink():
                    target.unlink()
                else:
                    target.rmdir()
        self.assertEqual(profile.UNINITIALIZED, profile.load(self.repo).state)


class ProfileFailsClosed(unittest.TestCase):
    """ARCH §7 — unknown configuration is never silently ignored."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / ".repohone").mkdir()
        self.path = self.tmp / ".repohone" / "profile.yaml"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _state(self, text):
        self.path.write_text(text)
        return profile.load(self.tmp)

    def test_template_is_active(self):
        self.assertEqual(profile.ACTIVE, self._state(profile.template()).state)

    def test_missing_profile_is_uninitialized(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(profile.UNINITIALIZED, profile.load(self.tmp).state)

    def test_unknown_key_is_invalid(self):
        self.assertEqual(profile.INVALID, self._state("profile_version: 1\nsurprise: 1\n").state)

    def test_future_profile_version_is_invalid(self):
        self.assertEqual(profile.INVALID, self._state("profile_version: 2\n").state)

    def test_incompatible_core_requirement_is_invalid(self):
        self.assertEqual(profile.INVALID,
                         self._state('profile_version: 1\nrequires_core: ">=99.0"\n').state)

    def test_unparseable_profile_is_invalid(self):
        self.assertEqual(profile.INVALID, self._state("\t: broken: [\n").state)

    def test_bad_enum_value_is_invalid(self):
        self.assertEqual(profile.INVALID,
                         self._state("profile_version: 1\nreasoning_egress: everything\n").state)

    def test_boolean_is_not_an_integer_profile_version(self):
        self.assertEqual(profile.INVALID,
                         self._state("profile_version: true\n").state)

    def test_duplicate_keys_are_invalid_even_when_values_match(self):
        result = self._state("profile_version: 1\nprofile_version: 1\n")
        self.assertEqual(profile.INVALID, result.state)
        self.assertIn("duplicate", result.problems[0])

    def test_duplicate_privacy_keys_never_choose_the_less_private_value(self):
        result = self._state(
            "profile_version: 1\ncontent_policy: hash_only\ncontent_policy: redacted\n")
        self.assertEqual(profile.INVALID, result.state)

    def test_every_known_key_rejects_duplicates(self):
        values = {
            "profile_version": "1", "requires_core": '">=0.4,<0.5"',
            "reasoning_egress": "interactive", "content_policy": "hash_only",
            "tool_capture": "false",
        }
        for key, value in values.items():
            with self.subTest(key=key):
                lines = ["profile_version: 1"] if key != "profile_version" else []
                lines += [f"{key}: {value}", f"{key}: {value}"]
                self.assertEqual(profile.INVALID,
                                 self._state("\n".join(lines) + "\n").state)

    def test_noncanonical_version_types_fail_closed(self):
        for value in ("false", '"1"', "null", "[]", "{}"):
            with self.subTest(value=value):
                self.assertEqual(profile.INVALID,
                                 self._state(f"profile_version: {value}\n").state)


class Ordinals(Fixture):
    """ARCH §17 — atomic, monotonic, never derived from git refs."""

    def setUp(self):
        super().setUp()
        self.checkout = identity.checkout_id(self.repo)
        state.initialize(self.checkout)

    def test_allocation_is_scoped_per_turn_and_class(self):
        a = [state.allocate_ordinal(self.checkout, "rh_a", "t1", "prompt") for _ in range(3)]
        b = [state.allocate_ordinal(self.checkout, "rh_a", "t2", "prompt") for _ in range(2)]
        c = [state.allocate_ordinal(self.checkout, "rh_a", "", "session_end") for _ in range(2)]
        self.assertEqual([1, 2, 3], a)
        self.assertEqual([1, 2], b)
        self.assertEqual([1, 2], c)

    def test_deleting_a_ref_does_not_reuse_its_ordinal(self):
        first = state.allocate_ordinal(self.checkout, "rh_a", "t1", "prompt")
        git(["update-ref", "-d", f"refs/repohone/snapshots/rh_a/t1/prompt/{first}"],
            self.repo, check=False)
        self.assertEqual(first + 1,
                         state.allocate_ordinal(self.checkout, "rh_a", "t1", "prompt"))

    def test_concurrent_allocation_never_duplicates(self):
        results, lock = [], threading.Lock()

        def worker():
            got = [state.allocate_ordinal(self.checkout, "rh_a", "t9", "prompt")
                   for _ in range(15)]
            with lock:
                results.extend(got)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(results), len(set(results)))
        self.assertEqual(sorted(results), list(range(1, len(results) + 1)))

    def test_state_db_uses_wal(self):
        with state.connect(paths.state_db(self.checkout)) as conn:
            self.assertEqual("wal", conn.execute("PRAGMA journal_mode").fetchone()[0].lower())


class Snapshots(Fixture):
    """ARCH §16, §16.1 — what a snapshot covers and what it claims."""

    def setUp(self):
        super().setUp()
        self.checkout = identity.checkout_id(self.repo)
        state.initialize(self.checkout)
        (self.repo / ".gitignore").write_text("build/\n*.env\n")
        (self.repo / "secret.env").write_text("TOKEN=x\n")
        (self.repo / "build").mkdir()
        (self.repo / "build" / "o.bin").write_text("artifact\n")
        (self.repo / "new_service.py").write_text("# created by the agent\n")

    def _capture(self, cls="prompt", ordinal=1, turn="t1"):
        return snapshot.capture(self.repo, self.checkout, "rh_a", cls, ordinal, turn,
                                record.now())

    def _tree_files(self, tree):
        return git(["ls-tree", "-r", "--name-only", tree], self.repo).split()

    def test_captured_file_set_includes_untracked_non_ignored(self):
        files = self._tree_files(self._capture().tree)
        self.assertIn("new_service.py", files)
        self.assertIn("calc.py", files)

    def test_captured_file_set_excludes_ignored(self):
        files = self._tree_files(self._capture().tree)
        self.assertNotIn("secret.env", files)
        self.assertFalse([f for f in files if f.startswith("build/")])

    def test_the_real_index_is_never_modified(self):
        before = git(["status", "--porcelain"], self.repo)
        self._capture()
        self.assertEqual(before, git(["status", "--porcelain"], self.repo))

    def test_assume_unchanged_cannot_hide_working_tree_content(self):
        git(["update-index", "--assume-unchanged", "calc.py"], self.repo)
        (self.repo / "calc.py").write_text("def mean(xs):\n    return 99\n")
        snap = self._capture()
        captured = git(["show", f"{snap.tree}:calc.py"], self.repo)
        self.assertIn("return 99", captured)
        # Capture changed only its private index; the developer's hint survives.
        self.assertTrue(git(["ls-files", "-v", "calc.py"], self.repo).startswith("h "))

    def test_skip_worktree_cannot_hide_a_present_working_tree_edit(self):
        target = self.repo / "hidden-skip.txt"
        target.write_text("base\n")
        git(["add", "hidden-skip.txt"], self.repo)
        git(["commit", "-qm", "tracked"], self.repo)
        git(["update-index", "--skip-worktree", "hidden-skip.txt"], self.repo)
        target.write_text("hidden local edit\n")
        snap = self._capture()
        captured = git(["show", f"{snap.tree}:hidden-skip.txt"], self.repo)
        self.assertEqual("hidden local edit", captured)
        self.assertTrue(git(["ls-files", "-v", "hidden-skip.txt"], self.repo)
                        .startswith("S "))

    def test_ref_layout_matches_the_contract(self):
        """Refs carry checkout ownership: worktrees share one ref store."""
        self.assertEqual("refs/repohone/snapshots/co-1/rh_a/t1/prompt/1",
                         snapshot.ref_for("rh_a", "prompt", 1, "t1", "co-1"))
        self.assertEqual("refs/repohone/snapshots/co-1/rh_a/t1/stop-failure/2",
                         snapshot.ref_for("rh_a", "stop_failure", 2, "t1", "co-1"))
        self.assertEqual("refs/repohone/snapshots/co-1/rh_a/session-end/3",
                         snapshot.ref_for("rh_a", "session_end", 3, None, "co-1"))
        self.assertEqual("refs/repohone/snapshots/co-1",
                         snapshot.checkout_namespace("co-1"))

    def test_an_unsafe_turn_id_stays_collision_free(self):
        a = snapshot.safe_segment("weird/id:one")
        b = snapshot.safe_segment("weird/id:two")
        self.assertNotEqual(a, b)
        self.assertNotIn("/", a)
        self.assertNotIn(":", a)

    def test_snapshots_survive_gc(self):
        snap = self._capture()
        git(["gc", "--prune=now", "--quiet"], self.repo)
        self.assertEqual(snap.commit, git(["rev-parse", snap.ref], self.repo))

    def test_snapshots_never_enter_branch_history(self):
        self._capture()
        self.assertNotIn("repohone", git(["log", "--format=%s", "HEAD"], self.repo))

    def test_snapshots_are_not_pushed(self):
        self._capture()
        bare = self.tmp / "bare.git"
        git(["init", "-q", "--bare", str(bare)], self.tmp)
        git(["remote", "set-url", "origin", str(bare)], self.repo)
        git(["push", "-q", "--all", "origin"], self.repo, check=False)
        git(["push", "-q", "--tags", "origin"], self.repo, check=False)
        pushed = git(["for-each-ref", "--format=%(refname)"], bare)
        self.assertNotIn("repohone", pushed)

    def test_an_occupied_ordinal_is_an_integrity_error(self):
        """ARCH §17 — never a silent replacement."""
        snap = self._capture()
        (self.repo / "later.py").write_text("changed\n")
        with self.assertRaises(snapshot.IntegrityError):
            snapshot.capture(self.repo, self.checkout, "rh_a", "prompt", snap.ordinal,
                             "t1", record.now())

    def test_a_moving_tree_is_refused_not_guessed(self):
        stop = threading.Event()

        def churn():
            i = 0
            staged = self.tmp / "churn.tmp"
            while not stop.is_set():
                # Atomic, so `git add` never reads a half-written file and fails instead.
                staged.write_text(f"v = {i}\n")
                os.replace(staged, self.repo / "churn.py")
                i += 1
                time.sleep(0.005)

        writer = threading.Thread(target=churn)
        writer.start()
        time.sleep(0.05)
        try:
            with self.assertRaises(snapshot.UnstableTree):
                snapshot.stable_tree(self.repo,
                                     snapshot._scratch_index(self.checkout, "rh_a", self.repo),
                                     time.monotonic() + 3)
        finally:
            stop.set()
            writer.join()

    def test_a_quiet_tree_settles(self):
        tree, attempts = snapshot.stable_tree(
            self.repo, snapshot._scratch_index(self.checkout, "rh_a", self.repo),
            time.monotonic() + 5)
        self.assertTrue(tree)
        self.assertLessEqual(attempts, snapshot.STABILITY_ATTEMPTS)

    def test_a_checkout_between_tree_and_metadata_never_mixes_states(self):
        """A stable file tree is not enough: HEAD can change immediately after it."""
        original_branch = git(["branch", "--show-current"], self.repo)
        git(["checkout", "-qb", "other"], self.repo)
        (self.repo / "calc.py").write_text("branch = 'other'\n")
        git(["add", "calc.py"], self.repo)
        git(["commit", "-qm", "other state"], self.repo)
        git(["checkout", "-q", original_branch], self.repo)

        real = snapshot.stable_tree
        switched = False

        def checkout_after_tree(*args, **kwargs):
            nonlocal switched
            result = real(*args, **kwargs)
            if not switched:
                switched = True
                git(["checkout", "-q", "other"], self.repo)
            return result

        snapshot.stable_tree = checkout_after_tree
        try:
            snap = self._capture()
        finally:
            snapshot.stable_tree = real

        head_contents = git(["show", f"{snap.head}:calc.py"], self.repo)
        tree_contents = git(["show", f"{snap.tree}:calc.py"], self.repo)
        self.assertEqual(head_contents, tree_contents)
        self.assertEqual("other", snap.branch)
        self.assertEqual("branch = 'other'", tree_contents)

    def test_submodule_working_state_is_captured_separately(self):
        sub = self.tmp / "sub"
        sub.mkdir()
        git(["init", "-q"], sub)
        git(["config", "user.email", "d@e.com"], sub)
        git(["config", "user.name", "D"], sub)
        (sub / "lib.py").write_text("x = 1\n")
        git(["add", "-A"], sub)
        git(["commit", "-qm", "sub"], sub)
        git(["-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendor"],
            self.repo)
        git(["commit", "-qm", "add submodule"], self.repo)
        (self.repo / "vendor" / "lib.py").write_text("x = 2  # edited in the submodule\n")

        snap = self._capture()
        self.assertTrue(snap.submodules, "submodule working state was invisible")
        child = snap.submodules[0]
        self.assertEqual("vendor", child["path"])
        blob = git(["show", f"{child['tree']}:lib.py"], self.repo / "vendor")
        self.assertIn("edited in the submodule", blob)

    def test_a_declared_path_without_an_in_tree_gitlink_is_not_followed(self):
        sibling = self.tmp / "sibling"
        sibling.mkdir()
        git(["init", "-q"], sibling)
        git(["config", "user.email", "d@e.com"], sibling)
        git(["config", "user.name", "D"], sibling)
        (sibling / "private.py").write_text("outside = True\n")
        git(["add", "-A"], sibling)
        git(["commit", "-qm", "sibling"], sibling)
        (self.repo / ".gitmodules").write_text(
            '[submodule "outside"]\n\tpath = ../sibling\n\turl = ../sibling\n')

        snap = self._capture()

        self.assertEqual([], snap.submodules)
        self.assertTrue(any("submodule path" in error
                            for error in snap.errors))
        self.assertEqual("", git(["for-each-ref", "--format=%(refname)",
                                  "refs/repohone/"], sibling))

    def test_root_and_submodule_settle_as_one_hierarchy(self):
        source = self.tmp / "source"
        source.mkdir()
        git(["init", "-q", "-b", "main"], source)
        git(["config", "user.email", "d@e.com"], source)
        git(["config", "user.name", "D"], source)
        (source / "value.txt").write_text("A\n")
        git(["add", "value.txt"], source)
        git(["commit", "-qm", "A"], source)
        child_a = git(["rev-parse", "HEAD"], source)
        (source / "value.txt").write_text("B\n")
        git(["commit", "-qam", "B"], source)
        child_b = git(["rev-parse", "HEAD"], source)

        original_branch = git(["branch", "--show-current"], self.repo)
        git(["-c", "protocol.file.allow=always", "submodule", "add", "-q",
             str(source), "vendor"], self.repo)
        git(["checkout", "-q", child_a], self.repo / "vendor")
        git(["add", ".gitmodules", "vendor"], self.repo)
        git(["commit", "-qm", "root points to A"], self.repo)
        git(["checkout", "-qb", "other"], self.repo)
        git(["checkout", "-q", child_b], self.repo / "vendor")
        git(["add", "vendor"], self.repo)
        git(["commit", "-qm", "root points to B"], self.repo)
        git(["checkout", "-q", original_branch], self.repo)
        git(["-c", "protocol.file.allow=always", "submodule", "update", "-q"],
            self.repo)

        real = snapshot.stable_tree
        switched = False

        def switch_root_during_child(repo, *args, **kwargs):
            nonlocal switched
            result = real(repo, *args, **kwargs)
            if Path(repo).resolve() == (self.repo / "vendor").resolve() and not switched:
                switched = True
                git(["checkout", "-q", "other"], self.repo)
                git(["-c", "protocol.file.allow=always", "submodule", "update", "-q"],
                    self.repo)
            return result

        snapshot.stable_tree = switch_root_during_child
        try:
            snap = self._capture()
        finally:
            snapshot.stable_tree = real

        root_gitlink = git(["ls-tree", snap.tree, "vendor"], self.repo).split()[2]
        self.assertEqual(child_b, root_gitlink)
        self.assertEqual(root_gitlink, snap.submodules[0]["head"])
        self.assertEqual([], snap.errors)


class ScratchIndexIsolation(Fixture):
    """Regressions: the scratch index was keyed by repo path and placed wherever
    REPOHONE_TMP pointed, which lost snapshots three different ways."""

    def setUp(self):
        super().setUp()
        self.checkout = identity.checkout_id(self.repo)
        state.initialize(self.checkout)

    def test_each_session_gets_its_own_index(self):
        """Sharing one GIT_INDEX_FILE makes concurrent sessions collide on index.lock."""
        a = snapshot._scratch_index(self.checkout, "rh_a", self.repo)
        b = snapshot._scratch_index(self.checkout, "rh_b", self.repo)
        self.assertNotEqual(a, b)

    def test_concurrent_sessions_all_keep_their_snapshots(self):
        results, lock = [], threading.Lock()

        def worker(name):
            try:
                snap = snapshot.capture(self.repo, self.checkout, f"rh_{name}", "prompt",
                                        1, "t1", record.now())
                with lock:
                    results.append(snap.tree)
            except Exception as exc:
                with lock:
                    results.append(f"FAILED: {type(exc).__name__}: {exc}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in "abcdef"]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual([], [r for r in results if str(r).startswith("FAILED")])
        self.assertEqual(6, len(results))

    def test_a_deep_repository_path_does_not_overflow_the_filename(self):
        index = snapshot._scratch_index(self.checkout, "rh_a", Path("/" + "x" * 300))
        self.assertLess(len(index.name), 255)

    def test_scratch_never_lands_inside_the_repository(self):
        """ARCH §16.1 — RepoHone scratch is excluded from the captured file set.
        An index file inside the repo means no tree can ever settle."""
        os.environ["REPOHONE_TMP"] = str(self.repo / ".rh-scratch")
        try:
            index = snapshot._scratch_index(self.checkout, "rh_a", self.repo)
            self.assertFalse(str(index.resolve()).startswith(str(self.repo.resolve())))
            snap = snapshot.capture(self.repo, self.checkout, "rh_a", "prompt", 1, "t1",
                                    record.now())
            files = git(["ls-tree", "-r", "--name-only", snap.tree], self.repo).split()
            self.assertFalse([f for f in files if ".rh-scratch" in f])
        finally:
            os.environ.pop("REPOHONE_TMP", None)


class SnapshotWorkHappensOutsideTheRecordLock(Fixture):
    """ARCH §9.1, §21 — git must not run while the record file is locked."""

    def test_the_snapshot_is_taken_before_the_record_is_opened(self):
        from repohone import capture as capture_mod
        order = []
        real_plan, real_open = capture_mod._plan, record.open_record

        def traced_plan(*a, **kw):
            order.append("snapshot")
            return real_plan(*a, **kw)

        def traced_open(*a, **kw):
            order.append("lock")
            return real_open(*a, **kw)

        self.init_repo()
        capture_mod._plan = traced_plan
        record.open_record = traced_open
        try:
            capture_mod.handle(claude.to_event(
                {"hook_event_name": "UserPromptSubmit", "session_id": "s-lock",
                 "cwd": str(self.repo), "prompt": "x", "prompt_id": "t1"}))
        finally:
            capture_mod._plan = real_plan
            record.open_record = real_open
        self.assertEqual(["snapshot", "lock"], order)


class ModelSwitch(Fixture):
    """ARCH §32 — the model path is SessionStart.model, then PostModelSwitch."""

    def test_a_switch_is_recorded_against_the_next_turn(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="one", prompt_id="t1")
        self.fire("Stop", prompt_id="t1")
        self.fire("PostModelSwitch", to_model="claude-opus-5")
        rec = self.the_record()
        switches = [m for m in rec["models"] if m["source"] == "model_switch"]
        self.assertEqual(1, len(switches), "PostModelSwitch was dropped")
        self.assertEqual("claude-opus-5", switches[0]["model"])
        self.assertEqual(2, switches[0]["from_turn"])
        self.assert_conformant(rec)


class SubmoduleFailuresAreRecorded(Fixture):
    """ARCH §16.1 — absent evidence is honest; silently absent evidence is not."""

    def test_a_submodule_that_cannot_be_snapshotted_is_reported(self):
        checkout = identity.checkout_id(self.repo)
        state.initialize(checkout)
        sub = self.tmp / "sub"
        sub.mkdir()
        for args in (["init", "-q"], ["config", "user.email", "d@e.com"],
                     ["config", "user.name", "D"]):
            git(args, sub)
        (sub / "lib.py").write_text("x = 1\n")
        git(["add", "-A"], sub)
        git(["commit", "-qm", "sub"], sub)
        git(["-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "vendor"],
            self.repo)
        git(["commit", "-qm", "add submodule"], self.repo)

        refs = self.repo / ".git" / "modules" / "vendor" / "refs"
        subprocess.run(["chmod", "-R", "a-w", str(refs)], check=False)
        try:
            snap = snapshot.capture(self.repo, checkout, "rh_a", "prompt", 1, "t1",
                                    record.now())
            self.assertTrue(snap.errors, "a failed submodule vanished from the record")
            self.assertIn("submodule vendor", snap.errors[0])
        finally:
            subprocess.run(["chmod", "-R", "u+w", str(refs)], check=False)


LOCK_WORKER = """
import os, sys
sys.path.insert(0, %r)
os.environ["REPOHONE_DATA_DIR"] = sys.argv[2]
from repohone import record
record.fcntl = None                      # the path Windows takes
if sys.argv[1] == "nolock":
    import contextlib
    record._locked = lambda path, suffix=".lock": contextlib.nullcontext()
from repohone import capture
from repohone.adapters import claude as adapter
for i in range(6):
    capture.handle(adapter.to_event({
        "hook_event_name": "UserPromptSubmit", "session_id": "lock-test",
        "prompt_id": "t%%d" %% i, "prompt": "p", "cwd": sys.argv[3]}))
""" % str(CORE)


class LockingWithoutFlock(Fixture):
    """ARCH §18 on hosts with no `fcntl`. Writes used to be simply unlocked
    there, so the ordering guarantee held on one platform and not the other."""

    def run_workers(self, mode, count=5):
        script = self.tmp / f"worker-{mode}.py"
        script.write_text(LOCK_WORKER)
        procs = [subprocess.Popen(
            [sys.executable, str(script), mode, str(self.data), str(self.repo)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(count)]
        for proc in procs:
            proc.wait()
        return self.the_record("lock-test")

    def events_and_faults(self, rec):
        events = sum(len(t["prompt_events"]) for t in rec["turns"])
        return events, check_record(rec)

    def test_the_fallback_lock_loses_no_events(self):
        self.init_repo()
        events, faults = self.events_and_faults(self.run_workers("lock"))
        self.assertEqual(30, events, "events were lost without flock")
        self.assertEqual([], faults)

    def test_every_event_is_either_recorded_or_refused(self):
        """The invariant that matters under contention: an event may be turned
        away, but it may never simply vanish. The old fallback proceeded
        unlocked after its deadline and lost updates with no trace."""
        self.init_repo()
        rec = self.run_workers("lock")
        checkout = identity.peek_checkout_id(self.repo)
        recorded = sum(len(t["prompt_events"]) for t in rec["turns"])
        refused = len([r for r in state.refusals(checkout)
                       if "could not take" in r["reason"]])
        self.assertEqual(30, recorded + refused,
                         f"{30 - recorded - refused} event(s) disappeared")
        self.assertEqual([], check_record(rec))

    def test_without_any_lock_the_same_load_corrupts_the_record(self):
        """Proves the lock above is doing the work, not the filesystem."""
        self.init_repo()
        events, faults = self.events_and_faults(self.run_workers("nolock"))
        self.assertTrue(events < 30 or faults,
                        "unlocked concurrent writes produced a clean record, so "
                        "this fixture cannot demonstrate the lock")

    def test_an_abandoned_lock_is_broken_rather_than_waited_on(self):
        lock = self.tmp / "x.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(f"999999 {time.time() - record.LOCK_WAIT_S - 5}")
        self.assertTrue(record._abandoned(lock))
        with record._exclusive_file(lock):
            self.assertTrue(lock.is_file())
        self.assertFalse(lock.is_file())

    def test_a_live_holders_lock_is_not_broken(self):
        lock = self.tmp / "y.lock"
        lock.write_text(f"{os.getpid()} {time.time() - record.LOCK_WAIT_S - 5}")
        self.assertFalse(record._abandoned(lock))

    def test_doctor_names_the_strategy_in_use(self):
        self.init_repo()
        found = [c for c in doctor.run(self.repo) if c[0] == "file locking"]
        self.assertEqual(1, len(found))
        self.assertIn("flock" if record.fcntl is not None else "lock files",
                      found[0][2])


class TheFallbackLockFailsClosed(Fixture):
    """Regression: after its deadline the fallback yielded anyway, entering the
    protected section without owning it — two writers inside at once, which is
    exactly what the lock exists to prevent."""

    def setUp(self):
        super().setUp()
        self._fcntl, record.fcntl = record.fcntl, None
        self._wait, record.LOCK_WAIT_S = record.LOCK_WAIT_S, 0.4
        self.addCleanup(lambda: setattr(record, "fcntl", self._fcntl))
        self.addCleanup(lambda: setattr(record, "LOCK_WAIT_S", self._wait))

    def test_a_second_holder_is_refused_rather_than_admitted(self):
        lock = self.tmp / "contended"
        entered = []

        def hold():
            with record._locked(lock):
                entered.append("first")
                time.sleep(1.2)

        keeper = threading.Thread(target=hold)
        keeper.start()
        time.sleep(0.2)
        try:
            with self.assertRaises(record.LockUnavailable), record._locked(lock):
                entered.append("second entered without the lock")
        finally:
            keeper.join()
        self.assertEqual(["first"], entered)

    def test_the_lock_is_available_again_afterwards(self):
        lock = self.tmp / "sequential"
        with record._locked(lock):
            pass
        with record._locked(lock):
            pass

    def test_two_stale_lock_recoverers_never_overlap(self):
        lock = self.tmp / "stale.lock"
        lock.write_text(f"999999 {time.time() - record.LOCK_WAIT_S - 5}")
        checked = threading.Event()
        resume = threading.Event()
        original = record._abandoned
        active = 0
        maximum = 0
        errors = []

        def paused_check(path):
            stale = original(path)
            if stale and threading.current_thread().name == "first":
                checked.set()
                resume.wait(3)
            return stale

        def contender():
            nonlocal active, maximum
            try:
                with record._exclusive_file(lock):
                    active += 1
                    maximum = max(maximum, active)
                    time.sleep(0.15)
                    active -= 1
            except Exception as exc:
                errors.append(exc)

        record._abandoned = paused_check
        try:
            first = threading.Thread(target=contender, name="first")
            second = threading.Thread(target=contender, name="second")
            first.start()
            self.assertTrue(checked.wait(3))
            second.start()
            time.sleep(0.05)
            resume.set()
            first.join(3)
            second.join(3)
            self.assertFalse(first.is_alive() or second.is_alive())
        finally:
            resume.set()
            record._abandoned = original
        self.assertEqual([], errors)
        self.assertEqual(1, maximum)

    def test_capture_records_the_event_it_could_not_take_the_lock_for(self):
        self.init_repo()
        checkout = identity.checkout_id(self.repo)
        held = record.record_path(checkout, identity.session_id("claude-code", "s-1"))
        lock = held.with_suffix(".ordering")
        entered = []

        def hold():
            with record._locked(lock, suffix=".ordering.lock"):
                entered.append("held")
                time.sleep(1.2)

        keeper = threading.Thread(target=hold)
        keeper.start()
        time.sleep(0.2)
        try:
            from repohone import capture as capture_mod
            from repohone.adapters import claude as adapter
            result = capture_mod.handle(adapter.to_event(
                {"hook_event_name": "UserPromptSubmit", "session_id": "s-1",
                 "prompt_id": "t1", "prompt": "work", "cwd": str(self.repo)}))
        finally:
            keeper.join()
        self.assertFalse(result["captured"])
        self.assertIn("could not take", result["reason"])
        self.assertTrue(state.refusals(checkout),
                        "a lost event left no trace for doctor")


class EveryCaptureWaitUsesTheHostBudget(Fixture):
    def test_native_flock_stops_before_the_hook_deadline(self):
        if record.fcntl is None:
            self.skipTest("fcntl is unavailable")
        lock = self.tmp / "native-deadline"
        entered = threading.Event()
        release = threading.Event()

        def hold():
            with record._locked(lock):
                entered.set()
                release.wait(2)

        keeper = threading.Thread(target=hold)
        keeper.start()
        self.assertTrue(entered.wait(1))
        started = time.monotonic()
        try:
            with self.assertRaises(record.LockUnavailable):
                with deadline.budget(0.15, reserve=0):
                    with record._locked(lock):
                        pass
        finally:
            release.set()
            keeper.join(2)
        self.assertLess(time.monotonic() - started, 0.6)

    def test_sqlite_contention_stops_before_the_hook_deadline(self):
        checkout = "deadline-checkout"
        state.initialize(checkout)
        blocker = sqlite3.connect(str(paths.state_db(checkout)), isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")
        started = time.monotonic()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                with deadline.budget(0.15, reserve=0):
                    state.ensure_session(
                        checkout, "rh_deadbeef", "fixture", "one", record.now())
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        self.assertLess(time.monotonic() - started, 0.6)


class DurableInstallation(Fixture):
    """Regression: hooks pointed at an absolute path inside the source checkout,
    so Core stopped working the moment that checkout moved."""

    def test_the_cli_is_runnable_as_a_module(self):
        for entry in (["-m", "repohone"], ["-m", "repohone.cli"],
                      ["-m", "repohone.hook"]):
            proc = subprocess.run([sys.executable] + entry + ["--help"],
                                  cwd=str(CORE), capture_output=True, text=True,
                                  env=dict(os.environ, PYTHONPATH=str(CORE)))
            self.assertEqual(0, proc.returncode, f"{entry}: {proc.stderr[:200]}")
            self.assertIn("repohone", proc.stdout)

    def test_the_project_declares_console_scripts(self):
        text = (ROOT / "pyproject.toml").read_text()
        self.assertIn('repohone = "repohone.cli:main"', text)
        self.assertIn('repohone-hook = "repohone.hook:run"', text)

    def test_install_uses_the_installed_package_when_importable(self):
        with contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            cli.cmd_install(argparse.Namespace(path=None))
        hooks = json.loads((claude.plugin_dir() / "hooks" / "hooks.json").read_text())["hooks"]
        args = hooks["UserPromptSubmit"][0]["hooks"][0]["args"]
        if args[3] == "":
            self.assertEqual(claude.hook_args(None), args)
            self.assertEqual("", err.getvalue())
        else:
            self.assertEqual(claude.hook_args(str(claude.plugin_dir() / "core")), args)
            self.assertIn("runs a copy", out.getvalue())

    def test_deinit_recognises_both_entry_forms(self):
        for args in (["-m", "repohone.hook", "hook"], ["/src/repohone_hook.py", "hook"]):
            self.assertTrue(claude.owns_hook(
                {"type": "command", "command": "python3", "args": args}), args)
        self.assertFalse(claude.owns_hook(
            {"type": "command", "command": "make", "args": ["lint"]}))

    def test_a_source_install_runs_a_pinned_copy_not_the_working_tree(self):
        """Regression: hooks imported the source tree, so every edit to it changed
        what capture ran between turns, and a broken edit stopped capture (§4)."""
        self.install()
        copied = claude.plugin_dir() / "core" / "repohone"
        self.assertEqual((CORE / "repohone" / "capture.py").read_text(),
                         (copied / "capture.py").read_text())
        self.assertFalse(list(copied.rglob("__pycache__")))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, cli.cmd_uninstall(argparse.Namespace(
                path=None, purge=False, dry_run=False, yes=True)))
        self.assertFalse(copied.exists())


class ABrokenInstallStaysSilent(Fixture):
    """Claude Code treats a hook's exit 2 as a blocking decision: the prompt is
    erased, the tool call denied, the stop refused.

    Regression: install pointed hooks at `<source>/repohone_hook.py`, and Python
    exits 2 when it cannot open a script. Moving the source tree, or committing
    the shared settings file for a teammate, blocked every prompt."""

    PAYLOAD = json.dumps({"hook_event_name": "UserPromptSubmit", "session_id": "s1",
                          "prompt_id": "t1", "prompt": "hello", "cwd": "."})

    def run_hook(self, command, args, cwd=None):
        env = {k: v for k, v in self.env.items() if k != "PYTHONPATH"}
        return subprocess.run([command, *args], input=self.PAYLOAD, text=True,
                              capture_output=True, env=env,
                              cwd=str(cwd or tempfile.gettempdir()))

    def installed_entry(self):
        hooks = json.loads(self.install().read_text())["hooks"]
        return hooks["UserPromptSubmit"][0]["hooks"][0]

    def test_a_script_path_that_moved_exited_2(self):
        proc = self.run_hook(sys.executable, ["/moved/away/repohone_hook.py", "hook"])
        self.assertEqual(2, proc.returncode, "the hazard this class guards against")

    def test_a_core_that_moved_or_was_uninstalled_exits_0_and_says_nothing(self):
        for source in ("/moved/away/core", ""):
            with self.subTest(source=source or "installed package"):
                proc = self.run_hook(sys.executable, claude.hook_args(source))
                self.assertEqual((0, "", ""), (proc.returncode, proc.stdout, proc.stderr))

    def test_the_installed_hook_still_captures(self):
        self.init_repo()
        entry = self.installed_entry()
        proc = self.run_hook(entry["command"], entry["args"], cwd=self.repo)
        self.assertEqual(0, proc.returncode, proc.stderr)
        checkout = identity.peek_checkout_id(self.repo)
        self.assertEqual(1, len(record.list_sessions(checkout)))

    def test_install_writes_nothing_into_any_repository(self):
        self.init_repo()
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "profile"], self.repo)
        self.installed_entry()
        self.assertEqual("", git(["status", "--porcelain", "--ignored"], self.repo))
        self.assertEqual({"enabledPlugins": {claude.PLUGIN_ID: False}},
                         json.loads((claude.config_dir() / "settings.json").read_text()))

    def test_init_removes_what_an_older_install_left(self):
        shared = self.repo / ".claude" / "settings.json"
        shared.parent.mkdir(exist_ok=True)
        shared.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [
            {"type": "command", "command": "make lint"},
            {"type": "command", "command": sys.executable,
             "args": [str(HOOK), "hook"]}]}]}}))
        (self.repo / ".mcp.json").write_text(json.dumps({"mcpServers": {
            "repohone": {"command": "python3"}, "theirs": {"command": "x"}}}))
        skill = self.repo / ".claude" / "skills" / "repohone-rule"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("old")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cli.cmd_init(argparse.Namespace(path=str(self.repo), force=False))
        left = json.loads(shared.read_text())["hooks"]["Stop"][0]["hooks"]
        self.assertEqual([{"type": "command", "command": "make lint"}], left)
        self.assertEqual({"mcpServers": {"theirs": {"command": "x"}}},
                         json.loads((self.repo / ".mcp.json").read_text()))
        self.assertFalse(skill.exists())
        self.assertIn("left by an older per-repository install", out.getvalue())

    def test_doctor_names_a_hook_that_cannot_import_core(self):
        self.init_repo()
        self.install(source="/moved/away/core")
        status, detail = {name: (status, detail)
                          for name, status, detail in doctor.run(self.repo)}["plugin"]
        self.assertEqual("fail", status)
        self.assertIn("cannot import RepoHone from /moved/away/core", detail)

    def test_doctor_warns_about_hooks_an_older_install_shared(self):
        self.init_repo()
        shared = self.repo / ".claude" / "settings.json"
        shared.parent.mkdir(exist_ok=True)
        shared.write_text(json.dumps(claude.settings(sys.executable, str(CORE))))
        rows = {name: (status, detail) for name, status, detail in doctor.run(self.repo)}
        self.assertEqual("warn", rows["older install"][0])

    def test_init_keeps_its_setting_out_of_everything_committed(self):
        """The shared settings file is committed; turning RepoHone on is one
        developer's choice. No global ignore rule may hide a regression."""
        quiet = {"GIT_CONFIG_GLOBAL": os.devnull, "XDG_CONFIG_HOME": str(self.tmp / "xdg")}
        with mock.patch.dict(os.environ, quiet), contextlib.redirect_stdout(io.StringIO()):
            cli.cmd_init(argparse.Namespace(path=str(self.repo), force=False))
            ignored = git_ignored(self.repo, ".claude/settings.local.json")
        self.assertFalse((self.repo / ".claude" / "settings.json").exists())
        local = json.loads((self.repo / ".claude" / "settings.local.json").read_text())
        self.assertEqual({claude.PLUGIN_ID: True}, local["enabledPlugins"])
        self.assertEqual(0, ignored)


class CaptureNeedsThisDevelopersConsent(Fixture):
    """The profile is committed, so it is the project's decision. Capturing one
    developer's sessions takes that developer's own acceptance (§5).

    Regression: capture checked only the profile, so once hooks run everywhere
    a teammate's commit would have started capturing everyone."""

    def write_profile(self, text=None):
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(text or profile.template())

    def announced(self):
        out = self.fire("SessionStart", source="startup").stdout
        return json.loads(out)["hookSpecificOutput"]["additionalContext"] if out else None

    def test_a_committed_profile_alone_captures_nothing(self):
        self.write_profile()
        self.fire("UserPromptSubmit", prompt="client work", prompt_id="t1")
        self.assertIsNone(identity.peek_checkout_id(self.repo))
        self.assertFalse(self.data.exists() and any(self.data.rglob("rh_*.json")))

    def test_acceptance_binds_the_configuration_not_its_comments(self):
        self.write_profile()
        profile.accept(self.repo, profile.load(self.repo))
        self.write_profile(profile.template() + "# reviewed in PR 12\n")
        self.fire("UserPromptSubmit", prompt="one", prompt_id="t1")
        self.assertEqual(1, len(record.list_sessions(identity.peek_checkout_id(self.repo))))
        self.write_profile(profile.template().replace("content_policy: redacted",
                                                      "content_policy: hash_only"))
        self.fire("UserPromptSubmit", session="s-2", prompt="two", prompt_id="t2")
        self.assertEqual(1, len(record.list_sessions(identity.peek_checkout_id(self.repo))),
                         "a changed configuration was captured under the old consent")

    def test_the_announcer_asks_until_accepted_and_again_after_a_change(self):
        self.write_profile()
        self.assertIn("not capturing this user's sessions", self.announced())
        profile.accept(self.repo, profile.load(self.repo))
        self.assertIn("RepoHone is capturing this session", self.announced())
        self.write_profile(profile.template().replace("tool_capture: true", "tool_capture: false"))
        self.assertIn("changed since the user accepted it", self.announced())

    def test_init_accepts_a_committed_profile_without_rewriting_it(self):
        text = profile.template().replace("tool_capture: true", "tool_capture: false")
        self.write_profile(text)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(0, cli.cmd_init(argparse.Namespace(path=str(self.repo), force=False)))
        self.assertEqual(text, (self.repo / ".repohone" / "profile.yaml").read_text())
        self.assertIs(True, profile.accepted(self.repo, profile.load(self.repo)))
        self.assertEqual((True, "local"), claude.enabled_here(self.repo))
        self.assertIn("tool capture off", out.getvalue())

    def test_deinit_turns_it_off_for_this_developer_and_keeps_the_profile(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cli.cmd_init(argparse.Namespace(path=str(self.repo), force=False))
            cli.cmd_deinit(argparse.Namespace(path=str(self.repo), purge=False,
                                              dry_run=False, yes=True))
        prof = profile.load(self.repo)
        self.assertEqual(profile.ACTIVE, prof.state)
        self.assertIsNone(profile.accepted(self.repo, prof))
        self.assertEqual((False, "local"), claude.enabled_here(self.repo))

    def test_the_clones_worktrees_share_one_acceptance(self):
        self.init_repo()
        git(["add", "-A"], self.repo)
        git(["commit", "-qm", "profile"], self.repo)
        tree = self.tmp / "tree"
        git(["worktree", "add", "-q", str(tree)], self.repo)
        self.assertIs(True, profile.accepted(tree, profile.load(tree)))


class InstallOncePerMachine(Fixture):
    """One plugin, off in every repository until `init` turns it on in one; and
    an uninstall that visits no repository (§4, §30)."""

    def uninstall(self, *flags):
        args = argparse.Namespace(path=None, purge="--purge" in flags,
                                  dry_run="--dry-run" in flags, yes="--yes" in flags)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return cli.cmd_uninstall(args)

    def test_install_is_off_everywhere_until_init(self):
        self.install()
        self.assertEqual((False, "user"), claude.enabled_here(self.repo))
        with contextlib.redirect_stdout(io.StringIO()):
            cli.cmd_init(argparse.Namespace(path=str(self.repo), force=False))
        self.assertEqual((True, "local"), claude.enabled_here(self.repo))
        self.assertEqual(0, git_ignored(self.repo, claude.SETTINGS_FILE))

    def test_uninstall_removes_the_plugin_and_only_its_own_setting(self):
        settings = claude.config_dir() / "settings.json"
        settings.parent.mkdir(parents=True)
        mine = {"theme": "dark", "enabledPlugins": {"other@market": True}}
        settings.write_text(json.dumps(mine))
        self.install()
        self.assertEqual(3, self.uninstall())
        self.assertTrue(claude.plugin_dir().exists(), "uninstall acted without --yes")
        self.assertEqual(0, self.uninstall("--yes"))
        self.assertFalse(claude.plugin_dir().exists())
        self.assertEqual(mine, json.loads(settings.read_text()))

    def test_a_directory_that_is_not_ours_is_left_alone(self):
        manifest = claude.plugin_dir() / ".claude-plugin" / "plugin.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"name": "someone-else"}))
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(1, cli.cmd_install(argparse.Namespace(path=None)))
        self.assertEqual(1, self.uninstall("--yes"))
        self.assertEqual({"name": "someone-else"}, json.loads(manifest.read_text()))

    def test_the_skill_runs_the_core_the_hooks_run(self):
        self.install()
        launcher = claude.plugin_dir() / "bin" / "repohone"
        proc = subprocess.run([str(launcher), "--version"], capture_output=True, text=True,
                              env={k: v for k, v in self.env.items() if k != "PYTHONPATH"})
        self.assertEqual((0, f"repohone {CORE_VERSION}"), (proc.returncode, proc.stdout.strip()))
        skill = (claude.plugin_dir() / "skills" / claude.SKILL / "SKILL.md").read_text()
        self.assertIn(f"{launcher} rule \"", skill)
        self.assertNotIn("{repohone}", skill)


class ACopyIsNotTheSameCheckout(Fixture):
    """The checkout id lives in `.git/repohone/`, so `cp -R` carries it along.

    Regression: a copy wrote into the original's telemetry store, and
    `deinit --purge` in the copy deleted the original's session records."""

    def prompt(self, repo, session):
        payload = {"hook_event_name": "UserPromptSubmit", "session_id": session,
                   "prompt_id": "t1", "prompt": "work", "cwd": str(repo)}
        return subprocess.run([sys.executable, str(HOOK), "hook"],
                              input=json.dumps(payload), cwd=str(repo),
                              capture_output=True, text=True, env=self.env)

    def copied(self):
        self.init_repo()
        self.prompt(self.repo, "original-session")
        copy = self.tmp / "copy"
        shutil.copytree(self.repo, copy, symlinks=True)
        return copy

    def test_a_copy_gets_its_own_identity_and_store(self):
        copy = self.copied()
        original = identity.peek_checkout_id(self.repo)
        self.assertIsNone(identity.peek_checkout_id(copy),
                          "an uncaptured copy claimed the original's id")
        self.assertEqual(str((self.repo / ".git").resolve()), identity.copy_of(copy))
        self.prompt(copy, "copy-session")
        mine = identity.peek_checkout_id(copy)
        self.assertNotEqual(original, mine)
        self.assertEqual(original, identity.peek_checkout_id(self.repo))
        self.assertEqual(1, len(record.list_sessions(original)))
        self.assertEqual(1, len(record.list_sessions(mine)))

    def test_a_moved_checkout_keeps_its_identity(self):
        self.init_repo()
        self.prompt(self.repo, "before-move")
        before = identity.peek_checkout_id(self.repo)
        moved = self.tmp / "moved"
        self.repo.rename(moved)
        self.prompt(moved, "after-move")
        self.assertEqual(before, identity.peek_checkout_id(moved))
        self.assertEqual(2, len(record.list_sessions(before)))

    def test_parallel_first_events_in_a_copy_agree_on_one_identity(self):
        copy = self.copied()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            list(pool.map(lambda n: self.prompt(copy, f"copy-{n}"), range(6)))
        mine = identity.peek_checkout_id(copy)
        self.assertEqual(6, len(record.list_sessions(mine)))
        self.assertEqual(1, len(record.list_sessions(identity.peek_checkout_id(self.repo))))

    def test_purging_an_uncaptured_copy_leaves_the_original_alone(self):
        copy = self.copied()
        original = identity.peek_checkout_id(self.repo)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            cli.cmd_deinit(argparse.Namespace(path=str(copy), purge=True,
                                              dry_run=False, yes=True))
        self.assertEqual(1, len(record.list_sessions(original)))

    def test_a_purge_that_would_delete_another_trees_sessions_is_refused(self):
        """A store shared before copies were told apart holds both trees' sessions."""
        copy = self.copied()
        original = identity.peek_checkout_id(self.repo)
        session = record.list_sessions(original)[0]
        rec = record.load(original, session)
        rec["repository"]["root"] = str(copy)
        record._atomic_write(record.record_path(original, session), rec)
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = cli.cmd_deinit(argparse.Namespace(path=str(self.repo), purge=True,
                                                     dry_run=False, yes=True))
        self.assertEqual(1, code)
        self.assertIn("refusing to purge", err.getvalue())
        self.assertEqual([session], record.list_sessions(original))
        self.assertTrue((self.repo / ".repohone" / "profile.yaml").exists(),
                        "a refused deinit still changed something")


class ADamagedRecordIsNeverRestarted(Fixture):
    """Regression: `open_record` asked `path.exists()`, which is False for a
    dangling link, so capture began a fresh record mid-session — the fourth
    turn written as turn 1, with no refusal. That is wrong evidence; a refused
    event is only missing evidence, and doctor can say so."""

    def _three_turns(self):
        self.init_repo()
        for turn in (1, 2, 3):
            self.fire("UserPromptSubmit", prompt=f"turn {turn}", prompt_id=f"t{turn}")
            self.fire("Stop", prompt_id=f"t{turn}", last_assistant_message="done")
        checkout = identity.peek_checkout_id(self.repo)
        return checkout, record.record_path(
            checkout, identity.session_id("claude-code", "s-1"))

    def test_every_damaged_shape_is_refused_not_restarted(self):
        for shape in ("dangling link", "directory", "unreadable"):
            with self.subTest(shape=shape):
                self.tearDown()
                self.setUp()
                checkout, target = self._three_turns()
                target.unlink()
                if shape == "dangling link":
                    target.symlink_to(target.with_name("gone.json"))
                elif shape == "directory":
                    target.mkdir()
                else:
                    target.write_text("{}")
                    os.chmod(target, 0)
                try:
                    self.fire("UserPromptSubmit", prompt="turn 4", prompt_id="t4")
                    self.fire("Stop", prompt_id="t4", last_assistant_message="done")
                    still_damaged = (target.is_symlink() or target.is_dir()
                                     or not os.access(target, os.R_OK))
                finally:
                    if not target.is_symlink() and target.is_file():
                        os.chmod(target, 0o644)
                self.assertTrue(still_damaged, "capture wrote over a damaged record")
                self.assertEqual(2, len(state.refusals(checkout)))

    def test_a_rejected_event_does_not_restart_a_damaged_record_either(self):
        """`_reject` writes through `open_record` without reading the record
        first, so on this path the write boundary's own check is all there is."""
        checkout, target = self._three_turns()
        target.unlink()
        target.symlink_to(target.with_name("gone.json"))
        self.fire("UserPromptSubmit", prompt="no turn id")      # takes the reject path
        self.assertTrue(target.is_symlink(), "the reject path wrote a fresh record")
        self.assertTrue(state.refusals(checkout))

    def test_reading_a_damaged_record_is_not_reading_a_missing_one(self):
        """Every read-only caller — diagnosis, rule marking — relies on this."""
        checkout, target = self._three_turns()
        target.unlink()
        target.symlink_to(target.with_name("gone.json"))
        session = identity.session_id("claude-code", "s-1")
        with self.assertRaises(record.MalformedArtifact):
            record.load(checkout, session)

    def test_an_intact_record_still_gains_its_turn(self):
        checkout, target = self._three_turns()
        self.fire("UserPromptSubmit", prompt="turn 4", prompt_id="t4")
        rec = record.load(checkout, identity.session_id("claude-code", "s-1"))
        self.assertEqual([1, 2, 3, 4], [t["index"] for t in rec["turns"]])


class DoctorSeesWhatIsBroken(Fixture):
    """Regression: an active profile with no hooks, a refused event and a deleted
    submodule ref all read as healthy — the states a developer most needs told."""

    def checks(self):
        return {c[0]: (c[1], c[2]) for c in doctor.run(self.repo)}

    def test_an_active_profile_without_the_plugin_is_a_failure(self):
        self.init_repo()
        status, detail = self.checks()["plugin"]
        self.assertEqual("fail", status)
        self.assertIn("repohone install", detail)

    def test_an_unsupported_record_version_is_a_failure(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        target = next(paths.records_dir(checkout).glob("*.json"))
        value = json.loads(target.read_text())
        value["schema_version"] = 99
        target.write_text(json.dumps(value))
        status, detail = self.checks()["records"]
        self.assertEqual(doctor.FAIL, status)
        self.assertIn("unsupported", detail)

    def test_a_malformed_record_is_a_failure_not_a_healthy_count(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        target = next(paths.records_dir(checkout).glob("*.json"))
        target.write_text("{malformed", encoding="utf-8")

        status, detail = self.checks()["records"]

        self.assertEqual(doctor.FAIL, status)
        self.assertIn("unreadable JSON", detail)

    def test_a_record_with_the_wrong_artifact_identity_is_a_failure(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        target = next(paths.records_dir(checkout).glob("*.json"))
        value = json.loads(target.read_text())
        value["schema"] = "repohone.proposal/v1"
        target.write_text(json.dumps(value))

        status, detail = self.checks()["records"]

        self.assertEqual(doctor.FAIL, status)
        self.assertIn("expected", detail)

    def test_a_header_only_record_is_rejected_by_runtime_and_doctor(self):
        self.init_repo()
        checkout = identity.checkout_id(self.repo)
        session = identity.session_id("claude-code", "header-only")
        target = record.record_path(checkout, session)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({
            "schema": "repohone.session/v1", "schema_version": 1,
            "contract_version": CONTRACT_VERSION}))

        with self.assertRaises(record.MalformedArtifact):
            record.load(checkout, session)
        status, detail = self.checks()["records"]
        self.assertEqual(doctor.FAIL, status)
        self.assertIn("missing required property", detail)

    def test_a_record_copied_to_another_session_is_never_appended(self):
        self.init_repo()
        for session in ("host-A", "host-B"):
            self.fire("UserPromptSubmit", session=session, prompt=session, prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        a = record.record_path(checkout, identity.session_id("claude-code", "host-A"))
        b = record.record_path(checkout, identity.session_id("claude-code", "host-B"))
        a.write_bytes(b.read_bytes())
        before = a.read_bytes()

        self.fire("UserPromptSubmit", session="host-A", prompt="must be refused",
                  prompt_id="t2")

        self.assertEqual(before, a.read_bytes())
        status, detail = self.checks()["records"]
        self.assertEqual(doctor.FAIL, status)
        self.assertIn("claims session_id", detail)

    def test_an_early_rejection_preserves_malformed_canonical_evidence(self):
        self.init_repo()
        self.fire("UserPromptSubmit", session="damaged", prompt="one", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        target = record.record_path(
            checkout, identity.session_id("claude-code", "damaged"))
        target.write_bytes(b"{damaged evidence")
        before = target.read_bytes()

        self.fire("UserPromptSubmit", session="damaged", prompt="missing turn id")

        self.assertEqual(before, target.read_bytes())
        self.assertEqual([], list(target.parent.glob("*.corrupt-*")))
        self.assertEqual(doctor.FAIL, self.checks()["records"][0])

    def test_incompatible_state_metadata_stops_capture_and_fails_doctor(self):
        self.init_repo()
        self.fire("SessionStart", session="state-test", source="startup")
        checkout = identity.peek_checkout_id(self.repo)
        session = identity.session_id("claude-code", "state-test")
        target = record.record_path(checkout, session)
        before = target.read_bytes()
        conn = sqlite3.connect(str(paths.state_db(checkout)))
        try:
            conn.executemany("UPDATE meta SET value=? WHERE key=?", [
                ("999", "schema_version"), ("9.9", "contract_version"),
                ("another-checkout", "checkout_id")])
            conn.commit()
        finally:
            conn.close()

        self.fire("UserPromptSubmit", session="state-test", prompt="must be refused",
                  prompt_id="t1")

        self.assertEqual(before, target.read_bytes())
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))
        status, detail = self.checks()["state db"]
        self.assertEqual(doctor.FAIL, status)
        self.assertIn("schema_version", detail)

    def test_a_supported_historical_session_index_can_resume(self):
        self.init_repo()
        self.fire("SessionStart", session="historical", source="startup")
        checkout = identity.peek_checkout_id(self.repo)
        session = identity.session_id("claude-code", "historical")
        with state.connect(paths.state_db(checkout)) as conn:
            conn.execute("UPDATE sessions SET contract_version='1.15' "
                         "WHERE session_id=?", (session,))

        self.fire("UserPromptSubmit", session="historical", prompt="resume",
                  prompt_id="t1")

        self.assertEqual(1, len(record.load(checkout, session)["turns"]))
        self.assertEqual(doctor.OK, self.checks()["state db"][0])

    def test_an_unsupported_session_index_fails_doctor(self):
        self.init_repo()
        self.fire("SessionStart", session="future-row", source="startup")
        checkout = identity.peek_checkout_id(self.repo)
        session = identity.session_id("claude-code", "future-row")
        with state.connect(paths.state_db(checkout)) as conn:
            conn.execute("UPDATE sessions SET contract_version='99.0' "
                         "WHERE session_id=?", (session,))

        status, detail = self.checks()["state db"]
        self.assertEqual(doctor.FAIL, status)
        self.assertIn("unsupported contract_version", detail)

    def test_a_refusal_survives_when_sqlite_is_unavailable(self):
        checkout = "fallback-checkout"
        with mock.patch.object(
                state, "ensure_schema",
                side_effect=sqlite3.DatabaseError("database unavailable")):
            recorded = state.record_refusal(
                checkout, "turn.tool", "database unavailable",
                "2026-01-01T00:00:00.000Z")

        self.assertTrue(recorded)
        self.assertEqual("turn.tool", state.refusals(checkout)[0]["kind"])

    def test_unwritable_record_storage_is_a_failure(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        storage = paths.checkout_dir(checkout)
        old_mode = storage.stat().st_mode
        storage.chmod(0o500)
        try:
            status, detail = self.checks()["storage writable"]
        finally:
            storage.chmod(old_mode)
        self.assertEqual(doctor.FAIL, status)
        self.assertIn(str(storage), detail)

    def test_installed_hooks_are_reported(self):
        self.init_repo()
        self.install()
        status, detail = self.checks()["plugin"]
        self.assertEqual("ok", status)
        self.assertIn(f"{len(claude.HOOK_TIMEOUTS)} hooks in", detail)

    def test_a_hook_whose_interpreter_is_gone_is_not_healthy(self):
        self.init_repo()
        self.install()
        self.assertEqual("ok", self.checks()["plugin"][0])

        def gone(hooks):
            for groups in hooks.values():
                for group in groups:
                    for entry in group["hooks"]:
                        entry["command"] = "/nonexistent/python"
        self.rewrite_plugin_hooks(gone)
        status, detail = self.checks()["plugin"]
        self.assertEqual("fail", status)
        self.assertIn("/nonexistent/python", detail)

    def test_a_hook_whose_entry_script_is_gone_is_not_healthy(self):
        self.init_repo()
        self.install()
        self.rewrite_plugin_hooks(lambda hooks: hooks.update({
            event: [{"hooks": [{"type": "command", "command": sys.executable,
                                "args": ["/gone/repohone_hook.py", "hook"]}]}]
            for event in claude.HOOK_TIMEOUTS}))
        status, detail = self.checks()["plugin"]
        self.assertEqual("fail", status)
        self.assertIn("does not exist", detail)

    def test_missing_lifecycle_hooks_are_not_healthy(self):
        """A partial install captures partial evidence, silently."""
        self.init_repo()
        self.install()

        def only_prompts(hooks):
            kept = hooks["UserPromptSubmit"]
            hooks.clear()
            hooks["UserPromptSubmit"] = kept
        self.rewrite_plugin_hooks(only_prompts)
        status, detail = self.checks()["plugin"]
        self.assertEqual("fail", status)
        self.assertIn("are not hooked", detail)
        self.assertIn(f"of {len(claude.HOOK_TIMEOUTS)} events", detail)

    def test_a_non_executable_hook_command_is_not_healthy(self):
        self.init_repo()
        self.install()
        dead = self.tmp / "not-executable"
        dead.write_text("#!/bin/sh\n")
        dead.chmod(0o644)
        self.rewrite_plugin_hooks(lambda hooks: hooks.update({
            event: [{"hooks": [{"type": "command", "command": str(dead),
                                "args": [str(HOOK), "hook"]}]}]
            for event in claude.HOOK_TIMEOUTS}))
        status, detail = self.checks()["plugin"]
        self.assertEqual("fail", status)
        self.assertIn("not executable", detail)

    def test_a_refused_event_is_surfaced(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        self.fire("Stop", prompt_id="does-not-exist", last_assistant_message="x")
        status, detail = self.checks()["capture errors"]
        self.assertEqual("warn", status)
        self.assertIn("could not be captured", detail)

    def test_an_event_refused_before_any_record_exists_is_still_visible(self):
        """It is refused before a session is known, so there is no record to put
        it in — and a dropped event must not read as a quiet checkout."""
        self.init_repo()
        self.fire("UserPromptSubmit", session="", prompt="lost work", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        self.assertEqual([], record.list_sessions(checkout))
        status, detail = self.checks()["capture errors"]
        self.assertEqual("fail", status)
        self.assertIn("no session id", detail)

    def test_a_checkout_that_captured_nothing_is_a_failure_not_a_warning(self):
        """Regression: refusals were counted but never compared with what
        landed, so 'every turn lost' and 'three of four hundred lost' printed
        identically — and a record with no turns looks like a quiet session."""
        self.init_repo()
        for turn in range(1, 5):
            self.fire("UserPromptSubmit", prompt=f"work {turn}")   # no prompt_id
            self.fire("Stop", last_assistant_message="done")
        status, detail = self.checks()["capture errors"]
        self.assertEqual("fail", status, detail)
        self.assertIn("not capturing", detail)

    def test_a_few_losses_among_many_captured_turns_stay_a_warning(self):
        self.init_repo()
        for turn in range(1, 5):
            self.fire("UserPromptSubmit", prompt=f"work {turn}", prompt_id=f"t{turn}")
            self.fire("Stop", prompt_id=f"t{turn}", last_assistant_message="done")
        self.fire("Stop", prompt_id="never-prompted", last_assistant_message="x")
        status, detail = self.checks()["capture errors"]
        self.assertEqual("warn", status, detail)

    def test_an_unreadable_records_directory_is_not_an_empty_one(self):
        """Regression: `list_sessions` globbed, and the audit used `rglob`; both
        yield nothing for a directory they cannot read, so doctor reported
        `records: ok, 0`."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        records = paths.records_dir(identity.peek_checkout_id(self.repo))
        os.chmod(records, 0)
        try:
            status, detail = self.checks()["records"]
        finally:
            os.chmod(records, 0o755)
        self.assertEqual("fail", status, detail)

    def test_capture_errors_says_when_it_cannot_count(self):
        """The refusal-versus-recorded comparison needs the record list; the
        records check can fail for other reasons, so this one must say so too."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        records = paths.records_dir(checkout)
        os.chmod(records, 0)
        try:
            _, status, detail = doctor._capture_errors(checkout)
            refs_status, refs_detail = self.checks()["snapshot refs"]
        finally:
            os.chmod(records, 0o755)
        self.assertEqual("fail", status, detail)
        self.assertIn("cannot be listed", detail)
        self.assertEqual("fail", refs_status)
        self.assertIn("cannot be checked", refs_detail,
                      "an unlistable record set is not a dangling ref")

    def test_the_audit_reports_a_directory_it_cannot_walk_into(self):
        """`rglob` skipped an unreadable directory without a word. Records are
        listed separately, so this uses one only the audit walks."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        hidden = paths.checkout_dir(identity.peek_checkout_id(self.repo)) / "proposals"
        hidden.mkdir(parents=True, exist_ok=True)
        os.chmod(hidden, 0)
        try:
            status, detail = self.checks()["records"]
        finally:
            os.chmod(hidden, 0o755)
        self.assertEqual("fail", status, detail)
        self.assertIn("cannot be read", detail)

    def test_a_clean_checkout_reports_no_capture_errors(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        self.assertEqual("ok", self.checks()["capture errors"][0])

    def test_the_global_hook_error_log_is_never_reported_as_healthy(self):
        self.init_repo()
        checkout = identity.checkout_id(self.repo)
        state.initialize(checkout)
        paths.error_log().parent.mkdir(parents=True, exist_ok=True)
        paths.error_log().write_text(
            "2026-01-01T00:00:00.000Z hook: unexpected failure\n",
            encoding="utf-8")
        status, detail = self.checks()["capture errors"]
        self.assertEqual("warn", status)
        self.assertIn("errors.log", detail)

    def test_invalid_utf8_in_the_fallback_log_is_visible(self):
        self.init_repo()
        checkout = identity.checkout_id(self.repo)
        state.initialize(checkout)
        target = paths.checkout_dir(checkout) / "capture-refusals.jsonl"
        target.write_bytes(b'{"at":"t","kind":"x","reason":"y"}\n\xff\n')
        status, detail = self.checks()["capture errors"]
        self.assertEqual("warn", status)
        self.assertTrue(any(row["kind"] == "refusal-log" for row in state.refusals(checkout)))

    def test_doctor_does_not_crash_on_an_invalid_record_filename(self):
        self.init_repo()
        checkout = identity.checkout_id(self.repo)
        target = paths.records_dir(checkout) / "bad.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
        rows = doctor.run(self.repo)
        records = next(row for row in rows if row[0] == "records")
        self.assertEqual("fail", records[1])

    def test_a_deleted_submodule_ref_is_not_healthy(self):
        self.init_repo()
        sub = self.tmp / "vendor"
        sub.mkdir()
        for args in (["init", "-q"], ["config", "user.email", "d@e.com"],
                     ["config", "user.name", "D"]):
            git(args, sub)
        (sub / "lib.py").write_text("x = 1\n")
        git(["add", "-A"], sub)
        git(["commit", "-qm", "sub"], sub)
        git(["-c", "protocol.file.allow=always", "submodule", "add", "-q",
             str(sub), "vendor"], self.repo)
        git(["commit", "-qm", "add submodule"], self.repo)
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        vendor = self.repo / "vendor"
        self.assertNotEqual("", git(["for-each-ref", "refs/repohone/"], vendor))
        self.assertEqual("ok", self.checks()["snapshot refs"][0])
        for ref in git(["for-each-ref", "--format=%(refname)", "refs/repohone/"],
                       vendor).splitlines():
            git(["update-ref", "-d", ref.strip()], vendor)
        status, detail = self.checks()["snapshot refs"]
        self.assertEqual("fail", status)
        self.assertIn("vendor/", detail)

    def test_the_plugin_must_be_on_in_this_checkout(self):
        self.init_repo()
        self.install()
        status, detail = self.checks()["enabled here"]
        self.assertEqual(("fail", True), (status, "off at user scope" in detail))
        claude.set_enabled(self.repo / claude.SETTINGS_FILE, True)
        self.assertEqual(("ok", "on at local scope"), self.checks()["enabled here"])

    def test_a_plugin_on_in_every_repository_is_a_warning(self):
        self.init_repo()
        self.install()
        claude.set_enabled(claude.config_dir() / "settings.json", True)
        status, detail = self.checks()["enabled here"]
        self.assertEqual("warn", status)
        self.assertIn("every repository", detail)

    def test_consent_is_checked_for_this_exact_configuration(self):
        self.init_repo()
        self.assertEqual("ok", self.checks()["consent"][0])
        target = self.repo / ".repohone" / "profile.yaml"
        target.write_text(target.read_text().replace("tool_capture: true", "tool_capture: false"))
        status, detail = self.checks()["consent"]
        self.assertEqual(("fail", True), (status, "changed since you accepted" in detail))
        profile.withdraw(self.repo)
        status, detail = self.checks()["consent"]
        self.assertEqual(("fail", True), (status, "not accepted" in detail))

    def test_what_an_older_install_left_is_named(self):
        self.init_repo()
        local = self.repo / claude.SETTINGS_FILE
        local.parent.mkdir(exist_ok=True)
        local.write_text(json.dumps(claude.settings(sys.executable, str(CORE))))
        status, detail = self.checks()["older install"]
        self.assertEqual("warn", status)
        self.assertIn(f"{len(claude.HOOK_TIMEOUTS)} RepoHone hook(s) in {claude.SETTINGS_FILE}",
                      detail)


class PurgeIsComplete(Fixture):
    """Regression: purge cleared only the root store, so a submodule kept a
    durable snapshot of the same work, and the installed hooks stayed behind."""

    def add_submodule(self, name="vendor"):
        sub = self.tmp / name
        sub.mkdir()
        for args in (["init", "-q"], ["config", "user.email", "d@e.com"],
                     ["config", "user.name", "D"]):
            git(args, sub)
        (sub / "lib.py").write_text("x = 1\n")
        git(["add", "-A"], sub)
        git(["commit", "-qm", "sub"], sub)
        git(["-c", "protocol.file.allow=always", "submodule", "add", "-q",
             str(sub), name], self.repo)
        git(["commit", "-qm", "add submodule"], self.repo)
        return self.repo / name

    def deinit(self, purge=True):
        return cli.cmd_deinit(argparse.Namespace(
            path=str(self.repo), purge=purge, yes=True, dry_run=False))

    def test_purge_removes_refs_in_initialized_submodules(self):
        self.init_repo()
        vendor = self.add_submodule()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        self.assertNotEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))
        self.assertNotEqual("", git(["for-each-ref", "refs/repohone/"], vendor),
                            "fixture did not pin a submodule ref")
        with contextlib.redirect_stdout(io.StringIO()):
            self.deinit()
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], vendor),
                         "a submodule kept this checkout's snapshot after purge")

    def test_purge_keeps_submodule_refs_owned_by_another_checkout(self):
        self.init_repo()
        vendor = self.add_submodule()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        mine = git(["for-each-ref", "--format=%(refname)", "refs/repohone/"], vendor)
        other = mine.replace(identity.peek_checkout_id(self.repo), "some-other-checkout")
        git(["update-ref", other, git(["rev-parse", mine], vendor)], vendor)
        with contextlib.redirect_stdout(io.StringIO()):
            self.deinit()
        self.assertEqual(other,
                         git(["for-each-ref", "--format=%(refname)", "refs/repohone/"],
                             vendor),
                         "purge reached another checkout's submodule refs")

    def test_purge_finds_a_submodule_missing_from_the_current_branch(self):
        self.init_repo()
        original = git(["branch", "--show-current"], self.repo)
        git(["checkout", "-qb", "with-submodule"], self.repo)
        vendor = self.add_submodule()
        self.fire("UserPromptSubmit", prompt="work", prompt_id="t1")
        self.assertNotEqual("", git(["for-each-ref", "--format=%(refname)",
                                     "refs/repohone/"], vendor))
        git(["checkout", "-q", original], self.repo)
        self.assertFalse((self.repo / ".gitmodules").exists())
        ref_check = {name: (status, detail)
                     for name, status, detail in doctor.run(self.repo)}["snapshot refs"]
        self.assertEqual("ok", ref_check[0])
        self.assertIn("2 under", ref_check[1],
                      "doctor missed the retained historical submodule store")

        with contextlib.redirect_stdout(io.StringIO()):
            status = self.deinit()

        self.assertEqual(0, status)
        module_git = self.repo / ".git" / "modules" / "vendor"
        self.assertEqual(
            "", subprocess.run(
                ["git", f"--git-dir={module_git}", "for-each-ref",
                 "--format=%(refname)", "refs/repohone/"],
                capture_output=True, text=True, check=True).stdout.strip())

    def test_deinit_removes_its_own_hooks_and_keeps_the_projects(self):
        self.init_repo()
        settings = self.repo / ".claude" / "settings.json"
        settings.parent.mkdir(exist_ok=True)
        ours = claude.settings(sys.executable, str(CORE))["hooks"]
        settings.write_text(json.dumps({"hooks": {**ours, "Stop": ours["Stop"] + [
            {"hooks": [{"type": "command", "command": "make lint"}]}]}}))
        local = self.repo / ".claude" / "settings.local.json"
        local.write_text(json.dumps({"hooks": ours}))
        with contextlib.redirect_stdout(io.StringIO()):
            self.deinit()
        left = json.loads(settings.read_text()).get("hooks", {})
        entries = [e for ev in left.values() for g in ev for e in g["hooks"]]
        self.assertEqual([{"type": "command", "command": "make lint"}], entries)
        self.assertEqual({"enabledPlugins": {claude.PLUGIN_ID: False}},
                         json.loads(local.read_text()))

    def test_deinit_leaves_unreadable_settings_untouched(self):
        self.init_repo()
        settings = self.repo / ".claude" / "settings.json"
        settings.parent.mkdir(exist_ok=True)
        settings.write_text("{not json")
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(0, self.deinit())
        self.assertEqual("{not json", settings.read_text())


class Redaction(unittest.TestCase):
    def test_one_secret_counts_once(self):
        from repohone.redact import redact
        for text in ("token = ghp_abcdefghijklmnop1234", "api_key: sup3rs3cretvalue123"):
            _, hits = redact(text)
            self.assertEqual(1, hits, f"{text!r} was masked more than once")

    def test_two_secrets_count_twice(self):
        from repohone.redact import redact
        _, hits = redact("AKIAIOSFODNN7EXAMPLE and password=hunter2xyz")
        self.assertEqual(2, hits)


class OrphanedRefs(Fixture):
    """Losing state.db while refs survive would restart ordinals at 1 and make
    every snapshot an integrity error; doctor has to say so."""

    def test_doctor_reports_refs_without_ordinal_state(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        paths.state_db(checkout).unlink()
        rows = doctor.run(self.repo)
        refs_row = [r for r in rows if r[0] == "snapshot refs"][0]
        self.assertEqual(doctor.FAIL, refs_row[1])
        self.assertIn("ordinal state", refs_row[2])


class SessionShapes(Fixture):
    """ARCH §16, §18, §20.1 — the real event sequences, end to end."""

    def setUp(self):
        super().setUp()
        self.init_repo()

    def test_a_normal_turn(self):
        self.fire("SessionStart", source="startup")
        self.fire("UserPromptSubmit", prompt="add a docstring", prompt_id="t1")
        (self.repo / "calc.py").write_text('"""m."""\n')
        self.fire("Stop", prompt_id="t1", effort="medium", model="claude-opus-5")
        rec = self.the_record()
        turn = rec["turns"][0]
        self.assertEqual("stop", turn["completion"])
        self.assertEqual("observed", turn["completion_source"])
        self.assertEqual("claude-opus-5", turn["model"])
        self.assert_conformant(rec)

    def test_type_ahead_extends_one_turn(self):
        """ARCH §18 — one prompt_id, two submissions, one turn, prompt/1 preserved."""
        self.fire("UserPromptSubmit", prompt="first", prompt_id="t1")
        self.fire("UserPromptSubmit", prompt="corrective follow-up", prompt_id="t1")
        self.fire("Stop", prompt_id="t1")
        rec = self.the_record()
        self.assertEqual(1, len(rec["turns"]), "type-ahead created a phantom turn")
        events = rec["turns"][0]["prompt_events"]
        self.assertEqual([1, 2], [e["ordinal"] for e in events])
        self.assertEqual("first", events[0]["content"]["text"])
        self.assert_conformant(rec)

    def test_an_interrupted_turn_is_derived_on_claude(self):
        """ARCH §32 — Claude fires no event; the adapter derives it."""
        self.fire("UserPromptSubmit", prompt="refactor", prompt_id="t1")
        self.fire("UserPromptSubmit", prompt="never mind", prompt_id="t2")
        self.fire("Stop", prompt_id="t2")
        rec = self.the_record()
        self.assertEqual("interrupted", rec["turns"][0]["completion"])
        self.assertEqual("derived", rec["turns"][0]["completion_source"])
        self.assertEqual([], rec["turns"][0]["stop_events"])
        self.assert_conformant(rec)

    def test_an_interrupted_turn_is_closed_by_session_end(self):
        self.fire("UserPromptSubmit", prompt="refactor", prompt_id="t1")
        self.fire("SessionEnd", reason="clear")
        rec = self.the_record()
        self.assertEqual("interrupted", rec["turns"][0]["completion"])
        self.assertEqual(1, len(rec["session_end_snapshots"]))
        self.assert_conformant(rec)

    def test_stop_failure_records_the_error_class(self):
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.fire("StopFailure", prompt_id="t1", error={"type": "rate_limit"})
        rec = self.the_record()
        turn = rec["turns"][0]
        self.assertEqual("stop_failure", turn["completion"])
        self.assertEqual("observed", turn["completion_source"])
        self.assertEqual("rate_limit", turn["stop_events"][0]["error"])
        self.assert_conformant(rec)

    def test_repeated_stop_events_are_kept_separately(self):
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.fire("Stop", prompt_id="t1", stop_hook_active=False)
        self.fire("Stop", prompt_id="t1", stop_hook_active=True)
        rec = self.the_record()
        stops = rec["turns"][0]["stop_events"]
        self.assertEqual([1, 2], [s["ordinal"] for s in stops])
        self.assertTrue(stops[1]["host_continuation"])
        self.assert_conformant(rec)

    def test_a_session_that_dies_mid_turn_stays_pending(self):
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        rec = self.the_record()
        self.assertEqual("pending", rec["turns"][0]["completion"])
        self.assertIsNone(rec["turns"][0]["completion_source"])
        self.assertEqual("observed", rec["lifecycle"]["state"])
        self.assert_conformant(rec)

    def test_workspace_change_between_turns_is_detected(self):
        """ARCH §20 — evidence that source changed outside the agent's cycle."""
        self.fire("UserPromptSubmit", prompt="one", prompt_id="t1")
        self.fire("Stop", prompt_id="t1")
        (self.repo / "calc.py").write_text("# edited by a human\n")
        self.fire("UserPromptSubmit", prompt="two", prompt_id="t2")
        self.fire("Stop", prompt_id="t2")
        rec = self.the_record()
        self.assertTrue(rec["turns"][1]["workspace_changed_before_turn"])
        self.assert_conformant(rec)

    def test_nothing_is_attributable_after_an_interrupt(self):
        """ARCH §20.1 — an interrupted turn has no moment at which the agent stopped."""
        self.fire("UserPromptSubmit", prompt="one", prompt_id="t1")
        self.fire("UserPromptSubmit", prompt="two", prompt_id="t2")
        rec = self.the_record()
        self.assertIsNone(rec["turns"][1]["workspace_changed_before_turn"])

    def test_the_record_stays_in_the_observed_phase(self):
        """ARCH §19 — Phase 1 delivers the structure, not reconciliation."""
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.fire("Stop", prompt_id="t1")
        self.fire("SessionEnd", reason="clear")
        rec = self.the_record()
        self.assertEqual("observed", rec["lifecycle"]["state"])
        self.assertEqual("pending", rec["reconciliation"]["outcome"])
        self.assertIsNone(rec["lifecycle"]["reconciled_at"])
        self.assert_conformant(rec)

    def test_versions_are_recorded_never_guessed(self):
        """ARCH §22.1 — both version fields, neither inferred."""
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        rec = self.the_record()
        self.assertEqual(1, rec["schema_version"])
        self.assertEqual(CONTRACT_VERSION, rec["contract_version"])
        self.assertEqual("repohone.session/v1", rec["schema"])

    def test_prompts_are_redacted_before_storage(self):
        self.fire("UserPromptSubmit", prompt="deploy with AKIAIOSFODNN7EXAMPLE now",
                  prompt_id="t1")
        rec = self.the_record()
        content = rec["turns"][0]["prompt_events"][0]["content"]
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", content["text"])
        self.assertEqual(1, content["redactions"])

    def test_resume_registers_once_and_appends_a_start(self):
        self.fire("SessionStart", source="startup")
        self.fire("SessionStart", source="resume")
        rec = self.the_record()
        self.assertEqual(["startup", "resume"],
                         [s["source"] for s in rec["lifecycle"]["starts"][-2:]])


class AdapterMapping(unittest.TestCase):
    """ARCH §32 — Claude specifics live here, not in the generic contract."""

    def test_hook_names_map_to_canonical_events(self):
        for hook, canonical in (("UserPromptSubmit", "turn.prompt"), ("Stop", "turn.stop"),
                                ("StopFailure", "turn.failed"), ("SessionEnd", "session.end")):
            event = claude.to_event({"hook_event_name": hook, "session_id": "s", "cwd": "."})
            self.assertEqual(canonical, event.kind)

    def test_claude_never_produces_an_observed_interrupt(self):
        """The Claude adapter has no interrupt event at all; capture derives it."""
        self.assertNotIn("turn.interrupted", claude.EVENTS.values())

    def test_unknown_hooks_are_ignored(self):
        self.assertIsNone(claude.to_event({"hook_event_name": "SomethingNew"}))

    def test_generated_settings_use_the_exec_form(self):
        cfg = claude.settings("/usr/bin/python3", "/opt/repohone_hook.py")
        entry = cfg["hooks"]["Stop"][0]["hooks"][0]
        self.assertEqual("command", entry["type"])
        self.assertEqual("/usr/bin/python3", entry["command"])
        self.assertIn("/opt/repohone_hook.py", entry["args"])
        self.assertNotIn("$CLAUDE_PROJECT_DIR", json.dumps(cfg))


class VersionHandling(Fixture):
    """ARCH §22 — a reader never applies current meaning to an unknown version."""

    def _write(self, name, rec):
        d = paths.records_dir(identity.checkout_id(self.repo))
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(rec))
        return identity.checkout_id(self.repo)

    def test_a_supported_record_loads(self):
        self.init_repo()
        self.fire("UserPromptSubmit", session="supported", prompt="go", prompt_id="t1")
        co = identity.peek_checkout_id(self.repo)
        session = identity.session_id("claude-code", "supported")
        self.assertIsNotNone(record.load(co, session))

    def test_a_newer_schema_version_is_refused(self):
        co = self._write("rh_f0700001", {"schema": "repohone.session/v1",
                                        "schema_version": 2, "contract_version": "1.4"})
        with self.assertRaises(record.UnsupportedVersion):
            record.load(co, "rh_f0700001")

    def test_a_newer_contract_version_is_refused(self):
        co = self._write("rh_9eae0001", {"schema": "repohone.session/v1",
                                       "schema_version": 1, "contract_version": "9.9"})
        with self.assertRaises(record.UnsupportedVersion):
            record.load(co, "rh_9eae0001")

    def test_inspection_can_still_read_what_analysis_refuses(self):
        co = self._write("rh_f0700001", {"schema": "repohone.session/v1",
                                        "schema_version": 2, "contract_version": "1.4"})
        self.assertIsNotNone(record.load(co, "rh_f0700001", strict=False))

    def test_capture_neither_mutates_nor_snapshots_for_a_future_record(self):
        self.init_repo()
        session_id = identity.session_id("claude-code", "future-session")
        checkout = self._write(session_id, {
            "schema": "repohone.session/v1", "schema_version": 99,
            "contract_version": CONTRACT_VERSION, "sentinel": "unchanged"})
        target = paths.records_dir(checkout) / f"{session_id}.json"
        before = target.read_text()

        result = self.fire("UserPromptSubmit", session="future-session",
                           prompt="must not be captured", prompt_id="t1")

        self.assertEqual(0, result.returncode)
        self.assertEqual(before, target.read_text())
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))


class RealPayloads(Fixture):
    """Payloads captured verbatim from Claude Code 2.1.267.

    Hand-written fixtures once used a bare `effort` string, which the host does
    not send; these keep the adapter honest against the shapes that actually
    arrive.
    """

    SESSION_START = {"hook_event_name": "SessionStart", "source": "startup",
                     "scratchpad_dir": "/tmp/scratch"}
    PROMPT = {"hook_event_name": "UserPromptSubmit", "permission_mode": "acceptEdits",
              "prompt_id": "b776eb34-490b-4078-912c-0e1ca9f7633c",
              "prompt": "Rename the parameter xs to values in calc.py. Just edit.",
              "scratchpad_dir": "/tmp/scratch"}
    STOP = {"hook_event_name": "Stop", "background_tasks": [], "effort": {"level": "high"},
            "last_assistant_message": "Done.", "permission_mode": "acceptEdits",
            "prompt_id": "b776eb34-490b-4078-912c-0e1ca9f7633c",
            "session_crons": [], "stop_hook_active": False}
    SESSION_END = {"hook_event_name": "SessionEnd", "reason": "other",
                   "prompt_id": "b776eb34-490b-4078-912c-0e1ca9f7633c"}

    def _fire_real(self, payload):
        full = dict(payload, session_id="real-1", cwd=str(self.repo),
                    transcript_path="/tmp/real-1.jsonl")
        return subprocess.run([sys.executable, str(HOOK), "hook"], input=json.dumps(full),
                              cwd=str(self.repo), capture_output=True, text=True, env=self.env)

    def test_a_real_session_produces_a_conformant_record(self):
        self.init_repo()
        for payload in (self.SESSION_START, self.PROMPT, self.STOP, self.SESSION_END):
            proc = self._fire_real(payload)
            self.assertEqual(0, proc.returncode, proc.stderr)
        rec = self.the_record("real-1")
        self.assert_conformant(rec)

    def test_effort_arrives_as_an_object(self):
        self.init_repo()
        self._fire_real(self.PROMPT)
        self._fire_real(self.STOP)
        rec = self.the_record("real-1")
        self.assertEqual("high", rec["turns"][0]["stop_events"][0]["effort"])

    def test_the_last_assistant_message_is_stored_redacted(self):
        self.init_repo()
        self._fire_real(self.PROMPT)
        self._fire_real(self.STOP)
        message = self.the_record("real-1")["turns"][0]["stop_events"][0]["last_assistant_message"]
        self.assertEqual("Done.", message["text"])
        self.assertEqual(0, message["redactions"])

    def test_claude_exposes_no_model_so_it_stays_unobserved(self):
        """ARCH open question 2 — settled for `claude -p` on 2.1.267: no hook
        payload carries a model, so the record says so rather than guessing."""
        self.init_repo()
        for payload in (self.SESSION_START, self.PROMPT, self.STOP):
            self._fire_real(payload)
        rec = self.the_record("real-1")
        self.assertEqual([], rec["models"])
        self.assertIsNone(rec["turns"][0]["model"])
        self.assert_conformant(rec)

    def test_one_start_entry_per_observed_start(self):
        self.init_repo()
        self._fire_real(self.SESSION_START)
        starts = self.the_record("real-1")["lifecycle"]["starts"]
        self.assertEqual(["startup"], [s["source"] for s in starts])


class ToolCapture(Fixture):
    """Phase 2 — PreToolUse. What the agent consulted before acting is what
    separates MISSING_CONTEXT from AGENT_REASONING_ERROR (LEARNING_PLAN §6)."""

    PRE_TOOL = {"hook_event_name": "PreToolUse", "prompt_id": "t1",
                "permission_mode": "acceptEdits", "effort": {"level": "high"},
                "tool_use_id": "toolu_01JGXpix8JrzqJvpQJ5dDJuK"}

    def fire_raw(self, payload, session="s-1"):
        full = dict(payload, session_id=session, cwd=str(self.repo),
                    transcript_path="/tmp/x")
        return subprocess.run([sys.executable, str(HOOK), "hook"], input=json.dumps(full),
                              cwd=str(self.repo), capture_output=True, text=True, env=self.env)

    def _events(self, session="s-1"):
        rec = self.the_record(session)
        turns = rec["extensions"].get("tool_events", {}).get("turns", {})
        return rec, [e for data in turns.values() for e in data["events"]]

    def test_tool_use_is_recorded_against_its_turn(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Read", {"file_path": str(self.repo / "calc.py")})
        self._tool("Bash", {"command": "pytest -q"})
        self.fire("Stop", prompt_id="t1")
        rec, events = self._events()
        self.assertEqual(["Read", "Bash"], [e["tool"] for e in events])
        self.assertEqual("calc.py", events[0]["detail"]["file_path"])
        self.assertEqual("pytest -q", events[1]["detail"]["command"])
        self.assert_conformant(rec)

    def test_file_contents_are_never_stored(self):
        """Write.tool_input carries the whole file; the snapshot already has it."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Write", {"file_path": str(self.repo / "big.py"),
                             "content": "SUPER_SECRET_BODY " * 200})
        self.fire("Stop", prompt_id="t1")
        rec, events = self._events()
        self.assertNotIn("SUPER_SECRET_BODY", json.dumps(rec))
        self.assertEqual({"file_path": "big.py"}, events[0]["detail"])

    def test_edit_strings_are_never_stored(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Edit", {"file_path": str(self.repo / "calc.py"),
                            "old_string": "OLD_SECRET", "new_string": "NEW_SECRET"})
        self.fire("Stop", prompt_id="t1")
        rec, _ = self._events()
        self.assertNotIn("OLD_SECRET", json.dumps(rec))
        self.assertNotIn("NEW_SECRET", json.dumps(rec))

    def test_unknown_input_fields_are_not_captured(self):
        """Allowlist: a tool that gains a secret-bearing field stays unrecorded."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("FutureTool", {"mystery_field": "LEAKED_VALUE"})
        self.fire("Stop", prompt_id="t1")
        rec, events = self._events()
        self.assertNotIn("LEAKED_VALUE", json.dumps(rec))
        self.assertEqual("FutureTool", events[0]["tool"])

    def test_secrets_in_commands_are_redacted(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Bash", {"command": "export TOKEN=ghp_abcdefghijklmnop1234 && make"})
        self.fire("Stop", prompt_id="t1")
        rec, events = self._events()
        self.assertNotIn("ghp_abcdefghijklmnop1234", json.dumps(rec))
        self.assertIn("make", events[0]["detail"]["command"])

    def test_absolute_paths_do_not_identify_the_machine(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Bash", {"command": f"cat {self.repo}/calc.py"})
        self._tool("Read", {"file_path": "/etc/passwd"})
        self.fire("Stop", prompt_id="t1")
        rec, events = self._events()
        self.assertEqual("cat <repo>/calc.py", events[0]["detail"]["command"])
        self.assertEqual("<outside-repo>", events[1]["detail"]["file_path"])

    def test_tool_capture_can_be_disabled(self):
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(
            "profile_version: 1\ntool_capture: false\n")
        profile.accept(self.repo, profile.load(self.repo))
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Read", {"file_path": str(self.repo / "calc.py")})
        self.fire("Stop", prompt_id="t1")
        rec = self.the_record()
        self.assertNotIn("tool_events", rec["extensions"])

    def test_a_non_boolean_tool_capture_is_invalid(self):
        (self.repo / ".repohone").mkdir(exist_ok=True)
        (self.repo / ".repohone" / "profile.yaml").write_text(
            "profile_version: 1\ntool_capture: maybe\n")
        self.assertEqual(profile.INVALID, profile.load(self.repo).state)

    def test_pre_tool_use_never_rewrites_the_record(self):
        """It fires on every tool call; touching the record each time would make
        the hook quadratic and contend on the lock."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        path = record.record_path(identity.peek_checkout_id(self.repo),
                                  identity.session_id("claude-code", "s-1"))
        before = path.stat().st_mtime_ns
        time.sleep(0.01)
        for _ in range(5):
            self._tool("Read", {"file_path": str(self.repo / "calc.py")})
        self.assertEqual(before, path.stat().st_mtime_ns,
                         "PreToolUse rewrote the session record")

    def _tool(self, name, tool_input, tool_use_id="tu1", prompt_id="t1"):
        payload = dict(self.PRE_TOOL, tool_name=name, tool_input=tool_input,
                       tool_use_id=tool_use_id, prompt_id=prompt_id)
        return self.fire_raw(payload)

    def test_sqlite_does_not_keep_tool_events_after_the_session_ends(self):
        """ARCH §21.1 — SQLite owns coordination, the record owns history."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        for i in range(5):
            self._tool("Read", {"file_path": str(self.repo / "calc.py")}, f"tu{i}")
        self.fire("Stop", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        with state.connect(paths.state_db(checkout)) as conn:
            before = conn.execute("SELECT count(*) FROM tool_events").fetchone()[0]
        self.assertEqual(5, before)

        self.fire("SessionEnd", reason="clear")
        with state.connect(paths.state_db(checkout)) as conn:
            after = conn.execute("SELECT count(*) FROM tool_events").fetchone()[0]
        self.assertEqual(0, after, "tool events accumulated in SQLite after folding")
        turn = self.the_record()["extensions"]["tool_events"]["turns"]["t1"]
        self.assertEqual(5, turn["count"], "history was lost, not just staged")

    def test_a_resumed_session_keeps_earlier_turns(self):
        """Staged rows are cleared at session end; a wholesale re-fold would then
        overwrite the turns already folded into the record."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="one", prompt_id="t1")
        self._tool("Read", {"file_path": str(self.repo / "calc.py")}, "tu1")
        self.fire("Stop", prompt_id="t1")
        self.fire("SessionEnd", reason="clear")

        self.fire("UserPromptSubmit", prompt="two", prompt_id="t2")
        self._tool("Write", {"file_path": str(self.repo / "new.py")}, "tu2", prompt_id="t2")
        self.fire("Stop", prompt_id="t2")
        turns = self.the_record()["extensions"]["tool_events"]["turns"]
        self.assertIn("t1", turns, "the first turn's tool evidence was lost on resume")
        self.assertIn("t2", turns)

    def test_folding_twice_is_idempotent(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Read", {"file_path": str(self.repo / "calc.py")})
        self.fire("Stop", prompt_id="t1")
        first = self.the_record()["extensions"]["tool_events"]
        self.fire("SessionEnd", reason="clear")
        self.assertEqual(first["turns"]["t1"]["events"],
                         self.the_record()["extensions"]["tool_events"]["turns"]["t1"]["events"])

    def test_uninitialized_repository_records_no_tool_use(self):
        """ARCH §5 — tool inputs are capture, and capture needs initialization."""
        self._tool("Bash", {"command": "cat ~/.aws/credentials"})
        self.assertFalse(self.data.exists())

    def test_the_hook_stays_silent_and_fast(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        proc = self._tool("Read", {"file_path": str(self.repo / "calc.py")})
        self.assertEqual(0, proc.returncode)
        self.assertEqual("", proc.stdout)

    def test_pre_tool_use_is_registered_with_a_short_timeout(self):
        cfg = claude.settings("/usr/bin/python3", "/opt/hook.py")
        self.assertIn("PreToolUse", cfg["hooks"])
        self.assertLessEqual(cfg["hooks"]["PreToolUse"][0]["hooks"][0]["timeout"], 10)

    def test_a_tool_state_failure_is_durable_and_visible_to_doctor(self):
        self.init_repo()
        prompt = claude.to_event({
            "hook_event_name": "UserPromptSubmit", "session_id": "s-1",
            "prompt_id": "t1", "prompt": "go", "cwd": str(self.repo)})
        self.assertTrue(capture.handle(prompt)["captured"])
        tool = claude.to_event({
            "hook_event_name": "PreToolUse", "session_id": "s-1",
            "prompt_id": "t1", "tool_name": "Read", "tool_use_id": "broken",
            "tool_input": {"file_path": str(self.repo / "calc.py")},
            "cwd": str(self.repo)})
        with mock.patch.object(state, "append_tool_event",
                               side_effect=OSError("state write failed")):
            result = capture.handle(tool)

        checkout = identity.peek_checkout_id(self.repo)
        self.assertFalse(result["captured"])
        self.assertTrue(result["refusal_recorded"])
        self.assertIn("state write failed", state.refusals(checkout)[0]["reason"])
        self.assertEqual(doctor.WARN, doctor._capture_errors(checkout)[1])


class ToolOutcomes(Fixture):
    """PostToolUse. PreToolUse alone records intent: a blocked call and a
    successful one are indistinguishable without a result event."""

    def _pre(self, name, tool_input, tool_use_id="tu1"):
        return self.fire("PreToolUse", prompt_id="t1", tool_name=name,
                         tool_use_id=tool_use_id, tool_input=tool_input)

    def _post(self, name, response, tool_use_id="tu1", duration_ms=12):
        return self.fire("PostToolUse", prompt_id="t1", tool_name=name,
                         tool_use_id=tool_use_id, tool_response=response,
                         duration_ms=duration_ms)

    def _events(self):
        rec = self.the_record()
        turns = rec["extensions"].get("tool_events", {}).get("turns", {})
        return rec, [e for d in turns.values() for e in d["events"]]

    def test_a_successful_call_records_its_outcome(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._pre("Bash", {"command": "echo fine"})
        self._post("Bash", {"stdout": "fine", "stderr": "", "interrupted": False})
        self.fire("Stop", prompt_id="t1")
        _, events = self._events()
        self.assertTrue(events[0]["outcome_observed"])
        self.assertEqual(4, events[0]["outcome"]["stdout_chars"])
        self.assertEqual(12, events[0]["outcome"]["duration_ms"])

    def test_a_call_with_no_result_event_is_marked_unobserved(self):
        """Verified on 2.1.267: a failed or blocked call fires no PostToolUse."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._pre("Bash", {"command": "false"})
        self.fire("Stop", prompt_id="t1")
        _, events = self._events()
        self.assertFalse(events[0]["outcome_observed"])
        self.assertIsNone(events[0]["outcome"])

    def test_outcomes_never_store_command_output(self):
        """stdout carries the same file contents and secrets the inputs do."""
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._pre("Bash", {"command": "cat secrets"})
        self._post("Bash", {"stdout": "AKIAIOSFODNN7EXAMPLE leaked",
                            "stderr": "SECRET_STDERR", "interrupted": False})
        self.fire("Stop", prompt_id="t1")
        rec, events = self._events()
        blob = json.dumps(rec)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", blob)
        self.assertNotIn("SECRET_STDERR", blob)
        self.assertEqual(27, events[0]["outcome"]["stdout_chars"])

    def test_a_result_for_an_unrecorded_call_is_not_invented(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._post("Bash", {"stdout": "x"}, tool_use_id="never-seen")
        self.fire("Stop", prompt_id="t1")
        _, events = self._events()
        self.assertEqual([], events)

    def test_the_adapter_documents_the_success_only_asymmetry(self):
        """Host knowledge stays in the adapter, as with interrupts (§32)."""
        self.assertTrue(claude.POST_TOOL_USE_ONLY_ON_SUCCESS)


class SubagentAttribution(Fixture):
    """A subagent's tool calls arrive under the parent's prompt_id with nothing
    to mark them, so start/stop events bracket them."""

    def _tool(self, name, tool_input, tid):
        return self.fire("PreToolUse", prompt_id="t1", tool_name=name,
                         tool_use_id=tid, tool_input=tool_input)

    def _events(self):
        turns = self.the_record()["extensions"]["tool_events"]["turns"]
        return [e for d in turns.values() for e in d["events"]]

    def test_subagent_work_is_separated_from_the_agents_own(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self._tool("Read", {"file_path": str(self.repo / "calc.py")}, "own")
        self.fire("SubagentStart", prompt_id="t1", agent_id="a1", agent_type="Explore")
        self._tool("Bash", {"command": "find src"}, "sub")
        self.fire("SubagentStop", prompt_id="t1", agent_id="a1", agent_type="Explore")
        self._tool("Write", {"file_path": str(self.repo / "out.py")}, "own2")
        self.fire("Stop", prompt_id="t1")
        events = self._events()
        self.assertEqual([None, "a1", None], [e.get("agent_id") for e in events])

    def test_overlapping_subagents_are_ambiguous_not_guessed(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.fire("SubagentStart", prompt_id="t1", agent_id="a1", agent_type="Explore")
        self.fire("SubagentStart", prompt_id="t1", agent_id="a2", agent_type="Explore")
        self._tool("Bash", {"command": "find src"}, "x")
        self.fire("SubagentStop", prompt_id="t1", agent_id="a1", agent_type="Explore")
        self.fire("Stop", prompt_id="t1")
        self.assertEqual("ambiguous", self._events()[0]["agent_id"])

    def test_markers_are_boundaries_not_tool_calls(self):
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.fire("SubagentStart", prompt_id="t1", agent_id="a1", agent_type="Explore")
        self._tool("Bash", {"command": "ls"}, "x")
        self.fire("SubagentStop", prompt_id="t1", agent_id="a1", agent_type="Explore")
        self.fire("Stop", prompt_id="t1")
        turn = list(self.the_record()["extensions"]["tool_events"]["turns"].values())[0]
        self.assertEqual(turn["count"], len(turn["events"]))
        self.assertEqual(1, turn["count"])


class ToolStorageBounds(Fixture):
    """Regression: the cap bounded only what was folded, not what was stored."""

    def test_storage_stops_at_the_cap(self):
        from repohone import capture as capture_mod
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        session = identity.session_id("claude-code", "s-1")
        cap = capture_mod.MAX_TOOL_EVENTS_PER_TURN
        for i in range(cap + 40):
            state.append_tool_event(checkout, session, "t1", record.now(), "Read",
                                    None, f"tu{i}", cap)
        with state.connect(paths.state_db(checkout)) as conn:
            stored = conn.execute("SELECT count(*) FROM tool_events").fetchone()[0]
        self.assertEqual(cap, stored, "tool event storage is unbounded")

    def test_at_cap_is_reported(self):
        from repohone import capture as capture_mod
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        checkout = identity.peek_checkout_id(self.repo)
        session = identity.session_id("claude-code", "s-1")
        cap = capture_mod.MAX_TOOL_EVENTS_PER_TURN
        for i in range(cap):
            state.append_tool_event(checkout, session, "t1", record.now(), "Read",
                                    None, f"tu{i}", cap)
        self.fire("Stop", prompt_id="t1")
        turn = self.the_record()["extensions"]["tool_events"]["turns"]["t1"]
        self.assertTrue(turn["at_cap"])


class UnattributedToolEvents(Fixture):
    """Regression: a tool call with no prompt_id folded under "", joining nothing."""

    def test_a_tool_call_without_a_turn_is_visibly_unattributed(self):
        from repohone import capture as capture_mod
        self.init_repo()
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.fire("PreToolUse", tool_name="Read",
                  tool_use_id="orphan", tool_input={"file_path": str(self.repo / "calc.py")})
        self.fire("Stop", prompt_id="t1")
        turns = self.the_record()["extensions"]["tool_events"]["turns"]
        self.assertIn(capture_mod.UNATTRIBUTED, turns)
        self.assertNotIn("", turns)


class PathScrubbing(unittest.TestCase):
    """Regression: substring replacement rewrote sibling directories."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp())
        self.repo = self.base / "a"
        self.sibling = self.base / "abc"
        self.repo.mkdir()
        self.sibling.mkdir()

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_a_sibling_directory_is_not_rewritten(self):
        from repohone.toolinput import minimize
        out = minimize({"command": f"diff {self.repo}/x.py {self.sibling}/x.py"},
                       self.repo, "redacted")
        self.assertIn("<repo>/x.py", out["command"])
        self.assertNotIn("<repo>bc", out["command"])

    def test_the_repository_root_itself_is_rewritten(self):
        from repohone.toolinput import minimize
        out = minimize({"command": f"ls {self.repo}"}, self.repo, "redacted")
        self.assertEqual("ls <repo>", out["command"])

    def test_a_relative_root_produces_no_bogus_forms(self):
        from repohone.toolinput import _variants
        self.assertEqual(set(), _variants("repo"))

    def test_a_root_followed_by_a_shell_operator_is_rewritten(self):
        """Regression: `cd <root> && …` kept the absolute path (found live)."""
        from repohone.toolinput import minimize
        for command, expected in ((f"cd {self.repo} && ls", "cd <repo> && ls"),
                                  (f'cd "{self.repo}"; ls', 'cd "<repo>"; ls'),
                                  (f"(cd {self.repo})", "(cd <repo>)")):
            self.assertEqual(expected,
                             minimize({"command": command}, self.repo, "redacted")["command"])

    def test_a_longer_name_sharing_the_prefix_is_left_alone(self):
        from repohone.toolinput import minimize
        for suffix in ("-old", ".bak", "_2", "2"):
            out = minimize({"command": f"ls {self.repo}{suffix}"}, self.repo, "redacted")
            self.assertNotIn("<repo>", out["command"], suffix)


class Lifecycle(Fixture):
    """ARCH §4, §15 — install, initialize, remove."""

    def _cli(self, *args):
        return subprocess.run([sys.executable, str(HOOK)] + list(args),
                              cwd=str(self.repo), capture_output=True, text=True, env=self.env)

    def test_init_creates_a_valid_profile(self):
        self.assertEqual(0, self._cli("init", "--path", str(self.repo)).returncode)
        self.assertEqual(profile.ACTIVE, profile.load(self.repo).state)

    def test_doctor_reports_an_uninitialized_repository(self):
        out = self._cli("doctor", "--path", str(self.repo)).stdout
        self.assertIn("UNINITIALIZED", out)

    def test_doctor_passes_after_init_and_install(self):
        self._cli("init", "--path", str(self.repo))
        self._cli("install", "--path", str(self.repo))
        proc = self._cli("doctor", "--path", str(self.repo))
        self.assertEqual(0, proc.returncode)
        self.assertNotIn("[FAIL]", proc.stdout)

    def test_doctor_fails_when_init_happened_without_install(self):
        """An active profile with no plugin captures nothing, silently."""
        self._cli("init", "--path", str(self.repo))
        proc = self._cli("doctor", "--path", str(self.repo))
        self.assertEqual(1, proc.returncode)
        self.assertIn("RepoHone is not installed on this machine", proc.stdout)

    def test_deinit_purge_removes_local_state_and_refs(self):
        self._cli("init", "--path", str(self.repo))
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.assertNotEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))
        self._cli("deinit", "--purge", "--yes", "--path", str(self.repo))
        self.assertEqual("", git(["for-each-ref", "refs/repohone/"], self.repo))
        prof = profile.load(self.repo)
        self.assertEqual(profile.ACTIVE, prof.state, "the project's profile was deleted")
        self.assertIsNone(profile.accepted(self.repo, prof))

    def test_deinit_without_confirmation_only_prints_the_plan(self):
        self._cli("init", "--path", str(self.repo))
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        refs = git(["for-each-ref", "refs/repohone/"], self.repo)

        proc = self._cli("deinit", "--purge", "--path", str(self.repo))

        self.assertEqual(3, proc.returncode)
        self.assertIn("deinitialization plan", proc.stdout)
        self.assertIn("no changes made", proc.stdout)
        self.assertEqual(profile.ACTIVE, profile.load(self.repo).state)
        self.assertEqual(refs, git(["for-each-ref", "refs/repohone/"], self.repo))

    def test_install_writes_exec_form_hooks(self):
        self.assertEqual(0, self._cli("install").returncode)
        hooks = json.loads((claude.plugin_dir() / "hooks" / "hooks.json").read_text())["hooks"]
        self.assertEqual(set(claude.HOOK_TIMEOUTS), set(hooks))
        entry = hooks["Stop"][0]["hooks"][0]
        self.assertEqual(("command", sys.executable), (entry["type"], entry["command"]))
        self.assertTrue(entry["args"])

    def test_a_finished_session_records_no_capture_errors(self):
        """Regression: session end wrote a column 1.23 had dropped, so every
        session ended with a capture error, and no test looked (found live)."""
        self._cli("init", "--path", str(self.repo))
        self.fire("SessionStart", source="startup")
        self.fire("UserPromptSubmit", prompt="go", prompt_id="t1")
        self.fire("Stop", prompt_id="t1")
        self.fire("SessionEnd", reason="prompt_input_exit")
        rec = self.the_record()
        self.assertEqual([], rec["capture"]["errors"])
        self.assertIsNotNone(rec["lifecycle"]["ended_at"])


if __name__ == "__main__":
    unittest.main()
