"""Git-native working-tree snapshots (ARCH §16, §16.1, §17).

A snapshot is the stable state of the captured file set that RepoHone observed
during one lifecycle event. It never claims to precede sibling hooks, and it is
never recorded from a tree that was still moving: instability yields `None` and
a capture error, because absent evidence is honest and wrong evidence is not.
"""
from __future__ import annotations

import hashlib
import os
import posixpath
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

from . import REF_NAMESPACE, gitcmd, paths

SNAPSHOT_TIMEOUT_S = float(os.environ.get("REPOHONE_SNAPSHOT_TIMEOUT_S", "10"))
STABILITY_ATTEMPTS = 3
MAX_SUBMODULE_DEPTH = 32

CLASS_SEGMENT = {"prompt": "prompt", "stop": "stop",
                 "stop_failure": "stop-failure", "session_end": "session-end"}

_SNAPSHOT_ENV = {
    "GIT_AUTHOR_NAME": "RepoHone", "GIT_AUTHOR_EMAIL": "repohone@localhost",
    "GIT_COMMITTER_NAME": "RepoHone", "GIT_COMMITTER_EMAIL": "repohone@localhost",
}

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class UnstableTree(RuntimeError):
    """The captured file set kept changing; no authoritative state exists."""


class IntegrityError(RuntimeError):
    """An allocated destination already holds different content (§17)."""


@dataclass
class Snapshot:
    ref: str
    cls: str
    ordinal: int
    commit: str
    tree: str
    head: Optional[str]
    branch: Optional[str]
    taken_at: str
    path: Optional[str]
    submodules: List[dict]
    errors: List[str] = field(default_factory=list)

    def as_record(self) -> dict:
        return {"ref": self.ref, "class": self.cls, "ordinal": self.ordinal,
                "commit": self.commit, "tree": self.tree, "head": self.head,
                "branch": self.branch, "taken_at": self.taken_at,
                "path": self.path, "submodules": self.submodules}


def safe_segment(value: str) -> str:
    """Ref-safe and collision-free: a rewritten id keeps a hash of the original."""
    if not value:
        return "_"
    cleaned = _UNSAFE.sub("-", value).strip("-.") or "_"
    if cleaned != value:
        cleaned += "-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return cleaned


def checkout_namespace(checkout_id: str) -> str:
    """Worktrees of one repository share a single ref store, so a ref must say
    which checkout owns it — otherwise one worktree's purge destroys another's
    evidence, and two worktrees can collide on a reused host session id."""
    return f"{REF_NAMESPACE}/{safe_segment(checkout_id)}"


def ref_for(session_id: str, cls: str, ordinal: int, turn_key: Optional[str],
            checkout_id: str) -> str:
    segment = CLASS_SEGMENT[cls]
    base = f"{checkout_namespace(checkout_id)}/{session_id}"
    if cls == "session_end":
        return f"{base}/{segment}/{ordinal}"
    return f"{base}/{safe_segment(turn_key or '_')}/{segment}/{ordinal}"


def _index_dir(checkout_id: str, repo: Path) -> Path:
    """Scratch never lives inside the repository: the index file would land in
    the captured file set and no tree could ever settle (ARCH §16.1)."""
    candidate = paths.scratch_dir(checkout_id) / "index"
    try:
        candidate.resolve().relative_to(Path(repo).resolve())
    except ValueError:
        return paths.ensure(candidate)
    return paths.ensure(paths.checkout_dir(checkout_id) / "tmp" / "index")


def _scratch_index(checkout_id: str, session_id: str, repo: Path) -> Path:
    """Per session, not per repo: concurrent sessions sharing one GIT_INDEX_FILE
    collide on index.lock and lose snapshots. The path is hashed because a deep
    repo path overflows NAME_MAX. The real index is never touched."""
    key = hashlib.sha256(str(Path(repo).resolve()).encode("utf-8")).hexdigest()[:12]
    return _index_dir(checkout_id, repo) / f"{safe_segment(session_id)}-{key}.idx"


def _write_tree_once(repo: Path, index: Path, deadline: float) -> str:
    env = {"GIT_INDEX_FILE": str(index)}
    # The scratch index is seeded from the real one so staged and sparse-checkout
    # state is preserved.  ``assume-unchanged`` is only a local performance hint,
    # though: inheriting it makes ``git add -A`` silently retain an old blob while
    # the working file contains different bytes.  Clear it in scratch only.
    flagged = gitcmd.index_flagged_paths(repo, env)
    assumed = [path for path in flagged if _is_assume_unchanged(repo, index, path)]
    for start in range(0, len(assumed), 100):
        gitcmd.run(["update-index", "--no-assume-unchanged", "--",
                    *assumed[start:start + 100]], repo, env=env,
                   timeout=max(0.5, deadline - time.monotonic()))
    # Sparse entries absent from disk must keep skip-worktree or they would be
    # captured as deletions. A present path can contain a hidden local edit, so
    # clear the bit only for those entries in the scratch index.
    assumed_set = set(assumed)
    skipped = [path for path in flagged
               if path not in assumed_set and (repo / path).exists()]
    for start in range(0, len(skipped), 100):
        gitcmd.run(["update-index", "--no-skip-worktree", "--",
                    *skipped[start:start + 100]], repo, env=env,
                   timeout=max(0.5, deadline - time.monotonic()))
    remaining = max(0.5, deadline - time.monotonic())
    gitcmd.run(["add", "-A", "--", "."], repo, env=env, timeout=remaining)
    remaining = max(0.5, deadline - time.monotonic())
    return gitcmd.run(["write-tree"], repo, env=env, timeout=remaining)


def _is_assume_unchanged(repo: Path, index: Path, path: str) -> bool:
    """Distinguish assume-unchanged from skip-worktree in the scratch index."""
    env = {"GIT_INDEX_FILE": str(index)}
    out = gitcmd.run(["ls-files", "-v", "--", path], repo, env=env)
    return bool(out and out[0].islower())


def stable_tree(repo: Path, index: Path, deadline: float) -> Tuple[str, int]:
    """Two staging passes must agree. `add -A` re-stats the whole captured file
    set, so a file still being written yields a different tree and we retry."""
    previous = None
    for attempt in range(1, STABILITY_ATTEMPTS + 1):
        tree = _write_tree_once(repo, index, deadline)
        if previous == tree:
            return tree, attempt
        previous = tree
        if time.monotonic() > deadline:
            break
    raise UnstableTree(f"captured file set did not settle in {STABILITY_ATTEMPTS} passes")


def _seed_index(repo: Path, index: Path, refresh: bool = False) -> None:
    if index.exists() and not refresh:
        return
    try:
        gd = gitcmd.git_dir(repo)
        if gd is None:
            return
        real = Path(gd) / "index"          # a worktree's .git is a file, not a dir
        if real.exists():
            # The real index is replaced atomically by Git. Copy it through an
            # atomic rename too, so concurrent captures never observe a partial
            # refresh of their scratch index.
            index.parent.mkdir(parents=True, exist_ok=True)
            handle, staged = tempfile.mkstemp(dir=str(index.parent),
                                               prefix=index.name + ".")
            try:
                with os.fdopen(handle, "wb") as output:
                    output.write(real.read_bytes())
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(staged, index)
            except BaseException:
                try:
                    os.unlink(staged)
                except OSError:
                    pass
                raise
        elif refresh:
            index.unlink(missing_ok=True)
    except OSError:
        if refresh:
            raise


def _path_version(path: Path):
    """A monotonic-enough token for Git control files.

    HEAD can move A -> B -> A while a snapshot is being staged. Comparing only
    its final value misses that ABA transition, so include the filesystem
    versions of the worktree HEAD, index and HEAD reflog as well as their
    semantic values. A false positive only loses a snapshot; a false negative
    would manufacture historical evidence.
    """
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _git_identity(repo: Path):
    """Read one self-consistent Git identity and its transition token."""
    git_dir = gitcmd.git_dir(repo)
    if git_dir is None:
        raise UnstableTree("Git worktree identity is unavailable")
    git_dir = Path(git_dir)
    watched = (git_dir / "HEAD", git_dir / "index", git_dir / "logs" / "HEAD")
    for _attempt in range(STABILITY_ATTEMPTS):
        before = tuple(_path_version(path) for path in watched)
        head = gitcmd.head(repo)
        branch = gitcmd.branch(repo)
        after = tuple(_path_version(path) for path in watched)
        if before == after:
            return head, branch, after
    raise UnstableTree("Git identity changed while it was being read")


def _commit(repo: Path, tree: str, head: Optional[str], message: str,
            deadline: float) -> str:
    args = ["commit-tree", tree]
    if head:
        args += ["-p", head]
    args += ["-m", message]
    remaining = max(0.5, deadline - time.monotonic())
    return gitcmd.run(args, repo, env=_SNAPSHOT_ENV, timeout=remaining)


def _pin(repo: Path, ref: str, commit: str) -> None:
    existing = gitcmd.run(["rev-parse", "--verify", "--quiet", ref], repo, check=False)
    if existing and existing != commit:
        raise IntegrityError(f"{ref} already points at {existing}, refusing to replace")
    if existing == commit:
        return
    object_format = gitcmd.run(["rev-parse", "--show-object-format"], repo,
                               check=False) or "sha1"
    zero = "0" * (64 if object_format == "sha256" else 40)
    try:
        # Create-only. A second capture racing this one must never replace the
        # first capture after both observed an absent ref.
        gitcmd.run(["update-ref", ref, commit, zero], repo)
    except gitcmd.GitError as exc:
        winner = gitcmd.run(["rev-parse", "--verify", "--quiet", ref], repo,
                            check=False)
        if winner and winner != commit:
            raise IntegrityError(
                f"{ref} was concurrently assigned to {winner}") from exc
        if winner != commit:
            raise


def _gitlinks(repo: Path, tree: str) -> Dict[str, str]:
    """Every gitlink in the accepted root tree, keyed by its literal path."""
    out = gitcmd.run(["ls-tree", "-r", "-z", tree], repo, check=False)
    found: Dict[str, str] = {}
    for entry in out.split("\0"):
        if not entry or "\t" not in entry:
            continue
        header, name = entry.split("\t", 1)
        parts = header.split()
        if len(parts) == 3 and parts[0] == "160000" and parts[1] == "commit":
            found[name] = parts[2]
    return found


def _declared_submodule_paths(repo: Path, tree: str) -> List[str]:
    """Read .gitmodules from *tree*, never from a later working-tree state."""
    out = gitcmd.run(
        ["config", "--blob", f"{tree}:.gitmodules", "--get-regexp",
         r"^submodule\..*\.path$"], repo, check=False)
    found = []
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            found.append(parts[1].strip())
    return found


def _contained_submodule(repo: Path, relative: str) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve one portable in-repository submodule path without following links."""
    if (not relative or "\x00" in relative or "\n" in relative or "\r" in relative
            or "\\" in relative or PurePosixPath(relative).is_absolute()
            or posixpath.normpath(relative) != relative):
        return None, f"submodule path {relative!r} is not a normalized relative path"
    parts = PurePosixPath(relative).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        return None, f"submodule path {relative!r} leaves the repository"
    root = repo.resolve()
    target = root.joinpath(*parts)
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return None, f"submodule path {relative!r} passes through a symbolic link"
    try:
        target.resolve(strict=False).relative_to(root)
    except ValueError:
        return None, f"submodule path {relative!r} leaves the repository"
    return target, None


def _submodule_checkouts(repo: Path, tree: str) -> Tuple[List[Tuple[str, Path]], List[str]]:
    """Initialized child checkouts that are members of the accepted tree."""
    links = _gitlinks(repo, tree)
    declared = _declared_submodule_paths(repo, tree)
    found: List[Tuple[str, Path]] = []
    errors: List[str] = []
    seen = set()
    for relative in declared:
        if relative in seen:
            errors.append(f"duplicate submodule path {relative!r} in .gitmodules")
            continue
        seen.add(relative)
        target, problem = _contained_submodule(repo, relative)
        if problem:
            errors.append(problem)
            continue
        if relative not in links:
            errors.append(
                f".gitmodules path {relative!r} is not a gitlink in the captured tree")
            continue
        assert target is not None
        marker = target / ".git"
        if not marker.exists():
            continue                 # declared but not initialized
        if marker.is_symlink():
            errors.append(f"submodule {relative!r} has a symbolic .git entry")
            continue
        top = gitcmd.toplevel(target)
        if top is None or top.resolve() != target.resolve():
            errors.append(f"submodule {relative!r} is not its own Git working tree")
            continue
        found.append((relative, target))
    for relative in sorted(set(links) - seen):
        errors.append(
            f"gitlink {relative!r} has no matching path in the captured .gitmodules")
    return found, errors


def _common_git_dir(repo: Path) -> Optional[Path]:
    value = gitcmd.run(
        ["rev-parse", "--path-format=absolute", "--git-common-dir"], repo,
        check=False)
    return Path(value).resolve() if value else None


def _module_stores(repo: Path) -> List[Tuple[str, Path]]:
    """Git stores retained for current or historical submodule checkouts."""
    common = _common_git_dir(repo)
    modules = common / "modules" if common else None
    if modules is None or not modules.is_dir():
        return []
    root = repo.resolve()
    stores: List[Tuple[str, Path]] = []
    seen = set()
    for config in modules.rglob("config"):
        git_dir = config.parent.resolve()
        if git_dir in seen or not (git_dir / "HEAD").exists() \
                or not (git_dir / "objects").is_dir():
            continue
        seen.add(git_dir)
        worktree = gitcmd.run(
            [f"--git-dir={git_dir}", "config", "--path", "--get", "core.worktree"],
            repo, check=False)
        label = f"<module:{git_dir.relative_to(modules)}>"
        if worktree:
            candidate = (git_dir / worktree).resolve()
            try:
                label = candidate.relative_to(root).as_posix()
            except ValueError:
                pass
        stores.append((label, git_dir))
    return stores


def refs_by_store(repo_root, checkout_id: str) -> Dict[str, set]:
    """Snapshot refs in the root and every retained submodule Git store."""
    repo = Path(repo_root)
    namespace = checkout_namespace(checkout_id)
    stores: List[Tuple[str, Optional[Path]]] = [("", None), *_module_stores(repo)]
    found: Dict[str, set] = {}
    for label, git_dir in stores:
        args = ([f"--git-dir={git_dir}"] if git_dir else [])
        listed = gitcmd.run(
            [*args, "for-each-ref", "--format=%(refname)", namespace + "/"],
            repo, check=False)
        found[label] = {line.strip() for line in listed.splitlines() if line.strip()}
    return found


def purge_refs(repo_root, checkout_id: str,
               historical: Iterable[dict] = ()) -> tuple:
    """Remove this checkout's refs from root, retained modules and old records.

    Historical exact refs also clean evidence written by older versions through
    a now-missing or invalid submodule path. They are deleted only when the ref
    still points to the commit recorded by RepoHone.
    """
    repo = Path(repo_root)
    namespace = checkout_namespace(checkout_id)
    removed, failures = 0, []
    removed_keys = set()

    for item in historical:
        relative, ref, commit = item.get("path"), item.get("ref"), item.get("commit")
        if not isinstance(relative, str) or not isinstance(ref, str) \
                or not isinstance(commit, str) or not ref.startswith(namespace + "/"):
            continue
        target = repo / relative
        if gitcmd.toplevel(target) is None:
            continue
        current = gitcmd.run(["rev-parse", "--verify", "--quiet", ref], target,
                             check=False)
        if not current:
            continue
        if current != commit:
            failures.append(
                f"{relative}: {ref} points at {current}, expected recorded {commit}")
            continue
        try:
            gitcmd.run(["update-ref", "-d", ref, commit], target)
            removed += 1
            removed_keys.add((str(gitcmd.git_dir(target)), ref))
        except (gitcmd.GitError, OSError) as exc:
            failures.append(f"{relative}: {ref}: {exc}")

    stores: List[Tuple[str, Optional[Path]]] = [(".", None), *_module_stores(repo)]
    for label, git_dir in stores:
        prefix = [f"--git-dir={git_dir}"] if git_dir else []
        listed = gitcmd.run(
            [*prefix, "for-each-ref", "--format=%(refname)", namespace + "/"],
            repo, check=False)
        for ref in (line.strip() for line in listed.splitlines()):
            if not ref:
                continue
            try:
                commit = gitcmd.run([*prefix, "rev-parse", "--verify", ref], repo)
                key = (str(git_dir or gitcmd.git_dir(repo)), ref)
                gitcmd.run([*prefix, "update-ref", "-d", ref, commit], repo)
                if key not in removed_keys:
                    removed += 1
                    removed_keys.add(key)
            except (gitcmd.GitError, OSError) as exc:
                failures.append(f"{label}: {ref}: {exc}")
        left = gitcmd.run(
            [*prefix, "for-each-ref", "--format=%(refname)", namespace + "/"],
            repo, check=False)
        if left.strip():
            failures.append(f"{label}: snapshot refs remain after purge")
    return removed, failures


@dataclass
class _Candidate:
    repo: Path
    tree: str
    head: Optional[str]
    branch: Optional[str]
    identity: tuple
    path: Optional[str]
    children: List[_Candidate] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    has_initialized_children: bool = False


def _candidate_signature(candidate: _Candidate):
    # A parent `git add` may refresh a submodule's real index stat cache while
    # observing it, replacing the index inode without changing repository
    # state. HEAD and its reflog remain transition counters for A -> B -> A
    # checkouts; the captured tree covers index and working-file content.
    transition = (candidate.identity[0], candidate.identity[2])
    return (candidate.tree, candidate.head, candidate.branch, transition,
            candidate.has_initialized_children,
            candidate.path, tuple(candidate.errors),
            tuple(_candidate_signature(child) for child in candidate.children))


def _collect_candidate(repo: Path, checkout_id: str, session_id: str,
                       deadline: float, depth: int = 0,
                       relative: Optional[str] = None,
                       ancestors=frozenset()) -> _Candidate:
    git_dir = gitcmd.git_dir(repo)
    if git_dir is None:
        raise UnstableTree("Git worktree identity is unavailable")
    resolved_git_dir = Path(git_dir).resolve()
    if resolved_git_dir in ancestors:
        raise UnstableTree(f"submodule cycle reaches {relative or '.'}")

    before = _git_identity(repo)
    index = _scratch_index(checkout_id, session_id, repo)
    _seed_index(repo, index, refresh=True)
    tree, _tree_attempts = stable_tree(repo, index, deadline)
    after = _git_identity(repo)
    if before != after:
        raise UnstableTree(f"Git identity changed while capturing {relative or '.'}")

    candidate = _Candidate(repo, tree, before[0], before[1], before[2], relative)
    children, errors = _submodule_checkouts(repo, tree)
    candidate.has_initialized_children = bool(children)
    candidate.errors.extend(errors)
    if depth >= MAX_SUBMODULE_DEPTH:
        if children:
            candidate.errors.append(
                f"submodule depth exceeds supported limit {MAX_SUBMODULE_DEPTH}")
        return candidate

    next_ancestors = frozenset(set(ancestors) | {resolved_git_dir})
    for name, child_repo in children:
        child_path = name if relative is None else f"{relative}/{name}"
        try:
            child = _collect_candidate(
                child_repo, checkout_id, session_id, deadline, depth + 1,
                child_path, next_ancestors)
            candidate.children.append(child)
        except (UnstableTree, IntegrityError, gitcmd.GitError, OSError) as exc:
            candidate.errors.append(
                f"submodule {child_path}: {type(exc).__name__}: {exc}")
    return candidate


def _materialise_candidate(candidate: _Candidate, ref: str, cls: str,
                           ordinal: int, taken_at: str, deadline: float
                           ) -> Tuple[Snapshot, List[Tuple[Path, Snapshot]]]:
    commit = _commit(candidate.repo, candidate.tree, candidate.head,
                     f"repohone {cls}/{ordinal}", deadline)
    children: List[dict] = []
    errors = list(candidate.errors)
    descendant_planned: List[Tuple[Path, Snapshot]] = []
    for child_candidate in candidate.children:
        child, child_entries = _materialise_candidate(
            child_candidate, ref, cls, ordinal, taken_at, deadline)
        children.append(child.as_record())
        errors.extend(child.errors)
        descendant_planned.extend(child_entries)
    snap = Snapshot(ref=ref, cls=cls, ordinal=ordinal, commit=commit,
                    tree=candidate.tree, head=candidate.head,
                    branch=candidate.branch, taken_at=taken_at,
                    path=candidate.path, submodules=children, errors=errors)
    planned = [(candidate.repo, snap), *descendant_planned]
    return snap, planned


def _pin_hierarchy(planned: List[Tuple[Path, Snapshot]]) -> None:
    if not planned:
        return
    root_snapshot = planned[0][1]
    added: List[Tuple[Path, str, str]] = []
    failed_paths: List[str] = []

    def below_failed(path: Optional[str]) -> bool:
        return bool(path and any(path == failed or path.startswith(failed + "/")
                                 for failed in failed_paths))

    def prune(records: List[dict], path: str) -> None:
        for child in list(records):
            if child.get("path") == path:
                records.remove(child)
                return
            prune(child.get("submodules") or [], path)

    try:
        for position, (repo, snap) in enumerate(planned):
            if below_failed(snap.path):
                continue
            existing = gitcmd.run(
                ["rev-parse", "--verify", "--quiet", snap.ref], repo, check=False)
            try:
                _pin(repo, snap.ref, snap.commit)
            except (IntegrityError, gitcmd.GitError, OSError) as exc:
                if position == 0:
                    raise
                assert snap.path is not None
                failed_paths.append(snap.path)
                prune(root_snapshot.submodules, snap.path)
                root_snapshot.errors.append(
                    f"submodule {snap.path}: {type(exc).__name__}: {exc}")
                continue
            if not existing:
                added.append((repo, snap.ref, snap.commit))
    except BaseException:
        for repo, ref, commit in reversed(added):
            gitcmd.run(["update-ref", "-d", ref, commit], repo, check=False)
        raise


def capture(repo_root, checkout_id: str, session_id: str, cls: str, ordinal: int,
            turn_key: Optional[str], taken_at: str,
            _depth: int = 0, _rel: Optional[str] = None) -> Snapshot:
    """Raises UnstableTree or IntegrityError; the caller degrades the event."""
    repo = Path(repo_root)
    deadline = time.monotonic() + SNAPSHOT_TIMEOUT_S
    ref = ref_for(session_id, cls, ordinal, turn_key, checkout_id)

    previous = None
    accepted = None
    for _attempt in range(STABILITY_ATTEMPTS):
        try:
            current = _collect_candidate(
                repo, checkout_id, session_id, deadline, _depth, _rel)
        except (UnstableTree, gitcmd.GitError, OSError):
            previous = None
            if time.monotonic() > deadline:
                break
            continue
        if not current.has_initialized_children:
            # stable_tree already performs two whole-file-set passes and the
            # Git identity brackets them. A second hierarchy pass is needed
            # only when independently mutable child repositories are included.
            accepted = current
            break
        if previous is not None and _candidate_signature(previous) == \
                _candidate_signature(current):
            accepted = current
            break
        previous = current
        if time.monotonic() > deadline:
            break
    if accepted is None:
        raise UnstableTree(
            f"repository hierarchy did not settle in {STABILITY_ATTEMPTS} attempts")
    snap, planned = _materialise_candidate(
        accepted, ref, cls, ordinal, taken_at, deadline)
    _pin_hierarchy(planned)
    return snap
