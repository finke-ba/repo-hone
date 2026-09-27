"""SessionStart announcer (ARCH §10).

An explicit model-provider egress surface. Output is minimal, sanitized and
non-evidentiary: no source, no transcript, no identity, no absolute paths.
"""
from __future__ import annotations

from typing import Optional

from . import gitcmd, profile

BUDGET_CHARS = 800

# The slash commands are user-only: the agent names them, and the user runs them.
UNINITIALIZED_TEXT = ("RepoHone is available but not initialized here.\n"
                      "Ask whether the user wants to initialize it (`/repohone:init`).")
INVALID_TEXT = ("RepoHone is installed but its project profile is not usable, so capture "
                "is suspended.\nSuggest `/repohone:doctor`.")
NOT_ACCEPTED_TEXT = ("RepoHone is set up in this project but is not capturing this user's "
                     "sessions.\nAsk whether they want it on here (`/repohone:init`) or "
                     "not to be asked again (`/repohone:deinit`).")
CHANGED_TEXT = ("This project's RepoHone profile changed since the user accepted it, so "
                "capture is paused.\nAsk whether they want to review and accept it "
                "(`/repohone:init`).")
# Where capture is on, the one thing the agent must do: an agent that only saves
# a stated rule to its own memory leaves RepoHone without it.
CAPTURING_TEXT = ("RepoHone is capturing this session. When the user states a standing "
                  "project rule, record it with the repohone-rule skill, even if you also "
                  "save it to your memory.")


def message(cwd) -> Optional[str]:
    """None outside a RepoHone repository."""
    root = gitcmd.toplevel(cwd)
    if root is None:
        return None
    prof = profile.load(root)
    if prof.state == profile.UNINITIALIZED:
        return _bounded(UNINITIALIZED_TEXT)
    if prof.state == profile.INVALID:
        return _bounded(INVALID_TEXT)
    consent = profile.accepted(root, prof)
    if consent is None:
        return _bounded(NOT_ACCEPTED_TEXT)
    if consent is False:
        return _bounded(CHANGED_TEXT)
    return _bounded(CAPTURING_TEXT)


def _bounded(text: str) -> str:
    """No repository detail needs to leave the machine to offer initialization."""
    if len(text) <= BUDGET_CHARS:
        return text
    return text[:BUDGET_CHARS - 1].rstrip() + "…"
