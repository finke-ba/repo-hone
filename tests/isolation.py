"""Imported by every test module before RepoHone: install writes Claude Code's
configuration, and a test must never reach the developer's real one. Tests may
themselves run inside a Claude Code session, whose id must not leak in."""
import os
import tempfile

os.environ["CLAUDE_CONFIG_DIR"] = tempfile.mkdtemp(prefix="repohone-claude-config-")
os.environ.pop("CLAUDE_CODE_SESSION_ID", None)
