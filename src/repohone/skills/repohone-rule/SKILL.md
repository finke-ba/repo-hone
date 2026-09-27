---
name: repohone-rule
description: Use whenever the developer states a standing project rule or convention — "we always X", "never do Y here", "that's a standing rule for this project" — including when you also save it to your own memory. Records it as a rule candidate bound to this session and turn. Local only; makes no model call and changes nothing.
---

# Marking a project rule

When a correction carries a **standing rule** rather than a one-off preference,
record it by running:

```bash
{repohone} rule "<the rule in the developer's own words>"
```

RepoHone already captured what happened; the command attaches the developer's
own words to it so the two can be reviewed together later.

## When to run it

Run it when the developer states something that should hold **beyond the current
task**. Saving the rule to your own memory does not replace this: your memory is
yours, and this records it for the project.

- "we always use the logger here, never print()"
- "handlers must not import persistence directly"
- "never commit without running make check"

## When not to run it

- A one-off preference for this task: "make this one synchronous".
- A factual correction with no rule behind it: "that file is `api.py`, not `apis.py`".
- The developer merely changing their mind about what they want built.
- Anything you inferred. The rule has to be something they said.

If you are unsure, ask: *"Is that a standing rule for this project, or just for
this change?"* One question is cheaper than a wrong candidate.

## What it does and does not do

```
rule  → a candidate is stored locally, bound to this session and turn
      → no model call and no change to the project
      → it IS consent to analyse this one candidate later
      → it is NOT consent to analyse anything else, and never
        consent to change the project
```

Under the default `interactive` egress policy the developer still approves the
moment evidence is sent, so marking does not start an analysis by itself.

It is **not** consent to change the project. Do not follow it with edits to
`CLAUDE.md`, `AGENTS.md`, lint config, or anything else. If the developer
wants the rule enforced, that is a separate, explicit decision they make
themselves.

## The session and turn

The command learns which session it runs in from the host and which turn is in
progress from RepoHone's own record, so you pass neither. If it says it could
not identify them, tell the developer the rule was not recorded. Do not retry
with a session or turn you chose: a guessed one files the rule against the
wrong evidence.

## The statement

Quote the rule in the developer's own words, not your paraphrase.

## After running it

Say one line — that you have noted it as a project rule, which
`/repohone:diagnose` can analyse — and carry on with the task. Do not summarise
the command's output or explain RepoHone.
