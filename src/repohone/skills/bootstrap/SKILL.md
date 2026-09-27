---
name: bootstrap
description: Find candidate project rules in this repository's instructions and history
disable-model-invocation: true
argument-hint: "[--include-remote]"
allowed-tools: Bash({repohone} bootstrap), Bash({repohone} bootstrap --all)
---

# Bootstrap

What the developer typed after the command: `$ARGUMENTS`

1. Run `{repohone} bootstrap`. Show the review set as printed: these are
   candidates only, and nothing is adopted.
2. Only if they asked for pull-request and CI history (`--include-remote`): run
   `{repohone} bootstrap --include-remote`, which reads GitHub through `gh`, and
   show the result as printed.

## Rules

- Run RepoHone only as written here, ids spelled out literally: no shell
  variables, or the approval will not match.
- Do not ask in the chat before the command that needs approval. Claude Code
  shows the developer an approval prompt; that is the one approval. If they
  decline, say so and stop: never retry it or reach the same result another way.
- Show what the developer approves exactly as RepoHone printed it, in a code
  block. Summarise anything else in a sentence or two, claiming nothing the output
  does not say.
- If RepoHone picks no target, relay what it printed and ask one question.
