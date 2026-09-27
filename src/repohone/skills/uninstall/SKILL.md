---
name: uninstall
description: Remove RepoHone's plugin from this machine, optionally deleting all its local evidence
disable-model-invocation: true
argument-hint: "[--purge]"
allowed-tools: Bash({repohone} uninstall --dry-run:*)
---

# Uninstall RepoHone

What the developer typed after the command: `$ARGUMENTS`

1. Run `{repohone} uninstall --dry-run`, adding `--purge` only if they asked for
   all local evidence to be deleted.
2. Show the plan as printed.
3. Run `{repohone} uninstall --yes`, with `--purge` if and only if step 1 had it.
4. Show the result as printed, then one line: this session keeps its hooks until
   it ends, and installing again takes a terminal.

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
