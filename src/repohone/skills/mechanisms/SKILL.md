---
name: mechanisms
description: What checks and tools this repository already has, and optionally what each costs to run
disable-model-invocation: true
argument-hint: "[--probe [name]]"
allowed-tools: Bash({repohone} mechanisms), Bash({repohone} mechanisms --probe)
---

# Mechanisms

What the developer typed after the command: `$ARGUMENTS`

1. Run `{repohone} mechanisms`. Summarise what it found in a sentence or two.
2. Only if they asked to measure (`--probe`): run `{repohone} mechanisms --probe`,
   which lists what it would run and changes nothing, and show that as printed.
   Then run `{repohone} mechanisms --probe --only <name>` for a name they gave, or
   `{repohone} mechanisms --probe --yes` for all of them, with a Bash timeout of
   600000 ms. Show the result as printed.

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
