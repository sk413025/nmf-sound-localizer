# Codex Capability Baseline

Last updated: 2026-03-11

This file records the Codex-native baseline that all assessment agents must use before making recommendations about this repository.

## Why this baseline exists

`Codex-native` should not be defined by repository taste alone. It must be grounded in:

- current official Codex product surfaces
- current local CLI behavior
- current local feature availability
- current repository-level Codex integrations such as `AGENTS.md` and project-local skills

## Official product references

- Codex product page: `https://openai.com/codex/`
- Codex cloud documentation: `https://developers.openai.com/codex/cloud`
- Docs MCP guide: `https://developers.openai.com/resources/docs-mcp`

These references establish that Codex is a coding agent with support for persistent repository instructions, multi-agent workflows, tooling surfaces, and reusable task specialization patterns.

## Local CLI snapshot

The local `codex --help` snapshot for this repository shows these directly relevant command surfaces:

- `exec`
- `review`
- `apply`
- `resume`
- `fork`
- `mcp`
- `mcp-server`
- `app`
- `app-server`
- `cloud`
- `features`

Implication: this repository can be organized around reusable command entrypoints, resumable workflows, and agent handoff patterns instead of ad hoc terminal instructions.

## Local feature snapshot

The local `codex features list` snapshot showed the following relevant states on 2026-03-11:

- `multi_agent = true`
- `shell_tool = true`
- `unified_exec = true`
- `shell_snapshot = true`
- `personality = true`
- `memories = false`
- `plugins = false`

Implication: recommendations should assume strong support for shell execution and multi-agent work, but should not assume long-term memory or plugin-based integrations.

## Repository-level Codex primitives already present

This repository already uses several Codex-native primitives:

- `AGENTS.md` as persistent repository guidance
- a project-local Nature Communications skill under `.codex/skills/`
- deterministic paper scripts under `scripts/paper/`
- manuscript and figure workspaces with explicit source files

Implication: the assessment should prefer extending these existing primitives over inventing a separate parallel workflow.

## What Codex-native means for this branch

For this repository, a Codex-native branch should have:

- clear persistent instructions for manuscript work
- concise onboarding for fresh agents
- deterministic commands for manuscript health checks
- machine-readable or at least consistently structured evidence flow
- task boundaries that multiple agents can own without stepping on each other
- direct linkage between claims, figures, provenance, and submission outputs

## What should not be assumed

Agents must not assume:

- hidden long-term memory across sessions
- automatic understanding of local manuscript conventions without reading `AGENTS.md`
- plugin or MCP integrations that are not explicitly configured
- that figure generation alone is the main target of the branch

## Derived evaluation questions

Every specialist report should answer the subset of these questions that matches its role:

- Is `AGENTS.md` sufficient for a fresh Codex instance to begin useful manuscript work safely?
- Should part of the manuscript workflow be encoded as a project-local skill?
- Are the current paper commands deterministic enough for autonomous validation?
- Are manuscript claims traceable enough for evidence-grounded agent writing?
- Can work be split cleanly across multiple agents without manuscript drift?
- Does the branch favor manuscript-first collaboration over figure-first convenience?

## Preflight gate

No specialist assessment is valid unless it explicitly cites this baseline and uses it to justify recommendations.
