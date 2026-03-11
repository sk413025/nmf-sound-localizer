# Role Packet: Agent Experience and Onboarding Agent

## Objective

Evaluate how quickly and safely a fresh Codex instance can understand this branch and begin useful work.

## Required inputs

- `docs/codex-native-assessment/CODEX_CAPABILITY_BASELINE.md`
- `docs/codex-native-assessment/SHARED_RUBRIC.md`
- `AGENTS.md`
- `.codex/skills/nature-communications-submission/SKILL.md`
- `docs/codex-native-assessment/README.md`

## Focus

- onboarding path clarity
- instruction density and context cost
- discoverability of source-of-truth files
- whether this branch should expose more task-specific skills or shorter quickstarts

## Questions to answer

- What should a new Codex agent read first?
- Is `AGENTS.md` sufficient but too dense?
- Would a short `START_HERE_AGENT.md` materially improve launch speed?
- Which recurring tasks should be skill-backed instead of rediscovered each time?

## Required outputs

- a recommended onboarding path with 5 steps or fewer
- the top context bottlenecks
- recommendations for agent-facing documents or skills
- risks of agent confusion, duplication, or drift

## Failure conditions

Your report should be considered weak if it talks only about human onboarding or ignores Codex-native primitives such as `AGENTS.md` and local skills.
