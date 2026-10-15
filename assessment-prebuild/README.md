# Principal AI Engineer assessment: pre-build pack

Prepared for Paul B. Isaac's on 14 October 2026.

This is a design and implementation handoff, not a completed assessment submission. No provider calls, executable task corpus, trained adapters, measured evaluation, drill output or video have been created. All costs in the design are planning assumptions. Provider model IDs, prices, availability and complete endpoint terms must be verified during implementation.

## Contents

- `design.pdf`: the three-page pre-build design.
- `design.md`: editable source; page markers correspond to the PDF.
- `JETBRAINS_BUILD_BRIEF.md`: phased implementation instructions, evidence requirements and failure drills.
- `sources.md`: checked sources, what they support, and unresolved provider eligibility.

## Start here

Read the design, make it your own, then freeze it as the repository's first commit before writing assessment code. Use the build brief in a new JetBrains Codex project. The brief asks the implementing agent to inspect the actual workstation before selecting GPU dependencies and to report missing credentials without printing them. It does not authorise spending on new services or sending messages.

The intended final demo command is `./run.sh demo --offline`; it does not exist in this pre-build pack. The deliverable must eventually contain real recordings, a real trained checkpoint, the base weights, dependency/runtime assets and a working command. A command that prints canned reports will not satisfy the brief.

## Schedule

Deadline: Saturday 17 October 2026, end of day UK time (BST, UTC+1).

| Date | Work | Exit condition |
| --- | --- | --- |
| Wed 14 October | Review/freeze design; verify tools and provider rights | Design-only first commit; actual feasibility known |
| Thu 15 October | Task/oracle corpus; sandbox; durable fleet runner | Three real providers recorded; crash/failover/budget drills work |
| Fri 16 October | Curate; train actual adapters; evaluate | Reloadable weights and honest gate report |
| Sat 17 October | Fresh offline rehearsal; serving; video; submission | Complete reviewable package; target completion by 18:00 BST |

Protect time for the fresh-machine offline rehearsal and for explaining the code cold. An honest NO SHIP is a valid result when the data do not establish an improvement.
