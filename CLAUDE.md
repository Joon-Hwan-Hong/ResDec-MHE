# Project Guidelines for Claude

## Critical: Design Fidelity Rules

### NEVER Modify Designs Without Explicit Approval

When implementing from design documents (`docs/plans/*.md`):

1. **No Unauthorized Removals**: Never remove components, parameters, or features from the documented design without explicit user approval
2. **No Unauthorized Additions**: Never add components, layers, or features not in the original design without explicit user approval
3. **No "Simplifications"**: Do not simplify designs unilaterally - what appears unnecessary may have important purposes

### When Presenting Designs

1. **Match Original Exactly**: Present designs that match the documented specifications exactly
2. **Flag ALL Deviations**: If you believe a change is beneficial, explicitly flag it:
   ```
   DEVIATION FROM DESIGN DOC:
   - Original: [what the doc says]
   - Proposed: [what you're suggesting]
   - Rationale: [why you think this is better]
   - REQUIRES APPROVAL before implementation
   ```
3. **Ask, Don't Assume**: When in doubt, ask the user rather than making assumptions

### During Implementation

1. **Follow TDD**: Write tests first, implement to pass tests
2. **Verify Against Design**: Before marking complete, verify implementation matches design doc
3. **Document Deviations**: If implementation must deviate from design (e.g., API incompatibility), document why and get approval

## Tool Usage Rules

### File Reading
- **Always use the `Read` tool** to examine file contents. Never use `cat`, `head`, `tail`, `sed`, `awk`, or other shell commands to read files. This applies to both the main agent and all subagents.
- Use `Grep` for content search, `Glob` for file discovery. Never use shell `grep`, `find`, or `rg` via Bash.

### Testing Strategy
- **Only run tests for affected modules.** After implementing changes to specific files, run only the test files that cover those modules (e.g., `uv run pytest tests/unit/analysis/test_ccc_importance.py -x -q`).
- **Do NOT run the full test suite (`pytest tests/`)** after every batch or task. The full suite takes ~6 minutes and most modules are unaffected by any given change.
- **Run the full suite exactly once** at the very end of all implementation work, as a final regression check before declaring done.
- When dispatching subagents, instruct them to run only the relevant test files, not the entire suite.

## Planning Workflow

### BANNED: Native Plan Mode Tools

**NEVER use `EnterPlanMode` or `ExitPlanMode`.** These are Claude Code's built-in planning tools and they MUST NOT be used in this project. They conflict with the superpowers planning workflow, display stale/wrong plan content, and cause confusion.

When planning is needed:
1. Use `superpowers:brainstorming` to resolve design decisions
2. Use `superpowers:writing-plans` to create the implementation plan
3. Save plans to `docs/plans/YYYY-MM-DD-<feature-name>.md` using the `Write` tool
4. Present the plan to the user directly in chat for approval — do NOT use `ExitPlanMode`
5. After user approves, proceed with `superpowers:subagent-driven-development`

If you find yourself reaching for `EnterPlanMode` or `ExitPlanMode`, STOP. Use the superpowers workflow instead.

### REQUIRED: Invoke Skills via `Skill` Tool

**Every superpowers skill MUST be invoked via the `Skill` tool before ANY work begins for that stage.** This applies to ALL skills: `brainstorming`, `writing-plans`, `subagent-driven-development`, `executing-plans`, `receiving-code-review`, `verification-before-completion`, `requesting-code-review`, etc.

- The `Skill` tool call MUST be the literal first action — before reading files, before analyzing, before running commands
- Do NOT skip invocation because the skill content is already in context (e.g., from a previous session or system reminder)
- Do NOT "announce" using a skill without actually calling the `Skill` tool
- Do NOT do the skill's work and then invoke the skill retroactively — invoke FIRST, work SECOND
- The `Skill` tool call is the contract — context in memory is not a substitute

## Project Structure

- Design documents: `docs/plans/`
- Source code: `src/`
- Tests: `tests/unit/`, `tests/integration/`

## Key Design Documents

- Architecture: `docs/plans/2026-01-13-cognitive-resilience-model-design-part1-architecture.md`
- Training/Ops: `docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md`
- RegionHandler: `docs/plans/2026-01-27-region-handler-design.md`
