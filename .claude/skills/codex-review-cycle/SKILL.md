---
name: codex-review-cycle
description: A comprehensive code review cycle against a design document, before solidifying codebase for production
---

# Codex Review Cycle

Comprehensive code review and fix cycle: Codex reviews codebase against a design doc, Claude verifies findings, brainstorms fixes with dual perspectives, writes implementation plan, executes it, runs full test suite, conducts final review with meta-review, verifies completion, and commits.

**Announce at start:** "Starting Codex Review Cycle against [design doc path]."

## Arguments

`$ARGUMENTS`: Path to design document. Default: `docs/plans/2026-01-13-cognitive-resilience-model-design-part2-training-ops.md`

If `$ARGUMENTS` is empty or not provided, use the default path above.

## Cycle ID and Artifacts

At the start of Stage 1, determine the **Cycle ID**: `YYYY-MM-DD-HHmm` (date and time, e.g. `2026-02-09-1430`). Use this same Cycle ID for ALL artifacts throughout the entire cycle, even if the cycle spans multiple days.

All intermediate artifacts are saved to `docs/reviews/` for persistence across context compression and session boundaries. Create `docs/reviews/` if it does not exist.

| Stage | Artifact | Path |
|-------|----------|------|
| 1 | Raw Codex output | `docs/reviews/{CYCLE_ID}-codex-raw-output.md` |
| 2 | Verified findings | `docs/reviews/{CYCLE_ID}-codex-verified-findings.md` |
| 4 | Implementation plan | `docs/plans/{CYCLE_ID}-codex-review-fixes.md` |
| 6 | Pre-execution base SHA | Recorded in plan or noted in conversation |
| 9 | Meta-review | `docs/reviews/{CYCLE_ID}-meta-review.md` |

## The Process

### Stage 1: Run Codex Review

1. Determine the Cycle ID (`YYYY-MM-DD-HHmm`) and announce it
2. Read the prompt template at `.claude/skills/codex-review-cycle/codex-review-prompt.txt`
3. Replace `{DESIGN_DOC}` with the resolved design doc path
4. Create `.claude/skills/codex-review-cycle/tmp/` if it does not exist
5. Write the resolved prompt to `.claude/skills/codex-review-cycle/tmp/codex-review-prompt-resolved.txt`
6. Run Codex via Bash (timeout: 600000ms):

```bash
codex exec \
  -s read-only \
  -o .claude/skills/codex-review-cycle/tmp/codex-review-output.md \
  - < .claude/skills/codex-review-cycle/tmp/codex-review-prompt-resolved.txt
```

Note: Model and reasoning effort are inherited from `~/.codex/config.toml`. To override, add `-m <model>` and/or `-c model_reasoning_effort="<level>"`.

If Codex takes longer than 10 minutes, run in background via Bash and poll the output file.

7. Copy Codex output to `docs/reviews/{CYCLE_ID}-codex-raw-output.md`

**USER GATE:** None — proceeds automatically to Stage 2.

### Stage 2: Verify Claims (Receiving Code Review)

1. **FIRST ACTION — no exceptions:** Call the `Skill` tool with `skill: "superpowers:receiving-code-review"`. Do NOT read files, do NOT analyze findings, do NOT do ANY work until the Skill tool call has returned.
2. Read `docs/reviews/{CYCLE_ID}-codex-raw-output.md` (the persisted copy from Stage 1, NOT the /tmp file)
3. **Early exit check:** If Codex identified no issues, present the clean result to the user and end the cycle. No need to proceed through remaining stages.
4. Treat Codex output as **external reviewer** feedback. For EACH finding:
   - **Read the actual code** at every line number cited — confirm the code matches the claim
   - **Read the design doc** sections relevant to the finding — check if the doc specifies requirements the finding ignores (e.g., GPU strategy, data formats, operational modes)
   - **Search the design doc changelog** for documented deviations — but do NOT dismiss findings just because the changelog acknowledges them. A changelog entry explains WHY something was done; it does not mean the issue is resolved
   - **Check test coverage** — are there tests for the specific code path in question?
   - **Consider ALL operational modes** — does the finding apply to DDP/multi-GPU? Bayesian AND deterministic paths? Training AND inference? Do not assume single-GPU or single-mode unless the design doc explicitly limits scope
   - **Check downstream consumers** — if a finding is about data quality (NaN, placeholders, fallbacks), verify whether downstream code guards against bad data
   - Flag claims that are incorrect, outdated, or hallucinated
5. **Severity assessment rules:**
   - Do NOT downgrade severity based on assumptions about project scope — verify scope against the design doc
   - Do NOT dismiss findings as "well-tested" without confirming tests cover the SPECIFIC scenario (e.g., DDP + SVI, not just SVI alone)
   - Do NOT dismiss doc mismatches as "sync issues" — if the spec text contradicts the code, it misleads readers
   - Findings that are documented in the changelog but not fixed in the spec text are STILL actionable (doc fix needed)
   - Findings where the code works but lacks inline explanation will be re-flagged by future reviewers — add documentation as an action item
6. Present categorized findings to user: **Verified**, **Unverified/Questionable**, **Incorrect**
7. Save categorized findings to `docs/reviews/{CYCLE_ID}-codex-verified-findings.md`

**USER GATE:** User reviews verified findings and confirms which to address. If no verified findings warrant action, end the cycle here.

### Stage 3: Brainstorm Solutions

**FIRST ACTION — no exceptions:** Call the `Skill` tool with `skill: "superpowers:brainstorming"`. Do NOT analyze findings, do NOT propose solutions, do NOT do ANY work until the Skill tool call has returned.

For each verified finding, explore solutions from two perspectives:

1. **Senior Systems Architect**: Code quality, maintainability, DRY, integration patterns, downstream engineering impact
2. **Scientific Principal Investigator**: Research aims, model correctness, training reliability, scientific validity

For each decision point:
- Present 2-3 approaches with pros/cons
- Assess downstream impact of each choice
- Recommend an approach with reasoning

**USER GATE:** User approves design decisions.

### Stage 4: Write Implementation Plan

**FIRST ACTION — no exceptions:** Call the `Skill` tool with `skill: "superpowers:writing-plans"`. Do NOT start drafting the plan, do NOT outline tasks, do NOT do ANY work until the Skill tool call has returned.

Save plan to `docs/plans/{CYCLE_ID}-codex-review-fixes.md`.

### Stage 5: Plan Examination (Self-Audit)

Before proceeding, explicitly verify by cross-referencing `docs/reviews/{CYCLE_ID}-codex-verified-findings.md`:
- Are ALL verified findings addressed in the plan?
- Are implementation steps concrete (exact file paths, code, commands)?
- Any ambiguous steps needing clarification?
- Does the plan account for integration between fixes?
- Are test strategies defined for each fix?

Present the audit results to the user.

**USER GATE:** User approves the final plan.

### Stage 6: Execute Plan

1. Record the pre-execution base SHA: `git rev-parse HEAD` — save this for Stage 8
2. **FIRST ACTION — no exceptions:** Call the `Skill` tool with `skill: "superpowers:subagent-driven-development"` or `skill: "superpowers:executing-plans"`. Do NOT create tasks, do NOT dispatch subagents, do NOT edit any source files until the Skill tool call has returned.

Execute on the current branch — do NOT set up a git worktree. Follow the batch execution workflow: execute tasks in batches by dependency, report between batches, apply feedback, continue.

Do NOT commit during execution — all commits happen in Stage 11.

**USER GATE:** User reviews between batches as defined by the skill.

### Stage 7: Full Test Suite

Run: `uv run pytest tests/ -x -q`

If failures: fix before proceeding.

**USER GATE:** None — automated.

### Stage 8: Final Review

**FIRST ACTION — no exceptions:** Call the `Skill` tool with `skill: "superpowers:requesting-code-review"`. Do NOT dispatch a review subagent, do NOT analyze diffs, do NOT do ANY work until the Skill tool call has returned.

Review all changes from the base SHA recorded in Stage 6 to current HEAD.

**USER GATE:** None — proceeds automatically to Stage 9.

### Stage 9: Meta-Review

Conduct a meta-review by cross-referencing `docs/reviews/{CYCLE_ID}-codex-verified-findings.md`:
1. For each verified issue, confirm it was addressed in the implementation
2. Identify any findings that were deprioritized or deferred
3. Summarize: what changed, what remains, what was intentionally deferred
4. Assess overall codebase readiness for production training
5. Save meta-review to `docs/reviews/{CYCLE_ID}-meta-review.md`

**USER GATE:** User reviews and approves meta-review.

### Stage 10: Verification Before Completion

**FIRST ACTION — no exceptions:** Call the `Skill` tool with `skill: "superpowers:verification-before-completion"`. Do NOT run tests, do NOT check git status, do NOT make ANY completion claims until the Skill tool call has returned.

Then run all verification commands and confirm output before making any success claims.

**USER GATE:** None — automated verification.

### Stage 11: Commit Changes

Commit all changes from the cycle to git:

```bash
git add <all changed files from Stages 6-7>
git commit -m "+ <basic condensed description of all changes>"
```

Commit message format: `"+ {short description}"` — keep it concise, one line or two summarizing the batch of fixes.

**USER GATE:** None — automated.

## Resuming Mid-Cycle

If a session ends or context is lost during execution, resume as follows:

1. List files in `docs/reviews/` and `docs/plans/` to find the Cycle ID and determine last completed stage
2. Read the verified findings (`docs/reviews/{CYCLE_ID}-codex-verified-findings.md`) and the plan (`docs/plans/{CYCLE_ID}-codex-review-fixes.md`)
3. Check implementation progress: use `git diff` and `git log` to see what has already been changed
4. Skip to the next incomplete stage and continue from there

| If you have... | Resume from |
|----------------|-------------|
| Raw Codex output only | Stage 2 |
| Verified findings saved | Stage 3 |
| Plan written | Stage 5 (re-audit) or Stage 6 |
| Partial execution | Stage 6 (continue remaining tasks) |
| All tasks done, no test run | Stage 7 |
| Tests pass, no review | Stage 8 |

**Announce when resuming:** "Resuming Codex Review Cycle {CYCLE_ID} from Stage [N]. Reading artifacts from docs/reviews/."

## Integration

**Required skills — MUST invoke each via the `Skill` tool at the start of its stage:**
- `superpowers:receiving-code-review` — Stage 2
- `superpowers:brainstorming` — Stage 3
- `superpowers:writing-plans` — Stage 4
- `superpowers:subagent-driven-development` OR `superpowers:executing-plans` — Stage 6
- `superpowers:requesting-code-review` — Stage 8
- `superpowers:verification-before-completion` — Stage 10

**CRITICAL — Skill Invocation Rule:** Every skill listed above MUST be invoked via the `Skill` tool at the start of its respective stage. Do NOT skip invocation because:
- The skill's instructions are already in context from a previous session or system reminder
- The skill was loaded in a prior stage of the same cycle
- Context compression preserved the skill's content
- You "already know" what the skill says

The `Skill` tool call is the mandatory contract. Having instructions in memory is NOT a substitute. This rule exists because skills may have been updated, and because the invocation triggers the correct workflow state.

## Red Flags

- **Never skip skill invocation** — invoke every required skill via the `Skill` tool at the start of its stage, even if you already have the instructions in context
- Never skip Stage 2 verification — Codex can hallucinate issues
- Never implement unverified findings
- Never skip the plan self-audit (Stage 5)
- Never proceed past a USER GATE without explicit user approval
- Never run Codex with write access (`-s read-only` only)
- Never skip dual-perspective analysis in Stage 3
- Never claim completion without running verification-before-completion
- Never set up a git worktree — execute on current branch
- **Never assume project scope without reading the design doc** — e.g., do not assume "single-GPU" when the design doc specifies multi-GPU DDP. Always verify operational requirements (GPU strategy, training modes, data formats) against the design doc before downgrading or dismissing findings
- **Never dismiss a finding just because the changelog documents it** — a changelog entry explains a decision but does not eliminate the need for code fixes, doc updates, or inline comments that prevent future re-flagging
- **For every finding marked "not actionable", specify what documentation/comment change would prevent it from being flagged again** — if nothing can prevent re-flagging, the finding IS actionable