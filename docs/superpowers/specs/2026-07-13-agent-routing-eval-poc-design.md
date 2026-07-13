# DeerFlow Agent Routing Evaluation POC Design

**Status:** Approved design for implementation planning  
**Date:** 2026-07-13  
**Primary goal:** Produce a credible proof of concept that measures DeerFlow skill-routing accuracy and uses an LLM judge for the smaller set of cases where process and output quality cannot be evaluated reliably with deterministic rules.

## 1. Problem

The current worktree has proven that Inspect AI can execute DeerFlow and retain a normalized trace, but its implementation effort is concentrated in a generic assertion framework rather than in a credible agent-evaluation loop.

Observed repository state at design time:

- `backend/skill_eval/assertion_engine.py` contains 329 lines and 24 assertion types.
- `backend/tests/skill_eval/test_assertion_engine.py` contains 704 lines.
- The committed demo dataset contains only two cases.
- Those cases reference `gcp-cloud-run` and `no-write-todos-in-pro`, which are not present in the current `skills/public/` tree.
- Retained Inspect logs prove basic execution and trace capture, but do not contain an aligned baseline or a statistically meaningful routing benchmark.
- One passing `skill_loaded` score was inferred from the configured candidate list even though the retained trace recorded `used=false`, so it did not prove that the skill had actually been selected.
- A real tool-using case timed out after 300 seconds, showing that runtime stability and bounded execution need to be separated from model routing quality.
- The current all-or-nothing assertion score combines infrastructure validity, route behavior, process behavior, and final-answer quality into one binary result.

The POC therefore needs to replace assertion breadth with a narrow, observable evaluation claim:

> Given a user request and two competing academic skills, which route does the real DeerFlow runtime select, how consistently does it select it, and does a small set of fully executed tasks produce a sound trace and useful output?

## 2. Goals

The POC MUST:

1. Exercise the real DeerFlow skill-discovery and loading path rather than a standalone prompt classifier.
2. Evaluate one skill boundary deeply: `systematic-literature-review` versus `academic-paper-review` versus no skill.
3. Run 20 manually labeled routing cases for three epochs each.
4. Derive the observed route from runtime evidence rather than from case-authored assertions.
5. Report a three-class confusion matrix, per-class metrics, macro metrics, infrastructure validity, and route stability.
6. Fully execute four representative cases and evaluate their observable trace, final answer, and artifacts with an LLM judge.
7. Produce retained Inspect logs plus machine-readable and human-readable summaries from one command.
8. Distinguish model routing failures, agent execution failures, and judge failures.
9. Remove the generic deterministic assertion DSL from the POC path.

## 3. Non-goals

The POC MUST NOT:

- evaluate every built-in skill;
- claim statistically significant general agent quality from four full executions;
- execute all 60 routing runs to task completion;
- build a general expression language or plugin registry for assertions;
- use an LLM judge to replace directly observable routing facts;
- inspect or expose hidden model chain-of-thought;
- build a new web dashboard;
- add production regression management, historical trend storage, or CI gating beyond the POC command's exit status;
- evaluate external search quality, arXiv availability, or sandbox performance as if they were routing decisions.

## 4. Design Principles

- **Facts before judgment.** Successful skill-file loads are deterministic runtime facts. Semantic process and output quality require judgment.
- **One benchmark, one claim.** The routing track measures route selection only.
- **Bound full execution.** Only four representative cases run through the complete skill workflow.
- **No case-level assertion lists.** Cases describe prompts and labels, not implementation-specific tool-call recipes.
- **Errors remain visible.** Infrastructure and judge failures are reported separately and cannot silently disappear from denominators or become passing scores.
- **Evidence is retained.** Every result points back to Inspect logs and stable trace evidence identifiers.
- **Real runtime, constrained surface.** Routing runs use the real DeerFlow client with exactly two available skills.
- **Clean cutover.** Obsolete assertion code and demo cases are removed instead of retained as parallel conventions.

## 5. Two-Track Architecture

```text
RoutingCase JSONL
        |
        +------------------------------+
        |                              |
        v                              v
Track A: routing benchmark       Track B: quality evaluation
20 cases x 3 epochs              4 tagged cases x 1 epoch
        |                              |
DeerFlowRoutingProbe             DeerFlow full runner
        |                              |
real DeerFlowClient stream       complete observable trace
        |                              + final answer + artifacts
RoutingObserver                       |
        |                              v
RouteObservation                 LLM quality judge
        |                              |
deterministic routing scorer     structured QualityJudgment
        |                              |
        +--------------+---------------+
                       v
                  POC aggregator
                       |
        Inspect logs + summary.json + summary.md
```

### 5.1 Track A: routing benchmark

Track A runs all 20 cases for three epochs. It exposes only the two candidate skills to the real DeerFlow runtime. It consumes real stream events and stops before long task execution once the current skill-loading tool-call batch has settled.

The track measures:

- which skill description the model inspected;
- which candidate `SKILL.md` file was successfully loaded;
- whether the runtime loaded neither skill;
- whether the model attempted to load both candidate skills in the same routing decision;
- whether the run failed before a valid routing observation could be formed.

### 5.2 Track B: complete quality evaluation

Track B selects four `quality`-tagged cases and allows the real agent to complete the task. It retains the normalized trace, final answer, and generated artifacts. A separate judge model evaluates route appropriateness, observable process quality, and output quality against shared route-level rubrics.

Track B demonstrates the end-to-end evaluation contract. It is not a broad statistical benchmark.

## 6. Data Model

### 6.1 Routing cases

```python
from typing import Literal

from pydantic import BaseModel, Field


RouteLabel = Literal[
    "systematic-literature-review",
    "academic-paper-review",
    "none",
]


class RoutingCase(BaseModel):
    id: str
    input: str
    expected_route: RouteLabel
    rationale: str
    tags: list[str] = Field(default_factory=list)
```

Field semantics:

- `id`: Stable identifier used to align epochs and report evidence.
- `input`: Unmodified user request sent to DeerFlow.
- `expected_route`: Human-reviewed benchmark label.
- `rationale`: Concise explanation used for dataset review. It is never sent to the evaluated agent or judge.
- `tags`: Dataset slicing dimensions such as `explicit`, `implicit`, `sibling-collision`, `unrelated`, and `quality`.

The POC case model intentionally has no `assertions`, `target`, `required_skills`, `candidate_skills`, or `difficulty`. Candidate skills are task-level constants; output targets belong to the quality rubric rather than to routing cases.

### 6.2 Routing evidence

```python
from typing import Literal

from pydantic import BaseModel, Field


class RouteEvidence(BaseModel):
    id: str
    kind: Literal["described", "load_requested", "loaded", "load_failed"]
    skill: str
    tool_call_id: str
    detail: str | None = None


class RouteObservation(BaseModel):
    observed_route: RouteLabel | Literal["ambiguous"] | None
    evidence: list[RouteEvidence] = Field(default_factory=list)
    completed: bool
    errors: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
```

`observed_route=None` is reserved for a run that cannot form a valid routing observation. A valid no-skill decision is normalized to the explicit label `"none"`.

### 6.3 Quality judgment

```python
from typing import Literal

from pydantic import BaseModel, Field


class QualityJudgment(BaseModel):
    recommended_route: RouteLabel
    route_quality: int
    process_quality: int
    output_quality: int
    overall_quality: int
    fatal_error: bool = False
    reasons: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
```

All numeric scores MUST be integers from 0 through 4. Evidence entries MUST reference stable identifiers in the judge payload, such as `tool_call[3]`, `tool_result[3]`, `artifact[report.md]`, or `final_answer`.

## 7. Dataset Design

The routing dataset contains exactly 20 manually reviewed cases:

| Expected route | Count | Coverage |
|---|---:|---|
| `systematic-literature-review` | 8 | Explicit SLR/survey, implicit “the literature,” annotated bibliography, cross-paper comparison, time windows, and citation formats. |
| `academic-paper-review` | 6 | One arXiv URL, one attached paper, single-study summary, methodology critique, peer review, and strengths/weaknesses. |
| `none` | 6 | Factual question, general news search, coding, translation, conceptual explanation, and a non-academic task. |

The existing `systematic-literature-review/evals/trigger_eval_set.json` is a seed, not the final benchmark. Its target-positive cases are useful, but its negative set has only one explicit sibling-skill case. The POC dataset replaces unrelated negatives as needed to achieve the approved 8/6/6 balance.

Exactly four cases carry the `quality` tag:

1. One explicit, bounded SLR request for five papers and a specified citation format.
2. One implicit but bounded multi-paper synthesis request.
3. One single-paper review against a public, stable paper source.
4. One ordinary academic concept question that should use neither candidate skill.

The bounded SLR prompts MUST specify paper count and output format so full runs do not expand to the skill's default scope unexpectedly.

### 7.1 Case authoring contract

Case authors describe user intent and the human route label. They MUST NOT encode the expected implementation as tool-call assertions.

Authoring rules:

1. Write the request as a natural user would. Autonomous-routing cases MUST NOT name a skill, request skill activation, or use a slash command.
2. Give each case one primary boundary: multi-paper synthesis, one-paper review, or neither candidate.
3. Explain the discriminating semantic feature in one concise `rationale`. Do not list expected tools, files, steps, or output substrings.
4. Use stable IDs after a case enters retained results. Material prompt or label changes create a new case ID so historical run hashes remain interpretable.
5. Use tags for analysis slices, not for hidden scoring behavior.
6. Reserve `quality` for bounded cases that are safe to execute fully.

Examples:

```json
{"id":"slr-implicit-rlhf-001","input":"What does the literature say about RLHF?","expected_route":"systematic-literature-review","rationale":"The phrase 'the literature' requests synthesis across multiple papers.","tags":["implicit","multi-paper"]}
{"id":"paper-review-arxiv-001","input":"Review this paper and assess its methodology: https://arxiv.org/abs/2310.06825","expected_route":"academic-paper-review","rationale":"A single specified paper requires depth-first review rather than a literature survey.","tags":["explicit","sibling-collision"]}
{"id":"none-concept-001","input":"Explain the difference between precision and recall.","expected_route":"none","rationale":"A direct conceptual explanation requires neither candidate academic skill.","tags":["direct-answer"]}
```

The initial suite SHOULD contain all of these case shapes:

- obvious target-skill positives;
- implicit target-skill positives without the words “systematic” or “survey”;
- sibling-skill collisions that differ mainly by one paper versus multiple papers;
- near-boundary `none` cases in the academic domain;
- a small number of unrelated `none` sanity checks.

Quality cases have additional authoring constraints:

- SLR requests specify a bounded count of three to five papers and an output format.
- Single-paper requests use a stable, publicly reachable paper source.
- `none` requests remain useful tasks rather than artificial “do nothing” prompts.
- Case-specific expectations remain prohibited; shared route rubrics own semantic process and output evaluation.

### 7.2 Label review workflow

Before a case is admitted:

1. The author writes `input`, `expected_route`, `rationale`, and tags.
2. A second reviewer independently labels the `input` without seeing the author's label or rationale.
3. Agreement admits the case after schema validation.
4. Disagreement is resolved by reviewing the two candidate skill descriptions and rewriting or removing genuinely ambiguous prompts.
5. The final rationale records the discriminating boundary, not the review discussion.

The judge's later `recommended_route` disagreement can open a label review, but it never changes the dataset automatically.

## 8. Route Observation Semantics

### 8.1 Candidate surface

Every routing run configures exactly these available skills:

```text
systematic-literature-review
academic-paper-review
```

This intentionally creates a sibling collision for academic requests while excluding unrelated skills from the POC claim.

### 8.2 Discovery versus selection

- A successful `describe_skill(name)` call records `kind="described"`.
- A candidate `read_file` call targeting its `SKILL.md` records `kind="load_requested"`.
- A successful correlated tool result records `kind="loaded"`.
- A failed correlated tool result records `kind="load_failed"`.
- `described` alone never counts as route selection.
- `load_requested` alone never counts as route selection.
- The first settled routing batch containing one successful candidate load selects that skill.
- If the same assistant tool-call batch successfully loads both candidates, the result is `ambiguous`.
- If the agent finishes a normal response without a successful candidate load, the result is `none`.
- A configuration failure, stream exception, timeout, malformed trace, or missing terminal outcome produces an infrastructure error rather than `none`.

The observer waits for all candidate skill-read calls issued in the current assistant tool-call batch to settle before stopping. This preserves the ability to detect an ambiguous same-turn decision without allowing the agent to continue into the expensive skill workflow.

### 8.3 Explicit slash activation

The 20 POC prompts do not use slash activation. Slash activation bypasses autonomous routing and therefore does not belong in the benchmark. Existing DeerFlow slash behavior remains outside this POC's scoring claim.

## 9. Deterministic Routing Scoring

Each valid epoch produces one of four observed labels:

```text
systematic-literature-review
academic-paper-review
none
ambiguous
```

Infrastructure failures are not labels.

A routing epoch passes when `observed_route == expected_route`. `ambiguous` always fails.

The report MUST contain:

- a 3-row expected-route confusion matrix with four observed columns, including `ambiguous`;
- per-class precision, recall, and F1 for the three benchmark classes;
- macro precision, macro recall, and macro F1;
- valid-run rate over all 60 planned routing runs;
- per-case stability across the three epochs;
- expected and observed labels for every failed case;
- stable evidence references for every route decision;
- separate infrastructure error details.

For metric computation:

- valid routing observations enter classification metrics;
- `ambiguous` contributes a false negative for the expected class and never a true positive;
- infrastructure failures do not enter the classification matrix, but they reduce valid-run rate;
- all three raw epoch results remain visible; the aggregator MUST NOT majority-vote them into a single hidden result.

## 10. LLM Judge Design

### 10.1 Judge responsibility

The judge evaluates semantic properties that are brittle or misleading as deterministic assertions:

- whether the selected route is appropriate for the request;
- whether observable tool use follows a coherent process;
- whether tool results are used correctly;
- whether the final answer and generated artifacts satisfy the task;
- whether errors, repetitions, unsupported claims, or trace/output contradictions materially reduce quality.

The judge does not decide the benchmark's deterministic observed route and does not replace its confusion matrix.

### 10.2 Judge inputs

The judge receives:

1. The original user request.
2. The names and descriptions of both candidate skills.
3. The actual observed route.
4. A complete normalized observable trace containing stable message, tool-call, tool-result, and error identifiers.
5. The final answer.
6. Generated artifact paths and bounded content snapshots.
7. All three shared route-level rubrics.

The judge MUST NOT receive:

- `expected_route`;
- case `rationale`;
- hidden reasoning or chain-of-thought;
- manually selected favorable trace excerpts.

The judge first selects `recommended_route` independently. The report marks disagreement between the judge recommendation and the human benchmark label as `label_review_needed`; it does not mutate the label automatically.

### 10.3 Shared rubrics

#### Systematic literature review

Assess whether the run:

- handles multi-paper scope and requested constraints coherently;
- uses a relevant and bounded retrieval process;
- synthesizes findings across papers instead of listing papers independently;
- produces internally consistent citations and requested artifacts;
- reports limitations and avoids unsupported claims.

#### Academic paper review

Assess whether the run:

- remains grounded in the specified single paper;
- identifies the paper's method, contribution, evidence, strengths, weaknesses, and limitations;
- distinguishes statements from the paper from the agent's critique;
- produces a useful and coherent review.

#### No skill

Assess whether the run:

- answers the request directly;
- avoids unnecessary skill loading and tool work;
- remains correct, relevant, and proportionate to the request.

#### Common process rubric

Assess observable execution only:

- tool choice and ordering are coherent;
- tool errors are handled rather than ignored;
- repeated or unused calls are penalized;
- final claims are supported by retrieved evidence;
- final output agrees with the retained trace and artifacts.

### 10.4 Score anchors

All four quality fields use the same anchors:

| Score | Meaning |
|---:|---|
| 0 | No evaluable result or completely wrong. |
| 1 | Severe omissions or largely unusable. |
| 2 | Partially satisfies the task with material problems. |
| 3 | Satisfies the task with sound evidence and no major defect. |
| 4 | Excellent, well-supported, efficient, and complete. |

A quality case passes only when:

- `fatal_error` is false;
- `route_quality >= 3`;
- `process_quality >= 3`;
- `output_quality >= 3`;
- the judgment cites at least one trace/tool evidence ID;
- the judgment cites `final_answer` or at least one artifact evidence ID.

### 10.5 Judge model and failure handling

The judge model is an explicit `judge_model` parameter and SHOULD differ from the evaluated agent model.

The judge MUST emit schema-valid structured JSON. If parsing fails, the harness MAY issue one repair request containing only the original judge output, the schema, and the parse error. The repair request MUST ask for format correction, not reconsideration. A second failure produces `judge_failure`.

A judge API error, invalid evidence reference, out-of-range score, or unparseable output is a judge failure. It MUST NOT silently become a zero score or a pass.

## 11. Evidence Normalization and Limits

The judge payload uses stable identifiers:

```text
message[0]
tool_call[0]
tool_result[0]
error[0]
artifact[report.md]
final_answer
```

The payload includes all observable tool calls and errors. Large results and artifacts are truncated using deterministic head-and-tail limits with original byte counts and SHA-256 digests. Truncation markers MUST remain visible. The harness MUST NOT select excerpts based on whether they make the agent look successful.

Generated artifacts are collected only from declared run output locations. Paths outside the run workspace and output directory are not read into judge context.

## 12. Reproducibility Record

Every POC run records:

- evaluated model identity;
- judge model identity;
- Inspect AI and DeerFlow versions;
- SHA-256 of both candidate `SKILL.md` files;
- SHA-256 of the routing case file;
- task arguments and epoch index;
- runtime/config identity without secrets;
- start and end timestamps;
- Inspect log path;
- summary schema version.

Comparisons between runs are valid only when these identities are visible. The report does not claim regression or improvement when relevant model, skill, case, or runtime identities are missing.

## 13. Command and Outputs

The POC exposes one command from `backend/`:

```bash
AGENT_MODEL=your-agent-model JUDGE_MODEL=your-judge-model \
  uv run python -m skill_eval.poc
```

`AGENT_MODEL` and `JUDGE_MODEL` are required explicit inputs. The command fails during preflight before creating a run when either is missing or unknown.

The command:

1. validates configuration, model names, skills, and the 20-case dataset;
2. runs the routing task for three epochs;
3. runs the four `quality` cases to completion for one epoch;
4. invokes the quality judge;
5. aggregates all results;
6. writes retained outputs.

Outputs:

```text
Inspect logs under `logs/`
`eval-results/{run_id}/summary.json`
`eval-results/{run_id}/summary.md`
```

`summary.json` is the machine-readable source for metrics and per-case results. `summary.md` is the POC demonstration surface and contains:

- run identities and hashes;
- confusion matrix;
- per-class and macro metrics;
- valid-run and stability metrics;
- failed and unstable routing cases with evidence references;
- four structured judge results;
- agent execution and judge failures;
- explicit pass/fail evaluation against acceptance criteria.

Inspect View remains the detailed trace viewer. The POC does not build another UI.

Exit codes:

- `0`: The command completed and all POC acceptance criteria passed.
- `1`: The command completed with valid evidence, but one or more quality thresholds failed.
- `2`: Configuration, runtime, aggregation, or judge infrastructure prevented a valid POC result.

## 14. Error Taxonomy

| Type | Examples | Metric treatment |
|---|---|---|
| `routing_failure` | Wrong skill, no skill when one was expected, skill when none was expected, ambiguous same-turn load. | Included in routing classification metrics. |
| `agent_execution_failure` | Stream timeout, unhandled tool error, empty final answer in a full quality run. | Reduces routing valid-run rate when applicable; quality case is a fatal failure. |
| `judge_failure` | API failure, invalid structured output, invalid evidence references. | Reported separately; invalidates that quality judgment. |
| `label_review_needed` | Judge independently recommends a route different from the human label. | Diagnostic only; label remains unchanged until human review. |

Errors include case ID, epoch, stage, concise message, and Inspect evidence location.

## 15. Implementation Shape

The implementation retains Inspect and the real DeerFlow adapter boundary, but replaces the assertion-centric POC path.

Expected core modules:

```text
backend/skill_eval/
  case_schema.py          # RoutingCase and route labels
  trace_schema.py         # Observable trace contracts retained for full quality runs
  routing.py              # RouteEvidence, RouteObservation, RoutingObserver
  judge.py                # Shared rubrics, evidence bundle, structured judge
  report.py               # Metrics, aggregation, JSON and Markdown summaries
  poc.py                  # One-command orchestration entrypoint
  dataset_loader.py       # RoutingCase JSONL to Inspect Samples
  inspect_solver.py       # Routing-probe and full-run solver wiring
  inspect_scorer.py       # Deterministic route scorer and quality judge adapter
  adapters/
    deerflow.py           # Real stream execution for probe and full modes

backend/evals/
  skills_routing_eval.py
  skills_quality_eval.py

backend/cases/
  literature_skill_routing.jsonl
```

Clean cutover removes:

- the 24-type assertion registry;
- `SkillAssertionSpec` and assertion names;
- `skill_assertion_scorer()`;
- assertion-engine-specific tests;
- mock demo cases that reference missing skills;
- mock-runner behavior that exists only to prove the obsolete assertion path.

Reusable stream conversion and tool-call/result correlation remain. Unit tests use scripted stream events rather than a second fake agent behavior convention.

## 16. Test Strategy

Tests defend observable contracts rather than source structure.

### 16.1 Routing observer

Cover:

- successful `describe_skill` followed by successful target load;
- description without load;
- load request with failed result;
- two successful candidate loads in one assistant tool-call batch;
- normal completion with no skill load;
- unrelated file reads;
- stream exception and timeout;
- evidence ordering and stable IDs.

### 16.2 Routing scorer and aggregation

Cover:

- all three correct classes;
- wrong-class predictions;
- `ambiguous` treatment;
- infrastructure failures excluded from classification but included in valid-run rate;
- confusion matrix counts;
- per-class and macro precision/recall/F1;
- three-epoch stability;
- no majority-vote loss of raw epoch data;
- exit-code selection.

### 16.3 Judge

Use a fake model boundary to cover:

- complete normalized trace and artifact payload construction;
- omission of expected label and rationale;
- deterministic truncation and hashes;
- structured result parsing and score ranges;
- valid and invalid evidence references;
- one format-only repair attempt;
- terminal judge failure;
- quality pass/fail calculation.

Tests MUST NOT assert exact natural-language judge prose.

### 16.4 End-to-end verification

Verification order:

1. Run focused unit tests for `backend/tests/skill_eval/`.
2. Run one real short case from each route class.
3. Inspect the three traces and route evidence manually.
4. Run the full POC command.
5. Verify summary metrics against raw result counts.
6. Manually compare all four judge judgments with their Inspect traces and artifacts.

## 17. Acceptance Criteria

The POC is complete when all delivery criteria below are met.

### 17.1 Harness criteria

- One command executes both tracks and writes all declared outputs.
- All 20 routing cases run for three epochs.
- All four quality cases run for one epoch.
- Every result retains model, skill, dataset, and log identities.
- Every route result contains inspectable evidence.
- All four quality attempts produce valid judge judgments; judge infrastructure failures prevent a successful POC exit.
- Focused unit tests and the real smoke sequence pass.

### 17.2 Routing quality criteria

- Valid-run rate across 60 planned routing epochs is at least 95%.
- Macro precision is at least 0.80.
- Macro recall is at least 0.80.
- Every routing failure and infrastructure failure is represented in the report.

Case stability is reported but is not a hard POC gate.

### 17.3 Full quality criteria

- At least three of the four full quality cases pass the approved judge thresholds.
- Every judgment contains valid trace/tool evidence and final-answer or artifact evidence.
- Judge/human label disagreements are surfaced for review.

A quality failure is an evaluation finding, not a harness failure, provided the evidence and judgment are valid. The command returns exit code `1` when agent quality thresholds fail and exit code `2` only when a valid evaluation could not be produced.

## 18. Risks and Mitigations

### Model nondeterminism

Mitigation: three routing epochs, raw epoch retention, and explicit stability reporting.

### Judge bias or self-confirmation

Mitigation: separate judge model, no expected label or rationale in judge input, independent route recommendation, structured evidence references, and manual review of all four POC judgments.

### Full SLR cost and latency

Mitigation: only two SLR quality cases, bounded paper counts, one quality epoch, explicit timeouts, and early stopping for all routing-only runs.

### External network instability

Mitigation: classify execution failures separately, retain tool errors, and never reinterpret network failure as routing failure. The POC report states when external failures limit quality conclusions.

### Early-stop resource leakage

Mitigation: the routing runner closes the stream after the current skill-load batch settles and verifies worker/thread completion before returning. A leaked worker is an infrastructure failure.

### Trace volume

Mitigation: retain raw Inspect logs, send only deterministic bounded evidence bundles to the judge, and include hashes and original sizes for truncated content.

### Label ambiguity

Mitigation: every case carries a human rationale, sibling cases are explicit, and judge disagreement is reported as `label_review_needed` rather than automatically changing ground truth.

## 19. Decision Summary

The POC deliberately stops being a generic assertion framework.

- Routing accuracy is measured from real DeerFlow runtime facts.
- Semantic process and output quality are assessed by an evidence-grounded LLM judge on four bounded complete runs.
- Twenty balanced cases and three epochs provide a credible routing demonstration.
- One command retains raw evidence and produces a concise report.
- The implementation removes assertion breadth that does not contribute to this claim.
