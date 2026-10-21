# OELLM 9B mixed-difficulty math discovery + RL — 2026-10-05

Status: submitted on LUMI; **no survey, training, or evaluation result is claimed here yet**.
The campaign is under `/scratch/project_465002530/users/bmoell/oellm-rlvr/runs/oellm9b-math-discovery-20261005`.

## Question and design

Can an additional verifier-backed RL run improve the Phase B step-64 OpenEuroLLM 9B
checkpoint on frozen mathematical tests, while producing a checkpoint-specific
difficulty map for a large, mixed-difficulty set of problems?

The 10,000-problem pool is drawn from the pinned local copy of
`birgermoell/oellm-math-rlvr` **train split only**. It contains 128 unique semantic groups per
non-English EU24 language (2,944 rows) and 7,056 English rows, spanning source
difficulty labels 1–5. Previous Phase A/B train and multilingual evaluation
semantic groups are excluded. The pool builder fails if counts or disjointness
do not match. TMAX receives this pool as one file with weight 1.0; its data
loader shuffles indices from a seed, so easy/hard source labels are mixed rather
than sequenced as a curriculum. The source labels are **not** model-specific
difficulty claims.

The frozen starting checkpoint is job `22488151`, Phase B step 64. Its survey
samples eight completions per problem at temperature 1 with a 2,048-token cap.
`survey/predictions.jsonl` records **all** attempts; the complete per-problem
annotation, including 0/8 and 8/8 groups, is `survey/difficulty.csv`. This
avoids the selection bias from the trainer's active sampler. Individual 8-sample
rates are coarse bins with wide Wilson intervals, not precise rankings.

The two-node, `small-g` DAPO run uses the same mixed pool for 256 updates
(16,384 accepted trajectories at eight prompts × eight samples per update), LR
`5e-7`, full export and restart state every 64 updates. Active sampling can
overdraw or repeat prompts, so **accepted trajectories do not imply 2,048
unique trained problems**; the survey does guarantee difficulty measurements
for 10,000 distinct semantic groups if it completes.

A CPU quality gate requires 10,000 complete problem groups, 80,000 attempts,
8–80% sample accuracy, at least 20% mixed-reward groups, at most 15% length
stops, and at most 2% unbalanced think tags. Training depends on that gate
succeeding; evaluation depends on training succeeding and requires a complete
step-256 export. Paired greedy evaluation compares against the Phase B step-64
parent on the frozen 512-row multilingual procedural holdout and the 1,024-row
external-family DAPO-Math diagnostic. A gain on the former alone is not a claim
of broad math-reasoning improvement.

## Submitted jobs

| Job | Purpose | Dependency |
|---|---|---|
| `22546515` | Build and audit 10,000-problem pool on CPU | none |
| `22546519` | Frozen-checkpoint survey and difficulty annotations | afterok `22546515` |
| `22546521` | Fail-closed survey quality gate | afterok `22546519` |
| `22546599` | Two-node 256-update RL training | afterok `22546521` |
| `22546601` | Paired held-out evaluation | afterok `22546599` |

Earlier submitted jobs `22546260`, `22546267`, `22546367`, `22546370`,
`22546475`, `22546478`, and `22546487` were cancelled while still pending
while correcting the container code path and enforcing `split=train`. They
produced no data or optimizer updates.

## Interpretation and next decision

Once all jobs complete, audit pool manifests, actual survey source/language
counts, the accepted and filtered training rollouts, gradients, checkpoint
completeness, and paired evaluation. Continue past 256 updates only if held-out
math improves without material external-family regression or increased
truncation/repetition. A stronger claim requires independently sourced,
verifier-audited non-procedural evaluation; the current source remains heavily
procedural.
