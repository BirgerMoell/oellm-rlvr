# Reasoning-RL reference experiment on LUMI

This experiment answers two questions in order:

1. Can `oellm-rlvr` optimize the real `openeurollm/oellm-9b-256k-sft` checkpoint on non-agentic,
   single-turn reasoning tasks without corrupting answer form or ordinary reasoning behavior?
2. Is the resulting evidence strong enough to keep this stack as the reasoning-RL reference while the main
   agentic path is built with SkyRL and Harbor?

It deliberately does not mix math and code in one first run. The existing math and sandboxed-code
qualifications show that both rollout paths execute; a mixed curriculum would make a small before/after signal
hard to attribute. Run the GSM8K reference first, then a separate code reference, and only then test a scheduled
mixture.

## Interpretation warning

GSM8K is a diagnostic for this checkpoint, not a clean estimate of unseen generalization. The published SFT
model card records 49,498 rows from a source named `open_math_2_gsm8k_50k`, and it does not document a final
post-mixture benchmark-decontamination pass. The official GSM8K test split remains disjoint from the rows used
by this RL run, but it may not be disjoint from earlier SFT data. Report the paired change as an optimization
and pipeline signal, never as a new state-of-the-art or uncontaminated benchmark score.

## Fixed protocol

- Parent checkpoint: `openeurollm/oellm-9b-256k-sft`, evaluated from the immutable local snapshot.
- Dataset: `openai/gsm8k`, configuration `main`, revision
  `740312add88f781978c0658806c59bc2815b9866`.
- RL data: all 7,473 official train rows. No assistant answer or reference rationale is retained in the learner
  parquet.
- Evaluation data: 64 seeded official test rows reserved for prompt/decoding calibration and the remaining
  1,255 untouched rows for the primary paired comparison. Neither split enters training.
- Prompt: the `concise` profile requests at most four short calculation lines, no `<think>` tags or repeated
  checking, and one final numeric `\boxed{...}` line. The rejected `natural` calibration remains available for
  reproducing the parent model's verbosity failure.
- Optimizer: DPPO, LR `1e-6`, 10 updates of 64 accepted trajectories, eight samples per prompt, active sampling.
- Compute: two LUMI-G nodes, eight learner GCDs and eight TP=1 vLLM rollout GCDs.
- Primary comparison: greedy decoding on the 1,255-row primary split, native tokenizer chat template, maximum
  1,024 new tokens.
- Qualitative comparison: 24 blinded paired traces stratified across improvement, regression, both-correct, and
  both-wrong transitions.

Ten updates are a bounded signal experiment, not a converged RL recipe. If it passes, the next run should
extend the same immutable protocol to 50–100 updates and add a clean held-out or newly authored task family.

### Parent calibration result

The fixed 64-row calibration split showed that simply increasing the generation budget does not cure the
parent's looping behavior:

| Prompt | Max tokens | Accuracy | Box present | Length stop | High repetition |
|---|---:|---:|---:|---:|---:|
| natural | 512 | 25.0% | 28.1% | 71.9% | 35.9% |
| natural | 1,024 | 40.6% | 54.7% | 45.3% | 60.9% |
| concise | 512 | 32.8% | 56.3% | 43.8% | 18.8% |
| concise | 1,024 | 39.1% | 76.6% | 21.9% | 35.9% |
| concise | 2,048 | 39.1% | 78.1% | 20.3% | 35.9% |

Use concise/1,024: doubling to 2,048 produced no accuracy gain and only a 1.6-point length-stop reduction.
All 64 concise/2,048 outputs used `<think>` tags despite the instruction, so tag use is reported rather than
silently treated as compliance. The parent's 21.9% concise/1,024 length-stop rate is an observed starting
condition. The training-rollout gate is therefore 30%; the paired candidate must not increase the primary
length-stop or repetition rates by more than five points.

## 1. Prepare the pinned data

Download the two Parquet files on a networked login node and record their checksums. The revision must appear
in both URLs:

```bash
REV=740312add88f781978c0658806c59bc2815b9866
ROOT=/scratch/project_465002530/users/bmoell/oellm-rlvr
mkdir -p "$ROOT/data/sources/gsm8k-$REV" "$ROOT/data/gsm8k-main-740312ad"
curl -L "https://huggingface.co/datasets/openai/gsm8k/resolve/$REV/main/train-00000-of-00001.parquet" \
  -o "$ROOT/data/sources/gsm8k-$REV/train.parquet"
curl -L "https://huggingface.co/datasets/openai/gsm8k/resolve/$REV/main/test-00000-of-00001.parquet" \
  -o "$ROOT/data/sources/gsm8k-$REV/test.parquet"

singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl \
  /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif \
  /scratch/project_465002530/users/bmoell/venvs/oellm-rlvr/bin/python -m oellm_rlvr.cli prepare-gsm8k \
  --train-source "$ROOT/data/sources/gsm8k-$REV/train.parquet" \
  --test-source "$ROOT/data/sources/gsm8k-$REV/test.parquet" \
  --output-dir "$ROOT/data/gsm8k-main-740312ad" --revision "$REV" --prompt-style concise
```

`manifest.json` records both source and generated SHA-256 values, row counts, the prompt protocol, and the
zero-overlap checks. `test-calibration.parquet` contains the 64 rows allowed for protocol tuning;
`test-primary.parquet` contains the other 1,255 rows and is the only primary comparison artifact. Do not use
any test parquet in an RL config. Test files contain reference rationales for offline audit; the training file
intentionally does not.

## 2. Run a bounded evaluator smoke

Before allocating two training nodes, prove that the exact model, tokenizer, test artifact, native chat
template, vLLM build, answer extraction, and result writer work together:

```bash
MODEL=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts/models/oellm-9b-256k-sft
CALIBRATION=/scratch/project_465002530/users/bmoell/oellm-rlvr/data/gsm8k-main-740312ad/test-calibration.parquet
sbatch --export=ALL,MODEL="$MODEL",TOKENIZER="$MODEL",DATASET="$CALIBRATION",TAG=gsm8k-parent-calibration \
  scripts/lumi_reasoning_eval.sbatch
```

This is a seeded diagnostic subset. Require 64 prediction rows, a terminal summary, finite metrics, no load
error, and a low length-stop/high-repetition rate before continuing.

## 3. Run the paired parent baseline

Use the untouched primary artifact. Keep this exact command's decoding fields for the candidate:

```bash
PRIMARY=/scratch/project_465002530/users/bmoell/oellm-rlvr/data/gsm8k-main-740312ad/test-primary.parquet
sbatch --export=ALL,MODEL="$MODEL",TOKENIZER="$MODEL",DATASET="$PRIMARY",TAG=gsm8k-parent-primary,MAX_NEW_TOKENS=1024,MAX_MODEL_LEN=3072 \
  scripts/lumi_reasoning_eval.sbatch
```

The evaluator appends one record at a time and resumes an interrupted run by `(id, sample_index)`. Re-submit
the identical command to resume. Never reuse a tag for another checkpoint or protocol.

## 4. Run ten online RL updates

```bash
CONFIG=configs/lumi-reasoning-gsm8k-oellm9b-10step.yaml
oellm-rlvr validate --config "$CONFIG"
oellm-rlvr topology --config "$CONFIG"
oellm-rlvr doctor --config "$CONFIG"
oellm-rlvr render-slurm --config "$CONFIG" --output reasoning-gsm8k-10step.sbatch
sbatch reasoning-gsm8k-10step.sbatch
```

The run requests at most six node-hours / 48 GCD-hours, but billing is actual runtime. It saves restart state
at steps 5 and 10 and a full Hugging Face model at step 10. Active sampling is important: a relatively strong
parent may yield all-correct or all-wrong eight-sample groups, which have no policy-gradient signal.

## 5. Evaluate the immutable step-10 export

Resolve the one directory containing `.checkpoint_complete`; do not point evaluation at the mutable parent
output directory:

```bash
CANDIDATE=$(find /scratch/project_465002530/users/bmoell/oellm-rlvr/outputs/reasoning-gsm8k-oellm9b-10step \
  -type f -name .checkpoint_complete -printf '%h\n' | sort | tail -1)
test -r "$CANDIDATE/model.safetensors"
sbatch --export=ALL,MODEL="$CANDIDATE",TOKENIZER="$CANDIDATE",DATASET="$PRIMARY",TAG=gsm8k-candidate-primary,MAX_NEW_TOKENS=1024,MAX_MODEL_LEN=3072 \
  scripts/lumi_reasoning_eval.sbatch
```

Then compare exact paired samples and create the blinded reasoning audit:

```bash
ROOT=/scratch/project_465002530/users/bmoell/oellm-rlvr
oellm-rlvr compare-reasoning \
  --baseline "$ROOT/evals/gsm8k-parent-primary/predictions.jsonl" \
  --candidate "$ROOT/evals/gsm8k-candidate-primary/predictions.jsonl" \
  --output "$ROOT/evals/gsm8k-comparison.json"
oellm-rlvr make-reasoning-audit \
  --baseline "$ROOT/evals/gsm8k-parent-primary/predictions.jsonl" \
  --candidate "$ROOT/evals/gsm8k-candidate-primary/predictions.jsonl" \
  --output "$ROOT/evals/gsm8k-reasoning-audit.md" --count 24
```

The automatic report contains accuracy, pass@k/majority metrics, paired wrong→right and right→wrong counts,
a paired-bootstrap 95% interval, exact McNemar test, boxed-answer form, structural-reasoning presence, any and
unbalanced `<think>` tags, reference-marker leakage, repetition, length stops, and response length. Structural flags do not
establish that reasoning is logically sound. Rate the blinded audit before opening its `.key.json` mapping.

## Decision table

| Outcome | Requirement | Decision |
|---|---|---|
| Infrastructure pass | 10 finite optimizer steps; at least 8 nonzero gradients; all weight syncs complete; policy lag ≤4; final model and restart state readable | The LUMI reasoning pipeline works |
| Reward-signal pass | Mixed rewards in accepted groups; zero-std ≤80%; training-rollout truncation ≤30%; verifier/system errors ≤2% | The data/verifier produces trainable signal |
| Strong model signal | Accuracy delta >0 and paired-bootstrap lower bound >0 | Extend to 50–100 updates |
| Weak but safe signal | Point estimate ≥0, interval crosses 0, and no form/reasoning regression | Repeat with more updates or a higher-powered clean holdout |
| No-go | Significant accuracy loss, >5-point form/structure loss, >5-point rise in repetition/length stops, or blinded logic/clarity regression | Stop; inspect reward/prompt/KL and do not scale |

Also run a protocol-matched MATH-500 and IFEval parent/candidate retention slice through the existing
OpenEuroLLM post-training evaluation control plane. Those checks should stay outside this repository's rollout
engine; this repository retains the exact RL artifacts and the close-coupled GSM8K diagnostic.

## Relationship to the agentic track

This is the reasoning reference proposed in the post-training discussion: single-turn, no tools, verifiable
numeric reward. The primary project goal remains agentic RL. After this result is archived:

1. reproduce a small code-agent run with the existing Apptainer/hidden-test path;
2. define one trajectory interface between `oellm-rlvr` artifacts and Harbor tasks;
3. qualify SkyRL + Harbor on LUMI with a small prepared sandbox set, including restart and offline operation;
4. compare throughput, failure taxonomy, reward integrity, and checkpoint quality before choosing the long-run
   agentic stack.

Do not make Harbor integration a condition for this GSM8K result. The point of the reference is to isolate the
learner/rollout/verifier loop before multi-turn environment failures enter the system.
