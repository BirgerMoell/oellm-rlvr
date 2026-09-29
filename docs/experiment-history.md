# Experiment history

This index records **what ran**, rather than telling a new user what to submit. Each linked qualification
record carries the detailed inputs, job IDs, artifacts, and limitations. Plans and unsubmitted configurations
are listed separately. Status below is as checked on 2026-09-29; consult LUMI Slurm and the output directories
for live job status.

## OpenEuroLLM 9B math sequence

| Stage | Evidence | What it established | What it did not establish |
|---|---|---|---|
| Initial instruct-SFT math dry run and cold restart | [Five-update qualification](qualification-oellm9b-instruct-math-dapo-2026-09-28.md), jobs `22395497` and `22395748` | Rollouts, rewards, five finite learner updates, weight transfer, complete exports, and restart on LUMI | Held-out quality improvement |
| English procedural-math canary | [16-update qualification](qualification-oellm9b-instruct-math-canary-2026-09-28.md), job `22398668`, plus held-out evaluation and restart jobs | All 16 updates and a cold restart worked; frozen 1,024-prompt in-domain exact accuracy rose from 42.97% to 44.63% at step 16 | Broad math or multilingual improvement |
| Multilingual phase-A admission | [Math phase record](oellm9b-math-promotion-2026-09-28.md), jobs `22401601` and `22401602` | The proposed 80% EU / 20% English data mix had useful model-specific reward variance | A trained multilingual checkpoint |
| First 32-update phase-A attempt | Job `22402030` | Nothing about learning: it timed out during model initialization, before an update or checkpoint | 32-update stability or any quality change |
| Repaired startup, one update | Job `22406985`, control commit `5ee2e1e` | Node-local checkpoint staging and bounded learner startup completed; 64 rollouts, one finite nonzero-gradient update, weight sync, and complete export | Sustained 32-update training or multilingual improvement; the exact original startup failure mechanism remains unproven |

The next **proposed** step is a fresh 32-update phase A from the tested control revision, with new output paths
and a newly rendered Slurm script. At its boundary, evaluate the saved checkpoints against the frozen
multilingual and external-family math sets before considering phase B. The old script for job `22402030` was
rendered from a pre-fix checkout and must not be reused. See the [phase plan](oellm9b-math-promotion-2026-09-28.md)
for the training mix and evaluation gates. This is a proposal, not a completed result.

## Other reference experiments

- [GSM8K reasoning reference](qualification-gsm8k-reasoning-2026-09-05.md): a completed 32-update
  infrastructure and reasoning experiment. Its outcome is diagnostic for that checkpoint and dataset; it
  does not stand in for clean external reasoning evaluation.
- [Qwen3.5 multilingual reasoning](qualification-qwen35-eu-reasoning-2026-09-14.md) and
  [multilingual math](qualification-qwen35-eu-math-2026-09-14.md): control-model qualifications and data
  profiling, not OpenEuroLLM 9B results.
- [Harbor agentic rollout](qualification-harbor-agentic-2026-09-07.md) and
  [SkyRL–Harbor training](qualification-harbor-training-2026-09-08.md): evidence for the separate agentic
  path, not a completion of the 9B multilingual math phase.
- [Earlier LUMI qualifications](qualification-2026-08-24.md): smoke tests, weight transfer, and math/code
  reward-signal investigations.

The [advanced reference](advanced-run-reference.md) preserves older command examples and the original
configuration catalogue. It is historical material; adapt and revalidate before using it for a new run.
