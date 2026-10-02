# Experiment history

This index records **what ran**, rather than telling a new user what to submit. Each linked qualification
record carries the detailed inputs, job IDs, artifacts, and limitations. Plans and unsubmitted configurations
are listed separately. Status below is as checked on 2026-10-02; consult LUMI Slurm and the output directories
for live job status.

## OpenEuroLLM 9B math sequence

| Stage | Evidence | What it established | What it did not establish |
|---|---|---|---|
| Initial instruct-SFT math dry run and cold restart | [Five-update qualification](qualification-oellm9b-instruct-math-dapo-2026-09-28.md), jobs `22395497` and `22395748` | Rollouts, rewards, five finite learner updates, weight transfer, complete exports, and restart on LUMI | Held-out quality improvement |
| English procedural-math canary | [16-update qualification](qualification-oellm9b-instruct-math-canary-2026-09-28.md), job `22398668`, plus held-out evaluation and restart jobs | All 16 updates and a cold restart worked; frozen 1,024-prompt in-domain exact accuracy rose from 42.97% to 44.63% at step 16 | Broad math or multilingual improvement |
| Multilingual phase-A admission | [Math phase record](oellm9b-math-promotion-2026-09-28.md), jobs `22401601` and `22401602` | A nominal 80% EU / 20% English profile had useful model-specific reward variance | The later Phase A backend did **not** realize that final mix: its factors selected 2,355 EU and 51 English rows |
| First 32-update phase-A attempt | Job `22402030` | Nothing about learning: it timed out during model initialization, before an update or checkpoint | 32-update stability or any quality change |
| Repaired startup, one update | Job `22406985`, control commit `5ee2e1e` | Node-local checkpoint staging and bounded learner startup completed; 64 rollouts, one finite nonzero-gradient update, weight sync, and complete export | Sustained 32-update training or multilingual improvement; the exact original startup failure mechanism remains unproven |
| Multilingual math Phase A | Job `22432251`, 32 updates, final [experimental checkpoint](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-phase-a-32step-experimental) from the [16-step canary](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-canary-step16-experimental) | Two-node TMAX/DAPO run completed with nonzero gradients and a full step-32 export; frozen multilingual exact accuracy rose 36.33% → 38.87% over 512 paired prompts | Broad math improvement: exact McNemar p=0.0596; DAPO-Math remained near floor (4.10% → 4.59%); the nominal 80/20 configuration produced a ~97.9/2.1 transformed-pool ratio, not an 80/20 final mix |

The Phase A release cards and exact run artifacts are also tracked under
[`releases/oellm9b-math-rlvr-2026-10-02`](../releases/oellm9b-math-rlvr-2026-10-02/).
The old script for job `22402030` was rendered from a pre-fix checkout and must not be reused.
The next **proposed** step is profiling the harder Phase B components and making a separate
promotion decision; Phase B training has not run. See the [phase plan](oellm9b-math-promotion-2026-09-28.md).

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
