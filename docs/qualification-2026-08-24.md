# LUMI qualification record — 2026-08-24

This record distinguishes an infrastructure qualification from a training-signal qualification. Both bounded
one-node jobs completed the full rollout, learner, final-save, and learner-to-vLLM weight-sync path. The base
checkpoint did not produce positive rewards on these selected samples, so this is not evidence that the current
model/dataset mix is ready for a scaled active-sampling run.

## Pinned runtime

- LUMI partition: `dev-g`, one node, eight MI250X GCDs per job
- training container: `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`
- PyTorch: `2.10.0+rocm7.0`; HIP: `7.0.51831`
- Ray: `2.54.0`; vLLM: `0.22.1`
- backend commit: `3f80d37042402b8363f39c9535723b0d4cb8de54`
- checkpoint: staged `Qwen3.5-2B`
- topology: four learner GCDs, four TP=1 vLLM engines, four learner data-parallel ranks

## Results

| Job | Input | Result | Runtime evidence |
|---|---|---|---|
| `21495571` | Eight English difficulty-5 math rows from distinct subdomains | `COMPLETED 0:0` in 6m10s | 32 completions, four reward groups, learner step, final model save, post-step weight sync 0.24s |
| `21495572` | Eight code rows with sandbox-only hidden stdin/stdout tests | `COMPLETED 0:0` in 8m04s | 32 six-turn agent rollouts, six bash calls/rollout, tool failure rate 0, five submissions, learner step, final save, post-step weight sync 0.24s |

The code run exercised the `slurm_apptainer` backend through hundreds of same-node overlapping Slurm steps.
Hidden tests were uploaded only on submission. The run did not reproduce the earlier false `test.sh`-missing
error after nested `SLURM_LABELIO` was disabled.

## Signal result and interpretation

The math batch had `math_correct_rate=0`, and the five submitted code trajectories all recorded reward 0.
Both jobs therefore reported `grad_norm=0`. This is acceptable for the bounded infrastructure smoke because
zero-standard-deviation filtering is disabled there; it is not acceptable for production RLVR.

### Post-run verifier audit — 2026-08-26

The code rewards from job `21495572` cannot be interpreted as model correctness. Its rollout artifacts show
that submitted trajectories reached the deferred verifier, but the generated `test.sh` embedded literal
newlines inside the Python source string passed to `write_text`. Python raised `SyntaxError: unterminated
string literal`, so every submission was forced to reward zero before any hidden case ran. Commit `ac9bcc2`
escapes that generation boundary and adds a regression test that compiles the exact heredoc. Code task data
must be regenerated after this fix; changing only the training checkout does not repair already packed tests.

The math diagnosis remains valid: all four sampled difficulty-5 prompt groups had uniform reward zero, so
group-normalized advantages and the gradient were correctly zero. This was a dataset/checkpoint signal
problem, not a learner, optimizer, model-save, or weight-synchronization failure.

Before scaling, build a curriculum that gives the starting policy nonzero within-prompt reward variance. Good
next experiments are easier math/code strata, an SFT checkpoint with stronger tool-submission behavior, and a
small active-sampling qualification. Require reward-0 and reward-1 completions inside prompt groups, nonzero
gradient norm, bounded resampling, and a second successful weight update before using the four-node profile.

## LUMI-specific findings

- The training SIF cannot reliably start nested setuid or user-namespace Apptainer. `slurm_apptainer` executes
  the host Singularity runtime in an overlapping same-node Slurm step and binds only isolated episode paths.
- An outer `srun --label` exports `SLURM_LABELIO=1`; the nested backend must override it to `0` or exact protocol
  output becomes `0: EXISTS` and the deferred verifier misreads it.
- FLA context parallelism in this container can fail AMD Triton MLIR compilation on variable rollout shapes
  (`chunk_delta_h`: operation destroyed but still has uses). LUMI profiles therefore use
  `sequence_parallel_size: 1`; re-enable it only after a real learner-step test in a replacement runtime.
- A six-turn code horizon plus turns-remaining and final-step warnings produced five submissions. The original
  longer horizon frequently exhausted the response budget without submitting.

## Reproduction

Generate the revision-pinned samples and task sandbox exactly as shown in the top-level README, then run:

```bash
oellm-rlvr doctor --config configs/lumi-math-qwen35-2b-smoke.yaml
oellm-rlvr doctor --config configs/lumi-code-qwen35-2b-smoke.yaml
oellm-rlvr render-slurm --config configs/lumi-math-qwen35-2b-smoke.yaml --output math-hf-smoke.sbatch
oellm-rlvr render-slurm --config configs/lumi-code-qwen35-2b-smoke.yaml --output code-hf-smoke.sbatch
sbatch math-hf-smoke.sbatch
sbatch code-hf-smoke.sbatch
```

Do not infer success from the top-level exit code alone. Check the learner metric table for `weight_sync`, inspect
code rollout timings for `done=True`, verify that no environment traceback occurred, and apply the signal gates
above before increasing node count.
