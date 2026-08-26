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

The initial math interpretation was also incomplete. A later difficulty-2 probe (`21536420`) again reported
four reward-zero groups and `grad_norm=0`, but its decoded artifact contains valid answers that the backend
rejected. All eight completions for the linear-equation prompt end in `\boxed{27}`, and several fraction
completions end in the exact `-73/20`. The published Parquet represents each answer as a singleton list;
the backend's multi-verifier transform then wrapped that list again and passed `["27"]`, rather than `"27"`,
to `MathVerifier`. Commit `99613f0` flattens the singleton at sampling time and rejects ambiguous
multi-answer rows.

The repaired-verifier code probe (`21536421`) showed a separate visibility error. Six trajectories invoked
the now-valid hidden-test runner and none reproduced the generated-heredoc syntax failure, but 26 of 32
trajectories never submitted. Their decoded messages show that the policy received only a generic request to
solve an unnamed coding task. The task archive retained `instruction.md`, while the backend intentionally
copied only environment seeds into the sandbox. Commit `99613f0` puts the complete problem statement and
`/workspace/solution.py` contract in the policy-visible message.

### Easier-stratum signal probes — 2026-08-26

Job `21536744` replayed the fixed scalar-answer contract on English difficulty-2 math and completed `0:0` in
6m44s. The verifier now behaved correctly and accepted 16 of 32 trajectories. The gradient was still zero:
the four eight-sample groups were internally constant (`8/8`, `0/8`, `8/8`, `0/8`), so grouped DPPO correctly
assigned every trajectory advantage zero. This demonstrates why an aggregate 50% accuracy is not enough for
RLVR; variation must occur among samples for the same prompt.

The first one-row fraction canary attempt (`21537331`) failed before model initialization because TMAX requires
eight dataset rows to fill four prompt slots across two asynchronous steps. The v4 sampler therefore repeats
the one calibrated row eight times with explicit copy indices. Job `21537536` then completed all 64 rollouts,
but incorrectly assigned reward zero to every one. Decoding and replaying the exact artifact through the
pinned verifier showed 56 equivalent answers and eight wrong answers. The root cause was the verifier's
symbolic-equivalence timeout: `MathVerifier` runs in an executor thread, while `signal.signal` only works on
the main thread; the caught exception silently became a false negative. Commit `b238d5a` makes that timeout
thread-aware and rejects extracted expressions over 512 characters before un-timed worker-thread parsing.

Job `21537886` reran the same v4 fraction canary at commit `b238d5a` and completed `0:0` in 6m35s. Every
16-sample prompt group contained 14 rewards of 1 and two rewards of 0. The artifact reports mean reward
`0.875`, zero-std fraction `0`, and advantages from `-0.875` to `0.125`; the learner reports
`grad_norm=1.24`, no truncations, and a 0.24s post-update weight sync. This qualifies the math reward-to-update
path and confirms that the earlier zero gradient was a verifier/threading failure, not a lack of correct
model responses.

Job `21536745` used regenerated English difficulty-1 code tasks with the actual problem in the model-visible
message. It completed `0:0` in 23m14s and produced the first verified code learning signal: two of 32
trajectories submitted passing solutions, one in each of two eight-sample prompt groups. The rollout artifact
records rewards `{0, 1}`, advantages from `-0.125` to `0.875`, and the learner reports `grad_norm=0.21` followed
by a 0.24s weight sync. No environment timeout occurred. Thirty trajectories did not submit, two completions
were length-truncated, and tool-format failures occurred in eight trajectories, so commit `a5a2869` adds a
concise system prompt naming the editable path and submission marker and reduces the signal profile's
per-turn budget from 2,048 to 1,024 tokens. This last change targets rollout efficiency; it is not needed to
interpret the already successful gradient-path result.

Both the difficulty-1 code probe and calibrated difficulty-2 math canary now pass the narrow grouped-signal
test. Difficulty should only be changed after verifier schema and policy-visible task contracts have been
replayed against saved trajectories.

### Active-sampling and second-update qualification — 2026-08-26

Job `21538347` ran the committed `lumi-math-qwen35-2b-active-sampling.yaml` profile from commit `64f33c9`
with active sampling enabled, a two-step asynchronous window, filtered-rollout retention, and a 30-minute
Slurm limit. It completed `0:0` in 4m59s and saved a complete final 2B model after 128 episodes.

Both 64-episode learner steps retained four mixed reward groups and produced nonzero gradients. Step 1 had
mean reward `0.890625`, advantages from `-0.9375` to `0.125`, `grad_norm=1.02`, and a 0.24s online weight
sync. Step 2 used the synchronized policy at `model_step=1`, had mean reward `0.875`, advantages from
`-0.875` to `0.125`, and `grad_norm=1.18`. No rollout was stale or truncated. The combined 128-row artifact
has mean reward `0.8828125`, eight of eight mixed prompt groups, zero zero-standard-deviation groups, and a
nonzero-advantage fraction of 1.0.

No prompt needed filtering or replenishment in this calibrated run because every generated group already had
reward variance. This qualifies active-sampling mode, the inter-step learner-to-vLLM update, and a second
successful optimizer update, but it does not yet exercise the filtered-prompt replenishment branch. The next
scaling gate remains the runbook's two-node weight-sync qualification. In parallel, rerun the code signal
profile with the concise control prompt before selecting a larger code curriculum.

### OELLM 9B real-checkpoint qualification — 2026-08-26

The staged `openeurollm/oellm-9b-256k-sft` artifact was checked against Hugging Face revision
`aa328efb891af0174b634af2704252eccda2154a`. It is a dense 9,101,947,904-parameter BF16
`Qwen3ForCausalLM` checkpoint (16.95 GiB of safetensors) with a 262,144-token configured context. The
qualification deliberately used a 3,072-token training pack and 2,048-token response cap; it did not attempt
to exercise the model's maximum context.

The first online job, `21539700`, used the published difficulty-5 sample with active sampling. The full
infrastructure path succeeded: one vLLM copy loaded on the rollout node, eight ZeRO-3 learner ranks formed on
the other node, and the two-rank cross-node weight-transfer group initialized over Libfabric in 4.24 seconds.
The selected curriculum was nevertheless unusable for this checkpoint: the first 128 inspected completions
all had reward zero, and the filtered artifact grew to 136 rows before cancellation. Active sampling correctly
filtered every zero-standard-deviation group and continued replenishing, so the job was canceled after 8m36s
instead of spending the full allocation without an optimizer update.

Commit `a0b00ee` adds `make-math-calibration` and the bounded
`lumi-math-oellm9b-256k-sft-ladder-2node.yaml` profile. The generated 16-row integer ladder has SHA-256
`6bafb197a5796b354ccfdb015a3be1656920471f5e3be69da8c9cba676cb424e`. It repeats eight arithmetic
levels across two asynchronous prompt slots, disables active resampling, and retains constant groups so a
calibration job is guaranteed to terminate.

Job `21540106` ran that profile from the exact public commit and completed `0:0` in 23m28s. LUMI allocated
two nodes (16 MI250X GCDs); the topology used eight learner GCDs plus one TP=1 rollout GCD and left seven
GCDs reserved. Cold parallel checkpoint reads spent roughly 12 minutes in filesystem I/O before recovering.
After that delay, vLLM loaded 17.02 GiB in 14.36 seconds, the eight-rank learner communicator completed, and
the trainer-to-rollout communicator completed across the nodes in 4.51 seconds.

Both 64-episode DPPO updates had real grouped reward signal and nonzero gradients:

| Step | Correct | Mean reward | Mixed groups | Advantage range | Gradient norm |
|---|---:|---:|---:|---:|---:|
| 1 | 48/64 | 0.750000 | 7/8 | -0.875 to 0.750 | 0.34 |
| 2 | 49/64 | 0.765625 | 7/8 | -0.875 to 0.375 | 0.40 |

The combined rollout artifact contains 128 trajectories, mean reward `0.7578125`, 14/16 mixed prompt
groups, a zero-standard-deviation fraction of `0.125`, and nonzero advantages on 112/128 trajectories. All
128 generations stopped normally; none truncated or timed out. Step 1 recorded a 1.38-second online weight
sync, no stale result was dropped, and the final model was saved as an 18,203,942,400-byte safetensors file
with `.checkpoint_complete`. This qualifies real 9B checkpoint loading, two-node native weight transfer,
math rollout verification, two nonzero-gradient learner updates, and final full-model save. It does not
qualify 256k-context training or the wider one-trainer-plus-eight-engine communicator.

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
