# Architecture

## Control plane versus training backend

`oellm-rlvr` is the reproducible layer around the pinned `OpenEuroLLM/tmax-reproduction` backend. The backend owns the model-specific learner, vLLM actors, DPPO/GRPO losses, Ray placement groups, environment pool, and native weight-transfer implementation. This repository owns configuration, scheduling, data/verifier contracts, launch safety, and health policy.

Keeping that boundary prevents an AMD fork and an NVIDIA fork from drifting. PyTorch still exposes ROCm devices through `torch.cuda`; vLLM's NCCL weight-transfer API reaches RCCL in the LUMI image. The CUDA profile uses the same argv and Ray topology with `singularity exec --nv` instead of `--rocm`.

## Allocation and actor placement

A Slurm allocation starts one Ray process per node. Ray then places two disjoint resource sets:

- learner bundles from `training.learner_gpus_per_node`;
- rollout bundles from `rollout.engines × rollout.tensor_parallel_size`.

The validator requires their sum to fit inside `nodes × gpus_per_node`. It also verifies that the rollout batch is large enough for the learner's sequence/data-parallel layout. Spare GPUs are allowed, but accidental oversubscription is not.

For the one-node smoke, Ray schedules four learner GCDs and four TP=1 vLLM engines. The four-node production example follows the TMAX split: two eight-GCD learner bundles plus sixteen one-GCD rollout engines.

## Online loop

1. The streaming loader samples prompts and puts requests on Ray queues.
2. vLLM actors generate multiple completions per prompt from policy version `p`.
3. Math verifiers compare extracted answers with ground truth. Code actors execute multi-turn bash calls in task sandboxes and run hidden tests on submission.
4. Active sampling removes all-equal groups that carry no relative-policy signal.
5. The learner computes centered advantages and a DPPO/GRPO update.
6. Native vLLM weight transfer broadcasts the new learner weights without serializing a checkpoint.
7. New rollouts record their model step, making policy lag observable.

`async_steps` and `inflight_updates` overlap generation/verification with learning. They improve utilization but make policy-lag limits essential.

## Failure domains

- Slurm owns node lifetime and terminates the whole allocation on a failed driver.
- The job trap terminates the background Ray step.
- Environment reset failures can become reward zero (`SWERL_RESET_FAILURE_ZERO_REWARD=1`) but are separately counted as errors.
- Code commands and test suites have independent timeouts.
- Backend revision, container, visible GPU count, accelerator type, imports, and native weight-transfer support are checked before training.
- The health gate stops promotion when reward groups are constant, outputs truncate, verifiers fail, or rollout policies lag too far behind.

## Trajectory contract

`TrajectoryRecord` stores the task and policy identities, completion, verifier result, token usage, entropy, and learner version. The append-only JSONL store is intentionally simple and recoverable. Large runs should write one shard per actor/node and compact to Parquet after the job; multiple workers should not contend on one shared file.
