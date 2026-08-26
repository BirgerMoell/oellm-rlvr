# Data and verifier contracts

## Math

The backend-compatible minimum row is:

```json
{"messages":[{"role":"user","content":"Compute 2 + 3."}],"ground_truth":"5","dataset":"math"}
```

`make-math-smoke` emits JSONL or Parquet based on the output suffix. Production math data should preserve source/license/provenance outside the training columns, pin a revision, remove evaluation contamination, and test every verifier on adversarial completions before training.

`sample-math` selects distinct semantic groups from the published OpenEuroLLM Math RLVR Parquet shard
and preserves all provenance and verifier columns. The policy still receives only `messages`. Use
`--min-difficulty` and `--max-difficulty` to select an inclusive curriculum band, and `--diverse-by` to
avoid filling a probe with one subdomain. Use `--subdomain` for an exact subdomain match when replaying a
known calibration prompt; do not mistake that narrow canary for a production data mixture.
`--copies` can repeat each selected row to meet the backend's prefill/async minimum for a controlled canary;
it must not be used to manufacture diversity in a production curriculum.

The online transform supports several verifiers per row by wrapping a scalar `ground_truth` and scalar
`dataset` into aligned lists. A published singleton such as `["5"]` must therefore be flattened to `"5"`
before training; otherwise the transform produces `[["5"]]` and sends a list, rather than a string, to
`MathVerifier`. `sample-math` performs this normalization and rejects multi-answer rows until they have an
explicit aligned multi-verifier contract.

The lightweight `MathVerifier` extracts the last `\\boxed{...}`, then `<final>...</final>`, then an explicit answer line, then the last nonempty line. It supports normalized exact strings and bounded rational arithmetic without calling `eval`. The online backend uses its own pinned ground-truth verifier implementation; the lightweight verifier is for dataset QA and regression tests.

## Code

`pack-code` accepts one manifest or a YAML list of manifests. Each manifest has an ID, instruction, task SIF, visible seed files, hidden test files, and step limit. It writes:

- `train.parquet`, including `messages`, `tools`, and the exact TMAX `env_config` shape;
- `task-data/<id>/...` for a directly mounted run;
- `task-data.tar.gz` for a Hugging Face dataset repository.

The flat per-sample environment shape is:

```json
{
  "max_steps": 8,
  "env_name": "swerl_vanillux_sandbox",
  "task_id": "add-two-integers",
  "image": "/shared/containers/python-3.12.sif"
}
```

`dataset` is the Open-Instruct verifier dispatch key, not a provenance label. Math samples use `math` and
code sandbox samples use `passthrough`; the original dataset identity is retained in `oellm_source_dataset`.

At load time, the pinned backend merges this with the run-level environment config and passes `task_id`,
`image`, and `max_steps` to the Ray environment actor. LUMI smoke profiles use `slurm_apptainer`, which asks
slurmd to launch host Singularity from the containerized Ray actor. Prepared Apptainer remains available for
prebuilt production task states, and CUDA systems can use Docker or Apptainer.

## Test design

Tests must be deterministic, bounded, and resistant to hard-coded answers. Use multiple public/hidden cases, property checks where practical, isolated temporary state, and explicit timeouts. Do not use network access or external services. Treat syntax/test failures as valid reward zero; label sandbox launch, missing image, missing test file, timeout, OOM, and parser failures as infrastructure errors as well as reward zero.

Binary reward is the safest initial contract. Introduce partial reward only when every component has a precise interpretation and cannot be farmed without solving the task. Version verifier code and store the version with every trajectory.

`sample-code` accepts the published OpenEuroLLM Code RLVR Parquet shard. It converts each Python
`stdin_stdout` hidden test into `cases.json` plus a bounded sandbox runner. The generated training row
contains only the policy-visible messages and TMAX environment configuration; hidden tests never enter
the model context. Code sampling supports the same inclusive difficulty band and can diversify by
`generator_family`. The generated verifier is compiled in the unit tests; this specifically guards the
shell-heredoc escaping boundary, where Python string escapes must survive one generation layer.

The complete problem statement and output contract are part of the policy-visible message. The backend
copies environment seeds to `/workspace` but deliberately does not expose `instruction.md` or hidden tests;
storing the problem only in the task-data archive leaves the agent with no task to solve.

The committed code profiles also override the backend's generic agent prompt with
`prompts/code-agent-system.txt`. It tells the policy the editable path and submission marker, bounds the
thought and shell-command size, and discourages unproductive filesystem and package exploration. Relative
`task.system_prompt_file` values are resolved against the configuration file, so rendered Slurm commands keep
working regardless of the caller's current directory. The signal profile uses a 1,024-token per-turn budget:
this is long enough for a compact implementation but short enough to reduce malformed, truncated JSON tool
calls and make a six-turn rollout practical on LUMI.

## Local verifier QA

The standalone `CodeVerifier` can execute a `TaskSpec` against an Apptainer image. A local subprocess runner exists only for trusted repository tests and requires an explicit unsafe flag. This verifier does not replace the online SWERL environment; it allows fast validation of candidate extraction, task files, commands, timeouts, and result schemas before consuming GPU allocation time.
