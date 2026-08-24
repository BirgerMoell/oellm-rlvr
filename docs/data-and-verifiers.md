# Data and verifier contracts

## Math

The backend-compatible minimum row is:

```json
{"messages":[{"role":"user","content":"Compute 2 + 3."}],"ground_truth":"5","dataset":"math"}
```

`make-math-smoke` emits JSONL or Parquet based on the output suffix. Production math data should preserve source/license/provenance outside the training columns, pin a revision, remove evaluation contamination, and test every verifier on adversarial completions before training.

`sample-math` selects distinct semantic groups from the published OpenEuroLLM Math RLVR Parquet shard
and preserves all provenance and verifier columns. The policy still receives only `messages`.

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

At load time, the pinned backend merges this with the run-level environment config and passes `task_id`, `image`, and `max_steps` to the Ray environment actor. The code pool uses `apptainer` or `prepared_apptainer` on LUMI and Docker/Apptainer on CUDA systems.

## Test design

Tests must be deterministic, bounded, and resistant to hard-coded answers. Use multiple public/hidden cases, property checks where practical, isolated temporary state, and explicit timeouts. Do not use network access or external services. Treat syntax/test failures as valid reward zero; label sandbox launch, missing image, missing test file, timeout, OOM, and parser failures as infrastructure errors as well as reward zero.

Binary reward is the safest initial contract. Introduce partial reward only when every component has a precise interpretation and cannot be farmed without solving the task. Version verifier code and store the version with every trajectory.

`sample-code` accepts the published OpenEuroLLM Code RLVR Parquet shard. It converts each Python
`stdin_stdout` hidden test into `cases.json` plus a bounded sandbox runner. The generated training row
contains only the policy-visible messages and TMAX environment configuration; hidden tests never enter
the model context.

## Local verifier QA

The standalone `CodeVerifier` can execute a `TaskSpec` against an Apptainer image. A local subprocess runner exists only for trusted repository tests and requires an explicit unsafe flag. This verifier does not replace the online SWERL environment; it allows fast validation of candidate extraction, task files, commands, timeouts, and result schemas before consuming GPU allocation time.
