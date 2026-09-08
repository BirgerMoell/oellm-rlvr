"""Project-owned SkyRL Harbor training entrypoint with policy-version evidence.

SkyRL's upstream Harbor learner is used without changing its optimization
logic.  This thin entrypoint only records the inference client's monotonic
weight version before every rollout batch so a qualification run can prove
that a later batch was sampled after an optimizer update and weight sync.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import ray
import yaml
from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
)
from examples.train_integrations.harbor.harbor_generator import HarborGenerator
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


class VersionedHarborGenerator(HarborGenerator):
    """Harbor generator that emits the exact sampler weight version."""

    async def generate(self, input_batch: Any, disable_tqdm: bool = False) -> Any:
        version = getattr(self.inference_engine_client, "weight_version", None)
        print(f"OELLM_HARBOR_POLICY_VERSION={version}", flush=True)
        output = await super().generate(input_batch, disable_tqdm=disable_tqdm)
        metrics = output.get("rollout_metrics")
        if isinstance(metrics, dict) and isinstance(version, int):
            metrics["generate/policy_weight_version"] = version
        return output


class VersionedHarborExp(HarborExp):
    """Use the upstream trainer with the evidence-producing generator."""

    def get_generator(self, cfg: Any, tokenizer: Any, inference_engine_client: Any) -> VersionedHarborGenerator:
        return VersionedHarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: HarborSkyRLConfig) -> None:
    VersionedHarborExp(cfg).run()


def main(argv: list[str] | None = None) -> None:
    overrides = sys.argv[1:] if argv is None else argv
    cfg = HarborSkyRLConfig.from_cli_overrides(overrides)
    defaults_path = Path(HARBOR_DEFAULT_CONFIG)
    defaults = yaml.safe_load(defaults_path.read_text())
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
