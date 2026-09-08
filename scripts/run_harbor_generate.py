"""Learner-off Harbor rollout entrypoint with real repeated-prompt sampling.

SkyRL v0.3.0's upstream ``main_harbor_generate`` is a debugging helper: it
hard-codes ``repetition_id=0`` and therefore ignores
``generator.n_samples_per_prompt``.  This project-owned entrypoint preserves
the upstream setup while expanding each prompt into the requested group.
"""

from __future__ import annotations

import asyncio
import sys

import ray
import yaml
from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HARBOR_DEFAULT_CONFIG,
    HarborSkyRLConfig,
    _deep_merge,
)
from examples.train_integrations.harbor.entrypoints.main_harbor_generate import HarborGenerateExp
from skyrl.train.generators.base import GeneratorInput, TrajectoryID
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from oellm_rlvr.harbor_sampling import expand_prompt_repetitions


class RepeatedHarborGenerateExp(HarborGenerateExp):
    def run(self) -> None:
        generator = self._setup_generator()
        prompts, trajectory_ids = expand_prompt_repetitions(
            self.train_dataset,
            int(self.cfg.generator.n_samples_per_prompt),
            TrajectoryID,
        )
        input_batch = GeneratorInput(
            prompts=prompts,
            trajectory_ids=trajectory_ids,
            env_classes=None,
            env_extras=None,
            sampling_params=None,
        )
        asyncio.run(generator.generate(input_batch))


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg) -> None:
    RepeatedHarborGenerateExp(cfg).run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])
    with open(HARBOR_DEFAULT_CONFIG) as handle:
        defaults = yaml.safe_load(handle)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)
    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be explicitly set for Harbor generation")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
