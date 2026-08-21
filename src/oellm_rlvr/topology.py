from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import RunConfig


@dataclass(frozen=True)
class Topology:
    capacity_gpus: int
    learner_gpus: int
    rollout_gpus: int
    spare_gpus: int
    learner_nodes: int
    rollout_engines: int
    tensor_parallel_size: int
    data_parallel_ranks: int
    samples_per_step: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def build_topology(config: RunConfig) -> Topology:
    capacity = config.platform.nodes * config.platform.gpus_per_node
    learners = sum(config.training.learner_gpus_per_node)
    rollout = config.rollout.engines * config.rollout.tensor_parallel_size
    return Topology(
        capacity_gpus=capacity,
        learner_gpus=learners,
        rollout_gpus=rollout,
        spare_gpus=capacity - learners - rollout,
        learner_nodes=len(config.training.learner_gpus_per_node),
        rollout_engines=config.rollout.engines,
        tensor_parallel_size=config.rollout.tensor_parallel_size,
        data_parallel_ranks=learners // config.training.sequence_parallel_size,
        samples_per_step=config.rollout.unique_prompts * config.rollout.samples_per_prompt,
    )
