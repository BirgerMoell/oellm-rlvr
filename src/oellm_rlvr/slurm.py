from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from .config import RunConfig


def render_slurm(config: RunConfig, config_path: str | Path) -> str:
    template_text = files("oellm_rlvr.templates").joinpath("lumi.sbatch.j2").read_text()
    template = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=True).from_string(
        template_text
    )
    control_root = Path(__file__).resolve().parents[2]
    return template.render(
        config=config,
        config_path=str(Path(config_path).resolve()),
        control_root=str(control_root),
        accelerator_flag="--rocm" if config.platform.accelerator == "rocm" else "--nv",
        module_lines="\n".join(f"module load {module}" for module in config.platform.modules),
        bind_value=",".join(config.platform.binds),
    )
