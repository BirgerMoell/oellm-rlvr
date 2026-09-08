from pathlib import Path

import pytest
from pydantic import ValidationError

from oellm_rlvr.campaign import CampaignConfig, load_campaign, render_campaign_markdown

ROOT = Path(__file__).parents[1]
CAMPAIGN = ROOT / "campaigns/lumi-9b-end-to-end-dry-run.yaml"
PROGRESSIVE_CAMPAIGN = ROOT / "campaigns/lumi-oellm9b-progressive-rl.yaml"


def test_lumi_campaign_validates_and_stays_inside_budget() -> None:
    campaign = load_campaign(CAMPAIGN)
    summary = campaign.summary()

    assert summary["valid"] is True
    assert summary["stage_count"] == 14
    assert summary["ready_stage_count"] == 4
    assert summary["required_expected_gcd_hours"] == 290
    assert summary["required_hard_ceiling_gcd_hours"] == 720
    assert summary["campaign_hard_ceiling_gcd_hours"] == 750
    assert summary["contingency_hard_ceiling_gcd_hours"] == 96
    assert "agentic-9b-one-step" in summary["build_required"]
    assert len(summary["build_required"]) == 9


def test_campaign_markdown_exposes_schedule_and_gaps() -> None:
    rendered = render_campaign_markdown(load_campaign(CAMPAIGN))

    assert "# OpenEuroLLM 9B full-stack RL dry run" in rendered
    assert "`agentic-9b-one-step`" in rendered
    assert "Work required before launch" in rendered
    assert "290 expected / 720 hard-ceiling GCD-hours" in rendered


def test_campaign_rejects_forward_dependency() -> None:
    campaign = load_campaign(CAMPAIGN).model_dump()
    campaign["stages"][0]["needs"] = [campaign["stages"][1]["id"]]

    with pytest.raises(ValidationError, match="must appear earlier"):
        CampaignConfig.model_validate(campaign)


def test_campaign_rejects_required_budget_above_ceiling() -> None:
    campaign = load_campaign(CAMPAIGN).model_dump()
    campaign["hard_ceiling_gcd_hours"] = 100

    with pytest.raises(ValidationError, match="above campaign ceiling"):
        CampaignConfig.model_validate(campaign)


def test_progressive_campaign_is_strictly_sequential_and_has_valid_mixes() -> None:
    campaign = load_campaign(PROGRESSIVE_CAMPAIGN)
    stage_ids = [stage.id for stage in campaign.stages]

    assert campaign.control_checkpoints == ["Qwen/Qwen3.5-9B"]
    assert stage_ids.index("reasoning-train") < stage_ids.index("math-train")
    assert stage_ids.index("math-train") < stage_ids.index("code-train")
    assert stage_ids.index("code-train") < stage_ids.index("agentic-train")
    for stage in campaign.stages:
        if stage.curriculum:
            assert sum(window.fraction for window in stage.curriculum) == pytest.approx(1.0)
            for window in stage.curriculum:
                assert sum(window.mixture.values()) == pytest.approx(1.0)


def test_campaign_rejects_invalid_curriculum_fraction() -> None:
    campaign = load_campaign(PROGRESSIVE_CAMPAIGN).model_dump()
    curriculum = next(stage["curriculum"] for stage in campaign["stages"] if stage["curriculum"])
    curriculum[0]["fraction"] = 0.1

    with pytest.raises(ValidationError, match="window fractions must sum to 1"):
        CampaignConfig.model_validate(campaign)


def test_progressive_campaign_render_exposes_checkpoint_flow_and_mix() -> None:
    rendered = render_campaign_markdown(load_campaign(PROGRESSIVE_CAMPAIGN))

    assert "Qwen/Qwen3.5-9B" in rendered
    assert "## Checkpoint flow and curricula" in rendered
    assert "`<frozen-parent>`" in rendered
    assert "math_easy=50%" in rendered
