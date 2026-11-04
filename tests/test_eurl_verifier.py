import json
import os
from types import ModuleType, SimpleNamespace

import pytest

from oellm_rlvr.compat import eurl_score, patch_eurl_verifier

ROOT = os.environ.get("OELLM_EUROPEAN_RL", os.path.expanduser("~/AI-Sweden/european-rl"))
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "eurl")), reason="european-rl checkout missing")


def _row(lang="pol", seed=3):
    os.environ["OELLM_EUROPEAN_RL"] = ROOT
    import sys
    sys.path.insert(0, ROOT)
    from eurl.core import load
    env = load("finance_amount_due")
    env.reset(seed, "train", lang)
    return env, {**env.ground_truth(), "mode": "default"}


def test_reference_scores_and_modes_are_enforced():
    env, contract = _row()
    ref = env.reference_completion()
    assert eurl_score(ref, contract) == 1.0
    assert eurl_score("<think>x</think>\n" + ref, contract) == 0.0  # think tags in default mode
    assert eurl_score("<think>x</think>\n" + ref, {**contract, "mode": "thinking"}) == 1.0
    assert eurl_score(ref, {**contract, "mode": "thinking"}) == 0.0  # no closed trace
    assert eurl_score("ANSWER: 0", contract) == 0.0


def test_verifier_registration():
    class VerifierFunction:
        def __init__(self, name, verifier_config=None, weight=1.0):
            self.name = name

    module = ModuleType("fake_gt")
    module.VerifierFunction = VerifierFunction
    module.VerificationResult = lambda score, reasoning="": SimpleNamespace(score=score, reasoning=reasoning)
    assert patch_eurl_verifier(module) and not patch_eurl_verifier(module)
    env, contract = _row("fin", 5)
    v = module.EurlTypedVerifier()
    assert v.name == "eurl_typed"
    assert v(None, env.reference_completion(), json.dumps(contract)).score == 1.0
    assert v(None, "x", "not json").score == 0.0
