import json

import pytest

from oellm_rlvr import eu_tasks as E


@pytest.mark.parametrize("lang", E.LANGS)
def test_parse_number_reads_every_locale_format(lang):
    for x in (0.5, 12.3, 1234.56, 98765.4, 1234567.89):
        assert E.parse_number(E.fmt_money(x, lang) + " " + E.LOCALE[lang]["cur"]) == pytest.approx(x)


@pytest.mark.parametrize("text,value", [("1,234", 1234), ("1.234", 1234), ("0.125", 0.125), ("12,5", 12.5), ("€ 7", 7),
                                        ("−3,50 €", -3.5), ("1 234,56 kr", 1234.56), ("1,234,567.8", 1234567.8)])
def test_parse_number_cases(text, value):
    assert E.parse_number(text) == pytest.approx(value)


def qa_label(kind, answers, mode="default"):
    return {"suite": "qa", "kind": kind, "answers": answers, "mode": mode}


def test_qa_grading():
    year = qa_label("year", ["1879"])
    assert E.grade_qa("He was born in 1879.\nANSWER: 1879", year) == "correct"
    assert E.grade_qa("ANSWER: 1878", year) == "wrong"
    assert E.grade_qa("ANSWER: 1878 or 1879", year) == "wrong"
    assert E.grade_qa("ANSWER: UNKNOWN", year) == "abstain"
    assert E.grade_qa("ANSWER: Jag vet inte", year) == "abstain"
    assert E.grade_qa("1879", year) == "no_answer"
    assert E.grade_qa("<think>x</think>ANSWER: 1879", year) == "no_answer"  # think tag in default mode
    city = qa_label("entity", ["Uppsala", "Upsala"], mode="thinking")
    assert E.grade_qa("<think>hmm</think>\nANSWER: **Uppsala**", city) == "correct"
    assert E.grade_qa("<think>hmm</think>\nANSWER: Uppsala, Sweden", city) == "correct"
    assert E.grade_qa("<think>hmm</think>\nANSWER: Uppsala or Stockholm", city) == "wrong"
    assert E.grade_qa("<think>hmm\nANSWER: Uppsala", city) == "no_answer"  # unclosed trace
    assert E.score_qa("<think>a</think>ANSWER: Malmö", city) == -1.0
    assert E.score_qa("<think>a</think>ANSWER: UNKNOWN", city) == 0.0


def test_doc_scoring_kinds():
    assert E.score_doc("Sum...\nANSWER: 1.234,56 EUR", {"kind": "number", "value": 1234.56, "tol": 0.011}) == 1.0
    assert E.score_doc("ANSWER: 1234.58", {"kind": "number", "value": 1234.56, "tol": 0.011}) == 0.0
    assert E.score_doc("ANSWER: 2027-03-14", {"kind": "date", "value": "2027-03-14"}) == 1.0
    assert E.score_doc("ANSWER: 14.03.2027", {"kind": "date", "value": "2027-03-14"}) == 0.0
    assert E.score_doc("ANSWER: K-104, K-110", {"kind": "ids", "value": ["K-110", "K-104"], "pattern": r"K-\d+"}) == 1.0
    assert E.score_doc("ANSWER: K-104", {"kind": "ids", "value": ["K-110", "K-104"], "pattern": r"K-\d+"}) == 0.0
    assert E.score_doc("ANSWER: INV-1 or INV-2", {"kind": "id", "value": "INV-1", "pattern": r"INV-\d+"}) == 0.0
    gold = {"name": "Anna Weber", "customer_number": "12-34567", "order_date": "2025-07-01", "amount": 99.5, "city": "Köln"}
    reply = '```json\n{"name": "Anna Weber", "customer_number": "12-34567", "order_date": "2025-06-01", "amount": "99,50", "city": "Koln"}\n```'
    assert E.score_doc(reply, {"kind": "json", "value": gold}) == pytest.approx(0.8)


def test_generators_are_deterministic_and_self_consistent():
    for family in E.FAMILIES:
        for variant in E.variants(family):
            a = E.make_doc_task(family, "sv", 2, variant, 7)
            assert a == E.make_doc_task(family, "sv", 2, variant, 7)
            label = a["label"]
            gold = json.dumps(label["value"]) if label["kind"] == "json" else "ANSWER: " + (
                ", ".join(label["value"]) if label["kind"] == "ids" else str(label["value"]))
            assert E.score_doc(gold, label) == 1.0


def test_open_instruct_verifiers_registered():
    import types

    from oellm_rlvr.compat import patch_eu_task_verifiers

    class VerificationResult:
        def __init__(self, score, reasoning=None):
            self.score, self.reasoning = score, reasoning

    class VerifierFunction:
        def __init__(self, name, verifier_config=None, weight=1.0):
            self.name = name

    module = types.ModuleType("fake_ground_truth_utils")
    module.VerificationResult, module.VerifierFunction = VerificationResult, VerifierFunction
    assert patch_eu_task_verifiers(module) is True
    assert patch_eu_task_verifiers(module) is False
    qa, doc = module.EUQAVerifier(), module.EUDocVerifier()
    assert qa.name == "eu_qa" and doc.name == "eu_doc"
    label = json.dumps({"suite": "qa", "kind": "year", "answers": ["1901"], "mode": "thinking"})
    assert qa([], "recall...\n</think>\nANSWER: 1901", label).score == 1.0
    assert qa([], "recall...\n</think>\nANSWER: UNKNOWN", label).score == 0.0
    assert qa([], "recall... ANSWER: 1901", label).score == -1.0                       # unclosed trace
    label = json.dumps({"suite": "doc", "kind": "number", "value": 12.5, "tol": 0.011, "mode": "default"})
    assert doc([], "12,50 + 0\nANSWER: 12,50 EUR", label).score == 1.0
    assert doc([], "ANSWER: 12.5", "not json").score == 0.0
