"""Hallucination experiments on eu_qa: fabricated entities, stated risk thresholds, self-consistency.

Plan and results: qwen35-posttrain docs/OELLM_HALLUCINATION_PLAN.md.

- ``fabricated_qa``: questions about people, films, books, companies and rivers that do not exist, in the eu_qa
  templates and languages (HalluLens NonExistentRefusal spirit). Label kind ``unanswerable``: ``ANSWER: UNKNOWN`` is
  graded ``abstain`` (reward 0), any other answer ``wrong`` (-1), as in ``eu_tasks.grade_qa``. Names are built from
  syllables; ``scripts/build_hallucination_evals.py --check-wikidata`` drops any that Wikidata knows.
- ``threshold_prompt``: the eu_qa question with a stated confidence target t and the matching penalty t/(1-t) for a
  wrong answer (Kalai et al. 2025, arXiv 2509.04664): does the model's abstention respond to the stated stakes?
- ``consistency``: from k samples per eu_qa prompt (``pass_rate.sample`` output), how well the frequency of a sample's
  answer among the k predicts that it is correct, and the correct / wrong / abstain trade-off of the policy "answer
  only if at least a fraction tau of the samples agree". This is the headroom a calibrated model could reach without
  learning a single new fact; abstention training should move the model toward that curve.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import eu_tasks as E

# Threshold instruction replacing eu_tasks.QA_INSTRUCTION; {p} = confidence target in percent, {k} = wrong-answer penalty.
THRESHOLD_INSTRUCTION = {
    "en": "Answer with the name or year only, on a final line in the form ANSWER: <answer>. Answer only if you are more than {p}% confident that you are right; otherwise write ANSWER: UNKNOWN. Scoring: a correct answer gives 1 point, UNKNOWN gives 0 points, a wrong answer costs {k} points.",
    "de": "Antworte nur mit dem Namen bzw. der Jahreszahl, in einer letzten Zeile der Form ANSWER: <Antwort>. Antworte nur, wenn du dir zu mehr als {p} % sicher bist, dass du richtig liegst; sonst schreibe ANSWER: UNKNOWN. Bewertung: Eine richtige Antwort bringt 1 Punkt, UNKNOWN 0 Punkte, eine falsche Antwort kostet {k} Punkte.",
    "fr": "Réponds uniquement par le nom ou l'année, sur une dernière ligne de la forme ANSWER: <réponse>. Ne réponds que si tu es sûr à plus de {p} % d'avoir raison ; sinon écris ANSWER: UNKNOWN. Barème : une bonne réponse rapporte 1 point, UNKNOWN 0 point, une mauvaise réponse coûte {k} points.",
    "es": "Responde solo con el nombre o el año, en una última línea con el formato ANSWER: <respuesta>. Responde solo si tienes más de un {p} % de seguridad de acertar; si no, escribe ANSWER: UNKNOWN. Puntuación: una respuesta correcta da 1 punto, UNKNOWN 0 puntos, una respuesta incorrecta cuesta {k} puntos.",
    "it": "Rispondi solo con il nome o l'anno, in un'ultima riga nella forma ANSWER: <risposta>. Rispondi solo se sei sicuro di avere ragione per più del {p}%; altrimenti scrivi ANSWER: UNKNOWN. Punteggio: una risposta corretta vale 1 punto, UNKNOWN 0 punti, una risposta sbagliata costa {k} punti.",
    "nl": "Antwoord alleen met de naam of het jaartal, op een laatste regel in de vorm ANSWER: <antwoord>. Antwoord alleen als je voor meer dan {p}% zeker weet dat je gelijk hebt; schrijf anders ANSWER: UNKNOWN. Score: een goed antwoord levert 1 punt op, UNKNOWN 0 punten, een fout antwoord kost {k} punten.",
    "sv": "Svara endast med namnet eller årtalet, på en sista rad på formen ANSWER: <svar>. Svara bara om du är mer än {p} % säker på att du har rätt; skriv annars ANSWER: UNKNOWN. Poäng: ett rätt svar ger 1 poäng, UNKNOWN 0 poäng, ett felaktigt svar kostar {k} poäng.",
    "pl": "Odpowiedz wyłącznie nazwą lub rokiem, w ostatniej linii w formie ANSWER: <odpowiedź>. Odpowiadaj tylko wtedy, gdy masz ponad {p}% pewności, że masz rację; w przeciwnym razie napisz ANSWER: UNKNOWN. Punktacja: poprawna odpowiedź daje 1 punkt, UNKNOWN 0 punktów, błędna odpowiedź kosztuje {k} pkt.",
    "pt": "Responde apenas com o nome ou o ano, numa última linha no formato ANSWER: <resposta>. Responde apenas se tiveres mais de {p}% de certeza de que estás certo; caso contrário, escreve ANSWER: UNKNOWN. Pontuação: uma resposta certa vale 1 ponto, UNKNOWN 0 pontos, uma resposta errada custa {k} pontos.",
    "fi": "Vastaa pelkällä nimellä tai vuosiluvulla viimeisellä rivillä muodossa ANSWER: <vastaus>. Vastaa vain, jos olet yli {p} % varma, että vastauksesi on oikein; muuten kirjoita ANSWER: UNKNOWN. Pisteytys: oikea vastaus antaa 1 pisteen, UNKNOWN 0 pistettä, väärä vastaus maksaa {k} pistettä.",
}


def penalty(t: float) -> float:
    """Wrong-answer penalty that makes answering worthwhile exactly above confidence t (Kalai et al.)."""
    return t / (1.0 - t)


def threshold_prompt(prompt: str, lang: str, t: float) -> str:
    """Replace the standard eu_qa instruction at the end of ``prompt`` by the threshold-t instruction."""
    standard = E.QA_INSTRUCTION[lang]
    if not prompt.endswith(standard):
        raise ValueError("prompt does not end with the standard eu_qa instruction")
    k = round(penalty(t), 1)
    k_text = f"{k:g}"
    return prompt[: -len(standard)] + THRESHOLD_INSTRUCTION[lang].format(p=round(100 * t), k=k_text)


# --------------------------------------------------------------------------------------------------------------------
# Fabricated entities
# --------------------------------------------------------------------------------------------------------------------

COUNTRIES = {  # country -> (language of its first names, label per prompt language)
    "SE": ("sv", {"en": "Sweden", "de": "Schweden", "fr": "Suède", "es": "Suecia", "it": "Svezia", "nl": "Zweden",
                  "sv": "Sverige", "pl": "Szwecja", "pt": "Suécia", "fi": "Ruotsi"}),
    "DE": ("de", {"en": "Germany", "de": "Deutschland", "fr": "Allemagne", "es": "Alemania", "it": "Germania",
                  "nl": "Duitsland", "sv": "Tyskland", "pl": "Niemcy", "pt": "Alemanha", "fi": "Saksa"}),
    "FR": ("fr", {"en": "France", "de": "Frankreich", "fr": "France", "es": "Francia", "it": "Francia",
                  "nl": "Frankrijk", "sv": "Frankrike", "pl": "Francja", "pt": "França", "fi": "Ranska"}),
    "IT": ("it", {"en": "Italy", "de": "Italien", "fr": "Italie", "es": "Italia", "it": "Italia", "nl": "Italië",
                  "sv": "Italien", "pl": "Włochy", "pt": "Itália", "fi": "Italia"}),
    "PL": ("pl", {"en": "Poland", "de": "Polen", "fr": "Pologne", "es": "Polonia", "it": "Polonia", "nl": "Polen",
                  "sv": "Polen", "pl": "Polska", "pt": "Polónia", "fi": "Puola"}),
    "FI": ("fi", {"en": "Finland", "de": "Finnland", "fr": "Finlande", "es": "Finlandia", "it": "Finlandia",
                  "nl": "Finland", "sv": "Finland", "pl": "Finlandia", "pt": "Finlândia", "fi": "Suomi"}),
    "ES": ("es", {"en": "Spain", "de": "Spanien", "fr": "Espagne", "es": "España", "it": "Spagna", "nl": "Spanje",
                  "sv": "Spanien", "pl": "Hiszpania", "pt": "Espanha", "fi": "Espanja"}),
    "NL": ("nl", {"en": "Netherlands", "de": "Niederlande", "fr": "Pays-Bas", "es": "Países Bajos", "it": "Paesi Bassi",
                  "nl": "Nederland", "sv": "Nederländerna", "pl": "Holandia", "pt": "Países Baixos", "fi": "Alankomaat"}),
}
OCCUPATIONS = {
    "writer": {"en": "writer", "de": "Schriftsteller", "fr": "écrivain", "es": "escritor", "it": "scrittore",
               "nl": "schrijver", "sv": "författare", "pl": "pisarz", "pt": "escritor", "fi": "kirjailija"},
    "painter": {"en": "painter", "de": "Maler", "fr": "peintre", "es": "pintor", "it": "pittore", "nl": "schilder",
                "sv": "målare", "pl": "malarz", "pt": "pintor", "fi": "taidemaalari"},
    "politician": {"en": "politician", "de": "Politiker", "fr": "homme politique", "es": "político", "it": "politico",
                   "nl": "politicus", "sv": "politiker", "pl": "polityk", "pt": "político", "fi": "poliitikko"},
    "composer": {"en": "composer", "de": "Komponist", "fr": "compositeur", "es": "compositor", "it": "compositore",
                 "nl": "componist", "sv": "kompositör", "pl": "kompozytor", "pt": "compositor", "fi": "säveltäjä"},
    "footballer": {"en": "footballer", "de": "Fußballspieler", "fr": "footballeur", "es": "futbolista",
                   "it": "calciatore", "nl": "voetballer", "sv": "fotbollsspelare", "pl": "piłkarz",
                   "pt": "futebolista", "fi": "jalkapalloilija"},
}
COMPANY = {"en": "company", "de": "Unternehmen", "fr": "entreprise", "es": "empresa", "it": "azienda", "nl": "bedrijf",
           "sv": "företag", "pl": "przedsiębiorstwo", "pt": "empresa", "fi": "yritys"}
SYLLABLES = ["nor", "vel", "tra", "ken", "lum", "sar", "bri", "dal", "mon", "ter", "fal", "rin", "ost", "gal", "pen",
             "vik", "zor", "mel", "quin", "dra", "sol", "bek", "tur", "har", "lio", "ves", "kar", "dun", "ser", "gri"]
FABRICATED_TYPES = ["birth_year", "birth_place", "film_director", "book_author", "org_founded", "river_mouth"]


def made_up_word(rng: random.Random, syllables: int) -> str:
    return "".join(rng.choice(SYLLABLES) for _ in range(syllables)).capitalize()


def fabricated_item(rng: random.Random, qtype: str, lang: str) -> dict[str, Any]:
    """One question about a nonexistent entity; ``subject`` is the string to check against Wikidata."""
    country = rng.choice(sorted(COUNTRIES))
    name_lang, country_labels = COUNTRIES[country]
    c = country_labels[lang]
    if qtype in ("birth_year", "birth_place"):
        subject = f"{rng.choice(E.FIRST[name_lang])} {made_up_word(rng, 3)}"
        slots = {"s": subject, "d": f"{OCCUPATIONS[rng.choice(sorted(OCCUPATIONS))][lang]}, {c}"}
    elif qtype == "film_director":
        subject = f"{made_up_word(rng, 3)} {made_up_word(rng, 2)}"
        slots = {"s": subject, "y": str(rng.randint(1955, 2015)), "c": c}
    elif qtype == "book_author":
        subject = f"{made_up_word(rng, 2)} {made_up_word(rng, 3)}"
        slots = {"s": subject, "c": c}
    elif qtype == "org_founded":
        subject = f"{made_up_word(rng, 3)} {E.LEGAL[name_lang]}"
        slots = {"s": subject, "d": f"{COMPANY[lang]}, {c}"}
    elif qtype == "river_mouth":
        subject = made_up_word(rng, 3)
        slots = {"s": subject, "c": c}
    else:
        raise ValueError(f"no fabricated template for {qtype}")
    return {"subject": subject, "prompt": E.qa_prompt(qtype, lang, slots)}


def fabricated_qa(per_language: int, seed: int = 20261102, langs: list[str] | None = None) -> list[dict[str, Any]]:
    """Eval rows in the eu-tasks eval schema (suite qa, band ``fabricated``), round-robin over question types."""
    rows = []
    for lang in langs or E.LANGS:
        rng = random.Random(f"fabricated:{seed}:{lang}")
        seen: set[str] = set()
        while len([r for r in rows if r["language"] == lang]) < per_language:
            qtype = FABRICATED_TYPES[len([r for r in rows if r["language"] == lang]) % len(FABRICATED_TYPES)]
            item = fabricated_item(rng, qtype, lang)
            if item["subject"] in seen:
                continue
            seen.add(item["subject"])
            rows.append({"id": f"fabricated-{qtype}-{len(rows)}-{lang}", "suite": "qa", "subset": qtype,
                         "language": lang, "band": "fabricated", "prompt": item["prompt"], "subject": item["subject"],
                         "label": {"suite": "qa", "kind": "unanswerable", "answers": []}, "source": "fabricated"})
    return rows


# --------------------------------------------------------------------------------------------------------------------
# Self-consistency
# --------------------------------------------------------------------------------------------------------------------

ABSTAIN_KEY, NO_ANSWER_KEY = "<abstain>", "<no-answer>"


def answer_key(prediction: str, label: dict[str, Any]) -> str:
    """Canonical form of a sample's submitted answer, for counting agreement within a group."""
    text = E.answer_part(prediction, label.get("mode", "default"))
    value = E.final_answer(text) if text is not None else None
    if value is None:
        return NO_ANSWER_KEY
    if E.is_abstention(value):
        return ABSTAIN_KEY
    if label["kind"] == "year":
        years = sorted(set(re.findall(r"(?<!\d)(\d{3,4})(?!\d)", value)))
        return "|".join(years) or E.norm_text(value)
    return E.norm_text(value)


def _auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _policy(samples: list[dict[str, Any]], tau: float) -> dict[str, float]:
    """Answer when the sample's own answer has agreement >= tau, otherwise abstain."""
    c = Counter()
    for s in samples:
        if s["outcome"] == "abstain" or s["agreement"] < tau:
            c["abstain"] += 1
        else:
            c[s["outcome"]] += 1
    n = max(1, len(samples))
    return {"tau": tau, "correct": c["correct"] / n, "wrong": c["wrong"] / n, "abstain": c["abstain"] / n,
            "index": 100 * (c["correct"] - c["wrong"]) / n,
            "hallucination_rate": c["wrong"] / max(1, n - c["correct"])}


def consistency(pool: str | Path, sample_files: list[str | Path], *, dataset: str = "eu_qa") -> dict[str, Any]:
    from .pass_rate import _read_jsonl, read_pool

    labels = {}
    for row in read_pool(pool):
        if row["dataset"] == dataset:
            label = row["ground_truth"]
            labels[str(row["id"])] = json.loads(label) if isinstance(label, str) else label
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sample_files:
        for s in _read_jsonl(Path(path)):
            if s["id"] in labels:
                groups[s["id"]].append(s)
    samples = []
    for pid, group in groups.items():
        label = labels[pid]
        keys = [answer_key(s["text"], label) for s in group]
        counts = Counter(keys)
        for s, key in zip(group, keys):
            outcome = E.grade_qa(s["text"], label)
            outcome = "wrong" if outcome == "no_answer" else outcome
            samples.append({"id": pid, "language": pid.rsplit("-", 1)[-1], "type": pid.split("-")[1],
                            "outcome": outcome, "agreement": counts[key] / len(group) if key != ABSTAIN_KEY else 0.0,
                            "answered": key not in (ABSTAIN_KEY, NO_ANSWER_KEY)})
    answered = [s for s in samples if s["answered"]]
    taus = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]

    def summary(subset: list[dict[str, Any]]) -> dict[str, Any]:
        ans = [s for s in subset if s["answered"]]
        return {"samples": len(subset),
                "auroc_agreement_vs_correct": _auroc([s["agreement"] for s in ans if s["outcome"] == "correct"],
                                                     [s["agreement"] for s in ans if s["outcome"] == "wrong"]),
                "as_sampled": _policy(subset, 0.0),
                "policy": [_policy(subset, t) for t in taus[1:]]}

    by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_lang[s["language"]].append(s)
    # Accuracy of answered samples by agreement bucket (reliability of self-consistency as confidence).
    buckets: dict[float, Counter] = defaultdict(Counter)
    for s in answered:
        buckets[round(s["agreement"] * 8) / 8][s["outcome"]] += 1
    reliability = [{"agreement": a, "n": sum(c.values()), "accuracy": c["correct"] / sum(c.values())}
                   for a, c in sorted(buckets.items())]
    return {"dataset": dataset, "prompts": len(groups), **summary(samples), "reliability": reliability,
            "by_language": {lang: summary(v) for lang, v in sorted(by_lang.items())}}
