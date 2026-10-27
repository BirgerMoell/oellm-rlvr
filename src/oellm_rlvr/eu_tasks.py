"""EU task suites: calibrated factual QA and document tasks, in ten EU languages.

Two suites in the spirit of the Artificial Analysis Intelligence Index, sized for ~9B models. Each one is a generator
with a programmatic check, so one module yields a held-out eval set and an unbounded RL pool:

- ``eu_qa`` (AA-Omniscience spirit): short factual questions built from Wikidata (CC0) facts about European people,
  places, works and organisations, from head to long-tail entities. Reward +1 correct, 0 abstain (``ANSWER: UNKNOWN``),
  -1 wrong or missing. The eval reports accuracy, abstention, hallucination rate (wrong / not correct) and the
  knowledge index 100 * (correct - wrong) / n.
- ``eu_doc`` (GDP.pdf / AA-Briefcase spirit): procedurally generated invoices, tables, contracts, meeting schedules and
  letters with locale-formatted numbers and dates; questions with exact answers (amounts, counts, ID sets, ISO dates,
  JSON fields). Reward 1 / 0 (JSON: fraction of fields correct).

Replies end with a line ``ANSWER: <answer>`` (the keyword is fixed in every language; JSON tasks reply with the object).
Thinking-mode replies are scored on the text after exactly one closed ``</think>``; default-mode replies must contain no
think tag. Each document family has one question variant held out of the RL pool (``HELD_OUT``) so the eval also
measures transfer to an unseen question type.
"""
from __future__ import annotations

import calendar
import datetime as dt
import json
import random
import re
import unicodedata
from typing import Any

LANGS = ["en", "de", "fr", "es", "it", "nl", "sv", "pl", "pt", "fi"]

# --------------------------------------------------------------------------------------------------------------------
# Answer extraction and normalisation (shared)
# --------------------------------------------------------------------------------------------------------------------

ANSWER_LINE = re.compile(r"ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)
ABSTAIN = ["unknown", "i don't know", "i do not know", "weiß ich nicht", "ich weiß es nicht", "unbekannt",
           "je ne sais pas", "inconnu", "no lo sé", "no lo se", "desconocido", "non lo so", "sconosciuto",
           "weet ik niet", "onbekend", "vet inte", "jag vet inte", "okänt", "nie wiem", "nieznany", "não sei",
           "desconhecido", "en tiedä", "tuntematon"]


def answer_part(prediction: str, mode: str) -> str | None:
    """Text the model submits: the whole reply (default mode, no think tags) or the text after one closed trace."""
    text = prediction.replace("<|im_end|>", "")
    if mode == "default":
        return None if "<think>" in text or "</think>" in text else text.strip()
    if text.count("</think>") != 1 or "<think>" in text.split("</think>", 1)[1]:
        return None
    return text.split("</think>", 1)[1].strip()


def final_answer(text: str) -> str | None:
    """Value of the last ``ANSWER:`` line (markdown emphasis and trailing period stripped)."""
    hits = ANSWER_LINE.findall(text.replace("**", ""))
    if not hits:
        return None
    value = hits[-1].strip().strip("`*_ ").rstrip(".").strip()
    return value or None


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ß", "ss").replace("ø", "o").replace("æ", "ae").replace("ł", "l").replace("đ", "d")
    s = re.sub(r"[\"'“”„‘’«»‚()\[\]]", "", s)
    s = re.sub(r"[^\w]+", " ", s)
    return " ".join(s.split())


def parse_number(s: str) -> float | None:
    """First number in ``s``, accepting en/de/fr/sv-style grouping and decimal separators."""
    s = s.replace(" ", " ").replace(" ", " ").replace("−", "-")
    m = re.search(r"-?\d[\d ,.']*", s)
    if not m:
        return None
    t = m.group(0).rstrip(" .,'").replace(" ", "").replace("'", "")
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        t = t.replace("." if dec == "," else ",", "").replace(dec, ".")
    elif "," in t:
        parts = t.split(",")
        t = t.replace(",", "") if len(parts) > 2 or len(parts[1]) == 3 else t.replace(",", ".")
    elif t.count(".") > 1 or (t.count(".") == 1 and len(t.split(".")[1]) == 3 and t.split(".")[0].lstrip("-") not in ("", "0")):
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return None


def is_abstention(value: str) -> bool:
    v = norm_text(value)
    return any(v == norm_text(p) or (len(v) < 40 and norm_text(p) in v) for p in ABSTAIN)


# --------------------------------------------------------------------------------------------------------------------
# Suite 1: calibrated factual QA (eu_qa)
# --------------------------------------------------------------------------------------------------------------------

QA_INSTRUCTION = {
    "en": "Answer with the name or year only, on a final line in the form ANSWER: <answer>. If you are not sure, write ANSWER: UNKNOWN. A wrong answer counts against you; UNKNOWN does not.",
    "de": "Antworte nur mit dem Namen bzw. der Jahreszahl, in einer letzten Zeile der Form ANSWER: <Antwort>. Wenn du dir nicht sicher bist, schreibe ANSWER: UNKNOWN. Eine falsche Antwort zählt negativ, UNKNOWN nicht.",
    "fr": "Réponds uniquement par le nom ou l'année, sur une dernière ligne de la forme ANSWER: <réponse>. Si tu n'es pas sûr, écris ANSWER: UNKNOWN. Une mauvaise réponse est pénalisée, UNKNOWN ne l'est pas.",
    "es": "Responde solo con el nombre o el año, en una última línea con el formato ANSWER: <respuesta>. Si no estás seguro, escribe ANSWER: UNKNOWN. Una respuesta incorrecta penaliza; UNKNOWN no.",
    "it": "Rispondi solo con il nome o l'anno, in un'ultima riga nella forma ANSWER: <risposta>. Se non sei sicuro, scrivi ANSWER: UNKNOWN. Una risposta sbagliata viene penalizzata, UNKNOWN no.",
    "nl": "Antwoord alleen met de naam of het jaartal, op een laatste regel in de vorm ANSWER: <antwoord>. Als je het niet zeker weet, schrijf dan ANSWER: UNKNOWN. Een fout antwoord telt negatief, UNKNOWN niet.",
    "sv": "Svara endast med namnet eller årtalet, på en sista rad på formen ANSWER: <svar>. Om du inte är säker, skriv ANSWER: UNKNOWN. Ett felaktigt svar räknas negativt, UNKNOWN gör det inte.",
    "pl": "Odpowiedz wyłącznie nazwą lub rokiem, w ostatniej linii w formie ANSWER: <odpowiedź>. Jeśli nie masz pewności, napisz ANSWER: UNKNOWN. Błędna odpowiedź liczy się na minus, UNKNOWN nie.",
    "pt": "Responde apenas com o nome ou o ano, numa última linha no formato ANSWER: <resposta>. Se não tiveres a certeza, escreve ANSWER: UNKNOWN. Uma resposta errada é penalizada; UNKNOWN não.",
    "fi": "Vastaa pelkällä nimellä tai vuosiluvulla viimeisellä rivillä muodossa ANSWER: <vastaus>. Jos et ole varma, kirjoita ANSWER: UNKNOWN. Väärä vastaus lasketaan miinukseksi, UNKNOWN ei.",
}

# {s} subject label, {d} disambiguation ("occupation, country" / "class, country"), {c} country, {y} year
QA_TEMPLATES = {
    "birth_year": {
        "en": "In which year was {s} ({d}) born?", "de": "In welchem Jahr wurde {s} ({d}) geboren?",
        "fr": "En quelle année est né(e) {s} ({d}) ?", "es": "¿En qué año nació {s} ({d})?",
        "it": "In che anno è nato/a {s} ({d})?", "nl": "In welk jaar is {s} ({d}) geboren?",
        "sv": "Vilket år föddes {s} ({d})?", "pl": "W którym roku urodził(a) się {s} ({d})?",
        "pt": "Em que ano nasceu {s} ({d})?", "fi": "Minä vuonna {s} ({d}) syntyi?"},
    "birth_place": {
        "en": "In which city or town was {s} ({d}) born?", "de": "In welcher Stadt oder welchem Ort wurde {s} ({d}) geboren?",
        "fr": "Dans quelle ville ou localité est né(e) {s} ({d}) ?", "es": "¿En qué ciudad o localidad nació {s} ({d})?",
        "it": "In quale città o località è nato/a {s} ({d})?", "nl": "In welke stad of plaats is {s} ({d}) geboren?",
        "sv": "I vilken stad eller ort föddes {s} ({d})?", "pl": "W jakim mieście lub miejscowości urodził(a) się {s} ({d})?",
        "pt": "Em que cidade ou localidade nasceu {s} ({d})?", "fi": "Missä kaupungissa tai paikkakunnalla {s} ({d}) syntyi?"},
    "admin_capital": {
        "en": "What is the capital or administrative seat of {s} ({c})?", "de": "Was ist die Hauptstadt bzw. der Verwaltungssitz von {s} ({c})?",
        "fr": "Quel est le chef-lieu ou la capitale de {s} ({c}) ?", "es": "¿Cuál es la capital o sede administrativa de {s} ({c})?",
        "it": "Qual è il capoluogo o la sede amministrativa di {s} ({c})?", "nl": "Wat is de hoofdstad of bestuurszetel van {s} ({c})?",
        "sv": "Vad är huvudort eller administrativt centrum för {s} ({c})?", "pl": "Jaka jest stolica lub siedziba władz jednostki {s} ({c})?",
        "pt": "Qual é a capital ou sede administrativa de {s} ({c})?", "fi": "Mikä on alueen {s} ({c}) pääkaupunki tai hallinnollinen keskus?"},
    "film_director": {
        "en": "Who directed the film “{s}” ({y}, {c})?", "de": "Wer führte Regie beim Film „{s}“ ({y}, {c})?",
        "fr": "Qui a réalisé le film « {s} » ({y}, {c}) ?", "es": "¿Quién dirigió la película «{s}» ({y}, {c})?",
        "it": "Chi ha diretto il film «{s}» ({y}, {c})?", "nl": "Wie regisseerde de film ‘{s}’ ({y}, {c})?",
        "sv": "Vem regisserade filmen ”{s}” ({y}, {c})?", "pl": "Kto wyreżyserował film „{s}” ({y}, {c})?",
        "pt": "Quem realizou o filme «{s}» ({y}, {c})?", "fi": "Kuka ohjasi elokuvan ”{s}” ({y}, {c})?"},
    "book_author": {
        "en": "Who wrote the literary work “{s}” (author's country: {c})?", "de": "Wer schrieb das literarische Werk „{s}“ (Land der Autorin/des Autors: {c})?",
        "fr": "Qui a écrit l'œuvre littéraire « {s} » (pays de l'auteur : {c}) ?", "es": "¿Quién escribió la obra literaria «{s}» (país del autor: {c})?",
        "it": "Chi ha scritto l'opera letteraria «{s}» (paese dell'autore: {c})?", "nl": "Wie schreef het literaire werk ‘{s}’ (land van de auteur: {c})?",
        "sv": "Vem skrev det litterära verket ”{s}” (författarens land: {c})?", "pl": "Kto napisał utwór literacki „{s}” (kraj autora: {c})?",
        "pt": "Quem escreveu a obra literária «{s}» (país do autor: {c})?", "fi": "Kuka kirjoitti kirjallisen teoksen ”{s}” (kirjailijan maa: {c})?"},
    "org_founded": {
        "en": "In which year was {s} ({d}) founded?", "de": "In welchem Jahr wurde {s} ({d}) gegründet?",
        "fr": "En quelle année a été fondé(e) {s} ({d}) ?", "es": "¿En qué año se fundó {s} ({d})?",
        "it": "In che anno è stato fondato/a {s} ({d})?", "nl": "In welk jaar is {s} ({d}) opgericht?",
        "sv": "Vilket år grundades {s} ({d})?", "pl": "W którym roku założono {s} ({d})?",
        "pt": "Em que ano foi fundado(a) {s} ({d})?", "fi": "Minä vuonna {s} ({d}) perustettiin?"},
    "river_mouth": {
        "en": "Into which river, lake or sea does the river {s} ({c}) flow?", "de": "In welchen Fluss, welchen See oder welches Meer mündet der Fluss {s} ({c})?",
        "fr": "Dans quel cours d'eau, lac ou mer se jette la rivière {s} ({c}) ?", "es": "¿En qué río, lago o mar desemboca el río {s} ({c})?",
        "it": "In quale fiume, lago o mare sfocia il fiume {s} ({c})?", "nl": "In welke rivier, welk meer of welke zee mondt de rivier {s} ({c}) uit?",
        "sv": "I vilket vattendrag, vilken sjö eller vilket hav mynnar vattendraget {s} ({c})?", "pl": "Do jakiej rzeki, jeziora lub morza uchodzi rzeka {s} ({c})?",
        "pt": "Em que rio, lago ou mar desagua o rio {s} ({c})?", "fi": "Mihin jokeen, järveen tai mereen joki {s} ({c}) laskee?"},
}
YEAR_TYPES = {"birth_year", "org_founded"}


def qa_prompt(qtype: str, lang: str, slots: dict[str, str]) -> str:
    return QA_TEMPLATES[qtype][lang].format(**slots) + "\n\n" + QA_INSTRUCTION[lang]


def grade_qa(prediction: str, label: dict[str, Any]) -> str:
    """'correct' | 'abstain' | 'wrong' | 'no_answer' (no ANSWER line, or an unclosed / misplaced trace)."""
    text = answer_part(prediction, label.get("mode", "default"))
    if text is None:
        return "no_answer"
    value = final_answer(text)
    if value is None:
        return "no_answer"
    if is_abstention(value):
        return "abstain"
    if label["kind"] == "year":
        years = set(re.findall(r"(?<!\d)(\d{3,4})(?!\d)", value))
        return "correct" if years == {str(label["answers"][0])} else "wrong"
    v = norm_text(value)
    if not v or len(v.split()) > 12:
        return "wrong"
    for alias in label["answers"]:
        a = norm_text(alias)
        if not a:
            continue
        if v == a:
            return "correct"
        # alias as a whole-word span of a short reply ("Uppsala, Sweden"), but not inside a list of guesses
        if re.search(rf"(?<!\w){re.escape(a)}(?!\w)", v) and len(v.split()) <= len(a.split()) + 3 and not re.search(r"\b(or|oder|ou|o|oppure|eller|lub|albo|tai)\b", v):
            return "correct"
    return "wrong"


QA_REWARD = {"correct": 1.0, "abstain": 0.0, "wrong": -1.0, "no_answer": -1.0}


def score_qa(prediction: str, label: dict[str, Any]) -> float:
    return QA_REWARD[grade_qa(prediction, label)]


# --------------------------------------------------------------------------------------------------------------------
# Suite 2: document tasks (eu_doc) - locale resources
# --------------------------------------------------------------------------------------------------------------------

LOCALE = {
    #     thousands, decimal, date format, currency code, currency symbol after?  VAT rates (reduced..standard)
    "en": dict(th=",", dec=".", date="{d:02d}/{m:02d}/{y}", cur="EUR", vat=[13.5, 23]),
    "de": dict(th=".", dec=",", date="{d:02d}.{m:02d}.{y}", cur="EUR", vat=[7, 19]),
    "fr": dict(th=" ", dec=",", date="{d:02d}/{m:02d}/{y}", cur="EUR", vat=[5.5, 10, 20]),
    "es": dict(th=".", dec=",", date="{d:02d}/{m:02d}/{y}", cur="EUR", vat=[10, 21]),
    "it": dict(th=".", dec=",", date="{d:02d}/{m:02d}/{y}", cur="EUR", vat=[10, 22]),
    "nl": dict(th=".", dec=",", date="{d:02d}-{m:02d}-{y}", cur="EUR", vat=[9, 21]),
    "sv": dict(th=" ", dec=",", date="{y}-{m:02d}-{d:02d}", cur="SEK", vat=[6, 12, 25]),
    "pl": dict(th=" ", dec=",", date="{d:02d}.{m:02d}.{y}", cur="PLN", vat=[8, 23]),
    "pt": dict(th=" ", dec=",", date="{d:02d}/{m:02d}/{y}", cur="EUR", vat=[6, 13, 23]),
    "fi": dict(th=" ", dec=",", date="{d}.{m}.{y}", cur="EUR", vat=[10, 14, 25.5]),
}

LEGAL = {"en": "Ltd", "de": "GmbH", "fr": "SARL", "es": "S.L.", "it": "S.r.l.", "nl": "B.V.", "sv": "AB", "pl": "Sp. z o.o.",
         "pt": "Lda.", "fi": "Oy"}
SYLL = ["nor", "vel", "tra", "ken", "lum", "sar", "bri", "dal", "mon", "ter", "fal", "rin", "ost", "gal", "pen", "vik"]

FIRST = {
    "en": ["Aoife", "Ciarán", "Siobhán", "Liam", "Niamh", "Conor", "Emma", "Seán", "Gráinne", "Patrick"],
    "de": ["Lukas", "Anna", "Jonas", "Lea", "Felix", "Marie", "Paul", "Sophie", "Tobias", "Katrin"],
    "fr": ["Camille", "Louis", "Chloé", "Hugo", "Léa", "Antoine", "Manon", "Julien", "Élodie", "Mathieu"],
    "es": ["Lucía", "Javier", "Carmen", "Diego", "Marta", "Alejandro", "Elena", "Pablo", "Sofía", "Andrés"],
    "it": ["Giulia", "Marco", "Chiara", "Luca", "Francesca", "Matteo", "Sara", "Alessandro", "Elisa", "Davide"],
    "nl": ["Sanne", "Daan", "Femke", "Bram", "Lotte", "Joost", "Eva", "Thijs", "Anouk", "Ruben"],
    "sv": ["Elin", "Johan", "Maja", "Erik", "Sara", "Oskar", "Ida", "Gustav", "Linnea", "Anders"],
    "pl": ["Zofia", "Jakub", "Katarzyna", "Piotr", "Agnieszka", "Tomasz", "Magdalena", "Michał", "Joanna", "Paweł"],
    "pt": ["Inês", "João", "Beatriz", "Tiago", "Mariana", "Rui", "Catarina", "Pedro", "Ana", "Miguel"],
    "fi": ["Aino", "Mikko", "Laura", "Juha", "Emilia", "Antti", "Sanna", "Ville", "Kaisa", "Timo"],
}
LAST = {
    "en": ["Murphy", "Kelly", "O'Brien", "Walsh", "Byrne", "Ryan", "Doyle", "Kavanagh", "Brennan", "Fitzgerald"],
    "de": ["Müller", "Schmidt", "Becker", "Hoffmann", "Wagner", "Schulz", "Keller", "Weber", "Krüger", "Lehmann"],
    "fr": ["Martin", "Bernard", "Dubois", "Lefèvre", "Moreau", "Girard", "Laurent", "Fournier", "Rousseau", "Mercier"],
    "es": ["García", "Fernández", "López", "Martínez", "Sánchez", "Romero", "Navarro", "Torres", "Ruiz", "Castillo"],
    "it": ["Rossi", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco", "Bruno", "Gallo", "Conti"],
    "nl": ["de Jong", "Jansen", "de Vries", "Bakker", "Visser", "Smit", "Meijer", "Mulder", "de Boer", "Bos"],
    "sv": ["Lindqvist", "Andersson", "Berg", "Nyström", "Holm", "Ekström", "Sandberg", "Lund", "Forsberg", "Åkesson"],
    "pl": ["Nowak", "Kowalski", "Wiśniewska", "Wójcik", "Kamiński", "Lewandowska", "Zieliński", "Szymańska", "Dąbrowski", "Kozłowska"],
    "pt": ["Silva", "Santos", "Ferreira", "Pereira", "Oliveira", "Costa", "Rodrigues", "Martins", "Sousa", "Gomes"],
    "fi": ["Korhonen", "Virtanen", "Mäkinen", "Nieminen", "Hämäläinen", "Laine", "Heikkinen", "Koskinen", "Järvinen", "Lehtonen"],
}
CITIES = {
    "en": ["Dublin", "Cork", "Galway", "Limerick", "Waterford"], "de": ["Berlin", "Hamburg", "München", "Köln", "Leipzig"],
    "fr": ["Paris", "Lyon", "Marseille", "Toulouse", "Nantes"], "es": ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao"],
    "it": ["Roma", "Milano", "Torino", "Napoli", "Bologna"], "nl": ["Amsterdam", "Rotterdam", "Utrecht", "Eindhoven", "Groningen"],
    "sv": ["Stockholm", "Göteborg", "Malmö", "Uppsala", "Umeå"], "pl": ["Warszawa", "Kraków", "Gdańsk", "Wrocław", "Poznań"],
    "pt": ["Lisboa", "Porto", "Braga", "Coimbra", "Faro"], "fi": ["Helsinki", "Tampere", "Turku", "Oulu", "Espoo"],
}
ITEMS = {
    "en": ["printer paper (box)", "toner cartridge", "office chair", "desk lamp", "USB-C cable", "keyboard", "coffee beans (1 kg)", "cleaning service (hour)"],
    "de": ["Druckerpapier (Karton)", "Tonerkartusche", "Bürostuhl", "Schreibtischlampe", "USB-C-Kabel", "Tastatur", "Kaffeebohnen (1 kg)", "Reinigungsservice (Stunde)"],
    "fr": ["papier d'imprimante (carton)", "cartouche de toner", "chaise de bureau", "lampe de bureau", "câble USB-C", "clavier", "café en grains (1 kg)", "service de nettoyage (heure)"],
    "es": ["papel de impresora (caja)", "cartucho de tóner", "silla de oficina", "lámpara de escritorio", "cable USB-C", "teclado", "café en grano (1 kg)", "servicio de limpieza (hora)"],
    "it": ["carta per stampante (scatola)", "cartuccia di toner", "sedia da ufficio", "lampada da scrivania", "cavo USB-C", "tastiera", "caffè in grani (1 kg)", "servizio di pulizia (ora)"],
    "nl": ["printerpapier (doos)", "tonercartridge", "bureaustoel", "bureaulamp", "USB-C-kabel", "toetsenbord", "koffiebonen (1 kg)", "schoonmaakdienst (uur)"],
    "sv": ["skrivarpapper (kartong)", "tonerkassett", "kontorsstol", "skrivbordslampa", "USB-C-kabel", "tangentbord", "kaffebönor (1 kg)", "städtjänst (timme)"],
    "pl": ["papier do drukarki (karton)", "kaseta z tonerem", "krzesło biurowe", "lampka biurkowa", "kabel USB-C", "klawiatura", "kawa ziarnista (1 kg)", "usługa sprzątania (godzina)"],
    "pt": ["papel de impressora (caixa)", "cartucho de toner", "cadeira de escritório", "candeeiro de secretária", "cabo USB-C", "teclado", "café em grão (1 kg)", "serviço de limpeza (hora)"],
    "fi": ["tulostinpaperi (laatikko)", "värikasetti", "työtuoli", "pöytävalaisin", "USB-C-kaapeli", "näppäimistö", "kahvipavut (1 kg)", "siivouspalvelu (tunti)"],
}
DEPTS = {
    "en": ["Sales", "Finance", "IT", "Logistics", "HR", "Marketing"], "de": ["Vertrieb", "Finanzen", "IT", "Logistik", "Personal", "Marketing"],
    "fr": ["Ventes", "Finance", "Informatique", "Logistique", "RH", "Marketing"], "es": ["Ventas", "Finanzas", "Informática", "Logística", "RR. HH.", "Marketing"],
    "it": ["Vendite", "Finanza", "Informatica", "Logistica", "Risorse umane", "Marketing"], "nl": ["Verkoop", "Financiën", "IT", "Logistiek", "HR", "Marketing"],
    "sv": ["Försäljning", "Ekonomi", "IT", "Logistik", "HR", "Marknad"], "pl": ["Sprzedaż", "Finanse", "IT", "Logistyka", "Kadry", "Marketing"],
    "pt": ["Vendas", "Finanças", "Informática", "Logística", "Recursos Humanos", "Marketing"], "fi": ["Myynti", "Talous", "IT", "Logistiikka", "Henkilöstö", "Markkinointi"],
}

# Document vocabulary: invoice, customer, date, item, qty, unit price (excl. VAT), VAT, employee table headers,
# contract headers, meeting headers.
VOCAB = {
    "en": dict(invoice="Invoice", customer="Customer", date="Date", item="Item", qty="Qty", unit="Unit price excl. VAT", vat="VAT",
               emp=["Name", "Department", "City", "Start date", "Monthly salary"], con=["Contract", "Counterparty", "Start date", "Duration (months)", "Monthly fee"],
               meet=["Meeting", "Room", "Date", "Start", "End"], employees="Employee register", contracts="Contract register", meetings="Room bookings",
               con_rule="A contract ends exactly the stated number of months after its start date (same day of the month)."),
    "de": dict(invoice="Rechnung", customer="Kunde", date="Datum", item="Artikel", qty="Menge", unit="Einzelpreis netto", vat="MwSt.",
               emp=["Name", "Abteilung", "Stadt", "Eintrittsdatum", "Monatsgehalt"], con=["Vertrag", "Vertragspartner", "Beginn", "Laufzeit (Monate)", "Monatliche Gebühr"],
               meet=["Termin", "Raum", "Datum", "Beginn", "Ende"], employees="Personalliste", contracts="Vertragsübersicht", meetings="Raumbuchungen",
               con_rule="Ein Vertrag endet genau die angegebene Zahl von Monaten nach seinem Beginn (am selben Tag des Monats)."),
    "fr": dict(invoice="Facture", customer="Client", date="Date", item="Article", qty="Qté", unit="Prix unitaire HT", vat="TVA",
               emp=["Nom", "Service", "Ville", "Date d'entrée", "Salaire mensuel"], con=["Contrat", "Cocontractant", "Début", "Durée (mois)", "Redevance mensuelle"],
               meet=["Réunion", "Salle", "Date", "Début", "Fin"], employees="Registre du personnel", contracts="Registre des contrats", meetings="Réservations de salles",
               con_rule="Un contrat se termine exactement le nombre de mois indiqué après sa date de début (le même jour du mois)."),
    "es": dict(invoice="Factura", customer="Cliente", date="Fecha", item="Artículo", qty="Cant.", unit="Precio unitario sin IVA", vat="IVA",
               emp=["Nombre", "Departamento", "Ciudad", "Fecha de alta", "Salario mensual"], con=["Contrato", "Contraparte", "Inicio", "Duración (meses)", "Cuota mensual"],
               meet=["Reunión", "Sala", "Fecha", "Inicio", "Fin"], employees="Registro de personal", contracts="Registro de contratos", meetings="Reservas de salas",
               con_rule="Un contrato termina exactamente el número de meses indicado después de su fecha de inicio (el mismo día del mes)."),
    "it": dict(invoice="Fattura", customer="Cliente", date="Data", item="Articolo", qty="Q.tà", unit="Prezzo unitario IVA esclusa", vat="IVA",
               emp=["Nome", "Reparto", "Città", "Data di assunzione", "Stipendio mensile"], con=["Contratto", "Controparte", "Inizio", "Durata (mesi)", "Canone mensile"],
               meet=["Riunione", "Sala", "Data", "Inizio", "Fine"], employees="Registro del personale", contracts="Registro dei contratti", meetings="Prenotazioni sale",
               con_rule="Un contratto termina esattamente il numero di mesi indicato dopo la data di inizio (lo stesso giorno del mese)."),
    "nl": dict(invoice="Factuur", customer="Klant", date="Datum", item="Artikel", qty="Aantal", unit="Stukprijs excl. btw", vat="Btw",
               emp=["Naam", "Afdeling", "Stad", "Startdatum", "Maandsalaris"], con=["Contract", "Wederpartij", "Begin", "Looptijd (maanden)", "Maandelijkse vergoeding"],
               meet=["Vergadering", "Zaal", "Datum", "Begin", "Einde"], employees="Personeelsregister", contracts="Contractenregister", meetings="Zaalreserveringen",
               con_rule="Een contract eindigt precies het genoemde aantal maanden na de begindatum (op dezelfde dag van de maand)."),
    "sv": dict(invoice="Faktura", customer="Kund", date="Datum", item="Artikel", qty="Antal", unit="À-pris exkl. moms", vat="Moms",
               emp=["Namn", "Avdelning", "Ort", "Anställningsdatum", "Månadslön"], con=["Avtal", "Motpart", "Start", "Löptid (månader)", "Månadsavgift"],
               meet=["Möte", "Rum", "Datum", "Start", "Slut"], employees="Personalförteckning", contracts="Avtalsförteckning", meetings="Rumsbokningar",
               con_rule="Ett avtal upphör exakt det angivna antalet månader efter startdatumet (samma dag i månaden)."),
    "pl": dict(invoice="Faktura", customer="Klient", date="Data", item="Pozycja", qty="Ilość", unit="Cena jedn. netto", vat="VAT",
               emp=["Imię i nazwisko", "Dział", "Miasto", "Data zatrudnienia", "Pensja miesięczna"], con=["Umowa", "Kontrahent", "Początek", "Okres (miesiące)", "Opłata miesięczna"],
               meet=["Spotkanie", "Sala", "Data", "Początek", "Koniec"], employees="Rejestr pracowników", contracts="Rejestr umów", meetings="Rezerwacje sal",
               con_rule="Umowa kończy się dokładnie po podanej liczbie miesięcy od daty rozpoczęcia (tego samego dnia miesiąca)."),
    "pt": dict(invoice="Fatura", customer="Cliente", date="Data", item="Artigo", qty="Qtd.", unit="Preço unitário s/ IVA", vat="IVA",
               emp=["Nome", "Departamento", "Cidade", "Data de admissão", "Salário mensal"], con=["Contrato", "Contraparte", "Início", "Duração (meses)", "Mensalidade"],
               meet=["Reunião", "Sala", "Data", "Início", "Fim"], employees="Registo de pessoal", contracts="Registo de contratos", meetings="Reservas de salas",
               con_rule="Um contrato termina exatamente o número de meses indicado após a data de início (no mesmo dia do mês)."),
    "fi": dict(invoice="Lasku", customer="Asiakas", date="Päivämäärä", item="Tuote", qty="Määrä", unit="Yksikköhinta (alv 0 %)", vat="ALV",
               emp=["Nimi", "Osasto", "Kaupunki", "Aloituspäivä", "Kuukausipalkka"], con=["Sopimus", "Vastapuoli", "Alkaa", "Kesto (kk)", "Kuukausimaksu"],
               meet=["Kokous", "Huone", "Päivämäärä", "Alkaa", "Päättyy"], employees="Henkilöstöluettelo", contracts="Sopimusluettelo", meetings="Huonevaraukset",
               con_rule="Sopimus päättyy täsmälleen ilmoitetun kuukausimäärän kuluttua alkamispäivästä (samana kuukaudenpäivänä)."),
}

DOC_INSTRUCTION = {
    "en": "Use only the information above. Give the final answer on a last line in the form ANSWER: <answer>.",
    "de": "Verwende nur die obigen Angaben. Gib die endgültige Antwort in einer letzten Zeile der Form ANSWER: <Antwort> an.",
    "fr": "N'utilise que les informations ci-dessus. Donne la réponse finale sur une dernière ligne de la forme ANSWER: <réponse>.",
    "es": "Usa solo la información anterior. Da la respuesta final en una última línea con el formato ANSWER: <respuesta>.",
    "it": "Usa solo le informazioni sopra riportate. Dai la risposta finale in un'ultima riga nella forma ANSWER: <risposta>.",
    "nl": "Gebruik alleen de bovenstaande informatie. Geef het eindantwoord op een laatste regel in de vorm ANSWER: <antwoord>.",
    "sv": "Använd endast informationen ovan. Ge det slutliga svaret på en sista rad på formen ANSWER: <svar>.",
    "pl": "Korzystaj wyłącznie z powyższych informacji. Podaj ostateczną odpowiedź w ostatniej linii w formie ANSWER: <odpowiedź>.",
    "pt": "Usa apenas a informação acima. Dá a resposta final numa última linha no formato ANSWER: <resposta>.",
    "fi": "Käytä vain yllä olevia tietoja. Anna lopullinen vastaus viimeisellä rivillä muodossa ANSWER: <vastaus>.",
}

# Answer-format hints appended to questions: money (2 decimals, currency), integer, ISO date, ID list, single ID, name
HINT = {
    "en": dict(money="Give the amount in {cur} with two decimals.", int="Give a whole number.", date="Give the date as YYYY-MM-DD.",
               ids="List the IDs separated by commas.", id="Give the ID.", name="Give the full name."),
    "de": dict(money="Gib den Betrag in {cur} mit zwei Nachkommastellen an.", int="Gib eine ganze Zahl an.", date="Gib das Datum im Format JJJJ-MM-TT (YYYY-MM-DD) an.",
               ids="Nenne die IDs durch Kommas getrennt.", id="Nenne die ID.", name="Nenne den vollständigen Namen."),
    "fr": dict(money="Donne le montant en {cur} avec deux décimales.", int="Donne un nombre entier.", date="Donne la date au format AAAA-MM-JJ (YYYY-MM-DD).",
               ids="Liste les identifiants séparés par des virgules.", id="Donne l'identifiant.", name="Donne le nom complet."),
    "es": dict(money="Da el importe en {cur} con dos decimales.", int="Da un número entero.", date="Da la fecha en formato AAAA-MM-DD (YYYY-MM-DD).",
               ids="Enumera los identificadores separados por comas.", id="Da el identificador.", name="Da el nombre completo."),
    "it": dict(money="Indica l'importo in {cur} con due decimali.", int="Indica un numero intero.", date="Indica la data nel formato AAAA-MM-GG (YYYY-MM-DD).",
               ids="Elenca gli ID separati da virgole.", id="Indica l'ID.", name="Indica il nome completo."),
    "nl": dict(money="Geef het bedrag in {cur} met twee decimalen.", int="Geef een geheel getal.", date="Geef de datum als JJJJ-MM-DD (YYYY-MM-DD).",
               ids="Noem de ID's, gescheiden door komma's.", id="Geef de ID.", name="Geef de volledige naam."),
    "sv": dict(money="Ange beloppet i {cur} med två decimaler.", int="Ange ett heltal.", date="Ange datumet som ÅÅÅÅ-MM-DD (YYYY-MM-DD).",
               ids="Ange id:na separerade med kommatecken.", id="Ange id:t.", name="Ange det fullständiga namnet."),
    "pl": dict(money="Podaj kwotę w {cur} z dwoma miejscami po przecinku.", int="Podaj liczbę całkowitą.", date="Podaj datę w formacie RRRR-MM-DD (YYYY-MM-DD).",
               ids="Wymień identyfikatory oddzielone przecinkami.", id="Podaj identyfikator.", name="Podaj pełne imię i nazwisko."),
    "pt": dict(money="Indica o montante em {cur} com duas casas decimais.", int="Indica um número inteiro.", date="Indica a data no formato AAAA-MM-DD (YYYY-MM-DD).",
               ids="Lista os identificadores separados por vírgulas.", id="Indica o identificador.", name="Indica o nome completo."),
    "fi": dict(money="Anna summa valuutassa {cur} kahden desimaalin tarkkuudella.", int="Anna kokonaisluku.", date="Anna päivämäärä muodossa VVVV-KK-PP (YYYY-MM-DD).",
               ids="Luettele tunnisteet pilkuilla erotettuina.", id="Anna tunniste.", name="Anna koko nimi."),
}

# Questions per family and variant. Slots: {cust} customer, {dept}, {city}, {date} (locale-formatted), {room}, {con}
QUESTIONS = {
    "invoice": {
        "gross_total": {
            "en": "What is the total amount including VAT of all invoices together?", "de": "Wie hoch ist der Gesamtbetrag inklusive MwSt. aller Rechnungen zusammen?",
            "fr": "Quel est le montant total TTC de toutes les factures réunies ?", "es": "¿Cuál es el importe total con IVA de todas las facturas juntas?",
            "it": "Qual è l'importo totale IVA inclusa di tutte le fatture insieme?", "nl": "Wat is het totaalbedrag inclusief btw van alle facturen samen?",
            "sv": "Vad är det sammanlagda beloppet inklusive moms för alla fakturor?", "pl": "Jaka jest łączna kwota brutto (z VAT) wszystkich faktur razem?",
            "pt": "Qual é o montante total com IVA de todas as faturas juntas?", "fi": "Mikä on kaikkien laskujen yhteissumma arvonlisäveroineen?"},
        "vat_total": {
            "en": "How much VAT is charged in total across all invoices?", "de": "Wie viel MwSt. wird über alle Rechnungen insgesamt berechnet?",
            "fr": "Quel est le montant total de TVA facturé sur l'ensemble des factures ?", "es": "¿Cuánto IVA se cobra en total en todas las facturas?",
            "it": "Quanta IVA viene addebitata in totale su tutte le fatture?", "nl": "Hoeveel btw wordt er in totaal over alle facturen berekend?",
            "sv": "Hur mycket moms debiteras totalt på alla fakturor?", "pl": "Ile łącznie wynosi VAT na wszystkich fakturach?",
            "pt": "Quanto IVA é cobrado no total em todas as faturas?", "fi": "Paljonko arvonlisäveroa laskutetaan yhteensä kaikilla laskuilla?"},
        "max_invoice": {
            "en": "Which invoice has the highest total including VAT?", "de": "Welche Rechnung hat den höchsten Gesamtbetrag inklusive MwSt.?",
            "fr": "Quelle facture a le montant TTC le plus élevé ?", "es": "¿Qué factura tiene el importe total con IVA más alto?",
            "it": "Quale fattura ha l'importo totale IVA inclusa più alto?", "nl": "Welke factuur heeft het hoogste totaalbedrag inclusief btw?",
            "sv": "Vilken faktura har det högsta totalbeloppet inklusive moms?", "pl": "Która faktura ma najwyższą kwotę brutto?",
            "pt": "Qual fatura tem o montante total com IVA mais elevado?", "fi": "Millä laskulla on suurin kokonaissumma arvonlisäveroineen?"},
        "customer_net": {
            "en": "What is the total amount excluding VAT invoiced to {cust}?", "de": "Wie hoch ist der Nettobetrag (ohne MwSt.), der {cust} insgesamt in Rechnung gestellt wurde?",
            "fr": "Quel est le montant total HT facturé à {cust} ?", "es": "¿Cuál es el importe total sin IVA facturado a {cust}?",
            "it": "Qual è l'importo totale IVA esclusa fatturato a {cust}?", "nl": "Wat is het totaalbedrag exclusief btw dat aan {cust} is gefactureerd?",
            "sv": "Vad är det sammanlagda beloppet exklusive moms som fakturerats {cust}?", "pl": "Jaka jest łączna kwota netto zafakturowana dla {cust}?",
            "pt": "Qual é o montante total sem IVA faturado a {cust}?", "fi": "Mikä on asiakkaalle {cust} laskutettu kokonaissumma ilman arvonlisäveroa?"},
    },
    "table": {
        "count_filter": {
            "en": "How many employees work in {dept} in {city}?", "de": "Wie viele Beschäftigte arbeiten in der Abteilung {dept} in {city}?",
            "fr": "Combien d'employés travaillent au service {dept} à {city} ?", "es": "¿Cuántos empleados trabajan en {dept} en {city}?",
            "it": "Quanti dipendenti lavorano nel reparto {dept} a {city}?", "nl": "Hoeveel medewerkers werken bij {dept} in {city}?",
            "sv": "Hur många anställda arbetar på {dept} i {city}?", "pl": "Ilu pracowników pracuje w dziale {dept} w mieście {city}?",
            "pt": "Quantos funcionários trabalham em {dept} em {city}?", "fi": "Kuinka moni työntekijä työskentelee osastolla {dept} kaupungissa {city}?"},
        "sum_dept": {
            "en": "What is the sum of the monthly salaries in {dept}?", "de": "Wie hoch ist die Summe der Monatsgehälter in der Abteilung {dept}?",
            "fr": "Quelle est la somme des salaires mensuels du service {dept} ?", "es": "¿Cuál es la suma de los salarios mensuales de {dept}?",
            "it": "Qual è la somma degli stipendi mensili del reparto {dept}?", "nl": "Wat is de som van de maandsalarissen bij {dept}?",
            "sv": "Vad är summan av månadslönerna på {dept}?", "pl": "Ile wynosi suma pensji miesięcznych w dziale {dept}?",
            "pt": "Qual é a soma dos salários mensais em {dept}?", "fi": "Mikä on osaston {dept} kuukausipalkkojen summa?"},
        "earliest": {
            "en": "Who in {dept} has the earliest start date?", "de": "Wer in der Abteilung {dept} hat das früheste Eintrittsdatum?",
            "fr": "Qui, au service {dept}, a la date d'entrée la plus ancienne ?", "es": "¿Quién de {dept} tiene la fecha de alta más antigua?",
            "it": "Chi, nel reparto {dept}, ha la data di assunzione più vecchia?", "nl": "Wie bij {dept} heeft de vroegste startdatum?",
            "sv": "Vem på {dept} har det tidigaste anställningsdatumet?", "pl": "Kto w dziale {dept} ma najwcześniejszą datę zatrudnienia?",
            "pt": "Quem em {dept} tem a data de admissão mais antiga?", "fi": "Kenellä osastolla {dept} on aikaisin aloituspäivä?"},
        "avg_dept": {
            "en": "What is the average monthly salary in {dept}, rounded to two decimals?", "de": "Wie hoch ist das durchschnittliche Monatsgehalt in der Abteilung {dept}, auf zwei Nachkommastellen gerundet?",
            "fr": "Quel est le salaire mensuel moyen du service {dept}, arrondi à deux décimales ?", "es": "¿Cuál es el salario mensual medio de {dept}, redondeado a dos decimales?",
            "it": "Qual è lo stipendio mensile medio del reparto {dept}, arrotondato a due decimali?", "nl": "Wat is het gemiddelde maandsalaris bij {dept}, afgerond op twee decimalen?",
            "sv": "Vad är medelmånadslönen på {dept}, avrundad till två decimaler?", "pl": "Ile wynosi średnia pensja miesięczna w dziale {dept}, zaokrąglona do dwóch miejsc po przecinku?",
            "pt": "Qual é o salário mensal médio em {dept}, arredondado a duas casas decimais?", "fi": "Mikä on osaston {dept} keskimääräinen kuukausipalkka kahden desimaalin tarkkuudella?"},
        "count_after": {
            "en": "How many employees started after {date}?", "de": "Wie viele Beschäftigte sind nach dem {date} eingetreten?",
            "fr": "Combien d'employés sont entrés après le {date} ?", "es": "¿Cuántos empleados empezaron después del {date}?",
            "it": "Quanti dipendenti sono stati assunti dopo il {date}?", "nl": "Hoeveel medewerkers zijn na {date} begonnen?",
            "sv": "Hur många anställdes efter {date}?", "pl": "Ilu pracowników zatrudniono po {date}?",
            "pt": "Quantos funcionários foram admitidos depois de {date}?", "fi": "Kuinka moni työntekijä aloitti {date} jälkeen?"},
    },
    "contract": {
        "end_date": {
            "en": "On which date does contract {con} end?", "de": "An welchem Datum endet der Vertrag {con}?", "fr": "À quelle date le contrat {con} se termine-t-il ?",
            "es": "¿En qué fecha termina el contrato {con}?", "it": "In quale data termina il contratto {con}?", "nl": "Op welke datum eindigt contract {con}?",
            "sv": "Vilket datum upphör avtal {con}?", "pl": "W jakim dniu kończy się umowa {con}?", "pt": "Em que data termina o contrato {con}?",
            "fi": "Minä päivänä sopimus {con} päättyy?"},
        "ending_before": {
            "en": "Which contracts end before {date}?", "de": "Welche Verträge enden vor dem {date}?", "fr": "Quels contrats se terminent avant le {date} ?",
            "es": "¿Qué contratos terminan antes del {date}?", "it": "Quali contratti terminano prima del {date}?", "nl": "Welke contracten eindigen vóór {date}?",
            "sv": "Vilka avtal upphör före {date}?", "pl": "Które umowy kończą się przed {date}?", "pt": "Que contratos terminam antes de {date}?",
            "fi": "Mitkä sopimukset päättyvät ennen {date}?"},
        "last_ending": {
            "en": "Which contract ends last?", "de": "Welcher Vertrag endet zuletzt?", "fr": "Quel contrat se termine en dernier ?", "es": "¿Qué contrato termina el último?",
            "it": "Quale contratto termina per ultimo?", "nl": "Welk contract eindigt als laatste?", "sv": "Vilket avtal upphör sist?", "pl": "Która umowa kończy się najpóźniej?",
            "pt": "Que contrato termina em último lugar?", "fi": "Mikä sopimus päättyy viimeisenä?"},
    },
    "meeting": {
        "overlap": {
            "en": "Two meetings in the same room overlap in time. Which ones?", "de": "Zwei Termine im selben Raum überschneiden sich zeitlich. Welche?",
            "fr": "Deux réunions dans la même salle se chevauchent. Lesquelles ?", "es": "Dos reuniones en la misma sala se solapan en el tiempo. ¿Cuáles?",
            "it": "Due riunioni nella stessa sala si sovrappongono. Quali?", "nl": "Twee vergaderingen in dezelfde zaal overlappen in tijd. Welke?",
            "sv": "Två möten i samma rum överlappar i tid. Vilka?", "pl": "Dwa spotkania w tej samej sali nakładają się w czasie. Które?",
            "pt": "Duas reuniões na mesma sala sobrepõem-se no tempo. Quais?", "fi": "Kaksi samassa huoneessa pidettävää kokousta menee päällekkäin. Mitkä?"},
        "booked_minutes": {
            "en": "How many minutes in total is room {room} booked on {date}?", "de": "Wie viele Minuten ist Raum {room} am {date} insgesamt gebucht?",
            "fr": "Combien de minutes au total la salle {room} est-elle réservée le {date} ?", "es": "¿Cuántos minutos en total está reservada la sala {room} el {date}?",
            "it": "Per quanti minuti in totale è prenotata la sala {room} il {date}?", "nl": "Hoeveel minuten in totaal is zaal {room} op {date} gereserveerd?",
            "sv": "Hur många minuter totalt är rum {room} bokat den {date}?", "pl": "Ile minut łącznie sala {room} jest zarezerwowana w dniu {date}?",
            "pt": "Quantos minutos no total está a sala {room} reservada em {date}?", "fi": "Kuinka monta minuuttia huone {room} on yhteensä varattu {date}?"},
    },
}

HELD_OUT = {"invoice": "customer_net", "table": "avg_dept", "contract": "last_ending", "meeting": "booked_minutes"}

LETTER = {
    "en": ("Dear Sir or Madam,\n\nI am writing about my order of {od}. On {pd} I had already ordered goods for {pa}, which arrived without problems. "
           "This time, however, the delivery has not reached me. The order amount was {oa}. My customer number is {cn}.\n\n"
           "I would be grateful for a quick reply.\n\nKind regards,\n{name}\n{street}\n{city}"),
    "de": ("Sehr geehrte Damen und Herren,\n\nich schreibe Ihnen wegen meiner Bestellung vom {od}. Bereits am {pd} hatte ich Waren für {pa} bestellt, die problemlos ankamen. "
           "Diesmal ist die Lieferung jedoch nicht bei mir eingetroffen. Der Bestellbetrag betrug {oa}. Meine Kundennummer lautet {cn}.\n\n"
           "Über eine baldige Antwort würde ich mich freuen.\n\nMit freundlichen Grüßen\n{name}\n{street}\n{city}"),
    "fr": ("Madame, Monsieur,\n\nJe vous écris au sujet de ma commande du {od}. J'avais déjà commandé le {pd} des articles pour {pa}, qui sont arrivés sans problème. "
           "Cette fois, en revanche, la livraison ne m'est pas parvenue. Le montant de la commande était de {oa}. Mon numéro de client est {cn}.\n\n"
           "Je vous remercie par avance de votre réponse rapide.\n\nCordialement,\n{name}\n{street}\n{city}"),
    "es": ("Estimados señores:\n\nLes escribo en relación con mi pedido del {od}. El {pd} ya había pedido artículos por {pa}, que llegaron sin problemas. "
           "Esta vez, sin embargo, el envío no me ha llegado. El importe del pedido fue de {oa}. Mi número de cliente es {cn}.\n\n"
           "Les agradecería una respuesta rápida.\n\nAtentamente,\n{name}\n{street}\n{city}"),
    "it": ("Gentili signori,\n\nvi scrivo in merito al mio ordine del {od}. Il {pd} avevo già ordinato articoli per {pa}, arrivati senza problemi. "
           "Questa volta, però, la consegna non mi è arrivata. L'importo dell'ordine era di {oa}. Il mio numero cliente è {cn}.\n\n"
           "Vi sarei grato per una risposta rapida.\n\nCordiali saluti,\n{name}\n{street}\n{city}"),
    "nl": ("Geachte heer, mevrouw,\n\nIk schrijf u over mijn bestelling van {od}. Op {pd} had ik al artikelen besteld voor {pa}, die zonder problemen aankwamen. "
           "Deze keer is de levering echter niet bij mij aangekomen. Het bestelbedrag was {oa}. Mijn klantnummer is {cn}.\n\n"
           "Ik zie uw spoedige reactie graag tegemoet.\n\nMet vriendelijke groet,\n{name}\n{street}\n{city}"),
    "sv": ("Hej,\n\nJag skriver angående min beställning den {od}. Den {pd} hade jag redan beställt varor för {pa}, som kom fram utan problem. "
           "Den här gången har leveransen dock inte nått mig. Ordersumman var {oa}. Mitt kundnummer är {cn}.\n\n"
           "Jag vore tacksam för ett snabbt svar.\n\nVänliga hälsningar\n{name}\n{street}\n{city}"),
    "pl": ("Szanowni Państwo,\n\npiszę w sprawie mojego zamówienia z dnia {od}. Już {pd} zamówiłem towary na kwotę {pa}, które dotarły bez problemu. "
           "Tym razem jednak przesyłka do mnie nie dotarła. Kwota zamówienia wynosiła {oa}. Mój numer klienta to {cn}.\n\n"
           "Będę wdzięczny za szybką odpowiedź.\n\nZ poważaniem\n{name}\n{street}\n{city}"),
    "pt": ("Exmos. Senhores,\n\nEscrevo a propósito da minha encomenda de {od}. Em {pd} já tinha encomendado artigos no valor de {pa}, que chegaram sem problemas. "
           "Desta vez, porém, a entrega não me chegou. O valor da encomenda foi de {oa}. O meu número de cliente é {cn}.\n\n"
           "Agradeço uma resposta rápida.\n\nCom os melhores cumprimentos,\n{name}\n{street}\n{city}"),
    "fi": ("Hyvä vastaanottaja,\n\nkirjoitan tilauksestani, jonka tein {od}. Olin jo {pd} tilannut tuotteita {pa} arvosta, ja ne saapuivat ongelmitta. "
           "Tällä kertaa toimitus ei kuitenkaan ole saapunut minulle. Tilauksen summa oli {oa}. Asiakasnumeroni on {cn}.\n\n"
           "Toivon pikaista vastausta.\n\nYstävällisin terveisin\n{name}\n{street}\n{city}"),
}
STREET = {"en": "{n} Main Street", "de": "Hauptstraße {n}", "fr": "{n} rue de la Gare", "es": "Calle Mayor {n}", "it": "Via Roma {n}",
          "nl": "Dorpsstraat {n}", "sv": "Storgatan {n}", "pl": "ul. Długa {n}", "pt": "Rua Direita {n}", "fi": "Kauppakatu {n}"}
FORM_PROMPT = {
    "en": "Extract the following fields from the letter above as a JSON object with exactly these keys: name, customer_number, order_date (YYYY-MM-DD, the date of the order the letter is about), amount (number, the amount of that order), city. Reply with the JSON object only.",
    "de": "Extrahiere aus dem obigen Brief folgende Felder als JSON-Objekt mit genau diesen Schlüsseln: name, customer_number, order_date (YYYY-MM-DD, das Datum der Bestellung, um die es im Brief geht), amount (Zahl, der Betrag dieser Bestellung), city. Antworte nur mit dem JSON-Objekt.",
    "fr": "Extrais de la lettre ci-dessus les champs suivants sous forme d'objet JSON avec exactement ces clés : name, customer_number, order_date (YYYY-MM-DD, la date de la commande dont parle la lettre), amount (nombre, le montant de cette commande), city. Réponds uniquement avec l'objet JSON.",
    "es": "Extrae de la carta anterior los siguientes campos como un objeto JSON con exactamente estas claves: name, customer_number, order_date (YYYY-MM-DD, la fecha del pedido del que trata la carta), amount (número, el importe de ese pedido), city. Responde solo con el objeto JSON.",
    "it": "Estrai dalla lettera sopra i seguenti campi come oggetto JSON con esattamente queste chiavi: name, customer_number, order_date (YYYY-MM-DD, la data dell'ordine di cui parla la lettera), amount (numero, l'importo di quell'ordine), city. Rispondi solo con l'oggetto JSON.",
    "nl": "Haal uit de bovenstaande brief de volgende velden als JSON-object met precies deze sleutels: name, customer_number, order_date (YYYY-MM-DD, de datum van de bestelling waarover de brief gaat), amount (getal, het bedrag van die bestelling), city. Antwoord alleen met het JSON-object.",
    "sv": "Extrahera följande fält ur brevet ovan som ett JSON-objekt med exakt dessa nycklar: name, customer_number, order_date (YYYY-MM-DD, datumet för den beställning brevet gäller), amount (tal, beloppet för den beställningen), city. Svara endast med JSON-objektet.",
    "pl": "Wyodrębnij z powyższego listu następujące pola jako obiekt JSON z dokładnie tymi kluczami: name, customer_number, order_date (YYYY-MM-DD, data zamówienia, którego dotyczy list), amount (liczba, kwota tego zamówienia), city. Odpowiedz wyłącznie obiektem JSON.",
    "pt": "Extrai da carta acima os seguintes campos como um objeto JSON com exatamente estas chaves: name, customer_number, order_date (YYYY-MM-DD, a data da encomenda de que trata a carta), amount (número, o valor dessa encomenda), city. Responde apenas com o objeto JSON.",
    "fi": "Poimi yllä olevasta kirjeestä seuraavat kentät JSON-objektina, jossa on täsmälleen nämä avaimet: name, customer_number, order_date (YYYY-MM-DD, sen tilauksen päivämäärä, jota kirje koskee), amount (luku, kyseisen tilauksen summa), city. Vastaa pelkällä JSON-objektilla.",
}

FAMILIES = ["invoice", "table", "contract", "meeting", "form"]
LEVEL_SIZES = {"invoice": [2, 3, 4], "table": [8, 14, 22], "contract": [4, 7, 10], "meeting": [5, 8, 12]}


def fmt_money(x: float, lang: str) -> str:
    loc = LOCALE[lang]
    whole, frac = f"{x:,.2f}".split(".")
    return whole.replace(",", loc["th"]) + loc["dec"] + frac


def fmt_rate(r: float, lang: str) -> str:
    s = f"{r:g}"
    return (s.replace(".", LOCALE[lang]["dec"])) + " %" if lang in ("de", "fr", "sv", "pl", "fi") else s.replace(".", LOCALE[lang]["dec"]) + "%"


def fmt_date(d: dt.date, lang: str) -> str:
    return LOCALE[lang]["date"].format(d=d.day, m=d.month, y=d.year)


def add_months(d: dt.date, months: int) -> dt.date:
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def company(rng: random.Random, lang: str) -> str:
    name = "".join(rng.sample(SYLL, 2)).capitalize()
    return f"{name} {LEGAL[lang]}"


def person(rng: random.Random, lang: str) -> str:
    return f"{rng.choice(FIRST[lang])} {rng.choice(LAST[lang])}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def rand_date(rng: random.Random, start: dt.date, span_days: int, max_day: int = 28) -> dt.date:
    while True:
        d = start + dt.timedelta(days=rng.randrange(span_days))
        if d.day <= max_day:
            return d


def money_value(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 2)


def _hint(lang: str, key: str) -> str:
    return HINT[lang][key].format(cur=LOCALE[lang]["cur"])


def gen_invoice(rng: random.Random, lang: str, level: int, variant: str) -> tuple[str, dict]:
    v, loc = VOCAB[lang], LOCALE[lang]
    customers = [company(rng, lang) for _ in range(2)]
    invoices = []
    base_no = rng.randrange(1000, 9000)
    for k in range(LEVEL_SIZES["invoice"][level - 1]):
        lines = []
        for _ in range(rng.randint(2, 2 + level)):
            lines.append((rng.choice(ITEMS[lang]), rng.randint(1, 12), money_value(rng, 3, 400), rng.choice(loc["vat"])))
        invoices.append((f"INV-{base_no + k * rng.randint(1, 7)}", customers[k % 2] if k < 2 else rng.choice(customers),
                         rand_date(rng, dt.date(2025, 1, 1), 600), lines))
    if len({i[0] for i in invoices}) != len(invoices):
        return gen_invoice(rng, lang, level, variant)
    parts, nets, vats = [], [], []
    for no, cust, date, lines in invoices:
        net = sum(round(q * p, 2) for _, q, p, _ in lines)
        vat = sum(round(round(q * p, 2) * r / 100, 2) for _, q, p, r in lines)
        nets.append(round(net, 2)); vats.append(round(vat, 2))
        rows = [[it, str(q), fmt_money(p, lang), fmt_rate(r, lang)] for it, q, p, r in lines]
        parts.append(f"### {v['invoice']} {no}\n{v['customer']}: {cust}\n{v['date']}: {fmt_date(date, lang)}\n\n"
                     + md_table([v["item"], v["qty"], f"{v['unit']} ({loc['cur']})", v["vat"]], rows))
    rule = {"en": "VAT is calculated per line on the line amount (quantity × unit price) and rounded to cents.",
            "de": "Die MwSt. wird je Position auf den Positionsbetrag (Menge × Einzelpreis) berechnet und auf Cent gerundet.",
            "fr": "La TVA est calculée par ligne sur le montant de la ligne (quantité × prix unitaire) et arrondie au centime.",
            "es": "El IVA se calcula por línea sobre el importe de la línea (cantidad × precio unitario) y se redondea al céntimo.",
            "it": "L'IVA è calcolata per riga sull'importo della riga (quantità × prezzo unitario) e arrotondata al centesimo.",
            "nl": "De btw wordt per regel berekend over het regelbedrag (aantal × stukprijs) en afgerond op centen.",
            "sv": "Momsen beräknas per rad på radbeloppet (antal × à-pris) och avrundas till hela ören.",
            "pl": "VAT oblicza się dla każdej pozycji od jej wartości (ilość × cena jednostkowa) i zaokrągla do groszy.",
            "pt": "O IVA é calculado por linha sobre o valor da linha (quantidade × preço unitário) e arredondado ao cêntimo.",
            "fi": "Arvonlisävero lasketaan riveittäin rivin summasta (määrä × yksikköhinta) ja pyöristetään senttiin."}[lang]
    doc = "\n\n".join(parts) + "\n\n" + rule
    q = QUESTIONS["invoice"][variant][lang]
    if variant == "gross_total":
        label, hint = {"kind": "number", "value": round(sum(nets) + sum(vats), 2), "tol": 0.011}, "money"
    elif variant == "vat_total":
        label, hint = {"kind": "number", "value": round(sum(vats), 2), "tol": 0.011}, "money"
    elif variant == "max_invoice":
        gross = [n + t for n, t in zip(nets, vats)]
        if sorted(gross)[-1] - sorted(gross)[-2] < 1:
            return gen_invoice(rng, lang, level, variant)
        label, hint = {"kind": "id", "value": invoices[gross.index(max(gross))][0], "pattern": r"INV-\d+"}, "id"
    else:
        cust = rng.choice([c for c in customers if any(i[1] == c for i in invoices)])
        q = q.format(cust=cust)
        label, hint = {"kind": "number", "value": round(sum(n for n, i in zip(nets, invoices) if i[1] == cust), 2), "tol": 0.011}, "money"
    return f"{doc}\n\n{q} {_hint(lang, hint)}\n\n{DOC_INSTRUCTION[lang]}", label


def gen_table(rng: random.Random, lang: str, level: int, variant: str) -> tuple[str, dict]:
    v = VOCAB[lang]
    n = LEVEL_SIZES["table"][level - 1]
    depts, cities = DEPTS[lang][: 3 + level], CITIES[lang][: 2 + level]
    names = set()
    rows = []
    while len(rows) < n:
        name = person(rng, lang)
        if name in names:
            continue
        names.add(name)
        rows.append((name, rng.choice(depts), rng.choice(cities), rand_date(rng, dt.date(2008, 1, 1), 6200, 31), rng.randrange(2400, 7800, 10) + rng.choice([0, 0, 0.5])))
    doc = f"### {v['employees']} ({LOCALE[lang]['cur']})\n\n" + md_table(v["emp"], [[r[0], r[1], r[2], fmt_date(r[3], lang), fmt_money(r[4], lang)] for r in rows])
    q = QUESTIONS["table"][variant][lang]
    dept = rng.choice(sorted({r[1] for r in rows}))
    in_dept = [r for r in rows if r[1] == dept]
    if variant == "count_filter":
        city = rng.choice(cities)
        q = q.format(dept=dept, city=city)
        label, hint = {"kind": "number", "value": sum(1 for r in rows if r[1] == dept and r[2] == city), "tol": 0}, "int"
    elif variant == "sum_dept":
        q = q.format(dept=dept)
        label, hint = {"kind": "number", "value": round(sum(r[4] for r in in_dept), 2), "tol": 0.011}, "money"
    elif variant == "earliest":
        dates = sorted(r[3] for r in in_dept)
        if len(dates) > 1 and dates[0] == dates[1]:
            return gen_table(rng, lang, level, variant)
        q = q.format(dept=dept)
        label, hint = {"kind": "name", "value": min(in_dept, key=lambda r: r[3])[0]}, "name"
    elif variant == "avg_dept":
        q = q.format(dept=dept)
        label, hint = {"kind": "number", "value": round(sum(r[4] for r in in_dept) / len(in_dept), 2), "tol": 0.011}, "money"
    else:
        cut = rand_date(rng, dt.date(2012, 1, 1), 3600, 31)
        if any(r[3] == cut for r in rows):
            return gen_table(rng, lang, level, variant)
        q = q.format(date=fmt_date(cut, lang))
        label, hint = {"kind": "number", "value": sum(1 for r in rows if r[3] > cut), "tol": 0}, "int"
    return f"{doc}\n\n{q} {_hint(lang, hint)}\n\n{DOC_INSTRUCTION[lang]}", label


def gen_contract(rng: random.Random, lang: str, level: int, variant: str) -> tuple[str, dict]:
    v = VOCAB[lang]
    n = LEVEL_SIZES["contract"][level - 1]
    base = rng.randrange(100, 900)
    rows = []
    for k in range(n):
        start = rand_date(rng, dt.date(2023, 1, 1), 1000)
        rows.append((f"K-{base + 3 * k + rng.randint(0, 2)}", company(rng, lang), start, rng.choice([6, 9, 12, 18, 24, 30, 36]), money_value(rng, 80, 4000)))
    if len({r[0] for r in rows}) != n:
        return gen_contract(rng, lang, level, variant)
    ends = {r[0]: add_months(r[2], r[3]) for r in rows}
    doc = f"### {v['contracts']}\n\n" + md_table(v["con"], [[r[0], r[1], fmt_date(r[2], lang), str(r[3]), fmt_money(r[4], lang)] for r in rows]) + "\n\n" + v["con_rule"]
    q = QUESTIONS["contract"][variant][lang]
    if variant == "end_date":
        con = rng.choice(rows)[0]
        q = q.format(con=con)
        label, hint = {"kind": "date", "value": ends[con].isoformat()}, "date"
    elif variant == "ending_before":
        cut = rand_date(rng, min(ends.values()), max(1, (max(ends.values()) - min(ends.values())).days), 28)
        chosen = sorted(c for c, e in ends.items() if e < cut)
        if not chosen or len(chosen) == n or any(e == cut for e in ends.values()):
            return gen_contract(rng, lang, level, variant)
        q = q.format(date=fmt_date(cut, lang))
        label, hint = {"kind": "ids", "value": chosen, "pattern": r"K-\d+"}, "ids"
    else:
        order = sorted(ends.values())
        if order[-1] == order[-2]:
            return gen_contract(rng, lang, level, variant)
        label, hint = {"kind": "id", "value": max(ends, key=ends.get), "pattern": r"K-\d+"}, "id"
    return f"{doc}\n\n{q} {_hint(lang, hint)}\n\n{DOC_INSTRUCTION[lang]}", label


def gen_meeting(rng: random.Random, lang: str, level: int, variant: str) -> tuple[str, dict]:
    v = VOCAB[lang]
    n = LEVEL_SIZES["meeting"][level - 1]
    rooms = ["A1", "A2", "B1", "B2"][: 2 + (level > 1)]
    days = [dt.date(2026, 3, 2) + dt.timedelta(days=rng.randrange(0, 10)) for _ in range(2)]
    # Build non-overlapping bookings per (room, day), then (for "overlap") move one into a clash.
    slots: dict[tuple[str, dt.date], list[tuple[int, int]]] = {}
    rows = []
    tries = 0
    while len(rows) < n and tries < 500:
        tries += 1
        room, day = rng.choice(rooms), rng.choice(days)
        start = rng.randrange(8 * 60, 17 * 60, 15)
        end = start + rng.choice([30, 45, 60, 90, 120])
        if any(s < end and start < e for s, e in slots.get((room, day), [])):
            continue
        slots.setdefault((room, day), []).append((start, end))
        rows.append([f"M-{10 + len(rows) * 3 + rng.randint(0, 2)}", room, day, start, end])
    if len(rows) < n:
        return gen_meeting(rng, lang, level, variant)
    if variant == "overlap":
        a = rng.randrange(n)
        b = rng.choice([i for i in range(n) if i != a])
        rows[b][1], rows[b][2] = rows[a][1], rows[a][2]
        shift = rng.choice([15, 30])
        dur = rows[b][4] - rows[b][3]
        rows[b][3] = rows[a][3] + shift if rows[a][4] - rows[a][3] > shift else rows[a][3]
        rows[b][4] = rows[b][3] + dur
        clashes = {(i, j) for i in range(n) for j in range(i + 1, n) if rows[i][1] == rows[j][1] and rows[i][2] == rows[j][2]
                   and rows[i][3] < rows[j][4] and rows[j][3] < rows[i][4]}
        if clashes != {tuple(sorted((a, b)))} or rows[b][4] > 20 * 60:
            return gen_meeting(rng, lang, level, variant)
    rows.sort(key=lambda r: (r[2], r[3]))
    hm = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    doc = f"### {v['meetings']}\n\n" + md_table(v["meet"], [[r[0], r[1], fmt_date(r[2], lang), hm(r[3]), hm(r[4])] for r in rows])
    q = QUESTIONS["meeting"][variant][lang]
    if variant == "overlap":
        pair = sorted(r[0] for r in rows if any(r is not o and r[1] == o[1] and r[2] == o[2] and r[3] < o[4] and o[3] < r[4] for o in rows))
        label, hint = {"kind": "ids", "value": pair, "pattern": r"M-\d+"}, "ids"
    else:
        room, day = rng.choice(sorted({(r[1], r[2]) for r in rows}))
        q = q.format(room=room, date=fmt_date(day, lang))
        label, hint = {"kind": "number", "value": sum(r[4] - r[3] for r in rows if r[1] == room and r[2] == day), "tol": 0}, "int"
    return f"{doc}\n\n{q} {_hint(lang, hint)}\n\n{DOC_INSTRUCTION[lang]}", label


def gen_form(rng: random.Random, lang: str, level: int, variant: str) -> tuple[str, dict]:
    name = person(rng, lang)
    city = rng.choice(CITIES[lang])
    od = rand_date(rng, dt.date(2025, 6, 1), 400)
    pd = od - dt.timedelta(days=rng.randrange(20, 200))
    oa, pa = money_value(rng, 20, 2500), money_value(rng, 20, 2500)
    cn = f"{rng.randrange(10, 99)}-{rng.randrange(10000, 99999)}"
    cur = LOCALE[lang]["cur"]
    letter = LETTER[lang].format(od=fmt_date(od, lang), pd=fmt_date(pd, lang), oa=f"{fmt_money(oa, lang)} {cur}", pa=f"{fmt_money(pa, lang)} {cur}",
                                 cn=cn, name=name, street=STREET[lang].format(n=rng.randint(1, 120)), city=f"{rng.randrange(10000, 99999)} {city}")
    label = {"kind": "json", "value": {"name": name, "customer_number": cn, "order_date": od.isoformat(), "amount": oa, "city": city}}
    return f"{letter}\n\n---\n\n{FORM_PROMPT[lang]}", label


GENERATORS = {"invoice": gen_invoice, "table": gen_table, "contract": gen_contract, "meeting": gen_meeting, "form": gen_form}


def variants(family: str) -> list[str]:
    return list(QUESTIONS[family]) if family in QUESTIONS else ["extract"]


def make_doc_task(family: str, lang: str, level: int, variant: str, seed: int | str) -> dict[str, Any]:
    rng = random.Random(f"eu_doc:{family}:{lang}:{level}:{variant}:{seed}")
    prompt, label = GENERATORS[family](rng, lang, level, variant)
    return {"family": family, "variant": variant, "language": lang, "level": level, "prompt": prompt, "label": label,
            "held_out_variant": HELD_OUT.get(family) == variant}


# --------------------------------------------------------------------------------------------------------------------
# Suite 2 scoring
# --------------------------------------------------------------------------------------------------------------------

def last_json_object(text: str) -> dict | None:
    for start in [m.start() for m in re.finditer(r"\{", text)][::-1]:
        depth = 0
        for j in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                try:
                    value = json.loads(text[start:j + 1])
                except json.JSONDecodeError:
                    break
                return value if isinstance(value, dict) else None
    return None


def score_doc(prediction: str, label: dict[str, Any]) -> float:
    """1 / 0 for exact answers; JSON: fraction of the five fields that match."""
    text = answer_part(prediction, label.get("mode", "default"))
    if not text:
        return 0.0
    kind, gold = label["kind"], label["value"]
    if kind == "json":
        obj = last_json_object(text)
        if obj is None:
            return 0.0
        ok = 0
        for key, want in gold.items():
            got = obj.get(key)
            if got is None:
                continue
            if key == "amount":
                num = got if isinstance(got, (int, float)) and not isinstance(got, bool) else parse_number(str(got))
                ok += num is not None and abs(float(num) - want) < 0.011
            elif key == "customer_number":
                ok += re.sub(r"\s", "", str(got)) == want
            elif key == "order_date":
                ok += str(got).strip() == want
            else:
                ok += norm_text(str(got)) == norm_text(want)
        return ok / len(gold)
    value = final_answer(text)
    if value is None:
        return 0.0
    if kind == "number":
        num = parse_number(value)
        return float(num is not None and abs(num - float(gold)) <= max(label.get("tol", 0), 1e-9))
    if kind == "date":
        found = re.findall(r"\d{4}-\d{2}-\d{2}", value)
        return float(found == [gold])
    if kind == "id":
        found = set(re.findall(label["pattern"], value))
        return float(found == {gold})
    if kind == "ids":
        return float(set(re.findall(label["pattern"], value)) == set(gold))
    if kind == "name":
        return float(norm_text(value) == norm_text(gold))
    raise ValueError(f"unknown kind {kind}")


def score(prediction: str, label: dict[str, Any] | str) -> float:
    """RL reward for either suite; label carries ``suite`` ("qa" | "doc") and ``mode``."""
    if isinstance(label, str):
        label = json.loads(label)
    return score_qa(prediction, label) if label["suite"] == "qa" else score_doc(prediction, label)
