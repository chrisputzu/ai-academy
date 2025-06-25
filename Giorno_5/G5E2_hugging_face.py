from transformers import pipeline
import re
import os

# Inizializza la pipeline NER
ner_pipe = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", aggregation_strategy="simple")

# Regex per dati strutturati
regex_patterns = {
    "IBAN": r'\bIT\d{2}[A-Z0-9]{1,23}\b',
    "EMAIL": r'\b[\w\.-]+@[\w\.-]+\.\w{2,}\b',
    "CFISCALE": r'\b[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]\b',
    "CARTA_CREDITO": r'\b(?:\d[ -]*?){13,16}\b'
}

def mask_regex(text):
    """Maschera dati sensibili tramite regex."""
    for label, pattern in regex_patterns.items():
        text = re.sub(pattern, label, text)
    return text

def mask_ner(text, ner_entities):
    """Maschera le entità NER nel testo."""
    # Ordina entità in ordine decrescente per evitare shift negli indici
    sorted_entities = sorted(ner_entities, key=lambda x: x['start'], reverse=True)
    for ent in sorted_entities:
        text = text[:ent['start']] + ent['entity_group'] + text[ent['end']:]
    return text

# Percorsi dei file
paths = [
    'Giorno_5/Mail.txt',
    'Giorno_5/nota_di_credito.txt',
    'Giorno_5/ordine_di_acquisto.txt',
    'Giorno_5/Fattura.txt'
]

# Analizza ogni file
for path in paths:
    print(f"\nAnalizzando il file: {path}\n")

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    # Estrai entità NER
    ner_entities = ner_pipe(text)
    for entity in ner_entities:
        print(f"{entity['word']}: {entity['entity_group']}")

    # Maschera prima i dati da regex, poi quelli NER
    text_masked = mask_regex(text)
    text_masked = mask_ner(text_masked, ner_entities)

    print("\nTesto mascherato:\n")
    print(text_masked)
