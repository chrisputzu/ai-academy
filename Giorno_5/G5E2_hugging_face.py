
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

tokenizer = AutoTokenizer.from_pretrained("DeepMount00/Italian_NER_XXL")
model = AutoModelForTokenClassification.from_pretrained("DeepMount00/Italian_NER_XXL", ignore_mismatched_sizes=True)
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def reconstruct_entities(ner_output):
    entities = []
    current = None

    for token in ner_output:
        label = token["entity"]
        word = token["word"].replace("##", "")
        start = token["start"]
        end = token["end"]

        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {
                "label": label[2:],
                "text": word,
                "start": start,
                "end": end
            }
        elif label.startswith("I-") and current and current["label"] == label[2:]:
            current["text"] += word
            current["end"] = end
        else:
            if current:
                entities.append(current)
                current = None

    if current:
        entities.append(current)

    return entities

def mask_entities(text, entities):
    # Ordina entità in ordine decrescente per evitare problemi di shift dell’indice
    entities = sorted(entities, key=lambda e: e['start'], reverse=True)
    for ent in entities:
        label = ent["label"]
        start = ent["start"]
        end = ent["end"]
        replacement = label.upper()  # es: 'EMAIL'
        text = text[:start] + replacement + text[end:]
    return text

# Leggere il file txt
paths = ['Giorno_5/Mail.txt', 'Giorno_5/nota_di_credito.txt', 'Giorno_5/ordine_di_acquisto.txt', 'Giorno_5/Fattura.txt']



for path in paths:

    print(f"\nAnalizzando il file: {path}\n")

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

        ner_results = nlp(text)
        entities = reconstruct_entities(ner_results)

        print(entities)

        masked_text = mask_entities(text, entities)
        print(masked_text)

