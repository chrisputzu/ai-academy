from transformers import pipeline
import re
import os

# 1. Pipeline NER
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# 2. Documenti da anonimizzare
files = ["Mail.txt", "Fattura.txt", "nota.txt"]

docs_dir = "Giorno_7"

label2mask = {
    "PER": "[NOME]",
    "ORG": "[ORGANIZZAZIONE]",
    "LOC": "[LUOGO]",
    "MISC": "[ALTRO]",
}

iban_re = re.compile(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}")
piva_re = re.compile(r"\b\d{11}\b")
email_re = re.compile(r"[\w\.-]+@[\w\.-]+")
phone_re = re.compile(r"\b\d{3}[\s\d]{5,}\b")

for fname in files:
    path = os.path.join(docs_dir, fname)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 3. anonimizzazione con NER
    entities = ner(text)
    masked = text
    for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
        label = ent["entity_group"]
        replacement = label2mask.get(label)
        if replacement:
            masked = masked[: ent["start"]] + replacement + masked[ent["end"] :]

    # 4. ulteriori pattern sensibili
    masked = iban_re.sub("[IBAN]", masked)
    masked = piva_re.sub("[PIVA]", masked)
    masked = email_re.sub("[EMAIL]", masked)
    masked = phone_re.sub("[TEL]", masked)

    # 5. Salvataggio
    out_path = os.path.join(docs_dir, fname.replace(".txt", "_anon.txt"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(masked)
    print(f"Creato {out_path}")

