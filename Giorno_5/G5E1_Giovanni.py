import json
import re
from transformers import pipeline
import spacy

# 1. NER con Hugging Face
ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# 2. SpaCy solo per tokenizzazione (facoltativo)
nlp_spacy = spacy.load("en_core_web_sm")

# 3. Lettura mail
with open("Giorno_5/Mail.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 4. 1° passata: NER + masking
entities = ner(text)
masked_text = text
for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
    label = ent["entity_group"]
    start, end = ent["start"], ent["end"]
    replacement = {
        "PER": "[NOME]",
        "EMAIL": "[EMAIL]",
        "PHONE": "[TEL]",
    }.get(label, None)
    if replacement:
        masked_text = masked_text[:start] + replacement + masked_text[end:]

# 5. 2° passata: estrazione con TinyLlama
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
prompt = (
    "Estrai nome, email e numero di telefono dal testo:\n"
    f"{masked_text}\nRestituisci solo JSON con chiavi nome, email, telefono."
)
raw_output = pipe(prompt, max_new_tokens=256)[0]['generated_text']

# 6. parsing sicuro JSON
try:
    j = raw_output.find('{')
    k = raw_output.rfind('}') + 1
    output = json.loads(raw_output[j:k])
    print("Nome:", output.get("nome"))
    print("Email:", output.get("email"))
    print("Telefono:", output.get("telefono"))
except Exception as e:
    print("Errore parsing:", e)
    print("Risposta completa:", raw_output)
