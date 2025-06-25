from transformers import pipeline

ner_pipe = pipeline(
    "ner",
    model="Davlan/bert-base-multilingual-cased-ner-hrl",
    aggregation_strategy="simple"
)

text = "Leonardo da Vinci ha dipinto la Gioconda a Firenze."
entities = ner_pipe(text)

# Stampa risultato: "Leonardo → PERSON", ecc.
entities = ner_pipe(text)
for ent in entities:
    print(f"{ent['word']}: {ent['entity_group']}")
