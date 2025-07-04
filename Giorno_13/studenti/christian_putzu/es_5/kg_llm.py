# Business question
user_question = "Quali clienti hanno almeno un contratto attivo nel 2024?"

# Cypher query for Neo4j
cypher_query = """
MATCH (c:Cliente)-[:HA_CONTRATTO]->(contr:Contratto)
WHERE contr.stato = 'attivo' AND contr.anno = 2024
RETURN c.nome AS Cliente, contr.id AS ContrattoID
"""

# Simulated Cypher response (tabular)
cypher_result = [
    {"Cliente": "Mario Rossi", "ContrattoID": "C123"},
    {"Cliente": "Elena Bianchi", "ContrattoID": "C456"}
]

# Prompt for LLM
llm_prompt = f"""
Risposta Cypher:
{cypher_result}

Domanda utente: {user_question}

Riformula la risposta in linguaggio naturale chiaro per il business.
"""

# (Here you would normally call the LLM, but you can simulate the response)
llm_response = (
    "I clienti che hanno almeno un contratto attivo nel 2024 sono: "
    "Mario Rossi (contratto C123) ed Elena Bianchi (contratto C456)."
)