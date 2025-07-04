# Risposta Progettuale
 
## Obiettivo del Progetto

Il progetto ha l’obiettivo di processare documenti testuali contenenti informazioni potenzialmente sensibili, garantendo la protezione della privacy attraverso tecniche di anonimizzazione automatica, e fornendo analisi intelligenti tramite modelli di linguaggio.
 
## Strategia Adottata

Il team ha progettato una pipeline collaborativa basata sull’integrazione tra:
 
- Tecniche di **Named Entity Recognition (NER)** per l'anonimizzazione semantica dei dati sensibili.

- Mascheramento tramite **regex** per rilevare e oscurare pattern strutturati (es. IBAN, email, codice fiscale).

- Un layer di **analisi AI** mediante modelli LLM (Large Language Models) offerti da Azure OpenAI.

- **CrewAI**, una libreria per orchestrare agenti, compiti e strumenti in modo modulare, trasparente e riutilizzabile.
 
## Componenti Principali

- `NERAnonimizer`: modulo che esegue il mascheramento delle entità usando il modello pre-addestrato Davlan/bert-base-multilingual-cased-ner-hrl.

- `AzureProcessor`: modulo che analizza semanticamente il testo anonimo tramite un deployment di gpt-4.1 su Azure.

- `DocumentProcessor`: tool CrewAI personalizzato che combina anonimizzazione e analisi AI.

- `CrewAI`: framework per la gestione del flusso operativo e dell'agente "Document Privacy Analyst".
 
## Vantaggi delle Scelte Tecnologiche

- Modularità e chiarezza della pipeline.

- Scalabilità e portabilità: il progetto può operare in ambienti online (con Azure) o offline (solo con NER).

- Attenzione alla privacy: tutti i documenti vengono anonimizzati **prima** di essere elaborati da modelli AI.

- Facilità di test e automazione grazie alla struttura agent-task-crew.
 
## Conclusione

La soluzione realizzata risponde al problema iniziale in modo completo e flessibile. I documenti vengono processati con attenzione alla privacy, arricchiti da un layer di analisi intelligente, e gestiti tramite una pipeline facilmente estendibile o adattabile a contesti diversi.
 
 