# Prompt Template

Questo template è stato generato automaticamente per il progetto.

## Template

```
---

## Prompt Template: **Modulare e Riutilizzabile per Progetto Tecnico**

### **SEZIONE CONTESTO**  
```
# **Descrizione del Progetto**  
Descrivi brevemente il progetto, includendo obiettivi principali e scopo.  

**Nome Progetto**: {PROJECT_NAME}  
**Descrizione Generale**: {PROJECT_DESCRIPTION}  
- **Obiettivi Principali**:  
  1. {OBJECTIVE_1}  
  2. {OBJECTIVE_2}  
  3. {OBJECTIVE_3}  

**Tecnologie Utilizzate**:  
- Linguaggi: {LANGUAGES_USED}  
- Librerie/Framework:  
  1. {LIBRARY_1}  
  2. {LIBRARY_2}  
  3. {LIBRARY_3}  

**Architettura del Progetto**:  
- Struttura generale: {ARCHITECTURE_OVERVIEW}  
- Moduli chiave:  
  - {MODULE_1}  
  - {MODULE_2}  
  - {MODULE_3}  

**Obiettivi e Requisiti Funzionali**:  
1. {REQUIREMENT_1}  
2. {REQUIREMENT_2}  
3. {REQUIREMENT_3}  
```

---

### **SEZIONE ISTRUZIONI**  
```
# **Istruzioni per Implementazione o Modifica**  
Fornisci istruzioni dettagliate con placeholder che possono essere adattati a task specifici.  

### **Passaggi da Seguire per Completare il Task**:  
1. **Input e Setup**:  
   - Carica il file o dataset di esempio in formato: {INPUT_FORMAT}  
   - Configura le variabili d’ambiente utilizzando il file `{CONFIG_FILE}`.  

2. **Anonimizzazione e Analisi**:  
   - Utilizza il modulo `{ANONYMIZATION_MODULE}` per eseguire l'anonimizzazione dei dati con il seguente comando:  
     ```
     python {SCRIPT_NAME} --input {INPUT_PATH} --output {OUTPUT_PATH}
     ```  
   - Per implementare un nuovo modello di NER, inserisci il modello `{NEW_MODEL_NAME}` nella configurazione del modulo `{NER_MODULE}`.  

3. **Integrazione Multi-Agente**:  
   - Definisci gli agenti richiesti nel file `{AGENT_CONFIG_FILE}`.  
   - Avvia la pipeline tramite il comando:  
     ```
     python {AGENT_SCRIPT} --config {AGENT_CONFIG_PATH}
     ```  

4. **Modifica o Implementazione Specifica**:  
   - Sostituisci `{PLACEHOLDER_CODE_OR_FUNCTION}` nel modulo `{SPECIFIC_MODULE}` come segue:  
     ```
     def {FUNCTION_NAME}(params):
         # New implementation here
         return updated_result
     ```  

### **Dettagli di Configurazione**  
- File di configurazione richiesti:  
  - `{CONFIG_FILE_1}`  
  - `{CONFIG_FILE_2}`  

- Variabili d’ambiente chiave:  
  ```
  API_KEY={YOUR_API_KEY}  
  ENDPOINT={YOUR_ENDPOINT}  
  MODEL_NAME={MODEL_NAME}  
  ```  

### **Good Practices**  
- **Backup**: Effettua un backup dei dati caricati nella cartella `{BACKUP_FOLDER}` prima di processarli.  
- **Logging**: Utilizza sempre il modulo `{LOGGING_MODULE}` per monitorare l'esecuzione.  
```

---

### **SEZIONE ESEMPI**  
```
# **Codice di Esempio**  

### **Anonimizzazione con Modulistica NER**  
Esegui un mascheramento di dati sensibili utilizzando una regex e modelli NER.  
```python
from transformers import pipeline  
import re  

def anonymize_text(text):  
    # Named Entity Recognition  
    ner_model = pipeline("ner", model="{MODEL_NAME}", tokenizer="{TOKENIZER_NAME}")  
    entities = ner_model(text)  
    
    # Mascherare con regex entità sensibili  
    anonymized_text = re.sub(r"{PATTERN}", "{MASKING_VALUE}", text)  
    return anonymized_text  

input_text = "Informazioni sensibili: Nome=John, IBAN=DE89 3704 0044 0532 0130 00."  
print(anonymize_text(input_text))  
```  

### **Esempio di RAG Workflow con LangChain**  
Esegui il retrieval semantico su una knowledge base per rispondere a domande.  
```python
from langchain.chains import RetrievalQA  
from langchain.vectorstores import FAISS  
from langchain.llms.openai import OpenAI  

# Setup del modello e vector store  
vector_store = FAISS.load_local("{VECTOR_STORE_PATH}")  
qa_chain = RetrievalQA(llm=OpenAI(model="{GPT_MODEL}"), retriever=vector_store.as_retriever())  

# Domanda di esempio  
query = "Qual è l'analisi contenuta nel documento X?"  
response = qa_chain.run(query)  
print(response)  
```  

### **Orchestrazione Multi-Agente**  
Utilizza CrewAI per analisi distribuita.  
```python
from crewai.agent import Agent  
from crewai.orchestrator import Orchestrator  

# Definizione agenti  
agent1 = Agent(name="SentimentAnalysisAgent", task="{TASK}", model="{MODEL_NAME}")  
agent2 = Agent(name="SummarizationAgent", task="text_summary", model="{MODEL_NAME}")  

# Orchestrazione  
orchestrator = Orchestrator(agents=[agent1, agent2])  
orchestrator.run(input_data="{INPUT_PATH}")  
```  

---

### **SEZIONE OUTPUT**  
```
# **Formato Output Desiderato**  
Specifica come l'output deve essere strutturato per soddisfare i criteri.  

### **Formato e Struttura dei Dati**  
- Formato file: {OUTPUT_FORMAT}  
- Struttura dei dati:  
  ```json  
  {  
      "document_id": "{ID}",  
      "analysis_results": {  
          "anonymization_status": "{STATUS}",  
          "key_insights": [  
              "{INSIGHT_1}",  
              "{INSIGHT_2}"  
          ]  
      }  
  }  
  ```  

### **Criteri di Qualità dell’Output**  
1. **Accuratezza**: Dati anonimizzati al 100% con nessuna informazione sensibile visibile.  
2. **Completeness**: Ogni documento deve includere un set completo di analisi (anonimizzazione, sintesi, sentiment analysis).  
3. **Formato Consistente**: Risultati esportati come JSON, leggibile e standard.  

### **Guida per Validazione**  
Esegui un controllo di validazione su campioni usando il modulo `{VALIDATION_MODULE}` e il comando:  
```
python validate.py --input {OUTPUT_PATH} --schema {SCHEMA_PATH}  
```
```

---

Questo prompt template modulare offre una struttura completa per descrivere, istruire e contestualizzare un progetto basato su tecnologie avanzate con Placeholders chiaramente definiti. È progettato per essere riutilizzabile su diversi tipi di implementazioni simili al progetto **Agentic RAG**.
```

## Come utilizzare

1. Copia il template sopra
2. Sostituisci le variabili con i valori appropriati
3. Utilizza per generare documentazione simile

*Generato automaticamente il 2025-06-30 14:46:10*
