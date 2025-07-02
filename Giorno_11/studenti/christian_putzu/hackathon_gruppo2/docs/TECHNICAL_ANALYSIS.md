# Analisi Tecnica

## Panoramica

---
## 1. **PANORAMICA GENERALE**  

### Tipo di Progetto/Contenuto
Il progetto denominato **Agentic RAG** si concentra sull'implementazione di una pipeline di elaborazione documentale automatizzata, con lo scopo di:
- Garantire **anonimizzazione completa dei dati sensibili** nei documenti tramite tecniche avanzate di riconoscimento entità (NER) e regex.
- Offrire **analisi semantica** e risposte intelligenti basandosi su modelli LLM (Large Language Models).
- Costruire una **piattaforma scalabile** che integri tecnologie multi-agente per processi di analisi avanzata.

### Tecnologie Principali Identificate
- **Linguaggi**: Python.  
- **Framework & librerie principali**:
  - [LangChain](https://www.langchain.com): Per l'implementazione della Retrieval-Augmented Generation (RAG).
  - [Streamlit](https://streamlit.io): Per la costruzione dell'interfaccia web.
  - [Transformers](https://huggingface.co): Per tecniche di analisi NER con modelli come BERT.
  - [Azure OpenAI](https://azure.microsoft.com/en-us/services/openai/): Per GPT-4 e gestioni embedding per la similarità semantica.
  - CrewAI: Per orchestrazione di moduli multi-agente.
- **Soluzioni cloud**:
  - Azure Integration per GPT-4, embeddings e API OpenAI.

### Struttura Generale
- **Moduli Software**:
  - `NERAnonimizer` per l'anonimizzazione.
  - `AzureProcessor` per la gestione delle analisi tramite GPT-4.
  - Multi-agente CrewAI per un'analisi distribuita.
- **Struttura di Presentazione**:
  - Una dashboard interattiva basata su Streamlit.
- **Pipeline di Elaborazione**:
  Il progetto segue un flusso operativo ben definito: **Upload → Anonimizzazione → Analisi → RAG → Multi-Agent Processing → Risultati Finali**.

---

## 2. **ANALISI TECNICA**

### Linguaggi di Programmazione Utilizzati
- **Python**: Linguaggio principale identificato per tutti i livelli implementativi.

### Framework e Librerie Identificate
- **LangChain**: Utilizzata per il retrieval semantico dei documenti e la costruzione di chatbot avanzati utilizzando il paradigma Retrieval-Augmented Generation.
- **Transformers**: Libreria Hugging Face integrata per implementare modelli di Named Entity Recognition (NER) come `"Davlan/bert-base-multilingual-cased-ner-hrl"`.
- **Streamlit**: Utilizzato per l'interfaccia grafica (dashboard web interattiva).
- **FAISS (Facebook AI Similarity Search)**: Per la creazione di un Index semantico di vector embedding.
- **OpenAI Integration**: Per connettività alla piattaforma Azure e utilizzo di modelli GPT-4 e specifici embedding.
- **dotenv**: Per la gestione delle variabili di configurazione (.env).

### Pattern Architetturali Rilevati
- **Pipeline di Elaborazione Dati**:
  1. Analisi e anonimizzazione iniziale tramite moduli NER & regex.
  2. Creazione di una knowledge base con embeddings per query semantiche.
  3. Supporto multi-agente tramite CrewAI.
- **Architettura Layered** (a 5 livelli): Presentazione, Privacy, Semantica, Multi-Agent, Persistenza.
- **Design Orientato alla Privacy (Privacy by Design)**: Mascheramento dati prima di tutte le elaborazioni AI.

### File di Configurazione Trovati
- `.env`: File caricati tramite `load_dotenv` per centralizzare:
  - Chiavi e endpoint API di Azure.
  - Configurazioni di deploy LLM (`gpt-4o`) e modelli NER.

---

## 3. **STRUTTURA ORGANIZZATIVA**

### Organizzazione Cartelle/File
```
Gruppo_2/
├── 01_Agentic_RAG.py       # Codice principale backend
├── 01_risposta_progettuale.md   # Analisi e documentazione high-level
├── 02_Documentazione.md    # Documentazione approfondita tecnica
├── 02_schema_architetturale.md  # Schema architetturale dettagliato
├── 03_documenti/           # Documenti sample per test pipeline
    ├── email3.txt
    ├── email4.txt
    ├── notifica.txt
    ├── report2.txt
├── 04_documenti/           # Folder per documenti generici anonimi
└── .env                    # Configurazione sensibile (.gitignored)
```

### Moduli Principali
1. **`Config`**:
   - Gestisce tutte le configurazioni centrali, incluse variabili di ambiente.
2. **`NERAnonimizer`**:
   - Effettua mascheramento dati sensibili tramite regex e BERT NER.
3. **`AzureProcessor`**:
   - Connette e utilizza GPT-4 e altre capacità AI offerte da Azure OpenAI.
4. **`CrewAI`**:
   - Orchestrazione di agenti per processi paralleli distribuiti (analisi multi-agente).

### Punti di Ingresso (Entry Points)
- **`main()` in 01_Agentic_RAG.py**:
  - Inizializza la pipeline principale.
  - Caricamento documenti, setup agenti, esecuzione task CrewAI.
- **Streamlit Dashboard**:
  - Entry point utente per operazioni gestionali.

### Dipendenze Principali
- **Python Core Modules**:
  - `os`, `re`, `json`, `tempfile`, `pathlib`, `pandas`, `numpy`.
- **Cloud Services**:
  - Azure: API embedding, GPT-4 inclusa.
- **LLM e NLP Tools**:
  - LangChain, Transformers, FAISS.
- **Strumenti interattivi**:
  - Streamlit.
  
---

## 4. **CONTESTO FUNZIONALE**

### Funzionalità Principali
1. **Anonimizzazione Dati Sensibili**:
   - Mascheramento di dati sensibili come IBAN, email, numeri di carte tramite regex.
   - Riconoscimento di entità personali/organizzative attraverso NER multilingua.
2. **Analisi e RAG Integration**:
   - Recupero e analisi semantica con LangChain+FAISS.
   - Codifica e costruzione di knowledge base nel vector store.
3. **Processing Multi-Agente**:
   - CrewAI consente orchestrazione di analisi parallelizzate suddivise in specializzazioni come sentiment analysis o sintesi documentale.
4. **Reportistica ed Esportazione**:
   - Generazione di file JSON contenenti tutte le analisi.
   - Persistenza di cronologia, log, e risultati utente.

### API o Interfacce Esposte
- **LangChain Vector Retrieval**:
  - Accesso per query semantiche e augmented answers.
- **Streamlit GUI** (frontend integrato):
  - Tab dedicati per: caricamento file, anonimizzazione, analisi, chatbot interattivi, e gestione multi-agente.
  
### Processi di Business Identificati
- **Conformità Privacy (GDPR)**:
  - Tutta l’elaborazione avviene su dati anonimizzati.
  - Export finale contiene solo dati "sicuri".
- **Analisi Documentale Automatica**:
  - I documenti caricati subiscono un flusso standardizzato di:
    - Anonimizzazione.
    - Classificazione.
    - Sintesi semantica.
    - Risposte intelligenti via RAG.
  
### Workflow Principali
1. **Data Processing Workflow**:
   - Analisi completa basata sull'orchestrazione degli agenti CrewAI.
   - Editing manuale opzionale sull'interfaccia Streamlit.
2. **Collaborative Task Management**:
   - Crew di agenti specializzati automatizza la gestione di più task su documenti multipli.

---

**Complessivamente**, il progetto **Agentic RAG** implementa una pipeline avanzata focalizzata sull'anonimizzazione predittiva, analisi AI distribuita, e presentazione scalabile consolidando tecnologie moderne come NER, regex, LangChain e sistemi multi-agente per rispondere a bisogni legati a privacy, intelligenza aziendale e conformità normativa.

## Metadata

- **Generato il**: 2025-06-30 14:46:10
- **Tipo sorgente**: File ZIP
- **Nome sorgente**: Gruppo_2.zip
