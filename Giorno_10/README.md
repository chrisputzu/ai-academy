# 🔒 Anonimizzatore Documenti con AI

Sistema completo per anonimizzazione e analisi intelligente di documenti testuali con protezione privacy GDPR.

## 🚀 Funzionalità

- **🔐 Anonimizzazione Automatica**: NER + Regex per proteggere dati sensibili
- **💬 RAG Chatbot**: Chat intelligente sui documenti anonimizzati  
- **🤖 Multi-Agent AI**: 4 agenti CrewAI per analisi approfondite
- **📊 Dashboard Web**: Interfaccia Streamlit completa
- **📥 Export Risultati**: Download JSON strutturati

## 📋 Requisiti

- Python 3.8+
- Account Azure OpenAI
- Dipendenze in `requirements.txt`

## ⚙️ Installazione

1. **Clona il repository**
```bash
git clone <repo-url>
cd document_anonymizer
```

2. **Installa dipendenze**
```bash
pip install -r requirements.txt
```

3. **Configura variabili d'ambiente**
```bash
cp .env.example .env
```

Modifica `.env` con le tue credenziali Azure:
```
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_ENDPOINT_EMB=https://your-embedding-resource.openai.azure.com/
AZURE_API_KEY_EMB=your-embedding-api-key
```

4. **Avvia l'applicazione**
```bash
streamlit run main.py
```

## 🎯 Come Usare

### 1. Upload Documenti
- Carica file `.txt` nella tab "Upload"
- Supporta upload multipli

### 2. Anonimizzazione
- Vai alla tab "Anonimizzazione" 
- Clicca "Avvia Anonimizzazione"
- Revisiona e modifica entità rilevate
- Conferma i documenti

### 3. Analisi AI
- Tab "Analisi": Analisi Azure OpenAI per singoli documenti
- Tab "RAG": Chat interattiva con i documenti
- Tab "CrewAI": Analisi multi-agente avanzate

## 🤖 Agenti CrewAI

- **📄 Document Analyst**: Classificazione e analisi strutturale
- **😊 Sentiment Analyst**: Analisi emozioni e trend
- **🎯 Strategy Coordinator**: Sintesi executive e raccomandazioni

## 📁 Struttura Progetto

```
document_anonymizer/
├── main.py                 # App Streamlit principale
├── config.py              # Configurazioni sistema
├── anonymizer.py          # Sistema anonimizzazione NER+Regex
├── ai_processor.py        # Azure + RAG + CrewAI
├── ui_components.py       # Componenti UI riutilizzabili
├── utils.py               # Funzioni utility
├── requirements.txt       # Dipendenze Python
├── .env.example          # Template environment
└── README.md             # Questa documentazione
```

## 🔐 Privacy & Sicurezza

- **Privacy by Design**: Anonimizzazione prima di qualsiasi elaborazione AI
- **GDPR Compliant**: Nessun dato sensibile inviato ai modelli
- **Controllo Manuale**: Revisione ed editing delle entità rilevate
- **Tracciabilità**: Cronologia completa delle operazioni

## 🛠️ Entità Supportate

### Regex Pattern
- **IBAN**: Codici bancari italiani
- **EMAIL**: Indirizzi email
- **CF**: Codici fiscali italiani
- **CARD**: Numeri carte di credito
- **PHONE**: Numeri di telefono

### NER (Named Entity Recognition)
- **PER**: Nomi di persone
- **ORG**: Organizzazioni
- **LOC**: Luoghi
- **MISC**: Entità varie

## 📊 Tipi di Analisi CrewAI

### 🔍 Comprensiva
Analisi completa con tutti e 4 gli agenti per insights 360°

### 📄 Documentale  
Focus su classificazione, struttura e organizzazione documenti

### 😊 Sentiment
Analisi emozioni, soddisfazione e trend comunicazioni

### 🔍 RAG Avanzata
Query complesse con recupero semantico e correlazioni

### ⚙️ Personalizzata
Selezione manuale agenti per analisi su misura

## 🔧 Configurazione Avanzata

### Modelli Azure
Modifica in `config.py`:
```python
DEPLOYMENT_NAME = "gpt-4o"  # Tuo deployment chat
AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"  # Tuo deployment embedding
```

### Pattern Regex Personalizzati
Aggiungi in `config.py`:
```python
REGEX_PATTERNS = {
    # Pattern esistenti...
    "CUSTOM_PATTERN": r'your_regex_here'
}
```

## 🐛 Troubleshooting

### Errore Azure OpenAI
- Verifica credenziali in `.env`
- Controlla deployment names
- Verifica quota e limiti Azure

### Errore NER Model
- Controlla connessione internet
- Aumenta timeout download modello
- Usa cache Hugging Face

### Performance Lente
- Riduci dimensione documenti
- Usa meno chunks per RAG
- Ottimizza parametri CrewAI

## 📈 Esempi Query

### Business Intelligence
```
"Analizza i temi principali nei documenti e identifica possibili rischi operativi"
```

### Customer Service
```
"Valuta il sentiment nelle comunicazioni clienti e suggerisci miglioramenti"
```

### Compliance
```
"Verifica la conformità delle comunicazioni e identifica potenziali problemi legali"
```

### Strategic Analysis
```
"Fornisci un'analisi comprensiva con raccomandazioni strategiche actionable"
```

## 🤝 Contributi

1. Fork il progetto
2. Crea feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit modifiche (`git commit -m 'Add AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri Pull Request

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

## 📞 Supporto

Per supporto e domande:
- Apri una Issue su GitHub
- Contatta il team di sviluppo
- Consulta la documentazione Azure OpenAI

---

**⚡ Quick Start**: `pip install -r requirements.txt && streamlit run main.py`