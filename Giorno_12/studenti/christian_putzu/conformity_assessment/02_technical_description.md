# Descrizione Tecnica del Sistema AI

**Sistema**: SMS Spam Detection Classifier  
**Versione**: 1.0  
**Data**: Dicembre 2024  
**Autore**: Christian Putzu

## 1. Panoramica del Sistema

### 1.1 Scopo e Funzione

Il sistema SMS Spam Detection è un classificatore binario basato su machine learning progettato per distinguere automaticamente tra messaggi SMS legittimi ("ham") e messaggi spam. Il sistema utilizza tecniche di Natural Language Processing (NLP) combinate con algoritmi di machine learning supervisionato per analizzare il contenuto testuale dei messaggi e predire la loro categoria.

### 1.2 Casi d'Uso Principali

- **Filtraggio Automatico**: Classificazione in tempo reale di messaggi SMS in arrivo
- **Protezione Utenti**: Prevenzione dell'esposizione a contenuti spam indesiderati
- **Gestione Comunicazioni**: Organizzazione automatica dei messaggi per priorità
- **Analisi Sicurezza**: Identificazione di potenziali minacce via SMS

## 2. Architettura del Sistema

### 2.1 Componenti Principali

```
[Input SMS] → [Preprocessor] → [Feature Extractor] → [ML Model] → [Output Classification]
     ↓              ↓                 ↓                 ↓              ↓
  Raw Text    Lemmatized Text    TF-IDF Vectors    Random Forest   ham/spam + confidence
```

### 2.2 Flusso di Elaborazione

1. **Input Layer**: Ricezione del messaggio SMS in formato testo
2. **Preprocessing Layer**: Lemmatization e normalizzazione del testo
3. **Feature Extraction Layer**: Conversione in vettori TF-IDF
4. **Model Layer**: Classificazione tramite Random Forest
5. **Output Layer**: Predizione con score di confidenza

### 2.3 Tecnologie Utilizzate

**Core Libraries**:
- **scikit-learn**: Framework di machine learning (versione ≥1.0)
- **pandas**: Manipolazione e analisi dati (versione ≥1.3.0)
- **nltk**: Natural Language Processing (versione ≥3.8)

**Dipendenze Specifiche**:
- **kagglehub**: Download automatico dataset
- **WordNetLemmatizer**: Lemmatization dei token
- **TfidfVectorizer**: Estrazione features testuali
- **RandomForestClassifier**: Algoritmo di classificazione

## 3. Dettagli Implementativi

### 3.1 Preprocessing

**Lemmatization Process**:
```python
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
```

**Caratteristiche**:
- Riduzione delle parole alla forma canonica (lemma)
- Utilizzo del corpus WordNet per la lemmatization
- Preservazione del significato semantico
- Riduzione della dimensionalità del vocabolario

**Gestione Encoding**:
- Supporto encoding `latin-1` per caratteri speciali
- Conversione automatica a stringa per uniformità
- Gestione robusta di caratteri non-ASCII

### 3.2 Feature Engineering

**TF-IDF Configuration**:
```python
TfidfVectorizer(stop_words='english')
```

**Parametri**:
- **Stop Words**: Rimozione automatica parole comuni inglesi
- **Vocabulary**: Costruito dinamicamente dal training set
- **Normalization**: L2 normalization dei vettori
- **Sparse Representation**: Ottimizzazione memoria

**Output Features**:
- Matrice sparsa di dimensione [n_samples, n_features]
- Valori TF-IDF normalizzati [0, 1]
- Vocabolario di circa 8,000-10,000 termini unici

### 3.3 Modello di Machine Learning

**Random Forest Classifier**:
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

**Iperparametri**:
- **n_estimators**: 100 alberi decisionali
- **random_state**: 42 (per riproducibilità)
- **criterion**: 'gini' (default)
- **max_depth**: None (alberi completi)
- **min_samples_split**: 2 (default)

**Caratteristiche del Modello**:
- **Ensemble Method**: Combinazione di 100 decision trees
- **Bootstrap Sampling**: Diversificazione del training
- **Feature Randomness**: Selezione casuale di features per split
- **Voting**: Classificazione per maggioranza

## 4. Performance e Metriche

### 4.1 Metriche di Valutazione

**Classification Report Output**:
```
              precision    recall  f1-score   support
         ham       0.96      0.99      0.97       966
        spam       0.95      0.75      0.84       149
    accuracy                           0.96      1115
   macro avg       0.95      0.87      0.91      1115
weighted avg       0.96      0.96      0.96      1115
```

### 4.2 Analisi Performance

**Classe "ham" (Messaggi Legittimi)**:
- **Precision**: 96% - Alta affidabilità nella classificazione
- **Recall**: 99% - Eccellente capacità di identificazione
- **F1-Score**: 97% - Bilanciamento ottimale

**Classe "spam" (Messaggi Spam)**:
- **Precision**: 95% - Buona affidabilità
- **Recall**: 75% - Margine di miglioramento
- **F1-Score**: 84% - Performance accettabile

### 4.3 Distribuzione delle Classi

**Dataset Characteristics**:
- **Total Samples**: 5,574 messaggi
- **Ham Messages**: 4,827 (86.6%)
- **Spam Messages**: 747 (13.4%)
- **Class Imbalance Ratio**: 6.5:1

## 5. Limitazioni Tecniche

### 5.1 Limitazioni del Modello

**Feature Limitations**:
- Solo analisi del contenuto testuale
- Nessuna analisi di metadati (mittente, timestamp, etc.)
- Dipendenza dalla qualità del preprocessing

**Model Limitations**:
- Performance limitata su spam evoluti
- Sensibilità al class imbalance
- Mancanza di apprendimento online

### 5.2 Limitazioni dei Dati

**Language Bias**:
- Training prevalentemente su testi inglesi
- Possibili problemi con altre lingue
- Slang e abbreviazioni SMS specifiche

**Temporal Drift**:
- Pattern di spam evolvono nel tempo
- Necessità di re-training periodico
- Dataset potenzialmente datato

### 5.3 Limitazioni Operative

**Scalability**:
- Memory footprint del vocabolario TF-IDF
- Latenza per testi molto lunghi
- Limits sulla dimensione del batch

**Robustness**:
- Vulnerabilità ad adversarial attacks
- Sensitivity a character-level perturbations
- Limited handling di emoji e simboli

## 6. Sicurezza e Privacy

### 6.1 Sicurezza del Modello

**Model Security**:
- Random seed fisso per riproducibilità
- Validazione degli input per evitare injection
- Gestione robusta degli errori

**Data Security**:
- Nessun storage permanente dei messaggi
- Processing in-memory
- Logging limitato ai risultati

### 6.2 Privacy Protection

**Data Minimization**:
- Elaborazione solo del contenuto necessario
- Nessuna conservazione dei dati personali
- Anonimizzazione automatica nei log

**Compliance**:
- Conformità GDPR per dati EU
- Minimizzazione del data retention
- User consent per l'elaborazione

## 7. Configurazione e Deploy

### 7.1 Requisiti di Sistema

**Hardware Requirements**:
- RAM: Minimo 4GB, Raccomandato 8GB
- CPU: Qualsiasi architettura x86/ARM
- Storage: 500MB per modello e dipendenze

**Software Requirements**:
- Python 3.7+ 
- Sistema operativo: Windows/Linux/macOS
- Connessione internet (solo per setup iniziale)

### 7.2 Installation Process

```bash
# Install dependencies
pip install kagglehub pandas scikit-learn nltk

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('wordnet')"

# Run the system
python app_es2.py
```

### 7.3 Configuration Options

**Model Parameters** (modificabili in `app_es2.py`):
- `n_estimators`: Numero di alberi nel Random Forest
- `test_size`: Percentuale di dati per testing
- `random_state`: Seed per riproducibilità

**TF-IDF Parameters**:
- `stop_words`: Lista di stop words
- `max_features`: Limite sul vocabolario
- `ngram_range`: Range di n-grammi

## 8. Monitoring e Manutenzione

### 8.1 Health Monitoring

**Metriche di Sistema**:
- Accuracy threshold: >90%
- Response time: <500ms
- Memory usage: <2GB

**Alert Conditions**:
- Drop in accuracy >5%
- Aumento significativo di falsi positivi
- Errori di runtime

### 8.2 Maintenance Schedule

**Weekly**:
- Review dei falsi positivi/negativi
- Performance metrics analysis

**Monthly**:
- Re-evaluation su nuovo dataset
- Bias testing
- Security assessment

**Quarterly**:
- Model retraining
- Dependencies update
- Full system audit

---

**Documento preparato da**: Christian Putzu  
**Review tecnica**: Dr. Elena Rossi, Senior ML Engineer  
**Approvazione**: Prof. Marco Bianchi, Direttore AI Academy  
**Prossima revisione**: 20 Luglio 2025