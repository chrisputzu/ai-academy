# Registro delle Decisioni e Log

**Sistema**: SMS Spam Detection Classifier  
**Documento**: Decision Log & Audit Trail  
**Data**: Dicembre 2024  
**Versione**: 1.0  
**Maintainer**: Christian Putzu

## 1. Panoramica del Decision Log

### 1.1 Scopo

Questo documento mantiene un registro completo di tutte le decisioni progettuali, tecniche e operative prese durante lo sviluppo, implementazione e manutenzione del sistema SMS Spam Detection. Ogni entry include la rationale, le alternative considerate, l'impatto e gli approvatori.

### 1.2 Formato delle Entry

Ogni decisione è documentata con:
- **ID**: Identificativo univoco
- **Data**: Timestamp della decisione
- **Categoria**: Tipo di decisione (Tecnica/Operativa/Legale/Etica)
- **Descrizione**: Dettaglio della decisione
- **Rationale**: Motivazione e analisi
- **Alternative**: Opzioni considerate e scartate
- **Impatto**: Conseguenze previste
- **Approvatore**: Responsabile della decisione
- **Status**: Implementata/Pianificata/Revocata

## 2. Log delle Decisioni Architetturali

### DEC-001: Scelta dell'Algoritmo di Machine Learning

**Data**: 2024-12-20  
**Categoria**: Tecnica  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Decisione**: Utilizzo di Random Forest Classifier per la classificazione spam

**Rationale**:
- Buone performance su dati testuali sparsi
- Robustezza al overfitting
- Interpretabilità moderata delle feature importance
- Gestione nativa del class imbalance

**Alternative Considerate**:
1. **Logistic Regression**: Troppo semplice per pattern complessi
2. **SVM**: Performance inferiori su dataset di grandi dimensioni
3. **Neural Networks**: Complessità eccessiva per il task, black-box
4. **Naive Bayes**: Assunzioni di indipendenza troppo restrittive

**Impatto**:
- Performance: Accuracy ~96%, buon bilanciamento precision/recall
- Complessità: Gestibile per team di sviluppo
- Manutenzione: Ritraining periodico necessario
- Interpretabilità: Feature importance disponibili

**Metriche di Successo**:
- Accuracy > 95%
- Training time < 10 minuti
- Prediction latency < 100ms per messaggio

---

### DEC-002: Strategia di Feature Engineering

**Data**: 2024-12-20  
**Categoria**: Tecnica  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Decisione**: Utilizzo di TF-IDF con lemmatization per feature extraction

**Rationale**:
- TF-IDF cattura importanza relativa dei termini
- Lemmatization riduce dimensionalità preservando semantica
- Stop words removal elimina rumore
- Scaling naturale dei valori [0,1]

**Alternative Considerate**:
1. **Bag of Words**: Perdita di informazione sulla frequenza relativa
2. **Word2Vec embeddings**: Complessità eccessiva, limited benefit per spam detection
3. **N-grams senza lemmatization**: Vocabolario troppo esteso
4. **Character-level features**: Vulnerability ad adversarial attacks

**Impatto**:
- Dimensionalità: ~8,000-10,000 features dopo preprocessing
- Performance: Ottimale per il Random Forest
- Memoria: Matrice sparsa gestibile (~50MB)
- Preprocessing time: ~2 secondi per 1000 messaggi

**Parametri Scelti**:
```python
TfidfVectorizer(
    stop_words='english',
    max_features=None,  # Vocabolario completo
    ngram_range=(1,1),  # Solo unigrams
    lowercase=True
)
```

---

### DEC-003: Gestione del Class Imbalance

**Data**: 2024-12-20  
**Categoria**: Tecnica  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Decisione**: Utilizzo delle metriche bilanciate senza resampling

**Rationale**:
- Ratio 6.5:1 gestibile da Random Forest nativamente
- Preservation della distribuzione naturale dei dati
- Evitare bias introdotti da synthetic sampling
- Focus su precision per classe spam

**Alternative Considerate**:
1. **SMOTE Oversampling**: Rischio di overfitting su pattern sintetici
2. **Random Undersampling**: Perdita di informazioni importanti
3. **Cost-sensitive learning**: Complessità nella calibrazione dei pesi
4. **Threshold tuning**: Approccio più complesso per deployment

**Impatto**:
- Performance spam class: Precision 95%, Recall 75%
- Performance ham class: Precision 96%, Recall 99%
- Overall accuracy: 96%
- Generalization: Mantenuta distribuzione naturale

**Metriche di Monitoraggio**:
- F1-score bilanciato
- Precision/Recall per singola classe
- Confusion matrix review mensile

---

## 3. Log delle Decisioni di Compliance

### DEC-004: Classificazione del Rischio EU AI Act

**Data**: 2024-12-20  
**Categoria**: Legale/Compliance  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Decisione**: Classificazione del sistema come "Limited Risk"

**Rationale**:
- Sistema interagisce direttamente con utenti finali
- Nessun impatto su settori ad alto rischio (sanità, giustizia, etc.)
- Non manipola comportamenti umani subliminalmente
- Richiesta trasparenza sull'utilizzo AI

**Alternative Considerate**:
1. **Minimal Risk**: Sottostimerebbe l'impatto su comunicazioni
2. **High Risk**: Sovrastimerebbe considerando il dominio applicativo
3. **Prohibited**: Non applicabile al caso d'uso

**Impatto**:
- Compliance requirements: Trasparenza, documentazione, monitoring
- Audit frequency: Annuale invece che trimestrale
- Approval process: Interno invece che external notified body
- Documentation scope: Moderato invece che comprehensive

**Compliance Checklist**:
- [x] Risk assessment completato
- [x] Technical documentation preparata
- [x] User notification process implementato
- [x] Logging e audit trail attivati

---

### DEC-005: Strategia di Privacy e Data Protection

**Data**: 2024-12-20  
**Categoria**: Legale/Privacy  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Decisione**: Data minimization con processing in-memory

**Rationale**:
- Conformità GDPR principle di data minimization
- Riduzione superficie di attacco per data breaches
- Semplificazione compliance con data retention policies
- Elimination di rischi legati a permanent storage

**Alternative Considerate**:
1. **Local storage with encryption**: Complexity e residual risk
2. **Anonymization before processing**: Performance impact significativo
3. **User consent for each message**: UX degradation inaccettabile
4. **Cloud processing**: Data sovereignty concerns

**Impatto**:
- Privacy risk: Minimized attraverso ephemeral processing
- Performance: Optimal con in-memory operations
- Compliance: GDPR-compliant by design
- Audit: Simplified logging requirements

**Technical Implementation**:
```python
# No persistent storage
def process_message(message_text):
    # Process in memory only
    features = extract_features(message_text)
    prediction = model.predict(features)
    # No storage of original message
    return prediction
```

---

## 4. Log delle Decisioni Operative

### DEC-006: Strategia di Deployment e Monitoring

**Data**: 2024-12-20  
**Categoria**: Operativa  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Decisione**: Deployment locale con monitoring manual

**Rationale**:
- Controllo completo su execution environment
- Latenza minimizzata per processing
- Data governance semplificata
- Costi operativi contenuti

**Alternative Considerate**:
1. **Cloud deployment**: Data sovereignty concerns, ongoing costs
2. **Hybrid approach**: Complessità architetturale elevata
3. **Edge computing**: Overhead di gestione per benefici limitati
4. **Containerized deployment**: Overhead per single-instance system

**Impatto**:
- Response time: <500ms per message processing
- Availability: 99.5% target (manual restart se necessario)
- Scalability: Limited a single-machine throughput
- Maintenance: Manual oversight richiesta

**Monitoring Approach**:
- Daily automated health checks
- Weekly performance review
- Monthly bias assessment
- Quarterly comprehensive audit

---

### DEC-007: User Notification Strategy

**Data**: 2024-12-20  
**Categoria**: Operativa/UX  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Decisione**: Passive notification con opt-out capability

**Rationale**:
- Compliance con EU AI Act transparency requirements
- Bilanciamento tra compliance e user experience
- Informazione senza interruzione del flusso comunicativo
- User control preservata

**Alternative Considerate**:
1. **Active consent per ogni messaggio**: UX inaccettabile
2. **No notification**: Non-compliant con EU AI Act
3. **One-time consent**: Insufficiente per ongoing processing
4. **Detailed explanation popup**: UX disruption significativa

**Impatto**:
- Compliance: EU AI Act transparency requirement soddisfatto
- UX: Minimal disruption al normale flusso
- User control: Opt-out option mantenuta
- Legal risk: Mitigated attraverso adequate disclosure

**Implementation Details**:
- Subtle UI indicator per AI-processed messages
- Clear explanation in user documentation
- Easy-access opt-out mechanism
- Regular reminder di AI system usage

---

## 5. Log delle Modifiche e Correzioni

### MOD-001: Performance Tuning dopo Test Iniziali

**Data**: 2024-12-20  
**Categoria**: Tecnica/Correzione  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Modifica**: Ottimizzazione n_estimators da 50 a 100

**Trigger**: Initial testing mostrava recall subottimale per spam class (68%)

**Analisi**:
- Aumentando gli alberi da 50 a 100: recall migliorato al 75%
- Training time incrementato da 45s a 78s (accettabile)
- Overfitting risk mitigato dalla robustezza di Random Forest
- Memory footprint aumentato del 40% ma rimane gestibile

**Risultati**:
- Spam recall: 68% → 75% (+7 percentage points)
- Ham precision: 96% → 96% (invariato)
- Overall accuracy: 94% → 96% (+2 percentage points)
- Training time: 45s → 78s (+73%)

**Validation**:
- Cross-validation su 5 folds confermato
- Hold-out test su dataset esterno passato
- Performance degradation monitoring attivato

---

### MOD-002: Gestione Encoding Latino-1

**Data**: 2024-12-20  
**Categoria**: Tecnica/Bug Fix  
**Approvatore**: Christian Putzu (AI System Owner)  
**Status**: Implementata

**Problema**: UnicodeDecodeError durante il caricamento del dataset

**Root Cause**: Dataset contiene caratteri speciali non UTF-8 compatible

**Soluzione**: Explicit encoding='latin-1' nel pandas.read_csv()

**Rationale**:
- Preservazione di tutti i caratteri originali
- Compatibility con historical SMS data
- Robustezza per input futuri con caratteri speciali
- Minimal performance impact

**Testing**:
- Verifica load di tutti i 5,574 record
- Validation di processing completo pipeline
- Spot check su sample di messaggi con caratteri speciali
- Performance benchmark invariato

**Code Change**:
```python
# Before
df = pd.read_csv(csv_path)  # Fails on special characters

# After  
df = pd.read_csv(csv_path, encoding='latin-1')  # Handles all characters
```

---

## 6. Log degli Audit e Review

### AUD-001: Initial Security Review

**Data**: 2024-12-20  
**Categoria**: Security/Audit  
**Reviewer**: Christian Putzu (Self-review)  
**Status**: Completato

**Scope**: Security assessment del sistema pre-deployment

**Findings**:
1. **BASSO**: Nessun encryption per model file (pickle)
2. **BASSO**: Input validation limitata su lunghezza messaggi
3. **INFO**: Logging insufficiente per security events

**Actions Taken**:
1. Model file encryption non prioritaria (public dataset, no IP)
2. Input validation aggiunta con MAX_LENGTH constraint
3. Security logging enhancement pianificato per v1.1

**Security Posture**: ADEGUATO per deployment iniziale

**Next Review**: 3 mesi (Marzo 2025)

---

### AUD-002: Bias Testing Initial Assessment

**Data**: 2024-12-20  
**Categoria**: Fairness/Audit  
**Reviewer**: Christian Putzu (Self-review)  
**Status**: Completato

**Scope**: Bias assessment su disponibili demographic proxies

**Methodology**: 
- Language pattern analysis
- Message length bias testing
- Temporal pattern assessment (limitato)

**Findings**:
1. **MEDIO**: Possibile bias verso messaggi in inglese formale
2. **BASSO**: Slight bias contro messaggi molto brevi
3. **INFO**: Insufficient data per demographic analysis

**Mitigation Actions**:
1. Enhanced testing protocol development pianificato
2. Diversified dataset sourcing per future versions
3. Monitoring dashboard setup per ongoing bias detection

**Overall Assessment**: ACCETTABILE con monitoring requirement

**Next Review**: 1 mese (Gennaio 2025)

---

## 7. Log delle Pianificazioni Future

### PLAN-001: Multi-language Support

**Data**: 2024-12-20  
**Categoria**: Enhancement/Roadmap  
**Priority**: Medio  
**Timeline**: Q2 2025

**Descrizione**: Estensione del supporto a lingue diverse dall'inglese

**Rationale**:
- Mitigazione bias linguistico identificato
- Ampliamento user base potenziale
- Compliance migliorata con non-discrimination requirements

**Requirements**:
- Dataset multilingue per training
- Lemmatization support per lingue target
- Cultural adaptation delle stop words
- Extended testing framework

**Resources Needed**:
- Linguist consultant per data validation
- Additional compute per larger models
- Extended QA testing cycles

**Risk Assessment**: MEDIO (complexity, resource intensive)

---

### PLAN-002: Explainable AI Features

**Data**: 2024-12-20  
**Categoria**: Enhancement/Compliance  
**Priority**: Alto  
**Timeline**: Q1 2025

**Descrizione**: Implementazione di feature per spiegabilità delle decisioni

**Rationale**:
- Miglioramento trasparenza utente
- Support per audit e compliance
- Facilitation del feedback loop per bias detection
- User trust improvement

**Technical Approach**:
- SHAP values per feature importance explanation
- Confidence scoring enhancement
- Top-contributing terms identification
- Decision boundary visualization

**Acceptance Criteria**:
- Explanation generation < 100ms additional latency
- User-friendly explanation format
- Integration con existing UI
- Audit trail di explanations

**Stakeholder Approval**: Pending

---

## 8. Templates e Procedure

### 8.1 Template per Nuove Decisioni

```markdown
### DEC-XXX: [Titolo Decisione]

**Data**: YYYY-MM-DD  
**Categoria**: [Tecnica/Operativa/Legale/Etica]  
**Approvatore**: [Nome e Ruolo]  
**Status**: [Proposta/Implementata/Revocata]

**Decisione**: [Descrizione concisa]

**Rationale**:
- [Motivazione principale]
- [Considerazioni aggiuntive]

**Alternative Considerate**:
1. **Opzione 1**: Pro/Contro
2. **Opzione 2**: Pro/Contro

**Impatto**:
- [Impatto tecnico]
- [Impatto operativo]  
- [Impatto business]

**Metriche di Successo**:
- [Metrica 1]
- [Metrica 2]

**Review Date**: [Data revisione pianificata]
```

### 8.2 Procedure di Approval

**Decisioni Tecniche**:
- Owner: Technical Lead
- Review: AI System Owner (se impact >10% performance)
- Approval: AI System Owner

**Decisioni Operative**:
- Owner: AI System Owner
- Review: Technical Lead + Quality Assurance
- Approval: AI System Owner

**Decisioni Legali/Compliance**:
- Owner: AI System Owner
- Review: Legal team + DPO
- Approval: Legal + AI System Owner

**Decisioni Architetturali**:
- Owner: Technical Lead
- Review: Full Risk Committee
- Approval: AI System Owner + Risk Committee

---

**Documento mantenuto da**: Christian Putzu  
**Frequency di aggiornamento**: Ongoing (real-time)  
**Archive policy**: Mantenimento permanente  
**Access level**: Internal team + auditors 