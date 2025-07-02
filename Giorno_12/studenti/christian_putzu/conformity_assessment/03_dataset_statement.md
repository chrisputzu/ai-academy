# Dataset Statement - SMS Spam Collection

**Dataset**: SMS Spam Collection  
**Sistema**: SMS Spam Detection Classifier  
**Data**: Dicembre 2024  
**Versione**: 1.0  
**Autore**: Christian Putzu

## 1. Informazioni Generali del Dataset

### 1.1 Identificazione

- **Nome**: SMS Spam Collection Dataset
- **Fonte**: Kaggle (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Origine**: UCI Machine Learning Repository
- **Formato**: CSV (Comma Separated Values)
- **Encoding**: Latin-1
- **Dimensione**: ~500KB

### 1.2 Contenuto

- **Totale Record**: 5,574 messaggi SMS
- **Attributi**: 2 colonne principali (v1, v2)
  - `v1`: Etichetta della classe ("ham" o "spam")
  - `v2`: Contenuto testuale del messaggio
- **Periodo Temporale**: Non specificato nel dataset
- **Lingua Prevalente**: Inglese

## 2. Distribuzione delle Classi

### 2.1 Bilanciamento del Dataset

| Classe | Conteggio | Percentuale |
|--------|-----------|-------------|
| ham    | 4,827     | 86.6%       |
| spam   | 747       | 13.4%       |
| **Totale** | **5,574** | **100%**    |

### 2.2 Caratteristiche di Distribuzione

- **Ratio di Sbilanciamento**: 6.5:1 (ham:spam)
- **Impatto**: Necessità di tecniche per gestire class imbalance
- **Mitigazione**: Utilizzo di metriche bilanciate (F1-score, balanced accuracy)

## 3. Caratteristiche dei Dati

### 3.1 Analisi Testuale

**Lunghezza dei Messaggi**:
- Media: ~80 caratteri per messaggio
- Range: 2-910 caratteri
- Mediana: ~60 caratteri

**Vocabolario**:
- Termini unici: ~8,000-10,000 dopo preprocessing
- Stopwords: Rimosse automaticamente (inglese)
- Caratteristiche linguistiche: Abbreviazioni SMS, emoji, simboli

### 3.2 Qualità dei Dati

**Completezza**:
- Valori mancanti: 0% (verificato)
- Record duplicati: Non verificati esplicitamente
- Consistenza encoding: Gestita con latin-1

**Pulizia Applicata**:
- Rimozione colonne inutilizzate (v3, v4, v5)
- Lemmatization tramite WordNetLemmatizer
- Conversione a stringhe per uniformità

## 4. Provenienza e Raccolta

### 4.1 Metodologia di Raccolta

**Fonte Originale**: 
- SMS Spam Corpus v.0.1 Big (Jose Maria Gomez Hidalgo)
- British English SMS messages
- Grumbletext Web site SMS messages
- Caroline Tag's PhD Thesis SMS messages
- NUS SMS Corpus

**Processo di Aggregazione**:
- Combinazione di diversi corpora SMS
- Standardizzazione delle etichette
- Rimozione di duplicati (parziale)

### 4.2 Licenza e Diritti

**Licenza**: 
- Public Domain / Academic Use
- Disponibile tramite UCI ML Repository
- Kaggle redistribution sotto Creative Commons

**Limitazioni d'Uso**:
- Solo per scopi di ricerca e educazione
- Non per uso commerciale senza verifica dei diritti
- Rispetto privacy per dati originali

## 5. Bias e Limitazioni Identificate

### 5.1 Bias Linguistici

**Lingua**:
- **Predominanza Inglese**: ~95% dei messaggi in inglese
- **Dialetti**: Prevalentemente British English
- **Implicazioni**: Performance limitate per altre lingue

**Varietà Linguistica**:
- Gergo SMS specifico del periodo di raccolta
- Abbreviazioni temporalmente situate
- Mancanza di emoji moderne e unicode

### 5.2 Bias Temporali

**Periodo di Raccolta**:
- Dataset potenzialmente datato (epoca pre-smartphone)
- Pattern di spam evolutivi non catturati
- Tecniche moderne di spam non rappresentate

**Evoluzione del Linguaggio**:
- Cambiamenti nel linguaggio SMS
- Nuove forme di comunicazione (emoji, reaction)
- Shift culturali nella comunicazione

### 5.3 Bias Geografici e Culturali

**Rappresentatività Geografica**:
- Prevalenza di fonti britanniche
- Limitata diversità geografica
- Possibili bias culturali nella definizione di spam

**Contesto Sociale**:
- Definizione di spam culturalmente specifica
- Variazioni regionali nelle pratiche di marketing
- Differenze normative per paese

## 6. Misure di Controllo Qualità

### 6.1 Validazione Implementata

**Pre-processing Checks**:
```python
# Verifica valori mancanti
print(f"Missing values:\n{df.isnull().sum()}")

# Controllo dimensioni
print(f"DataFrame shape: {df.shape}")

# Verifica distribuzione classi
print(df['v1'].value_counts())
```

**Quality Assurance**:
- Verifica encoding corretto (latin-1)
- Controllo consistenza etichette
- Validazione lunghezza messaggi

### 6.2 Limitazioni non Risolte

**Problemi Identificati**:
- Possibili duplicati non rimossi
- Mancanza di metadata temporali
- Assenza di informazioni demografiche

**Raccomandazioni**:
- Implementare deduplication più robusta
- Considerare dataset più recenti per production
- Integrare con fonti dati multilinguistiche

## 7. Considerazioni Privacy e Sicurezza

### 7.1 Privacy Protection

**Anonimizzazione**:
- Nessuna informazione personale identificativa
- Contenuti già anonimizzati alla fonte
- Rimozione di numeri di telefono e indirizzi

**Data Minimization**:
- Utilizzo solo del contenuto necessario per classificazione
- Nessun storage permanente di messaggi personali
- Processing limitato al training del modello

### 7.2 Sicurezza dei Dati

**Storage**:
- Download temporaneo tramite kagglehub
- Nessuna persistenza locale prolungata
- Cleanup automatico dopo utilizzo

**Access Control**:
- Accesso limitato al solo personale autorizzato
- Logging degli accessi ai dati
- Audit trail per modifiche

## 8. Impatti e Considerazioni Etiche

### 8.1 Potenziali Impatti Negativi

**Amplificazione Bias**:
- Reinforcement di stereotipi linguistici
- Discriminazione verso varietà linguistiche minoritarie
- Penalizzazione di stili comunicativi non standard

**Fairness Concerns**:
- Performance non uniforme tra gruppi demografici
- Possibile over-filtering per specifiche comunità
- Under-representation di pattern spam recenti

### 8.2 Misure di Mitigazione

**Monitoring**:
- Testing regolare su sottogruppi linguistici
- Analisi delle performance per diverse varietà di inglese
- Feedback loop per identificare bias emergenti

**Improvement Actions**:
- Integrazione periodica di nuovi dati
- Diversificazione delle fonti
- Collaborazione con comunità multilinguistiche

## 9. Aggiornamento e Manutenzione

### 9.1 Strategia di Aggiornamento

**Frequenza**:
- Valutazione semestrale della rilevanza
- Aggiornamento annuale con nuovi dataset
- Monitoring continuo della performance degradation

**Criteri per l'Aggiornamento**:
- Drop significativo nelle performance (>5%)
- Emergenza di nuovi pattern spam non coperti
- Feedback negativo dagli utenti

### 9.2 Versionamento

**Version Control**:
- Tracking delle versioni del dataset utilizzate
- Documentazione dei cambiamenti tra versioni
- Backward compatibility testing

**Documentation**:
- Changelog dettagliato per ogni aggiornamento
- Impact assessment per ogni modifica
- Approval process per nuove versioni

## 10. Conclusioni e Raccomandazioni

### 10.1 Adeguatezza per l'Uso

Il dataset SMS Spam Collection è **ADEGUATO** per l'uso previsto con le seguenti considerazioni:

**Punti di Forza**:
- Dimensione sufficiente per training
- Etichette di alta qualità
- Rappresentatività del dominio SMS

**Limitazioni da Considerare**:
- Bias linguistico verso l'inglese
- Potenziale obsolescenza temporale
- Sbilanciamento delle classi

### 10.2 Raccomandazioni Operative

1. **Monitoraggio Continuo**: Implementare metriche di fairness
2. **Diversificazione**: Integrare dataset multilingue
3. **Aggiornamento**: Pianificare refresh periodico dei dati
4. **Validazione**: Testing regolare su nuovi pattern di spam

---

**Documento preparato da**: Christian Putzu  
**Data Steward**: Dr. Francesca Verdi, Data Protection Officer  
**Approvazione Legal**: Avv. Giuseppe Neri, Legal Counsel  
**Prossima revisione**: 20 Luglio 2025