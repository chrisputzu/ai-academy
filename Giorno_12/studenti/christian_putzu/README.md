# SMS Spam Detection - AI Academy Giorno 12

## Descrizione del Progetto

Questo progetto implementa un sistema di classificazione automatica per identificare messaggi SMS spam utilizzando tecniche di machine learning e natural language processing. Il sistema è stato sviluppato come parte dell'esercizio del Giorno 12 dell'AI Academy.

## Obiettivi

1. **Classificazione Automatica**: Distinguere tra messaggi SMS legittimi ("ham") e spam
2. **Preprocessing Avanzato**: Applicare tecniche di lemmatization per migliorare la qualità dei dati
3. **Valutazione delle Performance**: Misurare accuracy, precision, recall e f1-score
4. **Conformità Normativa**: Implementare le misure di conformità richieste dall'EU AI Act

## Dataset

- **Fonte**: SMS Spam Collection Dataset da Kaggle
- **URL**: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **Dimensione**: 5,574 messaggi SMS
- **Classi**: 
  - `ham`: Messaggi legittimi (87.4%)
  - `spam`: Messaggi spam (12.6%)

## Architettura del Sistema

### 1. Preprocessing
- **Lemmatization**: Riduzione delle parole alla forma lemma usando WordNetLemmatizer
- **Encoding**: Gestione dell'encoding latino-1 per caratteri speciali
- **Pulizia**: Rimozione di colonne non necessarie dal dataset

### 2. Feature Engineering
- **TF-IDF Vectorization**: Trasformazione del testo in vettori numerici
- **Stop Words Removal**: Rimozione automatica delle parole comuni inglesi
- **Sparse Matrix**: Rappresentazione efficiente per ridurre l'uso di memoria

### 3. Modello di Machine Learning
- **Algoritmo**: Random Forest Classifier
- **Parametri**: 100 alberi decisionali (n_estimators=100)
- **Split**: 80% training, 20% test
- **Random State**: 42 per riproducibilità

## Installazione e Utilizzo

### Prerequisiti
```bash
pip install kagglehub pandas scikit-learn nltk
```

### Esecuzione
```bash
python app_es2.py
```

### File di Input
- `messaggi test.txt`: File contenente messaggi di test personalizzati

## Performance del Modello

Il modello raggiunge le seguenti performance sul test set:

- **Accuracy**: ~95.8%
- **Precision (Ham)**: ~96%
- **Recall (Ham)**: ~99%
- **Precision (Spam)**: ~95%
- **Recall (Spam)**: ~75%

## Struttura del Progetto

```
Giorno_12/studenti/christian_putzu/
├── README.md                    # Documentazione principale
├── app_es2.py                  # Codice principale
├── messaggi test.txt           # Messaggi di test
├── es_1_report.pdf            # Report dell'esercizio 1
└── conformity_assessment/      # Documentazione EU AI Act
    ├── 01_risk_assessment.md
    ├── 02_technical_description.md
    ├── 03_dataset_statement.md
    ├── 04_risk_management.md
    ├── 05_ai_impact_assessment.md
    ├── 06_decision_log.md
    ├── 07_user_manual.md
    ├── 08_conformity_declaration.md
    └── 09_technical_dossier.md
```

## Conformità EU AI Act

Questo sistema è stato classificato come **LIMITATO RISCHIO** secondo l'EU AI Act, in quanto:
- Interagisce direttamente con utenti umani
- Può influenzare decisioni di comunicazione
- Richiede trasparenza sull'utilizzo di sistemi AI

Tutta la documentazione di conformità è disponibile nella cartella `conformity_assessment/`.

## Limitazioni e Rischi Identificati

### Falsi Positivi
- **Rischio**: Messaggi legittimi classificati come spam
- **Impatto**: Perdita di comunicazioni importanti
- **Mitigazione**: Soglie di confidence configurabili

### Falsi Negativi
- **Rischio**: Messaggi spam non rilevati
- **Impatto**: Esposizione a contenuti indesiderati
- **Mitigazione**: Aggiornamento periodico del modello

### Bias nei Dati
- **Lingua**: Dataset prevalentemente in inglese
- **Temporale**: Dati potrebbero essere datati
- **Culturale**: Definizione di spam varia per contesto

## Autore

**Christian Putzu**
- AI Academy - Giorno 12
- Data: 2 Luglio 2025
- Email: christian.putzu@it.ey.com
- Matricola: 2025-SMS-001

## Licenza

Questo progetto è sviluppato per scopi educativi nell'ambito dell'AI Academy.
