# Manuale Utente - Sistema SMS Spam Detection

**Sistema**: SMS Spam Detection Classifier  
**Versione**: 1.0  
**Data**: Dicembre 2024  
**Target**: Utenti finali e amministratori di sistema

## 1. Introduzione

### 1.1 Benvenuti

Questo manuale fornisce informazioni complete per utilizzare il sistema SMS Spam Detection, un classificatore automatico progettato per distinguere messaggi SMS legittimi da spam utilizzando tecnologie di Intelligenza Artificiale.

### 1.2 Cosa Fa Questo Sistema

Il sistema SMS Spam Detection:
- **Classifica automaticamente** i messaggi SMS in "ham" (legittimi) o "spam"
- **Utilizza l'Intelligenza Artificiale** per analizzare il contenuto testuale
- **Protegge gli utenti** da contenuti indesiderati e potenzialmente dannosi
- **Mantiene la privacy** processando i dati senza memorizzazione permanente

### 1.3 Importante: Uso dell'AI

⚠️ **AVVISO TRASPARENZA AI**: Questo sistema utilizza Intelligenza Artificiale per processare automaticamente i vostri messaggi SMS. Come richiesto dalla normativa EU AI Act, vi informiamo che:

- Il sistema prende decisioni automatiche sui vostri messaggi
- Può occasionalmente commettere errori (falsi positivi/negativi)
- Avete il diritto di controllare e contestare le decisioni
- I vostri dati sono processati secondo i principi GDPR

## 2. Come Funziona

### 2.1 Processo di Classificazione

```
[Messaggio SMS] → [Analisi AI] → [Classificazione] → [Risultato: ham/spam]
```

1. **Input**: Il sistema riceve il testo del messaggio SMS
2. **Preprocessing**: Il testo viene normalizzato e processato
3. **Analisi**: L'algoritmo AI analizza le caratteristiche del messaggio
4. **Classificazione**: Il sistema assegna una categoria (ham/spam)
5. **Output**: Viene fornito il risultato con livello di confidenza

### 2.2 Tecnologia Utilizzata

- **Machine Learning**: Random Forest con 100 alberi decisionali
- **Natural Language Processing**: Analisi semantica del testo
- **Feature Engineering**: Estrazione automatica delle caratteristiche rilevanti
- **Preprocessing**: Lemmatization e normalizzazione del testo

## 3. Utilizzo Base

### 3.1 Installazione

**Prerequisiti**:
- Python 3.7 o superiore
- Connessione internet (solo per setup iniziale)
- 4GB RAM (8GB raccomandati)

**Installazione**:
```bash
# 1. Installare le dipendenze
pip install kagglehub pandas scikit-learn nltk

# 2. Download risorse NLTK (solo prima volta)
python -c "import nltk; nltk.download('wordnet')"

# 3. Eseguire il sistema
python app_es2.py
```

### 3.2 Primo Utilizzo

1. **Avvio**: Eseguire `python app_es2.py`
2. **Download Dataset**: Il sistema scaricherà automaticamente il dataset di training
3. **Training**: Il modello verrà addestrato (circa 1-2 minuti)
4. **Test**: Il sistema testerà alcuni messaggi di esempio

### 3.3 Testing Personalizzato

**File di Test**: Creare un file `messaggi test.txt` con i messaggi da testare (uno per riga)

**Esempio**:
```
Free money! Click here now!
Meeting at 3pm tomorrow
Your account has been compromised
Happy birthday!
```

**Esecuzione**: Il sistema processerà automaticamente tutti i messaggi nel file

## 4. Interpretazione dei Risultati

### 4.1 Output del Sistema

Per ogni messaggio, il sistema fornisce:

```
Test Message: Free money! Click here now!
Predicted Label: spam
```

### 4.2 Livelli di Confidenza

Il sistema internamente calcola un livello di confidenza per ogni predizione:
- **Alta confidenza**: Decisione molto sicura
- **Media confidenza**: Decisione probabile ma non certa
- **Bassa confidenza**: Decisione incerta, potrebbero essere necessarie verifiche

### 4.3 Tipi di Errore Possibili

**Falsi Positivi** (Messaggi legittimi classificati come spam):
- Messaggi business con termini promozionali
- Comunicazioni con abbreviazioni eccessive
- Messaggi in lingue diverse dall'inglese

**Falsi Negativi** (Messaggi spam non rilevati):
- Spam sofisticati che imitano messaggi legittimi
- Nuovi pattern di spam non presenti nel training
- Messaggi con tecniche di evasion avanzate

## 5. Controllo e Personalizzazione

### 5.1 Configurazione Avanzata

**Parametri Modificabili** (in `app_es2.py`):

```python
# Numero di alberi nel Random Forest
n_estimators = 100  # Aumentare per maggiore accuracy

# Dimensione del test set
test_size = 0.2  # 20% per test, 80% per training

# Seed per riproducibilità
random_state = 42
```

### 5.2 Gestione degli Errori

**Se il sistema classifica erroneamente**:
1. Documentare il messaggio e la classificazione errata
2. Verificare se il messaggio presenta caratteristiche ambigue
3. Considerare se il pattern è nuovo/evoluto
4. Fornire feedback per miglioramenti futuri

**Procedure di Ricorso**:
- Conservare evidenza del messaggio originale
- Documentare l'impatto dell'errore
- Contattare l'amministratore di sistema
- Richiedere review manuale del caso

## 6. Privacy e Sicurezza

### 6.1 Protezione dei Dati

**Cosa Viene Processato**:
- Solo il contenuto testuale del messaggio
- Nessuna informazione personale o metadati
- Processing temporaneo in memoria

**Cosa NON Viene Memorizzato**:
- Contenuto completo dei messaggi
- Informazioni identificative personali
- Cronologia delle comunicazioni
- Dati di geolocalizzazione

### 6.2 Diritti dell'Utente (GDPR)

**I Vostri Diritti**:
- **Accesso**: Sapere come vengono processati i vostri dati
- **Rettifica**: Correggere informazioni inesatte
- **Cancellazione**: Richiedere la rimozione dei dati
- **Opposizione**: Opporsi al processing automatico
- **Portabilità**: Ottenere i vostri dati in formato leggibile

**Come Esercitare i Diritti**:
- Contattare: Dr. Francesca Verdi, Data Protection Officer
- Email: privacy@aiacademy.edu
- Tempo di risposta: Massimo 30 giorni

### 6.3 Opt-out dal Sistema

**Disattivare il Sistema AI**:
1. Modificare la configurazione per disabilitare l'AI
2. Utilizzare filtri basati su regole semplici
3. Processamento completamente manuale

**Impatti dell'Opt-out**:
- Perdita di protezione automatica da spam
- Maggiore esposizione a contenuti indesiderati
- Necessità di filtering manuale

## 7. Limitazioni e Considerazioni

### 7.1 Limitazioni Tecniche

**Lingue Supportate**:
- Primarily: Inglese
- Limited: Altre lingue europee
- Non supportate: Lingue non latine, caratteri speciali complessi

**Tipi di Contenuto**:
- Supportati: Testo semplice, abbreviazioni comuni
- Limitati: Emoji complessi, simboli speciali
- Non supportati: Immagini, file allegati

### 7.2 Limitazioni Operative

**Performance**:
- Latenza: ~500ms per messaggio
- Throughput: ~100 messaggi/minuto
- Memoria: Richiede ~2GB RAM per operazioni ottimali

**Manutenzione**:
- Aggiornamento modello: Trimestrale
- Monitoring bias: Mensile
- Review performance: Settimanale

### 7.3 Quando Non Utilizzare il Sistema

**Situazioni Inappropriate**:
- Messaggi di emergenza critici
- Comunicazioni legali sensibili
- Contenuti medici confidenziali
- Informazioni finanziarie riservate

**Alternative Consigliate**:
- Review manuale per contenuti critici
- Whitelist per mittenti fidati
- Sistemi specializzati per domini specifici

## 8. Risoluzione Problemi

### 8.1 Problemi Comuni

**Errore: "UnicodeDecodeError"**
- Causa: Caratteri speciali non supportati
- Soluzione: Verificare encoding del file di input

**Errore: "Model file not found"**
- Causa: Training non completato correttamente
- Soluzione: Rieseguire il training completo

**Performance Lente**
- Causa: Memoria insufficiente o file troppo grandi
- Soluzione: Ridurre batch size o aumentare RAM

### 8.2 Diagnostica

**Verificare Funzionamento**:
```python
# Test di base
test_message = "Hello, how are you?"
prediction = model.predict([test_message])
print(f"Prediction: {prediction}")
```

**Log di Sistema**:
- Verificare file di log per errori
- Monitorare utilizzo memoria
- Controllare tempi di risposta

### 8.3 Quando Contattare Support

**Contattare Immediatamente**:
- Classificazioni sistematicamente errate
- Problemi di privacy o sicurezza
- Errori tecnici persistenti
- Sospetti di bias o discriminazione

**Informazioni da Fornire**:
- Versione del sistema
- Messaggio di errore completo
- Esempi di messaggi problematici
- Configurazione utilizzata

## 9. Aggiornamenti e Manutenzione

### 9.1 Aggiornamenti Automatici

**Cosa Viene Aggiornato**:
- Modello di machine learning
- Dataset di training
- Algoritmi di preprocessing
- Metriche di performance

**Frequenza**:
- Aggiornamenti critici: Immediati
- Miglioramenti performance: Mensili
- Nuove funzionalità: Trimestrali

### 9.2 Backup e Recovery

**Backup Automatico**:
- Configurazione sistema
- Log delle decisioni
- Metriche performance
- User feedback

**Recovery Process**:
1. Identificazione del problema
2. Rollback alla versione stabile
3. Analisi root cause
4. Fix e re-deployment

## 10. Feedback e Miglioramento

### 10.1 Come Fornire Feedback

**Canali di Feedback**:
- Email: feedback@aiacademy.edu
- Form online: https://aiacademy.edu/feedback/sms-spam
- Issue tracking: https://github.com/aiacademy/sms-spam/issues

**Tipi di Feedback Utili**:
- Classificazioni errate
- Suggerimenti funzionalità
- Problemi di usabilità
- Questioni privacy

### 10.2 Utilizzo del Feedback

**Il Vostro Feedback Viene Utilizzato Per**:
- Migliorare accuracy del modello
- Identificare bias sistematici
- Sviluppare nuove funzionalità
- Ottimizzare user experience

**Processo di Miglioramento**:
1. Raccolta feedback utenti
2. Analisi pattern e trends
3. Sviluppo miglioramenti
4. Testing e validazione
5. Deployment aggiornamenti

---

## Contatti e Support

**Supporto Tecnico**: support@aiacademy.edu  
**Privacy Officer**: privacy@aiacademy.edu  
**Feedback Sistema**: feedback@aiacademy.edu  
**Emergenze**: emergency@aiacademy.edu

**Orari di Supporto**: Lunedì-Venerdì, 9:00-17:00  
**Tempo di Risposta**: 24-48 ore per questioni non critiche

---

**Versione Manuale**: 1.0  
**Ultimo Aggiornamento**: 2 Luglio 2025
**Prossima Revisione**: 20 Luglio 2025