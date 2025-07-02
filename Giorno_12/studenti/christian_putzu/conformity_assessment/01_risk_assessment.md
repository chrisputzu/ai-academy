# Risk Assessment - Sistema SMS Spam Detection

**Documento**: Valutazione del Rischio  
**Sistema**: SMS Spam Detection Classifier  
**Data**: Dicembre 2024  
**Versione**: 1.0  
**Autore**: Christian Putzu

## 1. Identificazione del Livello di Rischio

### Classificazione secondo EU AI Act

**LIVELLO DI RISCHIO IDENTIFICATO: LIMITATO**

### Giustificazione della Classificazione

Il sistema SMS Spam Detection rientra nella categoria **Limited Risk** per i seguenti motivi:

1. **Interazione con Utenti Umani**: Il sistema classifica automaticamente messaggi destinati a utenti finali
2. **Trasparenza Richiesta**: Gli utenti devono essere informati che stanno interagendo con un sistema AI
3. **Impatto Moderato**: Le decisioni del sistema influenzano la comunicazione ma non hanno impatti critici su sicurezza o diritti fondamentali

### Esclusioni da Altre Categorie

- **NON Proibito**: Non manipola comportamenti umani in modo subliminale
- **NON Alto Rischio**: Non è utilizzato in settori critici (sanità, sicurezza, giustizia, infrastrutture critiche)
- **NON Foundation Model**: È un modello specifico per una singola applicazione

## 2. Analisi dei Rischi Specifici

### 2.1 Falsi Positivi

**Descrizione**: Messaggi legittimi classificati erroneamente come spam

**Probabilità**: Media (5-25% dei casi spam)
**Impatto**: Medio
**Livello di Rischio**: MEDIO

**Conseguenze**:
- Perdita di comunicazioni importanti (business, emergenze, notifiche)
- Interruzione di servizi basati su SMS (2FA, OTP)
- Frustrazione dell'utente e perdita di fiducia

**Misure di Mitigazione**:
- Implementazione di whitelist per mittenti fidati
- Soglie di confidence configurabili
- Meccanismo di feedback utente per correzioni
- Revisione periodica dei falsi positivi

### 2.2 Falsi Negativi

**Descrizione**: Messaggi spam non rilevati dal sistema

**Probabilità**: Media (10-20% dei casi spam)
**Impatto**: Basso-Medio
**Livello di Rischio**: BASSO-MEDIO

**Conseguenze**:
- Esposizione a contenuti indesiderati
- Possibili tentativi di phishing o truffe
- Degradazione dell'esperienza utente

**Misure di Mitigazione**:
- Aggiornamento regolare del modello con nuovi pattern
- Sistema di reporting per spam non rilevato
- Combinazione con altri sistemi di sicurezza

### 2.3 Bias e Discriminazione

**Descrizione**: Il modello potrebbe discriminare basandosi su caratteristiche linguistiche o culturali

**Probabilità**: Media
**Impatto**: Medio
**Livello di Rischio**: MEDIO

**Aree di Bias Identificate**:
- **Linguistico**: Dataset prevalentemente in inglese
- **Culturale**: Definizioni di spam variano per contesto geografico
- **Temporale**: Pattern di spam evolutivi non catturati

**Misure di Mitigazione**:
- Monitoraggio delle performance per diversi gruppi demografici
- Diversificazione del dataset di training
- Test regolari su sottogruppi specifici

## 3. Valutazione dell'Impatto

### 3.1 Impatto sui Diritti Fondamentali

**Libertà di Comunicazione**: BASSO
- Il sistema può limitare la ricezione di alcuni messaggi
- Tuttavia, l'utente mantiene il controllo finale

**Privacy**: BASSO
- Il sistema analizza contenuto dei messaggi
- Non memorizza permanentemente i contenuti
- Processa solo metadati e pattern testuali

**Non-discriminazione**: MEDIO
- Possibili bias linguistici e culturali
- Richiede monitoraggio continuo

### 3.2 Impatto Operativo

**Affidabilità del Servizio**: MEDIO
- Sistema automatizzato riduce intervento umano
- Errori possono accumularsi senza supervisione

**Scalabilità**: BASSO
- Sistema progettato per gestire volumi elevati
- Performance degradano gradualmente con aumento del carico

## 4. Misure di Controllo del Rischio

### 4.1 Controlli Tecnici

1. **Monitoraggio delle Performance**
   - Tracking continuo di accuracy, precision, recall
   - Alert automatici per degradazione delle performance
   - Dashboard in tempo reale

2. **Validazione dei Dati**
   - Controlli di qualità sui dati di input
   - Rilevamento di drift nei dati
   - Sanitizzazione degli input

3. **Robustezza del Modello**
   - Test di stress con dati adversariali
   - Validazione su dataset esterni
   - Backup model per fallback

### 4.2 Controlli Organizzativi

1. **Governance**
   - Responsabile designato per il sistema AI
   - Processo di escalation per problemi critici
   - Review periodiche del sistema

2. **Training del Personale**
   - Formazione sulle limitazioni del sistema
   - Procedure per gestione degli errori
   - Sensibilizzazione sui bias

### 4.3 Controlli di Processo

1. **Testing e Validazione**
   - Test periodici su nuovi dataset
   - Validazione A/B testing
   - User acceptance testing

2. **Audit e Compliance**
   - Audit interni trimestrali
   - Log completi delle decisioni
   - Documentazione delle modifiche

## 5. Piano di Monitoraggio

### 5.1 Metriche di Monitoraggio

**Metriche Tecniche**:
- Accuracy, Precision, Recall (soglia minima: 90%)
- Latenza di risposta (soglia massima: 500ms)
- Throughput (messaggi/secondo)

**Metriche di Qualità**:
- Tasso di falsi positivi (<5%)
- Tasso di falsi negativi (<15%)
- Soddisfazione utente (>80%)

**Metriche di Conformità**:
- Incidenti di bias riportati
- Tempo di risoluzione dei problemi
- Coverage dei test di non-discriminazione

### 5.2 Frequenza di Review

- **Quotidiana**: Monitoraggio automatico delle metriche tecniche
- **Settimanale**: Analisi dei falsi positivi/negativi
- **Mensile**: Review delle performance e bias testing
- **Trimestrale**: Audit completo e aggiornamento documentazione
- **Annuale**: Rivalutazione completa del rischio

## 6. Conclusioni

Il sistema SMS Spam Detection presenta un profilo di rischio **LIMITATO** secondo l'EU AI Act. I principali rischi identificati sono gestibili attraverso le misure di mitigazione proposte. Il sistema richiede:

1. **Trasparenza**: Informazione agli utenti sull'uso dell'AI
2. **Monitoraggio**: Sorveglianza continua delle performance
3. **Aggiornamento**: Mantenimento periodico del modello
4. **Audit**: Verifiche regolari di conformità

Il sistema può essere deploiato in produzione con le appropriate misure di controllo implementate.

---

**Documento approvato da**: Prof. Marco Bianchi, Direttore AI Academy  
**Data approvazione**: 2 Luglio 2025
**Prossima revisione**: 20 Luglio 2025