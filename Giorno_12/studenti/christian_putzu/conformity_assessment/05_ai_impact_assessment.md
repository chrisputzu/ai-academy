# AI Impact Assessment (AIIA)

**Sistema**: SMS Spam Detection Classifier  
**Tipo Assessment**: AI Impact Assessment secondo EU AI Act  
**Data**: Dicembre 2024  
**Versione**: 1.0  
**Responsabile**: Christian Putzu

## 1. Executive Summary

### 1.1 Scopo dell'Assessment

Questo AI Impact Assessment (AIIA) valuta l'impatto del sistema SMS Spam Detection sui diritti fondamentali, la sicurezza e altri aspetti societali in conformità con i requisiti dell'EU AI Act per sistemi a rischio limitato.

### 1.2 Risultati Principali

**Classificazione Rischio**: LIMITATO  
**Impatto Complessivo**: MEDIO-BASSO  
**Raccomandazione**: APPROVA con mitigazioni implementate  

**Aree di Attenzione**:
- Potenziale bias linguistico
- Gestione falsi positivi
- Trasparenza verso gli utenti

## 2. Descrizione del Sistema AI

### 2.1 Funzionalità Principali

Il sistema implementa un classificatore binario per distinguere messaggi SMS legittimi da spam utilizzando:
- Natural Language Processing (NLP)
- Machine Learning supervisionato (Random Forest)
- Feature extraction tramite TF-IDF

### 2.2 Stakeholder Identificati

**Utenti Primari**:
- Destinatari di messaggi SMS
- Gestori di sistemi di messaggistica
- Amministratori IT

**Utenti Secondari**:
- Operatori di telecomunicazioni
- Autorità di regolamentazione
- Sviluppatori di applicazioni

**Soggetti Interessati**:
- Mittenti legittimi di messaggi
- Marketer e aziende
- Autorità per la protezione dei dati

## 3. Valutazione dell'Impatto sui Diritti Fondamentali

### 3.1 Diritto alla Privacy e Protezione dei Dati

**Impatto Identificato**: BASSO

**Analisi**:
- Il sistema processa contenuto testuale dei messaggi
- Nessun storage permanente dei messaggi personali
- Processing limitato alla classificazione

**Misure di Protezione**:
- Data minimization principle applicato
- Encryption dei dati in transito
- Access logging per audit trail
- Compliance GDPR implementata

**Residual Risk**: BASSO
- Mitigato attraverso controlli tecnici e organizzativi

### 3.2 Libertà di Espressione e Comunicazione

**Impatto Identificato**: MEDIO

**Analisi**:
- Possibilità di blocco messaggi legittimi (falsi positivi)
- Potenziale interferenza con comunicazioni commerciali legittime
- Limitazione indiretta della libertà di espressione

**Potenziali Conseguenze**:
- Interruzione di comunicazioni importanti
- Limitazione accesso a informazioni
- Discriminazione di stili comunicativi specifici

**Misure di Mitigazione**:
- Soglie di confidence configurabili
- Meccanismo di whitelist per mittenti fidati
- Sistema di feedback per correzioni
- Audit regolari delle decisioni del sistema

**Residual Risk**: MEDIO-BASSO
- Controlli implementati riducono significativamente l'impatto

### 3.3 Non-Discriminazione e Uguaglianza

**Impatto Identificato**: MEDIO

**Analisi**:
- Potenziale bias linguistico (prevalenza inglese nel training)
- Discriminazione verso varietà linguistiche minoritarie
- Possibile impatto disuguale su gruppi demografici

**Aree di Bias Identificate**:
- **Linguistico**: Performance inferiore per non-native speakers
- **Culturale**: Definizioni di spam culturalmente specifiche
- **Socioeconomico**: Stili comunicativi associati a status

**Misure di Mitigazione**:
- Testing periodico su sottogruppi demografici
- Monitoring metriche di fairness
- Diversificazione dataset di training
- Feedback loop per identificazione bias

**Residual Risk**: MEDIO
- Richiede monitoraggio continuo e aggiornamenti periodici

### 3.4 Diritto al Ricorso e Procedimento Equo

**Impatto Identificato**: BASSO

**Analisi**:
- Sistema fornisce classificazioni automatiche
- Utenti potrebbero non avere meccanismi di ricorso immediati
- Mancanza di spiegabilità delle decisioni

**Misure di Mitigazione**:
- Documentazione del processo decisionale
- Procedure di escalation per contestazioni
- Human oversight per casi complessi
- Logging completo per audit trail

**Residual Risk**: BASSO
- Adeguato framework di accountability implementato

## 4. Impatto sulla Sicurezza

### 4.1 Sicurezza degli Utenti

**Benefici**:
- Protezione da tentativi di phishing
- Riduzione esposizione a contenuti dannosi
- Prevenzione truffe via SMS

**Rischi**:
- Falsa sicurezza da falsi negativi
- Dipendenza eccessiva dal sistema automatico

**Valutazione Netta**: POSITIVO
- I benefici superano i rischi identificati

### 4.2 Sicurezza del Sistema

**Vulnerabilità Identificate**:
- Possibili adversarial attacks sui dati di input
- Dependency vulnerabilities nelle librerie
- Model poisoning (rischio basso)

**Controlli Implementati**:
- Input validation e sanitization
- Regular security updates
- Isolated execution environment
- Monitoring per anomalie

**Livello di Sicurezza**: BUONO
- Controlli proporzionati al livello di rischio

## 5. Impatto Sociale ed Economico

### 5.1 Impatto Economico

**Benefici Economici**:
- Riduzione costi di gestione spam manuale
- Miglioramento efficienza comunicazioni business
- Riduzione perdite da truffe SMS

**Costi Economici**:
- Costi di sviluppo e manutenzione
- Possibili perdite da falsi positivi
- Training e formazione del personale

**Valutazione Netta**: POSITIVO
- ROI positivo atteso nel medio termine

### 5.2 Impatto Sociale

**Benefici Sociali**:
- Miglioramento qualità comunicazioni
- Protezione utenti vulnerabili
- Riduzione stress da spam

**Rischi Sociali**:
- Potenziale esclusione digitale per alcuni gruppi
- Over-reliance su sistemi automatici
- Erosione competenze umane di discernimento

**Considerazioni**:
- Necessità di mantenere human-in-the-loop
- Educazione utenti sui limiti del sistema
- Monitoring impatti a lungo termine

## 6. Valutazione delle Alternative

### 6.1 Alternative Considerate

**Opzione 1: Nessun Filtro Automatico**
- Pro: Nessuna interferenza con comunicazioni
- Contro: Esposizione completa a spam e truffe
- Valutazione: INADEGUATO per protezione utenti

**Opzione 2: Filtro Basato su Regole**
- Pro: Trasparenza e predicibilità
- Contro: Facilmente aggirabile, manutenzione complessa
- Valutazione: LIMITATO in efficacia

**Opzione 3: ML con Human-in-the-Loop**
- Pro: Bilanciamento automazione/controllo umano
- Contro: Scalabilità limitata, costi maggiori
- Valutazione: BUONO ma costoso

**Opzione 4: Soluzione Ibrida (Scelta Attuale)**
- Pro: Efficacia elevata con controlli adeguati
- Contro: Complessità gestionale
- Valutazione: OTTIMALE per il contesto

### 6.2 Giustificazione della Scelta

La soluzione implementata rappresenta il miglior compromesso tra:
- Efficacia nella rilevazione spam
- Protezione dei diritti fondamentali
- Sostenibilità economica
- Conformità normativa

## 7. Misure di Mitigazione e Controllo

### 7.1 Misure Tecniche

**Implemented Controls**:
- Bias detection algoritmi
- Performance monitoring continuo
- Input validation robusta
- Output confidence scoring

**Planned Enhancements**:
- Explainable AI features
- Multi-language support
- Advanced fairness metrics
- Real-time bias correction

### 7.2 Misure Organizzative

**Governance**:
- Risk management committee
- Regular impact assessments
- Stakeholder engagement process
- Incident response procedures

**Training**:
- Staff awareness programs
- Technical competency development
- Ethics and bias training
- Regulatory compliance updates

### 7.3 Misure Procedurali

**Monitoring**:
- Continuous performance tracking
- Periodic bias audits
- User feedback collection
- Regulatory compliance checks

**Review Cycles**:
- Monthly performance review
- Quarterly bias assessment
- Semi-annual impact review
- Annual comprehensive audit

## 8. Piano di Monitoraggio Post-Deployment

### 8.1 Key Performance Indicators

**Technical KPIs**:
- Accuracy: >95% (target), >90% (threshold)
- False Positive Rate: <5% (target), <10% (threshold)
- Response Time: <500ms (target), <1000ms (threshold)

**Fairness KPIs**:
- Equalized Odds Difference: <10%
- Demographic Parity Difference: <15%
- Individual Fairness Violation Rate: <5%

**User Experience KPIs**:
- User Satisfaction Score: >80%
- Complaint Rate: <1%
- Appeal Success Rate: <20%

### 8.2 Review e Aggiornamento

**Review Schedule**:
- **Monthly**: Technical performance metrics
- **Quarterly**: Bias and fairness assessment
- **Semi-annually**: Comprehensive impact review
- **Annually**: Full AIIA update

**Trigger Events for Re-assessment**:
- Significant change in system functionality
- New regulatory requirements
- Identified bias or discrimination issues
- Major incident or security breach
- Substantial user complaints

## 9. Stakeholder Consultation

### 9.1 Processo di Consultazione

**Internal Stakeholders**:
- Technical development team
- Legal and compliance team
- Product management
- User experience team

**External Stakeholders**:
- User representatives
- Privacy advocacy groups
- Industry experts
- Regulatory bodies

### 9.2 Feedback Incorporato

**Key Themes from Consultation**:
- Emphasis on transparency and explainability
- Concerns about bias and discrimination
- Need for robust appeal mechanisms
- Importance of user control and choice

**Changes Made**:
- Enhanced bias monitoring procedures
- Improved user notification processes
- Strengthened appeal and feedback mechanisms
- Additional fairness testing requirements

## 10. Conclusioni e Raccomandazioni

### 10.1 Valutazione Complessiva

Il sistema SMS Spam Detection presenta un **profilo di rischio accettabile** per il deployment in produzione, con le seguenti considerazioni:

**Punti di Forza**:
- Benefici chiari per la protezione degli utenti
- Rischi limitati sui diritti fondamentali
- Controlli adeguati implementati
- Conformità con EU AI Act

**Aree di Attenzione**:
- Monitoraggio continuo del bias linguistico
- Gestione efficace dei falsi positivi
- Mantenimento della trasparenza verso gli utenti

### 10.2 Raccomandazioni

**Immediate Actions**:
1. Implementare sistema di notifica utenti sull'uso AI
2. Configurare monitoring automatico delle metriche di fairness
3. Stabilire processo di feedback e appeal
4. Documentare procedure operative

**Short-term (3-6 mesi)**:
1. Condurre primo audit completo di bias
2. Implementare dashboard di monitoring
3. Sviluppare materiali di training per staff
4. Stabilire partnership per dataset diversificati

**Long-term (6-12 mesi)**:
1. Valutare implementazione explainable AI
2. Espandere support multi-linguistico
3. Sviluppare advanced fairness algorithms
4. Pianificare assessment di conformità EU AI Act

### 10.3 Approvazione per Deployment

**Raccomandazione**: **APPROVA** per deployment in produzione

**Condizioni**:
- Implementazione di tutte le misure di mitigazione identificate
- Completamento del training del personale
- Attivazione dei sistemi di monitoring
- Preparazione delle procedure di incident response

---

**Assessment condotto da**: Christian Putzu  
**Review interno**: Dr. Elena Rossi, Senior ML Engineer  
**Approvazione legale**: Avv. Giuseppe Neri, Legal Counsel  
**Approvazione finale**: Prof. Marco Bianchi, Direttore AI Academy  
**Data di validità**: 2 Luglio 2025