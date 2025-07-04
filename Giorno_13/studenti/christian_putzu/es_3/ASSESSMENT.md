# Analisi di Conformità AI Act – Caso Pratico: Analisi automatica di email bancaria

## 1. Descrizione del sistema
L'applicazione sviluppata combina un modello NER locale e GPT-4 in cloud per analizzare e rispondere automaticamente alle richieste su documenti aziendali. Nel caso pratico, viene analizzata un'email bancaria contenente dati personali e bancari dell'utente.

## 2. Classificazione del rischio (Tier)
Secondo l'AI Act, il sistema rientra nella categoria rischio limitato se utilizzato solo per fornire informazioni o assistenza automatica, con obblighi di trasparenza verso l'utente. Tuttavia, se l'output del sistema viene utilizzato per prendere decisioni che impattano diritti, accesso a servizi essenziali o valutazioni finanziarie, potrebbe ricadere nella categoria alto rischio, con obblighi più stringenti.

Nel caso specifico (analisi e risposta automatica a richieste su email bancaria, senza decisioni automatizzate vincolanti), si considera rischio limitato.

## 3. Obblighi applicabili
- Trasparenza: L'utente deve essere informato che sta interagendo con un sistema di intelligenza artificiale.
- Protezione dati: Devono essere rispettati i principi del GDPR, in particolare per quanto riguarda il trattamento di dati personali e sensibili.
- Sicurezza: Il sistema deve essere progettato per ridurre al minimo i rischi di accesso non autorizzato o perdita di dati.

## 4. Scheda di valutazione rischi, adempimenti e comunicazione verso l'utente
| Aspetto                  | Rischio/Obbligo                  | Misure adottate/da adottare                |
|--------------------------|-----------------------------------|--------------------------------------------|
| Trattamento dati         | Rischio di violazione privacy     | Anonimizzazione dati sensibili, crittografia, policy GDPR |
| Trasparenza              | Obbligo di informare l'utente     | Messaggio chiaro all'utente sull'uso di AI |
| Sicurezza                | Rischio di accesso non autorizzato| Autenticazione, logging, controllo accessi |
| Accuratezza output       | Rischio di errori nell'analisi    | Validazione manuale in caso di dubbi       |
| Conservazione dati       | Rischio di retention eccessiva    | Policy di cancellazione periodica          |

## 5. Comunicazione verso l'utente
L'utente deve essere informato con un messaggio del tipo:
> "Questa risposta è stata generata da un sistema di intelligenza artificiale che analizza i dati tramite modelli NER locali e GPT-4 cloud. I tuoi dati sono trattati nel rispetto della normativa GDPR. Per dubbi o richieste di rettifica, contatta l'amministratore."

---
**Documento analizzato:** bank_email.txt
