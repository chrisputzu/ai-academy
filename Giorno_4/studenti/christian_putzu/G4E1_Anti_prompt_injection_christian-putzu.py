
def valida_prompt(prompt):
     # 0. Scrivi prompt base per il controllo
    """
    Convalida un prompt prima di spedirlo a un modello LLM.

    Controlli implementati:
    0. Descrizione del Prompt base per il controllo.
    1. Black-list di parole/frasi vietate case-insensitive.
    2. Limite di lunghezza in numero di caratteri.
    3. Esempio di controllo struttura: il prompt deve terminare con un punto di domanda.
    Restituisce True se tutti i controlli sono superati, altrimenti solleva ValueError.
    """
    # 1. Lista di parole/frasi da bloccare
    blacklist = [
        "ignora istruzioni",
        "resetta ruolo",
        "password",
        "secret token",
        "DROP TABLE",
        "rm-rf",
        "sudo",
        "admin"
    ]
    
    # 2. Controllo presenza parole vietate
    prompt_lower = prompt.lower()
    for parola in blacklist:
        # COMPLETA: controlla se la parola è presente nel prompt (case-insensitive)
        if parola.lower() in prompt_lower:
            raise ValueError(f"Prompt bloccato: contiene '{parola}'")
    
    # 3. (FACOLTATIVO) Limite sulla lunghezza del prompt
    max_length = 400  # es: massimo 400 caratteri
    # COMPLETA: controlla se il prompt è troppo lungo
    if len(prompt) > max_length:
        raise ValueError("Prompt troppo lungo")
    
    # 4. (FACOLTATIVO) Altri controlli (struttura, presenza variabili non consentite, ecc.)
    # AGGIUNGI ALTRI CONTROLLI QUI

    # Controllo presenza di variabili vietate
    variabili_vietate = [
        "${password}",
        "{{token}}",
        "{{user_id}}",
        "${secret_key}",
        "<script>",
    ]
    for var in variabili_vietate:
        if var.lower() in prompt_lower:
            raise ValueError(f"Prompt bloccato: contiene variabile vietata «{var}»")

    return True

# Esempio d’uso (DA COMPLETARE NEI PUNTI CON '...')
prompt_utente = input("Inserisci il prompt da controllare: ")
try:
    if valida_prompt(prompt_utente):
        print("Prompt accettato. Procedo con l’invio al modello.")
except ValueError as e:
    print("Errore:", e)
