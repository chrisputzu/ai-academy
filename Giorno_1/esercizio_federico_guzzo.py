
X_THRESHOLD = 10
Y_THRESHOLD = 5
LARGE_VALUE_THRESHOLD = 50

def calcola_valori_processati(lista_x, lista_y, valore_z):

    risultati = []
    # Un solo ciclo invece di due blocchi identici.
    # La logica if/else annidata è stata semplificata con 'elif'.
    for i in range(len(lista_x)):
        if lista_x[i] > X_THRESHOLD:
            risultati.append(lista_x[i] + lista_y[i] + valore_z)
        elif lista_y[i] > Y_THRESHOLD:
            risultati.append(lista_x[i] * 2 + lista_y[i])
        else:
            risultati.append(0)
    return risultati

def stampa_analisi_valori(lista_valori):
    """
    Stampa un'analisi dei valori di una lista.

    Questa funzione separa la logica di stampa da quella di calcolo,
    seguendo il principio di singola responsabilità.
    """
    for valore in lista_valori:
        if valore > LARGE_VALUE_THRESHOLD:
            print("Grande valore trovato!")
        elif valore == 0:
            print("Zero trovato!")
        else:
            print(f"Valore: {valore}") # Utilizzo di f-string per una formattazione più moderna.

def somma_e_controlla_positivi(a, b, c, d):
    """
    Somma quattro numeri e stampa un messaggio se i primi tre sono positivi.

    Sostituisce la vecchia funzione 'foo'.
    - Ha un nome più descrittivo.
    - La condizione 'if' annidata è stata appiattita per maggiore chiarezza.
    """
    if a > 0 and b > 0 and c > 0:
        print("I primi tre valori sono tutti positivi.")
    return a + b + c + d

def analizza_numeri(lista_di_numeri):
    """
    Analizza una lista di numeri, calcola la somma dei loro valori assoluti
    e stampa varie informazioni su ciascun numero.

    Sostituisce 'funzione_che_esamina_i_numeri'.
    - Nome più conciso e in stile snake_case.
    - Tutte le operazioni sono state consolidate in un unico ciclo per efficienza.
    - La logica della somma è stata chiarita usando abs().
    """
    somma_assoluta = 0
    print("\n--- Analisi della lista di numeri ---")
    for numero in lista_di_numeri:
        # 1. Calcolo della somma (logica originale: somma dei valori assoluti)
        somma_assoluta += abs(numero)

        # 2. Stampa dell'elemento
        print(f"Elemento: {numero}")

        # 3. Controllo parità
        if numero % 2 == 0:
            print(f"  -> Pari")
        else:
            print(f"  -> Dispari")
        
        # 4. Controllo grandezza
        if numero > 100:
            print(f"  -> Molto grande!")

    # La variabile 'result' intermedia è stata rimossa, la funzione ritorna direttamente il valore.
    return somma_assoluta

# Blocco di esecuzione principale: una buona pratica per rendere il codice riutilizzabile e organizzato.
if __name__ == "__main__":
    # Esempio di utilizzo per la prima funzione
    x_vals = [5, 12, 8]
    y_vals = [3, 6, 10]
    z_val = 20
    
    valori_calcolati = calcola_valori_processati(x_vals, y_vals, z_val)
    stampa_analisi_valori(valori_calcolati)

    # Esempio di utilizzo per la seconda funzione
    numeri_da_analizzare = [10, -5, 150, 23]
    somma_risultante = analizza_numeri(numeri_da_analizzare)
    print(f"\nLa somma dei valori assoluti è: {somma_risultante}")

    print("\nFine programma")