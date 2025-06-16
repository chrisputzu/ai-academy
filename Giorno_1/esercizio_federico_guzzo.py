def calcola_valori(x, y, z):
    """
    Calcola una lista di valori in base a regole sui parametri x, y e costante z.

    Parametri
    ----------
    x : lista di int o float
        Prima lista di input.
    y : lista di int o float
        Seconda lista di input.
    z : int o float
        Costante da sommare nei calcoli.

    Ritorna
    -------
    list
        Lista dei valori calcolati.
    """
    risultati = []
    for xi, yi in zip(x, y):
        if xi > 10:
            risultati.append(xi + yi + z)
        elif yi > 5:
            risultati.append(xi * 2 + yi)
        else:
            risultati.append(0)
    return risultati


def analizza_risultati(valori):
    """
    Analizza i valori calcolati e restituisce messaggi interpretativi.

    Parametri
    ----------
    valori : lista di int o float
        Valori prodotti da calcola_valori().

    Ritorna
    -------
    list di str
        Messaggi associati a ciascun valore.
    """
    messaggi = []
    for valore in valori:
        if valore > 50:
            messaggi.append("Grande valore trovato!")
        elif valore == 0:
            messaggi.append("Zero trovato!")
        else:
            messaggi.append(f"Valore: {valore}")
    return messaggi


def riepilogo_lista(numeri):
    """
    Calcola il totale e classifica i numeri come pari/dispari e grandi/piccoli.

    Parametri
    ----------
    numeri : lista di int
        Lista di numeri da analizzare.

    Ritorna
    -------
    int
        Totale calcolato.
    list di str
        Informazioni sulla parità di ciascun numero.
    list di str
        Informazioni sulla grandezza di ciascun numero.
    """
    totale = 0
    parita = []
    dimensione = []

    for numero in numeri:
        totale += numero if numero > 0 else -numero
        parita.append("Pari" if numero % 2 == 0 else "Dispari")
        dimensione.append("Molto grande" if numero > 100 else "Normale")

    return totale, parita, dimensione


def stampa_riepilogo(numeri):
    """
    Stampa un riepilogo completo della lista di numeri fornita.

    Parametri
    ----------
    numeri : lista di int
        Lista da analizzare.
    """
    totale, info_parita, info_dimensione = riepilogo_lista(numeri)
    print("Totale calcolato:", totale)
    for numero, parita, dimensione in zip(numeri, info_parita, info_dimensione):
        print(f"Elemento: {numero} - {parita} - {dimensione}")


def stampa_positività(a, b, c, d):
    """
    Stampa un messaggio se i primi tre parametri sono positivi.

    Parametri
    ----------
    a, b, c, d : int o float
        Numeri da verificare.

    Ritorna
    -------
    int o float
        Somma di tutti i parametri.
    """
    if all(val > 0 for val in [a, b, c]):
        print("Tutti positivi")
    return a + b + c + d


def main():
    """
    Funzione principale che esegue il flusso del programma.
    """
    x = [12, 3, 15, 7]
    y = [2, 8, 1, 6]
    z = 5

    risultati = calcola_valori(x, y, z)
    messaggi = analizza_risultati(risultati)

    print("\nRisultati analizzati:")
    for msg in messaggi:
        print(msg)

    print("\nAnalisi della lista x:")
    stampa_riepilogo(x)

    print("\nVerifica della positività:")
    somma = stampa_positività(1, 2, 3, 4)
    print("Somma:", somma)

    print("\nFine programma")


if __name__ == "__main__":
    main()
