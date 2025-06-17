def calcola_valori(x, y, incremento):
    """
    Calcola una lista di valori elaborati da x e y secondo condizioni specifiche.

    Parameters
    ----------
    x : list of int
        Lista di valori di input.
    y : list of int
        Lista di valori di input.
    incremento : int
        Valore costante da sommare in certi casi.

    Returns
    -------
    list of int
        Lista di valori elaborati.
    """
    risultati = []
    for xi, yi in zip(x, y):
        if xi > 10:
            risultati.append(xi + yi + incremento)
        elif yi > 5:
            risultati.append(xi * 2 + yi)
        else:
            risultati.append(0)
    return risultati


def descrivi_valori(valori):
    """
    Restituisce una lista di messaggi descrittivi per ogni valore.

    Parameters
    ----------
    valori : list of int
        Lista di valori da descrivere.

    Returns
    -------
    list of str
        Messaggi corrispondenti ai valori.
    """
    messaggi = []
    for val in valori:
        if val > 50:
            messaggi.append("Grande valore trovato!")
        elif val == 0:
            messaggi.append("Zero trovato!")
        else:
            messaggi.append(f"Valore: {val}")
    return messaggi


def somma_pesata(lista):
    """
    Calcola una somma pesata: aggiunge i positivi, sottrae i negativi.

    Parameters
    ----------
    lista : list of int
        Lista di numeri interi.

    Returns
    -------
    int
        Somma pesata dei numeri.
    """
    return sum(el if el > 0 else -el for el in lista)


def analizza_numeri(lista):
    """
    Restituisce analisi dettagliata di una lista: parità, grandezza, e valore grezzo.

    Parameters
    ----------
    lista : list of int
        Lista di numeri da analizzare.

    Returns
    -------
    dict
        Analisi in forma di dizionario con varie categorie.
    """
    analisi = {
        "pari": [el for el in lista if el % 2 == 0],
        "dispari": [el for el in lista if el % 2 != 0],
        "molto_grandi": [el for el in lista if el > 100],
        "tutti": lista
    }
    return analisi


def somma_condizionale(a, b, c, d):
    """
    Restituisce la somma di 4 numeri e stampa un messaggio se tutti positivi.

    Parameters
    ----------
    a, b, c, d : int
        Valori da sommare.

    Returns
    -------
    int
        Somma dei valori.
    """
    if all(val > 0 for val in [a, b, c]):
        print("Tutti positivi")
    return a + b + c + d


def main():
    x = [12, 4, 15]
    y = [6, 3, 8]
    incremento = 5

    risultati = calcola_valori(x, y, incremento)
    messaggi = descrivi_valori(risultati)

    for msg in messaggi:
        print(msg)

    total = somma_pesata(risultati)
    print("Somma pesata:", total)

    analisi = analizza_numeri(risultati)
    print("Numeri pari:", analisi["pari"])
    print("Numeri dispari:", analisi["dispari"])
    print("Numeri molto grandi:", analisi["molto_grandi"])

    print("Fine programma")


if __name__ == "__main__":
    main()
