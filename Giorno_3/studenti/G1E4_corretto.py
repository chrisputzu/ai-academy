
def calcola_valori(lista_x, lista_y, z):
    """
    Calcola una lista di valori in base a regole definite sugli elementi di lista_x e lista_y.

    Parameters
    ----------
    lista_x : list of int or float
        Lista dei primi valori di input.
    lista_y : list of int or float
        Lista dei secondi valori di input (stessa lunghezza di lista_x).
    z : int or float
        Valore costante da sommare in certe condizioni.

    Returns
    -------
    list of int or float
        Lista dei valori calcolati.
    """
    risultati = []

    for x, y in zip(lista_x, lista_y):
        if x > 10:
            risultato = x + y + z
        elif y > 5:
            risultato = x * 2 + y
        else:
            risultato = 0
        risultati.append(risultato)
    return risultati

def stampa_analisi(valori):
    """
    Analizza e stampa messaggi specifici sui valori calcolati.

    Parameters
    ----------
    valori : list of int or float
        Lista di valori su cui effettuare l’analisi.

    Returns
    -------
    None
    """
    for valore in valori:
        if valore > 50:
            print("Grande valore trovato!")
        elif valore == 0:
            print("Zero trovato!")
        else:
            print(f"Valore: {valore}")

def tutti_positivi(*args):
    """
    Verifica se tutti i valori forniti sono positivi.

    Parameters
    ----------
    *args : int or float
        Qualsiasi numero di argomenti numerici.

    Returns
    -------
    bool
        True se tutti i valori sono positivi, False altrimenti.
    """
    return all(a > 0 for a in args)

def calcola_totale(lista):
    """
    Calcola il totale di una lista sommando i positivi e sottraendo i negativi.

    Parameters
    ----------
    lista : list of int or float
        Lista dei valori.

    Returns
    -------
    int or float
        Totale calcolato.
    """
    return sum(el if el > 0 else -el for el in lista)
    

def stampa_report(lista):
    """
    Stampa varie analisi sugli elementi della lista.

    Parameters
    ----------
    lista : list of int or float
        Lista di valori.

    Returns
    -------
    None
    """

    for el in lista:
        print("Pari:" if el % 2 == 0 else "Dispari:", el)
        print("Elemento:", el)
        if el > 100:
            print("Molto grande:", el)


def main():
    # Esempio di esecuzione 
    x = [10, 20, 15, 5]
    y = [5, 12, 8, 3]
    z = 2

    valori = calcola_valori(x, y, incremento)

    print("\n--- Analisi valori ---")
    stampa_analisi(valori)

    print("\n--- Verifica positività ---")
    print("Tutti positivi?", tutti_positivi(*x, *y, incremento))

    print("\n--- Totale ---")
    totale = calcola_totale(valori)
    print("Totale:", totale)

    print("\n--- Report dettagliato ---")
    stampa_report(valori)

if __name__ == "__main__":
    main()


