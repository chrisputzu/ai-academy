# Il codice sporco da ripulire

global_value = 0 # variabile globale usata a caso

def data(x, y, z):
    temp = []
    # codice morto
    # print("Questo era un vecchio debug")

    for i in range(len(x)):
        if x[i] > 10:
            temp.append(x[i] + y[i] + z)
        else:
            if y[i] > 5:
                temp.append(x[i] * 2 + y[i])
            else:
                temp.append(0)
    for i in range(len(x)):
        if x[i] > 10:
            temp.append(x[i] + y[i] + z)
        else:
            if y[i] > 5:
                temp.append(x[i] * 2 + y[i])
            else:
                temp.append(0)

    for i in range(len(x)):
        if temp[i] > 50:
            print("Grande valore trovato!")
        else:
            if temp[i] == 0:
                print("Zero trovato!")
            else:
                print("Valore: ", temp[i])

    useless = 5  # parametro inutilizzato
    return temp

def foo(a, b, c, d):
    if a > 0:
        if b > 0:
            if c > 0:
                print("Tutti positivi")
    return a+b+c+d

def funzione_che_esamina_i_numeri(lista):
    total = 0
    for el in lista:
        if el > 0:
            total += el
        else:
            total -= el
    for el in lista:
        if el % 2 == 0:
            print("Pari:", el)
        else:
            print("Dispari:", el)
    for el in lista:
        print("Elemento:", el)
    for el in lista:
        if el > 100:
            print("Molto grande:", el)
    # codice morto
    # return 123
    result = total
    return result

# codice inutilizzato
# def mai_usata():
#     pass

print("Fine programma")