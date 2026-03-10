'''
**1.**-  Crea una lista con datos numéricos aleatorios entre 0 y 99. Haz un menú usando while que muestre las siguientes opciones:

1. Añade elemento a la lista.
2. Elimina elemento de la lista.
3. Muestra la media de los datos.
4. Muestra la mediana de los datos.
5. Muestra el elemento máximo (debes recorrer la lista y comparar los elementos en vez de usar una función o método).
6. Salir.
'''


import random
import numpy as np

lista = []
continuar = True

for i in range(5):
    numero = random.randint(0, 99)
    lista.append(numero)



while continuar:
    print("1. Anade un elemento")
    print("2. Elimina elemento")
    print("3. Media")
    print("4. Mediana")
    print("5. Máximo")
    print("6. Salir")
    #opcion = input("Elige una opcion: ")
    try:
        opcion = int(input("elija una opcion"))
    except ValueError:
        print("no es un numero valido")

    if opcion == "1":
        nuevo = int(input("introduce el numero a anadir"))
        lista.append(nuevo)
        print("lista actual", lista)
        break

    elif opcion == "2":
        print(lista)
        borrar = int(input("Numero a borrar"))
        lista.remove(borrar)
        print(lista)
        break


    elif opcion == "3":
        print(lista)
        media = np.mean(lista)
        print(media)
        break

    elif opcion == "4":
        print(lista)
        mediana = np.median(lista)
        print(mediana)
        break

    elif opcion == "5":
        print (lista)
        maximo = 0
        for numero in lista:
            if numero>maximo:
                maximo=numero
        print("El valor máximo es " + str(maximo))
        break

    elif opcion == "6":
        continuar = False