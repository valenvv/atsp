#Goloso sacado de google y modificado  https://www.geeksforgeeks.org/travelling-salesman-problem-greedy-approach/
#creo que sería vecino mas cercano(?)
from typing import DefaultDict
import numpy as np

BIG_NUMBER = 2147483647

# Función para encontrar la ruta de costo mínimo para todas las rutas
def greedy1(tsp):
    total_cost = 0  # Costo total del recorrido
    counter = 0  # Contador de ciudades visitadas
    j = 0  # Índice de la ciudad actual en la matriz
    i = 0  # Índice de la ciudad actual en la matriz
    min_cost = BIG_NUMBER  # Costo mínimo para moverse a la siguiente ciudad
    visitedRouteList = DefaultDict(int)  # Lista de ciudades visitadas

    # Comenzando desde la ciudad con índice 0, es decir, la primera ciudad
    visitedRouteList[0] = 1
    route = [0] * len(tsp)  # Ruta inicializada

    # Recorrer la matriz de adyacencia tsp[][]
    while counter < len(tsp) - 1:
        min_cost = BIG_NUMBER
        next_city = -1

        # Buscar la siguiente ciudad no visitada con el menor costo
        for j in range(len(tsp)):
            if tsp[i][j] != -1 and visitedRouteList[j] == 0 and tsp[i][j] < min_cost:
                min_cost = tsp[i][j]
                next_city = j

        # Verificar si se encontró una ciudad válida para continuar
        if next_city == -1:
            print("No se encontró una ruta válida")
            return

        route[counter] = next_city  # Actualizar la ruta con la siguiente ciudad
        visitedRouteList[next_city] = 1  # Marcar la siguiente ciudad como visitada
        total_cost += min_cost  # Sumar el costo mínimo al costo total
        i = next_city  # Moverse a la siguiente ciudad
        counter += 1  # Incrementar el contador de ciudades visitadas

    # Volver a la ciudad inicial
    total_cost += tsp[i][0]
    route[counter] = 0  # Añadir la ciudad inicial al final de la ruta

    # Imprimir el resultado
    print("El costo mínimo es:", total_cost)
    print("La ruta es:", [x + 1 for x in route])


#Heurística de Inserción: construye la ruta insertando la ciudad que minimiza el incremento del costo total en cada paso

# Función para encontrar la ruta de costo mínimo utilizando la heurística de inserción
def greedy2(tsp):
    n = len(tsp)
    visited = [False] * n
    route = []

    # Comenzando desde la ciudad con índice 0
    current_city = 0
    route.append(current_city)
    visited[current_city] = True

    # Insertar la segunda ciudad, la más cercana a la primera
    min_dist = float('inf')
    next_city = -1
    for j in range(1, n):
        if tsp[current_city][j] < min_dist:
            min_dist = tsp[current_city][j]
            next_city = j

    route.append(next_city)
    visited[next_city] = True

    # Construir la ruta insertando ciudades una por una
    for _ in range(n - 2):
        best_increase = float('inf')
        best_city = -1
        best_position = -1

        for city in range(n):
            if not visited[city]:
                for i in range(1, len(route)):
                    increase = tsp[route[i-1]][city] + tsp[city][route[i]] - tsp[route[i-1]][route[i]]
                    if increase < best_increase:
                        best_increase = increase
                        best_city = city
                        best_position = i

        route.insert(best_position, best_city)
        visited[best_city] = True

    # Volver a la ciudad inicial
    route.append(route[0])
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += tsp[route[i]][route[i + 1]]

    # Imprimir el resultado
    print("El costo mínimo es:", total_cost)
    print("La ruta es:", [x + 1 for x in route])


# Código principal
if __name__ == "__main__":
    # Matriz de entrada (Ejemplo)
    tsp = [[-1, 10, 15, 20],
           [5, -1, 9, 10],
           [6, 13, -1, 12],
           [8, 8, 9, -1]]

    # Llamada a la función
    greedy1(tsp)
    greedy2(tsp)
