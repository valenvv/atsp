import os
import json
import matplotlib.pyplot as plt
import math
import pprint
import copy
import numpy as np

def atsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        start_idx = lines.index('EDGE_WEIGHT_SECTION\n') + 1
        dimension = int(lines[3].split()[1])
        
        matrix = []
        for line in lines[start_idx:]:
            if line.strip() == 'EOF':
                break
            row = list(map(int, line.split()))
            matrix.extend(row)
        
        matrix = [matrix[i:i + dimension] for i in range(0, len(matrix), dimension)]
        
    return matrix, dimension

def tour_hamiltoniano_basico(dimension):
    tour = list(range(dimension)) + [0]
    return tour

def vecino_mas_cercano(matrix, nodo_inicial):
    n = len(matrix)
    noVisitado = set(range(n))
    tour = [nodo_inicial]
    current_node = nodo_inicial
    noVisitado.remove(current_node)
    
    while noVisitado:
        costo_minimo = float('inf')
        next_node = None
        
        for neighbor in noVisitado:
            costo_adelante = matrix[current_node][neighbor]
            costo_atras = matrix[neighbor][current_node]
            cost = min(costo_adelante, costo_atras)
            
            if cost < costo_minimo:
                costo_minimo = cost
                next_node = neighbor
        
        tour.append(next_node)
        current_node = next_node
        noVisitado.remove(current_node)
    
    tour.append(nodo_inicial)
    
    return tour

UNDEF = float('inf')

def elegir_nodo(visitado, matrix, solucion, mas_cercano=True):
    cost = UNDEF
    w = -1
    for u in solucion:
        for v in range(len(matrix)):
            if not visitado[v] and ((mas_cercano and matrix[u][v] < cost) or (not mas_cercano and matrix[u][v] > cost) or cost == UNDEF):
                cost = matrix[u][v]
                w = v
    return w

def insertar_nodo(v, matrix, solucion):
    min_u = solucion[-1]
    min_w = solucion[0]
    min_diff = matrix[min_u][v] + matrix[v][min_w] - matrix[min_u][min_w]
    for i in range(len(solucion) - 1):
        u = solucion[i]
        w = solucion[i + 1]
        diff = matrix[u][v] + matrix[v][w] - matrix[u][w]
        if diff < min_diff:
            min_diff = diff
            min_u = u
            min_w = w
    return min_u, min_w

def atsp_insercion(matrix, mas_cercano=True):
    dimension = len(matrix)
    visitado = [False] * dimension
    solucion = [0, 1, 2]
    visitado[0] = visitado[1] = visitado[2] = True
    costo = matrix[0][1] + matrix[1][2] + matrix[2][0]

    while len(solucion) < dimension:
        min_v = elegir_nodo(visitado, matrix, solucion, mas_cercano)
        z, w = insertar_nodo(min_v, matrix, solucion)
        costo += matrix[z][min_v] + matrix[min_v][w] - matrix[z][w]
        visitado[min_v] = True
        if w == solucion[0]:
            solucion.append(min_v)
        else:
            solucion.insert(solucion.index(w), min_v)
    
    solucion.append(solucion[0])
    return costo, solucion

def costo_tour(matrix, tour):
    costo = 0
    n = len(tour)
    for i in range(n - 1):
        u = tour[i]
        v = tour[i + 1]
        costo += matrix[u][v]
    return costo

def do_2opt(tour, i, j):
    tour[i+1:j+1] = reversed(tour[i+1:j+1])

def atsp_2opt(matrix, tour_inicial):
    n = len(tour_inicial)
    tour_actual = tour_inicial[:]
    mejora = True
    while mejora:
        mejora = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                old_cost = matrix[tour_actual[i-1]][tour_actual[i]] + matrix[tour_actual[j]][tour_actual[j+1]]
                new_cost = matrix[tour_actual[i-1]][tour_actual[j]] + matrix[tour_actual[i]][tour_actual[j+1]]
                if new_cost < old_cost:
                    do_2opt(tour_actual, i, j)
                    mejora = True
                    break
            if mejora:
                break
    return tour_actual, costo_tour(matrix, tour_actual)

def relocate(matrix, tour):
    n = len(tour)
    mejora = True
    while mejora:
        mejora = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                nuevo_tour = tour[:i] + tour[i+1:j] + [tour[i]] + tour[j:]
                if costo_tour(matrix, nuevo_tour) < costo_tour(matrix, tour):
                    tour = nuevo_tour[:]
                    mejora = True
    return tour, costo_tour(matrix, tour)

def goloso_con_busqueda_local(matrix, nodo_inicial, heuristica_constructiva):
    if heuristica_constructiva == "vecino_mas_cercano":
        tour_inicial = vecino_mas_cercano(matrix, nodo_inicial)
    elif heuristica_constructiva == "atsp_insercion":
        costo, tour_inicial = atsp_insercion(matrix)
    else:
        raise ValueError("Heurística constructiva no válida.")

    tour_mejorado_2opt, costo_mejorada_2opt = atsp_2opt(matrix, tour_inicial)
    print("costo ej3 con 2-opt:", costo_mejorada_2opt)

    tour_mejorado_relocate, costo_mejorada_relocate = relocate(matrix, tour_inicial)
    print("costo ej3 con relocate:", costo_mejorada_relocate)

def VND(matrix, tour):
    k_max = 2
    k = 0
    while k < k_max:
        if k == 0:
            tour_nuevo, costo_nuevo_tour = relocate(matrix, tour)
        elif k == 1:
            tour_nuevo, costo_nuevo_tour = atsp_2opt(matrix, tour)

        if costo_nuevo_tour < costo_tour(matrix, tour):
            tour = tour_nuevo[:]
            k = 0
        else:
            k += 1
    return tour, costo_tour(matrix, tour)

def main():
    filename = 'ftv170.atsp'
    matrix, dimension = atsp_file(filename)

    tour_inicial = tour_hamiltoniano_basico(dimension)
    
    cost, solucion = atsp_insercion(matrix)
    print("El costo mínimo es atsp insercion:", cost)

    tour = vecino_mas_cercano(matrix, 0)
    costo = costo_tour(matrix, tour)
    print("El costo mínimo es vecino cercano:", costo)
    
    initial_path = tour_hamiltoniano_basico(dimension)
    optimal_path, optimal_length = atsp_2opt(matrix, initial_path)
    print("Longitud minima 2opt:", optimal_length)

    cost, solucion = atsp_insercion(matrix)
    
    optimal_path2, optimal_length2 = relocate(matrix, initial_path)
    print("Longitud minima relocate:", optimal_length2)
    
    vnd, cost_vnd = VND(matrix, solucion)
    print("VND costo:", cost_vnd)
    
    goloso_con_busqueda_local(matrix, 0, "atsp_insercion")

if __name__ == "__main__":
    main()

    
