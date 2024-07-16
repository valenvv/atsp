#esto me tiro chat
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
import pprint
import copy
import numpy as np



# def aggregate_input_data(folder_paths):
#     distance_matrices = []
#     dimensions = []

#     for folder_path in folder_paths:
#         with open(folder_path, 'r') as file:
#             file_data = file.read().strip()
#             lines = file_data.split("\n")
            
#             # Find dimension
#             dimension = None
#             for line in lines:
#                 if line.startswith("DIMENSION:"):
#                     dimension = int(line.split(":")[1].strip())
#                     dimensions.append(dimension)
#                     break
            
#             if dimension is None:
#                 raise ValueError("Dimension not found in file: {}".format(folder_path))
            
#             # Find distance matrix
#             found_edge_weight_section = False
#             distance_matrix = []
#             for line in lines:
#                 if found_edge_weight_section:
#                     if line.strip() == "EOF":
#                         break
#                     row = list(map(int, line.split()))
#                     distance_matrix.append(row)
#                 elif line.strip() == "EDGE_WEIGHT_SECTION":
#                     found_edge_weight_section = True
            
#             if not found_edge_weight_section:
#                 raise ValueError("EDGE_WEIGHT_SECTION not found in file: {}".format(folder_path))
            
#             distance_matrices.append(distance_matrix)
    
#     return distance_matrices, dimensions

# def nearest_neighbor(distance_matrix, dimension):
#     unvisited = set(range(dimension))
#     tour = [0]
#     unvisited.remove(0)

#     while unvisited:
#         last = tour[-1]
#         next_city = min(unvisited, key=lambda city: distance_matrix[last][city])
#         tour.append(next_city)
#         unvisited.remove(next_city)

#     return tour

# def calculate_tour_cost(tour, distance_matrix):
#     cost = 0
#     for i in range(len(tour) - 1):
#         cost += distance_matrix[tour[i]][tour[i + 1]]
#     cost += distance_matrix[tour[-1]][tour[0]]  # returning to the start
#     return cost

# def main(folder_paths):
#     distance_matrices, dimensions = aggregate_input_data(folder_paths)
    
#     all_tours = []
#     all_costs = []
    
#     for i in range(len(folder_paths)):
#         distance_matrix = distance_matrices[i]
#         dimension = dimensions[i]
        
#         tour = nearest_neighbor(distance_matrix, dimension)
#         cost = calculate_tour_cost(tour, distance_matrix)
        
#         all_tours.append(tour)
#         all_costs.append(cost)
        
#         print(f"Tour for {folder_paths[i]}:", tour)
#         print(f"Cost for {folder_paths[i]}:", cost)
#         print()
    
#     return all_tours, all_costs

# if __name__ == '__main__':
#     # Example folder paths containing the data files
#     folder_paths = [
#         'br17.atsp',
#         'ft53.atsp',
#         'ft70.atsp'
#     ]
#     jol= aggregate_input_data( folder_paths)
#     print(jol)
######################## purebas############################
############### EJERCICIO 1 #####################################
'''Algoritmo goloso 1'''
def vecino_mas_cercano(G, nodo_inicial):
    # Initialize variables
    n = len(G.nodes)
    noVisitado = set(G.nodes)
    tour = [nodo_inicial]
    current_node = nodo_inicial
    noVisitado.remove(current_node)
    
    while noVisitado:
        costo_minimo = float('inf')
        next_node = None
        
        # Find the neighbor with the minimum edge cost (forward or backward)
        for neighbor in noVisitado:
            costo_adelante = G[current_node][neighbor]['weight']
            costo_atras = G[neighbor][current_node]['weight']
            cost = min(costo_adelante, costo_atras)
            
            if cost < costo_minimo:
                costo_minimo = cost
                next_node = neighbor
        
        # Move to the next node
        tour.append(next_node)
        current_node = next_node
        noVisitado.remove(current_node)
    
    # Return to the start node to complete the tour
    tour.append(nodo_inicial)
    
    return tour

'''algoritmo goloso 2''' 
import networkx as nx

UNDEF = float('inf')

def elegir_nodo(visitado, G, solucion, mas_cercano=True):
    cost = UNDEF
    w = -1
    for u in solucion:
        for v in G.nodes:
            if not visitado[v] and ((mas_cercano and G[u][v]['weight'] < cost) or (not mas_cercano and G[u][v]['weight'] > cost) or cost == UNDEF): #aca iene en cuenta la asimetria pero no entendi como
                cost = G[u][v]['weight']
                w = v
    return w

def insertar_nodo(v, G, solucion):
    min_u = solucion[-1]
    min_w = solucion[0]
    min_diff = G[min_u][v]['weight'] + G[v][min_w]['weight'] - G[min_u][min_w]['weight']
    for i in range(len(solucion) - 1):
        u = solucion[i]
        w = solucion[i + 1]
        diff = G[u][v]['weight'] + G[v][w]['weight'] - G[u][w]['weight']
        if diff < min_diff:
            min_diff = diff
            min_u = u
            min_w = w
    return min_u, min_w

def atsp_insercion(G, mas_cercano=True):
    visitado = {node: False for node in G.nodes}
    # Start with an initial cycle of three nodes
    solucion = [0, 1, 2]
    visitado[0] = visitado[1] = visitado[2] = True
    costo = G[0][1]['weight'] + G[1][2]['weight'] + G[2][0]['weight']

    while len(solucion) < len(G.nodes):
        min_v = elegir_nodo(visitado, G, solucion, mas_cercano)
        z, w = insertar_nodo(min_v, G, solucion)
        costo += G[z][min_v]['weight'] + G[min_v][w]['weight'] - G[z][w]['weight']
        visitado[min_v] = True
        if w == solucion[0]:
            solucion.append(min_v)
        else:
            solucion.insert(solucion.index(w), min_v)
    
    # Close the tour
    solucion.append(solucion[0])
    return costo, solucion

############### EJERCICIO 2 #####################################

def longitud_tour(G, tour):
    #Calcula la longitud total del camino en un grafo G
    longitud = 0
    n = len(tour)
    for i in range(n - 1):
        u = tour[i]
        v = tour[i + 1]
        longitud += G[u][v]['weight']
    return longitud


'''Busqueda local 1: 2-OPT'''
def do_2opt(tour, i, j):
    #Realiza un intercambio 2-opt en el camino
    tour[i+1:j+1] = reversed(tour[i+1:j+1])

def atsp_2opt(G, tour_inicial):
    #Aplica el operador 2-opt para ATSP en un grafo G y un camino inicial
    n = len(tour_inicial)
    tour_actual = tour_inicial[:]
    longitud_actual = longitud_tour(G, tour_actual)

    mejora = True
    while mejora:
        mejora = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Realizar el intercambio 2-opt
                nuevo_tour = tour_actual[:]
                do_2opt(nuevo_tour, i, j)
                nueva_longitud = longitud_tour(G, nuevo_tour)

                # Verificar si el nuevo camino es mejor
                if nueva_longitud < longitud_actual:
                    tour_actual = nuevo_tour[:]
                    longitud_actual = nueva_longitud
                    mejora = True
                    break
            if mejora:
                break
    
    return tour_actual, longitud_actual

'''Busqueda local 2: SWAP'''
# Función de Intercambio (Swap)
def swap(route, i, j):
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

# Función para aplicar el operador swap para ATSP en un grafo G y un tour inicial
def atsp_swap(G, tour_inicial):
    n = len(tour_inicial)
    tour_actual = tour_inicial[:]
    longitud_actual = longitud_tour(G, tour_actual)

    mejora = True
    while mejora:
        mejora = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Realizar el intercambio swap
                nuevo_tour = tour_actual[:]
                swap(nuevo_tour, i, j)
                nueva_longitud = longitud_tour(G, nuevo_tour)

                # Verificar si el nuevo tour es mejor
                if nueva_longitud < longitud_actual:
                    tour_actual = nuevo_tour[:]
                    longitud_actual = nueva_longitud
                    mejora = True
    
    return tour_actual, longitud_actual

################ EJERCICIO 3 #############################
def goloso_con_busqueda_local(G, nodo_inicial):
    # Algoritmo constructivo (vecino más cercano)
    tour_inicial = vecino_mas_cercano(G, nodo_inicial)

    # Búsqueda local (por ejemplo, 2-opt)
    tour_mejorado, longitud_mejorada = atsp_2opt(G, tour_inicial)

    return tour_mejorado, longitud_mejorada


####################### EJERCICIO EXTRA ?? ###########################################
def relocate(G, tour):
    n = len(tour)
    mejora = True
    while mejora:
        mejora = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                new_tour = tour[:i] + tour[i+1:j] + [tour[i]] + tour[j:]
                if longitud_tour(G, new_tour) < longitud_tour(G, tour):
                    tour = new_tour[:]
                    mejora = True
    return tour, longitud_tour(G, tour)

def VND(G, tour): #clase metaheuristica
    k_max = 3
    k = 0
    while k < k_max:
        if k == 0:
            new_tour, longitud_tour_new = relocate(G, tour)
        elif k == 1:
            new_tour, longitud_tour_new = atsp_swap(G, tour)
        elif k == 2:
            new_tour, longitud_tour_new = atsp_2opt(G, tour)

        if longitud_tour_new < longitud_tour(G, tour):
            tour = new_tour[:]
            k = 0
        else:
            k += 1
    return tour, longitud_tour(G, tour)


# Ejemplo de uso
def main():
    # # Create a directed graph for ATSP
    # G = nx.DiGraph()

    # # Add edges with weights (distances) - cate
    # edges = [
    #     (0, 1, 30), (0, 2, 20), (0, 3, 15),
    #     (1, 0, 10), (1, 2, 20), (1, 3, 30),
    #     (2, 0, 20), (2, 1, 25), (2, 3, 15),
    #     (3, 0, 15), (3, 1, 10), (3, 2, 10)
    # ]
    # G.add_weighted_edges_from(edges)
   # Create a directed graph for ATSP
    G = nx.DiGraph()

    # Add edges with weights (distances) - br17
    edges = [
        (0, 1, 3), (0, 2, 5), (0, 3, 48), (0, 4, 48), (0, 5, 8), (0, 6, 8), (0, 7, 5), (0, 8, 5), (0, 9, 3), (0, 10, 3), (0, 11, 0), (0, 12, 3), (0, 13, 5), (0, 14, 8), (0, 15, 8), (0, 16, 5),
        (1, 0, 5), (1, 2, 3), (1, 3, 48), (1, 4, 48), (1, 5, 8), (1, 6, 8), (1, 7, 5), (1, 8, 5), (1, 9, 0), (1, 10, 0), (1, 11, 3), (1, 12, 0), (1, 13, 3), (1, 14, 8), (1, 15, 8), (1, 16, 5),
        (2, 0, 5), (2, 1, 3), (2, 3, 72), (2, 4, 72), (2, 5, 48), (2, 6, 48), (2, 7, 24), (2, 8, 24), (2, 9, 3), (2, 10, 3), (2, 11, 5), (2, 12, 3), (2, 13, 0), (2, 14, 48), (2, 15, 48), (2, 16, 24),
        (3, 0, 48), (3, 1, 48), (3, 2, 74), (3, 4, 0), (3, 5, 6), (3, 6, 6), (3, 7, 12), (3, 8, 12), (3, 9, 48), (3, 10, 48), (3, 11, 48), (3, 12, 48), (3, 13, 74), (3, 14, 6), (3, 15, 6), (3, 16, 12),
        (4, 0, 48), (4, 1, 48), (4, 2, 74), (4, 3, 0), (4, 5, 6), (4, 6, 6), (4, 7, 12), (4, 8, 12), (4, 9, 48), (4, 10, 48), (4, 11, 48), (4, 12, 48), (4, 13, 74), (4, 14, 6), (4, 15, 6), (4, 16, 12),
        (5, 0, 8), (5, 1, 8), (5, 2, 50), (5, 3, 6), (5, 4, 6), (5, 6, 8), (5, 7, 8), (5, 8, 8), (5, 9, 8), (5, 10, 8), (5, 11, 50), (5, 12, 0), (5, 13, 0), (5, 14, 8), (5, 15, 8), (5, 16, 8),
        (6, 0, 8), (6, 1, 8), (6, 2, 50), (6, 3, 6), (6, 4, 6), (6, 5, 0), (6, 7, 8), (6, 8, 8), (6, 9, 8), (6, 10, 8), (6, 11, 50), (6, 12, 0), (6, 13, 0), (6, 14, 8), (6, 15, 8), (6, 16, 8),
        (7, 0, 5), (7, 1, 5), (7, 2, 26), (7, 3, 12), (7, 4, 12), (7, 5, 8), (7, 6, 8), (7, 8, 0), (7, 9, 5), (7, 10, 5), (7, 11, 5), (7, 12, 5), (7, 13, 26), (7, 14, 8), (7, 15, 8), (7, 16, 0),
        (8, 0, 5), (8, 1, 5), (8, 2, 26), (8, 3, 12), (8, 4, 12), (8, 5, 8), (8, 6, 8), (8, 7, 0), (8, 9, 5), (8, 10, 5), (8, 11, 5), (8, 12, 5), (8, 13, 26), (8, 14, 8), (8, 15, 8), (8, 16, 0),
        (9, 0, 3), (9, 1, 0), (9, 2, 3), (9, 3, 48), (9, 4, 48), (9, 5, 8), (9, 6, 8), (9, 7, 5), (9, 8, 5), (9, 10, 3), (9, 11, 0), (9, 12, 3), (9, 13, 8), (9, 14, 8), (9, 15, 5), (9, 16, 3),
        (10, 0, 3), (10, 1, 0), (10, 2, 3), (10, 3, 48), (10, 4, 48), (10, 5, 8), (10, 6, 8), (10, 7, 5), (10, 8, 5), (10, 9, 0), (10, 11, 3), (10, 12, 0), (10, 13, 8), (10, 14, 8), (10, 15, 5), (10, 16, 3),
        (11, 0, 0), (11, 1, 3), (11, 2, 5), (11, 3, 48), (11, 4, 48), (11, 5, 8), (11, 6, 8), (11, 7, 5), (11, 8, 5), (11, 9, 3), (11, 10, 3), (11, 12, 8), (11, 13, 8), (11, 14, 5), (11, 15, 8), (11, 16, 5),
        (12, 0, 3), (12, 1, 0), (12, 2, 3), (12, 3, 48), (12, 4, 48), (12, 5, 8), (12, 6, 8), (12, 7, 5), (12, 8, 5), (12, 9, 3), (12, 10, 0), (12, 11, 8), (12, 13, 8), (12, 14, 5), (12, 15, 8), (12, 16, 5),
        (13, 0, 5), (13, 1, 3), (13, 2, 0), (13, 3, 72), (13, 4, 72), (13, 5, 48), (13, 6, 48), (13, 7, 24), (13, 8, 24), (13, 9, 3), (13, 10, 3), (13, 11, 5), (13, 12, 3), (13, 14, 48), (13, 15, 48), (13, 16, 24),
        (14, 0, 8), (14, 1, 8), (14, 2, 50), (14, 3, 6), (14, 4, 6), (14, 5, 0), (14, 6, 0), (14, 7, 8), (14, 8, 8), (14, 9, 8), (14, 10, 8), (14, 11, 50), (14, 12, 0), (14, 13, 0), (14, 15, 8), (14, 16, 8),
        (15, 0, 8), (15, 1, 8), (15, 2, 50), (15, 3, 6), (15, 4, 6), (15, 5, 0), (15, 6, 0), (15, 7, 8), (15, 8, 8), (15, 9, 8), (15, 10, 8), (15, 11, 50), (15, 12, 0), (15, 13, 0), (15, 14, 8), (15, 16, 8),
        (16, 0, 5), (16, 1, 5), (16, 2, 26), (16, 3, 12), (16, 4, 12), (16, 5, 8), (16, 6, 8), (16, 7, 0), (16, 8, 0), (16, 9, 5), (16, 10, 5), (16, 11, 5), (16, 12, 5), (16, 13, 26), (16, 14, 8), (16, 15, 8)
    ]
    G.add_weighted_edges_from(edges)
    # Verificar G
    # print("Nodos en G:", G.nodes)
    # print("Aristas en G:", G.edges(data=True))


    # Run the modified greedy insertion algorithm
    # cost, solucion = atsp_insercion(G)
    # print("El costo mínimo es:", cost)
    # print("La ruta es:", solucion)

    # Run the modified nearest neighbor algorithm
    # tour = vecino_mas_cercano(G, 0)
    # print("Tour:", tour)

    # Camino inicial
    initial_path = [1, 0, 3, 2, 1]
    # initial_path = [0, 3, 1, 2, 0] esta es sol de goloso
    # Aplicar 2-opt para ATSP
    optimal_path, optimal_length = atsp_2opt(G, initial_path)

    print("Camino optimo:", optimal_path)
    print("Longitud minima:", optimal_length)



    optimal_path1, optimal_length1 =atsp_swap(G, initial_path)
    print("Camino optimo:", optimal_path1)
    print("Longitud minima:", optimal_length1)

    vnd= VND(G, initial_path)
    print(vnd)

if __name__ == "__main__":
    main()
    
