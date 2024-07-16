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
    # Create a directed graph for ATSP
    G = nx.DiGraph()

    # Add edges with weights (distances)
    edges = [
        (0, 1, 30), (0, 2, 20), (0, 3, 15),
        (1, 0, 10), (1, 2, 20), (1, 3, 30),
        (2, 0, 20), (2, 1, 25), (2, 3, 15),
        (3, 0, 15), (3, 1, 10), (3, 2, 10)
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
    
