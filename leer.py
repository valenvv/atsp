#esto me tiro chat
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
import pprint
import copy
import numpy as np


import gzip
import shutil



# Leer el archivo descomprimido
def read_tsplib_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    # Procesar las líneas para obtener los datos del ATSP
    node_count = int(lines[3].split(':')[1].strip())
    weights = []
    start_index = lines.index('EDGE_WEIGHT_SECTION\n') + 1
    for line in lines[start_index:start_index + node_count]:
        weights.append(list(map(int, line.split())))
    return weights

# Usar la función para leer datos y construir el dígrafo
weights = read_tsplib_file('br171')
G = nx.DiGraph()
for i in range(len(weights)):
    for j in range(len(weights[i])):
        if i != j:
            G.add_edge(i, j, weight=weights[i][j])

# Añadir esto después de construir G
print("Grafo construido:", G.nodes, G.edges(data=True))


def vecino_mas_cercano(G, nodo_inicial):
    n = len(G.nodes)
    noVisitado = set(G.nodes)
    tour = [nodo_inicial]
    current_node = nodo_inicial
    noVisitado.remove(current_node)
    
    while noVisitado:
        costo_minimo = float('inf')
        next_node = None
        
        for neighbor in noVisitado:
            if neighbor in G[current_node]:
                costo_adelante = G[current_node][neighbor]['weight']
                costo_atras = G[neighbor][current_node]['weight']
                cost = min(costo_adelante, costo_atras)
                
                if cost < costo_minimo:
                    costo_minimo = cost
                    next_node = neighbor
            else:
                print(f"El nodo {neighbor} no es accesible desde el nodo {current_node}")
        
        if next_node is None:
            break
        
        tour.append(next_node)
        current_node = next_node
        noVisitado.remove(current_node)
    
    tour.append(nodo_inicial)  # Asegura que se complete el tour
    
    return tour





print(vecino_mas_cercano(G,0))
