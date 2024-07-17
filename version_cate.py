import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time

# Función para extraer la matriz de pesos de un archivo ATSP y construir un grafo dirigido con NetworkX
def atsp_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Buscar la sección de EDGE_WEIGHT_SECTION
        start_idx = lines.index('EDGE_WEIGHT_SECTION\n') + 1
        dimension = int(lines[3].split()[1])
        
        # Leer la matriz de pesos
        matrix = []
        for line in lines[start_idx:]:
            if line.strip() == 'EOF':
                break
            row = list(map(int, line.split()))
            matrix.extend(row)
        
        # Convertir la lista en una matriz
        matrix = [matrix[i:i + dimension] for i in range(0, len(matrix), dimension)]
        
    return matrix, dimension

def create_graph_from_matrix(matrix, dimension):
    G = nx.DiGraph()
    
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                G.add_edge(i, j, weight=matrix[i][j])
    
    return G

def tour_hamiltoniano_basico(G):
    nodos = list(G.nodes)
    nodos.sort() 
    
    # Construir circuito hamiltoniano
    tour = nodos + [nodos[0]]  # Agregar el primer nodo al final para cerrar el circuito
    
    return tour


##################################### EJERCICIO 1 #####################################
'''Algoritmo goloso 1'''
def vecino_mas_cercano(G, nodo_inicial): # O(n^2)
    n = len(G.nodes) 
    noVisitado = set(G.nodes)
    tour = [nodo_inicial]
    current_node = nodo_inicial
    noVisitado.remove(current_node)
    
    while noVisitado:
        costo_minimo = float('inf')
        next_node = None
        
        # Encontrar el vecino con el menor costo de arista (ya sea hacia adelante o hacia atrás)
        for neighbor in noVisitado:
            costo_adelante = G[current_node][neighbor]['weight']
            costo_atras = G[neighbor][current_node]['weight']
            cost = min(costo_adelante, costo_atras)
            
            if cost < costo_minimo:
                costo_minimo = cost
                next_node = neighbor

        tour.append(next_node)
        current_node = next_node
        noVisitado.remove(current_node)
        
    tour.append(nodo_inicial)
    
    return tour

'''algoritmo goloso 2''' 
UNDEF = float('inf')

def elegir_nodo(visitado, G, solucion, mas_cercano=True): # O(n^2)
    cost = UNDEF
    w = -1
    for u in solucion:
        for v in G.nodes:
            # Considera nodos no visitados y decide basado en la heurística de cercanía o lejanía
            if not visitado[v] and ((mas_cercano and G[u][v]['weight'] < cost) or (not mas_cercano and G[u][v]['weight'] > cost) or cost == UNDEF):
                cost = G[u][v]['weight']
                w = v
    return w

def insertar_nodo(v, G, solucion): # O(n)
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

def atsp_insercion(G, mas_cercano=True): # O(n^3)
    visitado = {node: False for node in G.nodes}
    # Comenzar con un ciclo inicial de tres nodos
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
    
    solucion.append(solucion[0])
    return costo, solucion

##################################### EJERCICIO 2 #####################################

def costo_tour(G, tour): # O(n)
    costo = 0
    n = len(tour)
    for i in range(n - 1):
        u = tour[i]
        v = tour[i + 1]
        costo += G[u][v]['weight']
    return costo


'''Busqueda local 1: 2-OPT'''
def do_2opt(tour, i, j): # O(n)
    #Realiza un intercambio 2-opt
    tour[i+1:j+1] = reversed(tour[i+1:j+1])

def atsp_2opt(G, tour_inicial): # O(n^4)
    #Aplica el operador 2-opt para ATSP en un grafo G y un tour inicial
    n = len(tour_inicial) 
    tour_actual = tour_inicial.copy()
    costo_actual = costo_tour(G, tour_actual)

    mejora = True
    while mejora:
        mejora = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Realizar el intercambio 2-opt
                nuevo_tour = tour_actual.copy()
                do_2opt(nuevo_tour, i, j)
                nuevo_costo= costo_tour(G, nuevo_tour)

                # Verificar si el nuevo camino es mejor
                if nuevo_costo < costo_actual:
                    tour_actual = nuevo_tour.copy()
                    costo_actual = nuevo_costo
                    mejora = True
                    break
            if mejora:
                break
    
    return tour_actual, costo_actual

'''Busqueda local 2: RELOCATE'''
def relocate(G, tour): # O(n^4)
    n = len(tour)
    mejora = True
    while mejora:
        mejora = False
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                # Intenta reubicar el nodo en la posición i en la posición j del tour
                nuevo_tour = tour[:i] + tour[i+1:j] + [tour[i]] + tour[j:]
                if costo_tour(G, nuevo_tour) < costo_tour(G, tour):
                    tour = nuevo_tour[:]
                    mejora = True
    return tour, costo_tour(G, tour)


##################################### EJERCICIO 3 #####################################

def goloso_con_busqueda_local_2opt(G, nodo_inicial, heuristica_constructiva): # O(n^4)
    if heuristica_constructiva == "vecino_mas_cercano": 
        inicio = time.time()
        tour_inicial = vecino_mas_cercano(G, nodo_inicial)  # O(n^2)
    elif heuristica_constructiva == "atsp_insercion":
        inicio = time.time()
        costo, tour_inicial = atsp_insercion(G)  # O(n^3)
    else:
        raise ValueError("Heurística constructiva no válida.")

    # Búsqueda local 2-opt
    tour_mejorado_2opt, costo_mejorada_2opt = atsp_2opt(G, tour_inicial)  # O(n^4)
    print("costo ej3 con 2-opt:", costo_mejorada_2opt)
    print("Tiempo 2opt:", time.time() - inicio, "segundos")
    
   
def goloso_con_busqueda_local_relocate(G, nodo_inicial, heuristica_constructiva): # O(n^4)
    
    if heuristica_constructiva == "vecino_mas_cercano": 
        inicio = time.time()
        tour_inicial = vecino_mas_cercano(G, nodo_inicial)  # O(n^2)
    elif heuristica_constructiva == "atsp_insercion":
        inicio = time.time()
        costo, tour_inicial = atsp_insercion(G)  # O(n^3)
    else:
        raise ValueError("Heurística constructiva no válida.")


    # Búsqueda local relocate
    tour_mejorado_relocate, costo_mejorada_relocate = relocate(G, tour_inicial)  # O(n^4)
    print("costo ej3 con relocate:", costo_mejorada_relocate)
    print("Tiempo ej3 relocate:", time.time() -inicio, "segundos")
   
    


####################### Para la experimentacion ###########################################

def VND(G, tour): # O(n^4)
    k_max = 2
    k = 0
    while k < k_max:
        if k == 0:
            tour_nuevo, costo_nuevo_tour = relocate(G, tour) 
        elif k == 1:
            tour_nuevo, costo_nuevo_tour = atsp_2opt(G, tour)

        if costo_nuevo_tour < costo_tour(G, tour):
            tour = tour_nuevo[:]
            k = 0
        else:
            k += 1
    return tour, costo_tour(G, tour)

############## ejercicio extra #######################
''' Reduccion de ATSP a TSP'''
def atsp_to_tsp(G):
    # Crear un nuevo grafo para TSP
    G_tsp = nx.Graph()

    # Para cada nodo en el grafo ATSP, crea dos nodos en el grafo TSP
    for node in G.nodes:
        G_tsp.add_node(f"{node}_in")
        G_tsp.add_node(f"{node}_out")
        G_tsp.add_edge(f"{node}_in", f"{node}_out", weight=0)

    # Para cada arista en el grafo ATSP, agrega dos aristas en el grafo TSP
    for u, v, data in G.edges(data=True):
        w = data['weight']
        G_tsp.add_edge(f"{u}_out", f"{v}_in", weight=w)

    return G_tsp


def visualize_graphs(G_atsp, G_tsp):
    plt.figure(figsize=(12, 5))

    # Visualización del grafo ATSP
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G_atsp)
    labels = nx.get_edge_attributes(G_atsp, 'weight')
    nx.draw(G_atsp, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G_atsp, pos, edge_labels=labels)
    plt.title('Grafo ATSP')

    # Visualización del grafo TSP transformado
    plt.subplot(1, 2, 2)
    pos_tsp = nx.spring_layout(G_tsp)
    labels_tsp = nx.get_edge_attributes(G_tsp, 'weight')
    nx.draw(G_tsp, pos_tsp, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G_tsp, pos_tsp, edge_labels=labels_tsp)
    plt.title('Grafo TSP Transformado')

    plt.tight_layout()
    plt.show()


def main():

    # Elejir instancia:
    
    # filename = 'br17.atsp'
    # filename = 'ft53.atsp'
    # filename = 'ft70.atsp'
    # filename = 'ftv33.atsp'  
    # filename = 'ftv35.atsp'
    # filename = 'ftv38.atsp'
    # filename = 'ftv44.atsp'  
    # filename = 'ftv47.atsp'
    # filename = 'ftv55.atsp'
    # filename = 'ftv64.atsp'
    # filename = 'ftv70.atsp'
    # filename = 'ftv170.atsp'  
    # filename = 'kro124p.atsp' 
    # filename = 'p43.atsp'      
    # filename = 'rbg323.atsp'
    # filename = 'rbg358.atsp'
    # filename = 'rbg403.atsp'
    # filename = 'rbg443.atsp'
    # filename = 'ry48p.atsp' 
    filename ='ejerextra.txt'
    
    matrix, dimension = atsp_file(filename)
    G = create_graph_from_matrix(matrix, dimension)
    tour_inicial = tour_hamiltoniano_basico(G)

    # Medir tiempo para Inserción 
    inicio = time.time()
    cost, solucion = atsp_insercion(G)
    print("El costo mínimo es atsp insercion:", cost)
    print("Tiempo de ejecución de atsp insercion:", time.time() - inicio, "segundos")

    # Medir tiempo para veciono mas cercano
    inicio = time.time()
    tour = vecino_mas_cercano(G, 0)
    costo = costo_tour(G, tour)
    print("El costo mínimo es vecino cercano:", costo)
    print("Tiempo de ejecución de vecino cercano:", time.time() - inicio, "segundos")


    # Medir tiempo para VND - Inserción
    inicio = time.time()
    vnd, cost_vnd = VND(G, solucion)
    print("VND costo con Inserción:", cost_vnd)
    print("Tiempo de ejecución de VND Inserción:", time.time() - inicio, "segundos")

    # Medir tiempo para VND - vecino mas cercano
    solucion2 = vecino_mas_cercano(G,0)
    inicio = time.time()
    vnd, cost_vnd = VND(G, solucion2)
    print("VND costo con vecino:", cost_vnd)
    print("Tiempo de ejecución de VND con vecino:", time.time() - inicio, "segundos")

    # Medir tiempo para goloso_con_busqueda_local Inserción
    print("Tiempo de ejecución de goloso con búsqueda local Inserción:")
    goloso_con_busqueda_local_2opt(G, 0, "atsp_insercion")
    goloso_con_busqueda_local_relocate(G, 0, "atsp_insercion")
    

    # Medir tiempo para goloso_con_busqueda_local vecino mas cercano
    print("Tiempo de ejecución de goloso con búsqueda local VC:")
    goloso_con_busqueda_local_2opt(G, 0, "vecino_mas_cercano")
    goloso_con_busqueda_local_relocate(G, 0, "vecino_mas_cercano")


    '''
    ### EJER extra ###
    # Transformar grafo ATSP a grafo TSP
    G_tsp = atsp_to_tsp(G)
    visualize_graphs(G, G_tsp)
    '''
    

if __name__ == "__main__":
    main()