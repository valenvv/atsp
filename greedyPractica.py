import numpy as np

UNDEF = float('inf')

def elegir_nodo(visitado, G, solucion, mas_cercano=True):
    cost = UNDEF
    w = -1
    for u in solucion:
        for v in range(len(G)):
            if not visitado[v] and ((mas_cercano and G[u][v] < cost) or (not mas_cercano and G[u][v] > cost) or cost == UNDEF):
                cost = G[u][v]
                w = v
    return w

def insertar_nodo(v, G, solucion):
    min_u = solucion[-1]
    min_w = solucion[0]
    min_diff = G[min_u][v] + G[v][min_w] - G[min_u][min_w]
    for i in range(len(solucion) - 1):
        u = solucion[i]
        w = solucion[i + 1]
        diff = G[u][v] + G[v][w] - G[u][w]
        if diff < min_diff:
            min_diff = diff
            min_u = u
            min_w = w
    return min_u, min_w

def tsp_insercion(G, mas_cercano=True):
    visitado = [False] * len(G)
    solucion = [0, 1, 2]
    visitado[0] = visitado[1] = visitado[2] = True
    cost = G[0][1] + G[1][2] + G[2][0]

    while len(solucion) < len(G):
        min_v = elegir_nodo(visitado, G, solucion, mas_cercano)
        z, w = insertar_nodo(min_v, G, solucion)
        cost += G[z][min_v] + G[min_v][w] - G[z][w]
        visitado[min_v] = True
        if w == solucion[0]:
            solucion.append(min_v)
        else:
            solucion.insert(solucion.index(w), min_v)
    return cost, solucion

# Ejemplo de uso
if __name__ == "__main__":
    tsp = [[UNDEF, 10, 15, 20],
           [5, UNDEF, 9, 10],
           [6, 13, UNDEF, 12],
           [8, 8, 9, UNDEF]]

    costo, ruta = tsp_insercion(tsp)
    print("El costo mínimo es:", costo)
    print("La ruta es:", [x + 1 for x in ruta])
