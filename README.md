# ATSP Solver

Este proyecto implementa diversas heurísticas y algoritmos para resolver el problema del **Asymmetric Traveling Salesman Problem (ATSP)**. Incluye algoritmos constructivos, búsquedas locales, y técnicas híbridas como VND. También incorpora herramientas para transformar ATSP a TSP y visualizar grafos.

## Contenido del Proyecto

### Principales Funciones y Algoritmos

#### Lectura y construcción de grafos:
- `atsp_file(filename)`: Extrae la matriz de costos desde un archivo `.atsp`.
- `create_graph_from_matrix(matrix, dimension)`: Construye un grafo dirigido usando NetworkX.

#### Heurísticas constructivas:
- `vecino_mas_cercano(G, nodo_inicial)`: Algoritmo goloso basado en el vecino más cercano.
- `atsp_insercion(G, mas_cercano=True)`: Inserción secuencial basada en heurísticas de cercanía o lejanía.

#### Búsquedas locales:
- `atsp_2opt(G, tour_inicial)`: Mejora un tour mediante el operador 2-opt.
- `relocate(G, tour)`: Realiza reubicaciones de nodos para optimizar el tour.

#### Métodos híbridos:
- `VND(G, tour)`: Variable Neighborhood Descent con operadores `relocate` y `2-opt`.
- `goloso_con_busqueda_local_*`: Combina heurísticas golosas con búsquedas locales.

#### Transformación de ATSP a TSP:
- `atsp_to_tsp(G)`: Transforma un grafo ATSP a un grafo TSP equivalente.
- `visualizar_grafo(G_atsp, G_tsp)`: Visualiza el grafo original y el transformado.

---

## Experimentos y Comparaciones

El programa principal realiza experimentos para medir el rendimiento de los algoritmos en términos de costo y tiempo. Se incluyen:

- Comparación de heurísticas golosas.
- Evaluación del impacto de búsquedas locales (2-opt y relocate).
- Aplicación de VND con distintos métodos iniciales.
