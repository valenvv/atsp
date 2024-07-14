#esto me tiro chat
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import math
import pprint
import copy

def aggregate_input_data(folder_paths):
    distance_matrices = []
    dimensions = []

    for folder_path in folder_paths:
        with open(folder_path, 'r') as file:
            file_data = file.read().strip()
            lines = file_data.split("\n")
            
            # Find dimension
            dimension = None
            for line in lines:
                if line.startswith("DIMENSION:"):
                    dimension = int(line.split(":")[1].strip())
                    dimensions.append(dimension)
                    break
            
            if dimension is None:
                raise ValueError("Dimension not found in file: {}".format(folder_path))
            
            # Find distance matrix
            found_edge_weight_section = False
            distance_matrix = []
            for line in lines:
                if found_edge_weight_section:
                    if line.strip() == "EOF":
                        break
                    row = list(map(int, line.split()))
                    distance_matrix.append(row)
                elif line.strip() == "EDGE_WEIGHT_SECTION":
                    found_edge_weight_section = True
            
            if not found_edge_weight_section:
                raise ValueError("EDGE_WEIGHT_SECTION not found in file: {}".format(folder_path))
            
            distance_matrices.append(distance_matrix)
    
    return distance_matrices, dimensions

def nearest_neighbor(distance_matrix, dimension):
    unvisited = set(range(dimension))
    tour = [0]
    unvisited.remove(0)

    while unvisited:
        last = tour[-1]
        next_city = min(unvisited, key=lambda city: distance_matrix[last][city])
        tour.append(next_city)
        unvisited.remove(next_city)

    return tour

def calculate_tour_cost(tour, distance_matrix):
    cost = 0
    for i in range(len(tour) - 1):
        cost += distance_matrix[tour[i]][tour[i + 1]]
    cost += distance_matrix[tour[-1]][tour[0]]  # returning to the start
    return cost

def main(folder_paths):
    distance_matrices, dimensions = aggregate_input_data(folder_paths)
    
    all_tours = []
    all_costs = []
    
    for i in range(len(folder_paths)):
        distance_matrix = distance_matrices[i]
        dimension = dimensions[i]
        
        tour = nearest_neighbor(distance_matrix, dimension)
        cost = calculate_tour_cost(tour, distance_matrix)
        
        all_tours.append(tour)
        all_costs.append(cost)
        
        print(f"Tour for {folder_paths[i]}:", tour)
        print(f"Cost for {folder_paths[i]}:", cost)
        print()
    
    return all_tours, all_costs

if __name__ == '__main__':
    # Example folder paths containing the data files
    folder_paths = [
        'br17.atsp',
        'ft53.atsp',
        'ft70.atsp'
    ]
    jol= aggregate_input_data( folder_paths)
    print(jol)
  
