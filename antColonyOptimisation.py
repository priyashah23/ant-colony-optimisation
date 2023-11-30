import random

import numpy as np

MAXIMUM_NUMBER_OF_ITERATIONS = 10_000
TEST_NUMBER_OF_ITERATIONS = 1
TEST_SINGLE_ANT = 1


class Ant:
    # May need to include ant.cost
    def __init__(self, current_city: int, ant_memory: np.ndarray):
        self.current_city = current_city
        self.ant_memory = ant_memory

    def remove_current_node(self, current_node: int) -> None:
        for i in range(self.ant_memory.shape[0]):
            for j in range(self.ant_memory.shape[1]):
                if j == current_node and i != j:
                    self.ant_memory[i][j] = 0
        print(self.ant_memory)

    def compute_transition_probabilities(self, current_node, eta, tau, alpha, beta, t) -> list:
        # Calculate Numerator
        numerators = [(tau[current_node][i] * t)**alpha * (eta[current_node][i] * t) ** beta for i in range(self.ant_memory.shape[0])]

        # Calculate Denominator
        denominator = 0
        for element in numerators:
            denominator += element

        # Probabilities
        probabilities = [round(numerators[i]/denominator, 4) for i in range(self.ant_memory.shape[0])]
        return probabilities

def ant_system_algorithm(number_of_ants: int, adj_matrix: np.ndarray, tau: np.ndarray, alpha, beta):
    heuristic_matrix = construct_heuristic_matrix(adj_matrix)
    #initalise ants here
    ant_list = [Ant(random.randrange(1, adj_matrix.shape[0] + 1), adj_matrix.copy()) for i in range(1)] #TODO -change this

    for n in range(TEST_NUMBER_OF_ITERATIONS):
        for current_ant in ant_list: # number of ants
            # Remove the current city as a possibility (This will be looped)
            current_ant.remove_current_node(current_ant.current_city)
            probabilities = current_ant.compute_transition_probabilities(current_ant.current_city - 1, heuristic_matrix, tau, alpha, beta, TEST_NUMBER_OF_ITERATIONS)
            print(probabilities)



def perform_ant_tour():
    pass


def apply_local_search():
    pass


def update_pheromones():
    pass


def construct_heuristic_matrix(adj_matrix: np.ndarray) -> np.ndarray:

    eta = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if i !=j :
                eta[i][j] = round(1/adj_matrix[i][j], 4)
            else:
                eta[i][j] = 0

    return eta

def construct_pheromone_matrix():
    pass
