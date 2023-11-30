import random

import numpy as np

MAXIMUM_NUMBER_OF_ITERATIONS = 10_000
TEST_NUMBER_OF_ITERATIONS = 1
TEST_SINGLE_ANT = 1


class Ant:
    def __init__(self, starting_city: int, ant_memory: np.ndarray):
        self.starting_city = starting_city
        self.ant_memory = ant_memory

    def remove_current_node(self, current_node: int) -> None:
        for i in range(self.ant_memory.shape[0]):
            for j in range(self.ant_memory.shape[1]):
                if j == current_node and i != j:
                    self.ant_memory[i][j] = 0

    def compute_transition_probabilities(self):
        pass

def ant_system_algorithm(number_of_ants: int, local_search: bool, adj_matrix: np.ndarray):
    heuristic_matrix = construct_heuristic_matrix(adj_matrix)

    for n in range(TEST_NUMBER_OF_ITERATIONS):
        for i in range(TEST_SINGLE_ANT):
            initial_starting_city = random.randrange(0, adj_matrix.shape[0])
            print(initial_starting_city)
            current_ant = Ant(initial_starting_city, adj_matrix.copy())
            # Remove the current city as a possibility
            current_ant.remove_current_node(initial_starting_city)


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
