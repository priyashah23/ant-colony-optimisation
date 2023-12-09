import random
import numpy as np
import math

MAXIMUM_NUMBER_OF_FITNESS_EVALUATIONS = 10_000

class Ant:
    def __init__(self, current_city: int, city_space: np.ndarray):
        self.current_city = current_city
        self.city_space = city_space
        self.memory = [current_city]
        self.cost = 0

    def remove_current_node(self, current_node: int) -> None:
        rows_to_update = np.arange(self.city_space.shape[0])
        cols_to_update = current_node
        self.city_space[rows_to_update != cols_to_update, cols_to_update] = 0
    def compute_transition_probabilities(self, current_node, tau, alpha, beta, t) -> np.ndarray:
        # Calculate Numerators
        numerators = tau[current_node] ** alpha * (self.city_space[current_node] ** beta) * t
        # Calculate Denominator
        denominator = numerators.sum()
        # Probabilities
        probabilities = numerators / denominator

        return probabilities

    def update_memory(self, city):
        return self.memory.append(city)

    def update_cost(self, current_city, next_city, adj_matrix: np.ndarray):
        if len(self.memory) == adj_matrix.shape[0]:
            first_element = self.memory[0] - 1
            self.cost += adj_matrix[next_city - 1][first_element]
        self.cost += adj_matrix[current_city - 1][next_city - 1]

    def deposit_pheromone(self, pheromone_matrix: np.ndarray, elitism_weight: float) -> np.ndarray:
        e = elitism_weight * (1/self.cost)
        delta = 1/self.cost
        for x in range(len(self.memory)):
            if x == len(self.memory) - 1:
                first_element = self.memory[0] - 1
                last_element = self.memory[x] - 1
                pheromone_matrix[last_element][first_element] += delta + e
            else:
               pheromone_matrix[self.memory[x] - 1][self.memory[x + 1] - 1] += delta + e
        return pheromone_matrix

def elitism_algorithm(number_of_ants: int, adj_matrix: np.ndarray, tau: np.ndarray, alpha, beta, decay_factor, elitism_weight) -> list:
    best_ant = None
    best_cost = math.inf
    list_of_points = []

    heuristic_matrix = construct_heuristic_matrix(adj_matrix)
    number_of_fitness_evaluations = 0
    while number_of_fitness_evaluations <= MAXIMUM_NUMBER_OF_FITNESS_EVALUATIONS:
        # initialise ants here
        ant_list = [Ant(random.randrange(1, adj_matrix.shape[0] + 1), heuristic_matrix.copy()) for _ in range(number_of_ants)]
        # construct ant solutions
        for current_ant in ant_list:
            perform_ant_tour(adj_matrix.shape[0] - 1, current_ant, tau, alpha, beta, number_of_fitness_evaluations + 1, adj_matrix)
            number_of_fitness_evaluations += 1
            if current_ant.cost < best_cost:
                best_ant = current_ant
                best_cost = current_ant.cost
        minimum_cost = min(ant_list, key=lambda x: x.cost)
        list_of_points.append((number_of_fitness_evaluations, minimum_cost.cost))
        tau = evaporate_pheromones(decay_factor, tau)
        tau = best_ant.deposit_pheromone(tau, elitism_weight)
        for current_ant in ant_list:
            tau = current_ant.deposit_pheromone(tau, 0)
    return list_of_points



def perform_ant_tour(length: int, current_ant: Ant, tau, alpha, beta, n, adj_matrix):
    for _ in range(length):
        current_ant.remove_current_node(current_ant.current_city - 1)
        probabilities = current_ant.compute_transition_probabilities(current_ant.current_city - 1, tau, alpha, beta, n)
        select_next_node(current_ant, probabilities, adj_matrix)


def apply_local_search():
    pass


def evaporate_pheromones(decay_factor, tau: np.ndarray) -> np.ndarray:
    for i in range(tau.shape[0]):
        for j in range(tau.shape[1]):
                tau[i][j] = (1 - decay_factor) * tau[i][j]
    return tau

def construct_heuristic_matrix(adj_matrix: np.ndarray) -> np.ndarray:

    eta = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))

    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if i !=j :
                eta[i][j] = round(1/adj_matrix[i][j], 4)
            else:
                eta[i][j] = 0

    return eta

def select_next_node(current_ant: Ant, probabilities: np.ndarray, adj_matrix):
    cumulative_probabilities = np.cumsum(probabilities)
    random_probability = random.uniform(0, 1)
    for x in range(len(cumulative_probabilities)):
        if cumulative_probabilities[x] >= random_probability:
            current_city = current_ant.current_city
            # add current city to path
            current_ant.update_memory(x + 1)
            current_ant.current_city = x + 1
            next_city = current_ant.current_city
            # update cost
            current_ant.update_cost(current_city, next_city, adj_matrix)
            break
