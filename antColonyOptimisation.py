import random
import numpy as np
import math

MAXIMUM_NUMBER_OF_ITERATIONS = 10_000
TEST_NUMBER_OF_ITERATIONS = 10
TEST_SINGLE_ANT = 1


class Ant:
    def __init__(self, current_city: int, city_space: np.ndarray):
        self.current_city = current_city
        self.city_space = city_space
        self.memory = [current_city]
        self.cost = 0

    def remove_current_node(self, current_node: int) -> None:
        for i in range(self.city_space.shape[0]):
            for j in range(self.city_space.shape[1]):
                if j == (current_node) and i != j:
                    self.city_space[i][j] = 0

    def compute_transition_probabilities(self, current_node, tau, alpha, beta, t) -> list:
        # Calculate Numerator
        numerators = [(tau[current_node][i] * t) ** alpha * (self.city_space[current_node][i] * t) ** beta for i in range(self.city_space.shape[0])]
        # Calculate Denominator
        denominator = 0
        for element in numerators:
            denominator += element

        # Probabilities
        probabilities = [round(numerators[i]/denominator, 4) for i in range(self.city_space.shape[0])]

        return probabilities

    def update_memory(self, city):
        return self.memory.append(city)

    def update_cost(self, current_city, next_city, adj_matrix: np.ndarray):
        if len(self.memory) == adj_matrix.shape[0]:
            first_element = self.memory[0] - 1
            self.cost += adj_matrix[next_city - 1][first_element]
        self.cost += adj_matrix[current_city - 1][next_city - 1]

    def deposit_pheromone(self, pheromone_matrix: np.ndarray) -> np.ndarray:
        delta = 1/self.cost
        for x in range(len(self.memory)):
            if x == len(self.memory) - 1:
                first_element = self.memory[0] - 1
                last_element = self.memory[x] - 1
                pheromone_matrix[last_element][first_element] += delta
            else:
               pheromone_matrix[self.memory[x] - 1][self.memory[x + 1] - 1] += delta
        return pheromone_matrix

def ant_system_algorithm(number_of_ants: int, adj_matrix: np.ndarray, tau: np.ndarray, alpha, beta, decay_factor) -> list:
    list_of_points = []
    heuristic_matrix = construct_heuristic_matrix(adj_matrix)
    for n in range(MAXIMUM_NUMBER_OF_ITERATIONS):
        # initalise ants here
        ant_list = [Ant(random.randrange(1, adj_matrix.shape[0] + 1), heuristic_matrix.copy()) for _ in range(number_of_ants)]
        counter = 0
        print("Iteration: ", n)
        # construct ant solutions
        for current_ant in ant_list:
            perform_ant_tour(adj_matrix.shape[0] - 1, current_ant, tau, alpha, beta, n + 1, adj_matrix)
            print(f"Ant {counter} Memory: {current_ant.memory}")
            counter += 1
        tau = evaporate_pheromones(decay_factor, tau)
        minimum_cost = math.inf
        for current_ant in ant_list:
            print(current_ant.cost)
            tau = current_ant.deposit_pheromone(tau)
            if current_ant.cost < minimum_cost:
                minimum_cost = current_ant.cost
        list_of_points.append((n, minimum_cost))
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

def select_next_node(current_ant: Ant, probabilities: list, adj_matrix):
    # TODO - move into own method
    cumulative_probabilities = []
    for x in range(len(probabilities)):
        if x == 0:
            cumulative_probabilities.append(probabilities[x])
        else:
            cumulative_probabilities.append(cumulative_probabilities[x - 1] + probabilities[x])

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