import random
import numpy as np

MAXIMUM_NUMBER_OF_FITNESS_EVALUATIONS = 10_000


class Ant:
    """
    Class that
    """
    def __init__(self, current_city: int, city_space: np.ndarray):
        """
        Initialises the ants
        :param current_city: The current city the ant is on
        :param city_space: The
        """
        self.current_city = current_city
        self.city_space = city_space
        self.memory = [current_city]
        self.cost = 0

    def remove_current_node(self, current_node: int) -> None:
        """
        Removes the current city as a possible path for the ant to explore in the adjacency matrix
        :param current_node: Current city
        :return: Void
        """
        rows_to_update = np.arange(self.city_space.shape[0])
        cols_to_update = current_node
        self.city_space[rows_to_update != cols_to_update, cols_to_update] = 0

    def compute_transition_probabilities(self, current_node, tau, alpha, beta, t) -> np.ndarray:
        """
        Computes the transition probabilities
        :param current_node: the current city the ant is on
        :param tau: The pheromone matrix
        :param alpha: Alpha is a constant controlling how much of the pheromone is involved in the transition
        :param beta: Beta is a constant controlling how much the heuristic influences the transitions
        :param t: The current iteration
        :return: a numpy array containing the probabilities of the ant visiting the next node
        """
        # Calculate Numerators
        numerators = tau[current_node] ** alpha * (self.city_space[current_node] ** beta) * t
        # Calculate Denominator
        denominator = numerators.sum()
        # Probabilities
        probabilities = numerators/denominator

        return probabilities

    def update_memory(self, city) -> None:
        """
        Adds the city to the ant's memory
        :param city: The city the ant is going to visit
        :return: None
        """
        return self.memory.append(city)

    def update_cost(self, current_city, next_city, adj_matrix: np.ndarray) -> None:
        """
        This function updates the cost of the ants path from the current city to the next city
        :param current_city: the city the ant is currently on
        :param next_city: the next city the ant will visit
        :param adj_matrix: takes the adjacency matrix to find what the cost is to visit the next city
        :return: None
        """
        if len(self.memory) == adj_matrix.shape[0]:
            first_element = self.memory[0] - 1
            self.cost += adj_matrix[next_city - 1][first_element]
        self.cost += adj_matrix[current_city - 1][next_city - 1]

    def deposit_pheromone(self, pheromone_matrix: np.ndarray) -> np.ndarray:
        """
        This function is where an ant deposit pheromones on every path that the ant visits
        :param pheromone_matrix: The pheromone matrix that is being updated
        :return: The updated pheromone matrix where the current ant has deposited its pheromone
        """
        delta = 1/self.cost
        for x in range(len(self.memory)):
            if x == len(self.memory) - 1:
                first_element = self.memory[0] - 1
                last_element = self.memory[x] - 1
                pheromone_matrix[last_element][first_element] += delta
            else:
                pheromone_matrix[self.memory[x] - 1][self.memory[x + 1] - 1] += delta
        return pheromone_matrix

def ant_system_algorithm(number_of_ants: int, adj_matrix: np.ndarray, tau: np.ndarray, alpha, beta, decay_factor, Q) -> list:
    """
    :param number_of_ants:
    :param adj_matrix:
    :param tau:
    :param alpha:
    :param beta:
    :param decay_factor:
    :return:
    """
    list_of_points = []
    heuristic_matrix = construct_heuristic_matrix(adj_matrix, Q)
    number_of_fitness_evaluations = 0
    while number_of_fitness_evaluations <= MAXIMUM_NUMBER_OF_FITNESS_EVALUATIONS:
        # initialise ants here
        ant_list = [Ant(random.randrange(1, adj_matrix.shape[0] + 1), heuristic_matrix.copy()) for _ in
                    range(number_of_ants)]
        # construct ant solutions
        for current_ant in ant_list:
            perform_ant_tour(adj_matrix.shape[0] - 1, current_ant, tau, alpha, beta, number_of_fitness_evaluations + 1, adj_matrix)
            number_of_fitness_evaluations += 1
        minimum_cost = min(ant_list, key=lambda x: x.cost)
        list_of_points.append((number_of_fitness_evaluations, minimum_cost.cost))

        # Deposit Pheromones
        for current_ant in ant_list:
            tau = current_ant.deposit_pheromone(tau)

        # Evaporate Pheromones
        tau = evaporate_pheromones(decay_factor, tau)
    return list_of_points



def perform_ant_tour(length: int, current_ant: Ant, tau, alpha, beta, n, adj_matrix):
    """

    :param length:
    :param current_ant:
    :param tau:
    :param alpha:
    :param beta:
    :param n:
    :param adj_matrix:
    :return:
    """
    for _ in range(length):
        current_ant.remove_current_node(current_ant.current_city - 1)
        probabilities = current_ant.compute_transition_probabilities(current_ant.current_city - 1, tau, alpha, beta, n)
        select_next_node(current_ant, probabilities, adj_matrix)

def evaporate_pheromones(decay_factor, tau: np.ndarray) -> np.ndarray:
    """

    :param decay_factor:
    :param tau:
    :return:
    """
    for i in range(tau.shape[0]):
        for j in range(tau.shape[1]):
                tau[i][j] = (1 - decay_factor) * tau[i][j]
    return tau

def construct_heuristic_matrix(adj_matrix: np.ndarray, Q: int) -> np.ndarray:
    """
    :param adj_matrix:
    :return:
    """
    eta = np.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if i != j:
                eta[i][j] = round(Q/adj_matrix[i][j], 4)
            else:
                eta[i][j] = 0

    return eta

def select_next_node(current_ant: Ant, probabilities: np.ndarray, adj_matrix):
    """
    :param current_ant:
    :param probabilities:
    :param adj_matrix:
    :return:
    """
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
