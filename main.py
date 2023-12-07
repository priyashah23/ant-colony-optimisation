import antColonyOptimisation
import numpy as np
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import elitismACO


def obtain_city_info() -> list:
    cities = ET.parse("brazil58.xml")
    root = cities.getroot()
    graph = root.find("graph")
    vertices = graph.findall("vertex")
    return vertices

def create_adjacency_matrix(vertices : list) -> np.ndarray:
    """
    Construct an adjacency matrix based off the vertices from the XML file
    :param vertices: List of vertices
    :return: The adjacency matrix
    """
    size = len(vertices)
    adj_matrix = np.zeros((size, size))
    for i in range(size):
        edges = vertices[i].findall("edge")
        for edge in edges:
            neighbour = int(edge.text)
            adj_matrix[i][neighbour] += float(edge.get("cost"))
    return adj_matrix

def initialise_pheromones(size_x: int, size_y:int) -> np.ndarray:
    """
    Initialise the pheremone matrix to be a random number between 1 and 0
    :param size_x: Width of Adjacency Matrix
    :param size_y: Height of Adjacency Matrix
    :return: Pheremone Matrix
    """

    pheromones = np.zeros((size_x, size_y))
    for i in range(size_x):
        for j in range(size_y):
            pheromones[i][j] = random.uniform(0, 1)
    return pheromones


def create_first_trial(DECAY_FACTOR, NUMBER_OF_ANTS, ALPHA, BETA, adj_matrix):
    for trial in range(5):
        tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
        print(f"(Trial {trial + 1})")
        list_of_points = antColonyOptimisation.ant_system_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA, DECAY_FACTOR)
        x, y = zip(*list_of_points)
        plt.plot(x, y, label=f"Trial: {trial}")
    plt.title("Brazil: No. of Ants: 50, Decay_factor = 0.5, Alpha = 1, Beta = 2")
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Average Cost')
    plt.xscale('log')
    plt.savefig('experiments/brazil/ten_ants.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DECAY_FACTOR = 0.5
    NUMBER_OF_ANTS = 50
    LOCAL_SEARCH = False
    ALPHA = 1
    BETA = 2

    random.seed(10)
    vertices = obtain_city_info()
    adj_matrix = create_adjacency_matrix(vertices)
    # tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
    # list_of_points = elitismACO.elitism_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA, DECAY_FACTOR, 0.2)
    # x, y = zip(*list_of_points)
    # plt.plot(x, y, label=f"Trial: 1")
    # plt.xscale('log')
    # plt.show()

    create_first_trial(DECAY_FACTOR, NUMBER_OF_ANTS, ALPHA, BETA, adj_matrix)





