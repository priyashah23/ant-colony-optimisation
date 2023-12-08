import antColonyOptimisation
import numpy as np
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import elitismACO


def obtain_city_info(name: int) -> list:
    """
    Reads the xml file
    :return: a list of vertices from the city
    """
    city = "burma14" if name == 1 else "brazil58"
    cities = ET.parse(f"{city}.xml")
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
    Initialise the pheromone matrix to be a random number between 1 and 0
    :param size_x: Width of Adjacency Matrix
    :param size_y: Height of Adjacency Matrix
    :return: Pheromone Matrix
    """
    pheromones = np.random.uniform(0, 1, size=(size_x, size_y))
    return pheromones


def test_against_multiple_ant_colonies():
    pass


def create_first_trial(DECAY_FACTOR, NUMBER_OF_ANTS, ALPHA, BETA, adj_matrix, city):
    for trial in range(5):
        tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
        print(f"(Trial {trial + 1})")
        list_of_points = antColonyOptimisation.ant_system_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA, DECAY_FACTOR)
        x, y = zip(*list_of_points)
        plt.plot(x, y, label=f"Trial: {trial}")
    plt.title(f"{city}: No. of Ants: {NUMBER_OF_ANTS}, Decay_factor = 0.5, Alpha = 1, Beta = 2")
    plt.legend()
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Ant Fitness')
    plt.xscale('log')
    plt.savefig(f'experiments/{city}/number_of_ants_{NUMBER_OF_ANTS}.png')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DECAY_FACTOR = 0.5
    NUMBER_OF_ANTS = 100
    LOCAL_SEARCH = False
    ALPHA = 1
    BETA = 2

    random.seed()
    city_selection = int(input("Select 1: Burma and Select 2: Brazil\n"))
    vertices = obtain_city_info(city_selection)
    adj_matrix = create_adjacency_matrix(vertices)
    city = "burma" if city_selection == 1 else "brazil"
    create_first_trial(DECAY_FACTOR, NUMBER_OF_ANTS, ALPHA, BETA, adj_matrix, city)





