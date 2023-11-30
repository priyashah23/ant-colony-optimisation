import antColonyOptimisation
import numpy as np
import random
import xml.etree.ElementTree as ET

def obtain_city_info() -> list:
    cities = ET.parse("burma14.xml")
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
            pheromones[i][j] += random.uniform(0, 1)
    return pheromones

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DECAY_FACTOR = 0
    NUMBER_OF_ANTS = 5
    LOCAL_SEARCH = False
    ALPHA = 1
    BETA = 2

    random.seed(10)
    vertices = obtain_city_info()
    adj_matrix = create_adjacency_matrix(vertices)
    tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
    antColonyOptimisation.ant_system_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA)



