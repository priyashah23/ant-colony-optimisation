import antColonyOptimisation
import numpy as np
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

import elitismACO
import menu
import rankBasedSystem


def obtain_city_info(name: int) -> list:
    """
    Reads the xml file
    :return: a list of vertices from the city
    """
    city = "burma14" if name == 1 else "brazil58"
    try:
        cities = ET.parse(f"{city}.xml")
    except:
        print("Unable to read file. Try and see whether brazil and burma are in the same directory")
        exit(-1)
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

def create_first_trial(parameters: list, adj_matrix : np.ndarray, city, Q):
    """

    :param parameters:
    :param adj_matrix:
    :param city:
    :param Q:
    :return:
    """
    NUMBER_OF_ANTS = parameters[1]
    DECAY_FACTOR, ALPHA, BETA = parameters[0], parameters[2], parameters[3]

    for trial in range(5):
        tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
        print(f"(Trial {trial + 1})")
        list_of_points = antColonyOptimisation.ant_system_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA,
                                                                    DECAY_FACTOR, Q)
        print(list_of_points)
        x, y = zip(*list_of_points)
        plt.plot(x, y, label=f"Trial: {trial}")
    plt.title(f"{city}: No. of Ants: {NUMBER_OF_ANTS}, Decay_factor = {DECAY_FACTOR}, Alpha = {ALPHA}, Beta = {BETA}, Q={Q}")
    plt.legend()
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Heuristic Function')
    plt.xscale('log')
    plt.show()


def plot_different_ant_colonies(DECAY_FACTOR, NUMBER_OF_ANTS, ALPHA, BETA, adj_matrix, city):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(50, 10))
    for trial in range(3):
        print(f"(Trial {trial + 1})")
        # Regular aco optimisation
        tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
        aco_list_of_points = antColonyOptimisation.ant_system_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA, DECAY_FACTOR)
        aco_x, aco_y = zip(*aco_list_of_points)

        # Elitism Approach
        tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
        list_of_points = elitismACO.elitism_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA, DECAY_FACTOR, 0.5)
        elite_x, elite_y = zip(*list_of_points)

        # Rank Based System
        tau = initialise_pheromones(adj_matrix.shape[0], adj_matrix.shape[1])
        rank_based_points = rankBasedSystem.rank_based_algorithm(NUMBER_OF_ANTS, adj_matrix, tau, ALPHA, BETA, DECAY_FACTOR, 0.5, 6)
        x, y = zip(*rank_based_points)

        # Print out the graphs
        ax[trial].title.set_text(f"{city.upper()}: No. of Ants: {NUMBER_OF_ANTS}, Trial: {trial + 1}")
        ax[trial].set_xlabel('Number of Evaluations')
        ax[trial].set_ylabel('Heuristic Cost')
        ax[trial].set_xscale('log')
        ax[trial].plot(aco_x, aco_y, label="AS System")
        ax[trial].plot(elite_x, elite_y, label="Elitism")
        ax[trial].plot(x, y, label=f"Rank_Based System")
        ax[trial].legend(loc="upper right", fontsize=12)
    plt.savefig(f'experiments/{city}/different_aco_variations.png')


if __name__ == '__main__':
    Q = 1
    random.seed()
    # User selects which city and which parameters
    city_selection = menu.selection_menu()
    parameters = menu.select_parameters()

    # Get city information from files
    vertices = obtain_city_info(city_selection)
    adj_matrix = create_adjacency_matrix(vertices)
    city = "burma" if city_selection == 1 else "brazil"
    create_first_trial(parameters, adj_matrix, city, Q)

    #plot_different_ant_colonies(DECAY_FACTOR, NUMBER_OF_ANTS, ALPHA, BETA, adj_matrix, city)






