import numpy as np
import xml.etree.ElementTree as ET

def obtain_city_info():
    cities = ET.parse("burma14.xml")
    root = cities.getroot()
    graph = root.find("graph")
    vertices = graph.findall("vertex")
    create_adjacency_matrix(vertices)

def create_adjacency_matrix(vertices : list):
    size = len(vertices)
    adj_matrix = np.zeros((size, size))
    print(adj_matrix)
    for i in range(size):
        edges = vertices[i].findall("edge")
        for edge in edges:
            neighbour = int(edge.text)
            adj_matrix[i][neighbour] += float(edge.get("cost"))
    print(adj_matrix)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    obtain_city_info()


