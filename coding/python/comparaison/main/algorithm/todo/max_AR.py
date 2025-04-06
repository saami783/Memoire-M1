# https://jicse.sbu.ac.ir/article_103538_6f2b9197b249065d3050975235c171b4.pdf
import random
import networkx as nx

def maxA(graph: nx.Graph):
    G = graph.copy()
    cover = set()

    while G.number_of_edges() > 0:
        zero_degree_nodes = [node for node, degree in G.degree() if degree == 0]
        G.remove_nodes_from(zero_degree_nodes)

        if G.number_of_edges() == 0:
            break

        degrees = dict(G.degree())
        min_degree = min(degrees.values())
        min_nodes = [node for node in degrees if degrees[node] == min_degree]

        maxadjacent = set()
        for node in min_nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
            max_neighbor = max(neighbors, key=lambda n: degrees[n])
            maxadjacent.add(max_neighbor)

        cover.update(maxadjacent)
        G.remove_nodes_from(min_nodes)

    return list(cover)



def maxAR(graph: nx.Graph):
    G = graph.copy()
    cover = set()

    while G.number_of_edges() > 0:
        # supprime les sommets de degré 0
        zero_degree_nodes = [node for node, degree in G.degree() if degree == 0]
        G.remove_nodes_from(zero_degree_nodes)

        if G.number_of_edges() == 0:
            break

        # trouve les sommets de degré minimum
        min_degree = min(dict(G.degree()).values())
        min_nodes = [node for node, degree in G.degree() if degree == min_degree]

        selected = set()
        for node in min_nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue

            max_neighbor = max(neighbors, key=lambda n: G.degree[n])
            rand_neighbor = random.choice(neighbors)

            # choix de l’un des deux avec probabilité ½
            chosen = random.choice([max_neighbor, rand_neighbor])
            selected.add(chosen)

        cover.update(selected)
        G.remove_nodes_from(selected)

    return list(cover)