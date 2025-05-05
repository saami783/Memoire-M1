import networkx as nx
import numpy as np
from solveur import minimum_vertex_cover


def generate_erdos_renyi_graph(num_nodes=20, prob=0.2, seed=54):
    graph = nx.erdos_renyi_graph(num_nodes, prob, seed=seed)
    return graph, minimum_vertex_cover(graph)

def extract_graph_properties():
    """Extrait toutes les caractéristiques nécessaires d'un graphe"""
    graph, cover_size = generate_erdos_renyi_graph()
    nb_sommets = graph.number_of_nodes()
    nb_aretes = graph.number_of_edges()
    densite = (2 * nb_aretes) / (nb_sommets * (nb_sommets - 1)) if nb_sommets > 1 else 0
    degrees = [deg for _, deg in graph.degree()]
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    avg_degree = np.mean(degrees) if degrees else 0
    num_deg_one = sum(1 for deg in degrees if deg == 1)
    triangle_counts = nx.triangles(graph)
    count_triangles = sum(triangle_counts.values()) // 3
    total_triangles = sum(1 for deg in degrees if deg >= 2)
    fraction_closed_triangles = count_triangles / total_triangles if total_triangles > 0 else 0
    avg_count_triangles = count_triangles / nb_aretes if nb_aretes > 0 else 0
    avg_clustering = nx.average_clustering(graph)
    local_efficiency = nx.local_efficiency(graph) if nx.is_connected(graph) else 0
    is_connected = int(nx.is_connected(graph))
    num_components = nx.number_connected_components(graph)

    return {
        "class": 0,  # À adapter selon vos besoins
        "nb_sommets": nb_sommets,
        "nb_aretes": nb_aretes,
        "densite": densite,
        "cover_size": cover_size,
        "instance_number": 1,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "avg_degree": avg_degree,
        "num_deg_one": num_deg_one,
        "count_triangles": count_triangles,
        "total_triangles": total_triangles,
        "fraction_closed_triangles": fraction_closed_triangles,
        "avg_count_triangles": avg_count_triangles,
        "max_count_triangles": max(triangle_counts.values(), default=0),
        "min_count_triangles": min(triangle_counts.values(), default=0),
        "avg_neighbor_deg": np.mean(list(nx.average_neighbor_degree(graph).values())),
        "sum_neighbor_deg": sum(nx.average_neighbor_degree(graph).values()),
        "degree_assortativity_coefficient": nx.degree_assortativity_coefficient(graph),
        "diameter": nx.diameter(graph) if is_connected else 0,
        "local_efficiency": local_efficiency,
        "avg_neighbor_clustering": avg_clustering,
        "avg_clustering_coefficient": avg_clustering,
        "is_acyclic": int(nx.is_forest(graph)),
        "is_bipartite": int(nx.is_bipartite(graph)),
        "is_connected": is_connected,
        "radius": nx.radius(graph) if is_connected else 0,
        "matching_number": len(nx.max_weight_matching(graph)),
        "largest_eigenvalue": np.max(np.linalg.eigvals(nx.laplacian_matrix(graph).todense())) if nb_sommets > 0 else 0,
        "num_components": num_components,
        "is_eularian": int(nx.is_eulerian(graph)) if is_connected else 0,
        "is_semi_eularian": int(nx.is_semieulerian(graph)) if is_connected else 0,
        "g_is_planar": int(is_planar(graph)),
        "twin_free": int(is_twin_free(graph))
    }

def is_planar(G):
    is_planar, _ = nx.check_planarity(G)
    return is_planar


def is_twin_free(G):
    for u in G.nodes():
        for v in G.nodes():
            if u != v and set(G.neighbors(u)) == set(G.neighbors(v)):
                return False
    return True