import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm

# Définition du modèle GNN pour la couverture minimale de sommets
class GNNVertexCover(nn.Module):
    """
    Modèle de réseau de neurones de graphes (GNN) pour prédire si chaque nœud doit être inclus
    dans la couverture minimale de sommets.

    Cette définition doit correspondre exactement à celle utilisée lors de l'entraînement.
    """
    def __init__(self, in_channels, hidden_channels):
        super(GNNVertexCover, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 2)  # 2 classes : dans la couverture ou non

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Fonction pour résoudre le problème de couverture minimale de sommets avec PuLP
def solve_vertex_cover_pulp(G):
    """
    Calcule la couverture minimale de sommets du graphe G en utilisant la programmation linéaire.

    Paramètres:
    - G: Un graphe NetworkX.

    Retourne:
    - vertex_cover: Liste des nœuds faisant partie de la couverture minimale.
    """
    problem = pulp.LpProblem("Minimum_Vertex_Cover", pulp.LpMinimize)
    node_vars = pulp.LpVariable.dicts("node", G.nodes(), 0, 1, cat='Binary')
    problem += pulp.lpSum(node_vars.values())

    for u, v in G.edges():
        problem += node_vars[u] + node_vars[v] >= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    vertex_cover = [v for v in G.nodes() if pulp.value(node_vars[v]) == 1]
    return vertex_cover

# Fonction pour calculer les features personnalisées d'un graphe
def compute_custom_features(G, node_list):
    """
    Calcule un vecteur de caractéristiques pour chaque nœud du graphe G.

    Paramètres:
    - G: Un graphe NetworkX.
    - node_list: Liste des nœuds de G.

    Retourne:
    - x: Tenseur de forme (num_nodes, num_features), où chaque ligne correspond aux caractéristiques d'un nœud.

    Caractéristiques calculées:
    - Placeholder pour les arêtes incidentes couvertes (initialisé à 0).
    - Degré du nœud.
    - Nombre de triangles dont le nœud fait partie.
    - Somme des degrés des voisins.
    - Nombre de voisins avec un degré de 1.
    - Degré moyen du graphe.
    - Degré minimum du graphe.
    - Degré maximum du graphe.
    """
    n_nodes = len(node_list)
    degrees = np.array([G.degree(node) for node in node_list])
    triangles = np.array([nx.triangles(G, node) for node in node_list])
    sum_degrees_neighbors = np.array([
        sum(G.degree(neighbor) for neighbor in G.neighbors(node))
        for node in node_list
    ])
    neighbors_degree_one = np.array([
        sum(1 for neighbor in G.neighbors(node) if G.degree(neighbor) == 1)
        for node in node_list
    ])
    mean_degree_graph = degrees.mean() if degrees.size > 0 else 0
    min_degree_graph = degrees.min() if degrees.size > 0 else 0
    max_degree_graph = degrees.max() if degrees.size > 0 else 0

    features = []
    for i in range(n_nodes):
        feature_vector = [
            0.0,                         # Arêtes incidentes déjà couvertes (initialisé à 0)
            degrees[i],                  # Degré du nœud
            triangles[i],                # Nombre de triangles
            sum_degrees_neighbors[i],    # Somme des degrés des voisins
            neighbors_degree_one[i],     # Nombre de voisins de degré 1
            mean_degree_graph,           # Degré moyen du graphe
            min_degree_graph,            # Degré minimum du graphe
            max_degree_graph             # Degré maximum du graphe
        ]
        features.append(feature_vector)

    x = torch.tensor(features, dtype=torch.float)
    return x

# Fonction pour vérifier si un ensemble de nœuds est une couverture de sommets
def is_vertex_cover(G, vertex_cover_nodes):
    """
    Vérifie si l'ensemble de nœuds donné est une couverture de sommets du graphe G.

    Paramètres:
    - G: Un graphe NetworkX.
    - vertex_cover_nodes: Ensemble de nœuds proposé comme couverture.

    Retourne:
    - is_cover: Booléen indiquant si l'ensemble est une couverture valide.
    - uncovered_edges: Liste des arêtes non couvertes (si l'ensemble n'est pas une couverture valide).
    """
    covered_edges = set()
    for node in vertex_cover_nodes:
        for neighbor in G.neighbors(node):
            edge = tuple(sorted((node, neighbor)))
            covered_edges.add(edge)
    all_edges = set(tuple(sorted(edge)) for edge in G.edges())
    uncovered_edges = all_edges - covered_edges
    is_cover = len(uncovered_edges) == 0
    return is_cover, uncovered_edges

# Fonction de l'algorithme itératif
def iterative_model_cover(G, model, device):
    """
    Implémente l'algorithme itératif décrit.

    Paramètres:
    - G: Un graphe NetworkX.
    - model: Le modèle GNN entraîné.
    - device: Le device (CPU ou GPU).

    Retourne:
    - solution_set: Ensemble des nœuds sélectionnés comme couverture.
    """
    solution_set = set()
    G_current = G.copy()

    while G_current.number_of_edges() > 0:
        node_list = list(G_current.nodes())
        if not node_list:
            break  # Plus de nœuds à traiter

        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Calcul des caractéristiques
        x = compute_custom_features(G_current, node_list)

        # Création de edge_index
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_current.edges()]
        if not edges:
            break  # Plus d'arêtes à traiter
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Création de l'objet Data
        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)

        # Prédiction avec le modèle
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1).cpu().numpy()

        # Trouver un sommet sélectionné par le modèle
        node_selected = False
        for node, label in zip(node_list, pred):
            if label == 1:
                solution_set.add(node)
                G_current.remove_node(node)
                node_selected = True
                break  # On arrête la boucle pour re-tester le graphe mis à jour

        if not node_selected:
            # Si aucun sommet n'est sélectionné, on choisit le sommet de degré maximum
            degrees = dict(G_current.degree())
            if degrees:
                max_degree_node = max(degrees, key=degrees.get)
                solution_set.add(max_degree_node)
                G_current.remove_node(max_degree_node)
            else:
                break  # Plus de sommets à traiter

    return solution_set

# Fonction de l'algorithme Maximum Degree Greedy
def maximum_degree_greedy(G):
    """
    Implémente l'algorithme glouton du degré maximum (MDG) pour la couverture de sommets.

    Paramètres:
    - G: Un graphe NetworkX.

    Retourne:
    - cover_set: Ensemble des nœuds sélectionnés comme couverture.
    """
    G_copy = G.copy()
    cover_set = set()

    while G_copy.number_of_edges() > 0:
        degrees = dict(G_copy.degree())
        max_degree_node = max(degrees, key=degrees.get)
        cover_set.add(max_degree_node)
        G_copy.remove_node(max_degree_node)

    return cover_set

# Fonction principale
def main():
    """
    Fonction principale du script.
    Génère plusieurs graphes de différentes classes, utilise les algorithmes pour prédire la couverture de sommets,
    compare avec la solution exacte, et affiche un bilan global pour chaque classe de graphes.
    """
    parser = argparse.ArgumentParser(description='Utilisation du modèle GNN pour la couverture minimale de sommets.')
    parser.add_argument('--model_path', type=str, default='gnn_vertex_cover_model.pth', help='Chemin du modèle entraîné')
    parser.add_argument('--num_graphs', type=int, default=100, help='Nombre de graphes à générer par classe')
    parser.add_argument('--num_nodes', type=int, default=20, help='Nombre de nœuds des graphes')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Nombre de canaux cachés dans le modèle GNN')
    args = parser.parse_args()

    # Détection du device (CPU ou GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Initialisation du modèle et chargement des poids
    in_channels = 8  # Nombre de caractéristiques d'entrée (doit correspondre à compute_custom_features)
    model = GNNVertexCover(in_channels=in_channels, hidden_channels=args.hidden_channels).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Définition des classes de graphes à tester
    graph_classes = {
        'Arbres': lambda: nx.random_tree(n=args.num_nodes),
        'Graphes aléatoires (p=0.1)': lambda: nx.erdos_renyi_graph(n=args.num_nodes, p=0.1),
        'Graphes aléatoires (p=0.3)': lambda: nx.erdos_renyi_graph(n=args.num_nodes, p=0.3),
        'Graphes aléatoires (p=0.5)': lambda: nx.erdos_renyi_graph(n=args.num_nodes, p=0.5),
        'Graphes réguliers (degré=3)': lambda: nx.random_regular_graph(d=3, n=args.num_nodes),
        'Graphes réguliers (degré=4)': lambda: nx.random_regular_graph(d=4, n=args.num_nodes),
        'Graphes complets': lambda: nx.complete_graph(n=args.num_nodes),
        'Graphes cycles': lambda: nx.cycle_graph(n=args.num_nodes),
        'Graphes bipartis complets (n/2, n/2)': lambda: nx.complete_bipartite_graph(n1=args.num_nodes//2, n2=args.num_nodes - args.num_nodes//2),
    }

    # Pour chaque classe de graphes, collecter les statistiques séparément
    for class_name, graph_generator in graph_classes.items():
        print(f"\n=== Traitement de la classe de graphes : {class_name} ===")
        total_graphs = 0

        # Pour l'algorithme itératif
        total_iterative_size_difference = 0
        iterative_approx_ratios = []
        iterative_max_approx_ratio = float('-inf')
        iterative_min_approx_ratio = float('inf')

        # Pour l'algorithme Maximum Degree Greedy
        total_mdg_size_difference = 0
        mdg_approx_ratios = []
        mdg_max_approx_ratio = float('-inf')
        mdg_min_approx_ratio = float('inf')

        # Progress bar pour la classe actuelle
        for i in tqdm(range(args.num_graphs), desc=f"Traitement des graphes {class_name}"):
            # Génération du graphe
            try:
                G = graph_generator()
            except nx.exception.NetworkXError as e:
                # Parfois, la génération peut échouer (par exemple, un graphe régulier avec les paramètres donnés peut ne pas exister)
                continue

            if G.number_of_edges() == 0:
                continue  # On ignore les graphes sans arêtes

            # Calcul de la couverture minimale avec PuLP
            exact_cover_nodes = solve_vertex_cover_pulp(G)
            if len(exact_cover_nodes) == 0:
                continue  # On ignore les graphes sans solution

            # Utilisation de l'algorithme itératif
            iterative_cover_nodes = iterative_model_cover(G, model, device)

            # Vérification si la solution itérative est une couverture valide
            is_cover_iterative, _ = is_vertex_cover(G, iterative_cover_nodes)

            # Utilisation de l'algorithme Maximum Degree Greedy
            mdg_cover_nodes = maximum_degree_greedy(G)

            # Vérification si la solution MDG est une couverture valide
            is_cover_mdg, _ = is_vertex_cover(G, mdg_cover_nodes)

            # Mise à jour des statistiques
            total_graphs += 1

            # Pour l'algorithme itératif
            iterative_size_difference = len(iterative_cover_nodes) - len(exact_cover_nodes)
            total_iterative_size_difference += iterative_size_difference

            if len(exact_cover_nodes) > 0:
                iterative_approx_ratio = len(iterative_cover_nodes) / len(exact_cover_nodes)
                iterative_approx_ratios.append(iterative_approx_ratio)
                iterative_max_approx_ratio = max(iterative_max_approx_ratio, iterative_approx_ratio)
                iterative_min_approx_ratio = min(iterative_min_approx_ratio, iterative_approx_ratio)

            # Pour l'algorithme MDG
            mdg_size_difference = len(mdg_cover_nodes) - len(exact_cover_nodes)
            total_mdg_size_difference += mdg_size_difference

            if len(exact_cover_nodes) > 0:
                mdg_approx_ratio = len(mdg_cover_nodes) / len(exact_cover_nodes)
                mdg_approx_ratios.append(mdg_approx_ratio)
                mdg_max_approx_ratio = max(mdg_max_approx_ratio, mdg_approx_ratio)
                mdg_min_approx_ratio = min(mdg_min_approx_ratio, mdg_approx_ratio)

        # Calcul des métriques moyennes pour l'algorithme itératif
        average_iterative_size_difference = total_iterative_size_difference / total_graphs if total_graphs > 0 else 0
        average_iterative_approx_ratio = sum(iterative_approx_ratios) / len(iterative_approx_ratios) if iterative_approx_ratios else 0

        # Calcul des métriques moyennes pour l'algorithme MDG
        average_mdg_size_difference = total_mdg_size_difference / total_graphs if total_graphs > 0 else 0
        average_mdg_approx_ratio = sum(mdg_approx_ratios) / len(mdg_approx_ratios) if mdg_approx_ratios else 0

        # Affichage du bilan pour l'algorithme itératif
        print(f"\n--- Statistiques pour l'algorithme itératif sur {class_name} ---")
        print(f"Nombre total de graphes traités : {total_graphs}")
        print(f"Différence moyenne de taille entre la solution itérative et la solution exacte : {average_iterative_size_difference:.2f}")
        print(f"Ratio d'approximation moyen : {average_iterative_approx_ratio:.2f}")
        print(f"Ratio d'approximation maximum : {iterative_max_approx_ratio:.2f}")
        print(f"Ratio d'approximation minimum : {iterative_min_approx_ratio:.2f}")

        # Affichage du bilan pour l'algorithme MDG
        print(f"\n--- Statistiques pour l'algorithme Maximum Degree Greedy sur {class_name} ---")
        print(f"Nombre total de graphes traités : {total_graphs}")
        print(f"Différence moyenne de taille entre la solution MDG et la solution exacte : {average_mdg_size_difference:.2f}")
        print(f"Ratio d'approximation moyen : {average_mdg_approx_ratio:.2f}")
        print(f"Ratio d'approximation maximum : {mdg_max_approx_ratio:.2f}")
        print(f"Ratio d'approximation minimum : {mdg_min_approx_ratio:.2f}")

if __name__ == '__main__':
    main()
