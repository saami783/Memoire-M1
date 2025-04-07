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
    mean_degree_graph = degrees.mean()
    min_degree_graph = degrees.min()
    max_degree_graph = degrees.max()

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

# Fonction principale
def main():
    """
    Fonction principale du script.
    Génère plusieurs graphes aléatoires, utilise le modèle pour prédire la couverture de sommets,
    compare avec la solution exacte, et affiche un bilan global.
    """
    parser = argparse.ArgumentParser(description='Utilisation du modèle GNN pour la couverture minimale de sommets.')
    parser.add_argument('--model_path', type=str, default='gnn_vertex_cover_model.pth', help='Chemin du modèle entraîné')
    parser.add_argument('--num_graphs', type=int, default=100, help='Nombre de graphes à générer')
    parser.add_argument('--num_nodes', type=int, default=20, help='Nombre de nœuds des graphes aléatoires')
    parser.add_argument('--probability', type=float, default=0.3, help='Probabilité de création d\'une arête pour les graphes aléatoires')
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

    # Listes pour stocker les métriques globales
    total_graphs = 0
    total_uncovered_edges = 0
    total_size_difference = 0
    total_approximation_ratios = []
    total_max_approx_ratio = float('-inf')
    total_min_approx_ratio = float('inf')

    # Listes pour les graphes avec couverture valide
    valid_graphs = 0
    valid_uncovered_edges = 0
    valid_size_differences = []
    valid_approx_ratios = []
    valid_max_approx_ratio = float('-inf')
    valid_min_approx_ratio = float('inf')

    # Listes pour les graphes avec couverture invalide
    invalid_graphs = 0
    invalid_uncovered_edges = 0
    invalid_size_differences = []
    invalid_approx_ratios = []
    invalid_max_approx_ratio = float('-inf')
    invalid_min_approx_ratio = float('inf')

    # Statistiques pour les graphes résultants après suppression des sommets sélectionnés
    invalid_graph_sizes = []
    invalid_new_approx_ratios = []
    invalid_new_max_approx_ratio = float('-inf')
    invalid_new_min_approx_ratio = float('inf')

    for i in tqdm(range(args.num_graphs), desc="Traitement des graphes"):
        # Génération d'un graphe aléatoire
        G = nx.erdos_renyi_graph(n=args.num_nodes, p=args.probability)
        if G.number_of_edges() == 0:
            print(f"Graphe {i+1}: Pas d'arêtes, passage au graphe suivant.")
            continue

        node_list = list(G.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}

        # Calcul des caractéristiques
        x = compute_custom_features(G, node_list)

        # Création de edge_index
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Création de l'objet Data
        data = Data(x=x, edge_index=edge_index)
        data = data.to(device)

        # Prédiction avec le modèle
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1).cpu().numpy()

        # Nœuds prédits comme faisant partie de la couverture
        predicted_cover_nodes = [node for node, label in zip(node_list, pred) if label == 1]

        # Calcul de la couverture minimale avec PuLP
        exact_cover_nodes = solve_vertex_cover_pulp(G)

        # Vérification si la solution du modèle est une couverture valide
        is_cover, uncovered_edges = is_vertex_cover(G, predicted_cover_nodes)

        # Calcul du ratio d'approximation
        if len(exact_cover_nodes) > 0:
            approximation_ratio = len(predicted_cover_nodes) / len(exact_cover_nodes)
            total_approximation_ratios.append(approximation_ratio)
            total_max_approx_ratio = max(total_max_approx_ratio, approximation_ratio)
            total_min_approx_ratio = min(total_min_approx_ratio, approximation_ratio)
        else:
            approximation_ratio = 0

        # Mise à jour des métriques globales
        total_graphs += 1
        total_uncovered_edges += len(uncovered_edges)
        size_difference = len(predicted_cover_nodes) - len(exact_cover_nodes)
        total_size_difference += size_difference

        # Mise à jour des métriques pour les graphes avec couverture valide ou invalide
        if is_cover:
            # Couverture valide
            valid_graphs += 1
            valid_uncovered_edges += len(uncovered_edges)
            valid_size_differences.append(size_difference)
            valid_approx_ratios.append(approximation_ratio)
            valid_max_approx_ratio = max(valid_max_approx_ratio, approximation_ratio)
            valid_min_approx_ratio = min(valid_min_approx_ratio, approximation_ratio)
        else:
            # Couverture invalide
            invalid_graphs += 1
            invalid_uncovered_edges += len(uncovered_edges)
            invalid_size_differences.append(size_difference)
            invalid_approx_ratios.append(approximation_ratio)
            invalid_max_approx_ratio = max(invalid_max_approx_ratio, approximation_ratio)
            invalid_min_approx_ratio = min(invalid_min_approx_ratio, approximation_ratio)

            # Suppression des sommets sélectionnés par le modèle du graphe
            G_sub = G.copy()
            G_sub.remove_nodes_from(predicted_cover_nodes)
            subgraph_size = G_sub.number_of_nodes()
            invalid_graph_sizes.append(subgraph_size)

            # Sélection des sommets restants comme nouvelle couverture proposée
            new_cover_nodes = list(set(G.nodes()) - set(predicted_cover_nodes))

            # Vérification si la nouvelle solution est une couverture valide
            is_new_cover, new_uncovered_edges = is_vertex_cover(G, new_cover_nodes)

            # Calcul du nouveau ratio d'approximation
            if len(exact_cover_nodes) > 0:
                new_approximation_ratio = len(new_cover_nodes) / len(exact_cover_nodes)
                invalid_new_approx_ratios.append(new_approximation_ratio)
                invalid_new_max_approx_ratio = max(invalid_new_max_approx_ratio, new_approximation_ratio)
                invalid_new_min_approx_ratio = min(invalid_new_min_approx_ratio, new_approximation_ratio)
            else:
                new_approximation_ratio = 0

    # Calcul des métriques moyennes globales
    average_uncovered_edges = total_uncovered_edges / total_graphs
    average_size_difference = total_size_difference / total_graphs
    average_approximation_ratio = sum(total_approximation_ratios) / len(total_approximation_ratios)
    valid_cover_percentage = (valid_graphs / total_graphs) * 100

    # Calcul des statistiques pour les graphes avec couverture valide
    if valid_graphs > 0:
        valid_average_uncovered_edges = valid_uncovered_edges / valid_graphs
        valid_average_size_difference = sum(valid_size_differences) / valid_graphs
        valid_average_approx_ratio = sum(valid_approx_ratios) / valid_graphs
    else:
        valid_average_uncovered_edges = 0
        valid_average_size_difference = 0
        valid_average_approx_ratio = 0

    # Calcul des statistiques pour les graphes avec couverture invalide
    if invalid_graphs > 0:
        invalid_average_uncovered_edges = invalid_uncovered_edges / invalid_graphs
        invalid_average_size_difference = sum(invalid_size_differences) / invalid_graphs
        invalid_average_approx_ratio = sum(invalid_approx_ratios) / invalid_graphs
    else:
        invalid_average_uncovered_edges = 0
        invalid_average_size_difference = 0
        invalid_average_approx_ratio = 0

    # Calcul des statistiques pour les graphes résultants après suppression des sommets sélectionnés
    if invalid_graph_sizes:
        avg_invalid_graph_size = sum(invalid_graph_sizes) / len(invalid_graph_sizes)
        max_invalid_graph_size = max(invalid_graph_sizes)
        min_invalid_graph_size = min(invalid_graph_sizes)
    else:
        avg_invalid_graph_size = max_invalid_graph_size = min_invalid_graph_size = 0

    if invalid_new_approx_ratios:
        avg_invalid_new_approx_ratio = sum(invalid_new_approx_ratios) / len(invalid_new_approx_ratios)
    else:
        avg_invalid_new_approx_ratio = 0

    # Affichage du bilan global
    print("\n=== Bilan Global ===")
    print(f"Nombre total de graphes traités : {total_graphs}")
    print(f"Pourcentage de couvertures valides proposées par le modèle : {valid_cover_percentage:.2f}%")
    print(f"Nombre moyen d'arêtes non couvertes par le modèle : {average_uncovered_edges:.2f}")
    print(f"Différence moyenne de taille entre la solution du modèle et la solution exacte : {average_size_difference:.2f}")
    print(f"Ratio d'approximation moyen : {average_approximation_ratio:.2f}")
    print(f"Ratio d'approximation maximum : {total_max_approx_ratio:.2f}")
    print(f"Ratio d'approximation minimum : {total_min_approx_ratio:.2f}")

    # Affichage des statistiques pour les graphes avec couverture valide
    print("\n=== Statistiques pour les couvertures valides ===")
    print(f"Nombre de graphes avec couverture valide : {valid_graphs}")
    if valid_graphs > 0:
        print(f"Nombre moyen d'arêtes non couvertes : {valid_average_uncovered_edges:.2f}")
        print(f"Différence moyenne de taille : {valid_average_size_difference:.2f}")
        print(f"Ratio d'approximation moyen : {valid_average_approx_ratio:.2f}")
        print(f"Ratio d'approximation maximum : {valid_max_approx_ratio:.2f}")
        print(f"Ratio d'approximation minimum : {valid_min_approx_ratio:.2f}")
    else:
        print("Aucune couverture valide n'a été proposée par le modèle.")

    # Affichage des statistiques pour les graphes avec couverture invalide
    print("\n=== Statistiques pour les couvertures invalides ===")
    print(f"Nombre de graphes avec couverture invalide : {invalid_graphs}")
    if invalid_graphs > 0:
        print(f"Nombre moyen d'arêtes non couvertes : {invalid_average_uncovered_edges:.2f}")
        print(f"Différence moyenne de taille : {invalid_average_size_difference:.2f}")
        print(f"Ratio d'approximation moyen : {invalid_average_approx_ratio:.2f}")
        print(f"Ratio d'approximation maximum : {invalid_max_approx_ratio:.2f}")
        print(f"Ratio d'approximation minimum : {invalid_min_approx_ratio:.2f}")
        print(f"Taille moyenne des graphes résultants après suppression des sommets : {avg_invalid_graph_size:.2f}")
        print(f"Taille maximale des graphes résultants : {max_invalid_graph_size}")
        print(f"Taille minimale des graphes résultants : {min_invalid_graph_size}")
        print(f"Ratio d'approximation moyen pour les sommets restants : {avg_invalid_new_approx_ratio:.2f}")
        print(f"Ratio d'approximation maximum pour les sommets restants : {invalid_new_max_approx_ratio:.2f}")
        print(f"Ratio d'approximation minimum pour les sommets restants : {invalid_new_min_approx_ratio:.2f}")
    else:
        print("Aucune couverture invalide n'a été proposée par le modèle.")

if __name__ == '__main__':
    main()
