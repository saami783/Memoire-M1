import os
import random
import networkx as nx
import pulp
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


# ============ 1) Solveur EXACT du Minimum Vertex Cover ============
def solve_min_vertex_cover(g: nx.Graph):
    """
    Résout exactement le MVC d'un graphe networkx grâce à PuLP.
    Retourne une liste binaire de longueur n (nb de nœuds)
    où 1 signifie que le nœud fait partie de la couverture.
    """
    # On récupère les nœuds sous forme d'une liste fixe pour avoir un index stable
    nodes = list(g.nodes())
    # On crée les variables binaires x_i
    x = pulp.LpVariable.dicts('x', nodes, cat=pulp.LpBinary)

    # Modèle de minimisation
    problem = pulp.LpProblem("MinimumVertexCover", pulp.LpMinimize)

    # Fonction objectif: minimiser la somme des x_i
    problem += pulp.lpSum([x[i] for i in nodes]), "MinimizeCoverSize"

    # Contraintes: pour chaque arête (u, v), x_u + x_v >= 1
    for (u, v) in g.edges():
        problem += x[u] + x[v] >= 1

    # Résolution
    problem.solve(pulp.PULP_CBC_CMD(msg=0))

    # Récupérer la solution
    solution = [int(x[i].varValue) for i in nodes]
    return solution


# ============ 2) Génération d'un graphe et features ============
def generate_random_graph(n=15, p=0.2, seed=None):
    """
    Génère un graphe aléatoire Erdős–Rényi G(n, p).
    """
    if seed is not None:
        random.seed(seed)
    g = nx.erdos_renyi_graph(n, p)

    # Éventuellement on évite un graphe totalement isolé
    if g.number_of_edges() == 0:
        return generate_random_graph(n, p, seed + 1 if seed is not None else None)
    return g


def get_node_features(g: nx.Graph):
    """
    Calcule de nombreuses features pour chaque nœud du graphe g.

    Les caractéristiques extraites (pour chaque nœud) incluent :
      - Degré
      - Coefficient de clustering local
      - Degré moyen des voisins
      - Betweenness centrality
      - Closeness centrality
      - Eigenvector centrality (si convergent)
      - PageRank
      - Nombre de triangles
      - Core number (k-core)
      - Harmonic centrality
      - Load centrality (si disponible)
      - Eccentricity (si le graphe est connexe)
      - Efficacité locale (moyenne des inverses des distances dans le sous-graphe des voisins)
      - Somme des degrés des voisins
      - Nombre de voisins de degré 1
      - Coefficient de clustering moyen des voisins
    Retourne un tenseur torch de dimension (nombre_de_noeuds, nombre_de_features).
    """
    nodes = list(g.nodes())
    n = len(nodes)

    degrees = np.array([g.degree(node) for node in nodes]).reshape(-1, 1)
    clustering = np.array([nx.clustering(g, node) for node in nodes]).reshape(-1, 1)
    avg_neighbor_deg = np.array([nx.average_neighbor_degree(g)[node] for node in nodes]).reshape(-1, 1)
    betweenness = np.array([nx.betweenness_centrality(g)[node] for node in nodes]).reshape(-1, 1)
    closeness = np.array([nx.closeness_centrality(g)[node] for node in nodes]).reshape(-1, 1)

    try:
        eigen = np.array([nx.eigenvector_centrality(g, max_iter=1000)[node] for node in nodes]).reshape(-1, 1)
    except Exception:
        eigen = np.zeros((n, 1))

    pagerank = np.array([nx.pagerank(g)[node] for node in nodes]).reshape(-1, 1)
    triangles = np.array([nx.triangles(g, node) for node in nodes]).reshape(-1, 1)
    core_number = np.array([nx.core_number(g)[node] for node in nodes]).reshape(-1, 1)
    harmonic = np.array([nx.harmonic_centrality(g)[node] for node in nodes]).reshape(-1, 1)

    try:
        load_cent = np.array([nx.load_centrality(g)[node] for node in nodes]).reshape(-1, 1)
    except Exception:
        load_cent = np.zeros((n, 1))

    try:
        eccentricity = np.array([nx.eccentricity(g, v=node) for node in nodes]).reshape(-1, 1)
    except Exception:
        eccentricity = np.zeros((n, 1))

    local_efficiency = []
    for node in nodes:
        neighbors = list(g.neighbors(node))
        if len(neighbors) > 1:
            subg = g.subgraph(neighbors)
            eff = 0
            count = 0
            lengths = dict(nx.all_pairs_shortest_path_length(subg))
            for u in neighbors:
                for v in neighbors:
                    if u != v and v in lengths[u]:
                        eff += 1.0 / lengths[u][v]
                        count += 1
            local_efficiency.append(eff / count if count > 0 else 0)
        else:
            local_efficiency.append(0)
    local_efficiency = np.array(local_efficiency).reshape(-1, 1)

    sum_neighbor_deg = []
    for node in nodes:
        neighs = list(g.neighbors(node))
        sum_deg = sum(g.degree(n) for n in neighs)
        sum_neighbor_deg.append(sum_deg)
    sum_neighbor_deg = np.array(sum_neighbor_deg).reshape(-1, 1)

    num_deg_one = []
    for node in nodes:
        neighs = list(g.neighbors(node))
        count = sum(1 for n in neighs if g.degree(n) == 1)
        num_deg_one.append(count)
    num_deg_one = np.array(num_deg_one).reshape(-1, 1)

    avg_neighbor_clustering = []
    for node in nodes:
        neighs = list(g.neighbors(node))
        if neighs:
            avg_clust = np.mean([nx.clustering(g, n) for n in neighs])
        else:
            avg_clust = 0
        avg_neighbor_clustering.append(avg_clust)
    avg_neighbor_clustering = np.array(avg_neighbor_clustering).reshape(-1, 1)

    features = np.hstack([
        degrees,
        clustering,
        avg_neighbor_deg,
        betweenness,
        closeness,
        eigen,
        pagerank,
        triangles,
        core_number,
        harmonic,
        load_cent,
        eccentricity,
        local_efficiency,
        sum_neighbor_deg,
        num_deg_one,
        avg_neighbor_clustering
    ])

    return torch.tensor(features, dtype=torch.float)


def build_data_from_nx(g: nx.Graph, cover_label: list):
    """
    Construit un Data (PyTorch Geometric) à partir d'un graphe networkx
    et du label binaire (cover_label).
    """
    nodes = list(g.nodes())
    node_idx_map = {node: i for i, node in enumerate(nodes)}

    edges = list(g.edges())
    edge_index = torch.tensor([[node_idx_map[u], node_idx_map[v]]
                               for (u, v) in edges], dtype=torch.long)
    # On double pour graphes non orientés
    edge_index = torch.cat([edge_index, edge_index.flip(dims=[1])], dim=0).T

    x = get_node_features(g)  # [n, num_features]
    y = torch.tensor(cover_label, dtype=torch.float)  # [n] (binaire)
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


# ============ 3) Dataset PyG ============
class VertexCoverDataset(Dataset):
    def __init__(self, n_graphs=1000, n_nodes=7, p=0.3):
        super().__init__()
        self.data_list = []

        for i in tqdm(range(n_graphs), desc="Génération des graphes"):
            g = generate_random_graph(n_nodes, p, seed=i)
            cover = solve_min_vertex_cover(g)
            data = build_data_from_nx(g, cover)
            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# ============ 4) GCN (modèle) ============
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Première couche + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Deuxième couche + Sigmoid pour binaire
        x = self.conv2(x, edge_index)
        x = torch.sigmoid(x)
        return x


# ============ 5) Fonctions d'entraînement & évaluation ============
def train_gnn(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data.x, data.edge_index)
        out = out.view(-1)
        y = data.y.view(-1)

        loss = F.binary_cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calcul de la précision
        preds = (out > 0.5).float()  # [0 ou 1]
        correct = (preds == y).sum().item()
        total_correct += correct
        total_nodes += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_nodes
    return avg_loss, accuracy


def eval_gnn(model, loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_nodes = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            out = out.view(-1)
            y = data.y.view(-1)

            loss = F.binary_cross_entropy(out, y)
            total_loss += loss.item()

            preds = (out > 0.5).float()
            correct = (preds == y).sum().item()
            total_correct += correct
            total_nodes += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * total_correct / total_nodes
    return avg_loss, accuracy


# ============ 6) Main avec apprentissage continu + barres de progression ============
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Utilisation du device :", device)

    # Paramètres
    num_train_graphs = 10000
    num_test_graphs = 100
    num_epochs = 40
    model_path = "model_checkpoint.pth"

    # -- Création du dataset
    train_dataset = VertexCoverDataset(n_graphs=num_train_graphs, n_nodes=7, p=0.3)
    test_dataset = VertexCoverDataset(n_graphs=num_test_graphs, n_nodes=70, p=0.2)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # -- Instancier le modèle
    model = GCN(in_channels=16, hidden_channels=16, out_channels=1).to(device)

    # -- Apprentissage continu : si un checkpoint existe, on le charge
    if os.path.exists(model_path):
        print(f"\n==> Chargement du modèle existant : {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("\n==> Pas de modèle existant, initialisation d'un nouveau GNN.")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # -- Entraînement
    print(f"\nEntraînement sur {num_epochs} époques...\n")

    for epoch in range(1, num_epochs+1):
        # On peut ajouter une barre de progression sur l'epoch,
        # mais si le dataset n'est pas très grand, ça peut être moins utile.
        train_loss, train_acc = train_gnn(model, train_loader, optimizer, device)
        test_loss, test_acc   = eval_gnn(model, test_loader, device)

        print(f"Époque {epoch}/{num_epochs}, "
              f"Loss train: {train_loss:.4f}, Acc train: {train_acc:.2f}% | "
              f"Loss test: {test_loss:.4f}, Acc test: {test_acc:.2f}%")

    # -- Sauvegarde du modèle pour réutilisation ultérieure
    torch.save(model.state_dict(), model_path)
    print(f"\nModèle sauvegardé dans {model_path}")

    # -- Démo sur un graphe "hors dataset"
    demo_g = generate_random_graph(n=7, p=0.2)
    cover_opt = solve_min_vertex_cover(demo_g)
    data_demo = build_data_from_nx(demo_g, cover_opt).to(device)

    model.eval()
    with torch.no_grad():
        out_demo = model(data_demo.x, data_demo.edge_index).view(-1)

    predicted = (out_demo > 0.5).int().cpu().numpy()
    ground_truth = cover_opt

    print("\nGraphe de démo (hors dataset) :")
    print("   Nœud | Label Optimal | Prédiction NN")
    for i in range(len(predicted)):
        print(f"   {i:4d} | {ground_truth[i]:13d} | {predicted[i]:14d}")

    print("Taille VC Optimal:", sum(cover_opt))
    print("Taille Couverture Prédite:", sum(predicted))


if __name__ == '__main__':
    main()
