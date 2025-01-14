import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import pulp
from tqdm import tqdm
import concurrent.futures
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Définition du modèle GNN pour la couverture minimale de sommets
class GNNVertexCover(nn.Module):
    """
    Modèle de réseau de neurones de graphes (GNN) pour prédire si chaque nœud doit être inclus
    dans la couverture minimale de sommets.

    Paramètres:
    - in_channels: Nombre de caractéristiques en entrée par nœud.
    - hidden_channels: Nombre de canaux cachés pour les couches GCN.
    - num_layers: Nombre de couches GCN dans le modèle.

    Suggestions d'amélioration:
    - Expérimenter avec différentes architectures de GNN (par ex. GATConv, SAGEConv).
    - Ajuster le nombre de couches et de neurones pour améliorer les performances.
    - Ajouter des mécanismes d'attention pour mieux capturer les relations entre les nœuds.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super(GNNVertexCover, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 2))  # 2 classes : dans la couverture ou non

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

# Fonction pour résoudre le problème de couverture minimale de sommets avec PuLP
def solve_vertex_cover_pulp(G):
    """
    Calcule la couverture minimale de sommets du graphe G en utilisant la programmation linéaire.

    Paramètres:
    - G: Un graphe NetworkX.

    Retourne:
    - vertex_cover: Liste des nœuds faisant partie de la couverture minimale.

    Suggestions d'amélioration:
    - Utiliser des heuristiques pour accélérer le calcul sur de grands graphes.
    - Intégrer des solutions approximatives lorsque le calcul exact est trop long.
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
    - Degré du nœud.
    - Degré moyen des voisins.
    - Écart-type des degrés des voisins.
    - Assortativité locale.
    - Coefficient de clustering local.
    - Efficacité locale.
    - Similarité de Jaccard moyenne avec les voisins.
    - Centralité de proximité.
    - Centralité d'intermédiarité.
    - Score PageRank.
    - Cohésion locale.
    - Distance moyenne aux voisins.
    - Nombre de triangles.
    - Somme des degrés des voisins.
    - Nombre de voisins de degré 1.
    - Degré moyen du graphe.
    - Degré minimum du graphe.
    - Degré maximum du graphe.

    Suggestions d'amélioration:
    - Normaliser les caractéristiques pour améliorer la convergence du modèle.
    - Ajouter des caractéristiques globales du graphe, comme le coefficient de clustering moyen.
    - Enrichir les caractéristiques avec des mesures de centralité (par ex. centralité de proximité).
    """
    n_nodes = len(node_list)
    degrees = np.array([G.degree(node) for node in node_list])
    degrees_dict = {node: degree for node, degree in zip(node_list, degrees)}
    triangles = np.array([nx.triangles(G, node) for node in node_list])

    # Initialisations
    sum_degrees_neighbors = np.zeros(n_nodes)
    num_neighbors_degree_one = np.zeros(n_nodes)
    avg_neighbor_degree = np.zeros(n_nodes)
    std_neighbor_degree = np.zeros(n_nodes)
    assortativity = np.zeros(n_nodes)
    clustering_coeffs = nx.clustering(G, nodes=node_list)
    clustering_coeffs_array = np.array([clustering_coeffs[node] for node in node_list])
    local_efficiency = np.zeros(n_nodes)
    avg_jaccard_similarity = np.zeros(n_nodes)
    closeness_centrality = nx.closeness_centrality(G)
    closeness_centrality_array = np.array([closeness_centrality[node] for node in node_list])
    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness_centrality_array = np.array([betweenness_centrality[node] for node in node_list])
    pagerank_scores = nx.pagerank(G)
    pagerank_array = np.array([pagerank_scores[node] for node in node_list])
    local_cohesion = np.zeros(n_nodes)
    avg_distance_to_neighbors = np.zeros(n_nodes)

    mean_degree_graph = degrees.mean()
    min_degree_graph = degrees.min()
    max_degree_graph = degrees.max()

    for idx, node in enumerate(node_list):
        neighbors = list(G.neighbors(node))
        neighbor_degrees = [degrees_dict[neighbor] for neighbor in neighbors]
        sum_degrees_neighbors[idx] = sum(neighbor_degrees)
        num_neighbors_degree_one[idx] = sum(1 for deg in neighbor_degrees if deg == 1)
        if neighbor_degrees:
            avg_neighbor_degree[idx] = np.mean(neighbor_degrees)
            std_neighbor_degree[idx] = np.std(neighbor_degrees)
            assortativity[idx] = avg_neighbor_degree[idx] - degrees[idx]
        else:
            avg_neighbor_degree[idx] = 0
            std_neighbor_degree[idx] = 0
            assortativity[idx] = 0

        # Efficacité locale
        k = degrees[idx]
        if k >= 2:
            neighbors_subgraph = G.subgraph(neighbors)
            path_lengths = dict(nx.all_pairs_shortest_path_length(neighbors_subgraph))
            efficiency = 0
            pairs = 0
            for u in neighbors_subgraph:
                for v in neighbors_subgraph:
                    if u != v:
                        try:
                            d = path_lengths[u][v]
                            efficiency += 1 / d
                        except KeyError:
                            pass
                        pairs += 1
            if pairs > 0:
                local_efficiency[idx] = efficiency / pairs
            else:
                local_efficiency[idx] = 0
        else:
            local_efficiency[idx] = 0

        # Similarité de Jaccard moyenne avec les voisins
        node_neighbors_set = set(neighbors)
        jaccard_similarities = []
        for neighbor in neighbors:
            neighbor_neighbors_set = set(G.neighbors(neighbor))
            intersection = node_neighbors_set & neighbor_neighbors_set
            union = node_neighbors_set | neighbor_neighbors_set
            if len(union) > 0:
                jaccard_sim = len(intersection) / len(union)
                jaccard_similarities.append(jaccard_sim)
        if jaccard_similarities:
            avg_jaccard_similarity[idx] = np.mean(jaccard_similarities)
        else:
            avg_jaccard_similarity[idx] = 0

        # Cohésion locale
        if len(neighbors) >= 2:
            subgraph = G.subgraph(neighbors)
            density = nx.density(subgraph)
        else:
            density = 0
        local_cohesion[idx] = density

        # Distance moyenne aux voisins
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=2)
        distances = [length for neighbor, length in lengths.items() if neighbor in neighbors]
        if distances:
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0
        avg_distance_to_neighbors[idx] = avg_distance

    features = []
    for i in range(n_nodes):
        feature_vector = [
            degrees[i],                       # Degré du nœud
            avg_neighbor_degree[i],           # Degré moyen des voisins
            std_neighbor_degree[i],           # Écart-type des degrés des voisins
            assortativity[i],                 # Assortativité locale
            clustering_coeffs_array[i],       # Coefficient de clustering local
            local_efficiency[i],              # Efficacité locale
            avg_jaccard_similarity[i],        # Similarité de Jaccard moyenne avec les voisins
            closeness_centrality_array[i],    # Centralité de proximité
            betweenness_centrality_array[i],  # Centralité d'intermédiarité
            pagerank_array[i],                # Score PageRank
            local_cohesion[i],                # Cohésion locale
            avg_distance_to_neighbors[i],     # Distance moyenne aux voisins
            triangles[i],                     # Nombre de triangles
            sum_degrees_neighbors[i],         # Somme des degrés des voisins
            num_neighbors_degree_one[i],      # Nombre de voisins de degré 1
            mean_degree_graph,                # Degré moyen du graphe
            min_degree_graph,                 # Degré minimum du graphe
            max_degree_graph                  # Degré maximum du graphe
        ]
        features.append(feature_vector)

    features = np.array(features)

    # Normalisation des caractéristiques
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    x = torch.tensor(features, dtype=torch.float)
    return x

# Fonction pour traiter un graphe et créer un objet Data
def process_graph(g6_str):
    """
    Traite un graphe au format graph6 et retourne un objet Data pour PyTorch Geometric.

    Paramètres:
    - g6_str: Chaîne de caractères du graphe au format graph6.

    Retourne:
    - data: Objet Data contenant les features, edge_index et labels.
    """
    G = nx.from_graph6_bytes(g6_str.encode())

    node_list = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # Calcul des caractéristiques
    x = compute_custom_features(G, node_list)

    # Calcul de la couverture minimale de sommets
    vertex_cover = solve_vertex_cover_pulp(G)

    # Création des labels
    y = torch.zeros(len(node_list), dtype=torch.long)
    for node in vertex_cover:
        y[node_to_idx[node]] = 1

    # Création de edge_index
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Création de l'objet Data
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# Fonction pour charger les graphes à partir d'un fichier .g6 et préparer les données
def load_graphs_with_features(file_path, num_workers=4):
    """
    Charge les graphes à partir d'un fichier au format graph6 (.g6), calcule les caractéristiques
    pour chaque nœud, et crée les objets Data pour PyTorch Geometric.

    Paramètres:
    - file_path: Chemin du fichier contenant les graphes au format .g6.
    - num_workers: Nombre de processus à utiliser pour le chargement en parallèle.

    Retourne:
    - graphs: Liste des objets Data prêts pour l'entraînement.
    """
    graphs = []
    
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f]
    
    # Utilisation de ProcessPoolExecutor pour le traitement parallèle avec map
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Utilisation de tqdm pour afficher la progression
        results = list(tqdm(executor.map(process_graph, lines), total=len(lines), desc="Traitement des graphes"))

    # Filtrer les résultats pour exclure les erreurs éventuelles
    for result in results:
        if result is not None:
            graphs.append(result)
        else:
            print("Un graphe n'a pas pu être traité et a été ignoré.")

    return graphs


# Fonction d'entraînement du modèle
def train(model, loader, optimizer, criterion, device, epochs=50):
    """
    Entraîne le modèle GNN sur les données fournies.

    Paramètres:
    - model: Le modèle GNN à entraîner.
    - loader: DataLoader contenant les données d'entraînement.
    - optimizer: Optimiseur pour la mise à jour des poids du modèle.
    - criterion: Fonction de perte.
    - device: Device sur lequel exécuter le modèle (cpu ou cuda).
    - epochs: Nombre d'époques pour l'entraînement.

    Retourne:
    - train_losses: Liste des pertes d'entraînement par époque.
    - train_accuracies: Liste des exactitudes d'entraînement par époque.

    Suggestions d'amélioration:
    - Implémenter un scheduler pour ajuster le taux d'apprentissage.
    - Ajouter une régularisation (par ex. dropout) pour éviter le surapprentissage.
    - Enregistrer les poids du modèle à chaque époque pour conserver les meilleures performances.
    """
    train_losses = []
    train_accuracies = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        print(f'Époque {epoch+1}/{epochs}, Perte moyenne : {avg_loss:.4f}, Exactitude : {accuracy*100:.2f}%')

    return train_losses, train_accuracies

# Fonction d'évaluation du modèle
def test(model, loader, device):
    """
    Évalue les performances du modèle sur les données de test.

    Paramètres:
    - model: Le modèle GNN à évaluer.
    - loader: DataLoader contenant les données de test.
    - device: Device sur lequel exécuter le modèle (cpu ou cuda).

    Retourne:
    - test_accuracy: Exactitude du modèle sur l'ensemble de test.
    - precision: Précision du modèle.
    - recall: Rappel du modèle.
    - f1: Score F1 du modèle.
    - confusion_mat: Matrice de confusion.
    - all_labels: Labels réels.
    - all_preds: Prédictions du modèle.

    Suggestions d'amélioration:
    - Utiliser une validation croisée pour une évaluation plus robuste.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(batch.y.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_accuracy = (all_preds == all_labels).sum().item() / all_labels.size(0)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    confusion_mat = confusion_matrix(all_labels, all_preds)

    print(f'Exactitude sur le test : {test_accuracy*100:.2f}%')
    print(f'Précision : {precision*100:.2f}%')
    print(f'Rappel : {recall*100:.2f}%')
    print(f'Score F1 : {f1*100:.2f}%')

    return test_accuracy, precision, recall, f1, confusion_mat, all_labels, all_preds

# Fonction principale
def main():
    """
    Fonction principale du script. Gère le chargement des données, l'entraînement et l'évaluation du modèle.

    Suggestions d'amélioration:
    - Ajouter des arguments de ligne de commande pour personnaliser les hyperparamètres.
    - Gérer les cas où le fichier de graphes est volumineux en chargeant les données par lots.
    """
    # Analyse des arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Entraînement d\'un GNN pour la couverture minimale de sommets.')
    parser.add_argument('--graph_file', type=str, default='graph8c.g6', help='Chemin du fichier de graphes au format .g6')
    parser.add_argument('--epochs', type=int, default=50, help='Nombre d\'époques pour l\'entraînement')
    parser.add_argument('--batch_size', type=int, default=32, help='Taille des batchs pour l\'entraînement')
    parser.add_argument('--hidden_channels', type=int, default=16, help='Nombre de canaux cachés dans le modèle GNN')
    parser.add_argument('--num_workers', type=int, default=4, help='Nombre de processus pour le chargement des données')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Taux d\'apprentissage pour l\'optimiseur')
    parser.add_argument('--num_layers', type=int, default=3, help='Nombre de couches GCN dans le modèle')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Répertoire pour sauvegarder les résultats')
    parser.add_argument('--train_all', action='store_true', help='Utiliser toutes les données pour l\'entraînement, sans ensemble de test')
    parser.add_argument('--load_model_path', type=str, default=None, help='Chemin vers le modèle pré-entraîné à charger')
    parser.add_argument('--save_model_path', type=str, default=None, help='Chemin pour sauvegarder le modèle entraîné')
    args = parser.parse_args()

    # Création du répertoire de sortie si nécessaire
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Détection du device (CPU ou GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device : {device}")

    # Charger les graphes et préparer les données
    graphs = load_graphs_with_features(args.graph_file, num_workers=args.num_workers)

    # Vérifier si des graphes ont été chargés
    if not graphs:
        print("Aucun graphe n'a été chargé. Vérifiez le fichier d'entrée.")
        return

    # Détermination du nombre de caractéristiques en entrée
    in_channels = graphs[0].num_node_features

    # Division des données en ensemble d'entraînement et de test
    if args.train_all:
        train_graphs = graphs
        test_graphs = []
    else:
        train_size = int(0.8 * len(graphs))
        train_graphs = graphs[:train_size]
        test_graphs = graphs[train_size:]

    # Création des DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    if test_graphs:
        test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
    else:
        test_loader = None

    # Initialisation du modèle
    if args.load_model_path:
        checkpoint = torch.load(args.load_model_path, map_location=device)
        # Vérifier que in_channels correspond
        if in_channels != checkpoint['in_channels']:
            raise ValueError("Le nombre de caractéristiques en entrée ne correspond pas entre les données et le modèle sauvegardé.")
        model = GNNVertexCover(
            in_channels=checkpoint['in_channels'],
            hidden_channels=checkpoint['hidden_channels'],
            num_layers=checkpoint['num_layers']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle chargé depuis {args.load_model_path}")
    else:
        model = GNNVertexCover(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers
        ).to(device)

    # Initialisation de l'optimiseur et de la fonction de perte
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()  # Utiliser NLLLoss car le modèle retourne log_softmax

    # Entraînement du modèle
    train_losses, train_accuracies = train(model, train_loader, optimizer, criterion, device, epochs=args.epochs)

    # Évaluation du modèle
    if test_loader:
        test_accuracy, precision, recall, f1, confusion_mat, all_labels, all_preds = test(model, test_loader, device)

        # Enregistrement des métriques d'entraînement pour tracer des courbes d'apprentissage
        epochs_range = range(1, args.epochs + 1)
        plt.figure()
        plt.plot(epochs_range, train_losses, label='Perte d\'entraînement')
        plt.xlabel('Époques')
        plt.ylabel('Perte')
        plt.title('Courbe de perte d\'entraînement')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'courbe_perte.png'))
        plt.close()

        plt.figure()
        plt.plot(epochs_range, train_accuracies, label='Exactitude d\'entraînement')
        plt.xlabel('Époques')
        plt.ylabel('Exactitude')
        plt.title('Courbe d\'exactitude d\'entraînement')
        plt.legend()
        plt.savefig(os.path.join(args.output_dir, 'courbe_exactitude.png'))
        plt.close()

        # Affichage et sauvegarde de la matrice de confusion
        plt.figure()
        labels = ['Non couvert', 'Couvert']
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=labels)
        disp.plot()
        plt.title('Matrice de confusion')
        plt.savefig(os.path.join(args.output_dir, 'matrice_confusion.png'))
        plt.close()

        # Sauvegarde des métriques dans un fichier texte
        with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
            f.write(f'Exactitude sur le test : {test_accuracy*100:.2f}%\n')
            f.write(f'Précision : {precision*100:.2f}%\n')
            f.write(f'Rappel : {recall*100:.2f}%\n')
            f.write(f'Score F1 : {f1*100:.2f}%\n')
    else:
        print("Aucun ensemble de test fourni, saut de l'évaluation.")

    # Sauvegarde du modèle entraîné
    if args.save_model_path:
        model_path = args.save_model_path
        torch.save({
            'model_state_dict': model.state_dict(),
            'in_channels': in_channels,
            'hidden_channels': args.hidden_channels,
            'num_layers': args.num_layers
        }, model_path)
        print(f'Modèle sauvegardé à {model_path}')

if __name__ == '__main__':
    main()

# TODO : ajouter kfold
# TODO : tester différentes architectures
