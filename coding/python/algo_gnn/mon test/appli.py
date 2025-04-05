import random
import networkx as nx
import pulp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


#########################################
# 1) Fonctions utilitaires pour les graphes
#########################################

def solve_min_vertex_cover(g: nx.Graph):
    """
    Résout exactement le MVC d'un graphe en utilisant PuLP.
    Retourne une liste binaire de longueur n (nombre de sommets)
    où 1 signifie que le sommet fait partie de la couverture optimale.
    """
    nodes = list(g.nodes())
    x_vars = pulp.LpVariable.dicts('x', nodes, cat=pulp.LpBinary)
    problem = pulp.LpProblem("MinimumVertexCover", pulp.LpMinimize)
    problem += pulp.lpSum([x_vars[i] for i in nodes]), "MinimizeCoverSize"
    for (u, v) in g.edges():
        problem += x_vars[u] + x_vars[v] >= 1
    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    solution = [int(x_vars[i].varValue) for i in nodes]
    return solution


def generate_random_graph(n=7, p=0.3, seed=None):
    """
    Génère un graphe aléatoire de type Erdős–Rényi G(n, p).
    Si le graphe généré est vide (aucune arête), il est régénéré.
    """
    if seed is not None:
        random.seed(seed)
    g = nx.erdos_renyi_graph(n, p)
    if g.number_of_edges() == 0:
        return generate_random_graph(n, p, seed + 1 if seed is not None else None)
    return g


def compute_vertex_features(g: nx.Graph):
    """
    Calcule un vecteur de caractéristiques pour chaque sommet du graphe.
    Caractéristiques calculées :
      - Degré
      - Coefficient de clustering
      - Centralité de proximité
      - Centralité d'intermédiarité
      - (Optionnel) Centralité par vecteur propre
    Retourne un tableau NumPy de dimension (n_sommets, num_features)
    """
    nodes = list(g.nodes())
    n = len(nodes)

    # Calcul de plusieurs propriétés
    degrees = np.array([g.degree(node) for node in nodes]).reshape(-1, 1)
    clustering = np.array([nx.clustering(g, node) for node in nodes]).reshape(-1, 1)
    closeness_dict = nx.closeness_centrality(g)
    closeness = np.array([closeness_dict[node] for node in nodes]).reshape(-1, 1)
    betweenness_dict = nx.betweenness_centrality(g)
    betweenness = np.array([betweenness_dict[node] for node in nodes]).reshape(-1, 1)
    # Pour des graphes petits, on peut calculer aussi l'eigenvector centrality
    try:
        eigenvector_dict = nx.eigenvector_centrality(g, max_iter=7000)
        eigenvector = np.array([eigenvector_dict[node] for node in nodes]).reshape(-1, 1)
    except Exception:
        eigenvector = np.zeros((n, 1))

    # Concaténation des caractéristiques
    features = np.hstack([degrees, clustering, closeness, betweenness, eigenvector])
    return features


#########################################
# 2) Création du dataset
#########################################

class VertexDataset(Dataset):
    """
    Dataset construit à partir de plusieurs graphes.
    Chaque exemple correspond à un sommet avec son vecteur de caractéristiques et son label (1 si dans le MVC, 0 sinon).
    """

    def __init__(self, num_graphs=7000, n_nodes=7, p=0.3):
        self.X = []  # features de sommets
        self.y = []  # labels (0 ou 1)

        for i in tqdm(range(num_graphs), desc="Génération des graphes"):
            g = generate_random_graph(n_nodes, p, seed=i)
            labels = solve_min_vertex_cover(g)  # liste binaire par sommet
            features = compute_vertex_features(g)  # tableau (n_nodes x num_features)

            # Pour chaque sommet, ajouter l'exemple
            for j in range(len(g.nodes())):
                self.X.append(features[j])
                self.y.append(labels[j])

        # Conversion en tableaux NumPy
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 1)

        # Normalisation des features
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

        # Conversion en tenseurs torch
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#########################################
# 3) Définition du petit réseau de neurones (MLP)
#########################################

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # sortie unique pour la classification binaire

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


#########################################
# 4) Entraînement et évaluation
#########################################

def train_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


#########################################
# 5) Application principale
#########################################

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Utilisation du device :", device)

    # Paramètres du dataset et de l'entraînement
    num_graphs = 1000  # nombre de graphes pour générer le dataset
    n_nodes = 7  # taille des graphes
    p = 0.3  # probabilité d'arête
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.01 # taux d'apprentissage

    # Création du dataset
    dataset = VertexDataset(num_graphs=num_graphs, n_nodes=n_nodes, p=p)
    # Diviser le dataset en train/test (80/20)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instanciation du modèle
    input_dim = dataset.X.shape[1]
    model = SimpleMLP(input_dim=input_dim, hidden_dim=16).to(device)
    criterion = nn.BCELoss()  # pour sortie sigmoïde et étiquette binaire
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement
    print("\nDébut de l'entraînement :")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        print(
            f"Époque {epoch}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    # --- Démo sur un nouveau graphe ---
    print("\n=== Démo sur un nouveau graphe ===")
    demo_g = generate_random_graph(n=n_nodes, p=p)
    optimal_cover = solve_min_vertex_cover(demo_g)
    print("Solution optimale (MVC) :", optimal_cover)
    print("Taille VC Optimal:", sum(optimal_cover))

    # Calcul des features pour tous les sommets du graphe
    demo_features = compute_vertex_features(demo_g)
    # Utiliser le scaler du dataset pour normaliser les features
    demo_features = dataset.scaler.transform(demo_features)
    demo_features = torch.tensor(demo_features, dtype=torch.float).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(demo_features)
    preds = (outputs > 0.5).int().cpu().numpy().flatten()
    print("Prédictions du modèle :", preds)
    print("Taille Couverture Prédite:", int(sum(preds)))

    # Affichage comparatif par sommet
    nodes = list(demo_g.nodes())
    for i, node in enumerate(nodes):
        print(f"Sommet {node:2d} | Label Optimal: {optimal_cover[i]} | Prédiction: {preds[i]}")

if __name__ == '__main__':
    main()
