import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

# === Composants du modèle ACI ===

class ACIEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_A = nn.Linear(input_dim, embed_dim)
        self.linear_A_comp = nn.Linear(input_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, A):
        A_comp = 1.0 - A - torch.eye(A.size(0), device=A.device)
        h = F.relu(self.linear_A(A))
        h_comp = F.relu(self.linear_A_comp(A_comp))
        Q = self.q_proj(h)
        K = self.k_proj(h_comp)
        V = self.v_proj(h)
        scores = torch.matmul(Q, K.T) / np.sqrt(self.embed_dim)
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

class ACIDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(0)
        output, _ = self.lstm(x)
        return torch.sigmoid(self.output_layer(output)).squeeze(0).squeeze(-1)

class ACINet(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = ACIEncoder(input_dim, embed_dim)
        self.decoder = ACIDecoder(embed_dim, hidden_dim)

    def forward(self, A):
        context = self.encoder(A)
        return self.decoder(context)

# === Utilitaires ===

def construct_vc(A, scores):
    C = set()
    covered = torch.zeros_like(A)
    for u, v in (A > 0).nonzero(as_tuple=False):
        if covered[u, v] == 0:
            chosen = u.item() if scores[u] >= scores[v] else v.item()
            C.add(chosen)
            covered[chosen, :] = 1
            covered[:, chosen] = 1
    return sorted(C)

def compute_vertex_cover_exact(A):
    G = nx.from_numpy_array(A.numpy())
    cover = set()
    uncovered_edges = set(G.edges())
    while uncovered_edges:
        degrees = dict(G.degree())
        max_node = max(degrees, key=degrees.get)
        cover.add(max_node)
        G.remove_node(max_node)
        uncovered_edges = set(G.edges())
    y = torch.zeros(A.shape[0])
    y[list(cover)] = 1.0
    return y

def compute_coverage_rate(A, cover):
    covered = torch.zeros_like(A)
    for v in cover:
        covered[v, :] = 1
        covered[:, v] = 1
    total = (A > 0).sum().item() / 2
    covered_edges = ((A * covered) > 0).sum().item() / 2
    return covered_edges / total if total > 0 else 1.0

def compute_f1(y_true, y_pred):
    return {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred)
    }

# === Wrapper principal ===

def aci(G: nx.Graph):
    torch.manual_seed(0)

    A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    max_nodes = 2000
    A_padded = torch.zeros((max_nodes, max_nodes))
    A_padded[:A.shape[0], :A.shape[1]] = A

    model = ACINet(input_dim=max_nodes, embed_dim=32, hidden_dim=64)
    model.load_state_dict(torch.load("main/algorithm/aci/aci_model.pt"))
    model.eval()

    with torch.no_grad():
        scores = model(A_padded)[:A.shape[0]]
        predicted_cover = construct_vc(A, scores)

    y = compute_vertex_cover_exact(A)
    predicted_size = len(predicted_cover)
    optimal_size = int(y.sum().item())
    rel_diff = (100 * (predicted_size - optimal_size) / optimal_size) if optimal_size > 0 else 0

    y_pred = torch.zeros_like(y)
    y_pred[predicted_cover] = 1.0
    coverage = compute_coverage_rate(A, predicted_cover)
    metrics = compute_f1(y.numpy(), y_pred.numpy())

    # print("✅ Predicted cover:", predicted_cover)
    # print("✅ True cover:", list(torch.nonzero(y).squeeze().numpy()))
    # print(f"✅ Taille de la couverture prédite : {predicted_size}")
    # print(f"✅ Taille de la couverture optimale : {optimal_size}")
    # print(f"📏 Écart relatif : {rel_diff:.2f}%")
    # print(f"🎯 Coverage rate: {coverage * 100:.2f}%")
    # print(f"🎯 F1-score: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")

    return predicted_cover

# === Exemple d'utilisation ===
if __name__ == "__main__":
    # Exemple : graphe aléatoire
    G = nx.gnp_random_graph(20, 0.2, seed=42)
    aci(G)
