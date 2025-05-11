import torch
import numpy as np
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn

from aci import *
import torch
import networkx as nx
import matplotlib.pyplot as plt

def compute_vertex_cover_exact(A):
    """
    Version simple : on utilise une heuristique de couverture gloutonne comme vérité terrain.
    Pour une version exacte, il faudrait résoudre un ILP avec OR-Tools ou Pyomo.
    """
    N = A.shape[0]
    G = nx.from_numpy_array(A.numpy())
    cover = set()
    uncovered_edges = set(G.edges())
    while uncovered_edges:
        node_degrees = dict(G.degree())
        max_node = max(node_degrees, key=node_degrees.get)
        cover.add(max_node)
        G.remove_node(max_node)
        uncovered_edges = set(G.edges())
    y = torch.zeros(N)
    y[list(cover)] = 1.0
    return y

def visualize_cover(A, cover):
    G = nx.from_numpy_array(A.numpy())
    pos = nx.spring_layout(G)
    colors = ['red' if i in cover else 'lightgray' for i in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500)
    plt.show()

def generate_random_graph(n_nodes: int, density: float = 0.1):
    G = nx.gnp_random_graph(n_nodes, density)
    A = nx.to_numpy_array(G)
    A = torch.tensor(A, dtype=torch.float32)
    return A

class ACIEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(ACIEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.linear_A = nn.Linear(input_dim, embed_dim)
        self.linear_A_comp = nn.Linear(input_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, A):
        # A: [N, N] adjacency matrix
        A_comp = 1.0 - A - torch.eye(A.size(0), device=A.device)  # complement matrix

        h = F.relu(self.linear_A(A))  # node embeddings from A
        h_comp = F.relu(self.linear_A_comp(A_comp))  # from complement

        Q = self.q_proj(h)
        K = self.k_proj(h_comp)
        V = self.v_proj(h)

        scores = torch.matmul(Q, K.T) / np.sqrt(self.embed_dim)
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)

        return context  # [N, embed_dim]

class ACIDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(ACIDecoder, self).__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(0)  # batch dimension
        output, _ = self.lstm(x)
        scores = torch.sigmoid(self.output_layer(output)).squeeze(0).squeeze(-1)
        return scores  # [N]

class ACINet(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(ACINet, self).__init__()
        self.encoder = ACIEncoder(input_dim, embed_dim)
        self.decoder = ACIDecoder(embed_dim, hidden_dim)

    def forward(self, A):
        context = self.encoder(A)
        scores = self.decoder(context)
        return scores  # probabilities for each node

def construct_vc(A, scores):
    """
    A: [N, N] torch.Tensor, adjacency matrix
    scores: [N] torch.Tensor, predicted scores from ACI model
    """
    N = A.size(0)
    C = set()
    covered = torch.zeros_like(A)

    # Phase d’extension
    edges = (A > 0).nonzero(as_tuple=False)
    for u, v in edges:
        if covered[u, v] == 0:
            if scores[u] >= scores[v]:
                C.add(u.item())
                covered[u, :] = 1
                covered[:, u] = 1
            else:
                C.add(v.item())
                covered[v, :] = 1
                covered[:, v] = 1

    # Phase de réduction
    to_check = list(C)
    for v in to_check:
        temp_C = C - {v}
        temp_covered = torch.zeros_like(A)
        for u in temp_C:
            temp_covered[u, :] = 1
            temp_covered[:, u] = 1
        if (temp_covered * A).sum() == A.sum():  # toutes les arêtes couvertes
            C.remove(v)

    return sorted(list(C))

def train_model(model, graphs, labels, epochs=50, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for A, y_true in zip(graphs, labels):
            scores = model(A)  # predicted probability [N]
            loss = criterion(scores, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

def test_model(model, A):
    model.eval()
    with torch.no_grad():
        scores = model(A)
        cover = construct_vc(A, scores)
    return cover


if __name__ == "__main__":
    torch.manual_seed(0)

    A = generate_random_graph(n_nodes=20, density=0.2)
    y = compute_vertex_cover_exact(A)

    model = ACINet(input_dim=A.shape[0], embed_dim=32, hidden_dim=64)

    train_model(model, [A], [y], epochs=100)

    predicted_cover = test_model(model, A)

    print("Predicted cover:", predicted_cover)
    print("True cover:", list(torch.nonzero(y).squeeze().numpy()))
    visualize_cover(A, predicted_cover)