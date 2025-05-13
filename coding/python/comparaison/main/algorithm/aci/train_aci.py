import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from aci import ACINet, construct_vc
from tqdm import tqdm

# Dataset personnalisé
class VertexCoverDataset(Dataset):
    def __init__(self, folder):
        self.A_files = sorted([f for f in os.listdir(folder) if f.startswith("A_")])
        self.y_files = sorted([f for f in os.listdir(folder) if f.startswith("y_")])
        self.folder = folder

    def __len__(self):
        return len(self.A_files)

    def __getitem__(self, idx):
        A = torch.load(os.path.join(self.folder, self.A_files[idx]))
        y = torch.load(os.path.join(self.folder, self.y_files[idx]))

        N = A.shape[0]
        max_dim = 2000

        padded_A = torch.zeros((max_dim, max_dim), dtype=A.dtype)
        padded_y = torch.zeros(max_dim)

        padded_A[:N, :N] = A
        padded_y[:N] = y

        return padded_A, padded_y


# Entraînement
def train(model, train_loader, val_loader, epochs=100, lr=1e-4, save_path="aci_model.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for A, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            A, y = A[0], y[0]  # batch size = 1
            scores = model(A)
            loss = criterion(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for A, y in val_loader:
                A, y = A[0], y[0]
                scores = model(A)
                loss = criterion(scores, y)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Sauvegarde du modèle
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé sous {save_path}")

if __name__ == "__main__":
    torch.manual_seed(0)

    # Chargement des datasets
    train_dataset = VertexCoverDataset("data/train")
    val_dataset = VertexCoverDataset("data/val")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Déterminer input_dim maximal (pour padding ou initialisation large)
    input_dim = 2000  # taille max des graphes
    model = ACINet(input_dim=input_dim, embed_dim=32, hidden_dim=64)

    train(model, train_loader, val_loader, epochs=100, lr=1e-4)
