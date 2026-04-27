from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch

from neural_network import TitanicNeuralNetwork, TitanicDataset


class NeuralNetworkLearning:
    """Trainer for TitanicNeuralNetwork: handles scaling, train/val split, training loop, and inference."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.score = None

    def fit(self, X_train, y_train):
        """Train the network, store roc_auc on validation split in self.score, return fitted model."""
        X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        X_train_nn = self.scaler.fit_transform(X_train_nn)
        X_val = self.scaler.transform(X_val_nn)

        train_dataset = TitanicDataset(X_train_nn, y_train_nn.values)
        val_dataset = TitanicDataset(X_val, y_val_nn.values)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model = TitanicNeuralNetwork(input_dim=X_train_nn.shape[1]).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(21):
            model.train()
            train_loss = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        model.eval()
        all_preds = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch)
                all_preds.append(pred.cpu())

        all_preds = torch.cat(all_preds).numpy()
        y_val_np = y_val_nn.values

        self.score = roc_auc_score(y_val_np, all_preds)
        print(f"Neural network cv score (roc_auc): {self.score:.4f}")

        return model

    def predict(self, model, X_test):
        """Return binary predictions (0/1) for X_test using the fitted scaler."""
        X_test_scaled = self.scaler.transform(X_test)

        test_dataset = TitanicDataset(X_test_scaled)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        model.eval()
        predictions = []

        with torch.no_grad():
            for X_batch in test_loader:
                X_batch = X_batch.to(self.device)
                pred = model(X_batch)
                predictions.extend((pred > 0.5).int().cpu().numpy())

        return predictions
