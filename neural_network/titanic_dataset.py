import numpy as np
import torch
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    """PyTorch dataset wrapping feature array and optional labels."""

    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(np.array(X))
        self.y = torch.FloatTensor(np.array(y)) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

