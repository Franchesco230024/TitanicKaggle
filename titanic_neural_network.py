import torch.nn as nn

class TitanicNeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.4),

            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(16, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        return self.network(x).squeeze()

