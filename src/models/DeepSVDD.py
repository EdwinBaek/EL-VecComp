import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepSVDD(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], output_size=1):
        super(DeepSVDD, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.ReLU())
            prev_size = hidden_size
        self.layers.append(nn.Linear(prev_size, output_size))

        self.center = None
        self.radius = 0

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, X, y=None, lr=0.001, epochs=100, batch_size=64, nu=0.1):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        X = torch.tensor(X, dtype=torch.float32)

        # Initialize center as mean of forward pass
        with torch.no_grad():
            self.center = self.forward(X).mean(dim=0)

        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch[0]
                optimizer.zero_grad()
                outputs = self.forward(batch)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

        # Compute final radius
        self.radius = self._get_radius(X, nu)

    def _get_radius(self, X, nu):
        dist = torch.sum((self.forward(X) - self.center) ** 2, dim=1)
        return np.quantile(dist.detach().numpy(), 1 - nu)

    def predict(self, X):
        self.eval()
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.forward(X)
            dist = torch.sum((outputs - self.center) ** 2, dim=1)
            scores = dist - self.radius
            predictions = (scores > 0).float()
        return predictions.numpy()