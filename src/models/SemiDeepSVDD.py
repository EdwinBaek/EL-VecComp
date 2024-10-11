import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class SemiDeepSVDD(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], output_size=1):
        super(SemiDeepSVDD, self).__init__()
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

    def fit(self, X, y, lr=0.001, epochs=100, batch_size=64, nu=0.1, beta=1.0):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Initialize center as mean of forward pass of normal samples
        normal_mask = y == 0
        with torch.no_grad():
            self.center = self.forward(X[normal_mask]).mean(dim=0)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)

                normal_mask = batch_y == 0
                abnormal_mask = batch_y == 1

                loss_normal = torch.mean(dist[normal_mask])
                loss_abnormal = torch.mean(torch.max(torch.zeros_like(dist[abnormal_mask]),
                                                     self.radius - dist[abnormal_mask]))

                loss = loss_normal + beta * loss_abnormal
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}')

        # Compute final radius
        self.radius = self._get_radius(X[normal_mask], nu)

    def _get_radius(self, X, nu):
        dist = torch.sum((self.forward(X) - self.center) ** 2, dim=1)
        return np.quantile(dist.detach().numpy(), 1 - nu)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
            dist = torch.sum((outputs - self.center) ** 2, dim=1)
            scores = dist - self.radius
            predictions = (scores > 0).float()
        return predictions