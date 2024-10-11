import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from ..models import MLP, DeepSVDD, SemiDeepSVDD


class IndividualLearner:
    def __init__(self, feature_type, input_size):
        if feature_type == 'continuous':
            self.model = MLP(input_size, [256, 128, 32], 2)
        else:
            self.model = RandomForestClassifier()

    def fit(self, X, y):
        if isinstance(self.model, nn.Module):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            # PyTorch training loop here
        else:
            self.model.fit(X, y)

    def predict(self, X):
        if isinstance(self.model, nn.Module):
            return self.model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
        else:
            return self.model.predict(X)


class MetaLearner(nn.Module):
    def __init__(self, input_size):
        super(MetaLearner, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.model(x)


class ELAMD(nn.Module):
    def __init__(self, individual_learners, meta_learner):
        super(ELAMD, self).__init__()
        self.individual_learners = individual_learners
        self.meta_learner = meta_learner

    def forward(self, x):
        individual_outputs = []
        for learner in self.individual_learners:
            if isinstance(learner, nn.Module):
                output = learner(x)
            else:
                output = torch.tensor(learner.predict_proba(x.cpu().numpy()), device=x.device)
            individual_outputs.append(output)

        meta_input = torch.cat(individual_outputs, dim=1)
        return self.meta_learner(meta_input)

    def predict(self, X_list):
        meta_features = []
        for learner, X in zip(self.individual_learners, X_list):
            meta_features.append(learner.predict(X))
        meta_features = torch.tensor(np.concatenate(meta_features, axis=1), dtype=torch.float32)
        return self.meta_learner(meta_features).argmax(dim=1)


class AnomalyDetector:
    def __init__(self, method='iforest'):
        if method == 'iforest':
            self.model = IsolationForest()
        elif method == 'deep_svdd':
            self.model = DeepSVDD()  # 별도 구현 필요

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)


class ELAMD_AnomalyDetection(nn.Module):
    def __init__(self, individual_learners, meta_learner, anomaly_detector):
        super(ELAMD_AnomalyDetection, self).__init__()
        self.individual_learners = individual_learners
        self.meta_learner = meta_learner
        self.anomaly_detector = anomaly_detector

    def forward(self, x):
        individual_outputs = []
        for learner in self.individual_learners:
            if isinstance(learner, nn.Module):
                output = learner(x)
            else:
                output = torch.tensor(learner.predict_proba(x.cpu().numpy()), device=x.device)
            individual_outputs.append(output)

        meta_input = torch.cat(individual_outputs, dim=1)
        meta_output = self.meta_learner(meta_input)

        if isinstance(self.anomaly_detector, nn.Module):
            anomaly_output = self.anomaly_detector(meta_output)
        else:
            anomaly_output = torch.tensor(self.anomaly_detector.predict(meta_output.detach().cpu().numpy()), device=x.device)

        return meta_output, anomaly_output

    def predict(self, X_list):
        meta_features = []
        for learner, X in zip(self.individual_learners, X_list):
            meta_features.append(learner.predict(X))
        meta_features = np.concatenate(meta_features, axis=1)
        return self.anomaly_detector.predict(meta_features)