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


class ELAMD:
    def __init__(self, feature_types, input_sizes):
        self.individual_learners = [IndividualLearner(ft, is_) for ft, is_ in zip(feature_types, input_sizes)]
        self.meta_learner = MetaLearner(len(self.individual_learners) * 2)

    def fit(self, X_list, y):
        # Train individual learners
        for learner, X in zip(self.individual_learners, X_list):
            learner.fit(X, y)

        # Generate meta-features
        meta_features = []
        for learner, X in zip(self.individual_learners, X_list):
            meta_features.append(learner.predict(X))
        meta_features = torch.tensor(np.concatenate(meta_features, axis=1), dtype=torch.float32)

        # Train meta-learner
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=0.001)
        # PyTorch training loop for meta-learner here

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


class ELAMD_AnomalyDetection:
    def __init__(self, feature_types, input_sizes, anomaly_method='iforest'):
        self.individual_learners = [IndividualLearner(ft, is_) for ft, is_ in zip(feature_types, input_sizes)]
        self.anomaly_detector = AnomalyDetector(method=anomaly_method)

    def fit(self, X_list, y):
        # Train individual learners
        for learner, X in zip(self.individual_learners, X_list):
            learner.fit(X, y)

        # Generate meta-features
        meta_features = []
        for learner, X in zip(self.individual_learners, X_list):
            meta_features.append(learner.predict(X))
        meta_features = np.concatenate(meta_features, axis=1)

        # Train anomaly detector
        self.anomaly_detector.fit(meta_features)

    def predict(self, X_list):
        meta_features = []
        for learner, X in zip(self.individual_learners, X_list):
            meta_features.append(learner.predict(X))
        meta_features = np.concatenate(meta_features, axis=1)
        return self.anomaly_detector.predict(meta_features)