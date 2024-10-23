import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from ..models import DeepSVDD, SemiDeepSVDD

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class IndividualLearner(nn.Module):
    def __init__(self, feature_type, input_size):
        super(IndividualLearner, self).__init__()
        if feature_type == 'continuous':
            self.model = MLP(input_size, [256, 128, 32], 2)
        else:
            self.model = RandomForestWrapper(input_size)

    def forward(self, x):
        return self.model(x)


class RandomForestWrapper(nn.Module):
    def __init__(self, input_size):
        super(RandomForestWrapper, self).__init__()
        self.rf = RandomForestClassifier()
        self.input_size = input_size
        self.is_fitted = False

    def forward(self, x):
        if self.training:
            return x  # During training, just return the input
        else:
            if not self.is_fitted:
                raise RuntimeError("RandomForestClassifier must be fitted before inference")
            return torch.tensor(self.rf.predict_proba(x.cpu().numpy()), device=x.device)

    def fit(self, X, y):
        self.rf.fit(X, y)
        self.is_fitted = True


class MetaLearner(nn.Module):
    def __init__(self, input_size):
        super(MetaLearner, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)


class ELAMD(nn.Module):
    def __init__(self, individual_learners, meta_learner):
        super(ELAMD, self).__init__()
        self.individual_learners = nn.ModuleList(individual_learners)
        self.meta_learner = meta_learner

    def forward(self, x):
        individual_outputs = [learner(xi) for learner, xi in zip(self.individual_learners, x)]
        meta_input = torch.cat(individual_outputs, dim=1)
        return self.meta_learner(meta_input)


class ELAMD_AnomalyDetection(nn.Module):
    def __init__(self, individual_learners, meta_learner, anomaly_detector):
        super(ELAMD_AnomalyDetection, self).__init__()
        self.individual_learners = nn.ModuleList(individual_learners)
        self.meta_learner = meta_learner
        self.anomaly_detector = anomaly_detector

    def forward(self, x):
        individual_outputs = [learner(xi) for learner, xi in zip(self.individual_learners, x)]
        meta_input = torch.cat(individual_outputs, dim=1)
        meta_output = self.meta_learner(meta_input)

        if isinstance(self.anomaly_detector, nn.Module):
            anomaly_output = self.anomaly_detector(meta_output)
        else:
            anomaly_output = torch.tensor(self.anomaly_detector.predict(meta_output.detach().cpu().numpy()), device=x[0].device)

        return meta_output, anomaly_output

    def fit_random_forests(self, X, y):
        for learner in self.individual_learners:
            if isinstance(learner.model, RandomForestWrapper):
                learner.model.fit(X, y)