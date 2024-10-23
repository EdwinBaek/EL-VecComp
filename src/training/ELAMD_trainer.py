import os
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ..preprocessing.ELAMD_dataset_loader import ELAMDDataset, collate_fn
from ..models.DeepSVDD import DeepSVDD
from ..models.SemiDeepSVDD import SemiDeepSVDD
from ..models.ELAMD import ELAMD, ELAMD_AnomalyDetection, MetaLearner, MLP, IndividualLearner, RandomForestWrapper
from ..utils.tools import EarlyStopping, adjust_learning_rate, visual


class ELAMDTrainer(object):
    def __init__(self, config):
        self.config = config
        self.model_name = config['model_name']
        self.dataset_name = config['dataset_name']

        self.individual_learners = {
            'MLP': MLP,
            'RandomForest': RandomForestClassifier
        }
        self.anomaly_detectors = {
            'IsolationForest': IsolationForest,
            'DeepSVDD': DeepSVDD,
            'SemiDeepSVDD': SemiDeepSVDD
        }

        # Train setting (GPU, model, loss, optim)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.rf_data = []  # RandomForest 학습을 위한 데이터 저장
        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()

    def _acquire_device(self):
        if self.config['use_gpu']:
            device = torch.device('cuda:{}'.format(self.config['gpu_id']))
            print('Use GPU: cuda:{}'.format(self.config['gpu_id']))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        individual_learners = []
        for feature_type, input_size in zip(self.config[self.model_name]['feature_types'], self.config[self.model_name]['input_sizes']):
            individual_learners.append(IndividualLearner(feature_type, input_size))

        meta_learner = MetaLearner(len(individual_learners) * 2)

        if self.config[self.model_name]['use_anomaly_detection']:
            AnomalyDetectorClass = self.anomaly_detectors[self.config[self.model_name]['anomaly_detector']]
            if issubclass(AnomalyDetectorClass, nn.Module):
                anomaly_detector = AnomalyDetectorClass(input_size=len(individual_learners) * 2)
            else:
                anomaly_detector = AnomalyDetectorClass()
            model = ELAMD_AnomalyDetection(individual_learners, meta_learner, anomaly_detector)
        else:
            model = ELAMD(individual_learners, meta_learner)

        return model

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config[self.model_name]['learning_rate'])
        return model_optim

    def create_data_loaders(self):
        train_dataset = ELAMDDataset(
            self.config, self.config[self.dataset_name]['lief_features_dir'], self.config[self.dataset_name]['train_labels']
        )
        valid_dataset = ELAMDDataset(
            self.config, self.config[self.dataset_name]['lief_features_dir'], self.config[self.dataset_name]['valid_labels']
        )
        test_dataset = ELAMDDataset(
            self.config, self.config[self.dataset_name]['lief_features_dir'], self.config[self.dataset_name]['test_labels']
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config[self.model_name]['batch_size'], collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config[self.model_name]['batch_size'], collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=self.config[self.model_name]['batch_size'], collate_fn=collate_fn)

        return train_loader, valid_loader, test_loader

    def train(self, train_loader, valid_loader):
        time_now = time.time()
        print(f"\nSTART Training at {time_now}\n")
        early_stopping = EarlyStopping(patience=self.config[self.model_name]['patience'], verbose=True)
        for epoch in range(self.config[self.model_name]['train_epochs']):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = [x.to(self.device) for x in batch_x]
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()

                if isinstance(self.model, ELAMD_AnomalyDetection):
                    outputs, _ = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                # RandomForest 학습을 위한 데이터 수집
                self.rf_data.extend(zip(batch_x, batch_y.cpu().numpy()))

            # 에폭이 끝날 때마다 RandomForest 학습
            self.train_random_forests()

            val_loss, val_metrics = self.evaluate(valid_loader)
            print(f"Epoch {epoch + 1}/{self.config[self.model_name]['train_epochs']}")
            print(f"Train Loss: {train_loss / len(train_loader):.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Metrics: {val_metrics}")

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_anomaly_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = [x.to(self.device) for x in batch_x]
                batch_y = batch_y.to(self.device)

                if isinstance(self.model, ELAMD_AnomalyDetection):
                    outputs, anomaly_outputs = self.model(batch_x)
                    all_anomaly_predictions.extend(anomaly_outputs.cpu().numpy())
                else:
                    outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        metrics = self.calculate_metrics(all_labels, all_predictions, all_anomaly_predictions)
        return total_loss / len(data_loader), metrics

    def test(self, data_loader):
        print("=" * 20 + " START TEST " + "=" * 20)
        self.model.eval()
        predictions = []
        anomaly_predictions = []
        true_labels = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = [x.to(self.device) for x in batch_x]

                if isinstance(self.model, ELAMD_AnomalyDetection):
                    outputs, anomaly_outputs = self.model(batch_x)
                    anomaly_predictions.extend(anomaly_outputs.cpu().numpy())
                else:
                    outputs = self.model(batch_x)

                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_y.numpy())

        metrics = self.calculate_metrics(true_labels, predictions, anomaly_predictions)
        print("Test Metrics:", metrics)
        return predictions, metrics

    def train_random_forests(self):
        X, y = zip(*self.rf_data)
        X = torch.cat(X, dim=0).cpu().numpy()
        y = np.array(y)
        if isinstance(self.model, ELAMD_AnomalyDetection):
            self.model.fit_random_forests(X, y)
        else:
            for learner in self.model.individual_learners:
                if isinstance(learner.model, RandomForestWrapper):
                    learner.model.fit(X, y)
        self.rf_data = []  # 데이터 초기화

    def calculate_metrics(self, true_labels, predictions, anomaly_predictions=None):
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1': f1_score(true_labels, predictions, average='weighted'),
        }

        if anomaly_predictions is not None:
            metrics['anomaly_auc'] = roc_auc_score(true_labels, anomaly_predictions)

        return metrics


# Voting ensemble strategy
def voting_ensemble(predictions):
    return np.argmax(np.bincount(predictions))


def main(config):
    print("=" * 100)
    print("Load ELAMD Dataset & Model ...")
    print("=" * 100)

    # Set seed
    fix_seed = 2024
    torch.set_num_threads(6)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Load model & load datasets
    exp = ELAMDTrainer(config)
    train_loader, valid_loader, test_loader = exp.create_data_loaders()

    if config['is_training']:
        exp.train(train_loader, valid_loader)
    else:
        predictions, metrics = exp.test(test_loader)

        # Voting ensemble
        model_name = config['model_name']
        if config[model_name]['use_voting']:
            ensemble_predictions = [voting_ensemble(pred) for pred in predictions]
            ensemble_metrics = exp.calculate_metrics(test_loader.dataset.labels, ensemble_predictions)
            print("Ensemble Metrics:", ensemble_metrics)