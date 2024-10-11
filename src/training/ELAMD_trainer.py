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
from ..preprocessing.data_loader import ELAMDDataset, collate_fn
from ..models import MLP, DeepSVDD, SemiDeepSVDD
from ..models.ELAMD import ELAMD, ELAMD_AnomalyDetection, MetaLearner
from ..utils.tools import EarlyStopping, adjust_learning_rate, visual


# Load global configuration yaml file
with open('./src/training/config/ELAMD_config.yaml', 'r') as file:
    model_config = yaml.safe_load(file)


class ELAMDTrainer(object):
    def __init__(self, config):
        self.config = config
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
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()

    def _build_model(self):
        individual_learners = []
        for feature_type, input_size in zip(model_config['feature_types'], model_config['input_sizes']):
            if feature_type == 'continuous':
                individual_learners.append(self.individual_learners['MLP'](input_size, [256, 128, 32], 2))
            else:
                individual_learners.append(self.individual_learners['RandomForest']())

        meta_learner = MetaLearner(len(individual_learners) * 2)

        if model_config['use_anomaly_detection']:
            anomaly_detector = self.anomaly_detectors[model_config['anomaly_detector']]()
            model = ELAMD_AnomalyDetection(individual_learners, meta_learner, anomaly_detector)
        else:
            model = ELAMD(individual_learners, meta_learner)

        return model

    def _acquire_device(self):
        if self.config['gpu']['use_gpu']:
            device = torch.device('cuda:{}'.format(self.config['gpu']['gpu_id']))
            print('Use GPU: cuda:{}'.format(self.config['gpu']['gpu_id']))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def create_data_loaders(self):
        train_dataset = ELAMDDataset(self.config, self.config['dir'][self.dataset_name]['train_labels'])
        valid_dataset = ELAMDDataset(self.config, self.config['dir'][self.dataset_name]['valid_labels'])
        test_dataset = ELAMDDataset(self.config, self.config['dir'][self.dataset_name]['test_labels'])

        train_loader = DataLoader(train_dataset, batch_size=model_config['batch_size'], collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=model_config['batch_size'], collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=model_config['batch_size'], collate_fn=collate_fn)

        return train_loader, valid_loader, test_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=model_config['learning_rate'])
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def train(self, train_loader, valid_loader):
        time_now = time.time()
        print(f"\nSTART Training at {time_now}\n")

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=model_config['patience'], verbose=True)

        for epoch in range(model_config['train_epochs']):
            self.model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = [x.to(self.device) for x in batch_x]
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss, val_accuracy = self.evaluate(valid_loader)
            print(f"Epoch {epoch + 1}/{self.config['hyperparams']['train_epochs']}")
            print(f"Train Loss: {train_loss / len(train_loader):.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = [x.to(self.device) for x in batch_x]
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return total_loss / len(data_loader), correct / total

    def test(self, data_loader):
        print("=" * 20 + f" START TEST " + "=" * 20)
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = [x.to(self.device) for x in batch_x]
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions


def main(config):
    print("=" * 80)
    print("Load ELAMD Dataset & Model ...")
    print("=" * 80)

    # Set seed
    fix_seed = 2024
    torch.set_num_threads(6)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Load model & load datasets
    exp = ELAMDTrainer(config)
    train_loader, valid_loader, test_loader = exp.create_data_loaders()
    exp.train(train_loader, valid_loader) if config['is_training'] else exp.test(test_loader)