import os
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from ..preprocessing.PANACEA_dataset_loader import PANACEADataset, collate_fn
from ..models import PANACEA, XAI
from ..utils.tools import EarlyStopping, adjust_learning_rate, visual

class PANACEATrainer:
    def __init__(self, config):
        self.config = config
        self.model_name = config['model_name']
        self.dataset_name = config['dataset_name']

        # Train setting (GPU, model, loss, optim)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
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
        model = EL_VecComp.Model(self.config).float()
        return model

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config[self.model_name]['learning_rate'])
        return model_optim

    def create_data_loaders(self):
        train_dataset = PANACEADataset(self.config, self.config[self.dataset_name]['train_labels'])
        valid_dataset = PANACEADataset(self.config, self.config[self.dataset_name]['valid_labels'])
        test_dataset = PANACEADataset(self.config, self.config[self.dataset_name]['test_labels'])

        train_loader = DataLoader(
            train_dataset, batch_size=self.config['hyperparams']['batch_size'], collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.config['hyperparams']['batch_size'], collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config['hyperparams']['batch_size'], collate_fn=collate_fn
        )

        return train_loader, valid_loader, test_loader

    def select_diverse_models(self, base_models, train_loader):
        feature_importances = []
        for model in base_models:
            importance = get_feature_importance(model, train_loader)
            feature_importances.append(importance)

        kmeans = KMeans(n_clusters=self.config['model']['num_ensemble_models'])
        cluster_labels = kmeans.fit_predict(feature_importances)

        selected_models = []
        for cluster in range(self.config['model']['num_ensemble_models']):
            cluster_models = [model for model, label in zip(base_models, cluster_labels) if label == cluster]
            selected_models.append(cluster_models[0])  # Select the first model from each cluster

        return selected_models

    def train_ensemble(self, train_loader, valid_loader):
        print("=" * 20 + " START Training " + "=" * 20)

        base_models = self.train_base_models(train_loader)
        selected_models = self.select_diverse_models(base_models, train_loader)

        self.ensemble_model = EnsembleModel(selected_models, self.config).to(self.device)
        optimizer = optim.Adam(self.ensemble_model.parameters(), lr=self.config['hyperparams']['learning_rate'])
        early_stopping = EarlyStopping(patience=self.config['hyperparams']['patience'], verbose=True)

        for epoch in range(self.config['hyperparams']['train_epochs']):
            self.ensemble_model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.ensemble_model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_loss, val_accuracy = self.evaluate(valid_loader)
            print(f"Epoch {epoch + 1}/{self.config['hyperparams']['train_epochs']}")
            print(f"Train Loss: {train_loss / len(train_loader):.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            early_stopping(val_loss, self.ensemble_model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, data_loader):
        self.ensemble_model.eval()
        total_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.ensemble_model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        return total_loss / len(data_loader), correct / total

    def test(self, test_loader):
        print("=" * 20 + " START TEST " + "=" * 20)
        self.ensemble_model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.ensemble_model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        return predictions


def main(config):
    print("=" * 80)
    print("Load PANACEA Dataset & Model ...")
    print("=" * 80)

    # Set seed
    fix_seed = 2024
    torch.set_num_threads(6)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Load model & load datasets
    trainer = PANACEATrainer(config)
    train_loader, valid_loader, test_loader = trainer.create_data_loaders()

    if config['is_training']:
        trainer.train_ensemble(train_loader, valid_loader)
    else:
        predictions = trainer.test(test_loader)
        # Here you can add code to save or further process the predictions