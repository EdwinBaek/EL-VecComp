import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from ..preprocessing.ELVecComp_dataset_loader import ELVecCompDataset, collate_fn
from ..models import EL_VecComp
from ..utils.tools import EarlyStopping, adjust_learning_rate, visual
from ..utils.training_logger import ModelTrainingLogger

class ELVecCompTrainer(object):
    def __init__(self, config, seq_model_name, nonseq_model_name):
        self.config = config
        self.seq_model_name = seq_model_name
        self.nonseq_model_name = nonseq_model_name
        self.model_name = config['model_name']
        self.dataset_name = config['dataset_name']
        self.embedding_name = config['word_embedding']

        # Train setting (GPU, model, loss, optim)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.criterion = self._select_criterion()
        self.optimizer = self._select_optimizer()
        
        # Logger 설정
        self.logger = ModelTrainingLogger(
            config,
            self.model_name,
            self.embedding_name,
            self.seq_model_name,
            self.nonseq_model_name
        )

    def _acquire_device(self):
        if self.config['use_gpu']:
            device = torch.device('cuda:{}'.format(self.config['gpu_id']))
            print('Use GPU: cuda:{}'.format(self.config['gpu_id']))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = EL_VecComp.Model(
            self.config, self.seq_model_name, self.nonseq_model_name
        ).float()
        return model

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config[self.model_name]['learning_rate'])
        return model_optim

    def create_data_loaders(self):
        train_dataset = ELVecCompDataset(self.config, self.config[self.dataset_name]['train_labels'])
        valid_dataset = ELVecCompDataset(self.config, self.config[self.dataset_name]['valid_labels'])
        test_dataset = ELVecCompDataset(self.config, self.config[self.dataset_name]['test_labels'])

        train_loader = DataLoader(
            train_dataset, batch_size=self.config[self.model_name]['batch_size'], collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=self.config[self.model_name]['batch_size'], collate_fn=collate_fn
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config[self.model_name]['batch_size'], collate_fn=collate_fn
        )

        return train_loader, valid_loader, test_loader

    def train(self, train_loader, valid_loader):
        print(f"START Training " + "=" * 80)

        # Create directories for saving results
        results_dir = os.path.join(
            self.config[self.dataset_name]['logs_dir'], 'train_and_validation'
            f'{self.embedding_name}_{self.seq_model_name}_{self.nonseq_model_name}'
        )
        os.makedirs(results_dir, exist_ok=True)

        # Initialize metrics tracking
        training_history = {
            'epoch': [],
            'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
        }
        
        # Best model 저장 dir 생성
        best_model_dir = os.path.join(
            self.config[self.dataset_name]['DL_models_dir'], self.model_name,
            f'{self.embedding_name}_{self.seq_model_name}_{self.nonseq_model_name}'
        )
        os.makedirs(best_model_dir, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.config[self.model_name]['patience'], verbose=True)
        best_metrics = {'val_f1': 0.0, 'epoch': 0}

        for epoch in range(self.config[self.model_name]['train_epochs']):
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            for batch_x, batch_y in train_loader:
                batch_x = {k: v.to(self.device) for k, v in batch_x.items()}
                batch_y = batch_y.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                # Calculate training accuracy
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            # Calculate training metrics
            train_metrics = self._calculate_metrics(train_labels, train_preds)
            train_avg_loss = train_loss / len(train_loader)

            # Evaluate on validation set
            val_loss, val_metrics = self.evaluate(valid_loader)

            # 로깅
            self.logger.update_training_history(
                epoch + 1, train_metrics, val_metrics, train_avg_loss, val_loss
            )
            self.logger.print_epoch_metrics(
                epoch + 1, self.config[self.model_name]['train_epochs'],
                train_metrics, val_metrics, train_avg_loss, val_loss
            )

            # Early stopping check
            early_stopping(val_loss, self.model, best_model_dir)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        self.logger.print_training_summary(best_model_dir)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = {k: v.to(self.device) for k, v in batch_x.items()}
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)

        return avg_loss, metrics

    def test(self, data_loader):
        print(f"START Test " + "=" * 80)
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = {k: v.to(self.device) for k, v in batch_x.items()}
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # metrics 계산
        test_loss = total_loss / len(data_loader)
        metrics = self._calculate_metrics(all_labels, all_preds)

        # 로깅
        self.logger.print_test_metrics(metrics, test_loss)
        self.logger.save_test_results(metrics, test_loss, all_labels, all_preds)

        results = {
            'predictions': np.array(all_preds),
            'true_labels': np.array(all_labels),
            'test_loss': test_loss,
            'metrics': metrics
        }

        return results

    def _calculate_metrics(self, labels, predictions):
        """Calculate precision, recall, and F1 score."""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = np.mean(np.array(labels) == np.array(predictions))

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def _print_metrics(self, metrics):
        """Print formatted metrics."""
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")


def main(config, seq_model_name, nonseq_model_name, is_training=True, model_pth=''):
    print("=" * 100)
    print("Load Ensemble Dataset & Model ...")
    print("=" * 100)

    # Set seed
    fix_seed = 2024
    torch.set_num_threads(6)
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # Load model & load datasets
    exp = ELVecCompTrainer(config, seq_model_name, nonseq_model_name)
    train_loader, valid_loader, test_loader = exp.create_data_loaders()
    exp.train(train_loader, valid_loader) if is_training else exp.test(test_loader)