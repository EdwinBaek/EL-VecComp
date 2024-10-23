import os
import pandas as pd
import numpy as np
from datetime import datetime


class ModelTrainingLogger:
    """모델 학습 과정과 결과를 로깅하는 클래스"""

    def __init__(self, config, model_name, embedding_name=None, seq_model_name=None, nonseq_model_name=None):
        """
        Args:
            config: 설정 정보를 담은 딕셔너리
            model_name: 모델 이름
            embedding_name: 임베딩 방식 이름 (선택)
            seq_model_name: Sequential 모델 이름 (선택)
            nonseq_model_name: Non-sequential 모델 이름 (선택)
        """
        self.config = config
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.seq_model_name = seq_model_name
        self.nonseq_model_name = nonseq_model_name

        # 로그 저장 디렉토리 설정
        self._setup_directories()

        # 훈련 기록 초기화
        self.training_history = {
            'epoch': [],
            'train_loss': [], 'train_accuracy': [], 'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
        }
        self.best_metrics = {'val_f1': 0.0, 'epoch': 0}

    def _setup_directories(self):
        """로그 저장을 위한 디렉토리 생성"""
        model_identifier = '_'.join(filter(None, [
            self.embedding_name,
            self.seq_model_name,
            self.nonseq_model_name
        ]))

        # 훈련/검증 로그 디렉토리
        self.train_val_dir = os.path.join(
            self.config[self.config['dataset_name']]['logs_dir'],
            'train_and_validation',
            model_identifier
        )

        # 테스트 로그 디렉토리
        self.test_dir = os.path.join(
            self.config[self.config['dataset_name']]['logs_dir'],
            'test',
            model_identifier
        )

        os.makedirs(self.train_val_dir, exist_ok=True)
        os.makedirs(self.test_dir, exist_ok=True)

    def update_training_history(self, epoch, train_metrics, val_metrics, train_loss, val_loss):
        """훈련 과정의 metrics 업데이트 및 저장"""
        # 훈련 기록 업데이트
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['train_accuracy'].append(train_metrics['accuracy'])
        self.training_history['train_precision'].append(train_metrics['precision'])
        self.training_history['train_recall'].append(train_metrics['recall'])
        self.training_history['train_f1'].append(train_metrics['f1'])
        self.training_history['val_loss'].append(val_loss)
        self.training_history['val_accuracy'].append(val_metrics['accuracy'])
        self.training_history['val_precision'].append(val_metrics['precision'])
        self.training_history['val_recall'].append(val_metrics['recall'])
        self.training_history['val_f1'].append(val_metrics['f1'])

        # CSV 파일로 저장
        pd.DataFrame(self.training_history).to_csv(
            os.path.join(self.train_val_dir, 'training_history.csv'),
            index=False
        )

        # 최고 성능 갱신 확인 및 저장
        if val_metrics['f1'] > self.best_metrics['val_f1']:
            self.best_metrics['val_f1'] = val_metrics['f1']
            self.best_metrics['epoch'] = epoch

            pd.DataFrame([{
                'best_epoch': self.best_metrics['epoch'],
                'best_val_f1': self.best_metrics['val_f1'],
                'train_loss': train_loss,
                'train_accuracy': train_metrics['accuracy'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_f1': train_metrics['f1'],
                'val_loss': val_loss,
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1']
            }]).to_csv(os.path.join(self.train_val_dir, 'best_metrics.csv'), index=False)

    def save_test_results(self, test_metrics, test_loss, true_labels, predictions):
        """테스트 결과 저장"""
        # 테스트 metrics 저장
        test_results = {
            'test_loss': test_loss,
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1']
        }
        pd.DataFrame([test_results]).to_csv(
            os.path.join(self.test_dir, 'test_metrics.csv'),
            index=False
        )

        # 예측 결과 저장
        pd.DataFrame({
            'true_labels': true_labels,
            'predictions': predictions
        }).to_csv(os.path.join(self.test_dir, 'test_predictions.csv'), index=False)

    def print_epoch_metrics(self, epoch, total_epochs, train_metrics, val_metrics, train_loss, val_loss):
        """에폭별 metrics 출력"""
        print(f"\nEpoch {epoch}/{total_epochs}" + "=" * 80)
        print("Training Metrics:")
        print(f"Loss: {train_loss:.4f}")
        self.print_metrics(train_metrics)

        print("\nValidation Metrics:")
        print(f"Loss: {val_loss:.4f}")
        self.print_metrics(val_metrics)

    def print_test_metrics(self, metrics, test_loss):
        """테스트 metrics 출력"""
        print("\nTest Metrics:")
        print(f"Loss: {test_loss:.4f}")
        self.print_metrics(metrics)

    @staticmethod
    def print_metrics(metrics):
        """metrics 출력 포맷팅"""
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")

    def print_training_summary(self, best_model_dir):
        """학습 완료 후 요약 정보 출력"""
        print(f"\nTraining completed.")
        print(f"Best validation F1: {self.best_metrics['val_f1']:.4f} at epoch {self.best_metrics['epoch']}")
        print(f"Best model saved to {best_model_dir}")
        print(f"Training history saved to {self.train_val_dir}")