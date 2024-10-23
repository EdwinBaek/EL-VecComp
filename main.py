import os
import sys
import yaml
import torch
import numpy as np

from src.models import statistical_coder
from src.analysis import static_analysis, cuckoo_parser, lief_feature_extractor
from src.analysis.utils import file_utils
from src.preprocessing import embedding_vector_extractor, dataset_processing
from src.training import embedding_trainer, ELVecComp_trainer, ELAMD_trainer, PANACEA_trainer

# Load configuration yaml file
def load_config():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    print(f"\nTASK: {config['task_name']}")
    print(f"Model : {config['model_name']}")
    print(f"Dataset : {config['dataset_name']}\n")
    return config


if __name__ == "__main__":
    config = load_config()

    task_type = config['task_name']
    dataset_name = config['dataset_name']

    # Static analysis 결과를 csv 파일로 저장
    if task_type == 'STATIC_ANALYSIS':
        static_analysis.main(config[dataset_name]['raw_dir'], config[dataset_name]['static_features_dir'])

    # LIEF library로 PE file로부터 feature 추출
    if task_type == 'LIEF_FEATURE_EXTRACT':
        lief_feature_extractor.main(
            config[dataset_name]['raw_dir'],
            config[dataset_name]['lief_features_dir'],
            label_files=[
                os.path.join(config[dataset_name]['labels_base_dir'], '1.trainSet.csv'),
                os.path.join(config[dataset_name]['labels_base_dir'], '2.preSet.csv'),
                os.path.join(config[dataset_name]['labels_base_dir'], '3.finalSet1.csv'),
                os.path.join(config[dataset_name]['labels_base_dir'], '4.finalSet2.csv')
            ]
        )

    # Static analysis, dynamic analysis 이후 데이터셋 생성
    elif task_type == 'PROCESSING_DATASET':
        # Cuckoo reports parsing
        cuckoo_parser.main(config[dataset_name]['reports_dir'], config[dataset_name]['dynamic_features_dir'])

        # Dataset을 EL-VecComp에서 활용 가능한 형태로 전처리
        dataset_processing.main(
            config, config[dataset_name]['dynamic_features_dir'], config[dataset_name]['static_features_dir'],
            config[dataset_name]['processed_feature_base_dir'], config[dataset_name]['labels_base_dir']
        )
    
    # 생성된 데이터셋을 기반으로 embedding model 구축 및 vecto compression 수행
    elif task_type == 'VECTOR_COMPRESSION':
        # 각 feature에 대한 embedding model 학습 (Word2Vec, GloVe, FastText, Graph2Vec)
        embedding_trainer.main(
            config[dataset_name]['processed_feature_base_dir'], config[dataset_name]['embedding_models_dir']
        )

        # Embedding model로부터 vector 추출
        embedding_vector_extractor.main(
            config[dataset_name]['embedding_models_dir'], config[dataset_name]['embedding_vectors_dir']
        )

        # Embedding vector로부터 statistical coding (arithmetic, huffman) 수행
        statistical_coder.main(
            config[dataset_name]['embedding_vectors_dir'], config[dataset_name]['arithmetic_vectors_dir']
        )
        statistical_coder.main(
            config[dataset_name]['embedding_vectors_dir'], config[dataset_name]['huffman_vectors_dir'], coding_type='huffman'
        )
    
    # Ensemble 딥 러닝 모델 학습 or inference
    elif task_type == 'TRAIN_DL_MODEL':
        if config['model_name'] == 'EL-VecComp':
            # EL-VecComp 모델 학습
            ELVecComp_trainer.main(config, seq_model_name='RNN', nonseq_model_name='MLP')
            ELVecComp_trainer.main(config, seq_model_name='Bi-RNN', nonseq_model_name='MLP')
            ELVecComp_trainer.main(config, seq_model_name='LSTM', nonseq_model_name='MLP')
            ELVecComp_trainer.main(config, seq_model_name='Bi-LSTM', nonseq_model_name='MLP')
            ELVecComp_trainer.main(config, seq_model_name='GRU', nonseq_model_name='MLP')
            ELVecComp_trainer.main(config, seq_model_name='Bi-GRU', nonseq_model_name='MLP')

            ELVecComp_trainer.main(config, seq_model_name='RNN', nonseq_model_name='Transformer')
            ELVecComp_trainer.main(config, seq_model_name='Bi-RNN', nonseq_model_name='Transformer')
            ELVecComp_trainer.main(config, seq_model_name='LSTM', nonseq_model_name='Transformer')
            ELVecComp_trainer.main(config, seq_model_name='Bi-LSTM', nonseq_model_name='Transformer')
            ELVecComp_trainer.main(config, seq_model_name='GRU', nonseq_model_name='Transformer')
            ELVecComp_trainer.main(config, seq_model_name='Bi-GRU', nonseq_model_name='Transformer')

        elif config['model_name'] == 'ELAMD':
            # ELAMD 모델 학습
            ELAMD_trainer.main(config)

        elif config['model_name'] == 'PANACEA':
            # EL-VecComp 모델 학습
            PANACEA_trainer.main(config)

    # Ensemble 딥 러닝 모델 inference
    elif task_type == 'INFERENCE_DL_MODEL':
        if config['model_name'] == 'EL-VecComp':
            # EL-VecComp 모델 inference (test)
            ELVecComp_trainer.main(
                config, seq_model_name='RNN', nonseq_model_name='MLP', is_training=False,
                model_pth=os.path.join(config['DL_models_dir'], config['model_name'], 'FastText_RNN_MLP/checkpoint_val_loss_0.5760.pth')
            )
            # ELVecComp_trainer.main(config, seq_model_name='Bi-RNN', nonseq_model_name='MLP')
            # ELVecComp_trainer.main(config, seq_model_name='LSTM', nonseq_model_name='MLP')
            # ELVecComp_trainer.main(config, seq_model_name='Bi-LSTM', nonseq_model_name='MLP')
            # ELVecComp_trainer.main(config, seq_model_name='GRU', nonseq_model_name='MLP')
            # ELVecComp_trainer.main(config, seq_model_name='Bi-GRU', nonseq_model_name='MLP')

            # ELVecComp_trainer.main(config, seq_model_name='RNN', nonseq_model_name='Transformer')
            # ELVecComp_trainer.main(config, seq_model_name='Bi-RNN', nonseq_model_name='Transformer')
            # ELVecComp_trainer.main(config, seq_model_name='LSTM', nonseq_model_name='Transformer')
            # ELVecComp_trainer.main(config, seq_model_name='Bi-LSTM', nonseq_model_name='Transformer')
            # ELVecComp_trainer.main(config, seq_model_name='GRU', nonseq_model_name='Transformer')
            # ELVecComp_trainer.main(config, seq_model_name='Bi-GRU', nonseq_model_name='Transformer')

        elif config['model_name'] == 'ELAMD':
            # ELAMD 모델 inference (test)
            ELAMD_trainer.main(config)

        elif config['model_name'] == 'PANACEA':
            # PANACEA 모델 inference (test)
            PANACEA_trainer.main(config)

    # src의 report를 dest로 옮기면서 MD5.json으로 이름을 변경함
    elif task_type == 'MOVE_REPORTS':
        dest_dir = config['KISA']['reports_dir']
        src1_dir = os.path.join(config['KISA']['dataset_base_dir'], 'cuckoo_reports_zips_temp/cuckoo_reports_finalset2_1/')
        src2_dir = os.path.join(config['KISA']['dataset_base_dir'], 'cuckoo_reports_zips_temp/cuckoo_reports_finalset2_2/')
        src3_dir = os.path.join(config['KISA']['dataset_base_dir'], 'cuckoo_reports_zips_temp/cuckoo_reports_finalset2_3/')
        src4_dir = os.path.join(config['KISA']['dataset_base_dir'], 'cuckoo_reports_zips_temp/cuckoo_reports_finalset2_4/')
        src5_dir = os.path.join(config['KISA']['dataset_base_dir'], 'cuckoo_reports_zips_temp/finalset1/')
        src6_dir = os.path.join(config['KISA']['dataset_base_dir'], 'cuckoo_reports_zips_temp/finalset2/')
        file_utils.move_reports(src1_dir, dest_dir)
        file_utils.move_reports(src2_dir, dest_dir)
        file_utils.move_reports(src3_dir, dest_dir)
        file_utils.move_reports(src4_dir, dest_dir)
        file_utils.move_reports(src5_dir, dest_dir)
        file_utils.move_reports(src6_dir, dest_dir)

    else:
        print("Select job...")