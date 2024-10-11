import os
import sys
import yaml
import torch
import numpy as np

from src.models import statistical_coder
from src.analysis import static_analysis, cuckoo_parser, lief_feature_extractor
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
        static_analysis.main(config['dir'][dataset_name]['raw'], config['dir'][dataset_name]['static_features'])

    # LIEF library로 PE file로부터 feature 추출
    if task_type == 'LIEF_FEATURE_EXTRACT':
        lief_feature_extractor.main(
            config['dir'][dataset_name]['raw'],
            config['dir'][dataset_name]['lief_features'],
            label_files=[
                os.path.join(config['dir'][dataset_name]['labels_base'], '1.trainSet.csv'),
                os.path.join(config['dir'][dataset_name]['labels_base'], '2.preSet.csv'),
                os.path.join(config['dir'][dataset_name]['labels_base'], '3.finalSet1.csv'),
                os.path.join(config['dir'][dataset_name]['labels_base'], '4.finalSet2.csv')
            ]
        )

    # Static analysis, dynamic analysis 이후 데이터셋 생성
    elif task_type == 'PROCESSING_DATASET':
        # Cuckoo reports parsing
        cuckoo_parser.main(config['dir'][dataset_name]['reports'], config['dir'][dataset_name]['dynamic_features'])

        # Dataset을 EL-VecComp에서 활용 가능한 형태로 전처리
        dataset_processing.main(
            config, config['dir'][dataset_name]['dynamic_features'], config['dir'][dataset_name]['static_features'],
            config['dir'][dataset_name]['processed_feature_base'], config['dir'][dataset_name]['labels_base']
        )
    
    # 생성된 데이터셋을 기반으로 embedding model 구축 및 vecto compression 수행
    elif task_type == 'VECTOR_COMPRESSION':
        # 각 feature에 대한 embedding model 학습 (Word2Vec, GloVe, FastText, Graph2Vec)
        embedding_trainer.main(
            config['dir'][dataset_name]['processed_feature_base'], config['dir'][dataset_name]['embedding_models']
        )

        # Embedding model로부터 vector 추출
        embedding_vector_extractor.main(
            config['dir'][dataset_name]['embedding_models'], config['dir'][dataset_name]['embedding_vectors']
        )

        # Embedding vector로부터 statistical coding (arithmetic, huffman) 수행
        statistical_coder.main(
            config['dir'][dataset_name]['embedding_vectors'], config['dir'][dataset_name]['arithmetic_vectors']
        )
        statistical_coder.main(
            config['dir'][dataset_name]['embedding_vectors'], config['dir'][dataset_name]['huffman_vectors'], coding_type='huffman'
        )
    
    # Ensemble 딥 러닝 모델 학습 or inference
    elif task_type == 'TRAIN_DL_MODEL':
        if config['model_name'] == 'EL-VecComp':
            # EL-VecComp 모델 학습
            ELVecComp_trainer.main(config)

        elif config['model_name'] == 'ELAMD':
            # ELAMD 모델 학습
            ELAMD_trainer.main(config)

        elif config['model_name'] == 'PANACEA':
            # EL-VecComp 모델 학습
            PANACEA_trainer.main(config)

    else:
        print("Select job...")