import os
import csv
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ELVecCompDataset(Dataset):
    def __init__(self, config, labels_file):
        self.config = config
        self.model_name = config['model_name']
        self.dataset_name = config['dataset_name']
        self.vector_dirs = config[self.dataset_name]['arithmetic_vectors_dir'] if config['coding_type'] == "ARITHMETIC" else config[self.dataset_name]['huffman_vectors_dir']
        self.file_names = []
        self.labels = []
        self.feature_types = ['api_calls', 'file_system', 'registry', 'opcodes', 'strings', 'import_table']
        self.word_to_vector = {}
        self.identifier_to_encoded_bytes = {}

        # Read labels and file names
        with open(labels_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.file_names.append(row[0])
                self.labels.append(int(row[1]))

        self.load_word_embeddings()    # Load word embeddings

    def load_word_embeddings(self):
        for feature_name in self.feature_types:
            embedding_file = os.path.join(
                self.vector_dirs, f"compressed_{self.config['word_embedding'].lower()}_{feature_name}_vectors.csv"
            )
            embeddings = pd.read_csv(embedding_file, index_col=0)
            self.word_to_vector[feature_name] = {
                word: self.bytes_to_float(bytes.fromhex(encoded_bytes))
                for word, encoded_bytes in embeddings['encoded_bytes'].items()
            }

    def bytes_to_float(self, encoded_bytes):
        if self.config['coding_type'] == "ARITHMETIC":
            return struct.unpack('!d', encoded_bytes)[0]
        else:
            int_value = int.from_bytes(encoded_bytes, byteorder='big')
            return int_value / (2 ** (8 * len(encoded_bytes)) - 1)

    def get_vector(self, feature_type, identifier):
        if identifier in self.word_to_vector[feature_type].index:
            encoded_bytes = bytes.fromhex(self.word_to_vector[feature_type].loc[identifier, 'encoded_bytes'])
            return torch.tensor([self.bytes_to_float(encoded_bytes)], dtype=torch.float32)
        else:
            return torch.tensor([0.0], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = self.labels[idx]
        feature_vectors = {feature_name: [] for feature_name in self.feature_types}

        # Load and convert vectors
        for feature_name in self.feature_types:
            if feature_name in ['api_calls', 'file_system', 'registry']:
                preprocessed_feature_dir = os.path.join(
                    self.config[self.dataset_name]['processed_dynamic_features_dir'], feature_name
                )
            else:
                preprocessed_feature_dir = os.path.join(
                    self.config[self.dataset_name]['processed_static_features_dir'], feature_name
                )

            preprocessed_file = os.path.join(preprocessed_feature_dir, f"{file_name}.csv")

            try:
                if os.path.exists(preprocessed_file):
                    with open(preprocessed_file, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(feature_vectors[feature_name]) >= self.config[self.model_name]['max_seq_length']:
                                break
                            word = row[0]
                            if word in self.word_to_vector[feature_name]:
                                feature_vectors[feature_name].append(self.word_to_vector[feature_name][word])
                            else:
                                feature_vectors[feature_name].append(0.0)
                else:
                    feature_vectors[feature_name].append(0.0)

            except Exception as e:
                print(f"Error processing file {preprocessed_file}: {str(e)}")
                feature_vectors[feature_name].append(0.0)

            # Pad or truncate to max_seq_length
            if feature_vectors[feature_name]:
                feature_vectors[feature_name] = torch.tensor(feature_vectors[feature_name][:self.config[self.model_name]['max_seq_length']], dtype=torch.float32)
                if len(feature_vectors[feature_name]) < self.config[self.model_name]['max_seq_length']:
                    padding = torch.zeros(self.config[self.model_name]['max_seq_length'] - len(feature_vectors[feature_name]))
                    feature_vectors[feature_name] = torch.cat([feature_vectors[feature_name], padding])
            else:
                feature_vectors[feature_name] = torch.zeros(self.config[self.model_name]['max_seq_length'])

        return feature_vectors, label

    def get_feature_dimensions(self):
        return {feature: 1 for feature in self.feature_types}


def collate_fn(batch):
    feature_vectors = {feature_name: [] for feature_name in batch[0][0].keys()}
    labels = []

    for item, label in batch:
        for feature_name, vectors in item.items():
            feature_vectors[feature_name].append(vectors)
        labels.append(label)

    # Stack tensors
    for feature_name in feature_vectors:
        feature_vectors[feature_name] = torch.stack(feature_vectors[feature_name]).unsqueeze(-1)

    return feature_vectors, torch.tensor(labels)