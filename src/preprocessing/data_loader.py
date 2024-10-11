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
        self.dataset_name = config['dataset_name']
        self.vector_dirs = config['dir'][self.dataset_name]['arithmetic_vectors'] if config['coding_type'] == "ARITHMETIC" else config['dir'][self.dataset_name]['huffman_vectors']
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
                self.vector_dirs, f"compressed_{self.config['coding_type'].lower()}_{feature_name}_vectors.csv"
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
                preprocessed_feature_dir = os.path.join(self.config['dir'][self.dataset_name]['processed_dynamic_features'], feature_name)
            else:
                preprocessed_feature_dir = os.path.join(self.config['dir'][self.dataset_name]['processed_static_features'], feature_name)

            preprocessed_file = os.path.join(preprocessed_feature_dir, f"{file_name}.csv")

            try:
                if os.path.exists(preprocessed_file):
                    with open(preprocessed_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(feature_vectors[feature_name]) >= self.config['sequential']['max_seq_length']:
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
                feature_vectors[feature_name] = torch.tensor(feature_vectors[feature_name][:self.config['sequential']['max_seq_length']], dtype=torch.float32)
                if len(feature_vectors[feature_name]) < self.config['sequential']['max_seq_length']:
                    padding = torch.zeros(self.config['sequential']['max_seq_length'] - len(feature_vectors[feature_name]))
                    feature_vectors[feature_name] = torch.cat([feature_vectors[feature_name], padding])
            else:
                feature_vectors[feature_name] = torch.zeros(self.config['sequential']['max_seq_length'])

        return feature_vectors, label

    def get_feature_dimensions(self):
        return {feature: 1 for feature in self.feature_types}


class ELAMDDataset(Dataset):
    def __init__(self, config, labels_file):
        self.config = config
        self.dataset_name = config['dataset_name']
        self.file_names = []
        self.labels = []

        # Read labels and file names
        with open(labels_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.file_names.append(row[0])
                self.labels.append(int(row[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Extract features
        features = self.extract_features(file_path)

        # Convert to tensor
        features = {k: torch.tensor(v, dtype=torch.float32) for k, v in features.items()}
        label = torch.tensor(label, dtype=torch.long)

        return features, label

    def extract_features(self, file_path):
        pe = PE.parse(file_path)

        features = {}
        features['byte_histogram'] = self.get_byte_histogram(pe)
        features['byte_entropy'] = self.get_byte_entropy_histogram(pe)
        features['strings'] = self.get_strings(pe)
        features['file_info'] = self.get_file_info(pe)
        features['header_info'] = self.get_header_info(pe)
        features['section_info'] = self.get_section_info(pe)
        features['import_info'] = self.get_import_info(pe)

        return features

    def get_byte_histogram(self, pe):
        # Implement byte histogram extraction
        pass

    def get_byte_entropy_histogram(self, pe):
        # Implement byte entropy histogram extraction
        pass

    def get_strings(self, pe):
        # Implement string extraction and feature hashing
        pass

    def get_file_info(self, pe):
        # Implement file info extraction
        pass

    def get_header_info(self, pe):
        # Implement header info extraction
        pass

    def get_section_info(self, pe):
        # Implement section info extraction
        pass

    def get_import_info(self, pe):
        # Implement import info extraction
        pass


class PANACEADataset(Dataset):
    def __init__(self, config, labels_file):
        self.config = config
        self.dataset_name = config['dataset_name']
        self.vector_dirs = config['dir'][self.dataset_name]['arithmetic_vectors'] if config['coding_type'] == "ARITHMETIC" else config['dir'][self.dataset_name]['huffman_vectors']
        self.labels = []
        self.file_names = []
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
                self.vector_dirs, f"compressed_{self.config['coding_type'].lower()}_{feature_name}_vectors.csv"
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
                preprocessed_feature_dir = os.path.join(self.config['dir'][self.dataset_name]['processed_dynamic_features'], feature_name)
            else:
                preprocessed_feature_dir = os.path.join(self.config['dir'][self.dataset_name]['processed_static_features'], feature_name)

            preprocessed_file = os.path.join(preprocessed_feature_dir, f"{file_name}.csv")

            try:
                if os.path.exists(preprocessed_file):
                    with open(preprocessed_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(feature_vectors[feature_name]) >= self.config['sequential']['max_seq_length']:
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
                feature_vectors[feature_name] = torch.tensor(feature_vectors[feature_name][:self.config['sequential']['max_seq_length']], dtype=torch.float32)
                if len(feature_vectors[feature_name]) < self.config['sequential']['max_seq_length']:
                    padding = torch.zeros(self.config['sequential']['max_seq_length'] - len(feature_vectors[feature_name]))
                    feature_vectors[feature_name] = torch.cat([feature_vectors[feature_name], padding])
            else:
                feature_vectors[feature_name] = torch.zeros(self.config['sequential']['max_seq_length'])

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