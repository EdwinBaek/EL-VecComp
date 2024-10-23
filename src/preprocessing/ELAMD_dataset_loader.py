import os
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ELAMDDataset(Dataset):
    def __init__(self, config, lief_features_dir, labels_file):
        self.config = config
        self.model_name = config['model_name']
        self.dataset_name = config['dataset_name']

        # Load all CSV files in the directory
        all_features = []
        for filename in os.listdir(lief_features_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(lief_features_dir, filename)
                df = pd.read_csv(file_path)
                all_features.append(df)

        # Concatenate all feature dataframes
        self.features_df = pd.concat(all_features, ignore_index=True)
        self.features_df.set_index('file_path', inplace=True)

        # Load labels
        self.labels_file_df = pd.read_csv(labels_file)

        # Merge features and labels
        self.data = self.labels_file_df.merge(self.features_df, left_on='filename', right_index=True, how='inner')

        # Separate features and labels
        self.features = self.data.drop(['filename', 'label'], axis=1)
        self.labels = self.data['label']

        # Define feature types and input sizes
        self.feature_types = config[self.model_name]['feature_types']
        self.input_sizes = config[self.model_name]['input_sizes']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features.iloc[idx].values.astype(np.float32)
        label = self.labels.iloc[idx]

        # Split features based on input sizes
        feature_list = []
        start = 0
        for size in self.input_sizes:
            feature_list.append(torch.tensor(features[start:start + size], dtype=torch.float32))
            start += size

        return feature_list, torch.tensor(label, dtype=torch.long)

    def get_feature_info(self):
        return self.feature_types, self.input_sizes


def collate_fn(batch):
    features = [[] for _ in range(len(batch[0][0]))]
    labels = []

    for item in batch:
        for i, feature in enumerate(item[0]):
            features[i].append(feature)
        labels.append(item[1])

    features = [torch.stack(feature) for feature in features]
    labels = torch.stack(labels)

    return features, labels