import os
import csv
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ELAMDDataset(Dataset):
    def __init__(self, lief_features_dir, labels_file):
        # Load features
        self.features_df = pd.read_csv(os.path.join(lief_features_dir, 'lief_features.csv'))
        self.features_df.set_index('file_name', inplace=True)

        # Load set (train/valid/test)
        self.labels_file_df = pd.read_csv(labels_file)

        # Merge features and labels
        self.data = self.labels_file_df.merge(self.features_df, left_on='filename', right_index=True, how='inner')

        # Separate features and labels
        self.features = self.data.drop(['filename', 'label'], axis=1)
        self.labels = self.data['label']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features.iloc[idx].values.astype(np.float32)
        label = self.labels.iloc[idx]

        return torch.tensor(features), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    features = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return features, labels