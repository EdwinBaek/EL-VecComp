import os
import csv
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class ELAMDDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.features = self.data.drop(['file_path', 'label'], axis=1)
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