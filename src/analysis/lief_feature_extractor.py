import os
import lief
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_features(file_path):
    try:
        pe = lief.parse(file_path)
        features = {}

        # Byte histogram
        with open(file_path, 'rb') as f:
            content = f.read()
        features['byte_histogram'] = np.histogram(np.frombuffer(content, dtype=np.uint8), bins=256, range=(0,256))[0]

        # Byte entropy histogram
        window_size = 2048
        entropy_values = []
        for i in range(0, len(content), window_size):
            window = content[i:i+window_size]
            _, counts = np.unique(np.frombuffer(window, dtype=np.uint8), return_counts=True)
            probabilities = counts / len(window)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropy_values.append(entropy)
        features['byte_entropy'] = np.histogram(entropy_values, bins=256, range=(0,8))[0]

        # File info
        features['file_size'] = os.path.getsize(file_path)

        # Header info
        features['number_of_sections'] = pe.header.numberOfSections
        features['timestamp'] = pe.header.time_date_stamps

        # Section info
        features['sections_mean_entropy'] = np.mean([s.entropy for s in pe.sections])
        features['sections_min_entropy'] = np.min([s.entropy for s in pe.sections])
        features['sections_max_entropy'] = np.max([s.entropy for s in pe.sections])

        # Import info
        features['number_of_imports'] = sum(len(lib.entries) for lib in pe.imports)

        return features
    except:
        return None

def main(pe_directory, output_dir, label_files=[]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_features = []
    file_paths = [os.path.join(pe_directory, f) for f in os.listdir(pe_directory)]

    # Load all label files
    labels_df = pd.concat([pd.read_csv(f, header=None, names=['filename', 'label']) for f in label_files])
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    for file_path in tqdm(file_paths, desc="Extracting features"):
        features = extract_features(file_path)
        if features is not None:
            file_name = os.path.basename(file_path)
            features['file_path'] = file_path
            features['label'] = labels_dict.get(file_name, -1)  # -1 for unknown label
            all_features.append(features)

    output_file = os.path.join(output_dir, 'ELAMD_LIEF_data.csv')
    df = pd.DataFrame(all_features)
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")