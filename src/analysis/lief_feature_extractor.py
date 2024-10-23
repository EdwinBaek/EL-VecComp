import os
import re
import lief
import pefile
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

logging.getLogger('lief').setLevel(logging.ERROR)


def extract_strings(content, min_length=4):
    strings = re.findall(b'[\x20-\x7e]{%d,}' % min_length, content)
    return [s.decode('ascii') for s in strings]


def hash_strings(strings, size=1024):
    if not strings:
        return np.zeros(size)
    c = Counter(strings)
    arr = np.zeros(size)
    for s, count in c.items():
        arr[hash(s) % size] += count
    return arr

def is_pe_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            return f.read(2) == b'MZ'
    except:
        return False


def extract_features(file_path):
    features = {
        'file_path': file_path,
        'byte_histogram': np.zeros(256),
        'byte_entropy': np.zeros(256),
        'file_size': 0,
        'number_of_sections': 0,
        'timestamp': 0,
        'sections_mean_entropy': 0,
        'sections_min_entropy': 0,
        'sections_max_entropy': 0,
        'number_of_imports': 0,
        'has_relocation_table': 0,
        'has_resource_table': 0,
        'string_features': np.zeros(1024)  # 추가된 부분
    }

    try:
        # File info
        features['file_size'] = os.path.getsize(file_path)

        # Byte histogram and entropy
        with open(file_path, 'rb') as f:
            content = f.read()
        features['byte_histogram'] = np.histogram(np.frombuffer(content, dtype=np.uint8), bins=256, range=(0,256))[0]

        window_size = 2048
        entropy_values = []
        for i in range(0, len(content), window_size):
            window = content[i:i+window_size]
            _, counts = np.unique(np.frombuffer(window, dtype=np.uint8), return_counts=True)
            probabilities = counts / len(window)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropy_values.append(entropy)
        features['byte_entropy'] = np.histogram(entropy_values, bins=256, range=(0,8))[0]

        # String information
        strings = extract_strings(content)
        features['string_features'] = hash_strings(strings)

        if not is_pe_file(file_path):
            print(f"{file_path} is not a PE file")
            return features

        # Try LIEF parsing
        try:
            pe = lief.parse(file_path)
            if pe is not None:
                if hasattr(pe, 'header'):
                    if hasattr(pe.header, 'numberof_sections'):
                        features['number_of_sections'] = pe.header.numberof_sections
                    elif hasattr(pe.header, 'numberOfSections'):
                        features['number_of_sections'] = pe.header.numberOfSections

                    if hasattr(pe.header, 'time_date_stamps'):
                        features['timestamp'] = pe.header.time_date_stamps

                if pe.sections:
                    section_entropies = [s.entropy for s in pe.sections]
                    features['sections_mean_entropy'] = np.mean(section_entropies)
                    features['sections_min_entropy'] = np.min(section_entropies)
                    features['sections_max_entropy'] = np.max(section_entropies)

                if hasattr(pe, 'imports'):
                    features['number_of_imports'] = sum(len(lib.entries) for lib in pe.imports)

                features['has_relocation_table'] = int(pe.has_relocations)
                features['has_resource_table'] = int(pe.has_resources)
            else:
                raise Exception("LIEF parsing failed")

        # If LIEF fails, try pefile
        except:
            try:
                pe = pefile.PE(file_path)
                features['number_of_sections'] = pe.FILE_HEADER.NumberOfSections
                features['timestamp'] = pe.FILE_HEADER.TimeDateStamp
                features['number_of_imports'] = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT) if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT') else 0
                features['has_relocation_table'] = int(hasattr(pe, 'DIRECTORY_ENTRY_BASERELOC'))
                features['has_resource_table'] = int(hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'))

                if hasattr(pe, 'sections'):
                    section_entropies = [section.get_entropy() for section in pe.sections]
                    features['sections_mean_entropy'] = np.mean(section_entropies)
                    features['sections_min_entropy'] = np.min(section_entropies)
                    features['sections_max_entropy'] = np.max(section_entropies)
            except:
                print(f"Both LIEF and pefile failed to parse {file_path}")

    except Exception as e:
        print(f"Unexpected error for {file_path}: {str(e)}")

    return features


def main(pe_directory, output_dir, label_files=[]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load all label files
    labels_df = pd.concat([pd.read_csv(f, header=None, names=['filename', 'label']) for f in label_files])
    labels_dict = dict(zip(labels_df['filename'], labels_df['label']))

    file_paths = [os.path.join(pe_directory, f) for f in os.listdir(pe_directory)]

    for file_path in tqdm(file_paths, desc="Extracting features"):
        features = extract_features(file_path)
        file_name = os.path.basename(file_path)
        features['label'] = labels_dict.get(file_name, -1)  # -1 for unknown label

        # Convert the features dictionary to a pandas DataFrame
        df = pd.DataFrame([features])

        # Create the output filename
        output_filename = os.path.splitext(file_name)[0] + '.csv'
        output_file_path = os.path.join(output_dir, output_filename)

        # Save the features to a CSV file
        df.to_csv(output_file_path, index=False)
        print(f"Features for {file_name} saved to {output_file_path}")

    print(f"Feature extraction completed. Results saved in {output_dir}")