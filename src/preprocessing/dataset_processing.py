import os
import csv
import random
from functools import partial
from concurrent.futures import ProcessPoolExecutor

# File reading and writing functions
def read_csv_file(file_path, skip_header=False):
    """Read a CSV file and return its contents as a list."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            if skip_header:
                next(reader)  # Skip header
            return list(reader)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def write_csv_file(file_path, data):
    """Write data to a CSV file."""
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

# Feature processing functions
def process_api_calls(src_file, dest_file, max_seq_length):
    """Process API calls feature."""
    data = read_csv_file(src_file)
    api_calls = [[row[0]] for row in data[:max_seq_length]]
    write_csv_file(dest_file, api_calls)

def process_file_system(src_file, dest_file, max_seq_length):
    """Process file system feature."""
    data = read_csv_file(src_file)
    file_system_data = [row for row in data[:max_seq_length]]
    write_csv_file(dest_file, file_system_data)

def process_registry(src_file, dest_file, max_seq_length):
    """Process registry feature (same as file system)."""
    process_file_system(src_file, dest_file, max_seq_length)

def process_opcodes(src_file, dest_file, max_seq_length):
    """Process opcodes feature."""
    data = read_csv_file(src_file)
    opcodes = [[row[0]] for row in data[:max_seq_length]]
    write_csv_file(dest_file, opcodes)

def process_strings(src_file, dest_file, max_seq_length):
    """Process strings feature."""
    data = read_csv_file(src_file)
    strings = [[row[0]] for row in data[:max_seq_length]]
    write_csv_file(dest_file, strings)

def process_import_table(src_file, dest_file, max_seq_length):
    """Process import table feature."""
    data = read_csv_file(src_file)
    imports = [[f"{row[0]}\\{row[1]}"] for row in data[:max_seq_length]]
    write_csv_file(dest_file, imports)

# Main processing functions
def process_file(process_func, src_dir, dest_dir, hash_value, max_seq_length):
    """Process a single file for a given feature."""
    try:
        src_file = os.path.join(src_dir, f"{hash_value}.csv")
        dest_file = os.path.join(dest_dir, f"{hash_value}.csv")
        if os.path.exists(src_file):
            process_func(src_file, dest_file, max_seq_length)
            return True
        else:
            print(f"Source file not found: {src_file}")
            return False
    except Exception as e:
        print(f"Error processing {hash_value}: {str(e)}")
        return False

def process_feature(feature_name, process_func, src_dir, dest_dir, executed_identifiers, max_seq_length):
    """Process all files for a given feature."""
    print(f"Pre-processing {feature_name}")
    os.makedirs(dest_dir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=4) as executor:
        process_partial = partial(process_file, process_func, src_dir, dest_dir, max_seq_length=max_seq_length)
        results = list(executor.map(process_partial, executed_identifiers))

    processed_count = sum(results)
    print(f"Processed {processed_count} out of {len(executed_identifiers)} files for {feature_name}")

# Dataset creation and splitting functions
def split_dataset(input_file, labels_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split the dataset into train, validation, and test sets."""
    data = read_csv_file(input_file, skip_header=False)
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    write_csv_file(os.path.join(labels_dir, 'train_set.csv'), train_data)
    write_csv_file(os.path.join(labels_dir, 'valid_set.csv'), val_data)
    write_csv_file(os.path.join(labels_dir, 'test_set.csv'), test_data)

    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")

def create_dataset_list(executed_csv_file, labels_dir, dataset_name):
    """Create a dataset list based on the executed files and existing labels."""
    if dataset_name not in ['KISA', 'PEMML', 'BODMAS', 'VirusShare', 'TEST']:
        print(f"Unsupported dataset name: {dataset_name}")
        return None

    print(f"Creating {dataset_name} labels...")

    executed_hashes = set(row[0] for row in read_csv_file(executed_csv_file))

    new_labels = {}
    for labels_file in os.listdir(labels_dir):
        for row in read_csv_file(os.path.join(labels_dir, labels_file)):
            file_name = os.path.splitext(row[0])[0]  # Remove extension
            if file_name in executed_hashes:
                new_labels[file_name] = row[1]

    output_file = os.path.join(labels_dir, 'dataset_list.csv')
    write_csv_file(output_file, [[file_name, label] for file_name, label in new_labels.items()])

    print(f"New labels file created: {output_file}")
    return output_file

# Main function
def main(config, dynamic_features_dir, static_features_dir, output_dir, labels_dir):
    """ Main function to orchestrate the entire data processing pipeline """
    max_seq_length = config['sequential']['max_seq_length']

    executed_identifiers_file = os.path.join(dynamic_features_dir, "executed_identifiers.csv")
    executed_identifiers = [row[0] for row in read_csv_file(executed_identifiers_file)]
    print(f"Found {len(executed_identifiers)} executed hashes")

    features = [
        ("DYNAMIC FEATURES : API calls", process_api_calls, os.path.join(dynamic_features_dir, "dynamic-api_calls"),
         os.path.join(output_dir, "dynamic", "api_calls")),
        ("DYNAMIC FEATURES : File System", process_file_system,
         os.path.join(dynamic_features_dir, "dynamic-file_changes"),
         os.path.join(output_dir, "dynamic", "file_system")),
        ("DYNAMIC FEATURES : Registry", process_registry,
         os.path.join(dynamic_features_dir, "dynamic-registry_changes"),
         os.path.join(output_dir, "dynamic", "registry")),
        ("STATIC FEATURES : opcodes", process_opcodes, os.path.join(static_features_dir, "opcodes"),
         os.path.join(output_dir, "static", "opcodes")),
        ("STATIC FEATURES : strings", process_strings, os.path.join(static_features_dir, "strings"),
         os.path.join(output_dir, "static", "strings")),
        ("STATIC FEATURES : Import Table", process_import_table, os.path.join(static_features_dir, "dlls"),
         os.path.join(output_dir, "static", "import_table")),
    ]

    for feature_name, process_func, src_dir, dest_dir in features:
        process_feature(feature_name, process_func, src_dir, dest_dir, executed_identifiers, max_seq_length)

    output_file_path = create_dataset_list(executed_identifiers_file, labels_dir, config['dataset_name'])
    if output_file_path:
        split_dataset(output_file_path, labels_dir)

    print("Dataset processing complete!")