import os
import csv
import json
import hashlib
from tqdm import tqdm
from collections import Counter

def save_executed_identifiers(hashes, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Hash'])
        for hash_value in hashes:
            writer.writerow([hash_value])

def extract_file_and_registry_changes(report):
    file_changes, registry_changes = [], []

    for process in report.get('behavior', {}).get('processes', []):
        for call in process.get('calls', []):
            if call['category'] in ['file', 'registry']:
                change = {
                    'time': call['time'],
                    'api': call['api'],
                    'status': 'Success' if call['status'] else 'Fail',
                }

                if call['category'] == 'file':
                    if 'filepath' in call['arguments']:
                        change['path'] = call['arguments']['filepath']
                    elif 'file_path' in call['arguments']:
                        change['path'] = call['arguments']['file_path']
                    file_changes.append(change)
                elif call['category'] == 'registry':
                    if 'regkey' in call['arguments']:
                        change['path'] = call['arguments']['regkey']
                    elif 'key_handle' in call['arguments']:
                        change['path'] = call['arguments']['key_handle']
                    registry_changes.append(change)

    # Sort changes by time
    file_changes.sort(key=lambda x: x['time'])
    registry_changes.sort(key=lambda x: x['time'])

    return file_changes, registry_changes

def save_changes_to_csv(changes, output_file, change_type):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'API', 'Status', 'Path'])
        for change in changes:
            writer.writerow([
                change['time'],
                change['api'],
                change['status'],
                clean_text(change.get('path', 'N/A'))
            ])

def clean_text(text):
    if isinstance(text, str):
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    return str(text)

def extract_features(report):
    features = {'dynamic': {}}
    feature_counts = {}

    # Extract MD5 hash and PE filename
    target = report.get('target', {}).get('file', {})
    md5_hash = target.get('md5', '')
    pe_name = target.get('name', '')  # Get the PE filename

    # Check if the sample was executed (has PID)
    processes = report.get('behavior', {}).get('processes', [])
    if not (len(processes) > 0 and any(p.get('pid') for p in processes)):
        # Return None if the sample was not executed
        return None, None

    # Dynamic features
    behavior = report.get('behavior', {})

    # API calls
    api_calls = []
    for process in processes:
        for call in process.get('calls', []):
            api = call.get('api', '')
            api_calls.append(api)
    features['dynamic']['api_calls'] = api_calls
    feature_counts['api_calls'] = len(api_calls)

    # File operations
    for op in ['file_opened', 'file_created', 'file_written', 'file_deleted']:
        features['dynamic'][op] = behavior.get('summary', {}).get(op, [])
        feature_counts[op] = len(features['dynamic'][op])

    # Registry operations
    for op in ['regkey_opened', 'regkey_read', 'regkey_written', 'regkey_deleted']:
        features['dynamic'][op] = behavior.get('summary', {}).get(op, [])
        feature_counts[op] = len(features['dynamic'][op])

    # Network activity
    network = report.get('network', {})
    features['dynamic']['dns_requests'] = [d['request'] for d in network.get('dns', [])]
    feature_counts['dns_requests'] = len(features['dynamic']['dns_requests'])
    features['dynamic']['http_requests'] = [
        f"{r.get('method', '')} {r.get('url', '')}" for r in network.get('http', [])
    ]
    feature_counts['http_requests'] = len(features['dynamic']['http_requests'])

    # Other behavioral features
    for feature in ['mutex', 'service_created', 'service_started', 'process_created', 'dll_loaded']:
        features['dynamic'][feature] = behavior.get('summary', {}).get(feature, [])
        feature_counts[feature] = len(features['dynamic'][feature])

    # Add file and registry changes
    file_changes, registry_changes = extract_file_and_registry_changes(report)
    features['dynamic']['file_changes'] = file_changes
    features['dynamic']['registry_changes'] = registry_changes
    feature_counts['file_changes'] = len(file_changes)
    feature_counts['registry_changes'] = len(registry_changes)

    return features, feature_counts, md5_hash, pe_name

def save_feature_to_csv(feature_name, feature_data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([feature_name])
        for item in feature_data:
            writer.writerow([clean_text(item)])

def main(reports_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Reports directory: {reports_dir}")
    print(f"Extracted features directory: {output_dir}")
    print("Extracting features from Cuckoo sandbox reports...")

    # 전체 파일 수 계산
    total_files = sum(1 for f in os.listdir(reports_dir) if f.endswith('.json'))

    # Initialize execution counter and feature counts for each file
    execution_count = {'executed': 0, 'not_executed': 0, 'error': 0}
    all_feature_counts = []
    executed_identifiers = []  # 새로운 리스트를 추가합니다

    # tqdm으로 진행 상황 표시
    with tqdm(total=total_files, desc="Processing reports", unit="file") as pbar:
        for filename in os.listdir(reports_dir):
            # json 형식이 아닌 cuckoo report는 제외함
            if not filename.endswith('.json'):
                continue
            file_path = os.path.join(reports_dir, filename)
            try:
                if not os.path.exists(file_path):
                    print(f"File does not exist: {file_path}")
                    execution_count['error'] += 1
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    features, feature_counts, md5_hash, pe_name = extract_features(report)
                    if features is None:
                        execution_count['not_executed'] += 1
                        continue
                    else:
                        execution_count['executed'] += 1

                        # Use MD5 hash if available, otherwise use PE filename
                        # If both are unavailable, use the JSON report filename
                        file_identifier = md5_hash or pe_name or os.path.splitext(filename)[0]
                        executed_identifiers.append(file_identifier)    # 실행된 샘플의 MD5 hash를 리스트에 추가

                        # Store feature counts for this file
                        feature_counts['file_name'] = file_identifier
                        all_feature_counts.append(feature_counts)

                        for feature_type, feature_dict in features.items():
                            for feature_name, feature_value in feature_dict.items():
                                if isinstance(feature_value, list):
                                    feature_dir = os.path.join(output_dir, f"{feature_type}-{feature_name}")
                                    output_file = os.path.join(feature_dir, f"{file_identifier}.csv")

                                    if feature_name in ['file_changes', 'registry_changes']:
                                        save_changes_to_csv(feature_value, output_file, feature_name)
                                    else:
                                        save_feature_to_csv(feature_name, feature_value, output_file)

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                execution_count['error'] += 1

            pbar.update(1)  # 진행 상황 업데이트

    # 실행된 샘플의 해시를 CSV 파일로 저장합니다
    executed_identifiers_file = os.path.join(output_dir, "executed_identifiers.csv")
    save_executed_identifiers(executed_identifiers, executed_identifiers_file)
    print(f"Hashes of executed samples have been saved to {executed_identifiers_file}")

    # Save execution statistics
    execution_stats_file = os.path.join(output_dir, "execution_statistics.json")
    with open(execution_stats_file, 'w') as f:
        json.dump(execution_count, f, indent=4)

    print(f"Execution statistics have been saved to {execution_stats_file}")

    # Save feature counts for each file
    feature_counts_file = os.path.join(output_dir, "feature_counts.json")
    with open(feature_counts_file, 'w') as f:
        json.dump(all_feature_counts, f, indent=4)

    print(f"Feature counts for each file have been saved to {feature_counts_file}")