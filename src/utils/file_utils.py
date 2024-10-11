""" File 처리 utils function """
import os
import csv
import json
import shutil
import hashlib
from ..adversarial_attacks.gym_malware.gym_malware.envs.utils import interface


def process_reports(src, dst):
    if not os.path.exists(src):
        os.makedirs(src)

    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {file_path}")
                        continue

                # 타겟 파일의 MD5 해시 추출
                md5_hash = data.get('target', {}).get('file', {}).get('md5', '')

                if md5_hash:
                    new_file_name = f"{md5_hash}.json"
                    new_file_path = os.path.join(dst, new_file_name)
                    shutil.copy(file_path, new_file_path)
                    print(f"Copied and renamed: {file_path} -> {new_file_path}")
                else:
                    print(f"MD5 hash not found in {file_path}")

        
# File reading and writing functions
def read_csv_file(file_path, skip_header=False):
    """ Read a CSV file and return its contents as a list """
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


def rename_to_sha256(directory):
    filename_mapping = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                bytes = f.read()
                sha256 = hashlib.sha256(bytes).hexdigest()
            new_filepath = os.path.join(directory, sha256)
            os.rename(filepath, new_filepath)
            filename_mapping.append((filename, sha256))
    return filename_mapping


def save_mapping_to_csv(mapping, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original Filename', 'SHA256 Filename'])
        writer.writerows(mapping)


def convert_and_save(directory_path, output_csv):
    filename_mapping = rename_to_sha256(directory_path)
    save_mapping_to_csv(filename_mapping, output_csv)


# KISA labels에 해당하는 src_dir의 file 중, malware에 해당하는 파일만 dest_dir로 이동함
def malware_move(src_dir, labels_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    malware_files = set()
    # Read all CSV files in labels_dir
    for labels_file in os.listdir(labels_dir):
        if labels_file.endswith('.csv'):
            for row in read_csv_file(os.path.join(labels_dir, labels_file)):
                file_name = row[0]
                label = row[1]
                if label == '1':
                    malware_files.add(file_name)

    # Move files from src_dir to dest_dir
    moved_count = 0
    for file_name in os.listdir(src_dir):
        if file_name in malware_files:
            src_path = os.path.join(src_dir, file_name)
            dest_path = os.path.join(dest_dir, file_name)
            shutil.move(src_path, dest_path)
            moved_count += 1

    print(f"Moved {moved_count} malware files to {dest_dir}")


if __name__ == "__main__":
    # process_reports 함수
    # src = "../../dataset/reports/benign_reports1"
    # dst = "../../dataset/reports/benign"
    # process_reports(src, dst)
    
    # malware_move 함수 : src_dir의 malware만 SAMPLE_PATH로 이동
    src1_dir = "/path/to/source/directory"
    src2_dir = "/path/to/source/directory"
    labels_dir = "/path/to/labels/directory"
    pe_directory = interface.SAMPLE_PATH
    malware_move(src1_dir, labels_dir, pe_directory)
    malware_move(src2_dir, labels_dir, pe_directory)

    output_csv_path = os.path.join(os.path.dirname(pe_directory), "KISA_filename_to_sha256_list.csv")
    convert_and_save(pe_directory, output_csv_path)

    print(f"Files in {pe_directory} have been renamed to their SHA256 hashes.")
    print(f"Filename mapping has been saved to {output_csv_path}")

    # 변환된 파일 목록 확인
    available_sha256 = interface.get_available_sha256()
    print(f"Available SHA256 files: {len(available_sha256)}")

    # 변환된 파일 중 하나를 불러와 확인
    if available_sha256:
        sample_sha256 = available_sha256[0]
        sample_bytes = interface.fetch_file(sample_sha256)
        print(f"Successfully fetched file with SHA256: {sample_sha256}")
        print(f"File size: {len(sample_bytes)} bytes")