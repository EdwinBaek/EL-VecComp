import os
import re
import csv
import json
import string
import logging
import pefile
import capstone
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_opcodes(pe, section):
    try:
        code = section.get_data()
        cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
        return [(i.mnemonic, i.op_str) for i in cs.disasm(code, 0x1000)]
    except Exception as e:
        logging.error(f"Opcode extraction error: {e}")
        return []

def extract_api_calls(pe):
    try:
        return [imp.name.decode('ascii', errors='ignore') for entry in pe.DIRECTORY_ENTRY_IMPORT
                for imp in entry.imports if imp.name]
    except Exception as e:
        logging.error(f"API calls extraction error: {e}")
        return []

def extract_dlls_and_functions(pe):
    try:
        dll_functions = {}
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode('ascii', errors='ignore')
            functions = [imp.name.decode('ascii', errors='ignore') for imp in entry.imports if imp.name]
            dll_functions[dll_name] = functions
        return dll_functions
    except Exception as e:
        logging.error(f"DLL and function extraction error: {e}")
        return {}

def is_meaningful_string(s, min_length=2):
    # 최소 길이 확인
    if len(s) < min_length:
        return False

    # 알파벳이나 숫자가 포함되어 있는지 확인
    if not re.search(r'[a-zA-Z0-9]', s):
        return False

    # 반복되는 문자가 너무 많은 경우 제외
    if len(set(s)) < len(s) / 3:
        return False

    # 특정 패턴의 의미 있는 문자열 포함 여부 확인
    meaningful_patterns = [
        r'\w+\.\w+',            # 파일명 또는 도메인 (예: example.com)
        r'(?i)https?://',       # URL
        r'(?i)[a-z]:\\',        # 윈도우 경로
        r'/\w+/\w+',            # Unix 경로
        r'\w+@\w+',             # 이메일 주소의 일부
        r'(?i)version',         # 버전 정보
        r'(?i)error|warning|info',    # 로그 관련 키워드
        r'[A-Z]{2,}',           # 대문자 약어 (예: API, DLL)
    ]

    if any(re.search(pattern, s) for pattern in meaningful_patterns):
        return True

    # 일반적인 단어나 명령어 포함 여부 확인
    common_words = {'the', 'and', 'or', 'if', 'for', 'while', 'function', 'return', 'class', 'int', 'char', 'bool'}
    words = re.findall(r'\w+', s.lower())
    if any(word in common_words for word in words):
        return True

    return False

def extract_strings(pe):
    try:
        strings = []
        printable = set(string.printable)
        for section in pe.sections:
            data = section.get_data()
            ascii_string = ''
            for byte in data:
                char = chr(byte)
                if char in printable:
                    ascii_string += char
                else:
                    if is_meaningful_string(ascii_string):
                        strings.append(ascii_string)
                    ascii_string = ''
            if is_meaningful_string(ascii_string):
                strings.append(ascii_string)

        return list(set(strings))  # 중복 제거
    except Exception as e:
        logging.error(f"String extraction error: {e}")
        return []

def save_feature_to_csv(feature_name, feature_data, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if feature_name == 'Opcodes':
            writer.writerow(['Opcode', 'Operand'])
            writer.writerows(feature_data)
        else:
            writer.writerow([feature_name])
            writer.writerows([[item] for item in feature_data])

def save_dll_functions_to_csv(dll_functions, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['DLL', 'Function'])
        for dll, functions in dll_functions.items():
            for func in functions:
                writer.writerow([dll, func])

def analyze_pe(file_path, output_dir):
    try:
        pe = pefile.PE(file_path)

        opcodes = extract_opcodes(pe, pe.sections[0])
        api_calls = extract_api_calls(pe)
        dll_functions = extract_dlls_and_functions(pe)
        strings = extract_strings(pe)

        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Save each feature to a separate CSV file in its own folder
        save_feature_to_csv('Opcodes', opcodes, os.path.join(output_dir, 'opcodes', f"{base_name}.csv"))
        save_feature_to_csv('API Calls', api_calls, os.path.join(output_dir, 'api_calls', f"{base_name}.csv"))
        save_dll_functions_to_csv(dll_functions, os.path.join(output_dir, 'dlls', f"{base_name}.csv"))
        save_feature_to_csv('Strings', strings, os.path.join(output_dir, 'strings', f"{base_name}.csv"))

        # Calculate feature counts for this file
        feature_counts = {
            'file_name': base_name,
            'opcodes': len(opcodes),
            'api_calls': len(api_calls),
            'dlls': len(dll_functions),
            'total_imported_functions': sum(len(funcs) for funcs in dll_functions.values()),
            'strings': len(strings)
        }

        logging.info(f"Analysis complete for {base_name}. Results saved to separate CSV files.")
        return True, feature_counts
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False, {}

def main(dataset_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # dataset 디렉토리 내 전체 파일 이름을 list로 저장
    pe_files = [f for f in os.listdir(dataset_dir)]
    total_files = len(pe_files)
    logging.info(f"Starting analysis of {total_files} PE files...")

    all_feature_counts = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_pe, os.path.join(dataset_dir, pe_file), output_dir) for pe_file in pe_files]

        success_count = 0
        with tqdm(total=total_files, desc="Analyzing PE files", unit="file") as pbar:
            for future in as_completed(futures):
                success, feature_counts = future.result()
                if success:
                    success_count += 1
                    all_feature_counts.append(feature_counts)
                pbar.update(1)

    logging.info(f"Analysis complete. Processed {success_count} out of {total_files} files successfully.")

    # Save feature counts for all files to a JSON file
    with open(os.path.join(output_dir, 'feature_counts.json'), 'w') as f:
        json.dump(all_feature_counts, f, indent=4)

    logging.info(f"Feature counts for all files saved to {os.path.join(output_dir, 'feature_counts.json')}")