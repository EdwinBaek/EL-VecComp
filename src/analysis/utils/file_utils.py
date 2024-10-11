"""
    File 처리 utils function
"""
import os
import json
import shutil

# src의 대량의 cuckoo report(json) 파일을 dst/{MD5}.csv로 변환 및 이동
def move_reports(src, dst):
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
                    shutil.move(file_path, new_file_path)
                    print(f"Copied and renamed: {file_path} -> {new_file_path}")
                else:
                    print(f"MD5 hash not found in {file_path}")

# 메인 함수 추가
def main():
    # BODMAS dataset
    bodmas_dst_path = './dataset/reports/BODMAS'
    # move_reports_main('./dataset/Cuckoo_Reports/BODMAS/bodmas1_reports', bodmas_dst_path)    # 11,261개
    # move_reports_main('./dataset/Cuckoo_Reports/BODMAS/bodmas3_reports', bodmas_dst_path)    # 9,004개
    # move_reports_main('./dataset/Cuckoo_Reports/BODMAS/bodmas4_reports', bodmas_dst_path)    # 8,926개
    # move_reports_main('./dataset/Cuckoo_Reports/BODMAS/bodmas5_reports', bodmas_dst_path)    # 9,004개
    # move_reports_main('./dataset/Cuckoo_Reports/BODMAS/bodmas6_reports', bodmas_dst_path)    # 9,006개

    # KISA dataset
    kisa_dst_path = './dataset/reports/KISA'
    # move_reports_main('./dataset/Cuckoo_Reports/KISA/kisa2_reports', kisa_dst_path)    # 2,441개
    # move_reports_main('./dataset/Cuckoo_Reports/KISA/kisa3_reports', kisa_dst_path)    # 9,977개
    # move_reports_main('./dataset/Cuckoo_Reports/KISA/kisa1_reports', kisa_dst_path)    # 591개
    # move_reports_main('./dataset/Cuckoo_Reports/KISA/kisa4_reports', kisa_dst_path)    # 10,686개
    # move_reports_main('./dataset/Cuckoo_Reports/KISA/kisa5_reports/kisa4_reports', kisa_dst_path)    # 592개

    # VirusShare dataset
    vs_dst_path = './dataset/reports/VirusShare'
    # move_reports_main('./dataset/Cuckoo_Reports/VirusShare/vs1_reports', vs_dst_path)    # 1,996개