import os
import json
import shutil
from pathlib import Path
from datetime import datetime

# src의 대량의 cuckoo report(json) 파일을 dst/{MD5}.csv로 변환 및 이동
def move_reports(src, dst):
    """
    JSON 보고서 파일들을 안전하게 이동하는 함수

    Args:
        src: 소스 디렉토리 경로
        dst: 대상 디렉토리 경로

    Returns:
        dict: 처리 결과 통계
    """
    if not os.path.exists(src):
        os.makedirs(src)
    if not os.path.exists(dst):
        os.makedirs(dst)

    stats = {
        'processed': 0,
        'moved': 0,
        'skipped': 0,
        'errors': 0,
        'duplicates': 0
    }

    # 중복 파일 로깅을 위한 파일 생성
    duplicate_log = Path(dst) / 'duplicate_files.log'

    for root, dirs, files in os.walk(src):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                stats['processed'] += 1

                try:
                    with open(file_path, 'r') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            print(f"Error decoding JSON in file: {file_path}")
                            stats['errors'] += 1
                            continue

                    # MD5 해시 추출
                    md5_hash = data.get('target', {}).get('file', {}).get('md5', '')

                    if not md5_hash:
                        print(f"MD5 hash not found in {file_path}")
                        stats['skipped'] += 1
                        continue

                    new_file_name = f"{md5_hash}.json"
                    new_file_path = os.path.join(dst, new_file_name)

                    # 파일이 이미 존재하는 경우 처리
                    if os.path.exists(new_file_path):
                        stats['duplicates'] += 1

                        # 기존 파일과 새 파일의 내용 비교
                        with open(new_file_path, 'r') as existing_f:
                            existing_content = existing_f.read()
                        with open(file_path, 'r') as new_f:
                            new_content = new_f.read()

                        if existing_content == new_content:
                            # 내용이 동일한 경우 - 로그만 남기고 스킵
                            with open(duplicate_log, 'a') as log:
                                log.write(f"{datetime.now()}: Identical file skipped - {file_path}\n")
                            os.remove(file_path)  # 중복 파일 제거
                            continue
                        else:
                            # 내용이 다른 경우 - 타임스탬프를 추가하여 새 이름으로 저장
                            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            new_file_name = f"{md5_hash}_{timestamp}.json"
                            new_file_path = os.path.join(dst, new_file_name)

                            # 로그 기록
                            with open(duplicate_log, 'a') as log:
                                log.write(
                                    f"{datetime.now()}: Different content found - {file_path} -> {new_file_name}\n")

                    # 파일 이동
                    shutil.move(file_path, new_file_path)
                    stats['moved'] += 1
                    print(f"Moved: {file_path} -> {new_file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    stats['errors'] += 1

    # 처리 결과 출력
    print("\nProcessing Summary:")
    print(f"Total files processed: {stats['processed']}")
    print(f"Successfully moved: {stats['moved']}")
    print(f"Duplicates found: {stats['duplicates']}")
    print(f"Skipped (no MD5): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")