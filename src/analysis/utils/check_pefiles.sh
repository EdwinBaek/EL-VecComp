#!/bin/bash

# 사용법 함수
usage() {
    echo "사용법: $0 <소스_디렉토리> <목적지_디렉토리>"
    echo "예: $0 ./source_folder ./non_pe_files"
    exit 1
}

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
    echo "$1"
}

# 인자 확인
if [ "$#" -ne 2 ]; then
    usage
fi

# 디렉토리 설정
SOURCE_DIR="$1"
DEST_DIR="$2"
LOG_FILE="pe_sorter_log.txt"

# 디렉토리 존재 확인
if [ ! -d "$SOURCE_DIR" ]; then
    log "오류: 소스 디렉토리가 존재하지 않습니다: $SOURCE_DIR"
    exit 1
fi

# 목적지 디렉토리가 없으면 생성
mkdir -p "$DEST_DIR"

# pefile 라이브러리 설치 (Python 2용으로 설치)
python -m pip install pefile capstone

# Python 스크립트 생성 (Python 2 문법으로 작성)
cat << EOF > pe_checker.py
import os
import sys
import pefile

def is_pe_file(file_path):
    try:
        pefile.PE(file_path)
        return True
    except:
        return False

if __name__ == "__main__":
    file_path = sys.argv[1]
    if is_pe_file(file_path):
        print "PE"
    else:
        print "NON-PE"
EOF

# 파일 카운터 초기화
total_files=0
pe_files=0
non_pe_files=0

log "작업 시작: 소스 디렉토리 $SOURCE_DIR, 목적지 디렉토리 $DEST_DIR"

# 소스 디렉토리의 모든 파일을 순회
for file in "$SOURCE_DIR"/*; do
    if [ -f "$file" ]; then
        total_files=$((total_files + 1))

        # Python 스크립트를 실행하여 파일이 PE 형식인지 확인
        result=$(python pe_checker.py "$file")

        if [ "$result" = "NON-PE" ]; then
            # PE 형식이 아니면 파일을 DEST_DIR로 이동
            mv "$file" "$DEST_DIR/"
            if [ $? -eq 0 ]; then
                log "이동함: $file -> $DEST_DIR/$(basename "$file")"
                non_pe_files=$((non_pe_files + 1))
            else
                log "오류: 파일 이동 실패: $file"
            fi
        else
            log "PE 파일 (원본 유지): $file"
            pe_files=$((pe_files + 1))
        fi
    fi
done

# 임시 Python 스크립트 삭제
rm pe_checker.py

log "작업 완료"
log "총 처리된 파일: $total_files"
log "PE 파일 수: $pe_files"
log "non-PE 파일 수: $non_pe_files"
log "non-PE 파일은 $DEST_DIR 에 복사되었습니다."
log "모든 원본 파일은 $SOURCE_DIR 에 그대로 유지됩니다."