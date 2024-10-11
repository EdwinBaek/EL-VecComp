#!/bin/bash

# 사용법 확인
if [ "$#" -ne 4 ]; then
    echo "사용법: $0 <소스_폴더> <대상_폴더> <생성할_폴더_수> <폴더_접두사>"
    exit 1
fi

SOURCE_DIR="$1"
DEST_DIR="$2"
NUM_FOLDERS="$3"
FOLDER_PREFIX="$4"

# 대상 폴더 생성
mkdir -p "$DEST_DIR"

# 각 대상 폴더 생성
for i in $(seq 1 $NUM_FOLDERS); do
    mkdir -p "$DEST_DIR/${FOLDER_PREFIX}_$i"
done

# 파일 개수 확인
total_files=$(find "$SOURCE_DIR" -maxdepth 1 -type f | wc -l)
files_per_folder=$((total_files / NUM_FOLDERS + 1))

echo "총 파일 수: $total_files"
echo "각 폴더당 파일 수: $files_per_folder"

# 진행 상황 표시 함수
show_progress() {
    local current=$1
    local total=$2
    local percent=$((current * 100 / total))
    local completed=$((percent / 2))
    local remaining=$((50 - completed))

    printf "\r[%-${completed}s%-${remaining}s] %d%%" "$(printf '#%.0s' $(seq 1 $completed))" "$(printf '.%.0s' $(seq 1 $remaining))" "$percent"
}

# 파일 이동
count=0
folder_index=1
processed_files=0

find "$SOURCE_DIR" -maxdepth 1 -type f | while read file; do
    mv "$file" "$DEST_DIR/${FOLDER_PREFIX}_$folder_index/"
    count=$((count + 1))
    processed_files=$((processed_files + 1))

    if [ $count -eq $files_per_folder ] && [ $folder_index -lt $NUM_FOLDERS ]; then
        count=0
        folder_index=$((folder_index + 1))
    fi

    # 진행 상황 표시
    show_progress $processed_files $total_files
done

echo -e "\n파일 분배가 완료되었습니다."