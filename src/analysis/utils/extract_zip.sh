#!/bin/bash

# 사용법 함수
usage() {
    echo "사용법: $0 <zip_file> <destination_path> <number_of_files>"
    echo "  <zip_file>: 압축 해제할 ZIP 파일"
    echo "  <destination_path>: 파일을 압축 해제할 경로"
    echo "  <number_of_files>: 압축 해제할 파일의 개수"
    exit 1
}

# 매개변수 확인
if [ "$#" -ne 3 ]; then
    usage
fi

ZIP_FILE="$1"
DESTINATION="$2"
NUM_FILES="$3"

# 입력 확인
if [ ! -f "$ZIP_FILE" ]; then
    echo "오류: '$ZIP_FILE' 파일이 존재하지 않습니다."
    exit 1
fi

if [ ! -d "$DESTINATION" ]; then
    echo "'$DESTINATION' 디렉토리가 존재하지 않습니다. 생성합니다."
    mkdir -p "$DESTINATION"
fi

if ! [[ "$NUM_FILES" =~ ^[0-9]+$ ]]; then
    echo "오류: 파일 개수는 양의 정수여야 합니다."
    exit 1
fi

# 임시 파일 목록
TEMP_LIST="temp_file_list.txt"

# ZIP 파일 내용 나열 및 지정된 개수의 파일 선택
unzip -l "$ZIP_FILE" | tail -n +4 | head -n -2 | awk '{print $4}' | head -n "$NUM_FILES" > "$TEMP_LIST"

# 선택된 파일 압축 해제
while IFS= read -r file
do
    unzip -q "$ZIP_FILE" "$file" -d "$DESTINATION"
done < "$TEMP_LIST"

# 임시 파일 삭제
rm "$TEMP_LIST"

echo "압축 해제가 완료되었습니다. $NUM_FILES 개의 파일이 '$DESTINATION'에 압축 해제되었습니다."