#!/bin/bash

# Cuckoo 결과 디렉토리 설정
CUCKOO_RESULT_DIR="$HOME/.cuckoo/storage/analyses"

# HOME 디렉토리에 임시 작업 폴더 생성
TEMP_DIR="$HOME/cuckoo_temp_$(date +%s)"
mkdir -p "$TEMP_DIR"

# 현재 날짜와 시간을 파일명에 사용
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_ZIP="cuckoo_reports_$TIMESTAMP.zip"

# JSON에서 값을 추출하는 함수
extract_json_value() {
    sed -n 's/.*"'"$1"'"\s*:\s*"\([^"]*\)".*/\1/p' "$2" | head -n 1
}

# 모든 분석 폴더를 순회
for analysis_dir in "$CUCKOO_RESULT_DIR"/*; do
    if [ -d "$analysis_dir" ]; then
        report_file="$analysis_dir/reports/report.json"

        if [ -f "$report_file" ]; then
            # report.json에서 MD5 해시와 PE 파일명 추출
            md5_hash=$(extract_json_value "md5" "$report_file")

            if [ -n "$md5_hash" ]; then
                # 새 파일명 생성 (MD5_PE이름.json)
                new_filename="${md5_hash}.json"

                # 파일 복사 및 이름 변경
                mv "$report_file" "$TEMP_DIR/$new_filename"

                echo "처리됨: $new_filename"
            else
                echo "경고: $report_file 에서 MD5 해시 또는 PE 이름을 추출할 수 없습니다."
            fi
        else
            echo "경고: $analysis_dir 에 report.json 파일이 없습니다."
        fi
    fi
done

# 모든 처리된 파일을 하나의 ZIP 파일로 압축
(cd "$TEMP_DIR" && zip -r "$HOME/$OUTPUT_ZIP" .)

# 임시 디렉토리 삭제
rm -rf "$TEMP_DIR"

echo "모든 처리가 완료되었습니다. 결과 파일: $HOME/$OUTPUT_ZIP"