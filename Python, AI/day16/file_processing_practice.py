# p.216
# 1. 목표 : 파일 처리기 구헌
# 2. 요구사항
# 다양한 유형의 파일(텍스트, CSV, JSON, 바이너리)을 읽고 쓸 수 있어야 합니다
# 파일이 존재하지 않거나, 권한이 없거나, 형식이 잘못된 경우 등 다양한 오류 상황을 적절히 처리
# 사용자 정의 예외 계층 구조를 설계하고 구현
# 오류 발생 시 로깅을 통해 문제를 기록
# 모든 파일 작업은 컨텍스트 매니저(`with` 구문)를 사용

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="app.log",
    encoding="utf-8",
)
logger = logging.getLogger("fileproc")

# --- 사용자 정의 예외 계층 --- 
class FileProcessingError(Exception):
    pass

class FileOpenError(FileProcessingError):
    pass

class FilePermissionError(FileProcessingError):
    pass

class FileFormatError(FileProcessingError):
    pass

class FileIOError(FileProcessingError):
    pass


def _translate_open_errors(path, mode, encoding=None, newline=None):
    try:
        return open(path, mode, encoding=encoding, newline=newline)
    except FileNotFoundError as e:
        logger.error(f"[파일 존재 오류] '{path}' 경로를 찾을 수 없습니다.", exc_info=True)
        raise FileOpenError(f"파일을 찾을 수 없습니다: {path}") from e
    except PermissionError as e:
        logger.error(f"[파일 권한 오류] '{path}' 파일 접근 권한이 없습니다.", exc_info=True)
        raise FilePermissionError(f"권한이 없습니다: {path}") from e
    except OSError as e:
        logger.error(f"[I/O 오류] '{path}' 파일 열기 실패: {e}", exc_info=True)
        raise FileIOError(f"I/O 오류: {path} ({e})") from e


# --- 텍스트 파일 관리 ---
# 텍스트 파일 읽기
def read_text_file(read_file):
    logger.debug("텍스트 파일 읽기 시작")
    path = Path(read_file)
    with _translate_open_errors(path, "r", encoding="utf-8") as file:
        data = file.read()
    logger.info(f"{path} 텍스트 파일 읽기 성공")
    return data

# 텍스트 파일 쓰기
def write_text_file(write_file):
    logger.debug("텍스트 파일 쓰기 시작")
    path = Path(write_file)
    content = ["텍스트 파일 쓰기\n", "1.\n", "2.\n", "3.\n", "내용 추가\n"]
    with _translate_open_errors(path, "w", encoding="utf-8") as file:
        file.writelines(content)
    logger.info(f"{path} 텍스트 파일 쓰기 성공")


# --- 바이너리 파일 관리 --- 
# 바이너리 파일 읽기
def read_binary_file(read_file):
    logger.debug("바이너리 파일 읽기 시작")
    path = Path(read_file)
    with _translate_open_errors(path, "rb") as file:
        data = file.read()
    logger.info(f"{path} 바이너리 파일 읽기 성공 (크기: {len(data)} 바이트)")
    return data

# 바이너리 파일 쓰기
def write_binary_file(write_file):
    logger.debug("바이너리 파일 쓰기 시작")
    path = Path(write_file)
    payload = bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F])  # "Hello"
    with _translate_open_errors(path, "wb") as file:
        file.write(payload)
    logger.info(f"{path} 바이너리 파일 쓰기 성공")


# --- CSV 파일 관리 --- 
# CSV 파일 읽기
def read_csv_file(read_file):
    logger.debug("CSV 파일 읽기 시작")
    path = Path(read_file)
    try:
        with _translate_open_errors(path, "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            try:
                header = next(reader)
            except StopIteration:
                logger.warning(f"{path} CSV 헤더가 비어 있습니다.")
                return [], []
            rows = [row for row in reader]
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"[CSV 기타 오류] {path} 읽기 실패: {e}", exc_info=True)
        raise FileIOError(f"CSV 읽기 오류: {path} ({e})") from e

    logger.info(f"{path} CSV 파일 읽기 성공 (rows={len(rows)})")
    return header, rows

# CSV 파일 쓰기
def write_csv_file(write_file):
    logger.debug("CSV 파일 쓰기 시작")
    path = Path(write_file)
    header = ["이름", "나이", "도시"]
    rows = [
        ["홍길동", "30", "서울"],
        ["김철수", "25", "부산"],
        ["이영희", "28", "인천"],
    ]
    try:
        with _translate_open_errors(path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"[CSV 기타 오류] {path} 쓰기 실패: {e}", exc_info=True)
        raise FileIOError(f"CSV 쓰기 오류: {path} ({e})") from e

    logger.info(f"{path} CSV 파일 쓰기 성공 (rows={len(rows)})")


# --- JSON 파일 관리 --- 
# JSON 파일 읽기
def read_json_file(read_file):
    logger.debug("JSON 파일 읽기 시작")
    path = Path(read_file)
    try:
        with _translate_open_errors(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        logger.error(f"[파일 형식 오류] {path} 는 JSON 형식이 아닙니다: {e}", exc_info=True)
        raise FileFormatError(f"JSON 형식 오류: {path} ({e})") from e
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"[JSON 기타 오류] {path} 읽기 실패: {e}", exc_info=True)
        raise FileIOError(f"JSON 읽기 오류: {path} ({e})") from e

    logger.info(f"{path} JSON 파일 읽기 성공")
    return data

# JSON 파일 쓰기
def write_json_file(write_file):
    logger.debug("JSON 파일 쓰기 시작")
    path = Path(write_file)
    data = {"이름": "홍길동", "나이": 30, "도시": "서울"}
    try:
        with _translate_open_errors(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except FileProcessingError:
        raise
    except Exception as e:
        logger.error(f"[JSON 기타 오류] {path} 쓰기 실패: {e}", exc_info=True)
        raise FileIOError(f"JSON 쓰기 오류: {path} ({e})") from e

    logger.info(f"{path} JSON 파일 쓰기 성공")



write_text_file("hello.txt")
print("\n*텍스트 파일 내용*")
print(read_text_file("hello.txt"))

write_binary_file("hello.bin")
binary_data = read_binary_file("hello.bin")
print("\n*바이너리 파일 내용*")
print(f"크기: {len(binary_data)} 바이트")
print("데이터:", binary_data.decode("utf-8"))

write_csv_file("people.csv")
header, rows = read_csv_file("people.csv")
print("\n*CSV 파일 내용*")
print("컬럼:", header)
for row in rows:
    print("데이터:", row)

write_json_file("data.json")
data = read_json_file("data.json")
print("\n*JSON 파일 내용*")
import json
print(json.dumps(data, ensure_ascii=False, indent=2))



    
