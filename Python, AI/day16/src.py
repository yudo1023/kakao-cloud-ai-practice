# p.150
# 파일을 읽어서 -> 암호화 -> 복호화
def xor_encrypt_decrypt(input_file, output_file, key):
    try:
        with open(input_file, 'rb') as infile:
            data = infile.read
            #print(data)

        key_bytes = key.encode() if isinstance(key, str) else bytes([key])
        key_len = len(key_bytes)

        encryted_data = bytearray(len(data))
        for i in range(len(data)):
            encryted_data[i] = data[i] ^ key_bytes[i & key_len] # data/encryted_data의 길이와 key_bytes의 길이가 서로 다를 수 있어서

        with open(output_file, 'wb') as outfile:
            outfile.write(data)
    except Exception as e:
        print(f"오류 : {e}")


xor_encrypt_decrypt('example.txt', 'secret.enc', 'mykey123') # 암호화

xor_encrypt_decrypt('secret.enc', 'decrypted.txt', 'mykey123') # 복호화

# -- 텍스트 파일 처리--
# 텍스트 파일 읽기
with open('example.txt', 'r', encoding='utf-8') as file:
    for line in file:
        print(line, end='')

# 텍스트 파일 쓰기(한줄)
with open('output.txt', 'w', encoding='utf-8') as file:
    file.write('텍스트 파일 쓰기\n')

# 텍스트 파일 쓰기(여러줄)
lines = ['1.\n', '2.\n', '3.\n']
with open('output.txt', 'w', encoding='utf-8') as file:
    file.writelines(lines)

# 텍스트 파일 내용 추가
with open('output.txt', 'a', encoding='utf-8') as file:
    file.write('내용 추가\n')

# -- 바이너리 파일 처리--
# 바이너리 파일 읽기
with open('image.jpg', 'rb') as file:
    binary_data = file.read()
    print(f"파일 크기 : {len(binary_data)} 바이트")

# 바이너리 파일 쓰기
data = bytes([0x48, 0x65, 0x6C, 0x6C, 0x6F])
with open('binary_data.bin', 'wb') as file:
    file.write(data)
    print("바이너리 데이터를 파일에 기록했습니다.")

# 이미지 파일 복사하기
def copy_image(source_file, destination_file):
    try:
        with open(source_file, 'rb') as source:
            image_data = source.read()

        with open(destination_file, 'wb') as destination:
            destination.write(image_data)

        print(f"{source_file}이(가) {destination_file}로 복사되었습니다.")
        return True
    except FileNotFoundError:
        print(f"오류 : '{source_file}'파일을 찾을 수 없습니다.")
        return False
    except Exception as e:
        print(f"오류 발생 : {e}")
        return False
    
# 바이너리 데이터 문자열 처리
import base64

binary_data = b'Hello, binary world!'

encoded = base64.b64encode(binary_data)
print(f"Base64 인코딩 : {encoded}")
print(f"문자열로 변환 : {encoded.decode('ascii')}")

decoded = base64.b64decode(encoded)
print(f"원본 바이너리 데이터 : {decoded}")

# -- CSV 파일 처리--
# CSV 파일 읽기(기본)
import csv

with open('data.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)

    header = next(csv_reader)
    print(f"컬럼 : {header}")

    for row in csv_reader:
        print(f"데이터 : {row}")

# CSV 파일 읽기(딕셔너리 형태로)
with open('data.csv', 'r', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)

    for row in csv_reader:
        print(f"이름 : {row['이름']}, 나이 : {row['나이']}, 도시 : {row['도시']}")

# CSV 파일 쓰기
header = ['이름', '나이', '도시']
data = [
    ['홍길동', '30', '서울'],
    ['김철수', '25', '부산'],
    ['이영희', '28', '인천']
]

with open('new_data.csv', 'w', encoding='utf-8', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow(header)
    csv_writer.writerows(data)

    print("CSV 파일 생성 완료")

# -- JSON 파일 처리--
# JSON 파일 읽기
import json

with open('test.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)
    print(json.dumps(json_data))

# JSON 파일 쓰기
data = {
    "이름" : "홍길동",
    "나이" : 30,
    "도시" : "서울"
}

with open('new_date.json', 'w', encoding='utf-8') as file:
    json.dump(data, file)


# 파일이 존재하지 않거나, 권한이 없거나, 형식이 잘못된 경우 등 다양한 오류 상황을 적절히 처리

try:
    with open() as file:
        data = file.read()

# 파일이 존재하지 않을 경우
except FileNotFoundError as e:
    print(f"파일이 존재하지 않습니다. {e}")

# 파일 권한이 없는 경우
except PermissionError as e:
    print(f"파일 권한이 없습니다. {e}")

# 파일 형식이 잘못된 경우
except ValueError as e:
    print(f"파일 형식이 잘못되었습니다. {e}")