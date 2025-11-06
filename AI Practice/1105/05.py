# *KAKAO OCR API*

import requests
import json

def kakao_ocr_receipt(image_path, app_key):
   url = "https://dapi.kakao.com/v2/vision/text/ocr"  
   # 카카오 OCR API URL (일반 텍스트 OCR)

   # 이미지 파일 읽기
   with open(image_path, 'rb') as image_file:
       image_data = image_file.read()

   # 요청 헤더에 앱 키 설정
   headers = {
       'Authorization': f'KakaoAK {app_key}',
   }

   # 요청 파일 데이터 설정 (multipart/form-data)
   files = {
       'image': image_data,
   }

   # API 요청
   response = requests.post(url, headers=headers, files=files)

   # 응답 JSON으로 파싱
   result_json = response.json()

   # 인식된 텍스트 출력
   if response.status_code == 200:
       texts = []
       for item in result_json.get('result', []):
           for word_info in item.get('recognition_words', []):
               texts.append(word_info)
       return texts
   else:
       print("Error:", response.status_code, result_json)
       return None

if __name__ == "__main__":
   # 본인의 카카오 REST API 키 입력
   app_key = "YOUR_KAKAO_REST_API_KEY"
   # 영수증 이미지 경로 입력
   image_path = "1103/01_2.png"

   ocr_texts = kakao_ocr_receipt(image_path, app_key)
   if ocr_texts:
       print("=== 인식된 텍스트 ===")
       for text in ocr_texts:
           print(text)

