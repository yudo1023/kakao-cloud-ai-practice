# *Google Vision API*

import os
from google.cloud import vision
import io

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-vision-api-key.json'

def detect_text(image_path):
   client = vision.ImageAnnotatorClient()

   with io.open(image_path, 'rb') as image_file:
       content = image_file.read()

   image = vision.Image(content=content)



   # 이미지에서 텍스트 감지
   response = client.text_detection(image=image)
   texts = response.text_annotations

   if texts:
       print('Detected text:')
       print(texts[0].description)  # 전체 인식된 텍스트 출력
   else:
       print('No text detected.')

   if response.error.message:
       raise Exception(f'{response.error.message}')

if __name__ == "__main__":
   image_file_path = '1103/01_1.png'  # 한글 텍스트가 포함된 이미지 경로
   detect_text(image_file_path)