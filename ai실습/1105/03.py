# *trOCR*

from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
import requests
from io import BytesIO
from PIL import Image
import unicodedata

# 한국어용 TrOCR 모델 및 프로세서 불러오기
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr", use_fast=False)
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

# 이미지 URL
url = "https://raw.githubusercontent.com/ddobokki/ocr_img_example/master/g.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

pixel_values = processor(img, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values, max_length=64)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(pixel_values.shape)
print(generated_ids)


# 한글 정규화
generated_text = unicodedata.normalize("NFC", generated_text)

print(generated_text)