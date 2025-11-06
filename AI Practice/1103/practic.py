# %%
from PIL import Image
import pytesseract
import requests
import fitz
import io
import Levenshtein
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def ocr_space_api(image_path, api_key='K85097285888957', language='kor'):
    url_api = "https://api.ocr.space/parse/image"

    with open(image_path, 'rb') as f:
        response = requests.post(
            url_api,
            files={"filename": f},
            data={"apikey": api_key, "language": language},
            timeout=30
        )

    try:
        response.raise_for_status()
    except requests.HTTPError:
        print("HTTP error:", response.status_code)
        print(response.text)
        return None

    # JSON으로 파싱
    try:
        result = response.json()
    except ValueError:
        print("응답이 JSON이 아님:", response.text)
        return None

    # 에러 여부 검사
    if result.get("IsErroredOnProcessing"):
        print("OCR 처리 에러:", result.get("ErrorMessage") or result.get("ErrorDetails"))
        return None

    parsed = result.get("ParsedResults")
    if not parsed:
        print("ParsedResults 없음 - 응답:", result)
        return None

    text_detected = parsed[0].get("ParsedText", "")
    return text_detected

def extract_text_from_pdf(pdf_path):
   doc = fitz.open(pdf_path)
   full_text = ""

   for page_num in range(len(doc)):
       page = doc.load_page(page_num)
       text = page.get_text("text").replace('$', ' ')
       full_text += text

       image_list = page.get_images()
       for img_index, img in enumerate(image_list):
           xref = img[0]
           base_image = doc.extract_image(xref)
           image_bytes = base_image["image"]
           img_pil = Image.open(io.BytesIO(image_bytes))

           img_text = pytesseract.image_to_string(img_pil, lang='kor+eng')
           full_text += "\n[이미지 내 텍스트]\n" + img_text + "\n"

   return full_text

def calculate_accuracy(original_text, ocr_text):
   distance = Levenshtein.distance(original_text, ocr_text)
   max_len = max(len(original_text), len(ocr_text))
   accuracy = (1 - distance / max_len) * 100
   return accuracy

img = Image.open('sample.png')
tesseract_text = pytesseract.image_to_string(img, lang='kor+eng')
api_text = ocr_space_api('sample.png')
pdf_text = extract_text_from_pdf('sample.pdf')

original = """안녕하세요. OCR 인식 테스트입니다.
이 문장은 Tesseract와 OCR.space의 성능을 비교합니다.
Accuracy test with Korean + English MIXED text."""

original_pdf = """OCR 비교 실험용 문서입니다.
이 문장은 PDF 내부의 텍스트로 존재합니다.

[이미지 내 텍스트]
안녕하세요. OCR 인식 테스트입니다.
이 문장은 Tesseract와 OCR.space의 성능을 비교합니다.
Accuracy test with Korean + English MIXED text."""


tesseract_accuracy = calculate_accuracy(original, tesseract_text)
api_accuracy = calculate_accuracy(original, api_text)
pdf_accuracy = calculate_accuracy(original_pdf, pdf_text)

print("\n" + "="*60)
print("1. 이미지 내 텍스트 (Tesseract)")
print("="*60)
print(tesseract_text.strip())
print(f"\n➡️ 정확도: {tesseract_accuracy:.2f}%")

print("\n" + "="*60)
print("2. 이미지 내 텍스트 (OCR.space)")
print("="*60)
print(api_text.strip())
print(f"\n➡️ 정확도: {api_accuracy:.2f}%")

print("\n" + "="*60)
print("3. PDF 텍스트 및 이미지 내 텍스트 (PyMuPDF + Tesseract)")
print("="*60)
print(pdf_text.strip())
print(f"\n➡️ 정확도: {pdf_accuracy:.2f}%\n")

font_path = 'NanumGothic.ttf'
font_prop = fm.FontProperties(fname=font_path, size=12)

image_names = ['tesseract', 'ocr-space', 'pyMuPDF']
acuuracies = [tesseract_accuracy, api_accuracy, pdf_accuracy]

plt.bar(image_names, acuuracies)
plt.title("OCR 인식률 비교", fontproperties=font_prop)
plt.xlabel("이미지 이름", fontproperties=font_prop)
plt.ylabel("인식률(%)", fontproperties=font_prop)
plt.ylim(0,100)
plt.show()


# %%
