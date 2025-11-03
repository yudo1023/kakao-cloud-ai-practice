from PIL import Image
import pytesseract

img = Image.open('1103/01_kor.png')
text = pytesseract.image_to_string(img, lang='kor+eng')

print("추출된 텍스트: ", text)