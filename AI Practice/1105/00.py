# *OCR 성능 평가지표*
# Character Error Rate (CER)
# 문자 단위로 오인식된 문자 개수를 전체 문자수로 나눈 비율
# Word Error Rate (WER)
# 단어 단위로 오인식된 단어의 비율


import pytesseract
import easyocr
import numpy as np
import editdistance

# CER 계산 함수
def cer(reference, hypothesis):
   ref = list(reference)
   hyp = list(hypothesis)
   distance = editdistance.eval(ref, hyp)
   cer_value = distance / max(len(ref), 1)
   return cer_value

# WER 계산 함수
def wer(reference, hypothesis):
   ref = reference.split()
   hyp = hypothesis.split()
   distance = editdistance.eval(ref, hyp)
   wer_value = distance / max(len(ref), 1)
   return wer_value

# 이미지 경로 또는 이미지 배열
image_path = '1103/sample.png'

# Tesseract OCR 인식
tesseract_text = pytesseract.image_to_string(image_path, lang='kor')

# EasyOCR 인식 (한국어 포함)
reader = easyocr.Reader(['ko', 'en'])
easyocr_result = reader.readtext(image_path, detail=0, paragraph=True)
easyocr_text = ' '.join(easyocr_result)

# 기준 정답 텍스트 (수정 필요)
ground_truth = """안녕하세요. OCR 인식 테스트입니다.
이 문장은 Tesseract와 OCR.space의 성능을 비교합니다.
Accuracy test with Korean + English MIXED text."""

# CER, WER 계산
print("Tesseract CER:", cer(ground_truth, tesseract_text))
print("Tesseract WER:", wer(ground_truth, tesseract_text))
print("EasyOCR CER:", cer(ground_truth, easyocr_text))
print("EasyOCR WER:", wer(ground_truth, easyocr_text))