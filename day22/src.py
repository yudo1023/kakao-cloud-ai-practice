import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras import layers, models
import pytesseract
import cv2
import re

# # OCR 전처리 예제
# class OCRPreprocessor:
#     # 그레이스케일 변환(3채널 기준)
#     def convert_to_grayscale(self, image):
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image.copy()
        
#         return gray

#     # 이미지 사이즈 변환
#     def resize_image(self, image, target_height=800):
#         h, w = image.shape[:2]
#         if h < target_height:
#             scale = target_height / h
#             new_w = int(w*scale)
#             resized = cv2.resize(image, (new_w, target_height),
#                                  interpolation=cv2.INTER_CUBIC)
#         else:
#             resized = image

#         return resized
    
#     # 이진화 처리
#     def apply_threshold(self, image, method='adaptive'):
#         gray = self.convert_to_grayscale(image)

#         if method == 'simple':
#             _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         elif method == 'adaptive':
#             # 가중 평균 임계값으로 사용
#            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                   cv2.THRESH_BINARY, 11, 2)
#         elif method == 'otsu':
#             # 히스토그램에 대한 이진화
#             _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         return thresh
    
#     # 노이즈 제거
#     def remove_noise(self, image):
#         kernel = np.ones((3,3), np.uint8)
#         # 열림 연산(작은 노이즈 점 제거)
#         opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
#         # 닫힘 연산(텍스트 내부의 작은 구멍 매움)
#         closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#         # 가우시안 블러
#         denoised = cv2.GaussianBlur(closing, (3, 3), 0)

#         return denoised
    
#     # 기울기 보정
#     def correct_skew(self, image):
#         edges = cv2.Canny(image, 50, 150, apertureSize=3)
#         lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

#         if lines is not None:
#             angles = []
#             for rho, theta in lines[:, 0]:
#                 # 허프 변환에서 theta 0도 : 수직선, 90도 : 수평선, 180도 : 다시 수직선
#                 angle = np.degrees(theta) - 90
#                 angles.append(angle)
#             # 평균, 중앙값
#             median_angle = np.median(angles)
#             (h, w) = image.shape[:2]
#             center = (w //2, h//2)
#             M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

#             rotated = cv2.warpAffine(image, M, (w, h),
#                                      flags=,
#                                      borderMode=cv2.BORDER_REPLICATE)

#     def preprocess_pipeline(self, image):
#         # 단계별 저장
#         steps = []
#         step_names = []

#         # 0. 오리지널 이미지
#         steps.append(image.copy())
#         step_names.append('원본 이미지')

#         # 1. 그레이스케일 변환
#         gray = self.convert_to_grayscale(image)
#         steps.append(gray)
#         step_names.append('그레이스케일')

#         # 2. 이미지 사이즈 변환
#         resized = self.resize_image(gray)
#         steps.append(resized)
#         step_names.append('크기 조정')

#         # 3. 이진화 처리
#         thresh = self.apply_threshold(resized)
#         steps.append(thresh)
#         step_names.append('이진화')

#         # 4. 노이즈 제거
#         denoised = self.remove_noise(thresh)
#         steps.append(denoised)
#         step_names.append('노이즈 제거')

#         # 5. 기울기 보정
#         corrected = self.correct_skew(denoised)
#         steps.append(corrected)
#         step_names.append('기울기 보정')

#         return corrected

# # 샘픙 이미지
# def create_noisy_sample_image():
#     image = np.ones((300, 800, 3), dtype=np.uint8) * 255
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(image, 'test1', (50, 100), font, 1.5, (0,0,0), 2)
#     cv2.putText(image, 'test2', (50, 150), font, 1, (0,0,0), 2)
#     cv2.putText(image, 'test3', (50, 200), font, 1, (0,0,0), 2)

#     noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
#     noisy_image = cv2.add(image, noise)
#     h, w = noisy_image.shape[:2]
#     center = (w//2, h//2)
#     M = cv2.getRotationMatrix2D(center, 5, 1.0)
#     skewed_image = cv2.warpAffine(noisy_image, M, (w, h))

#     return skewed_image

# image = create_noisy_sample_image()
# original_text = pytesseract.image_to_string(image)
# print(original_text)

# # ocr 전처리
# preprocessor = OCRPreprocessor()
# prorocessed_image = preprocessor.preprocess_pipeline(image)
# print("전처리 후")
# processed_text=pytesseract.image_to_string(prorocessed_image)
# print(processed_text)


# Tesseract예제
class TesseractOCR:
    def get_available_languages(self):
        langs = pytesseract.get_languages()
        return langs
    
    def ocr_with_box(self, image, lang='eng'):
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        result_image = image.copy()

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 60:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['heifht'][i]
                text = data['text'][i]

                if text.strip():
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(result_image, text, (x, y -10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return result_image, data
        

class DocumentOCRDemo:
    def __init__(self):
        self.ocr = TesseractOCR()

    # 영수증 전처리
    def preprocess_receipt(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,5))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    # 샘플 영수증 이미지
    def create_sample_receipt(self):
        image = np.ones((600, 400, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [
            ('GROCERY STORE', (100, 50), 0.8, 2),
            ('Date: 01/15/2024', (50, 100), 0.6, 1),
            ('Item 1.      $5.99', (50, 150), 0.6, 1),
            ('Item 2.      $3.90', (50, 180), 0.6, 1),
            ('Item 3.      $12.25', (50, 210), 0.6, 1),
            ('Tax          $1.73', (50, 260), 0.6, 1),
            ('Total        $23.47', (50, 310), 0.7, 2),
            ('Thank you!', (120, 360), 0.6, 1),
        ]

        for text, pos, scale, thickness in texts:
            cv2.putText(image, text, pos, font, scale, (0, 0, 0), thickness)

        return image
    
    # 전처리된 영수증 이미지 가져오기
    def process_receipt_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            image = self.create_sample_receipt()
        processed = self.preprocess_receipt(image)
        config = r'--oem 3 --psm 4 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwyz.,:-$'
        text = pytesseract.image_to_string(processed, config=config)
        parsed_info = self.parse_receipt_image(text)

        return text, parsed_info, processed

    # 영수증 정보 파싱
    def parse_receipt_image(self, text):
        lines = text.split('\n')
        info = {
            'items':[],
            'total':None,
            'date':None,
            'store_name':None
        }
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 가격 패턴 찾기 ($xx.xx)
            price_pattern = r'\$?\d+\.\d{2}'
            if re.search(price_pattern, line):
                if 'total' in line.lower() or 'sum' in line.lower():
                    prices = re.findall(price_pattern, line)
                    if prices:
                        info['total'] = prices[-1]
                else:
                    info['items'].append(line)
            
            # 날짜 패턴 찾기 (MM/DD/YYYY)
            date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
            if re.search(date_pattern, line):
                info['date'].append(line)

    
ocr = TesseractOCR()
print("지원 언어: ", ocr.get_available_languages()[:10])

demo = DocumentOCRDemo()
receipt_text, parsed_info, proceesed_image = demo.process_receipt_image(None)
print("영수증 텍스트:")
print(receipt_text)
print("\n파싱된 정보")
for key, value in parsed_info.items():
    print(f"{key}: {value}")

sample_image = demo.create_sample_receipt()

plt.rc('font', family='Apple Gothic')
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))



# torch 예제
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import string
import random

class CRNN(nn.Module):
    def __init__(self, img_height, img_width, num_chars, num_classes, rnn_hidden=256):
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.num_chars = num_chars
        self.num_classes = num_classes

        # cnn
        self.cnn = nn.Sequential([
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), # 메모리 절약을 위해 연산 결과를 입력 텐서에 직접 저장
            nn.MaxPool2d(2,2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)), # 8x32 -> 4x32 (높이만 축소)

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2,1)), # 4x32 -> 2x32 (높이만 축소)

            nn.conv2d(512, 512, 2, 1, 0),
            nn.ReLU()

        ])

        # rnn
        self.rnn = nn.LSTM(512, rnn_hidden, bidirectional=True, batch_first=True)

        self.linear = nn.Linear(rnn_hidden*2, num_classes)
        

class SynthericTextDataset(Dataset):
    def __init__(self, num_samples=1000, img_height=32, img_width=128, max_text_len=100):
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_len = max_text_len

        self.chars = string.digits

        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self)}
        self.char_to_idx['<blank'] = 0 # CTC blank 토큰

        self.idx_to_char = {idx: char+1 for char, idx in self.char_to_idx.items()}
        self.num_classes = len(self.chars) + 1

        self.transform = transforms.Compose([
            transforms.ToTensor(), # PIL Image -> Tensor
            transforms.Normalize((0.5,),(0.8,))
        ])

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 랜덤 텍스트 생성
        # 텍스트로 부터 이미지 생성
        # 텍스트로 부터 라벨 생성

        return Image.fromarray(img_array)
    
class CRNNTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        self.optimizer = optim.adam(model.parameters(), lr=0.0005, weight_decay=13-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5) # 3 에포크마다 학습률 감소, 학습률은 0.5배로 감소
    
    def train_epoch(self, dataloader):
        self.model.train() # 훈련 모드 설정
        # self.model.eval() # 평가 모드 설정
        total_loss = 0
        num_batches = 0

        for batch_idx, (images, targets, texts) in enumerate(dataloader):
            images = images.to(self.device)
            targets = [target.to(self.device) for target in targets]

            # forward
            self.optimizer.zero_grad() # 그래디언트 초기화
            outputs = self.model(images)
            outputs = outputs.permute(1, 0, 2) # CTC 손실 계산을 위한 차원 순서 변경
            input_lengths = torch.full((images.size(0), ), outputs.size(0), dtype=torch.long)
            target_lengths = torch.tensor([len(target) from target in targets], dtype=torch.long)
            target_id = torch.cat(targets)
            loss = self.criterion()
            loss.backward()
            torch.nn.Unflatten.clip_grad_norm()

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1



def collate_fn(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    labels = list(labels)
    return images, labels, texts

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 디바이스', {device})


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, co)

model = CRNN(
    img_height=32,
    img_width=128,
    num_chars=len(train_dataset.chars),
    num_classes=train_dataset.num_classes
)

print(f"모델 파라미터 수: {sum(p.numel())}")

# train
triner = CRNNTrainer(model, device)
num_epochs = 5
for epoch in range(num_epochs):
