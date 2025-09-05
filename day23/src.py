# # ** CRNN **
# # OCR 기초_p.97crnn 예제
# from difflib import SequenceMatcher
# from typing import Dict, List, Union
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from PIL import Image, ImageDraw, ImageFont

# plt.rc('font', family='Apple SD Gothic Neo')

# class CRNN(nn.Module):
#     def __init__(self, img_height, img_width, num_chars, num_classes, rnn_hidden=256):
#         super().__init__()

#         self.img_height = img_height
#         self.img_width = img_width
#         self.num_chars = num_chars
#         self.num_classes = num_classes

#         # cnn
#         self.cnn = nn.Sequential([
#             nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True), # 메모리 절약을 위해 연산 결과를 입력 텐서에 직접 저장
#             nn.MaxPool2d(2,2),

#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2,2),

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((2,1)), # 8x32 -> 4x32 (높이만 축소)

#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d((2,1)), # 4x32 -> 2x32 (높이만 축소)

#             nn.conv2d(512, 512, 2, 1, 0),
#             nn.ReLU()

#         ])

#         # rnn
#         self.rnn = nn.LSTM(512, rnn_hidden, bidirectional=True, batch_first=True)

#         self.linear = nn.Linear(rnn_hidden*2, num_classes)

#     # 순전파  
#     def forward(self, x):
#         conv_features = self.cnn(x)
#         b, c, h ,w = conv_features.size()
#         conv_features = conv_features.view(b, c*h, w)
#         conv_features = conv_features.permute(0,2,1)
#         rnn_out, _ = self.rnn(conv_features) # RNN으로 순차 정보 처리
#         # 출력 레이어
#         output = self.linear(rnn_out)
#         output = F.log_softmax(output, dim=2)

#         return output


# class SyntheticTextDataset(Dataset):
#     def __init__(self, num_samples=1000, img_height=32, img_width=128, max_text_len=6):
#         self.num_samples = num_samples
#         self.img_height = img_height
#         self.img_width = img_width
#         self.max_text_len = max_text_len

#         self.chars = string.digits

#         self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.chars)}
#         self.char_to_idx['<blank>'] = 0 # CTC blank 토큰

#         self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
#         self.num_classes = len(self.chars) + 1

#         self.transform = transforms.Compose([
#             transforms.ToTensor(), # PIL Image -> Tensor
#             transforms.Normalize((0.5,),(0.8,))
#         ])

#     def __len__(self):
#         return self.num_samples
    
#     def __getitem__(self, idx):
#         # 랜덤 텍스트 생성
#         text_len = random.randint(2, min(4, self.max_text_len))
#         text = ''.join(random.choices(self.chars, k=text_len))
#         # 텍스트로 부터 이미지 생성
#         image = self.create_text_image(text)
#         # 텍스트를 인덱스 라벨로 반환
#         label = [self.char_to_idx[char] for char in text]
#         # 텐서 변환
#         if self.transform:
#             image = self.transform(image)

#         return image, torch.tensor(label, dtype=torch.long), text
    
#     # 텍스트로 이미지 생성
#     def create_text_image(self, text):
#         img = Image.new('L', (self.img_width, self.img_height), 255)
#         draw = ImageDraw.Draw(img)
#         try:
#             font = ImageFont.truetype("arial.ttf", 24)
#         except:
#             font = ImageFont.load_default()

#         try:
#             bbox = draw.textbbox((0, 0), text, font=font)
#             text_width = bbox[2] - bbox[0]
#             text_height = bbox[3] - bbox[1]
#         except AttributeError:
#             text_width, text_height = draw.textsize(text, font=font)

#         x = (self.img_width - text_width) // 2
#         y = (self.img_height - text_height) // 2

#         draw.text((x, y), text, fill=0, font=font)

#         img_array = np.array(img)
#         noise = np.random.normal(0, 5, img_array.shape) # 가우시안 노이즈
#         img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

#         return Image.fromarray(img_array)
    
# class CRNNTrainer:
#     def __init__(self, model, device='cpu'):
#         self.model = model.to(device)
#         self.device = device
#         self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
#         self.optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
#         self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5) # 3 에포크마다 학습률 감소, 학습률은 0.5배로 감소
    
#     def train_epoch(self, dataloader):
#         self.model.train() # 훈련 모드 설정
#         # self.model.eval() # 평가 모드 설정
#         total_loss = 0
#         num_batches = 0

#         for batch_idx, (images, targets, texts) in enumerate(dataloader):
#             images = images.to(self.device)
#             targets = [target.to(self.device) for target in targets]

#             # forward
#             self.optimizer.zero_grad() # 그래디언트 초기화
#             outputs = self.model(images)
#             outputs = outputs.permute(1, 0, 2) # CTC 손실 계산을 위한 차원 순서 변경
#             input_lengths = torch.full((images.size(0), ), outputs.size(0), dtype=torch.long)
#             target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
#             target_id = torch.cat(targets)
#             loss = self.criterion(outputs, target_id, input_lengths, target_lengths)
#             loss.backward()
#             torch.nn.Unflatten.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

#             self.optimizer.step()

#             total_loss += loss.item()
#             num_batches += 1

#             if batch_idx % 50 == 0:
#                 print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

#             # 학습률 스케줄링
#             self.scheduler.step()
#             return total_loss / num_batches
        
#         # CTC 디코딩
#         def decode_prediction(self, output, dataset):
#             pred_indices = torch.argmax(output, dim=2)
#             decoded_texts = []
#             for batch_idx in range(pred_indices.size(0)):
#                 indices = pred_indices[batch_idx].cpu().numpy()
#                 decoded_chars = []
#                 prev_idx = -1
#                 for idx in indices:
#                     if idx != 0 and idx != prev_idx:
#                         if idx in dataset.idx_to_char:
#                             decoded_chars.append(dataset.idx_to_char[idx])
#                     prev_idx = idx

#                 decoded_texts.append(''.join(decoded_chars))
            
#             return decoded_texts    

#         # 모델 평가 수행
#         def evaluate(self, dataloader, dataset, num_samples=10):
#             self.model.eval()
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for batch_idx, (images, targets, gt_text) in enumerate(dataloader):
#                     if batch_idx >= num_samples:
#                         break
#                     images = images.to(self.device)
#                     outputs = self.model(images)
#                     predicted_texts = self.decode_prediction(outputs, dataset)

#                     for pred, gt in zip(predicted_texts, gt_text):
#                         if pred == gt:
#                             corrct += 1
#                         total += 1

#                         print(f"GT: '{gt} | Pred: '{pred} | {'✔️' if pred == gt else '❌'}")

#             accuracy = corrct / total if total > 0 else 0
#             print(F"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
#             return accuracy

# def crnn_practice():
#     # 디바이스 설정
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 데이터셋 생성
#     train_dataset = SyntheticTextDataset(num_samples=1000, img_height=32, img_width=128)
#     test_dataset = SyntheticTextDataset(num_samples=100, img_height=32, img_width=128)
#     # 데이터 로더 생성
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

#     # 샘플 데이터 시각화
#     sample_image, sample_label, sample_text = train_dataset[0]
#     plt.figure(figsize=(10, 3))
#     plt.imshow(sample_image.squeeze(), cmap='gray')
#     plt.title(f"샘플 이미지: '{sample_text}'")
#     plt.axis('off')
#     plt.show()

#     # 모델 생성
#     model = CRNN(
#         img_height=32,
#         img_width=128,
#         num_chars=len(train_dataset.chars),
#         num_classes=train_dataset.num_classes
#     )

#     print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")

#     # 훈련
#     trainer = CRNNTrainer(model, device)
#     num_epochs = 5
#     for epoch in range(num_epochs):
#         avg_loss = trainer.train_epoch(train_loader)
#         current_lr = trainer.optimizer.param_groups[0]['lr']
#         print(f"평균 손실 : {avg_loss:.4f}, 학습률 : {current_lr:.6f}")

#         # 중간평가(2에포크마다)
#         if(epoch + 1) % 2 == 0:
#             trainer.evaluate(test_loader, test_dataset, num_samples=5)

#     final_accuracy = trainer.evaluate(test_loader, test_dataset, num_samples=20)

#     return model, train_dataset, test_dataset

# # 데이터로더용 collate 함수
# def collate_fn(batch):
#     images, labels, texts = zip(*batch)
#     images = torch.stack(images)
#     labels = list(labels)
#     return images, labels, texts

# model, train_dataset, test_dataset = crnn_practice()



# # ** TrOCR 모델 **
# # OCR 심화_p.12 TrOCR 예제
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests
# import torch
# import numpy as np

# class TrOCRSystem:
#     def __init__(self, model_name="microsoft/trocr-base-printed"):
#         self.processor = TrOCRProcessor.from_pretrained(model_name)
#         self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)

#         print(f"모델 로딩 완료 (Device: {self.device})")

#     def extract_text(self, image):
#         pixel_values = self.processor(image, return_tensors='pt').pixel_values
#         pixel_values = pixel_values.to(self.device)
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 pixel_values,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 max_length=256
#             )
#             generated_ids = outputs.sequences
#             token_scores = outputs.scores
#         generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         if token_scores:
#             token_probs=[]
#             for score in token_scores:
#                 probs = torch.softmax(score, dim=-1)
#                 max_prob = torch.max(probs).item()
#                 token_probs.append(max_prob)

#             confidence = sum(token_probs) / len(token_probs) if token_probs else 0
            
#             return generated_text, confidence
        
#         return generated_text

# # 샘플이미지 생성
# def create_test_image(text):
#     image_width = len(text) * 20 + 100
#     img_height = 60

#     img = Image.new('RGB', (image_width, img_height), 'white')
#     draw = ImageDraw.Draw(img)
#     try:
#         font = ImageFont.truetype("arial.ttf", 20)
#     except OSError:
#         font = ImageFont.load_default()

#     draw.text((20, 20), text, fill='black', font=font)

#     return img

# # TrOCR 모델
# ocr = TrOCRSystem()

# test_cases = [
#     "basic english",
#     "This is long english sentence",
#     "Machine Learning",
#     "2025년입니다",
#     "한글과 English and 123"
# ]

# for test_case in test_cases:
#     test_image = create_test_image(test_case)
#     text, confidence = ocr.extract_text(test_image)
#     print(f"인식된 텍스트: '{text}")
#     print(f"신뢰도: '{confidence:.1%}")




# # ** 한글지원 앙상블 모델 **
# # OCR 심화_p.62 앙상블 모델 예제
# import easyocr
# import cv2
# import numpy as np
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import torch
# from typing import List, Dict, Tuple, Union, Optional
# import re
# from PIL import Image

# class KoreanOCRemsemble:
#     def __init__(self):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.easy_ocr = easyocr.Reader(
#             ['ko', 'en'],
#             gpu=True
#         )
#         model_name = "microsoft/trocr-base-printed"
#         self.trorc_prcessor = TrOCRProcessor.from_pretrained(model_name)
#         self.trorc_model = VisionEncoderDecoderModel.from_pretrained(model_name)
#         self.trorc_model.to(self.device)

#     def extract_text_easy(self, image:np.ndarray):
#         results = self.easy_ocr.readtext(image)
#         extracted_texts = []
#         for result in results:
#             bbox, text, confidence = result
#             extracted_texts = ({
#                 'text': text,
#                 'confidence':confidence,
#                 'bbox':bbox,
#                 'engine':'EasyOCR'
#             })
#         return extracted_texts
    
#     def extract_text_trocr(self, image:np.ndarray):
#         if isinstance(image, np.ndarray):
#             image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         else:
#             image_pil = image

#         pixel_values = self.trorc_prcessor(image_pil, return_tensors='pt').pixel_values
#         pixel_values = pixel_values.to(self.device)

#         with torch.no_grad():
#             outputs = self.trorc_model.generate(
#                 pixel_values,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 max_length=256
#             )
#             generated_ids = outputs.sequences
#             token_scores = outputs.scores

#             if token_scores:
#                 token_probs = []
#                 for score in token_scores:
#                     probs = torch.softmax(score, dim=-1)
#                     max_prob = torch.max(probs).item()
#                     token_probs.append(max_prob)

#             confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

#         text = self.trorc_prcessor.batch_decode(generated_ids, skip_special_tokens=True)[0]

#         # easyOCR과 동일하게 결과 출력
#         return [{
#             'text': text,
#             'confidence': confidence,
#             'bbox': None,
#             'engine': 'TrOCR'
#         }] if text.strip() else []
    
#     def is_korean_text(self, text:str) -> float:
#         if not text:
#             return 0.0
#         korean_chars = sum(1 for char in text if 0xAC00 <= ord(char) <= 0xD7A3)
#         return korean_chars / len(text)
    
#     def is_english_text(self, text:str) -> float:
#         if not text:
#             return 0.0
        
#         total_chars = len([c for c in text if c.isalpha()])
#         if total_chars == 0:
#             return 0.0
        
#         englisth_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
#         ratio = englisth_chars / total_chars
#         return ratio

#     # 텍스트간 유사도 꼐산
#     def _calculate_text_similarity(self, text1, text2):
#         from difflib import SequenceMatcher
#         return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
#     # 텍스트 중복 여부 확인
#     def _is_duplicate_text(self, text:str, existing_results:List[Dict], threshold:0.8):
#         # 유사도를 기반으로 체크
#         for existing in existing_results:
#             similarity = self._calculate_text_similarity(text, existing['text'])
#             if similarity > threshold:
#                 return True
#         return False
    
#     # 텍스트 간 유사도 계산
#     def _should_replace_reuslt(self, new_result:Dict, existing_reuslt:Dict):
#         priority_scores = {'very_high':4, 'high': 3, 'medium': 2, 'low': 1}
#         new_priority_score = priority_scores.get(new_result['priority'], 0)
#         existing_priority_score = priority_scores.get(existing_reuslt['priority'], 0)
#         if new_priority_score > existing_priority_score:
#             return True
#         elif new_priority_score > existing_priority_score:
#             return False
        
#         # 우선순위가 같으면 신뢰도로 비교
#         english_ratio = self.is_english_text(new_result['text'])
#         korean_ratio = self.is_korean_text(new_result['text'])

#         new_confidence = new_result['confidence']
#         exsisting_confidence = existing_reuslt['confidence']

#         if english_ratio > 0.5 and korean_ratio < 0.5:
#             if new_result['engine'] == 'TrOCR':
#                 new_confidence += 0.15
#             if existing_reuslt['engine'] == 'TrOcr':
#                 exsisting_confidence += 0.15

#         if new_confidence > exsisting_confidence + 0.05:
#             return True
#         elif exsisting_confidence > new_confidence + 0.05:
#             return False
        
#         # 신뢰도가 비슷하면 엔진별 특성 고려
#         if english_ratio > 0/5 and korean_ratio < 0.5:
#             return new_result['engine'] == 'TrOCR'
#         elif korean_ratio > 0.5:
#             return new_result['engine'] == 'EasyOCR'
#         else:
#             return False

#     # OCR엔진의 결과를 피러링하고 병합
#     def filter_and_merge_results(self, easy_results:List[Dict], trocr_results:List[Dict]):
#         for result in easy_results:
#             english_ratio = self.is_english_text(result['text'])
#             korean_ratio = self.is_krean_text(result['text'])

#             all_results = []
#             if korean_ratio > 0.5: # 한글 위주
#                 result['priority'] = 'high'
#                 result['reason'] = 'Korean text - EasyOCR high'
#             elif english_ratio > 0.5: # 영어 위주
#                 result['priority'] = 'low'
#                 result['reason'] = 'English text - EasyOCR low'
#             elif korean_ratio > 0.2 and english_ratio > 0.2: # 혼용 텍스트
#                 result['priority'] = 'low'
#                 result['reason'] = 'Mixed text - EasyOCR high'
#             else:
#                 result['priority'] = 'low'
#                 result['reason'] = 'EasyOCR low'
#             all_results.append(result)

#             def get_priority_score(priority):
#                 if priority == 'very_high': return 0
#                 elif priority == 'high': return 1
#                 elif priority == 'medium': return 2
#                 else: return 3 # low

#             # 중복 제거
#             final_results = []
#             for result in all_results:
#                 best_existing = None
#                 for existing in final_results:
#                     if self._calculate_text_similarity(result['text'], existing['text']) > 0.8:
#                         best_existing = existing
#                         final_results.append(best_existing)
#                     else :
#                         final_results.remove(best_existing)
#                         final_results.append(result)

#             # sorting
#             all_results.sort(key=lambda x:(
#                 # 우선순위 적용 함수
#                 get_priority_score(x['priority']), -x['confidence']
#             ))

#             return final_results
            
#     # 앙상블 OCR 수행
#     def extract_text_emsenble(self, image_path:Union[str, np.ndarray]):
#         # 이미지 로드
#         image = cv2.imread(image_path)

#         # easy_ocr 결과 추출
#         easy_results = self.extract_text_easy(image_path)
#         print(f"easy results: {easy_results[0]['text']}")
#         print(f"   - EasyOCR: {len(easy_results)}개 텍스트 발견")

#         # trocr 결과 추출
#         trocr_results = self.extract_text_trocr(image_path)
#         print(f"trocr results: {trocr_results[0]['text']}")
#         print(f"   - TrOCR: {len(trocr_results)}개 텍스트 발견")

#         # 결과를 평가 -> 최종 결과 선택
#         merged_reuslts = self.filter_and_merge_results(
#             easy_results,
#             trocr_results
#         )
#         print(f"merged_results: {merged_reuslts}")

#         final_text = ' '.join([result['text'] for result in merged_reuslts if result['confidence'] > 0.5])
#         return {
#             'final_text': final_text,
#             'detailed_results': merged_reuslts,
#             'engine_stats': {
#                 'easy_count': len(easy_results),
#                 'trocr_count': len(trocr_results),
#                 'merged_count': len(merged_reuslts)
#             }
#         }


# def create_korean_test_image() -> List[np.ndarray]:
#     # 시스템 폰트 사용
#     from PIL import Image, ImageDraw, ImageFont
#     import numpy as np

#     try:
#         font_large = ImageFont.truetype("malgun.ttf", 40)
#         font_medium = ImageFont.truetype("malgun.ttf", 30)
#     except:
#         try:
#             font_large = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttf", 40)
#             font_medium = ImageFont.truetype("/System/Library/Fonts/AppleSDGothicNeo.ttf", 30)
#         except:
#             font_large = ImageFont.load_default()
#             font_medium = ImageFont.load_default()

#     texts = [
#         ("한글 OCR 성능 테스트", 50, 50, font_large),
#         ("Korean OCR Performance Test", 50, 100, font_medium),
#         ("Mixed Text 성능 테스트", 50, 150, font_medium),
#         ("숫자 1234 성능 테스트", 50, 200, font_medium)
#     ]
#     i = 0
#     img_list = []
#     for text, x, y, font in texts:
#         img = Image.new('RGB', (800, 400), 'white')
#         draw = ImageDraw.Draw(img)
#         draw.text((x, y), text, fill='black', font=font)
#         i += 1
#         img_list.append(img)
#         img.save(f'test_image{i}.png')

#     img_list = [cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB) for img in img_list]
#     return img_list
        

# # 테스트 데이터 생성
# test_images = create_korean_test_image()

# # OCR 앙상블 모델
# ocr_emsenble = KoreanOCRemsemble()

# for test_image in test_images:
#     result = ocr_emsenble.extract_text_emsenble(test_image)


# # ** 이미지 세그멘테이션 기초
# # 이미지_세그멘테이션 및 객체 검출 p.10 임계값 기반
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# def threshold_segmentation_demo():
#     image = cv2.imread('image.png', 0)

#     ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

#     _, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

#     plt.figure(figsize=(15, 5))

#     plt.subplot(1, 4, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')

#     plt.subplot(1, 4, 2)
#     plt.imshow(binary, cmap='gray')
#     plt.title('Binary Threshold')
#     plt.axis('off')

#     plt.subplot(1, 4, 3)
#     plt.imshow(otsu, cmap='gray')
#     plt.title("Otsu's Method")
#     plt.axis('off')

#     plt.subplot(1, 4, 4)
#     plt.imshow(adaptive, cmap='gray')
#     plt.title('Adaptive Threshold')
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

#     return binary, otsu, adaptive

# threshold_segmentation_demo()

# ** 이미지 세그멘테이션 기초
# 이미지_세그멘테이션 및 객체 검출 p.17 군집화
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.metrics.pairwise import euclidean_distances
import cv2
import numpy as np
import matplotlib.pyplot as plt

def clustering_segmentation_demo():
    image = cv2.imread('image.png')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    k = 5

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pixel_values)

    centers = np.uint8(kmeans.cluster_centers_)

    segmented_image = centers[labels.flatten()]

    segmented_image = segmented_image.reshape(image_rgb.shape)

    # mean shift 클러스터링
    # 1. 이미지 크기 줄여서 계산 속도 향상(시간여유가 있음 안해도 됨)
    small_image = cv2.resize(image_rgb, (image_rgb.shape[1]//2, image_rgb.shape[0]//2))
    small_pixel_values = small_image.reshape((-1, 3))
    small_pixel_values = np.float32(small_pixel_values)

    # 2. 픽셀 샘플링
    sample_size = min(10000, len(small_pixel_values))
    sample_indices = np.random.choice(len(small_pixel_values), sample_size, replace=False)
    sampled_pixels = small_pixel_values[sample_indices]

    # 3. mean shift 클러스터링 수행
    mean_shift = MeanShift(bandwidth=30)
    labels_ms_sample = mean_shift.fit_predict(sampled_pixels)

    # 4. 전체 이미지에 클러스터 결과 적용
    centers_ms = np.uint8(mean_shift.cluster_centers_)
    # 모든 픽셀에 대해 가장 가까운 클러스터 할당
    distances = euclidean_distances(small_pixel_values, centers_ms)
    labels_ms_full = np.argmin(distances, axis=1)
    segmented_image_ms = centers_ms[labels_ms_full]
    segmented_image_ms = segmented_image_ms.reshape(small_image.shape)

    # 원래 크키로 복원
    segmented_image_ms = cv2.resize(segmented_image_ms, (image_rgb.shape[1], image_rgb.shape[0]))

    plt.figure(figsize=(15,5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title(f'K-Means (k={k})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image_ms)
    plt.title('Mean Shift (Optimized)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return segmented_image, segmented_image_ms

clustering_segmentation_demo()
    



