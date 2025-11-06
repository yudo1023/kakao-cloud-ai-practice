import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
plt.rc('font', family='Apple SD Gothic Neo')

# # ** 인스턴스 세그멘테이션 **
# # p.59 Mask R-CNN 예제
# class MaskRcnnPredictor:
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.model = maskrcnn_resnet50_fpn(pretrained=True)
#         self.model.to(self.device)
#         self.model.eval()
#         self.class_name = [
#             '__background__','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#             'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#             'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
#             'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
#             'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
#             'sports ball', 'kite','baseball bat', 'baseball glove', 'skateboard',
#             'suftboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
#             'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
#             'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#             'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
#             'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#             'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
#             'clock', 'vase', 'scissors', 'teddy bear', 'hair drider', 'toothbrush'       
#             ]

#     # 이미지에서 인스턴스 세그멘테이션 수행
#     def predict(self, img_path, confidence_threshold=0.5):
#         # 이미지 로드 및 전처리
#         image = Image.open(img_path).convert('RGB')
#         # 이미지를 텐서로 변환 [0, 255] -> [0, 1]
#         image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)
#         # 모델 추론
#         with torch.no_grad():
#             predictions = self.model(image_tensor)
#         # 결과 처리
#         pred = predictions[0]
#         # 신뢰도가 임계값 이상만 선택
#         keep_idx = pred['scores'] > confidence_threshold
#         # 필터링된 결과 추출
#         boxes = pred['boxes'][keep_idx].cpu().numpy()
#         labels = pred['labels'][keep_idx].cpu().numpy()
#         scores = pred['scores'][keep_idx].cpu().numpy()
#         masks = pred['masks'][keep_idx].cpu().numpy()

#         return image, boxes, labels, scores, masks
    
#     # 예측 결과 시각화
#     def visualize_results(self, image, boxes, labels, scores, masks):
#         fig, axes = plt.subplots(1, 2, figsize=(15,7))
#         axes[0].imshow(image)
#         axes[0].set_title('Object Detection')
#         axes[0].axis('off')

#         colors = plt.cm.Set1(np.linspace(0, 1, len(masks)))

#         for i, mask in enumerate(masks):
#             mask_colored = np.zeros((*mask.shape[1:], 4))
#             mask_colored[:, :, :3] = colors[i][:3]
#             mask_colored[:, :, 3] = mask[0] * 0.7
#             axes[1].imshow(mask_colored)
        
#         plt.tight_layout()
#         plt.show()  

# # MaskRCNN구축(pretrained)
# predictor = MaskRcnnPredictor()
# image_path = 'test.jpg'
# image, boxes, labels, scores, masks = predictor.predict(image_path)
# predictor.visualize_results(image, boxes, labels, scores, masks)


from ultralytics import YOLO
import cv2

## p.82 YOLO 예제
# class YOLOSegmentation:
#     def __init__(self, model_name='yolov8n-seg.pt'):
#         self.model = YOLO(model_name)
    
#     def predict_and_visualize(self, image_path, confidence=0.5):
#         results = self.model(image_path, conf=confidence)

#         for r in results:
#             image = cv2.imread(image_path)
#             image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#             fig, axes = plt.subplots(1, 3, figsize=(18, 6))

#             axes[0].imshow(image_rgb)
#             axes[0].set_title('Original Image')
#             axes[0].axis('off')

#             img_with_boxes = r.plot()
#             axes[1].imshow(img_with_boxes)
#             axes[1].set_title('Detection Results')
#             axes[1].axis('off')

#             if r.masks is not None:
#                 masks = r.masks.data.cpu().numpy()
#                 combined_mask = np.zeros_like(image_rgb)

#                 for i, mask in enumerate(masks):
#                     mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))
#                     color = np.random.randint(0, 255, 3)
#                     colored_mask = np.zeros_like(image_rgb)
#                     colored_mask[mask_resized > 0.5] = color
#                     combined_mask = cv2.addWeighted(combined_mask, 1, colored_mask, 0.7, 0)
                
#                 result_img = cv2.addWeighted(image_rgb, 0.6, combined_mask, 0.4, 0)
#                 axes[2].imshow(result_img)
#                 axes[2].set_title('Segmentation Masks')
#             else:
#                 axes[2].text(0.5, 0.5, 'No masks detected',
#                              transform=axes[2].transAxes, ha='center', va='center')
#                 axes[2].set_title('No segmentation result')
                
#             axes[2].axis('off')
#             plt.tight_layout()
#             plt.show()

# # yolo 인스턴스 생성
# yolo_seg = YOLOSegmentation()

# image_path = 'test.jpg'
# yolo_seg.predict_and_visualize(image_path)


# ** 객체 검출 **
import matplotlib.patches as patches

# # p.88 바운딩 박스
# def visualize_bouding_box_formats():
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     img = np.ones((300, 400, 3)) * 0.8
#     x1, y1, x2, y2 = 50, 50, 200, 150
#     axes[0].imshow(img)
#     rect1 = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
#                               linewidth=2, edgecolor='red', facecolor='none')
#     axes[0].add_patch(rect1)
#     axes[0].text(x1, y1-5, f'({x1}, {y1})', color='red', fontsize=10)
#     axes[0].text(x2, y2+15, f'({x2}, {y2})', color='red', fontsize=10)
#     axes[0].set_title('(x1, y1, x2, y2) 형식')
#     axes[0].set_xlim(0, 400)
#     axes[0].set_ylim(300, 0)

#     x, y, w, h = 50, 50, 150, 100
#     axes[1].imshow(img)
#     rect2 = patches.Rectangle((x, y), w, h,
#                               linewidth=2, edgecolor='blue', facecolor='none')
#     axes[1].add_patch(rect2)
#     axes[1].text(x, y-5, f'({x}, {y})', color='blue', fontsize=10)
#     axes[1].text(x+w//2, y+h//2, f'w={w}, h={h}', color='blue', fontsize=10)
#     axes[1].set_title('(x, y, width, height) 형식')
#     axes[1].set_xlim(0, 400)
#     axes[1].set_ylim(300, 0)

#     plt.tight_layout()
#     plt.show()

# visualize_bouding_box_formats()

# # p.92 IoU
# def calculate_iou(box1, box2):
#     # 두 영역이 겹칠때
#     # 왼쪽 경계는 두 박스의 왼쪽 좌표 중 더 큰 값
#     # 위쪽 경계는 두 박스의 위쪽 좌표 중 더 큰 값
#     # 오른쪽 경계는 두 박스의 오른쪽 좌표 중 더 작은 값
#     # 아래쪽 경계는 두 박스의 아래쪽 좌표 중 더 작은 값
#     x1_inter = max(box1[0], box2[0])
#     y1_inter = max(box1[1], box2[1])
#     x2_inter = min(box1[2], box2[2])
#     y2_inter = min(box1[3], box2[3])

#     # 교집합 영역
#     if x2_inter > x1_inter and y2_inter > y1_inter:
#         intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
#     else:
#         intersection = 0
    
#     # 각 박스의 영역
#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     # 합집합 영역
#     union = area1 + area2 - intersection

#     # IoU 계산
#     iou = intersection / union if union > 0 else 0

#     return iou

# # IoU 시각화
# def visualize_iou():
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     cases = [
#         {'box1' : [50, 50, 150, 150], 'box2' : [100, 100, 200, 200], 'title':'IoU = 0.14'},
#         {'box1' : [50, 50, 150, 150], 'box2' : [75, 75, 175, 175], 'title':'IoU = 0.36'},
#         {'box1' : [50, 50, 150, 150], 'box2' : [60, 60, 140, 140], 'title':'IoU = 0.64'},
#     ]

#     for i, case in enumerate(cases):
#         box1, box2 = case['box1'], case['box2']

#         # 빈 이미지
#         img = np.ones((250, 250, 3)) * 0.9
#         axes[i].imshow(img)

#         # 첫 번째 박스
#         rect1 = patches.Rectangle((box1[0], box1[1]), box1[2] - box1[0], box1[3] - box1[1],
#                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
#         axes[i].add_patch(rect1)
#         # 두 번째 박스
#         rect2 = patches.Rectangle((box2[0], box2[1]), box2[2] - box2[0], box2[3] - box2[1],
#                                   linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
#         axes[i].add_patch(rect2)

#         iou = calculate_iou(box1, box2)
#         axes[i].set_title(f'{case["title"]} (실제 : {iou:.2f})')
#         axes[i].set_xlim(0, 250)
#         axes[i].set_ylim(250, 0)
#         axes[i].axis('off')

#     plt.tight_layout()
#     plt.show()

# visualize_iou()

# # p. 98 NMS
# def calculate_iou(box1, box2):
#     x1_inter = max(box1[0], box2[0])
#     y1_inter = max(box1[1], box2[1])
#     x2_inter = min(box1[2], box2[2])
#     y2_inter = min(box1[3], box2[3])

#     if x2_inter <= x1_inter or y2_inter <= y1_inter:
#         return 0
    
#     intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

#     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

#     union = area1 + area2 - intersection

#     return intersection/union

# def nms(boxes, scores, iou_threshold=0.5):
#     indices = np.argsort(scores)[::-1]
#     keep = []
#     while len(indices) > 0:
#         current = indices[0]
#         keep.append(current)
#         if len(indices) == 1:
#             break
#         current_box = boxes[current]
#         other_boxes = boxes[indices[1:]] # 현재 박스를 제외한 나머지 박스들

#         ious = []
#         for other_box in other_boxes:
#             iou = calculate_iou(current_box, other_box)
#             ious.append(iou)
#         ious = np.array(ious)

#         # 임계값보다 큰 박스들은 현재 박스와 중복을 간주해서 제거, 작은 박스들만 남김
#         # 높은 IoU: 같은 객체를 여러번 검출(불필요한 중복), 낮은 IoU: 서로 다른 객체를 검출
#         indices = indices[1:][ious < iou_threshold]
    
#     return keep

# def visualize_nms():
#     boxes = np.array([
#         [50, 50, 150, 150],
#         [60, 60, 160, 160],
#         [200, 100, 300, 200],
#         [210, 110, 310, 210],
#         [70, 70, 170, 170]
#     ])
#     scores = np.array([0.9, 0.8, 0.85, 0.7, 0.75])
#     keep_indicies = nms(boxes, scores, iou_threshold=0.3)

#     fig, axes = plt.subplots(1, 2, figsize=(15, 6))
#     img = np.ones((350, 400, 3)) * 0.9
#     axes[0].imshow(img)
#     axes[0].set_title('NMS 적용 전')
#     colors = ['red', 'blue', 'green', 'orange', 'purple']

#     for i, (box, score) in enumerate(zip(boxes, scores)):
#         x1, y1, x2, y2 = box
#         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
#                                  linewidth=2, edgecolor=colors[i],
#                                  facecolor=colors[i], alpha=0.3)
#         axes[0].add_patch(rect)
#         axes[0].text(x1, y1-5, f'Box {i+1}: {score:.2f}',
#                      color=colors[i], fontsize=10, weight='bold')
#     axes[0].set_xlim(0, 400)
#     axes[0].set_ylim(350, 0)
#     axes[0].axis('off')
#     axes[1].imshow(img)
#     axes[1].set_title('NMS 적용 후')

#     for i in keep_indicies:
#         box, score = boxes[i], scores[i]
#         x1, y1, x2, y2 = box
#         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
#                                  linewidth=3, edgecolor=colors[i],
#                                  facecolor=colors[i], alpha=0.3)
#         axes[1].add_patch(rect)
#         axes[1].text(x1, y1-5, f'Box {i+1}: {score:.2f}',
#                      color=colors[i], fontsize=10, weight='bold')
#     axes[1].set_xlim(0, 400)
#     axes[1].set_ylim(350, 0)
#     axes[1].axis('off')

#     plt.tight_layout()
#     plt.show()

# visualize_nms()

from sklearn.metrics import precision_recall_curve, average_precision_score
## p.109 mAP
# def calculate_map_example():
#     np.random.seed(42) # 실행할 때 마다 동일한 결과를 얻기 위해 랜덤 시드 고정

#     classes = ['person', 'car', 'bicyle']

#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))

#     total_ap = 0

#     for i, class_name in enumerate(classes):
#         n_samples = 100
#         y_true = np.random.randint(0, 2, n_samples)
#         y_scores = np.random.random(n_samples)
#         y_scores[y_true == 1] += 0.3
#         y_scores = np.clip(y_scores, 0, 1)
#         precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
#         ap = average_precision_score(y_true, y_scores)
        
#         # PR 곡선 시각화
#         axes[0, i].plot(recall, precision, linewidth=2, label=f'AP={ap:.3f}')
#         axes[0, i].fill_between(recall, precision, alpha=0.3) # 곡선 아래 부분 채우기
#         axes[0, i].set_xlabel('Recall')
#         axes[0, i].set_ylabel('Precision')
#         axes[0, i].set_title(f'{class_name} - PR Curve')
#         axes[0, i].legend()
#         axes[0, i].grid(True, alpha=0.3)
#         axes[0, i].set_xlim([0, 1])
#         axes[0, i].set_ylim([0, 1])
        
#         # 신뢰도 임계값에 따른 정밀도/재현율 변화
#         axes[1, i].plot(thresholds, precision[:-1], 'b-', label='Precision')
#         axes[1, i].plot(thresholds, recall[:-1], 'r-', label='Recall')
#         axes[1, i].set_xlabel('Confidence Threshold')
#         axes[1, i].set_ylabel('Score')
#         axes[1, i].set_title(f'{class_name} - Precision vs Recall')
#         axes[1, i].legend()
#         axes[1, i].grid(True, alpha=0.3)
#         axes[1, i].set_xlim([0, 1])
#         axes[1, i].set_ylim([0, 1])

#         total_ap += ap

#     map_score = total_ap / len(classes)
#     plt.suptitle(f'mAP = {map_score:.3f}', fontsize=16, fontweight = 'bold')
#     plt.tight_layout()
#     plt.show()

# calculate_map_example()

# ** chatbot **
from sklearn.calibration import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification

# # 챗봇 기본 개념 p.22 BERT 기반 의도 분류 예제
# class IntentClassifier:
#     def __init__(self, model_name = 'klue/bert-base'):
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = None
#         self.label_encoder = LabelEncoder()

#     def prepare_date(self, texts, labels):
#         encoded_labels = self.label_encoder.fit_transform(labels)
#         encodings = self.tokenizer(
#             texts,
#             truncation=True,
#             padding=True,
#             max_length=128,
#             return_tensors='pt'
#         )
#         return encodings, encoded_labels

#     def train(self, train_texts, train_labels):
#         num_labels = len(set(train_labels))
#         self.model = BertForSequenceClassification.from_pretrained(
#             'klue/bert-base',
#             num_labels=num_labels
#         )
#         train_encodings, train_labels_encoded = self.prepare_date(train_texts, train_labels)
        
#         class IntentDataset(torch.utils.data.Dataset):
#             def __init__(self, encodings, labels):
#                 self.encodings = encodings
#                 self.labels = labels

#             def __getitem__(self, idx):
#                 item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#                 item['labels'] = torch.tensor(self.labels[idx])
#                 return item

#             def __len__(self):
#                 return len(self.labels)
    
#         train_dataset = IntentDataset(train_encodings, train_labels_encoded)
#         batch_size = 16
#         train_dataloader = torch.utils.data.DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=True
#         )

#         learning_rate = 2e-5
#         weight_decay = 0.01
#         optimizer = torch.optim.AdamW(
#             self.model.parameters(),
#             lr=learning_rate,
#             weight_decay=weight_decay

#         )# AdamW = Adam + weight decay, bert 훈련에 최적화된 옵티마이저
#         self.model.train()
#         epochs = 3
#         for epoch in range(epochs):
#             total_loss = 0
#             for batch in train_dataloader:
#                 optimizer.zero_grad()
#                 outputs = self.model(**batch)
#                 loss = outputs.loss
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
#                 optimizer.step()
#                 total_loss += loss.item()

#             avg_loss = total_loss / len(train_dataset)
#             print(f"AVG Loss : {avg_loss:.4f}")

#     def predict(self, text):
#         inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#             predictions = torch.nn.functional.softmax(outputs.logits, dim=1)

#         predicted_class = torch.argmax(predictions, dim=1).item()
#         confidence = torch.max(predictions).item()
#         intent = self.label_encoder.inverse_transform([predicted_class])

#         return intent, confidence


# train_texts = [
#     "안녕하세요", "반갑습니다", "hello",
#     "날씨가 어때요?", "비가 와요?", "맑나요?",
#     "주문하고 싶어요", "메뉴 보여줘", "배달 가능해?"
# ]

# train_labels = [
#     "greeting", "greeting", "greeting",
#     "weather", "weather", "weather",
#     "order", "order", "order"
# ]

# test_text = "오늘 날씨 어때요?"

# classifier = IntentClassifier()
# classifier.train(train_texts, train_labels)

# intent, confidence = classifier.predict(test_text)
# print(f"의도: {intent}, 신뢰도: {confidence:.2f}")


# # ** openai api 활용 **
# p.54 OpenAI API 활용 챗봇 예제
from openai import OpenAI
import os
import dotenv
dotenv.load_dotenv()

class OpenAIChatbot:
    def __init__(self, api_key, model: str = 'gpt-3.5-trubo'):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.system_message = {
           "role": "system",
           "content": "당신은 까페 주문을 받는 친절한 직원입니다. 메뉴 추천과 주문 처리를 도와주세요."
        }
        
    def set_system_prompt(self, prompt: str):
        self.system_message["content"] = prompt

    def add_message(self, role, content):
        self.conversation_history.append({
            'role': role,
            'content': content
        })

    def get_response(self, user_message):
        # add_message
        self.add_message('user', user_message)

        messages = [self.system_message] + self.conversation_history
        response = self.client.chat.completions.create(
            meodel = self.model,
            messages = messages,
            max_tokens = 1000,
            temperature=0.7,
            presence_penalty=0.6,
            frequency_penalty=0.0
        )
        assistant_message = response.choices[0].message.content
        self.add_message('assitant', assistant_message)

        return assistant_message
    
api_key = os.environ.get("OPENAI_API_KEY")
chatbot = OpenAIChatbot(api_key)
chatbot.set_system_prompt(
    "당신은 카페 주문을 받는 친절한 직원입니다. "
    "메뉴 추천과 주문 처리를 도와주세요."
)

user_input = "안녕하세요, 추천 메뉴가 있나요?"
response = chatbot.get_response(user_input)
print(f"챗봇: {response}")

user_input = "달지 않은 음료를 원해요"
response = chatbot.get_response(user_input)
print(f"챗봇: {response}")