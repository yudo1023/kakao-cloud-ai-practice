from ultralytics.models.sam import Predictor as SAMPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지 로드 및 RGB 변환
image = cv2.imread("01.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Ultralytics SAM Predictor 초기화
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam_b.pt")
predictor = SAMPredictor(overrides=overrides)

# 이미지 설정
predictor.set_image(image_rgb)

# 바운딩 박스 좌표 지정 [x_min, y_min, x_max, y_max]
#box_prompt = [209, 239, 383, 546]
#box_prompt = [5, 200, 152, 374]
box_prompt = [240, 340, 280, 380]
# results는 Ultralytics 모델의 출력 결과 객체 리스트
results = predictor(bboxes=[box_prompt])

# 중심점 좌표로 지정 (x, y)
point_prompt = [[260, 360]] 
results2 = predictor(points=point_prompt, labels=[1])

# 첫 번째 결과를 가져옴
result = results[0]
result2 = results2[0]

# 결과 중 마스크 데이터는 .masks.data 에 있음 (3D 배열)
masks = result.masks.data  # shape: (num_masks, height, width)
masks2 = result2.masks.data
# 첫 번째 마스크 선택 (2D 배열)
mask = masks[0]
mask2 = masks2[0]

# 이미지와 마스크 겹쳐서 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.imshow(mask, alpha=0.6)  # 투명도 조절
plt.title("Ultralytics SAM result(box)")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_rgb)
plt.imshow(mask2, alpha=0.6)  # 투명도 조절
plt.title("Ultralytics SAM result(point)")
plt.axis('off')

plt.show()

# 마스크 및 분할 결과 저장
#cv2.imwrite("02/mask_result6.png", mask.cpu().numpy() * 255)