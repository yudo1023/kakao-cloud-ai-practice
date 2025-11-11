from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import requests
import os

if not os.path.exists("sam_vit_b_01ec64.pth"):
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    response = requests.get(url)
    with open("sam_vit_b_01ec64.pth", "wb") as f:
        f.write(response.content)

# 모델 로드 및 초기화
sam_checkpoint = "sam_vit_b_01ec64.pth"  # 다운받은 SAM 체크포인트 경로 지정
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# 이미지 로드
image = cv2.imread("01.jpg")  # 사용자 이미지 파일명 사용
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# 박스 프롬프트 지정
x_min, y_min, x_max, y_max =(209, 239, 383, 546) # 강쥐
#x_min, y_min, x_max, y_max =(5, 200, 152, 374) # 딸긔
# 개 영역에 직접 박스 좌표 지정 (예시: 이미지 내 개 부분 직접 측정 필요)
input_box = np.array([ [x_min, y_min], [x_max, y_max] ])  # 예: [[250,200],[450,520]] (좌표는 실제 이미지에 맞게 수정)
#masks, scores, logits = predictor.predict(box=input_box, multimask_output=False)

# 점 프롬프로 지정
#input_points = np.array([[300, 400]]) # 강쥐
input_points = np.array([[135, 290]]) # 딸긔
masks, scores, logits = predictor.predict(point_coords=input_points, multimask_output=False, point_labels=[1])


# 마스크 시각화 출력
plt.figure(figsize=(10,10))
plt.imshow(image_rgb)
mask = masks[0]
plt.imshow(mask, alpha=0.6)  # 개만 분할된 마스크 영역
plt.axis('off')
plt.title("SAM result with Box Prompt")
plt.show()

# 마스크 및 분할 결과 저장
cv2.imwrite("03/mask_result4.png", (mask * 255).astype(np.uint8))