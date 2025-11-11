from ultralytics import SAM
import cv2
import matplotlib.pyplot as plt

# SAM 모델 로드
model = SAM("sam_b.pt") 

# 이미지 경로
image_path = "01.jpg"

# 전체 이미지 분할 (자동)
results = model(image_path)

# 결과 마스크 추출
masks = results[0].masks.data.cpu().numpy()  # 여러 마스크가 있을 수 있음

# 첫 번째 마스크만 선택
mask = masks[6]

# 원본 이미지 로드
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 마스크를 이용해 세그멘테이션 결과 생성
segmented = image_rgb.copy()
segmented[mask == 0] = 0 

# 마스크 시각화
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Segmentation Mask")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Segmented Result")
plt.imshow(segmented)
plt.axis('off')

plt.show()

# 마스크 및 분할 결과 저장
cv2.imwrite("01/mask_result.png", mask * 255)  # 마스크 저장
cv2.imwrite("01/segmented_result7.png", cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))