# HSV로 얼굴 색상 기반 마스크

import cv2
import numpy as np

# 1. 이미지 로드
image = cv2.imread('selfie.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. 피부색 HSV 범위 정의
lower_skin = np.array([0, 10, 80])
upper_skin = np.array([30, 200, 255])

# 3. 피부색 영역 마스크 생성
mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)

# 4. 피부색 부분만 흰색으로 지우기
result = image.copy()
result[mask_skin > 0] = (255, 255, 255)

rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba[mask_skin > 0, 3] = 0 

# 5. 결과 표시
cv2.imshow("Original", image)
cv2.imshow("Skin Mask", mask_skin)
cv2.imshow("Face Color Removed (HSV)", result)
cv2.imwrite("Face_Color_Removed(HSV)_Transparent.png", rgba)
cv2.waitKey(0)
cv2.destroyAllWindows()
