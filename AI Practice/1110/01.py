# HSV를 활용한 파란색 객체 분할
import cv2
import numpy as np

# 1. 이미지 로드
image = cv2.imread('onion2.png')

# 2. BGR 이미지를 HSV로 변환
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 파란색 범위 정의 (HSV)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# 4. HSV 이미지에서 파란색 범위에 해당하는 픽셀만 추출하여 마스크 생성
mask_bg = cv2.inRange(hsv_image, lower_blue, upper_blue)

# 5. 원본 이미지에 마스크 적용 (파란색 영역만 남김)
result = cv2.bitwise_and(image, image, mask=mask_bg)

mask_fg = cv2.bitwise_not(mask_bg)
result2 = cv2.bitwise_and(image, image, mask=mask_fg)

rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
rgba_image[:, :, 3] = mask_fg

# 6. 결과 이미지 표시
cv2.imshow('Original Image', image)
cv2.imshow('Blue Mask', mask_bg)
cv2.imshow('bg', mask_fg)
cv2.imshow('Segmented Blue Object', result)
cv2.imshow('Segmented None Blue Object',result2)
#cv2.imshow('Onion Transparent',rgba_image)
cv2.imwrite("onion_transparent.png", rgba_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

