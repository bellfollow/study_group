import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 로드
image_path = '다운로드.png'  # 이미지 경로를 설정하세요
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 이미지를 읽음

# 가우시안 블러를 적용하여 이미지 노이즈 감소
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  # 가우시안 블러 적용, 커널 크기는 (5, 5)

# Canny 엣지 검출 (임계값 조정)
edges = cv2.Canny(blurred_img, 50, 150)  # 노이즈가 줄어든 이미지에 Canny 적용, 임계값을 조정

# 원본 이미지, 블러 처리 이미지, 엣지 검출 결과를 시각화
plt.figure(figsize=(15,5))

plt.subplot(131), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132), plt.imshow(blurred_img, cmap='gray')
plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])

plt.subplot(133), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
