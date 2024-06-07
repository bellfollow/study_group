# inference.py 파일
import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"

def predictor_inference(input_image, selected_points):
    # 입력 이미지 확인
    if input_image is None:
        raise ValueError("Input image is None")
   

    # 포인트 처리
    if selected_points:
        points = np.array([p for p, _ in selected_points])
        labels = np.array([int(l) for _, l in selected_points])
        print(points, "dd", labels, "dd")
    else:
        points, labels = None, None
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(input_image)  # 이미지 처리
    # 예측 수행
    
    masks, _, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=None,
        multimask_output=False,
    )

    # 첫 번째 마스크 추출
    mask = masks[0]

    # 이진화된 마스크 저장
    result_image = np.where(mask, 1, 0).astype(np.uint8)
    image_name = f"mask.png"
    cv2.imwrite(image_name, result_image * 255)
    print("done")
    return result_image * 255 
