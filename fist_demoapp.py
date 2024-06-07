import os
import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image
from text_editing_SDXL import BlendedLatentDiffusionSDXL
from inference import predictor_inference
import requests
import base64
import json

# 포인트 색상 및 마커
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

# Ollama API 설정
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava"

# 이미지를 Base64로 인코딩하는 함수
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# 이미지를 분석하고 프롬프트 단어와 결합하여 6단어 문장을 생성하는 함수
def generate_combined_prompt(image, user_prompt):
    # 이미지를 Base64로 인코딩
    image_base64 = encode_image_to_base64(image)
    
    # Llava API 호출
    llm_payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"'{user_prompt}' and the atmosphere in the picture you identified, please express it in 6 words sentence",
        "images": [image_base64]
    }
    response = requests.post(OLLAMA_API_URL, json=llm_payload, stream=True)
    
    # API 응답 디버깅을 위한 로그 메시지 추가
    print("API 응답 상태 코드:", response.status_code)
    
    combined_prompt = ""
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                # 각 라인을 JSON으로 파싱
                line_json = json.loads(line.decode('utf-8'))
                combined_prompt += line_json.get("response", "")
    else:
        combined_prompt = "프롬프트 생성 실패"
    
    print("최종 프롬프트:", combined_prompt)
    return combined_prompt

with gr.Blocks() as demo:
    # 이미지 세그멘테이션
    with gr.Tab(label='Image'):
        with gr.Row():
            with gr.Column():
                # 입력 이미지
                original_image = gr.State(value=None)   # 포인트가 없는 원본 이미지를 저장, 기본값은 None
                input_image = gr.Image(type="numpy")
                # 포인트 프롬프트
                with gr.Column():
                    selected_points = gr.State([])      # 포인트 저장
                    with gr.Row():
                        gr.Markdown('포인트 삭제를 위한 버튼입니다.')
                        undo_button = gr.Button('포인트 삭제')
                # 실행 버튼
                button = gr.Button("변환")
                
             # 블렌드 결과 표시
            with gr.Tab(label='결과'):
                blended_image = gr.Image(type='numpy')
        
    # 사용자가 이미지를 업로드하면, 원본 이미지는 `original_image`에 저장됩니다.
    def store_img(img):
        return img, []  # 새 이미지를 업로드할 때, `selected_points`는 비어 있어야 합니다.
    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )

    # 사용자가 이미지를 클릭하여 포인트를 선택하고, 이미지를 업데이트합니다.
    def get_point(img, sel_pix, evt: gr.SelectData):
        sel_pix.append((evt.index, 1))   # 전경 포인트를 추가합니다
        img_copy = img.copy()
        for point, label in sel_pix:
            cv2.circle(img_copy, tuple(point), 5, (0, 255, 0) if label == 1 else (0, 0, 255), -1)
        return np.array(img_copy)
    input_image.select(
        get_point,
        [input_image, selected_points],
        [input_image],
    )
       
    def blend_images(original_image, mask, prompt):
        init_image_path = "inputs/init_image.png"
        mask_path = "inputs/mask.png"
        Image.fromarray(original_image).save(init_image_path)
        Image.fromarray(mask).save(mask_path)
        
        args = lambda: None
        args.init_image = init_image_path
        args.mask = mask_path
        args.prompt = prompt
        args.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        args.batch_size = 1
        args.blending_start_percentage = 0.25
        args.device = "cuda"
        args.output_path = "outputs/blended_image.png"
        
        bld = BlendedLatentDiffusionSDXL.from_pretrained(args.model_path, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
        bld.args = args
        bld.to(args.device)
        
        results = bld.edit_image(
            blending_percentage=args.blending_start_percentage,
            prompt=[args.prompt] * args.batch_size,
        )
        
        results_flat = np.concatenate(results, axis=1)
        result_image = Image.fromarray(results_flat)
        result_image.save(args.output_path)
        
        return np.array(result_image)
    
    # 자동 실행 버튼 클릭 시
    def process_auto(input_image, selected_points, original_image, user_prompt):
        bgr_image = original_image[:, :, [2, 1, 0]]  # RGB to BGR
        mask_image = predictor_inference(bgr_image, selected_points)
        combined_prompt = generate_combined_prompt(original_image, user_prompt)  # original_image를 사용하여 프롬프트 생성
        blended_image = blend_images(original_image, mask_image, combined_prompt)
        print(combined_prompt)
        return blended_image

    # Ollama에서 텍스트 프롬프트를 입력받기 위한 Textbox 추가
    with gr.Row():
        prompt_input = gr.Textbox(label="바꾸고 싶은 배경의 키워드를 적어주세요")

    button.click(
        process_auto, 
        inputs=[input_image, selected_points, original_image, prompt_input],
        outputs=[blended_image]
    )

    # 선택된 포인트를 되돌립니다.
    def undo_points(orig_img, sel_pix):
        temp = orig_img.copy() if isinstance(orig_img, np.ndarray) else np.array(orig_img)
        # 포인트를 그립니다.
        if len(sel_pix) != 0:
            sel_pix.pop()
            for point, label in sel_pix:
                cv2.drawMarker(temp, tuple(point), colors[label], markerType=markers[label], markerSize=20, thickness=5)
        return temp
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image]
    )

demo.queue().launch(debug=True, share=True)
