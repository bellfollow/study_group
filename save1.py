# app.py 파일
import os
import cv2
import numpy as np
import gradio as gr
import torch
from PIL import Image
from text_editing_SDXL import BlendedLatentDiffusionSDXL, parse_args
from inference import predictor_inference

# points color and marker
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

with gr.Blocks() as demo:
    # Segment image
    with gr.Tab(label='Image'):
        with gr.Row():
            with gr.Column():
                # input image
                original_image = gr.State(value=None)   # store original image without points, default None
                input_image = gr.Image(type="numpy")
                # point prompt
                with gr.Column():
                    selected_points = gr.State([])      # store points
                    with gr.Row():
                        gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                        undo_button = gr.Button('Undo point')
                # run button
                button = gr.Button("Auto!")
                
            # show the image with mask
            with gr.Tab(label='Image+Mask'):
                output_image = gr.Image(type='numpy')
            # show only mask
            with gr.Tab(label='Mask'):
                output_mask = gr.Image(type='numpy')
            
             # 블렌드 결과 표시
            with gr.Tab(label='Blend Result'):
                blended_image = gr.Image(type='numpy')
        
    # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        return img, []  # when new image is uploaded, `selected_points` should be empty
    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )

    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, evt: gr.SelectData):
        sel_pix.append((evt.index, 1))   # append the foreground_point
        img_copy = img.copy()
        for point, label in sel_pix:
            cv2.circle(img_copy, tuple(point), 5, (0, 255, 0) if label == 1 else (0, 0, 255), -1)
        return np.array(img_copy)
    input_image.select(
        get_point,
        [input_image, selected_points],
        [input_image],
    )
       
    def blend_images(init_image, mask, prompt):
        init_image_path = "inputs/init_image.png"
        mask_path = "inputs/mask.png"
        Image.fromarray(init_image).save(init_image_path)
        Image.fromarray(mask).save(mask_path)
        
        args = parse_args()
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
    def process_auto(input_image, selected_points, original_image, prompt):
        output_image, mask = predictor_inference(input_image, selected_points)
        blended_image = blend_images(original_image, mask, prompt)
        return output_image, mask, blended_image
    
    button.click(
        process_auto, 
        inputs=[input_image, selected_points, original_image, gr.Textbox(label="프롬프트")],
        outputs=[output_image, output_mask, blended_image]
    )


    # undo the selected point
    def undo_points(orig_img, sel_pix):
        temp = orig_img.copy() if isinstance(orig_img, np.ndarray) else np.array(orig_img)
        # draw points
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
