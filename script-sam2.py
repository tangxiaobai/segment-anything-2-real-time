# script.py

import argparse
import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from sam2.build_sam import build_sam2_camera_predictor
from tqdm import tqdm  # 导入 tqdm

def main(all_ok_bboxes, output_video):
    # 设置工作目录
    os.chdir("/content/segment-anything-2-real-time/")

    checkpoint = "./checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

    out_dir = "/content/output"
    os.makedirs(out_dir, exist_ok=True)
    import shutil
    shutil.rmtree('/content/output', ignore_errors=True)
    os.makedirs('/content/output', exist_ok=True)

    cap = cv2.VideoCapture(output_video)

    if_init = False
    n = 1

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc="Processing frames") as pbar:  # 使用 tqdm 创建进度条
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            width, height = frame.shape[:2][::-1]
            if not if_init:
                predictor.load_first_frame(frame)
                if_init = True
                ann_frame_idx = 0  # the frame index we interact with

                # 遍历所有组的坐标
                for i, bbox_coords in enumerate(all_ok_bboxes):
                    ann_obj_id = i + 1  # 对象 ID 从 1 开始
                    bbox = np.array(bbox_coords, dtype=np.float32)

                    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
                    )

                print(ann_obj_id)
            else:
                out_obj_ids, out_mask_logits = predictor.track(frame)

                all_mask = np.zeros((height, width, 1), dtype=np.uint8)
                for i in range(0, len(out_obj_ids)):
                    out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                        np.uint8
                    ) * 255

                    all_mask = cv2.bitwise_or(all_mask, out_mask)

                #all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
                #frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
     

                # 假设 all_mask 和 frame 已经定义
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

                # 对 all_mask 进行膨胀操作
                dilated_mask = cv2.dilate(all_mask, kernel, iterations=1)
                #all_mask = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR)  # 将 all_mask 转换为三通道图像
                all_mask = dilated_mask.astype(np.uint8) * 255 
                # 使用掩膜从原始图片中提取部分区域
                cv2.imwrite('/content/tem.jpg', frame)
                image = cv2.imread('/content/tem.jpg')
                masked_image = cv2.bitwise_and(image, image, mask=all_mask)
                # 将掩膜应用于原始图片  
                blurred_image = cv2.GaussianBlur(frame, (21, 21), 500)  # 使用较大的核大小进行模糊
                # 将提取的部分区域叠加到模糊后的图片上
                blurred_image = cv2.bitwise_and(blurred_image, blurred_image, mask=~all_mask)
                    # 将提取的部分区域叠加到模糊后的图片上
                frame = np.where(dilated_mask[:, :, None] > 0, masked_image, blurred_image)
                # result 即为只保留 all_mask 遮罩内容的图像
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite("/content/output/" + str(n) + ".jpg", frame)
            n = n + 1
            pbar.update(1)  # 更新进度条

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video frames with SAM2.")
    parser.add_argument('--bboxes', type=str, required=True, help='Bboxes in JSON format')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')

    args = parser.parse_args()
    import pickle

    # 从文件中加载变量
    with open('/content/all_ok_bboxes.pkl', 'rb') as file:
        all_ok_bboxes = pickle.load(file)
    
    main(all_ok_bboxes, args.video)
