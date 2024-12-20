# @title #重新加载f
import os
import argparse
from typing import Tuple, Optional
import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from tqdm import tqdm
import sys
import json
sys.path.append('/content/florence-sam-colab')
from utils.video import generate_unique_name, create_directory, delete_directory
from utils.florence import load_florence_model, run_florence_inference, FLORENCE_DETAILED_CAPTION_TASK, FLORENCE_CAPTION_TO_PHRASE_GROUNDING_TASK, FLORENCE_OPEN_VOCABULARY_DETECTION_TASK
from utils.modes import IMAGE_INFERENCE_MODES, IMAGE_OPEN_VOCABULARY_DETECTION_MODE, IMAGE_CAPTION_GROUNDING_MASKS_MODE, VIDEO_INFERENCE_MODES
import pickle



VIDEO_SCALE_FACTOR = 0.5
VIDEO_TARGET_DIRECTORY = "tmp"
create_directory(directory_path=VIDEO_TARGET_DIRECTORY)

DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

FLORENCE_MODEL, FLORENCE_PROCESSOR = load_florence_model(device=DEVICE)

def main(output_video):
    os.chdir("/content/florence-sam-colab")
    import pickle
    # 从文件中加载变量
    with open('/content/texts.pkl', 'rb') as file:
        texts = pickle.load(file)
    print(texts)
    frame_generator = sv.get_video_frames_generator(output_video)
    frame = next(frame_generator)
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    width, height = frame.size

    #print(f"宽度: {width}, 高度: {height}")
    detections_list = []
    all_ok_bboxes = []
    men_count = 0
    for text in texts:
        _, result = run_florence_inference(
            model=FLORENCE_MODEL,
            processor=FLORENCE_PROCESSOR,
            device=DEVICE,
            image=frame,
            task=FLORENCE_OPEN_VOCABULARY_DETECTION_TASK,
            text=text
        )

        detections = sv.Detections.from_lmm(
            lmm=sv.LMM.FLORENCE_2,
            result=result,
            resolution_wh=frame.size
        )
        print(result)
        half_area = width * height * 0.5

        # 存储所有 the table 的边界框和面积
        table_bboxes = []
        table_areas = []
        given_area =1000
        # 统计 men 的数量
        #men_count = 0
        men_label_flag = False
        accflag = False
        with open('/content/accflag.pkl', 'wb') as file:
          pickle.dump(accflag, file)

        for bbox, label in zip(result['<OPEN_VOCABULARY_DETECTION>']['bboxes'], result['<OPEN_VOCABULARY_DETECTION>']['bboxes_labels']):
            if label == 'men':
                men_label_flag = True
                all_ok_bboxes.append([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                men_count += 1
                print('men add 1',men_count)
            if label == 'ping pong ball':
                # 计算当前 ping pong ball 的面积
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                # 检查面积是否不超过给定边界框的面积
                if area <= given_area:
                    all_ok_bboxes.append([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
            elif label == 'the table':
                # 计算当前 the table 的面积
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                table_bboxes.append([[bbox[0] - 100, bbox[1]], [bbox[2] + 100, bbox[3]]])
                table_areas.append(area)
            elif label == 'table tennis bat':
                all_ok_bboxes.append([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])

        # 找到面积最大的 the table
        if table_areas:
            max_area_index = table_areas.index(max(table_areas))
            max_area_bbox = table_bboxes[max_area_index]
            
            # 检查面积是否超过50%
            if max(table_areas) < half_area:
                all_ok_bboxes.append(max_area_bbox)
        # 如果 men 的数量为 1，则打印 result
        if False and men_count == 0:
            print('men_count == 0',men_count)
            # 保存变量
            accflag = True
            with open('/content/accflag.pkl', 'wb') as file:
                pickle.dump(accflag, file)
        if men_label_flag and men_count <= 1:
            print('men_count <= 1',men_count)
            # 保存变量
            accflag = True
            with open('/content/accflag.pkl', 'wb') as file:
                pickle.dump(accflag, file)

    #print(all_ok_bboxes)
    # 保存变量到文件
    with open('/content/all_ok_bboxes.pkl', 'wb') as file:
        pickle.dump(all_ok_bboxes, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a video.")
    parser.add_argument("output_video", type=str, help="Path to the output video file")
    args = parser.parse_args()
    all_ok_bboxes = main(args.output_video)
    #print(all_ok_bboxes)
