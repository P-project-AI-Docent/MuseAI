# backend/yolo_detect.py
import cv2
import numpy as np
from ultralytics import YOLO
import os

# ★ 프로젝트 기준 절대경로로 변환
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolo", "yolov8n.pt")

# ★ 변수 이름 오타 수정됨
yolo_model = YOLO(MODEL_PATH)

def detect_artwork_yolo(img):
    """
    YOLO로 bounding box 반환
    img: np.array (H,W,3)
    return: (x, y, w, h) or None
    """
    results = yolo_model(img, conf=0.25, verbose=False)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    # 가장 큰 bounding box 찾기
    candidates = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        candidates.append((area, (x1, y1, x2, y2)))

    candidates.sort(reverse=True)
    _, (x1, y1, x2, y2) = candidates[0]
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
