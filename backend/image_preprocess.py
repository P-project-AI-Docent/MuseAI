# backend/image_preprocess.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from backend.search_image import search_image

# ------------------------------------------------------------
# 0) YOLO 모델 경로 및 로드
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ai_docent/
YOLO_PATH = os.path.join(BASE_DIR, "yolo", "yolov8n.pt")  # YOLO 모델 파일 경로

if not os.path.exists(YOLO_PATH):
    raise FileNotFoundError(f"[ERROR] YOLO 모델이 존재하지 않습니다: {YOLO_PATH}")

yolo_model = YOLO(YOLO_PATH)


# ------------------------------------------------------------
# 1) 컬러 보정 & 대비 향상 (CLAHE + 화이트밸런스)
# ------------------------------------------------------------
def _gray_world_white_balance(bgr):
    # Gray-World WB: 채널 평균을 같게 맞춤
    b, g, r = cv2.split(bgr.astype(np.float32))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    b *= (mean_gray / (mean_b + 1e-6))
    g *= (mean_gray / (mean_g + 1e-6))
    r *= (mean_gray / (mean_r + 1e-6))
    balanced = cv2.merge([b, g, r])
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced

def _apply_clahe(bgr):
    # LAB 공간에서 L 채널에 CLAHE
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out

def enhance_colors(rgb):
    """ RGB → (WB + CLAHE) → RGB """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _gray_world_white_balance(bgr)
    bgr = _apply_clahe(bgr)
    rgb_out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb_out


# ------------------------------------------------------------
# 2) YOLO로 작품 박스 감지
# ------------------------------------------------------------
def detect_artwork_yolo(img_rgb):
    """
    img_rgb: np.ndarray (H,W,3) RGB
    return: (x1,y1,x2,y2) or None
    """
    # ultralytics는 RGB/ndarray 바로 입력 가능
    results = yolo_model.predict(img_rgb, conf=0.25, verbose=False)
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: (N,4)
    # 가장 큰 박스 선택
    max_area, best = 0, None
    for x1, y1, x2, y2 in boxes:
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area, best = area, (int(x1), int(y1), int(x2), int(y2))
    return best


# ------------------------------------------------------------
# 3) 사각형 정렬 & 퍼스펙티브 변환
# ------------------------------------------------------------
def _order_points(pts):
    # 4x2 좌표 → [tl, tr, br, bl] 정렬
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)  # x+y
    diff = np.diff(pts, axis=1)  # y-x

    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def _four_point_transform(image, pts):
    rect = _order_points(pts.astype(np.float32))
    (tl, tr, br, bl) = rect

    # 출력 폭, 높이 계산
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    maxW = max(1, maxW)
    maxH = max(1, maxH)

    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped


# ------------------------------------------------------------
# 4) 액자 안쪽 실제 작품 사각형(내부 프레임) 검출
#    (Canny → Morph close → Contours → 4점 근사 중 최대)
# ------------------------------------------------------------
def detect_inner_quad(rgb_crop):
    """
    입력: RGB crop (YOLO 박스 내부 이미지)
    출력: 4점 좌표 (x,y) 4개 또는 None
    """
    bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # 반사/노이즈에 강인하도록 블러 + Canny + Morph
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    h, w = gray.shape[:2]
    img_area = w * h
    best_quad = None
    best_area = 0

    # 큰 외곽부터 탐색
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < img_area * 0.05:  # 너무 작은 건 skip
            break

        # 다각형 근사
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 사각형 후보
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best_quad = approx.reshape(-1, 2)

    return best_quad


# ------------------------------------------------------------
# 5) 안전 Crop
# ------------------------------------------------------------
def _crop_rect(rgb, x1, y1, x2, y2):
    H, W = rgb.shape[:2]
    x1 = max(0, min(W - 1, x1))
    x2 = max(0, min(W - 1, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return rgb[y1:y2, x1:x2]


# ------------------------------------------------------------
# 6) 최종 파이프라인: YOLO → 내부프레임(사각형) → 퍼스펙티브 보정 → 색보정 → CLIP 검색
# ------------------------------------------------------------
def process_and_search_yolo_enhanced(img_rgb, topk=1):
    """
    1) YOLO로 작품 박스
    2) 박스 내부에서 '작품 사각형(액자 내부)' 감지
    3) 퍼스펙티브 보정 (비뚤어진 액자 정면화)
    4) CLAHE + 화이트밸런스 색 보정
    5) CLIP 이미지 검색
    """
    # 1) YOLO 박스
    box = detect_artwork_yolo(img_rgb)
    if box is None:
        # 실패 시: 전체 이미지 보정 → 검색 (완전 실패 방지)
        enhanced_full = enhance_colors(img_rgb)
        return search_image(enhanced_full, topk=topk)

    x1, y1, x2, y2 = box
    crop_rgb = _crop_rect(img_rgb, x1, y1, x2, y2)
    if crop_rgb is None or crop_rgb.size == 0:
        # YOLO box 이상 시 fallback
        enhanced_full = enhance_colors(img_rgb)
        return search_image(enhanced_full, topk=topk)

    # 2) 내부 사각형(작품) 찾기
    quad = detect_inner_quad(crop_rgb)

    # 3) 퍼스펙티브 보정
    if quad is not None:
        try:
            # quad는 crop 기준 좌표이므로 그대로 warp
            warped = _four_point_transform(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), quad)
            warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        except Exception:
            warped_rgb = crop_rgb
    else:
        # 내부 프레임을 못 찾으면 YOLO crop 그대로
        warped_rgb = crop_rgb

    # 4) 색/대비 보정
    enhanced = enhance_colors(warped_rgb)

    # 5) CLIP 검색
    return search_image(enhanced, topk=topk)
