"""
==========================================================
1조 Object Detection 라벨링 & 증강 통합 웹 서버
==========================================================
- Flask 백엔드
- 기능: 이미지 업로드, ROI 라벨링 저장, 호모그래피 증강,
         푸아송 노이즈 증강, 조명 증강, On-the-fly JSON 스키마 저장
- 프론트엔드: HTML5 Canvas + Vanilla JS
"""

import os
import json
import uuid
import math
import logging
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

# v1.1: 신규 증강 함수 임포트
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.augmentation_recipe import (
    apply_gamma_correction,
    apply_histogram_equalization,
    apply_cutout,
    apply_mosaic,
    _load_image_safe,
    # ── v2.0 신규 ──
    apply_copy_paste,
    transform_polygon_homography,
)

# ──────────────────────────────────────────────────────────
# 앱 초기화
# ──────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DATA = os.path.join(BASE_DIR, 'project_data')

# 폴더 구조 생성
DIRS = {
    'original':  os.path.join(PROJECT_DATA, 'images', 'original'),
    'augmented': os.path.join(PROJECT_DATA, 'images', 'augmented'),
    'labels':    os.path.join(PROJECT_DATA, 'labels'),
    'metadata':  os.path.join(PROJECT_DATA, 'metadata'),
    'logs':      os.path.join(PROJECT_DATA, 'logs'),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB

# 로깅 설정
logging.basicConfig(
    filename=os.path.join(DIRS['logs'], 'server.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# ──────────────────────────────────────────────────────────
# 유틸리티 함수
# ──────────────────────────────────────────────────────────

def generate_image_id():
    """고유 이미지 ID 생성"""
    return str(uuid.uuid4())[:8]


def draw_annotations_on_image(image: np.ndarray, annotations: list) -> np.ndarray:
    """
    증강 이미지 위에 bbox / polygon 라벨을 시각적으로 그려서
    좌표 변환이 올바르게 됐는지 확인할 수 있게 한다.

    - bbox  : 흰색 실선 사각형 + 좌상단 라벨 텍스트
    - polygon: 흰색 실선 다각형 + 각 꼭짓점 파란 점 + 첫 꼭짓점 라벨 텍스트
    """
    img = image.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.28, min(0.5, w / 1280 * 0.5))
    thickness = max(1, int(w / 640))

    for ann in annotations:
        label = ann.get('label', '')
        color = (255, 255, 255)  # 흰색

        if ann.get('type') == 'polygon':
            pts = ann.get('points', [])
            if len(pts) < 2:
                continue
            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_np], isClosed=True, color=color, thickness=thickness)
            # 꼭짓점 점 표시
            for pt in pts:
                cv2.circle(img, (int(pt[0]), int(pt[1])), max(2, thickness + 1),
                           (255, 120, 0), -1)  # 파란 점
            # 첫 꼭짓점 위에 라벨
            if pts:
                tx, ty = int(pts[0][0]), max(int(pts[0][1]) - 4, 0)
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(img, (tx - 1, ty - th - 2), (tx + tw + 2, ty + 2),
                              (0, 0, 0), -1)
                cv2.putText(img, label, (tx, ty), font, font_scale,
                            (0, 255, 80), thickness, cv2.LINE_AA)
        else:
            # bbox
            x = int(ann.get('x', 0))
            y = int(ann.get('y', 0))
            bw = int(ann.get('w', 0))
            bh = int(ann.get('h', 0))
            if bw <= 0 or bh <= 0:
                continue
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, thickness)
            # 라벨 텍스트 (박스 좌상단)
            ty = max(y - 4, 0)
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(img, (x - 1, ty - th - 2), (x + tw + 2, ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(img, label, (x, ty), font, font_scale,
                        (0, 255, 80), thickness, cv2.LINE_AA)

    return img


def draw_augmentation_labels(image: np.ndarray, params: dict) -> np.ndarray:
    """
    증강 이미지 우하단에 적용된 기법 목록을 반투명 오버레이로 표시한다.

    표시 형식 (예):
        ┌─────────────────────────────────┐
        │ Homography  θ=-2.3°             │
        │ Poisson     λ=28                │
        │ Gamma       γ=0.74              │
        │ CLAHE       clip=2.0            │
        │ Cutout      8%                  │
        │ Copy-Paste                      │
        └─────────────────────────────────┘
        (이미지 우하단 반투명 검정 배경 위 녹색 텍스트)

    Args:
        image  : BGR 이미지 numpy array
        params : run_augmentation_pipeline이 반환한 파라미터 dict

    Returns:
        텍스트가 오버레이된 이미지 (원본 수정 없음, 복사본 반환)
    """
    img = image.copy()
    h, w = img.shape[:2]

    # ── 적용된 기법 이름 + 파라미터 조합 ──
    lines = []
    if 'theta_deg' in params or 'h31' in params:
        theta = params.get('theta_deg', 0)
        lines.append(f"Homography  th={theta:.1f}deg")
    if 'poisson_lambda' in params:
        lam = params['poisson_lambda']
        lines.append(f"Poisson     lam={lam:.0f}")
    if params.get('specular_flare'):
        lines.append("Specular Flare")
    if params.get('shadow'):
        lines.append("Shadow")
    if 'gamma' in params:
        lines.append(f"Gamma       g={params['gamma']:.2f}")
    if 'clahe_clip_limit' in params:
        lines.append(f"CLAHE       clip={params['clahe_clip_limit']:.1f}")
    if 'cutout_area_max' in params:
        pct = int(params['cutout_area_max'] * 100)
        lines.append(f"Cutout      {pct}%")
    if params.get('mosaic'):
        lines.append("Mosaic")
    if params.get('copy_paste'):
        lines.append("Copy-Paste")

    if not lines:
        return img

    # ── 텍스트 렌더링 설정 ──
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 이미지 크기에 비례한 폰트 스케일 (640px 기준 0.38)
    font_scale = max(0.30, min(0.48, w / 1280 * 0.50))
    thickness = 1
    pad = 5
    line_h = int(font_scale * 38)   # 한 줄 높이

    # ── 가장 긴 줄 기준으로 박스 너비 계산 ──
    max_tw = max(
        cv2.getTextSize(t, font, font_scale, thickness)[0][0]
        for t in lines
    )
    box_w = max_tw + pad * 2
    box_h = len(lines) * line_h + pad * 2

    # ── 우하단 위치 ──
    margin = 8
    x0 = w - box_w - margin
    y0 = h - box_h - margin

    # ── 반투명 배경 (alpha 0.55) ──
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (w - margin, h - margin),
                  (20, 20, 20), cv2.FILLED)
    img = cv2.addWeighted(overlay, 0.60, img, 0.40, 0)

    # ── 텍스트 렌더링 ──
    for i, text in enumerate(lines):
        tx = x0 + pad
        ty = y0 + pad + (i + 1) * line_h
        # 그림자 (가독성 향상)
        cv2.putText(img, text, (tx + 1, ty + 1),
                    font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # 본문 (연두색)
        cv2.putText(img, text, (tx, ty),
                    font, font_scale, (80, 255, 100), thickness, cv2.LINE_AA)

    return img


def validate_annotation(ann):
    """
    단일 annotation 유효성 검사 (v2.0: bbox / polygon 분기).
    - type='polygon'  → label + points [[x,y]...] 검증
    - type='bbox' (기본) → label + x, y, w, h 검증
    """
    # v2.0: polygon 타입 분기
    if ann.get('type') == 'polygon':
        label = ann.get('label')
        if not isinstance(label, str) or len(label.strip()) == 0:
            return False, "polygon label은 비어있을 수 없습니다."
        points = ann.get('points')
        if not isinstance(points, (list, tuple)) or len(points) < 3:
            return False, "polygon은 3개 이상의 points가 필요합니다."
        for i, pt in enumerate(points):
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                return False, f"points[{i}]는 [x, y] 형식이어야 합니다."
            if not all(isinstance(v, (int, float)) for v in pt):
                return False, f"points[{i}] 좌표는 숫자여야 합니다."
        return True, "OK"

    # 기존 AABB 검증
    required = ['label', 'x', 'y', 'w', 'h']
    for key in required:
        if key not in ann:
            return False, f"'{key}' 필드가 누락되었습니다."
    if not isinstance(ann['label'], str) or len(ann['label'].strip()) == 0:
        return False, "label은 비어있을 수 없습니다."
    for key in ['x', 'y', 'w', 'h']:
        if not isinstance(ann[key], (int, float)) or ann[key] < 0:
            return False, f"'{key}'는 0 이상의 숫자여야 합니다."
    return True, "OK"


def clamp_bbox(x, y, w, h, img_w, img_h):
    """바운딩 박스를 이미지 경계 내로 클램핑"""
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    w = max(0, min(w, img_w - x))
    h = max(0, min(h, img_h - y))
    return x, y, w, h


# ──────────────────────────────────────────────────────────
# 증강 함수들
# ──────────────────────────────────────────────────────────

def apply_homography(image, h31, h32, theta_deg):
    """
    호모그래피 투영 변환 적용
    - h31, h32: 원근 기울기 파라미터
    - theta_deg: 중심 회전각 (도)
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    rad = math.radians(theta_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # 3x3 호모그래피 행렬 구성
    # 중심으로 이동 → 회전+원근 → 원래 위치로
    # 단순화: 직접 행렬 구성
    H = np.array([
        [cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a],
        [sin_a,  cos_a, cy * (1 - cos_a) - cx * sin_a],
        [h31,    h32,   1.0]
    ], dtype=np.float64)

    warped = cv2.warpPerspective(image, H, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE)
    return warped, H


def transform_bbox_by_homography(bbox, H, img_w, img_h):
    """
    호모그래피 행렬 H를 사용하여 바운딩 박스 좌표를 변환
    bbox = [x, y, w, h] → 4개 꼭짓점에 H 곱 → Min-Max Axis-aligned BBox
    """
    x, y, w, h = bbox
    corners = np.array([
        [x,     y,     1],
        [x + w, y,     1],
        [x + w, y + h, 1],
        [x,     y + h, 1],
    ], dtype=np.float64).T  # 3x4

    transformed = H @ corners  # 3x4
    # w'로 나누기 (동차 좌표 → 유클리드 좌표)
    transformed[0] /= transformed[2]
    transformed[1] /= transformed[2]

    xs = transformed[0]
    ys = transformed[1]

    new_x = max(0, float(np.min(xs)))
    new_y = max(0, float(np.min(ys)))
    new_r = min(img_w, float(np.max(xs)))
    new_b = min(img_h, float(np.max(ys)))

    return [new_x, new_y, new_r - new_x, new_b - new_y]


def apply_poisson_noise(image, lam):
    """
    Poisson Noise 합성
    lam: 센서 입사 광량 (λ). 클수록 밝고 상대 노이즈 줄어듦.
    """
    # 이미지를 0-1 범위로 정규화
    img_float = image.astype(np.float64) / 255.0
    # λ로 스케일링 후 푸아송 샘플링
    scaled = img_float * lam
    noisy = np.random.poisson(scaled).astype(np.float64)
    # 다시 0-255로 복원
    noisy = noisy / lam * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_specular_flare(image, intensity=0.4):
    """
    국소적 빛 반사(Specular Flare) 합성
    랜덤 위치에 가우시안 형태의 밝은 원형 빛을 합성
    """
    h, w = image.shape[:2]
    # 랜덤 위치
    cx = np.random.randint(w // 4, 3 * w // 4)
    cy = np.random.randint(h // 4, 3 * h // 4)
    radius = np.random.randint(min(w, h) // 8, min(w, h) // 3)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = np.exp(-(dist ** 2) / (2 * (radius / 2.5) ** 2))
    mask = (mask * intensity * 255).astype(np.float64)

    result = image.astype(np.float64)
    for c in range(3):
        result[:, :, c] += mask
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_shadow(image, intensity=0.5):
    """
    그라데이션 그림자 합성
    한쪽에서 다른 쪽으로 점진적으로 어두워지는 효과
    """
    h, w = image.shape[:2]
    # 랜덤 방향 (0: 좌→우, 1: 우→좌, 2: 상→하, 3: 하→상)
    direction = np.random.randint(0, 4)

    if direction == 0:
        grad = np.linspace(1.0, 1 - intensity, w)
        mask = np.tile(grad, (h, 1))
    elif direction == 1:
        grad = np.linspace(1 - intensity, 1.0, w)
        mask = np.tile(grad, (h, 1))
    elif direction == 2:
        grad = np.linspace(1.0, 1 - intensity, h)
        mask = np.tile(grad.reshape(-1, 1), (1, w))
    else:
        grad = np.linspace(1 - intensity, 1.0, h)
        mask = np.tile(grad.reshape(-1, 1), (1, w))

    result = image.astype(np.float64)
    for c in range(3):
        result[:, :, c] *= mask
    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────
# 증강 파이프라인 (V3: 디폴트 값 기반 자동 실행)
# ──────────────────────────────────────────────────────────

# 디폴트 파라미터 범위
DEFAULTS = {
    'homography': {
        'h31_range': (-0.001, 0.001),
        'h32_range': (-0.001, 0.001),
        'theta_range': (-5.0, 5.0),
    },
    'poisson': {
        'lambda_range': (20, 40),
    }
}


def _load_random_images_for_mosaic(original_dir, count=3, exclude_filename=''):
    """
    Mosaic용: original 폴더에서 무작위로 count개 이미지를 로드한다.
    exclude_filename: 기준 이미지 자신은 제외 (중복 방지)

    Returns:
        [(image_ndarray, []), ...]  — 어노테이션은 빈 리스트 (단순 배경용)
        dataset_index.json가 있으면 해당 어노테이션도 함께 로드 시도
    """
    pool = []
    try:
        all_files = [
            f for f in os.listdir(original_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            and f != exclude_filename
        ]
    except OSError:
        return pool

    if not all_files:
        return pool

    # dataset_index.json에서 어노테이션 매핑 시도
    ann_map = {}
    index_path = os.path.join(os.path.dirname(original_dir), '..', 'metadata', 'dataset_index.json')
    index_path = os.path.normpath(index_path)
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
            labels_dir = os.path.normpath(
                os.path.join(os.path.dirname(original_dir), '..', 'labels')
            )
            for entry in index:
                img_basename = os.path.basename(
                    entry.get('original_image_path', entry.get('original_image', ''))
                )
                label_rel = entry.get('label_json_path', entry.get('label_json', ''))
                label_full = os.path.join(
                    os.path.dirname(labels_dir), label_rel
                )
                if os.path.exists(label_full):
                    try:
                        with open(label_full, 'r', encoding='utf-8') as lf:
                            ldata = json.load(lf)
                        ann_map[img_basename] = ldata.get('annotations', [])
                    except Exception:
                        pass
        except Exception:
            pass

    # 무작위 샘플링
    chosen = np.random.choice(
        all_files,
        size=min(count, len(all_files)),
        replace=False
    ).tolist()

    for fname in chosen:
        fpath = os.path.join(original_dir, fname)
        img = _load_image_safe(fpath)
        if img is not None:
            anns = ann_map.get(fname, [])
            pool.append((img, anns))

    return pool


def run_augmentation_pipeline(image, annotations, num_augmented=5,
                              enable_homography=True,
                              enable_noise=True,
                              enable_specular=False,
                              enable_shadow=False,
                              label_sensitive_labels=None,
                              # ── v1.1 파라미터 ──
                              enable_gamma=False,
                              gamma_range=(0.6, 1.6),
                              enable_clahe=False,
                              clahe_clip_limit=2.0,
                              enable_cutout=False,
                              cutout_area_max=0.08,
                              enable_mosaic=False,
                              original_dir=None,
                              source_filename='',
                              mosaic_selected=None,   # 사용자가 직접 선택한 파일명 리스트
                              # ── v2.0 신규 파라미터 ──
                              enable_copy_paste=False):
    """
    V4 증강 파이프라인 (v1.1 업데이트).

    신규 파라미터:
        enable_gamma       : Gamma Correction 활성화
        gamma_range        : (min, max) 감마 샘플링 범위
        enable_clahe       : CLAHE 히스토그램 평탄화 활성화
        clahe_clip_limit   : CLAHE clipLimit (1.0~5.0)
        enable_cutout      : Cutout(Random Erasing) 활성화
        cutout_area_max    : Cutout 최대 면적 비율 (0.02~0.15)
        enable_mosaic      : Mosaic Augmentation 활성화
        original_dir       : Mosaic용 이미지 풀 경로
        source_filename    : Mosaic 시 자기 자신 제외용 파일명

    반환: [(augmented_image, augmented_annotations, augmentation_params), ...]
    """
    if label_sensitive_labels is None:
        label_sensitive_labels = []

    img_h, img_w = image.shape[:2]
    results = []

    # Mosaic용 이미지 풀 사전 로드 (반복 I/O 방지)
    mosaic_pool = []
    if enable_mosaic and original_dir:
        if mosaic_selected:
            # 사용자가 직접 선택한 이미지 파일명으로 풀 구성
            for fname in mosaic_selected:
                fpath = os.path.join(original_dir, fname)
                img = _load_image_safe(fpath)
                if img is not None:
                    mosaic_pool.append((img, []))
        else:
            # 선택이 없으면 기존처럼 랜덤 로드
            mosaic_pool = _load_random_images_for_mosaic(
                original_dir, count=9, exclude_filename=source_filename
            )

    for i in range(num_augmented):
        aug_img = image.copy()
        aug_anns = [dict(a) for a in annotations]  # 깊은 복사
        params = {}

        # ── [기존] 호모그래피 증강 ──
        if enable_homography:
            h31 = np.random.uniform(*DEFAULTS['homography']['h31_range'])
            h32 = np.random.uniform(*DEFAULTS['homography']['h32_range'])
            theta = np.random.uniform(*DEFAULTS['homography']['theta_range'])

            has_sensitive = any(
                a['label'] in label_sensitive_labels for a in aug_anns
            )
            if has_sensitive:
                theta = np.clip(theta, -3.0, 3.0)

            aug_img, H = apply_homography(aug_img, h31, h32, theta)

            for ann in aug_anns:
                # v2.0: polygon / bbox 분기
                if ann.get('type') == 'polygon' and ann.get('points'):
                    ann['points'] = transform_polygon_homography(
                        ann['points'], H, img_w, img_h
                    )
                else:
                    new_bbox = transform_bbox_by_homography(
                        [ann['x'], ann['y'], ann['w'], ann['h']], H, img_w, img_h
                    )
                    ann['x'], ann['y'], ann['w'], ann['h'] = new_bbox

            params['h31'] = round(h31, 6)
            params['h32'] = round(h32, 6)
            params['theta_deg'] = round(theta, 2)

        # ── [기존] 푸아송 노이즈 ──
        if enable_noise:
            lam = np.random.uniform(*DEFAULTS['poisson']['lambda_range'])
            aug_img = apply_poisson_noise(aug_img, lam)
            params['poisson_lambda'] = round(lam, 1)

        # ── [기존] 조명 증강 ──
        if enable_specular:
            aug_img = apply_specular_flare(aug_img)
            params['specular_flare'] = True

        if enable_shadow:
            aug_img = apply_shadow(aug_img)
            params['shadow'] = True

        # ── [v1.1 신규] Gamma Correction ──
        if enable_gamma:
            gamma_val = round(float(np.random.uniform(*gamma_range)), 2)
            try:
                aug_img = apply_gamma_correction(aug_img, gamma_val)
                params['gamma'] = gamma_val
            except Exception as e:
                logging.warning(f"Gamma Correction 실패: {e}")

        # ── [v1.1 신규] CLAHE ──
        if enable_clahe:
            try:
                aug_img = apply_histogram_equalization(
                    aug_img,
                    use_clahe=True,
                    clip_limit=clahe_clip_limit
                )
                params['clahe_clip_limit'] = clahe_clip_limit
            except Exception as e:
                logging.warning(f"CLAHE 실패: {e}")

        # ── [v1.1 신규] Cutout ──
        if enable_cutout:
            try:
                aug_img, aug_anns = apply_cutout(
                    aug_img, aug_anns,
                    area_ratio_min=0.02,
                    area_ratio_max=float(cutout_area_max),
                    fill_mode='mean',
                    drop_covered=True,
                    drop_threshold=0.5  # BBox의 50% 이상 가려지면 드롭
                )
                params['cutout_area_max'] = cutout_area_max
            except Exception as e:
                logging.warning(f"Cutout 실패: {e}")

        # ── [v1.1 신규] Mosaic ──
        if enable_mosaic:
            try:
                # 풀이 3장 미만이면 현재 이미지(자기 자신)를 복제해서 채운다.
                # 이렇게 하면 이미지 1장만 있어도 Mosaic이 작동한다.
                effective_pool = list(mosaic_pool)
                while len(effective_pool) < 3:
                    # 자기 자신을 복사본으로 추가 (얕은 복사 → 어노테이션은 같은 참조지만 safe)
                    effective_pool.append((image.copy(), [dict(a) for a in annotations]))

                chosen_idx = np.random.choice(
                    len(effective_pool), size=3, replace=False
                ).tolist()
                chosen_pool = [effective_pool[j] for j in chosen_idx]

                aug_img, aug_anns = apply_mosaic(
                    aug_img, aug_anns,
                    image_pool=chosen_pool,
                    output_w=img_w,
                    output_h=img_h
                )
                # 실제 외부 이미지 사용 여부 표시
                params['mosaic'] = True
                params['mosaic_self_dup'] = len(mosaic_pool) < 3
            except Exception as e:
                logging.warning(f"Mosaic 실패: {e}")

        # ── [v2.0 신규] Copy-Paste Augmentation ──
        # polygon 라벨이 하나 이상 있고 배경 풀이 로드된 경우에만 실행
        if enable_copy_paste:
            try:
                # 배경 풀 = Mosaic용 풀 재활용 (이미 로드되어 있으면 사용)
                bg_pool = [img for img, _ in mosaic_pool] if mosaic_pool else []

                # 풀이 없으면 직접 로드
                if not bg_pool and original_dir:
                    raw_pool = _load_random_images_for_mosaic(
                        original_dir, count=5, exclude_filename=source_filename
                    )
                    bg_pool = [img for img, _ in raw_pool]

                if bg_pool:
                    aug_img, aug_anns = apply_copy_paste(
                        aug_img, aug_anns,
                        background_pool=bg_pool,
                        num_pastes=1
                    )
                    params['copy_paste'] = True
            except Exception as e:
                logging.warning(f"Copy-Paste 실패: {e}")

        # ── 최종 좌표 클램핑 및 정수 변환 (v2.0: polygon/bbox 분기) ──
        for ann in aug_anns:
            if ann.get('type') == 'polygon':
                # polygon은 apply_copy_paste 내부에서 이미 clip 처리됨
                # 추가 안전 처리: 각 점 범위 재확인
                ann['points'] = [
                    [int(np.clip(px, 0, img_w)), int(np.clip(py, 0, img_h))]
                    for px, py in ann.get('points', [])
                ]
            else:
                ann['x'], ann['y'], ann['w'], ann['h'] = clamp_bbox(
                    ann['x'], ann['y'], ann['w'], ann['h'], img_w, img_h
                )
                ann['x'] = round(ann['x'])
                ann['y'] = round(ann['y'])
                ann['w'] = round(ann['w'])
                ann['h'] = round(ann['h'])

        results.append((aug_img, aug_anns, params))

    return results


# ──────────────────────────────────────────────────────────
# 라우트: 페이지
# ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ──────────────────────────────────────────────────────────
# 라우트: 이미지 업로드
# ──────────────────────────────────────────────────────────

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """이미지 파일 업로드 → original/ 폴더에 저장"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '파일이 없습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': '파일명이 비어 있습니다.'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(DIRS['original'], filename)
    file.save(filepath)

    logging.info(f"이미지 업로드: {filename}")
    return jsonify({
        'success': True,
        'message': '업로드 성공',
        'filename': filename,
        'url': f'/images/original/{filename}'
    })


@app.route('/images/original/<filename>')
def serve_original(filename):
    return send_from_directory(DIRS['original'], filename)


@app.route('/images/augmented/<filename>')
def serve_augmented(filename):
    return send_from_directory(DIRS['augmented'], filename)


# ──────────────────────────────────────────────────────────
# 라우트: 원본 이미지 목록 조회 (Mosaic 선택용)
# ──────────────────────────────────────────────────────────

@app.route('/api/images/list', methods=['GET'])
def list_original_images():
    """
    project_data/images/original/ 에 있는 이미지 목록을 반환한다.
    프론트엔드 Mosaic 이미지 선택 패널에서 사용.
    응답 형식: { "images": [{"filename": "cat.jpg", "url": "/images/original/cat.jpg"}, ...] }
    """
    try:
        files = [
            f for f in os.listdir(DIRS['original'])
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]
        files.sort()
        result = [
            {'filename': f, 'url': f'/images/original/{f}'}
            for f in files
        ]
        return jsonify({'images': result})
    except Exception as e:
        return jsonify({'images': [], 'error': str(e)})


# ──────────────────────────────────────────────────────────
# 라우트: 저장 + 증강 실행
# ──────────────────────────────────────────────────────────

@app.route('/api/save', methods=['POST'])
def save_data():
    """
    프론트엔드에서 전송된 데이터를 검증 → annotation 저장 → 증강 실행
    입력 형식:
    {
        "filename": "image.jpg",
        "image_width": 1280,
        "image_height": 720,
        "annotations": [
            {"label": "object", "x": 10, "y": 20, "w": 100, "h": 50, "label_sensitive": false}
        ],
        "augmentation": {
            "enable_homography": true,
            "enable_noise": true,
            "enable_specular": false,
            "enable_shadow": false,
            "num_augmented": 5
        }
    }
    """
    data = request.json
    if not data:
        return jsonify({'success': False, 'message': 'JSON 데이터가 없습니다.'}), 400

    filename = data.get('filename')
    annotations = data.get('annotations', [])
    img_w = data.get('image_width', 0)
    img_h = data.get('image_height', 0)
    aug_opts = data.get('augmentation', {})

    # ── 검증 ──
    if not filename:
        return jsonify({'success': False, 'message': 'filename이 필요합니다.'}), 400

    filepath = os.path.join(DIRS['original'], filename)
    if not os.path.exists(filepath):
        return jsonify({'success': False, 'message': '서버에 원본 이미지가 없습니다.'}), 404

    for i, ann in enumerate(annotations):
        valid, msg = validate_annotation(ann)
        if not valid:
            return jsonify({
                'success': False,
                'message': f'annotation[{i}] 오류: {msg}'
            }), 400
        # v2.0: polygon은 x,y,w,h 없으므로 bbox 전용 클램핑만 적용
        if ann.get('type') != 'polygon':
            ann['x'], ann['y'], ann['w'], ann['h'] = clamp_bbox(
                ann['x'], ann['y'], ann['w'], ann['h'], img_w, img_h
            )

    # ── 이미지 ID 생성 ──
    image_id = generate_image_id()
    base_name = os.path.splitext(filename)[0]

    # ── annotation JSON 저장 ──
    annotation_data = {
        'image_id': image_id,
        'image_name': filename,
        'image_width': img_w,
        'image_height': img_h,
        'task_type': 'object_detection',
        'annotations': annotations,
        'applied_augmentations': [],
        'saved_at': datetime.now().isoformat()
    }

    label_json_name = f"{base_name}_{image_id}.json"
    label_json_path = os.path.join(DIRS['labels'], label_json_name)
    with open(label_json_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, indent=2, ensure_ascii=False)

    # ── 증강 실행 ──
    augmented_paths = []
    augmented_params_list = []

    if aug_opts:
        # 이미지 로드 (한글 경로 대응)
        img_array = np.fromfile(filepath, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({'success': False, 'message': '이미지를 읽을 수 없습니다.'}), 500

        # Label Sensitive 라벨 목록 추출
        sensitive_labels = [
            a['label'] for a in annotations if a.get('label_sensitive', False)
        ]

        num_aug = aug_opts.get('num_augmented', 5)
        results = run_augmentation_pipeline(
            image, annotations,
            num_augmented=num_aug,
            enable_homography=aug_opts.get('enable_homography', True),
            enable_noise=aug_opts.get('enable_noise', True),
            enable_specular=aug_opts.get('enable_specular', False),
            enable_shadow=aug_opts.get('enable_shadow', False),
            label_sensitive_labels=sensitive_labels,
            # ── v1.1 신규 파라미터 파싱 ──
            enable_gamma=aug_opts.get('enable_gamma', False),
            gamma_range=(
                float(aug_opts.get('gamma_min', 0.6)),
                float(aug_opts.get('gamma_max', 1.6))
            ),
            enable_clahe=aug_opts.get('enable_clahe', False),
            clahe_clip_limit=float(aug_opts.get('clahe_clip_limit', 2.0)),
            enable_cutout=aug_opts.get('enable_cutout', False),
            cutout_area_max=float(aug_opts.get('cutout_area_max', 0.08)),
            enable_mosaic=aug_opts.get('enable_mosaic', False),
            original_dir=DIRS['original'],
            source_filename=filename,
            mosaic_selected=aug_opts.get('mosaic_selected_images') or None,
            # ── v2.0 신규 ──
            enable_copy_paste=aug_opts.get('enable_copy_paste', False),
        )

        ext = os.path.splitext(filename)[1]
        for idx, (aug_img, aug_anns, params) in enumerate(results):
            aug_filename = f"{base_name}_{image_id}_aug{idx}{ext}"
            aug_filepath = os.path.join(DIRS['augmented'], aug_filename)

            # ── bbox / polygon 라벨을 이미지에 그린다 (좌표 변환 시각적 확인) ──
            aug_img_drawn = draw_annotations_on_image(aug_img, aug_anns)
            # ── 증강 기법 오버레이 텍스트 추가 ──
            aug_img_labeled = draw_augmentation_labels(aug_img_drawn, params)

            # 유니코드 경로 안전 저장
            ok, buf = cv2.imencode(ext, aug_img_labeled)
            if ok:
                buf.tofile(aug_filepath)

            # 증강 annotation JSON 저장
            aug_ann_data = {
                'image_id': image_id,
                'augmented_index': idx,
                'source_image': filename,
                'augmented_image': aug_filename,
                'annotations': aug_anns,
                'augmentation_params': params,
                'saved_at': datetime.now().isoformat()
            }
            aug_json_name = f"{base_name}_{image_id}_aug{idx}.json"
            aug_json_path = os.path.join(DIRS['labels'], aug_json_name)
            with open(aug_json_path, 'w', encoding='utf-8') as f:
                json.dump(aug_ann_data, f, indent=2, ensure_ascii=False)

            augmented_paths.append(aug_filename)
            augmented_params_list.append(params)

        # annotation_data에 증강 정보 추가
        annotation_data['applied_augmentations'] = augmented_params_list
        with open(label_json_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)

    # ── dataset_index.json 갱신 ──
    index_path = os.path.join(DIRS['metadata'], 'dataset_index.json')
    if os.path.exists(index_path):
        with open(index_path, 'r', encoding='utf-8') as f:
            dataset_index = json.load(f)
    else:
        dataset_index = []

    dataset_index.append({
        'image_id': image_id,
        'original_image_path': f"images/original/{filename}",
        'label_json_path': f"labels/{label_json_name}",
        'augmented_image_paths': [f"images/augmented/{p}" for p in augmented_paths],
        'created_at': datetime.now().isoformat()
    })

    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_index, f, indent=2, ensure_ascii=False)

    # ── 로그 ──
    logging.info(
        f"저장 완료: image_id={image_id}, "
        f"annotations={len(annotations)}, "
        f"augmented={len(augmented_paths)}"
    )

    # ── 응답 ──
    return jsonify({
        'success': True,
        'message': '저장 및 증강 완료',
        'image_id': image_id,
        'original_image_path': f"images/original/{filename}",
        'label_json_path': f"labels/{label_json_name}",
        'augmented_image_paths': [f"images/augmented/{p}" for p in augmented_paths],
        'augmented_count': len(augmented_paths),
        'metadata_path': f"metadata/dataset_index.json"
    })


# ──────────────────────────────────────────────────────────
# 라우트: 내보내기 목록 조회
# ──────────────────────────────────────────────────────────

@app.route('/api/exports', methods=['GET'])
def list_exports():
    """저장된 데이터 인덱스 조회"""
    index_path = os.path.join(DIRS['metadata'], 'dataset_index.json')
    if not os.path.exists(index_path):
        return jsonify({'entries': []})
    with open(index_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify({'entries': data})


# ──────────────────────────────────────────────────────────
# 서버 시작
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 50)
    print("1조 Object Detection 라벨링 서버 시작")
    print(f"프로젝트 데이터 경로: {PROJECT_DATA}")
    print("http://localhost:5000 에서 접속하세요")
    print("=" * 50)
    app.run(debug=True, port=5000)
