"""
==========================================================
augmentation_recipe.py — On-the-fly 데이터 증강 파이프라인
==========================================================
역할:
  저장된 JSON "레시피"를 읽어, 원본 이미지에 수학적으로 정확한
  증강을 실시간으로 적용하고, 바운딩 박스 좌표를 동기화하는
  PyTorch Dataset 호환 클래스를 제공한다.

On-the-fly vs 물리적 저장 비교:
  ┌─────────────────┬──────────────────┬─────────────────────┐
  │                 │ 물리적 저장       │ On-the-fly (레시피) │
  ├─────────────────┼──────────────────┼─────────────────────┤
  │ 디스크 사용량    │ 원본 × 증강 수   │ 원본 + JSON만       │
  │ 재현성          │ 파일 있으면 재현  │ 동일 시드로 재현    │
  │ 유연성          │ 파라미터 변경 불가│ 언제든 파라미터 조정│
  │ 학습 속도       │ I/O 병목 발생    │ 메모리서 실시간 처리│
  └─────────────────┴──────────────────┴─────────────────────┘

수학적 핵심 — Homography BBox 변환:
  1. 원본 BBox의 4개 꼭짓점을 동차 좌표계(Homogeneous)로 표현
     → [x, y, 1]^T
  2. 3×3 Homography 행렬 H를 곱해 변환된 4점을 구함
     → H @ [x, y, 1]^T = [x', y', w']^T
  3. w'로 나눠 유클리드 좌표로 복원 (perspective divide)
     → (x'/w', y'/w')
  4. 변환된 4점의 min/max로 Axis-Aligned BBox(AABB) 계산

Poisson 노이즈 (물리적 근거):
  - 카메라 센서에 도달하는 광자(photon) 수는 Poisson 분포를 따름
  - 따라서 밝은 픽셀(광자 많음)일수록 절대 노이즈는 크지만
    상대적 노이즈(SNR)는 오히려 작아짐
  - Gaussian 노이즈는 밝기와 무관하게 노이즈를 가하므로 비물리적
"""

import math
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────
# 파라미터 샘플러: JSON 레시피 생성 모듈
# ──────────────────────────────────────────────────────────

# 프롬프트에 명시된 고정 파라미터 범위
AUG_PARAM_RANGES = {
    'h31'            : (-0.001,  0.001),   # X축 원근 기울기
    'h32'            : (-0.001,  0.001),   # Y축 원근 기울기
    'theta_deg'      : (-5.0,    5.0),     # 중심 회전각 (도)
    'poisson_lambda' : (20.0,    40.0),    # 센서 광량 λ
}


def sample_augmentation_params(
    n: int = 5,
    label_sensitive: bool = False,
    enable_homography: bool = True,
    enable_noise: bool = True,
    enable_specular: bool = False,
    enable_shadow: bool = False,
    seed: Optional[int] = None
) -> List[dict]:
    """
    무작위 증강 파라미터를 n개 샘플링하여 레시피 목록을 반환한다.

    Args:
        n              : 생성할 레시피 수
        label_sensitive: True이면 회전 범위를 ±3°로 축소 (방향 민감 라벨 보호)
        enable_*       : 각 증강 타입 활성화 여부
        seed           : 재현성을 위한 난수 시드 (None이면 랜덤)

    Returns:
        [
            {
                "h31": -0.0005,
                "h32":  0.0003,
                "theta_deg": 2.1,
                "poisson_lambda": 28.5,
                "specular_flare": false,
                "shadow": true
            },
            ...
        ]

    label_sensitive 방어 로직:
        방향 정보를 담은 라벨(화살표, 6↔9 등)은 큰 회전 시 의미가
        바뀐다. 이를 막기 위해 theta_deg 범위를 절반으로 축소한다.
    """
    if seed is not None:
        np.random.seed(seed)

    params_list = []
    for _ in range(n):
        p: Dict[str, Any] = {}

        if enable_homography:
            p['h31'] = round(float(np.random.uniform(*AUG_PARAM_RANGES['h31'])), 6)
            p['h32'] = round(float(np.random.uniform(*AUG_PARAM_RANGES['h32'])), 6)

            # label_sensitive이면 회전 범위를 ±3°로 제한
            theta_range = (-3.0, 3.0) if label_sensitive else AUG_PARAM_RANGES['theta_deg']
            p['theta_deg'] = round(float(np.random.uniform(*theta_range)), 2)

        if enable_noise:
            p['poisson_lambda'] = round(
                float(np.random.uniform(*AUG_PARAM_RANGES['poisson_lambda'])), 1
            )

        p['specular_flare'] = enable_specular
        p['shadow']         = enable_shadow

        params_list.append(p)

    return params_list


# ──────────────────────────────────────────────────────────
# 개별 증강 함수
# ──────────────────────────────────────────────────────────

def build_homography_matrix(
    h31: float, h32: float, theta_deg: float,
    img_w: int, img_h: int
) -> np.ndarray:
    """
    3×3 Homography 행렬을 구성한다.

    행렬 구조 (회전 + 원근 동시 처리):
    ┌                            ┐
    │  cos θ  -sin θ  tx         │
    │  sin θ   cos θ  ty         │   ← 아핀(회전+이동) 부분
    │   h31    h32     1         │   ← 원근(perspective) 행
    └                            ┘

    where:
        tx = cx(1 - cosθ) + cy sinθ   (회전 중심을 이미지 중앙으로)
        ty = cy(1 - cosθ) - cx sinθ

    h31, h32는 마지막 행(원근 행)의 계수로,
    이 값이 0이 아니면 평행선이 한 점으로 수렴하는 원근감이 생긴다.

    자료형 주의:
        np.float64를 명시하지 않으면 cv2.warpPerspective에서
        dtypes 불일치 오류가 발생할 수 있음.
    """
    cx, cy = img_w / 2.0, img_h / 2.0
    rad = math.radians(theta_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    # 회전 중심 이동 성분 (이미지 중앙을 기준으로 회전하기 위함)
    tx = cx * (1.0 - cos_a) + cy * sin_a
    ty = cy * (1.0 - cos_a) - cx * sin_a

    H = np.array([
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
        [h31,    h32,   1.0]
    ], dtype=np.float64)

    return H


def apply_homography(
    image: np.ndarray,
    h31: float, h32: float, theta_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지에 Homography 변환을 적용한다.

    경계 처리:
        [수정] BORDER_REPLICATE 사용: 원근 변환 후 비어있는 가장자리를
        가장 가까운 테두리 픽셀 값으로 채운다 (자연스러운 확장).

        [버그 원인] 기존 BORDER_REFLECT_101은 이미지를 거울 반사하여 채우기 때문에
        원근 변환 시 좌우/상하 대칭 선이 시각적으로 강하게 보이는 아티팩트 발생.
        특히 h31, h32 원근 파라미터가 적용될 때 이미지 경계가 "찢기는 선" 처럼 보임.

    Returns:
        (변환된 이미지, 3×3 H 행렬)
    """
    h, w = image.shape[:2]
    H = build_homography_matrix(h31, h32, theta_deg, w, h)
    warped = cv2.warpPerspective(
        image, H, (w, h),
        borderMode=cv2.BORDER_REPLICATE  # ← 수정: 반사 아님, 테두리 픽셀 확장
    )
    return warped, H


def transform_bbox_homography(
    bbox: List[float],
    H: np.ndarray,
    img_w: int,
    img_h: int
) -> List[int]:
    """
    바운딩 박스의 4개 꼭짓점에 Homography 행렬을 적용하고
    Axis-Aligned Bounding Box(AABB)로 변환한다.

    단계별 설명:
      1. bbox [x, y, w, h] → 4개 꼭짓점 좌표
         TL(x,y), TR(x+w,y), BR(x+w,y+h), BL(x,y+h)

      2. 동차 좌표계로 표현: 각 점에 z=1 추가
         [x, y] → [x, y, 1]^T

      3. H @ 꼭짓점_행렬 (3×4) → 변환된 동차 좌표 (3×4)

      4. perspective divide: x' = X/W, y' = Y/W
         (H의 3행이 [0,0,1]이 아닐 때 필요)

      5. 변환된 4점의 min/max로 AABB 계산 후 이미지 경계로 클리핑

    왜 꼭짓점 4개를 모두 변환하는가:
        직사각형에 원근 변환을 가하면 일반 사각형(Quadrilateral)이 됨.
        이 사각형의 최소 경계 직사각형(AABB)을 새 BBox로 사용해야
        변환된 객체를 완전히 포함할 수 있음.
    """
    x, y, w, h = bbox

    # 4개 꼭짓점 (x, y, 1) 형태로 구성 → (3×4) 행렬
    corners = np.array([
        [x,     y,     1.0],   # 좌상단 TL
        [x + w, y,     1.0],   # 우상단 TR
        [x + w, y + h, 1.0],   # 우하단 BR
        [x,     y + h, 1.0],   # 좌하단 BL
    ], dtype=np.float64).T     # → (3, 4) 형태

    # H @ corners: (3×3) @ (3×4) = (3×4)
    transformed = H @ corners

    # perspective divide: 동차 좌표의 3번째 값(W)으로 나누기
    # W가 0에 가까워지는 극단적 원근에서의 수치 안전성 확보
    W = transformed[2]
    W = np.where(np.abs(W) < 1e-10, 1e-10, W)  # 0 나누기 방지
    xs = transformed[0] / W
    ys = transformed[1] / W

    # AABB 계산 + 이미지 경계 클리핑
    new_x = float(np.clip(np.min(xs), 0, img_w))
    new_y = float(np.clip(np.min(ys), 0, img_h))
    new_x2 = float(np.clip(np.max(xs), 0, img_w))
    new_y2 = float(np.clip(np.max(ys), 0, img_h))

    new_w = new_x2 - new_x
    new_h = new_y2 - new_y

    return [round(int(new_x)), round(int(new_y)),
            round(int(new_w)), round(int(new_h))]


# ──────────────────────────────────────────────────────────
# [v2.0 신규] Polygon Homography 변환
# ──────────────────────────────────────────────────────────

def transform_polygon_homography(
    points: List[List[float]],
    H: np.ndarray,
    img_w: int,
    img_h: int
) -> List[List[int]]:
    """
    N개의 다각형 꼭짓점 전체에 Homography 행렬을 적용한다.

    AABB(4꼭짓점) 방식과의 차이:
      - AABB: 4개 코너만 변환 → 원근 후 사각형 외곽 박스 계산
      - Polygon: N개 점 모두 변환 → 실루엣이 그대로 유지됨
      - Instance Segmentation에서 정밀도가 훨씬 높음

    수학적 절차:
      1. N개 점을 동차 좌표계 (3, N) 행렬로 변환
         [[x1, x2, ..., xN],
          [y1, y2, ..., yN],
          [1,  1,  ...,  1 ]]

      2. H @ pts_h → (3, N) 변환된 동차 좌표

      3. Perspective Divide: x'_i = X_i / W_i, y'_i = Y_i / W_i
         (원근 나눗셈 — 평행 이동+회전만 있을 땐 W≈1이나 일반 경우 필수)

      4. 이미지 경계 (0~img_w, 0~img_h) 내로 np.clip 처리

    Args:
        points : [[x1,y1], [x2,y2], ...] 형식의 원본 이미지 좌표
        H      : 3×3 Homography 행렬 (float64)
        img_w  : 이미지 너비  (클리핑 상한)
        img_h  : 이미지 높이  (클리핑 상한)

    Returns:
        [[x1',y1'], [x2',y2'], ...] 변환 후 정수 좌표 목록
        점 수는 입력과 동일하게 유지됨 (N개)
    """
    if not points or len(points) < 1:
        return points

    pts = np.array(points, dtype=np.float64)  # (N, 2)
    N   = pts.shape[0]

    # 동차 좌표: (3, N) 행렬 구성
    # [[x1..xN], [y1..yN], [1..1]]
    ones    = np.ones((1, N), dtype=np.float64)
    pts_h   = np.vstack([pts.T, ones])          # (3, N)

    # H @ pts_h → (3, N) 변환된 동차 좌표
    transformed = H @ pts_h                     # (3, N)

    # Perspective Divide: W가 0에 가까울 때 수치 안전성 보장
    W = transformed[2]
    W = np.where(np.abs(W) < 1e-10, 1e-10, W)

    xs = transformed[0] / W   # (N,) — 변환 후 x 좌표 배열
    ys = transformed[1] / W   # (N,) — 변환 후 y 좌표 배열

    # 이미지 경계 클리핑 (0 ~ img_w/img_h)
    xs = np.clip(xs, 0, img_w)
    ys = np.clip(ys, 0, img_h)

    # (N, 2) 정수 좌표로 반환
    result = np.stack([xs, ys], axis=1)         # (N, 2)
    return [[round(int(x)), round(int(y))] for x, y in result]


def apply_poisson_noise(image: np.ndarray, lam: float) -> np.ndarray:
    """
    물리적 카메라 센서 노이즈를 모사하는 Poisson 노이즈 합성.

    원리:
      - 픽셀 값을 λ로 스케일링 (광량에 비례)
      - 스케일된 값을 기댓값으로 Poisson 샘플링
      - 다시 255 범위로 복원

    왜 Gaussian이 아닌가:
      - Gaussian 노이즈: σ² = 상수 (밝기에 무관)
      - Poisson 노이즈: σ² = μ (밝기에 비례)
        → 실제 카메라 센서에서 밝은 픽셀에 더 많은 광자가 도달하고
          그 분산도 광자 수에 비례함 (샷 노이즈, Shot noise)

    자료형 방어:
      - lam이 극단적으로 작으면 스케일된 값이 0에 수렴 → 모두 흑색
      - lam이 극단적으로 크면 Poisson → Gaussian 근사 (CLT)
      - np.clip으로 0~255 오버플로우 방지
    """
    if lam <= 0:
        raise ValueError(f"poisson_lambda는 양수여야 합니다. (입력: {lam})")

    # float64로 변환 (정밀도 확보)
    img_f = image.astype(np.float64) / 255.0

    # λ로 스케일 → 각 픽셀의 "광자 수" 기댓값
    scaled = img_f * lam

    # Poisson 샘플링 (정수값) → 분산 = 기댓값 = scaled
    noisy = np.random.poisson(scaled).astype(np.float64)

    # λ로 나눠 0~1 복원 → 255 스케일
    noisy_norm = (noisy / lam) * 255.0

    # 픽셀 값이 0~255를 벗어나지 않도록 클리핑
    return np.clip(noisy_norm, 0, 255).astype(np.uint8)


def apply_specular_flare(image: np.ndarray, intensity: float = 0.4) -> np.ndarray:
    """
    국소적 빛 반사(Specular Flare) 합성.
    랜덤 위치에 Gaussian 형태의 밝은 원형 빛을 덧씌운다.
    """
    h, w = image.shape[:2]
    cx = np.random.randint(w // 4, 3 * w // 4)
    cy = np.random.randint(h // 4, 3 * h // 4)
    radius = np.random.randint(min(w, h) // 8, min(w, h) // 3)

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    # Gaussian 형태의 밝기 마스크
    mask = np.exp(-(dist ** 2) / (2 * (radius / 2.5) ** 2))
    mask = (mask * intensity * 255.0)

    result = image.astype(np.float64)
    for c in range(3):
        result[:, :, c] += mask
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_shadow(image: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """
    그라데이션 그림자 합성.
    한쪽에서 다른 쪽으로 점진적으로 어두워지는 조명 효과.
    """
    h, w = image.shape[:2]
    direction = np.random.randint(0, 4)

    if direction == 0:
        grad = np.linspace(1.0, 1.0 - intensity, w)
        mask = np.tile(grad, (h, 1))
    elif direction == 1:
        grad = np.linspace(1.0 - intensity, 1.0, w)
        mask = np.tile(grad, (h, 1))
    elif direction == 2:
        grad = np.linspace(1.0, 1.0 - intensity, h)
        mask = np.tile(grad.reshape(-1, 1), (1, w))
    else:
        grad = np.linspace(1.0 - intensity, 1.0, h)
        mask = np.tile(grad.reshape(-1, 1), (1, w))

    result = image.astype(np.float64)
    for c in range(3):
        result[:, :, c] *= mask
    return np.clip(result, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════
# v1.1 신규 증강 함수
# ══════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────
# [신규] Gamma Correction & Histogram Equalization (CLAHE)
# ──────────────────────────────────────────────────────────

def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """
    감마 보정(Gamma Correction): V_out = 255 × (V_in / 255)^γ

    성능 최적화 — cv2.LUT(Look-Up Table) 활용:
      - 브루트 포스: 이미지 전체 픽셀(H×W×3)에 지수 연산 → O(HW) 비용
      - LUT 방식: 0~255 정수 256개에만 연산 → O(256) 선계산 후 O(HW) 테이블 참조
      - 결과: 수백만 픽셀 이미지에서 ~10× 속도 향상

    감마 값의 물리적 의미:
      γ < 1 (예: 0.5): 밝기 증가 (어두운 영역 디테일 복원, 야간 이미지)
      γ > 1 (예: 2.0): 밝기 감소 (과노출 이미지 보정)
      γ = 1           : 원본 유지 (항등 변환)

    Args:
        image: BGR 이미지 (H, W, 3) uint8
        gamma: 감마 값 (권장 범위: 0.4 ~ 2.5)

    Returns:
        감마 보정된 이미지 (H, W, 3) uint8
    """
    if gamma <= 0:
        raise ValueError(f"gamma는 양수여야 합니다. (입력: {gamma})")

    # 0~255 정수 256개에 대한 변환 테이블 사전 계산
    # np.arange(256): [0, 1, 2, ..., 255]
    # / 255.0: 정규화 → [0.0, ..., 1.0]
    # ** gamma: 거듭제곱 (핵심 감마 수식)
    # * 255.0: 역정규화 → [0.0, ..., 255.0]
    # np.clip → np.uint8: 0~255 정수로 안전 변환
    lut = np.array(
        [np.clip(255.0 * (i / 255.0) ** gamma, 0, 255) for i in range(256)],
        dtype=np.uint8
    )

    # cv2.LUT: 각 픽셀값을 테이블 인덱스로 사용해 즉시 변환 (벡터화)
    return cv2.LUT(image, lut)


def apply_histogram_equalization(
    image: np.ndarray,
    use_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_size: int = 8
) -> np.ndarray:
    """
    히스토그램 평탄화(Histogram Equalization) — YCrCb 색공간 활용.

    왜 BGR 직접 처리가 아닌 YCrCb인가:
      - BGR 각 채널을 독립적으로 equalizeHist하면 채널 간 비율이 깨져 색조 왜곡 발생
      - YCrCb는 밝기(Y)와 색차(Cr, Cb)를 분리한 색공간
        → Y 채널만 처리하면 색상(색조·채도) 완전 보존, 밝기 분포만 균등화

    일반 equalizeHist vs CLAHE(Contrast Limited AHE):
      equalizeHist : 전역 히스토그램 → 노이즈 과대 증폭 위험
      CLAHE        : 타일(tile_size×tile_size) 단위 지역 처리 + clipLimit로 증폭 제한
                     → 실제 영상 처리에서 CLAHE가 훨씬 자연스러운 결과

    Args:
        image      : BGR 이미지 (H, W, 3) uint8
        use_clahe  : True=CLAHE, False=일반 equalizeHist
        clip_limit : CLAHE 명암비 증폭 한계 (1.0~5.0 권장, 클수록 강함)
        tile_size  : CLAHE 타일 크기 (픽셀 단위, 8 또는 16 권장)

    Returns:
        히스토그램 평탄화된 BGR 이미지 (H, W, 3) uint8
    """
    # BGR → YCrCb 변환 (밝기/색차 분리)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Y(밝기) 채널만 추출
    y, cr, cb = cv2.split(ycrcb)

    if use_clahe:
        # CLAHE: 타일별 지역 히스토그램 균등화
        # clipLimit: 히스토그램 빈(bin) 최대 높이 제한 → 노이즈 과증폭 방지
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=(tile_size, tile_size)
        )
        y_eq = clahe.apply(y)
    else:
        # 전역 히스토그램 균등화 (빠르지만 노이즈 취약)
        y_eq = cv2.equalizeHist(y)

    # 균등화된 Y 채널로 교체 후 BGR로 복원
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


# ──────────────────────────────────────────────────────────
# [신규] Cutout (Random Erasing)
# ──────────────────────────────────────────────────────────

def apply_cutout(
    image: np.ndarray,
    annotations: List[dict],
    area_ratio_min: float = 0.02,
    area_ratio_max: float = 0.10,
    fill_mode: str = 'mean',
    drop_covered: bool = False,
    drop_threshold: float = 0.5
) -> Tuple[np.ndarray, List[dict]]:
    """
    Cutout (Random Erasing): 이미지의 임의 영역을 마스킹하여 오클루전 모사.

    수학적 처리:
      1. 전체 이미지 면적(S = H × W) 기준으로 마스크 면적 S_cut 무작위 결정
         S_cut = S × Uniform(area_ratio_min, area_ratio_max)
      2. 종횡비(aspect ratio) r = Uniform(0.3, 3.0) 무작위 결정
         cut_h = sqrt(S_cut / r), cut_w = sqrt(S_cut × r)
      3. 마스크 좌상단 좌표 (rx, ry) 무작위 샘플링 → 경계 클리핑
      4. 해당 영역 픽셀값을 fill_mode에 따라 채움

    BBox 처리 전략:
      공간적 변환(이동/회전)이 없으므로 기본적으로 BBox 좌표는 유지.
      drop_covered=True 시: Cutout이 BBox 면적의 drop_threshold 이상을 가리면 제거.
      예) drop_threshold=0.5 → BBox의 50% 이상이 가려지면 해당 BBox 드롭.
      (100% 기준은 현실적으로 거의 발동하지 않아 의미 없음)

    fill_mode:
      'black': 마스크 영역을 (0, 0, 0)으로 채움
      'mean' : 마스크 영역을 이미지 전체 채널별 평균값으로 채움
               → 배경이 다양한 경우 좀 더 자연스러운 결과

    Args:
        image           : BGR 이미지 (H, W, 3) uint8
        annotations     : BBox 어노테이션 리스트 [{'x','y','w','h','label',...}]
        area_ratio_min  : Cutout 면적 최솟값 (전체 면적 대비 비율, 예: 0.02 = 2%)
        area_ratio_max  : Cutout 면적 최댓값 (예: 0.10 = 10%)
        fill_mode       : 채움 방식 ('black' 또는 'mean')
        drop_covered    : True이면 임계값 이상 가려진 BBox 드롭
        drop_threshold  : BBox 면적 대비 가려진 비율 임계값 (기본 0.5 = 50%)

    Returns:
        (수정된 이미지, 처리된 어노테이션 리스트)
    """
    img_h, img_w = image.shape[:2]
    total_area = img_h * img_w

    # 마스크 면적 결정 (전체 면적의 area_ratio_min ~ area_ratio_max)
    cut_area = total_area * np.random.uniform(area_ratio_min, area_ratio_max)

    # 종횡비 무작위 결정 (0.3 ~ 3.0 범위로 너무 가늘거나 넓은 박스 방지)
    aspect_ratio = np.random.uniform(0.3, 3.0)

    # S_cut = cut_h × cut_w, r = cut_w / cut_h
    # → cut_h = sqrt(S_cut / r), cut_w = sqrt(S_cut × r)
    cut_h = int(np.sqrt(cut_area / aspect_ratio))
    cut_w = int(np.sqrt(cut_area * aspect_ratio))

    # 이미지 범위 내로 클리핑
    cut_h = min(cut_h, img_h)
    cut_w = min(cut_w, img_w)

    # 마스크 좌상단 좌표 무작위 결정
    rx = np.random.randint(0, max(1, img_w - cut_w))
    ry = np.random.randint(0, max(1, img_h - cut_h))

    # 마스크 우하단 좌표 (이미지 경계 클리핑)
    rx2 = min(rx + cut_w, img_w)
    ry2 = min(ry + cut_h, img_h)

    # 채움값 결정
    if fill_mode == 'mean':
        # 채널별 평균 픽셀값 계산 (float → uint8)
        fill_val = image.mean(axis=(0, 1)).astype(np.uint8)  # shape: (3,)
    else:
        fill_val = np.array([0, 0, 0], dtype=np.uint8)

    # NumPy 슬라이싱으로 마스크 영역 채우기 (인플레이스 아닌 복사본 수정)
    result = image.copy()
    result[ry:ry2, rx:rx2] = fill_val  # 브로드캐스팅: (cut_h, cut_w, 3) = (3,)

    # BBox 처리
    aug_anns = [dict(a) for a in annotations]  # 깊은 복사

    if drop_covered:
        kept = []
        for ann in aug_anns:
            if ann.get('type') == 'polygon':
                kept.append(ann)  # polygon은 면적 계산 복잡 → 유지
                continue

            bx, by, bw, bh = ann['x'], ann['y'], ann['w'], ann['h']
            bbox_area = bw * bh
            if bbox_area <= 0:
                kept.append(ann)
                continue

            # Cutout과 BBox의 교차 영역(intersection) 계산
            ix1 = max(rx,  bx)
            iy1 = max(ry,  by)
            ix2 = min(rx2, bx + bw)
            iy2 = min(ry2, by + bh)

            if ix2 <= ix1 or iy2 <= iy1:
                # 겹침 없음 → 무조건 유지
                kept.append(ann)
            else:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                coverage = inter_area / bbox_area  # 0.0 ~ 1.0
                # coverage < drop_threshold 이면 아직 절반 이상 보이므로 유지
                if coverage < drop_threshold:
                    kept.append(ann)
                # coverage >= drop_threshold 이면 너무 많이 가려졌으므로 드롭
        aug_anns = kept

    return result, aug_anns


# ──────────────────────────────────────────────────────────
# [신규] Mosaic Augmentation
# ──────────────────────────────────────────────────────────

def _load_image_safe(img_path: str) -> Optional[np.ndarray]:
    """유니코드/한글 경로를 안전하게 로드 (None이면 실패)."""
    try:
        arr = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def apply_mosaic(
    primary_image: np.ndarray,
    primary_annotations: List[dict],
    image_pool: List[Tuple[np.ndarray, List[dict]]],
    output_w: int,
    output_h: int
) -> Tuple[np.ndarray, List[dict]]:
    """
    Mosaic Augmentation: 4개 이미지를 하나의 캔버스에 사분면으로 배치.

    YOLOv4에서 처음 도입된 기법. 4가지 장면을 하나에 합성해
    다양한 컨텍스트와 소형 객체 탐지 성능을 동시에 향상시킨다.

    수학적 변환 (각 사분면 i에 대해):
      1. 교차점 (cx, cy) 무작위 결정
         cx = Uniform(output_w * 0.3, output_w * 0.7)
         cy = Uniform(output_h * 0.3, output_h * 0.7)

      2. 4개 사분면 정의:
         Q0(TL): canvas[0:cy,   0:cx  ]  ← primary_image
         Q1(TR): canvas[0:cy,   cx:W  ]
         Q2(BL): canvas[cy:H,   0:cx  ]
         Q3(BR): canvas[cy:H,   cx:W  ]

      3. 각 사분면에 대해:
         quad_w, quad_h = 사분면 크기
         scale_x = quad_w / src_w  (이미지를 사분면에 맞춰 리사이즈)
         scale_y = quad_h / src_h

         BBox 변환:
           new_x = bbox.x * scale_x + x_offset
           new_y = bbox.y * scale_y + y_offset
           new_w = bbox.w * scale_x
           new_h = bbox.h * scale_y

         AABB 클리핑 (캔버스 경계 초과 방지):
           new_x  = clip(new_x,       0, output_w)
           new_y  = clip(new_y,       0, output_h)
           new_x2 = clip(new_x+new_w, 0, output_w)
           new_y2 = clip(new_y+new_h, 0, output_h)
           → new_w = new_x2 - new_x, new_h = new_y2 - new_y
           → 유효 박스(w>1, h>1)만 유지

    Args:
        primary_image       : 기준 이미지 (Q0에 배치)
        primary_annotations : 기준 이미지의 BBox 리스트
        image_pool          : [(이미지, 어노테이션), ...] 최소 3개 필요
                              부족하면 primary_image로 패딩
        output_w, output_h  : 출력 캔버스 크기 (원본 해상도 권장)

    Returns:
        (모자이크 이미지 (output_h, output_w, 3), 전체 BBox 리스트)
    """
    # ── 4개 이미지 구성 (부족하면 primary로 패딩) ──
    all_images = [(primary_image, primary_annotations)]
    for img, anns in image_pool:
        all_images.append((img, anns))
        if len(all_images) == 4:
            break
    # 여전히 4개 미만이면 primary 반복 패딩
    while len(all_images) < 4:
        all_images.append((primary_image, primary_annotations))

    # ── 교차점 (cx, cy) 무작위 결정 (30%~70% 범위 내) ──
    cx = int(np.random.uniform(output_w * 0.3, output_w * 0.7))
    cy = int(np.random.uniform(output_h * 0.3, output_h * 0.7))

    # ── 출력 캔버스 생성 (검은색 배경) ──
    canvas = np.zeros((output_h, output_w, 3), dtype=np.uint8)

    # 사분면 정의: (x_start, y_start, x_end, y_end)
    quads = [
        (0,  0,  cx,       cy,       0,  0),   # Q0 TL: offset=(0, 0)
        (cx, 0,  output_w, cy,       cx, 0),   # Q1 TR: offset=(cx, 0)
        (0,  cy, cx,       output_h, 0,  cy),  # Q2 BL: offset=(0, cy)
        (cx, cy, output_w, output_h, cx, cy),  # Q3 BR: offset=(cx, cy)
    ]

    all_annotations: List[dict] = []

    for i, (x_s, y_s, x_e, y_e, x_off, y_off) in enumerate(quads):
        src_img, src_anns = all_images[i]

        quad_w = x_e - x_s  # 사분면 너비
        quad_h = y_e - y_s  # 사분면 높이

        if quad_w <= 0 or quad_h <= 0:
            continue  # 교차점이 경계에 붙은 극단적 경우 스킵

        src_h, src_w = src_img.shape[:2]

        # ── 이미지를 사분면 크기로 리사이즈 ──
        # INTER_LINEAR: 빠르고 품질 좋은 양선형 보간
        resized = cv2.resize(
            src_img, (quad_w, quad_h),
            interpolation=cv2.INTER_LINEAR
        )

        # ── 캔버스에 배치 ──
        canvas[y_s:y_e, x_s:x_e] = resized

        # ── BBox 좌표 변환 ──
        # scale_x = quad_w / src_w: 리사이즈 스케일 비율
        # x_offset = x_off: 사분면의 캔버스 상 x 시작점
        scale_x = quad_w / max(src_w, 1)
        scale_y = quad_h / max(src_h, 1)

        for ann in src_anns:
            bx = ann.get('x', 0) * scale_x + x_off
            by = ann.get('y', 0) * scale_y + y_off
            bw = ann.get('w', 0) * scale_x
            bh = ann.get('h', 0) * scale_y

            # AABB 클리핑 (캔버스 경계를 벗어난 부분 잘라냄)
            bx2 = bx + bw
            by2 = by + bh

            bx  = float(np.clip(bx,  0, output_w))
            by  = float(np.clip(by,  0, output_h))
            bx2 = float(np.clip(bx2, 0, output_w))
            by2 = float(np.clip(by2, 0, output_h))

            final_w = bx2 - bx
            final_h = by2 - by

            # 클리핑 후 유효한 박스만 유지 (1px 이상)
            if final_w >= 1.0 and final_h >= 1.0:
                new_ann = dict(ann)
                new_ann['x'] = round(int(bx))
                new_ann['y'] = round(int(by))
                new_ann['w'] = round(int(final_w))
                new_ann['h'] = round(int(final_h))
                new_ann['mosaic_source'] = i  # 디버깅용: 몇 번째 사분면 출처
                all_annotations.append(new_ann)

    return canvas, all_annotations


# ──────────────────────────────────────────────────────────
# [v2.0 신규] Copy-Paste Augmentation
# ──────────────────────────────────────────────────────────

def apply_copy_paste(
    primary_image: np.ndarray,
    primary_annotations: List[dict],
    background_pool: List[np.ndarray],
    num_pastes: int = 1
) -> Tuple[np.ndarray, List[dict]]:
    """
    Polygon 라벨에서 정밀 마스크를 생성하여 배경 이미지에 객체를 합성한다.

    핵심 아이디어:
      - 단순 bbox crop은 배경 픽셀을 함께 붙여 넣어 "배경 오염" 발생
      - Polygon 마스크를 사용하면 객체 실루엣만 정확히 추출하여
        배경 이미지에 자연스럽게 합성 가능

    단계별 로직:
      1. type='polygon'인 라벨에서 cv2.fillPoly로 2D 이진 마스크 생성
      2. Bitwise AND로 배경=0, 객체=원본 픽셀인 "누끼 이미지" 추출
      3. 배경 풀(pool)에서 무작위 배경 선택 및 크기 맞춤
      4. 임의의 위치 (dx, dy)에 붙여넣기 (마스크 영역만)
      5. Polygon 좌표를 dx, dy만큼 이동 → 라벨 동기화

    Args:
        primary_image       : 원본 이미지 (H, W, 3) BGR
        primary_annotations : 라벨 목록 (type='polygon' 포함)
        background_pool     : 배경으로 사용할 이미지 numpy array 목록
        num_pastes          : 1회 호출당 붙여넣기 횟수 (기본 1)

    Returns:
        (합성된 이미지, 업데이트된 라벨 목록)
        - polygon 라벨만 이동; bbox 라벨은 그대로 유지
        - 붙여넣기로 이미지 밖으로 나간 polygon 좌표는 클리핑
        - polygon이 없거나 배경 풀이 비어있으면 원본 그대로 반환
    """
    # ── 전제 조건 검사 ──
    polygon_anns = [a for a in primary_annotations if a.get('type') == 'polygon']
    if not polygon_anns or not background_pool:
        return primary_image, primary_annotations  # 조건 불충족 시 원본 반환

    img_h, img_w = primary_image.shape[:2]

    # ── 배경 이미지 선택 및 원본 크기에 맞게 리사이즈 ──
    bg_raw = background_pool[np.random.randint(len(background_pool))]
    background = cv2.resize(bg_raw, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

    result_image = background.copy()
    result_annotations = [dict(a) for a in primary_annotations]

    # ── 각 paste 횟수만큼 반복 ──
    for _ in range(num_pastes):
        # 이번 pass에 사용할 polygon 라벨 무작위 선택
        src_ann = polygon_anns[np.random.randint(len(polygon_anns))]
        pts_raw = src_ann.get('points', [])

        if len(pts_raw) < 3:
            continue

        # ── 1. Polygon 마스크 생성 (cv2.fillPoly) ──
        # points: [[x1,y1], ...] → cv2 요구 형식: (N,1,2) int32
        poly_pts = np.array(pts_raw, dtype=np.int32).reshape((-1, 1, 2))

        mask = np.zeros((img_h, img_w), dtype=np.uint8)  # 2D 이진 마스크
        cv2.fillPoly(mask, [poly_pts], 255)               # 내부: 255, 외부: 0

        # ── 2. Bitwise AND: 배경=0, 객체=원본 픽셀 ("누끼") ──
        # mask를 3채널로 확장하여 BGR 이미지와 연산
        mask_3ch  = cv2.merge([mask, mask, mask])         # (H, W, 3)
        obj_only  = cv2.bitwise_and(primary_image, mask_3ch)

        # 객체 Bounding Rect 계산 (최소 외접 사각형 → crop 범위)
        obj_rect  = cv2.boundingRect(poly_pts)            # (x, y, w, h)
        ox, oy, ow, oh = obj_rect
        if ow <= 0 or oh <= 0:
            continue

        # 객체 영역 크롭
        obj_crop  = obj_only[oy:oy+oh, ox:ox+ow]         # (oh, ow, 3)
        mask_crop = mask[oy:oy+oh, ox:ox+ow]              # (oh, ow)

        # ── 3. 붙여넣기 위치 (dx, dy) 무작위 결정 ──
        # 객체가 완전히 이미지 내에 들어오도록 범위 제한
        max_dx = max(0, img_w - ow)
        max_dy = max(0, img_h - oh)
        dx = int(np.random.randint(0, max_dx + 1))
        dy = int(np.random.randint(0, max_dy + 1))

        # 오프셋 (원본 객체 좌측상단 기준 → 새 위치 기준)
        # polygon 점들의 실제 이동량 = (dx - ox, dy - oy)
        shift_x = dx - ox
        shift_y = dy - oy

        # ── 4. 배경 이미지에 마스크 합성 ──
        # 마스크 영역만 덮어쓰기: result = bg * (1 - mask) + obj * mask
        dst_region = result_image[dy:dy+oh, dx:dx+ow]
        mask_inv   = cv2.bitwise_not(mask_crop)

        bg_part  = cv2.bitwise_and(dst_region, dst_region, mask=mask_inv)
        obj_part = cv2.bitwise_and(obj_crop,   obj_crop,   mask=mask_crop)
        result_image[dy:dy+oh, dx:dx+ow] = cv2.add(bg_part, obj_part)

        # ── 5. Polygon 좌표 이동 (라벨 동기화) ──
        # result_annotations 내 해당 ann을 찾아 points 업데이트
        for r_ann in result_annotations:
            if r_ann is not src_ann and r_ann.get('points') != src_ann.get('points'):
                continue
            if r_ann.get('type') != 'polygon':
                continue

            shifted = []
            for px, py in r_ann['points']:
                # 평행 이동 + 이미지 경계 클리핑
                nx = int(np.clip(px + shift_x, 0, img_w))
                ny = int(np.clip(py + shift_y, 0, img_h))
                shifted.append([nx, ny])
            r_ann['points'] = shifted
            break  # 첫 번째 매칭 ann만 처리

    return result_image, result_annotations


# ──────────────────────────────────────────────────────────
# 핵심 클래스: OnTheFlyAugmenter
# ──────────────────────────────────────────────────────────

class OnTheFlyAugmenter:
    """
    JSON 레시피를 읽어 이미지와 BBox를 실시간 증강하는 클래스.
    PyTorch의 Dataset.__getitem__ 인터페이스를 따른다.

    사용 예:
        augmenter = OnTheFlyAugmenter(recipe_dir='project_data/labels')
        sample = augmenter[0]
        image  = sample['image']         # numpy array (H, W, 3)
        bboxes = sample['annotations']   # 증강 후 AABB 좌표 목록

    PyTorch DataLoader 연동:
        from torch.utils.data import DataLoader
        loader = DataLoader(
            augmenter,
            batch_size=4,
            collate_fn=augmenter.collate_fn
        )
        for batch in loader:
            images     = batch['images']      # (N, H, W, 3)
            ann_lists  = batch['annotations'] # List[List[dict]]
    """

    def __init__(
        self,
        recipe_dir: str,
        original_img_root: str = '',
        image_size: Optional[Tuple[int, int]] = None
    ):
        """
        Args:
            recipe_dir      : JSON 레시피 파일들이 있는 디렉토리
            original_img_root: 원본 이미지 루트 경로
                              (레시피의 source_image가 상대경로일 때 prefix)
            image_size      : None이면 원본 크기 유지.
                              (width, height) 지정 시 리사이즈.
        """
        self.recipe_dir       = Path(recipe_dir)
        self.original_img_root = Path(original_img_root) if original_img_root else None
        self.image_size       = image_size

        # *_recipe_*.json 파일만 수집 (라벨 JSON과 구분)
        self.recipes = self._load_recipes()

    def _load_recipes(self) -> List[dict]:
        """레시피 JSON 파일들을 파싱하여 메모리에 적재한다."""
        recipes = []
        pattern = '*_recipe_*.json'

        for recipe_path in sorted(self.recipe_dir.glob(pattern)):
            try:
                with open(str(recipe_path), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['_recipe_path'] = str(recipe_path)
                recipes.append(data)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[경고] 레시피 파일 파싱 실패: {recipe_path} → {e}")

        return recipes

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int) -> dict:
        """
        idx번째 레시피를 읽어 원본 이미지에 증강을 적용하고
        (image, annotations) 쌍을 반환한다.

        Returns:
            {
                'image'       : np.ndarray (H, W, 3) uint8,
                'annotations' : [{'label':..., 'x':..., 'y':...,
                                  'w':..., 'h':...}, ...],
                'image_id'    : str,
                'recipe_id'   : str
            }
        """
        recipe = self.recipes[idx]
        params = recipe.get('augmentation_params', {})
        anns   = recipe.get('original_annotations', [])

        # ── 원본 이미지 로드 ──
        source_img_name = recipe['source_image']
        if self.original_img_root:
            img_path = self.original_img_root / source_img_name
        else:
            img_path = Path(source_img_name)

        # 한글/유니코드 경로 안전 로드
        img_array = np.fromfile(str(img_path), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {img_path}")

        img_h, img_w = image.shape[:2]
        aug_anns = [dict(a) for a in anns]  # 깊은 복사 (원본 수정 방지)

        # ── Homography 변환 + 좌표 동기화 (v2.0: bbox / polygon 분기) ──
        if 'h31' in params and 'h32' in params and 'theta_deg' in params:
            h31       = float(params['h31'])
            h32       = float(params['h32'])
            theta_deg = float(params['theta_deg'])

            image, H = apply_homography(image, h31, h32, theta_deg)

            for ann in aug_anns:
                if ann.get('type') == 'polygon' and ann.get('points'):
                    # v2.0: N개 polygon 꼭짓점 전체 변환
                    ann['points'] = transform_polygon_homography(
                        ann['points'], H, img_w, img_h
                    )
                else:
                    # 기존 AABB 변환
                    new_bbox = transform_bbox_homography(
                        [ann['x'], ann['y'], ann['w'], ann['h']],
                        H, img_w, img_h
                    )
                    ann['x'], ann['y'], ann['w'], ann['h'] = new_bbox

        # ── Poisson 노이즈 합성 ──
        if 'poisson_lambda' in params:
            lam = float(params['poisson_lambda'])
            # λ 범위 방어: 0 이하이면 노이즈 적용 건너뜀
            if lam > 0:
                image = apply_poisson_noise(image, lam)

        # ── 조명 증강 (선택) ──
        if params.get('specular_flare', False):
            image = apply_specular_flare(image)

        if params.get('shadow', False):
            image = apply_shadow(image)

        # ── 리사이즈 (선택) ──
        if self.image_size is not None:
            target_w, target_h = self.image_size
            scale_x = target_w / img_w
            scale_y = target_h / img_h

            image = cv2.resize(image, (target_w, target_h),
                               interpolation=cv2.INTER_LINEAR)

            # 리사이즈에 따른 BBox 좌표 스케일링
            for ann in aug_anns:
                ann['x'] = round(int(ann['x'] * scale_x))
                ann['y'] = round(int(ann['y'] * scale_y))
                ann['w'] = round(int(ann['w'] * scale_x))
                ann['h'] = round(int(ann['h'] * scale_y))

        return {
            'image'      : image,   # BGR numpy array (cv2 기본 형식)
            'annotations': aug_anns,
            'image_id'   : recipe.get('image_id', ''),
            'recipe_id'  : recipe.get('recipe_id', '')
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """
        PyTorch DataLoader의 collate_fn.
        다양한 크기의 이미지를 배치로 묶을 때 사용.

        이미지 크기가 모두 같을 때만 numpy stack이 가능하므로,
        image_size를 고정하거나 자체 패딩 로직을 추가해야 함.
        """
        return {
            'images'     : np.stack([b['image'] for b in batch]),
            'annotations': [b['annotations'] for b in batch],
            'image_ids'  : [b['image_id']     for b in batch],
            'recipe_ids' : [b['recipe_id']    for b in batch]
        }

    def visualize_sample(
        self,
        idx: int,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        idx번째 샘플에 증강을 적용하고 BBox를 시각화한다.
        디버깅 및 수학적 정확성 검수용.

        사용법:
            aug.visualize_sample(0, save_path='debug/check_000.jpg')
        """
        sample = self[idx]
        vis    = sample['image'].copy()

        for ann in sample['annotations']:
            x, y, w, h = ann['x'], ann['y'], ann['w'], ann['h']
            label       = ann.get('label', '?')

            # 박스 그리기
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 라벨 배경
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(vis, (x, y - th - 6), (x + tw + 4, y), (0, 255, 0), -1)
            cv2.putText(
                vis, label, (x + 2, y - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            # 유니코드 경로 안전 저장
            ok, buf = cv2.imencode('.jpg', vis)
            if ok:
                buf.tofile(save_path)

        return vis


# ──────────────────────────────────────────────────────────
# 단독 테스트
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("sample_augmentation_params() 테스트")
    print("=" * 60)

    params = sample_augmentation_params(n=3, seed=42, label_sensitive=False)
    for i, p in enumerate(params):
        print(f"  [{i}] {p}")

    print()
    print("label_sensitive=True (회전 축소):")
    params_s = sample_augmentation_params(n=3, seed=42, label_sensitive=True)
    for i, p in enumerate(params_s):
        theta = p.get('theta_deg', 'N/A')
        print(f"  [{i}] theta_deg={theta}")

    print("=" * 60)
    print("BBox Homography 변환 테스트")
    print("=" * 60)
    IMG_W, IMG_H = 640, 480
    bbox = [100, 100, 200, 150]  # x, y, w, h
    H = build_homography_matrix(0.0005, -0.0003, 3.0, IMG_W, IMG_H)
    new_bbox = transform_bbox_homography(bbox, H, IMG_W, IMG_H)
    print(f"  원본 BBox  : {bbox}")
    print(f"  변환 BBox  : {new_bbox}")
    area_orig = bbox[2] * bbox[3]
    area_new  = new_bbox[2] * new_bbox[3]
    print(f"  넓이 비율   : {area_new/area_orig:.3f} (1에 가까울수록 정확)")
    print("=" * 60)
