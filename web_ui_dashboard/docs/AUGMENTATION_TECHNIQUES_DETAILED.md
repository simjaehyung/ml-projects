# 🎬 Data Augmentation 기법 상세 분석

## 📋 목록

### v1.0 기본 증강
1. **Homography** (원근 + 회전)
2. **Poisson Noise** (물리적 카메라 노이즈)
3. **Specular Flare** (빛 반사)
4. **Gradient Shadow** (그라데이션 그림자)

### v1.1 고급 증강
5. **Gamma Correction** (비선형 밝기 조정)
6. **CLAHE** (지역 히스토그램 평탄화)
7. **Cutout** (오클루전 시뮬레이션)
8. **Mosaic** (4-이미지 합성)

### v2.0 신규 증강
9. **Polygon Homography** (다각형 기반 변환)
10. **Copy-Paste** (객체 누끼 합성)

---

## 1️⃣ Homography (원근 + 회전)

### 목적
3D 공간에서 카메라 시점 변화를 모사하여 원근감과 회전을 동시에 적용.

### 코드 위치
```
augmentation_recipe.py: apply_homography() [Line 169]
                        transform_bbox_homography() [Line 192]
```

### 작동 원리

#### 단계 1: 3×3 Homography 행렬 구성
```python
# 파라미터
h31: float       # X축 원근 기울기 (-0.001 ~ +0.001)
h32: float       # Y축 원근 기울기 (-0.001 ~ +0.001)
theta_deg: float # 중심 회전각 (-5° ~ +5°)

# 결과: 3×3 변환 행렬 H
# 이 행렬이 [x, y, 1]^T에 곱해져 새로운 이미지 좌표 생성
```

#### 단계 2: cv2.warpPerspective로 이미지 변환
```python
warped = cv2.warpPerspective(
    image, H, (w, h),
    borderMode=cv2.BORDER_REFLECT_101  # 경계 반사로 검은 여백 방지
)
```

#### 단계 3: BBox 좌표 동기화 (핵심!)
```
원본 BBox (직사각형, 4개 꼭짓점)
  ↓ [동차 좌표계 변환]
  각 점을 [x, y, 1] 형태로 변환
  ↓ [3×3 행렬 곱셈]
  H @ [x, y, 1]^T = [x', y', w']^T
  ↓ [Perspective Divide — 원근 나눗셈]
  x_final = x' / w'
  y_final = y' / w'
  ↓ [4개 변환된 점의 min/max]
  새로운 Axis-Aligned BBox (AABB) 계산
  ↓ [클리핑]
  0 ≤ x, y ≤ img_w, img_h 범위로 자르기
```

### 왜 4개 점을 모두 변환하는가?
직사각형에 원근 변환을 가하면 **일반 사각형(Quadrilateral)**이 됨.
이 사각형을 감싸는 최소 직사각형이 새로운 BBox.

### 코드 예시
```python
# 4개 꼭짓점을 (3×4) 행렬로 구성
corners = np.array([
    [x,     y,     1.0],    # 좌상단 TL
    [x + w, y,     1.0],    # 우상단 TR
    [x + w, y + h, 1.0],    # 우하단 BR
    [x,     y + h, 1.0],    # 좌하단 BL
], dtype=np.float64).T  # → (3, 4)

# H @ corners: (3×3) @ (3×4) = (3×4)
transformed = H @ corners

# Perspective Divide (0 나누기 방지)
W = transformed[2]
W = np.where(np.abs(W) < 1e-10, 1e-10, W)
xs = transformed[0] / W
ys = transformed[1] / W

# AABB 계산
new_x = np.clip(np.min(xs), 0, img_w)
new_y = np.clip(np.min(ys), 0, img_h)
new_x2 = np.clip(np.max(xs), 0, img_w)
new_y2 = np.clip(np.max(ys), 0, img_h)
```

### 수치 안전성
- **0 나누기 방지**: `W < 1e-10`일 때 `1e-10`으로 치환
- **오버플로우 방지**: `np.clip(0, img_w/img_h)` 경계 처리

---

## 2️⃣ Poisson Noise (물리적 카메라 노이즈)

### 목적
카메라 센서의 실제 노이즈 특성(샷 노이즈, Shot Noise) 모사.

### 코드 위치
```
augmentation_recipe.py: apply_poisson_noise() [Line 325]
```

### 왜 Gaussian이 아닌가?

| 특성 | Gaussian | Poisson |
|------|----------|---------|
| **분산** | σ² = 상수 | σ² = μ (기댓값에 비례) |
| **밝기 관계** | 밝기와 무관 | 밝을수록 노이즈 증가 |
| **물리** | 비물리적 | 실제 카메라 센서 |

### 작동 원리

#### 단계 1: 정규화
```python
img_f = image.astype(np.float64) / 255.0  # [0, 1] 범위
```

#### 단계 2: λ로 스케일 (광자 수 모사)
```python
# 각 픽셀의 "광자 수" 기댓값 = pixel_value × λ
# λ 범위: 20 ~ 40
scaled = img_f * lam
```

#### 단계 3: Poisson 샘플링
```python
# NumPy Poisson: 기댓값=scaled, 분산=scaled인 정수 표본
noisy = np.random.poisson(scaled).astype(np.float64)
```

#### 단계 4: 역정규화
```python
# λ로 나눠 0~1로 복원 → 255 스케일
noisy_norm = (noisy / lam) * 255.0

# 오버플로우 방지
result = np.clip(noisy_norm, 0, 255).astype(np.uint8)
```

### 수식
```
V_out = Poisson(V_in × λ) / λ × 255
     = Poisson(V_in × λ) / λ × 255

특징: 밝은 픽셀 → 분산 커짐
      어두운 픽셀 → 분산 작음 (거의 검정 유지)
```

### 에러 처리
```python
if lam <= 0:
    raise ValueError("poisson_lambda는 양수여야 합니다")
```

---

## 3️⃣ Specular Flare (빛 반사)

### 목적
렌즈에 반사되는 강한 빛 효과 추가 (선택적).

### 코드 위치
```
augmentation_recipe.py: apply_specular_flare() [Line 364]
```

### 작동 원리

#### 단계 1: 무작위 위치 결정
```python
# 이미지의 중앙 영역에서만 배치 (예쁜 효과)
cx = np.random.randint(w // 4, 3 * w // 4)
cy = np.random.randint(h // 4, 3 * h // 4)
radius = np.random.randint(min(w, h) // 8, min(w, h) // 3)
```

#### 단계 2: Gaussian 형 밝기 마스크 생성
```python
# 거리 계산 (중심에서의 유클리드 거리)
Y, X = np.ogrid[:h, :w]
dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

# Gaussian 함수: 중심이 밝고 가장자리로 갈수록 어두워짐
mask = np.exp(-(dist ** 2) / (2 * (radius / 2.5) ** 2))
mask = mask * intensity * 255.0  # intensity: 0~1 범위
```

#### 단계 3: 원본에 덧칠하기
```python
result = image.astype(np.float64)
for c in range(3):  # B, G, R 채널
    result[:, :, c] += mask  # 밝기 증가

result = np.clip(result, 0, 255).astype(np.uint8)
```

### 수식
```
V_out = V_in + mask
mask = intensity × 255 × exp(-(dist²) / (2σ²))
σ = radius / 2.5
```

---

## 4️⃣ Gradient Shadow (그라데이션 그림자)

### 목적
한쪽에서 다른 쪽으로 점진적으로 어두워지는 조명 효과.

### 코드 위치
```
augmentation_recipe.py: apply_shadow() [Line 386]
```

### 작동 원리

#### 단계 1: 방향 무작위 결정 (4가지)
```python
direction = np.random.randint(0, 4)
# 0: 좌→우 (왼쪽이 밝음)
# 1: 우→좌 (오른쪽이 밝음)
# 2: 상→하 (위가 밝음)
# 3: 하→상 (아래가 밝음)
```

#### 단계 2: 선형 그라데이션 마스크 생성
```python
# 예: 좌→우 방향 (direction=0)
grad = np.linspace(1.0, 1.0 - intensity, w)
# [1.0, ..., 0.5] (intensity=0.5일 경우)

# 행 방향으로 반복 확장
mask = np.tile(grad, (h, 1))  # shape: (h, w)
```

#### 단계 3: 곱셈으로 적용
```python
result = image.astype(np.float64)
for c in range(3):
    result[:, :, c] *= mask  # 곱셈 (감소)

result = np.clip(result, 0, 255).astype(np.uint8)
```

### 수식
```
V_out = V_in × mask
mask = Linspace(1.0, 1.0 - intensity, width)

특징: 곱셈이므로 원본 색상 비율 유지 (덧셈 아님)
```

---

## 5️⃣ Gamma Correction (비선형 밝기 조정)

### 목적
비선형 밝기 조정으로 다양한 광원 환경 모사.

### 코드 위치
```
augmentation_recipe.py: apply_gamma_correction() [Line 421]
```

### 수식
```
V_out = 255 × (V_in / 255)^γ

γ < 1 (예: 0.5): 밝기 증가 (어두운 부분 복원)
γ > 1 (예: 2.0): 밝기 감소 (과노출 보정)
γ = 1:           항등 변환 (원본)
```

### 성능 최적화 — LUT (Look-Up Table)

#### 브루트 포스 방식 (느림)
```python
# O(HW) 복잡도: 모든 픽셀에 지수 연산
for i in range(H):
    for j in range(W):
        for c in range(3):
            result[i,j,c] = 255 * (image[i,j,c]/255)**gamma
```
→ 1280×720×3 이미지 = 276만 연산!

#### LUT 방식 (빠름) — 실제 코드
```python
# Step 1: 0~255의 256개 정수에만 변환 테이블 사전 계산
lut = np.array(
    [np.clip(255.0 * (i / 255.0) ** gamma, 0, 255) 
     for i in range(256)],
    dtype=np.uint8
)
# lut = [V_out(0), V_out(1), ..., V_out(255)]

# Step 2: 각 픽셀값을 테이블 인덱스로 사용
result = cv2.LUT(image, lut)  # 벡터화 연산 (매우 빠름)
```
→ O(256) 선계산 + O(HW) 테이블 참조  
→ **~10배 속도 향상**

### 코드
```python
def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError(f"gamma는 양수여야합니다")
    
    # LUT 생성
    lut = np.array(
        [np.clip(255.0 * (i / 255.0) ** gamma, 0, 255) 
         for i in range(256)],
        dtype=np.uint8
    )
    
    # LUT 적용
    return cv2.LUT(image, lut)
```

---

## 6️⃣ CLAHE (Contrast Limited Adaptive Histogram Equalization)

### 목적
지역 히스토그램 평탄화로 명암비 증강 (색상 왜곡 최소화).

### 코드 위치
```
augmentation_recipe.py: apply_histogram_equalization() [Line 460]
```

### 왜 BGR 직접 처리가 아닌가?

#### BGR 직접 처리 (나쁜 예)
```python
# ❌ 각 채널을 독립적으로 평탄화
result[:,:,0] = cv2.equalizeHist(image[:,:,0])  # Blue
result[:,:,1] = cv2.equalizeHist(image[:,:,1])  # Green
result[:,:,2] = cv2.equalizeHist(image[:,:,2])  # Red

# 문제: 채널 간 비율이 깨져 색조 왜곡 발생
```

#### YCrCb 분리 처리 (좋은 예) — 실제 코드
```python
# Step 1: BGR → YCrCb (밝기/색차 분리)
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# Step 2: Y(밝기) 채널만 추출
y, cr, cb = cv2.split(ycrcb)

# Step 3: Y 채널만 처리
clahe = cv2.createCLAHE(
    clipLimit=2.0,              # 증폭 한계 (1~5)
    tileGridSize=(8, 8)         # 타일 크기
)
y_eq = clahe.apply(y)

# Step 4: 처리된 Y를 다시 조합 → BGR 복원
ycrcb_eq = cv2.merge([y_eq, cr, cb])
result = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
```

### CLAHE vs 일반 equalizeHist

| 특성 | equalizeHist | CLAHE |
|------|-------------|-------|
| **범위** | 전역 | 타일 단위 |
| **clipLimit** | 없음 | 있음 (노이즈 제한) |
| **결과** | 더 강함 | 더 자연스러움 |
| **노이즈** | 과증폭 가능 | 제어됨 |

### 파라미터
```python
clip_limit: float = 2.0      # 1.0 (약함) ~ 5.0 (강함)
tile_size: int = 8 또는 16   # 8×8 또는 16×16 픽셀 단위
```

---

## 7️⃣ Cutout (Random Erasing)

### 목적
이미지의 일부를 마스킹하여 부분 오클루전(가려짐) 모사.

### 코드 위치
```
augmentation_recipe.py: apply_cutout() [Line 515]
```

### 작동 원리

#### 단계 1: 마스크 면적 결정
```python
# 전체 이미지 면적 기준
total_area = img_h * img_w
cut_area = total_area × Uniform(area_ratio_min, area_ratio_max)
# 예: 2% ~ 10%
```

#### 단계 2: 종횡비(Aspect Ratio) 결정
```python
aspect_ratio = np.random.uniform(0.3, 3.0)
# 너무 가늘거나 넓지 않도록 제한

# 면적과 종횡비로 높이/너비 계산
# S_cut = cut_h × cut_w
# r = cut_w / cut_h
# → cut_h = sqrt(S_cut / r)
# → cut_w = sqrt(S_cut × r)

cut_h = int(np.sqrt(cut_area / aspect_ratio))
cut_w = int(np.sqrt(cut_area * aspect_ratio))
```

#### 단계 3: 무작위 위치에 배치
```python
rx = np.random.randint(0, max(1, img_w - cut_w))
ry = np.random.randint(0, max(1, img_h - cut_h))

rx2 = min(rx + cut_w, img_w)
ry2 = min(ry + cut_h, img_h)
```

#### 단계 4: 영역 채우기
```python
# fill_mode='mean': 이미지 평균값으로 채우기
fill_val = image.mean(axis=(0, 1)).astype(np.uint8)  # (3,)

# fill_mode='black': 검정(0,0,0)으로 채우기
fill_val = np.array([0, 0, 0], dtype=np.uint8)

# NumPy 슬라이싱으로 적용
result = image.copy()
result[ry:ry2, rx:rx2] = fill_val  # 브로드캐스팅
```

### BBox 처리

#### 공간 변환 없음
```
Cutout은 픽셀을 "지울" 뿐이고 좌표를 이동하지 않으므로
기본적으로 BBox 좌표는 변경 없음.
```

#### drop_covered 옵션
```python
# Cutout이 BBox를 100% 완전히 덮으면 어노테이션 제거
if drop_covered:
    fully_covered = (
        rx  <= bx      and
        ry  <= by      and
        rx2 >= bx + bw and
        ry2 >= by + bh
    )
    if not fully_covered:
        kept.append(ann)
```

---

## 8️⃣ Mosaic (4-이미지 합성)

### 목적
4개의 서로 다른 이미지를 조합하여 **맥락적 다양성** 증강.

### 코드 위치
```
augmentation_recipe.py: apply_mosaic() [Line 631]
```

### 작동 원리

#### 단계 1: 4개 이미지 풀에서 선택
```python
# image_pool = [(img1, ann1), (img2, ann2), (img3, ann3), (img4, ann4), ...]
# 무작위로 4개 선택 (또는 제공된 3개 + primary)
```

#### 단계 2: 출력 해상도를 4개 사분면으로 분할
```python
# 출력: (output_h, output_w)
# 중심점 (cx, cy)를 무작위로 선택 → 4개 사분면
#
#  ┌─────────────┬─────────────┐
#  │   (0,1)     │   (1,1)     │
#  │  img1       │  img2       │
#  ├─────────────┼─────────────┤
#  │   (0,0)     │   (1,0)     │
#  │  img3       │  img4       │
#  └─────────────┴─────────────┘
#        (cx)         (cx)
#        (cy)

cx = np.random.randint(0, output_w)
cy = np.random.randint(0, output_h)

# 4개 사분면 경계
quads = [
    (0,  cx, 0,  cy),   # 좌상: x_start, x_end, y_start, y_end
    (cx, output_w, 0, cy),      # 우상
    (0, cx, cy, output_h),      # 좌하
    (cx, output_w, cy, output_h) # 우하
]
```

#### 단계 3: 각 이미지를 사분면에 리사이즈
```python
for i, (x_s, x_e, y_s, y_e) in enumerate(quads):
    src_img = images[i]
    
    # 사분면 크기로 리사이즈
    quad_w = x_e - x_s
    quad_h = y_e - y_s
    resized = cv2.resize(src_img, (quad_w, quad_h),
                        interpolation=cv2.INTER_LINEAR)
    
    # 캔버스에 배치
    canvas[y_s:y_e, x_s:x_e] = resized
```

#### 단계 4: BBox 좌표 변환 (핵심!)
```
각 이미지가 사분면에 리사이즈되면서
스케일 팩터(scale_x, scale_y)와
오프셋(x_offset, y_offset)이 생김

예: 원본 이미지 (640×480) → 사분면 (320×240)으로 리사이즈
    scale_x = 320 / 640 = 0.5
    scale_y = 240 / 480 = 0.5
    x_offset = 사분면의 캔버스 x 시작점
    y_offset = 사분면의 캔버스 y 시작점

각 원본 BBox 좌표 (bx, by, bw, bh)를 변환:
    new_bx = bx × scale_x + x_offset
    new_by = by × scale_y + y_offset
    new_bw = bw × scale_x
    new_bh = bh × scale_y

그 후 이미지 경계(0~output_w, 0~output_h)로 클리핑
```

### 코드 예시
```python
# 스케일 계산
scale_x = quad_w / max(src_w, 1)
scale_y = quad_h / max(src_h, 1)

# 각 어노테이션 변환
for ann in src_anns:
    bx = ann['x'] * scale_x + x_offset
    by = ann['y'] * scale_y + y_offset
    bw = ann['w'] * scale_x
    bh = ann['h'] * scale_y
    
    # AABB 클리핑
    bx = np.clip(bx, 0, output_w)
    by = np.clip(by, 0, output_h)
    bx2 = np.clip(bx + bw, 0, output_w)
    by2 = np.clip(by + bh, 0, output_h)
    
    # 클리핑 후 유효 검사 (1px 이상)
    final_w = bx2 - bx
    final_h = by2 - by
    if final_w >= 1.0 and final_h >= 1.0:
        kept.append((bx, by, final_w, final_h))
```

---

## 9️⃣ Polygon Homography (v2.0)

### 목적
자유형 다각형(Instance Segmentation)에 동차 좌표계 변환 적용.

### 코드 위치
```
augmentation_recipe.py: transform_polygon_homography() [Line 258]
```

### AABB vs Polygon 비교

#### AABB (4점)
```
직사각형 4개 꼭짓점만 변환
→ 사각형이 됨
→ 사각형을 감싸는 새 직사각형이 BBox
→ 자유로운 모양 손실 ❌
```

#### Polygon (N점)
```
N개 점 전체 변환
→ 다각형 형태 유지
→ Instance Segmentation에 정확 ✅
```

### 작동 원리

#### 단계 1: 동차 좌표 행렬 구성
```python
# 입력: points = [[x1,y1], [x2,y2], ..., [xN,yN]]
# 변환: (3, N) 행렬 구성
pts = np.array(points, dtype=np.float64)  # (N, 2)

ones = np.ones((1, N), dtype=np.float64)
pts_h = np.vstack([pts.T, ones])  # (3, N)
# [[x1..xN],
#  [y1..yN],
#  [1..1]]
```

#### 단계 2: 행렬 곱셈
```python
transformed = H @ pts_h  # (3×3) @ (3×N) = (3×N)
```

#### 단계 3: Perspective Divide
```python
W = transformed[2]
W = np.where(np.abs(W) < 1e-10, 1e-10, W)  # 0 나누기 방지

xs = transformed[0] / W  # (N,)
ys = transformed[1] / W  # (N,)
```

#### 단계 4: 클리핑 및 반환
```python
xs = np.clip(xs, 0, img_w)
ys = np.clip(ys, 0, img_h)

result = [[round(int(x)), round(int(y))] for x, y in zip(xs, ys)]
```

### 코드
```python
def transform_polygon_homography(points, H, img_w, img_h):
    if not points:
        return points
    
    pts = np.array(points, dtype=np.float64)
    N = pts.shape[0]
    
    ones = np.ones((1, N), dtype=np.float64)
    pts_h = np.vstack([pts.T, ones])
    
    transformed = H @ pts_h
    
    W = transformed[2]
    W = np.where(np.abs(W) < 1e-10, 1e-10, W)
    
    xs = transformed[0] / W
    ys = transformed[1] / W
    
    xs = np.clip(xs, 0, img_w)
    ys = np.clip(ys, 0, img_h)
    
    return [[round(int(x)), round(int(y))] for x, y in zip(xs, ys)]
```

---

## 🔟 Copy-Paste Augmentation (v2.0)

### 목적
객체의 정밀한 경계(Polygon)를 마스크로 사용하여 배경 이미지에 합성.

### 코드 위치
```
augmentation_recipe.py: apply_copy_paste() [Line 773]
```

### 배경 오염 문제 해결

#### 기존 방식 (나쁜 예)
```
원본 이미지에서 BBox 영역만 crop
→ 배경 픽셀 함께 포함 (배경 오염)
```

#### Copy-Paste 방식 (좋은 예)
```
Polygon 마스크 → 정확한 객체만 추출
→ 배경 완전 교체 (깨끗함)
```

### 작동 원리

#### 단계 1: Polygon → 이진 마스크 생성
```python
# points: [[x1,y1], [x2,y2], ...]
poly_pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

# 2D 이진 마스크 (내부=255, 외부=0)
mask = np.zeros((img_h, img_w), dtype=np.uint8)
cv2.fillPoly(mask, [poly_pts], 255)
```

#### 단계 2: Bitwise AND로 누끼 추출
```python
# 배경: 0, 객체: 원본 픽셀값인 "누끼"
mask_3ch = cv2.merge([mask, mask, mask])  # (H, W, 3)
obj_only = cv2.bitwise_and(primary_image, mask_3ch)
```

#### 단계 3: 배경 이미지 선택 + 리사이즈
```python
bg_raw = background_pool[np.random.randint(len(background_pool))]
background = cv2.resize(bg_raw, (img_w, img_h),
                       interpolation=cv2.INTER_LINEAR)
```

#### 단계 4: 객체 Bounding Rect 계산
```python
# 최소 외접 사각형 (crop 범위)
obj_rect = cv2.boundingRect(poly_pts)
ox, oy, ow, oh = obj_rect

obj_crop = obj_only[oy:oy+oh, ox:ox+ow]
mask_crop = mask[oy:oy+oh, ox:ox+ow]
```

#### 단계 5: 무작위 위치 결정
```python
# 객체가 완전히 이미지 내에 들어오도록
max_dx = max(0, img_w - ow)
max_dy = max(0, img_h - oh)

dx = np.random.randint(0, max_dx + 1)
dy = np.random.randint(0, max_dy + 1)
```

#### 단계 6: 마스크 기반 합성
```python
# offset: (원본 객체 좌측상단 기준 → 새 위치 기준)
shift_x = dx - ox
shift_y = dy - oy

# 배경 + 객체 합성 (마스크 영역만)
dst_region = result_image[dy:dy+oh, dx:dx+ow]
mask_inv = cv2.bitwise_not(mask_crop)

bg_part = cv2.bitwise_and(dst_region, dst_region, mask=mask_inv)
obj_part = cv2.bitwise_and(obj_crop, obj_crop, mask=mask_crop)

result_image[dy:dy+oh, dx:dx+ow] = cv2.add(bg_part, obj_part)
```

#### 단계 7: Polygon 좌표 이동
```python
# 모든 점을 (shift_x, shift_y)만큼 평행 이동
shifted = []
for px, py in polygon_points:
    nx = int(np.clip(px + shift_x, 0, img_w))
    ny = int(np.clip(py + shift_y, 0, img_h))
    shifted.append([nx, ny])

ann['points'] = shifted
```

### 수식
```
누끼_이미지 = 원본 ⊗ 마스크  (⊗: bitwise AND)
합성_이미지 = 배경 ⊗ ¬마스크 + 누끼 ⊗ 마스크
라벨_이동 = 원본_점 + [shift_x, shift_y]
```

### 마스크 합성 상세
```
결과 = (배경 AND ¬마스크) OR (객체 AND 마스크)

배경을 AND ¬마스크하면:
- 마스크 영역 = 0 (검정)
- 그 외 영역 = 배경 유지

객체를 AND 마스크하면:
- 마스크 영역 = 객체
- 그 외 영역 = 0

두 결과를 OR (또는 ADD)하면:
- 최종: 배경 + 객체 완벽 합성
```

---

## 📊 증강 기법 비교표

| 기법 | 입력 | 출력 | 공간변환 | BBox처리 | 속도 |
|------|------|------|---------|---------|------|
| Homography | 이미지 | 이미지 | ✅ 회전+원근 | 4점 변환→AABB | 빠름 |
| Poisson | 이미지 | 이미지 | ❌ | 유지 | 보통 |
| Specular | 이미지 | 이미지 | ❌ | 유지 | 매우빠름 |
| Shadow | 이미지 | 이미지 | ❌ | 유지 | 매우빠름 |
| Gamma | 이미지 | 이미지 | ❌ | 유지 | **극빠름**(LUT) |
| CLAHE | 이미지 | 이미지 | ❌ | 유지 | 보통 |
| Cutout | 이미지+BBox | 이미지+BBox | ❌ | 유지/드롭 | 매우빠름 |
| Mosaic | 4×이미지+BBox | 이미지+BBox | ✅ 스케일+시프트 | 변환+클립 | 느림 |
| Poly Homo | 이미지+Poly | 이미지+Poly | ✅ 회전+원근 | N점 변환+클립 | 빠름 |
| Copy-Paste | 이미지+Poly | 이미지+Poly | ✅ 평행이동 | 좌표시프트 | 보통 |

---

## 🎯 Pipeline 순서

```python
run_augmentation_pipeline() 함수 내 실제 순서:

1. Homography 변환
   └─ 이미지 왜곡 + 4점 BBox 변환
2. Poisson Noise
   └─ 물리적 노이즈 합성
3. Specular Flare (선택)
   └─ 빛 반사 효과
4. Gradient Shadow (선택)
   └─ 그라데이션 그림자
5. Gamma Correction
   └─ 비선형 밝기 (LUT 최적화)
6. CLAHE
   └─ 명암비 증강 (색상 보존)
7. Cutout
   └─ 오클루전 시뮬레이션
8. Mosaic
   └─ 4-이미지 합성 (맥락 다양성)
9. Copy-Paste (v2.0)
   └─ 다각형 기반 객체 합성

반환: (증강_이미지, 변환된_라벨, 파라미터_기록)
```

---

## 📝 주요 설계 원칙

### 1. 데이터 무결성
- 모든 좌표 변환 후 **경계 클리핑** (0~img_w/h)
- 클리핑 후 최소 1px 크기 검증
- 0 나누기 방지 (`W < 1e-10` → 1e-10)

### 2. 성능 최적화
- **Gamma**: LUT로 10배 속도 향상
- **NumPy 벡터화**: 반복문 최소화
- **cv2 네이티브**: OpenCV C++ 백엔드 활용

### 3. 색상 왜곡 방지
- **CLAHE**: YCrCb 색공간으로 색상 보존
- **Shadow**: 곱셈 (덧셈 아님) → 색비율 유지

### 4. 수치 안전성
- 타입 변환 명시적 (float64 → uint8)
- `np.clip` 오버플로우 방지
- `dtype` 일관성 유지

---

**Last Updated**: 2026-03-14  
**Version**: v2.0 Complete
