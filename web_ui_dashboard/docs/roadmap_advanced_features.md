# 🚀 고도화 로드맵 — 다음 단계 구현 추천안

**작성일:** 2026년 3월 14일  
**기준:** 현재 구현 수준 분석 → 다음 단계 추천  
**대상:** 프로젝트 팀원

---

## 🎯 핵심 전략

현재 상태:
- ✅ **기초:** 라벨링 UI, Undo/Redo, 기본 검증, 안전한 저장
- ✅ **증강:** Homography (원근), Poisson 노이즈, 기본 조명

다음 단계는 **3가지 축**으로 진행 권장:

| 축 | 내용 | 난이도 |
|----|------|--------|
| **A축 (고도화)** | 이미지 증강 다양화 | ⭐⭐⭐ |
| **B축 (편의성)** | 라벨링 UI 개선 | ⭐⭐ |
| **C축 (통합)** | 데이터 포맷 및 분석 | ⭐⭐⭐ |

---

## A축: 이미지 증강 고도화

### A-1: CLAHE (Contrast Limited Adaptive Histogram Equalization) ⭐⭐⭐

#### 📌 목표
밝기 불균형이 있는 이미지의 명암비를 균등하게 개선. 예: 어두운 사진, 실내/실외 혼합 장면.

#### 🔍 원리

**일반 히스토그램 평탄화 vs CLAHE:**

```
일반 Histogram Equalization:
  전체 이미지의 픽셀 분포를 균등하게 펼침
  → 문제: 전역 처리라 노이즈 과대 증폭, 부자연스러움

CLAHE (Contrast Limited Adaptive):
  이미지를 타일로 분할 (예: 8×8 그리드)
  → 각 타일별로 독립적으로 히스토그램 평탄화
  → 타일 경계에서 선형 보간 (부드러운 전환)
  → 명암비 증폭 한계 설정 (clipLimit) → 노이즈 제어
```

#### 💻 구현 (OpenCV)

```python
import cv2

def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """
    CLAHE 명암비 개선
    
    Args:
        image: BGR 이미지 (H, W, 3)
        clip_limit: 명암비 제한값 (1.0~5.0 권장)
        tile_size: 타일 크기 (8, 16, 32 중 선택)
    
    Returns:
        개선된 이미지
    """
    # BGR → LAB 변환 (밝기 채널만 처리)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # CLAHE 적용 (L 채널에만)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, 
                             tileGridSize=(tile_size, tile_size))
    enhanced_l = clahe.apply(l_channel)
    
    # 다시 합치기
    enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
    
    # LAB → BGR 변환
    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return result
```

#### 📊 레시피 JSON 추가

```json
{
  "augmentation_params": {
    "clahe": {
      "enabled": true,
      "clip_limit": 2.5,
      "tile_size": 8
    }
  }
}
```

#### ✅ 언제 사용?
- 야외 vs 실내 조명이 섞인 이미지
- 어두운 물체를 탐지할 때
- 로우 라이트(Low-light) 이미지

#### ⏱️ 소요 시간: **1일**

---

### A-2: Color Jitter (색상 변형) ⭐⭐⭐

#### 📌 목표
색온도, 포화도, 밝기를 무작위로 변형하여 조명 다양성 확보.

#### 🔍 원리

```
원본 이미지 (RGB)
    ↓
HSV 변환 (Hue-Saturation-Value)
    ↓
각 채널 독립적으로 무작위 조정:
  - H (색상): ±θ 도 회전
  - S (포화도): ×λ_s 배수
  - V (명도): ±δ 조정
    ↓
RGB 복원
```

#### 💻 구현

```python
def apply_color_jitter(image, 
                       brightness=0.2, 
                       contrast=0.2, 
                       saturation=0.2, 
                       hue=0.1):
    """
    Color Jitter 적용
    
    Args:
        brightness: 명도 변화 범위 [0.0, 1.0]
                   (예: 0.2 = ±20%)
        contrast: 명암비 변화 범위
        saturation: 포화도 변화 범위
        hue: 색상 변화 범위 [0.0, 0.5]
             (0.5 = 180도 = 전체 색상환의 절반)
    
    Returns:
        색상 변형된 이미지
    """
    # HSV로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # Hue 조정 (0~180 범위)
    hue_delta = np.random.uniform(-hue * 180, hue * 180)
    h = (h + hue_delta) % 180
    
    # Saturation 조정
    sat_factor = np.random.uniform(1 - saturation, 1 + saturation)
    s = np.clip(s * sat_factor, 0, 255)
    
    # Value 조정
    val_factor = np.random.uniform(1 - brightness, 1 + brightness)
    v = np.clip(v * val_factor, 0, 255)
    
    # HSV 병합 및 BGR로 복원
    hsv_adjusted = cv2.merge([h, s, v]).astype(np.uint8)
    result = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    
    return result
```

#### 📊 레시피 JSON 추가

```json
{
  "augmentation_params": {
    "color_jitter": {
      "enabled": true,
      "brightness": 0.2,
      "contrast": 0.2,
      "saturation": 0.3,
      "hue": 0.1
    }
  }
}
```

#### ✅ 언제 사용?
- 카메라 설정이 다양한 데이터
- 실내/실외 조명 변화 모방
- 시간대 다양성 (아침/점심/저녁)

#### ⏱️ 소요 시간: **1일**

---

### A-3: Mixup (데이터 합성) ⭐⭐⭐⭐

#### 📌 목표
두 이미지를 선형 보간(blend)하여 새로운 학습 샘플 생성. 강력한 정규화 기법.

#### 🔍 원리

```
이미지1(I1, BBox1) + 이미지2(I2, BBox2)
    ↓
선형 보간: I_mixed = λ × I1 + (1-λ) × I2
        (λ: 0.0~1.0 사이 무작위)
    ↓
BBox 병합: 두 세트 모두 유지
           (오버랩 판정 후 필터링 가능)
    ↓
새로운 학습 샘플
```

#### 💻 구현

```python
def apply_mixup(image1, bboxes1,
                image2, bboxes2,
                alpha=0.5):
    """
    Mixup: 두 이미지 선형 보간
    
    Args:
        image1, image2: 이미지 (같은 크기 전제)
        bboxes1, bboxes2: BBox 리스트
        alpha: 혼합 비율 (0.5 = 50%-50%)
    
    Returns:
        (mixed_image, merged_bboxes)
    """
    # 크기 맞춰야 함 (리사이즈 필요시)
    if image1.shape != image2.shape:
        h, w = image1.shape[:2]
        image2 = cv2.resize(image2, (w, h))
    
    # 혼합 비율 샘플링 (Beta 분포 사용 권장)
    lam = np.random.beta(alpha, alpha)
    
    # 선형 보간
    mixed = (lam * image1 + (1 - lam) * image2).astype(np.uint8)
    
    # BBox 병합
    merged_bboxes = []
    for bbox in bboxes1:
        bbox['source'] = 'image1'
        bbox['weight'] = lam
        merged_bboxes.append(bbox)
    
    for bbox in bboxes2:
        bbox['source'] = 'image2'
        bbox['weight'] = 1 - lam
        merged_bboxes.append(bbox)
    
    return mixed, merged_bboxes
```

#### 📊 레시피 JSON (조정 필요)

```json
{
  "mixup": {
    "enabled": true,
    "partner_image_id": "e5f6g7h8",
    "alpha": 0.5,
    "lambda": 0.6
  }
}
```

#### ⚠️ 주의사항
- **이미지 쌍 관리:** 어떤 두 이미지를 혼합할지 결정 필요
- **BBox 오버랩:** 혼합 후 BBox들이 완전히 겹칠 수 있음 → 필터링 로직 필요
- **라벨 가중치:** 각 BBox에 신뢰도(confidence) 추가 가능

#### ✅ 언제 사용?
- 데이터셋 부족할 때
- 모델 정규화 (dropout/regularization 대체 효과)

#### ⏱️ 소요 시간: **2일** (BBox 관리 로직 복잡)

---

### A-4: Elastic Deformation (탄성 변형) ⭐⭐⭐⭐

#### 📌 목표
이미지를 탄성 재료처럼 휘어뜨리는 왜곡 변형. 매우 현실적인 변형.

#### 🔍 원리

```
1. 랜덤 벡터장 생성 (displacement field)
   - 각 픽셀 (x, y)에 대해 이동 거리 (dx, dy) 정의

2. 가우시안 필터로 평탄화
   - 부드러운 변형 (픽셀 단위 랜덤 X)

3. 각 픽셀을 새로운 위치로 이동
   - (x, y) → (x + dx, y + dy)

결과: 자연스러운 구부러짐 (라면 면발, 종이 주름 같은)
```

#### 💻 구현

```python
def apply_elastic_deformation(image, 
                              alpha=30, 
                              sigma=3):
    """
    Elastic Deformation
    
    Args:
        image: 입력 이미지 (H, W, 3)
        alpha: 변형 강도 (픽셀 단위)
        sigma: 가우시안 평탄화 정도
    
    Returns:
        변형된 이미지
    """
    h, w = image.shape[:2]
    
    # 1. 랜덤 벡터장 생성 (dx, dy)
    dx = np.random.randn(h, w) * alpha
    dy = np.random.randn(h, w) * alpha
    
    # 2. 가우시안 평탄화 (부드러운 변형)
    dx = cv2.GaussianBlur(dx, (5, 5), sigma)
    dy = cv2.GaussianBlur(dy, (5, 5), sigma)
    
    # 3. 격자 좌표 생성
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 4. 변형된 좌표
    x_new = (x + dx).astype(np.float32)
    y_new = (y + dy).astype(np.float32)
    
    # 5. 리맵핑 (이중 선형 보간)
    result = cv2.remap(image, x_new, y_new, 
                       cv2.INTER_LINEAR, 
                       borderMode=cv2.BORDER_REFLECT_101)
    
    return result
```

#### 📊 레시피 JSON

```json
{
  "augmentation_params": {
    "elastic_deformation": {
      "enabled": true,
      "alpha": 30,
      "sigma": 3
    }
  }
}
```

#### ⚠️ BBox 변환이 복잡함

```python
def transform_bbox_elastic(bbox, dx, dy, img_w, img_h):
    """
    Elastic deformation 후 BBox 변환 (근사)
    
    → 완벽한 수학적 변환은 어려움
    → 4개 꼭짓점 각각을 변형 후 AABB로 계산
    """
    x, y, w, h = bbox
    corners = np.array([
        [x,     y,     1],
        [x + w, y,     1],
        [x + w, y + h, 1],
        [x,     y + h, 1],
    ])
    
    # 각 꼭짓점에서 변형 벡터 샘플링
    transformed = []
    for cx, cy, _ in corners:
        cx_int, cy_int = int(np.clip(cx, 0, img_w-1)), \
                         int(np.clip(cy, 0, img_h-1))
        dx_val = dx[cy_int, cx_int]
        dy_val = dy[cy_int, cx_int]
        transformed.append([cx + dx_val, cy + dy_val])
    
    # AABB 계산
    xs = [p[0] for p in transformed]
    ys = [p[1] for p in transformed]
    
    new_x = max(0, min(xs))
    new_y = max(0, min(ys))
    new_w = min(img_w, max(xs)) - new_x
    new_h = min(img_h, max(ys)) - new_y
    
    return [int(new_x), int(new_y), int(new_w), int(new_h)]
```

#### ✅ 언제 사용?
- 유연한 물체 탐지 (옷, 종이, 식물)
- 실시간 변형 시뮬레이션

#### ⏱️ 소요 시간: **2일** (BBox 변환 로직)

---

### A-5: GridMask (부분 마스킹) ⭐⭐⭐

#### 📌 목표
이미지의 일부 영역을 체계적으로 제거하여 모델이 전체 컨텍스트를 보도록 강제.

#### 🔍 원리

```
┌──────────────────────────┐
│ ██ ░░ ██ ░░ ██ ░░ ██ ░░ │  ← 체스판 패턴
│ ░░ ██ ░░ ██ ░░ ██ ░░ ██ │
│ ██ ░░ ██ ░░ ██ ░░ ██ ░░ │
│ ░░ ██ ░░ ██ ░░ ██ ░░ ██ │
└──────────────────────────┘

██ = 마스킹된 영역 (검은색)
░░ = 유지된 영역

효과:
- 모델이 부분 정보만으로 객체 인식
- Dropout의 공간 버전 같은 정규화 효과
```

#### 💻 구현

```python
def apply_gridmask(image, 
                   ratio=0.5, 
                   d_min=32, 
                   d_max=224):
    """
    GridMask 적용
    
    Args:
        image: 입력 이미지
        ratio: 마스킹 비율 (0.5 = 50%)
        d_min, d_max: 그리드 셀 크기 범위 (픽셀)
    
    Returns:
        GridMask 적용된 이미지
    """
    h, w = image.shape[:2]
    
    # 1. 그리드 셀 크기 결정
    d = np.random.randint(d_min, d_max + 1)
    
    # 2. 마스크 생성 (체스판 패턴)
    mask = np.ones((h, w), dtype=np.uint8)
    
    for i in range(0, h, d):
        for j in range(0, w, d):
            if (i // d + j // d) % 2 == 0:
                mask[i:i+d, j:j+d] = 0  # 검은색
    
    # 3. 마스크 적용
    image_masked = image.copy()
    for c in range(3):  # BGR 채널
        image_masked[:, :, c] = image[:, :, c] * mask
    
    return image_masked
```

#### 📊 레시피 JSON

```json
{
  "augmentation_params": {
    "gridmask": {
      "enabled": true,
      "ratio": 0.5,
      "d_min": 32,
      "d_max": 224
    }
  }
}
```

#### ✅ 언제 사용?
- 모델의 강건성(robustness) 향상
- 오클루전(occlusion) 상황 모방

#### ⏱️ 소요 시간: **1일**

---

### A-6: AutoAugment / RandAugment ⭐⭐⭐⭐⭐

#### 📌 목표
여러 증강 기법을 조합하여 최적 정책을 자동 선택 (또는 무작위 선택).

#### 🔍 원리

```
AutoAugment:
  ├─ 강화학습으로 최적 증강 정책 학습
  └─ 시간 많이 걸리지만 매우 효과적

RandAugment (권장):
  ├─ 무작위로 N개의 기법 선택
  ├─ 강도(magnitude) M을 고정
  └─ 빠르고 구현 간단
```

#### 💻 구현 (RandAugment)

```python
class RandAugment:
    """
    RandAugment: 무작위로 N개의 기법을 magnitude M으로 적용
    """
    
    AUGMENTATIONS = {
        'shear_x': lambda img, m: apply_shear_x(img, m),
        'shear_y': lambda img, m: apply_shear_y(img, m),
        'translate_x': lambda img, m: apply_translate_x(img, m),
        'translate_y': lambda img, m: apply_translate_y(img, m),
        'rotate': lambda img, m: apply_rotate(img, m),
        'auto_contrast': lambda img, m: cv2.cvtColor(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        'invert': lambda img, m: 255 - img,
        'equalize': lambda img, m: apply_equalize(img),
        'solarize': lambda img, m: apply_solarize(img, m),
        'posterize': lambda img, m: apply_posterize(img, m),
        'contrast': lambda img, m: apply_contrast(img, m),
        'color': lambda img, m: apply_color(img, m),
        'brightness': lambda img, m: apply_brightness(img, m),
        'sharpness': lambda img, m: apply_sharpness(img, m),
    }
    
    def __init__(self, n=2, m=9):
        """
        Args:
            n: 적용할 기법 수 (예: 2개)
            m: 강도 레벨 (0~30, 클수록 강함)
        """
        self.n = n
        self.m = m / 30.0  # 정규화 (0~1)
    
    def __call__(self, image):
        ops = np.random.choice(
            list(self.AUGMENTATIONS.keys()),
            self.n,
            replace=False
        )
        
        for op_name in ops:
            image = self.AUGMENTATIONS[op_name](image, self.m)
        
        return image
```

#### 📊 레시피 JSON

```json
{
  "augmentation_params": {
    "rand_augment": {
      "enabled": true,
      "n": 2,
      "magnitude": 9
    }
  }
}
```

#### ✅ 언제 사용?
- 모든 상황에 범용적으로 적용 가능
- 빠르고 효과적인 기본 증강 정책

#### ⏱️ 소요 시간: **3일** (15가지 기법 각각 구현)

---

### A-7: 시간축 증강 (Temporal Augmentation) ⭐⭐⭐⭐⭐

#### 📌 목표
연속 프레임(비디오) 데이터에서 프레임 간 관계를 활용한 증강.

#### 🔍 원리

```
프레임 시퀀스: [F_t-1, F_t, F_t+1]
    ↓
Optical Flow 계산 (F_t-1 → F_t 이동 벡터)
    ↓
BBox도 동일하게 이동
    ↓
새로운 학습 샘플: (F_t + 이동된 BBox)
```

#### 💻 개요

```python
def apply_temporal_augmentation(frames, bboxes_seq, alpha=0.5):
    """
    Temporal Augmentation: 연속 프레임에서 BBox 추적
    
    Args:
        frames: [F_1, F_2, F_3, ...] 연속 프레임
        bboxes_seq: [[BBox_1_1, BBox_1_2, ...], [...], ...]
        alpha: 프레임 혼합 비율
    
    Returns:
        증강된 (프레임, BBox) 쌍들
    """
    # Optical flow 계산 (Lucas-Kanade, FlowNet 등)
    # BBox를 flow에 따라 이동
    # 프레임 간 보간
    pass
```

#### ⏱️ 소요 시간: **4~5일** (복잡도 매우 높음)

---

## B축: 라벨링 UI 개선

### B-1: 박스 선택 및 이동/리사이즈 ⭐⭐⭐

#### 📌 목표
이미 그려진 박스를 마우스로 선택 → 이동 또는 리사이즈.

#### 구현 아이디어

```javascript
// 마우스 이벤트 체계 변경
canvas.addEventListener('mousedown', (e) => {
    const box = getBoxAtPoint(mx, my);
    if (box !== -1) {
        state.selectedBoxIdx = box;
        state.dragMode = 'move';  // 또는 'resize'
    } else {
        // 새 박스 그리기 시작
        state.isDrawing = true;
    }
});

canvas.addEventListener('mousemove', (e) => {
    if (state.selectedBoxIdx !== -1) {
        // 선택된 박스 이동
        state.boxes[state.selectedBoxIdx].x += dx;
        state.boxes[state.selectedBoxIdx].y += dy;
    } else if (state.isDrawing) {
        // 새 박스 그리기
        state.currentBox = {...};
    }
});
```

#### ⏱️ 소요 시간: **2~3일**

---

### B-2: 라벨 직접 편집 (더블클릭) ⭐⭐

#### 📌 목표
박스 위에서 더블클릭 → 라벨 입력 필드 표시 → 수정.

```javascript
canvas.addEventListener('dblclick', (e) => {
    const box = getBoxAtPoint(mx, my);
    if (box !== -1) {
        // 인라인 텍스트 입력
        state.editingBoxIdx = box;
        showInlineEditor(box);
    }
});
```

#### ⏱️ 소요 시간: **1일**

---

### B-3: 클래스 검색 & 필터 ⭐⭐

#### 📌 목표
"car"만 보기, "person" 박스만 하이라이트 등의 필터링.

```javascript
// UI: 필터 입력 박스 추가
function filterBoxesByLabel(labelFilter) {
    return state.boxes.filter(b => 
        b.label.includes(labelFilter)
    );
}
```

#### ⏱️ 소요 시간: **1일**

---

## C축: 데이터 포맷 및 분석

### C-1: YOLO / COCO 자동 변환 ⭐⭐⭐

#### 📌 목표
저장된 JSON을 YOLO `.txt` 또는 COCO JSON 형식으로 변환.

#### 💻 구현

```python
def to_yolo_format(annotation_json_path, output_dir):
    """
    내부 JSON → YOLO format 변환
    
    YOLO 형식:
        class_id center_x_norm center_y_norm width_norm height_norm
    """
    with open(annotation_json_path, 'r') as f:
        data = json.load(f)
    
    img_w = data['image_width']
    img_h = data['image_height']
    
    lines = []
    for ann in data['annotations']:
        cx = (ann['x'] + ann['w'] / 2) / img_w
        cy = (ann['y'] + ann['h'] / 2) / img_h
        w = ann['w'] / img_w
        h = ann['h'] / img_h
        class_id = get_class_id(ann['label'])  # 라벨 → ID 매핑
        
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    output_file = os.path.join(output_dir, 
                               f"{data['image_id']}.txt")
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
```

#### ⏱️ 소요 시간: **1~2일**

---

### C-2: 데이터셋 통계 대시보드 ⭐⭐⭐

#### 📌 목표
웹 UI에 대시보드 추가:
- 클래스별 박스 수
- 이미지당 평균 박스 수
- BBox 크기 분포
- 증강 비율

#### 💻 구현 개요

```python
def compute_dataset_stats(metadata_index_path):
    """
    dataset_index.json을 읽어 통계 계산
    """
    with open(metadata_index_path, 'r') as f:
        index = json.load(f)
    
    stats = {
        'total_images': len(index),
        'total_boxes': 0,
        'class_distribution': {},
        'augmented_ratio': 0,
        'bbox_size_distribution': []
    }
    
    # ... 계산 로직
    
    return stats
```

```html
<!-- 대시보드 HTML 추가 -->
<div id="stats-panel">
    <h3>Dataset Statistics</h3>
    <canvas id="class-chart"></canvas>  <!-- Chart.js 사용 -->
    <canvas id="bbox-size-chart"></canvas>
</div>
```

#### ⏱️ 소요 시간: **2~3일** (Chart.js 연동 포함)

---

### C-3: 배치 업로드 및 병렬 처리 ⭐⭐⭐⭐

#### 📌 목표
여러 이미지를 한 번에 업로드 → 병렬로 처리.

#### 💻 구현 아이디어

```python
# Python: 병렬 처리
from concurrent.futures import ProcessPoolExecutor

def batch_save_all_data(data_list):
    """
    여러 이미지를 병렬로 처리
    """
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(save_all_data, data_list))
    
    return results
```

```javascript
// JavaScript: 다중 파일 업로드
fileInput.multiple = true;

fileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    for (const file of files) {
        await uploadFile(file);
        // 또는 모두 동시 업로드
    }
});
```

#### ⏱️ 소요 시간: **2~3일**

---

## D축: 고급 ML 기능

### D-1: 간단한 객체 탐지 모델 프리뷰 ⭐⭐⭐⭐⭐

#### 📌 목표
라벨링한 데이터로 경량 모델(MobileNet + SSD)을 학습하고, 웹 UI에서 실시간 프리뷰.

#### 구조

```
1. 백엔드: PyTorch/TensorFlow 경량 모델 학습
2. 웹소켓: 실시간 결과 전송
3. 프론트엔드: Canvas에 예측 BBox 표시
```

#### ⏱️ 소요 시간: **5~7일** (모델 학습, 웹소켓 설정)

---

### D-2: 부분 세그멘테이션 (초기) ⭐⭐⭐⭐⭐

#### 📌 목표
픽셀 단위 마스킹으로 업그레이드. 예: 사람의 정확한 실루엣.

#### 기술

```
SAM (Segment Anything Model) 활용:
  - 사용자가 클릭 또는 박스 그림
  - SAM이 객체 경계 자동 인식
  
또는 U-Net 기반 커스텀 모델
```

#### ⏱️ 소요 시간: **1주** (SAM 통합 시 짧음)

---

## 🎯 추천 구현 순서

### Phase 1 (1주): 필수 고도화
1. **A-1: CLAHE** (1일) — 기본적인 전처리 강화
2. **A-2: Color Jitter** (1일) — 색상 다양성
3. **B-1: 박스 선택/이동** (3일) — 라벨링 편의성

### Phase 2 (2주): 중심 확장
4. **A-3: Mixup** (2일) — 데이터 합성
5. **C-1: YOLO 변환** (2일) — 포맷 호환성
6. **C-2: 통계 대시보드** (2일) — 시각화

### Phase 3 (1주): 고도화
7. **A-4: Elastic Deformation** (2일) — 현실적 변형
8. **A-6: RandAugment** (3일) — 범용 증강 정책

### Phase 4 (2주+): 심화
9. **C-3: 배치 병렬 처리** (2~3일) — 성능 최적화
10. **D-1: 모델 프리뷰** (5~7일) — ML 통합

---

## 📊 난이도 vs 효과 분석

```
효과
│
│  A-6 (RandAugment)  ★★★★★
│  D-1 (Model Preview)★★★★★
│  
│  A-3 (Mixup)        ★★★★
│  A-4 (Elastic)      ★★★★
│  A-5 (GridMask)     ★★★
│  B-1 (편집)         ★★★
│  C-1,2 (포맷/대시)  ★★★
│
│  A-1,2 (CLAHE/Color)★★
│  B-2,3 (UI필터)     ★★
│
└───────────────────────────── 난이도
  ⭐  ⭐⭐  ⭐⭐⭐  ⭐⭐⭐⭐  ⭐⭐⭐⭐⭐
```

---

## 💡 최종 추천

**이미지 증강 우선순위:**
1. **A-2: Color Jitter** ← 1일이고 효과 좋음
2. **A-1: CLAHE** ← 실내/외 명암비 개선
3. **A-6: RandAugment** ← 최종 목표 (모든 기법 통합)
4. **A-3: Mixup** ← 데이터 부족 시
5. **A-4: Elastic** ← 유연성 물체용

**UI 개선:**
1. **B-1: 박스 이동/리사이즈** ← 가장 기다려지는 기능

**데이터 관리:**
1. **C-1: YOLO 변환** ← 즉시 필요 (모델 학습용)
2. **C-2: 대시보드** ← 데이터 품질 확인

**ML 통합:**
1. **D-1: 모델 프리뷰** ← 최종 목표 (검증 자동화)

---

## 🚀 다음 액션

```
Week 1:
┌─────────────────────────┐
│ 1. A-2 구현 (Color)     │
│ 2. B-1 시작 (박스 선택) │
└─────────────────────────┘

Week 2:
┌─────────────────────────┐
│ 3. B-1 완료             │
│ 4. A-1 구현 (CLAHE)     │
│ 5. C-1 구현 (YOLO)      │
└─────────────────────────┘

Week 3+:
┌─────────────────────────┐
│ 6. A-6 시작 (RandAug)   │
│ 7. C-2 구현 (대시보드)  │
│ 8. D-1 기획 (모델)      │
└─────────────────────────┘
```

---

**작성:** 2026년 3월 14일  
**최종 목표:** 총 2~3개월 안에 엔터프라이즈급 라벨링 플랫폼 완성 🎯
