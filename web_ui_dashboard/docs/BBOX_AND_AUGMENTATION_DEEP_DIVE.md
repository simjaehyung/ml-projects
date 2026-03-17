# BBox를 모델이 어떻게 받아들이는가, 그리고 BBox가 증강에 어떤 영향을 주는가

---

## 1. 모델은 BBox를 어떻게 "학습"하는가

### 1-1. 이미지 분류(Classification)와 객체 검출(Detection)의 구조적 차이

**분류 모델 (예: ResNet, VGG)**

```
입력: 이미지 1장 (H × W × 3)
        ↓
CNN (특징 추출)
        ↓
Fully Connected Layer
        ↓
출력: [고양이: 0.92, 개: 0.05, 새: 0.03]  ← 확률 하나짜리 벡터
```

**검출 모델 (예: YOLOv8, Faster R-CNN)**

```
입력: 이미지 1장 (H × W × 3)
        ↓
CNN (특징 추출)
        ↓
Detection Head
        ↓
출력: [
  {class: "고양이", x: 230, y: 150, w: 120, h: 90, confidence: 0.94},
  {class: "개",     x: 580, y: 320, w: 200, h: 160, confidence: 0.88}
]
```

검출 모델은 **이미지 내 여러 위치를 동시에 검색**하고, 각 위치마다 "여기에 뭔가 있는가? 있다면 어떤 클래스이고 정확히 어디서 어디까지인가"를 예측합니다.

---

### 1-2. 모델이 BBox를 어떻게 수치로 처리하는가

#### YOLO 방식 (그리드 기반)

이미지를 N×N 격자로 나눕니다 (예: 13×13, 26×26, 52×52).

```
이미지 (640×640)
┌───┬───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │   │
├───┼───┼───┼───┼───┼───┼───┤
│   │   │ ★ │   │   │   │   │  ← ★ : 고양이 중심이 이 셀에 있음
├───┼───┼───┼───┼───┼───┼───┤
│   │   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┴───┘
```

★ 셀을 담당하는 출력 노드가 예측하는 값:

```
[tx, ty, tw, th, objectness_score, class_prob_cat, class_prob_dog, ...]
  ↑   ↑   ↑   ↑
  │   │   │   └── 박스 높이 (로그 스케일 예측)
  │   │   └────── 박스 너비 (로그 스케일 예측)
  │   └────────── 셀 내 y 오프셋 (0~1)
  └────────────── 셀 내 x 오프셋 (0~1)
```

실제 손실 함수(Loss Function)가 학습하는 것:

```python
# YOLO의 학습 손실 (개념적 표현)
loss = (
    λ_coord × Σ[(x_pred - x_true)² + (y_pred - y_true)²]  # 중심 위치 오차
  + λ_coord × Σ[(√w_pred - √w_true)² + (√h_pred - √h_true)²]  # 크기 오차
  + λ_obj   × Σ[BCE(obj_pred, obj_true)]                        # 객체 유무 오차
  + λ_cls   × Σ[BCE(cls_pred, cls_true)]                        # 클래스 오차
)
```

**즉, BBox 좌표(x, y, w, h)는 이 Loss 계산의 직접적인 정답값(Ground Truth)으로 사용됩니다.**
BBox가 없으면 위치·크기 오차를 계산할 수 없고, 검출 학습 자체가 불가능합니다.

---

### 1-3. BBox가 없을 때 발생하는 문제

BBox 없이 이미지만 학습시키면:

```
학습 데이터: 고양이 사진 1000장
레이블: "cat" (위치 정보 없음)

모델이 배우는 것:
→ "이 이미지 전체가 고양이다"
→ 추론 시 이미지 전체에 하나의 박스를 쳐버림
→ 혹은 이미지 내 어디에서 찾아야 할지 몰라서 잘못된 위치를 예측
```

반면 BBox가 있으면:

```
학습 데이터: 고양이 사진 1000장 + 각 이미지의 고양이 위치/크기
모델이 배우는 것:
→ "이미지의 이 좌표 범위 안에 고양이가 있다"
→ 새로운 이미지에서 정확한 위치에 박스를 그릴 수 있음
```

---

## 2. BBox가 있을 때 증강이 어떻게 달라지는가

### 2-1. "이미지 단독 증강" vs "이미지+BBox 동기화 증강"

**이미지만 있는 경우의 증강:**

```python
# 단순히 이미지만 변환
augmented_image = cv2.flip(image, 1)  # 좌우 반전
# 끝 → 어디에 객체가 있는지 정보 없음
```

**이미지 + BBox가 있는 경우의 증강:**

```python
# 이미지 변환
augmented_image = cv2.flip(image, 1)  # 좌우 반전

# BBox 좌표도 반드시 함께 변환
# 원본: x=230, y=150, w=120, h=90 (이미지 너비=640)
# 반전 후: x = 640 - 230 - 120 = 290
new_x = img_width - bbox_x - bbox_w
new_bbox = [290, 150, 120, 90]  # y, w, h는 좌우 반전이므로 그대로
```

BBox 없이 이미지만 증강하면 "증강된 이미지에서 객체가 어디 있는지"는 영원히 알 수 없습니다.

---

### 2-2. 각 증강 기법별 BBox 처리 방식

#### ① Homography (원근 변환)

```
원본 이미지:                     Homography 적용 후:
┌──────────────────┐             ┌──────────────────┐
│                  │             │                  │
│    ┌──────┐      │             │      ┌────┐      │
│    │ CAT  │      │    →→→      │      │CAT │      │
│    └──────┘      │             │       └──┘       │
│                  │             │                  │
└──────────────────┘             └──────────────────┘
bbox: [230, 150, 120, 90]        bbox: [???] ← 반드시 재계산 필요
```

**BBox 재계산 과정 (우리 코드 방식):**

```python
# 원본 bbox의 4개 꼭짓점을 동차좌표로 표현
corners = np.array([
    [x,     y,     1],  # 좌상단
    [x + w, y,     1],  # 우상단
    [x + w, y + h, 1],  # 우하단
    [x,     y + h, 1],  # 좌하단
], dtype=np.float64).T  # shape: (3, 4)

# 3×3 호모그래피 행렬 H를 4개 꼭짓점에 동시 적용
transformed = H @ corners        # shape: (3, 4)

# 동차좌표 → 유클리드 좌표 변환 (원근 나눗셈)
xs = transformed[0] / transformed[2]
ys = transformed[1] / transformed[2]

# 변환된 4점을 감싸는 새 AABB(Axis-Aligned Bounding Box) 계산
new_x = max(0, min(xs))
new_y = max(0, min(ys))
new_w = min(img_w, max(xs)) - new_x
new_h = min(img_h, max(ys)) - new_y
```

시각적으로:

```
변환 전 bbox:           변환 후 4점:            새 AABB:
┌──────┐                ╱──╲                  ┌────────┐
│      │     →→→       ╱    ╲      →→→        │ ╱──╲   │
│      │              ╲    ╱                  │╱    ╲  │
└──────┘               ╲──╱                  └────────┘
직사각형              찌그러진 사각형          새 직사각형 (AABB)
```

왜 4개 꼭짓점이 필요한가:
직사각형에 원근 변환을 가하면 평행사변형 또는 임의의 사각형이 됩니다.
단순히 중심점만 이동시키면 박스 크기가 틀려집니다.
4점을 모두 변환한 뒤 min/max로 다시 감싸야 정확합니다.

---

#### ② Mosaic (4장 합성)

```
원본 4장:                   합성된 1장:
[이미지A][이미지B]     →    ┌─────┬─────┐
[이미지C][이미지D]          │  A  │  B  │
                            │bbox↗│bbox↗│
                            ├─────┼─────┤
                            │  C  │  D  │
                            │bbox↗│bbox↗│
                            └─────┴─────┘
```

**BBox 변환이 없으면 불가능한 증강입니다.**

각 이미지의 BBox 좌표를 새 캔버스 기준으로 변환해야 합니다:

```python
# 이미지A가 좌상단 (0,0)~(cx,cy) 영역에 배치된 경우
scale_x = cx / original_w  # 가로 축소 비율
scale_y = cy / original_h  # 세로 축소 비율
offset_x = 0               # 이미지A는 좌상단에서 시작
offset_y = 0

# 원본 BBox: [x, y, w, h]
new_x = x * scale_x + offset_x
new_y = y * scale_y + offset_y
new_w = w * scale_x
new_h = h * scale_y
# clip: 새 캔버스 경계 밖으로 나간 부분 잘라냄
```

BBox 없이 이미지만 4장 합친다면: 합성 이미지 안에서 각 객체가 어디에 있는지 알 수 없습니다.

---

#### ③ Cutout (Random Erasing)

```
원본:                         Cutout 적용:
┌──────────────────┐          ┌──────────────────┐
│                  │          │          ██████  │
│    ┌──────┐      │    →     │    ┌──────┐      │
│    │ CAT  │      │          │    │ C████│      │  ← Cutout이 bbox를 침범
│    └──────┘      │          │    └──────┘      │
│                  │          │                  │
└──────────────────┘          └──────────────────┘
```

**BBox가 있기 때문에 가능한 처리:**

```python
# Cutout 영역과 BBox의 겹침 비율 계산
intersection_area = overlap(cutout_box, bbox)
bbox_area = bbox_w * bbox_h

coverage_ratio = intersection_area / bbox_area

if coverage_ratio >= 1.0:
    # BBox가 100% 가려짐 → 이 BBox 제거 (잘못된 학습 방지)
    drop this bbox
else:
    # 부분적으로 가려짐 → BBox 유지 (현실에서도 가려진 객체 탐지 필요)
    keep bbox
```

BBox가 없으면: Cutout이 객체를 완전히 가려도 "여기에 객체가 있다"는 레이블이 남아 모델이 잘못 학습됩니다.

---

#### ④ Copy-Paste (우리 코드 v2.0)

BBox만으로는 구현이 **불가능**하고, Polygon 레이블이 있어야 합니다.

```
단계 1: Polygon으로 정확한 객체 마스크 생성
┌──────────────────┐
│                  │
│  ╭──────╮        │
│  │ CAT  │        │   → 마스크(흰색=객체, 검정=배경)
│  ╰──────╯        │
│                  │
└──────────────────┘

단계 2: 마스크로 배경 제거 (누끼)
┌──────────────────┐
│  000000000000000 │
│  00╭──────╮0000 │   → 객체 픽셀만 남음
│  00│ CAT  │0000 │
│  00╰──────╯0000 │
│  000000000000000 │
└──────────────────┘

단계 3: 배경 이미지에 붙여넣기 + Polygon 좌표 이동
new_points = [[px + dx, py + dy] for px, py in original_points]
```

Polygon 좌표가 없으면 마스크를 생성할 수 없어서, Copy-Paste 자체가 성립하지 않습니다.

---

#### ⑤ Label Preserving (방향성 보호)

```
원본:              잘못된 증강:       올바른 처리:
  ↑ (위쪽 화살표)    ↓ (아래쪽!)       증강 건너뜀
  label: "up"        label: "up"       label: "up" 유지

BBox에 label_sensitive: true가 있으면 → 반전/90도 회전 차단
```

**레이블 메타데이터(label_sensitive)도 BBox의 일부입니다.**
이미지만 있으면 어떤 객체가 방향에 민감한지 알 수 없습니다.

---

### 2-3. 증강 후 BBox가 정확한지 확인하는 방법

우리 코드는 증강된 이미지를 저장할 때 **bbox를 이미지 위에 그려서** 저장합니다.

```python
# app/app.py - draw_annotations_on_image()
aug_img_drawn = draw_annotations_on_image(aug_img, aug_anns)
```

결과물:
```
project_data/images/augmented/image_uuid_aug0.jpg
→ 이 이미지를 열면 흰색 박스와 라벨이 그려져 있음
→ 박스가 객체를 잘 감싸고 있으면 변환이 정확한 것
→ 박스가 엉뚱한 곳에 있으면 변환 버그가 있는 것
```

---

## 3. 요약 정리

| 질문 | 답변 |
|------|------|
| 모델은 BBox를 어떻게 받아들이는가 | Loss 함수의 정답값(Ground Truth)으로 직접 사용. 위치 오차·크기 오차를 수치로 계산 |
| BBox 없이 이미지만 학습하면 | 분류(Classification)는 가능, 위치 검출(Detection)은 불가능 |
| Homography 시 BBox 처리 | 4개 꼭짓점을 3×3 행렬로 변환 후 AABB 재계산 |
| Mosaic 시 BBox 처리 | 스케일 비율과 오프셋을 각 이미지에 적용해 좌표 변환 |
| Cutout 시 BBox 처리 | 100% 가려진 BBox는 제거. 그래야 잘못된 학습 방지 |
| Copy-Paste 시 Polygon 처리 | Polygon 좌표 기반 마스크로 누끼 추출 후 좌표 평행이동 |
| Label Preserving | BBox의 메타데이터를 보고 방향 민감 객체는 반전/회전 차단 |

**결론:** BBox는 단순한 "표시"가 아니라, 모델 학습의 정답 데이터이자 증강 기법들이 수학적으로 작동하기 위한 기준점입니다.
