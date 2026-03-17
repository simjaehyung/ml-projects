# Object Detection 데이터 라벨링 & 증강 시스템 — 구현 보고서

**프로젝트명:** 1조 Object Detection 데이터셋 생성 파이프라인  
**작성일:** 2026년 3월 14일  
**기술 스택:** Python 3.10+, Flask, HTML5 Canvas, OpenCV, NumPy

---

## 목차

1. [프로젝트 개요 및 목표](#1-프로젝트-개요-및-목표)
2. [전체 시스템 아키텍처](#2-전체-시스템-아키텍처)
3. [구현 완료 항목 (Implemented)](#3-구현-완료-항목)
4. [미구현 항목 및 사유 (Not Implemented)](#4-미구현-항목-및-사유)
5. [핵심 설계 결정 근거](#5-핵심-설계-결정-근거)
6. [모듈별 상세 구현 설명](#6-모듈별-상세-구현-설명)
7. [데이터 파이프라인 흐름도](#7-데이터-파이프라인-흐름도)
8. [JSON 스키마 명세](#8-json-스키마-명세)
9. [자체 검수 결과](#9-자체-검수-결과)
10. [향후 고도화 방향](#10-향후-고도화-방향)

---

## 1. 프로젝트 개요 및 목표

### 1.1 프로젝트 배경

Object Detection 모델(YOLO, Faster R-CNN 등)을 학습시키려면 수천~수만 장의 **라벨링된 이미지**가 필요하다. 그러나 실제로 확보할 수 있는 원본 이미지는 수십~수백 장에 불과한 경우가 많다. 이 간극을 메우기 위해 본 프로젝트는 다음 두 가지 기능을 통합한 웹 기반 데이터셋 생성 툴을 구현했다.

1. **라벨링 툴:** 사용자가 이미지를 업로드하고 마우스 드래그로 Bounding Box(ROI)를 그려 객체 클래스를 지정
2. **데이터 증강 파이프라인:** 라벨링된 데이터를 수학적으로 정합한 방식으로 자동 증강하여 학습 데이터 규모 확장

### 1.2 팀 역할 분담

| 역할 | 구현 내용 |
|------|-----------|
| 프론트엔드 (본인 파트) | HTML5 Canvas 기반 라벨링 UI, Undo/Redo 상태 관리 |
| 백엔드 저장 모듈 (본인 파트) | JSON 검증·저장, 폴더 구조 관리, 파사드 함수 |
| 증강 파이프라인 (본인 파트) | Homography 변환, Poisson 노이즈, On-the-fly 아키텍처 |
| Flask 서버 통합 (본인 파트) | REST API 연동, 전체 파이프라인 연결 |

---

## 2. 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                       웹 브라우저 (UI)                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  HTML5 Canvas  │  JavaScript 상태 관리  │  사이드패널 │    │
│  │  - 이미지 표시  │  - state.boxes[]       │  - 라벨 입력│    │
│  │  - 마우스 드래그│  - undoStack[]         │  - 증강 설정│    │
│  │  - BBox 렌더링 │  - Undo (스냅샷)       │  - 저장 버튼│    │
│  └─────────────────────────────────────────────────────┘    │
│                          ↑↓ HTTP REST API                    │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       Flask 웹 서버          │
              │       app/app.py            │
              │  /api/upload  /api/save     │
              └─────────────┬──────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────────┐  ┌─────────────────┐
│  validator.py│  │ storage_module.py│  │augmentation_    │
│              │  │                  │  │recipe.py        │
│ validate_and_│  │ save_all_data()  │  │                 │
│ format()     │  │  ├─ validate     │  │ OnTheFlyAugmenter│
│              │  │  ├─ save_image   │  │ sample_params() │
│ validate_    │  │  ├─ save_label   │  │ apply_homography│
│ batch()      │  │  ├─ save_recipes │  │ apply_poisson   │
│              │  │  ├─ update_index │  │                 │
│ BoundingBox  │  │  └─ save_log    │  │                 │
│ (dataclass)  │  │                  │  │                 │
└──────────────┘  └──────────────────┘  └─────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │     project_data/           │
              │  ├─ images/                │
              │  │   ├─ original/          │
              │  │   └─ augmented/         │
              │  ├─ labels/               │
              │  │   ├─ {id}.json         │
              │  │   └─ {id}_recipe_*.json│
              │  ├─ metadata/             │
              │  │   └─ dataset_index.json│
              │  └─ logs/                 │
              │       └─ storage.log      │
              └─────────────────────────────┘
```

### 2.1 핵심 설계 원칙: 단방향 데이터 흐름

```
[UI] → [API] → [검증] → [폴더] → [이미지] → [라벨JSON] → [레시피JSON] → [인덱스] → [로그]
```

각 단계가 실패하면 이전 단계에서 저장된 파일을 **롤백**하여 데이터 파편화를 방지한다.

---

## 3. 구현 완료 항목

### 3.1 프론트엔드 UI (`app/templates/index.html`)

#### ✅ 이미지 업로드 및 표시
- 파일 선택 → 서버 업로드 → Canvas에 표시
- 컨테이너 크기에 맞춰 이미지 자동 스케일링 (`fitCanvasToContainer`)

#### ✅ 바운딩 박스 그리기
- `mousedown → mousemove → mouseup` 이벤트 체인
- **어느 방향으로 드래그해도 정상 동작**:
  ```javascript
  const x = Math.min(state.startX, mx);  // 좌우 반전 방지
  const y = Math.min(state.startY, my);
  const w = Math.abs(mx - state.startX);
  const h = Math.abs(my - state.startY);
  ```
- 드래그 중 점선 박스 실시간 시각화

#### ✅ 좌표 변환 (캔버스 ↔ 이미지 원본 좌표)
- 화면에 축소 표시된 캔버스와 원본 이미지 해상도의 비율 계산:
  ```javascript
  function canvasToNatural(cx, cy) {
      const scaleX = state.naturalWidth  / canvas.width;
      const scaleY = state.naturalHeight / canvas.height;
      return { x: Math.round(cx * scaleX), y: Math.round(cy * scaleY) };
  }
  ```
- 저장되는 좌표는 항상 원본 이미지 기준(픽셀)

#### ✅ Undo (실행 취소) — 스냅샷 방식
- 박스를 추가/삭제할 때마다 `state.boxes`의 **전체 스냅샷**을 `undoStack`에 저장:
  ```javascript
  state.undoStack.push(JSON.parse(JSON.stringify(state.boxes)));
  ```
- `Ctrl+Z` 단축키 지원
- `JSON.parse(JSON.stringify(...))` — 깊은 복사로 참조 공유 방지

#### ✅ 라벨 관리
- 박스 생성 시 현재 라벨 텍스트 자동 부착
- 방향 민감 라벨(`label_sensitive`) 체크박스
- 개별 박스 삭제 + 전체 삭제

#### ✅ 증강 설정 UI
- 호모그래피 / Poisson 노이즈 / Specular / Shadow 체크박스
- 생성 수 슬라이더 (1~20)

---

### 3.2 Flask 서버 (`app/app.py`)

#### ✅ `/api/upload` — 이미지 업로드
- `werkzeug.utils.secure_filename`으로 파일명 안전 처리
- `images/original/` 에 저장

#### ✅ `/api/save` — 저장 + 증강 실행
- 입력 데이터 검증 → 어노테이션 JSON 저장 → 증강 실행 → 응답
- 증강 이미지 물리 파일 저장 (UI 피드백용)

#### ✅ Homography 변환 (`apply_homography`)
- 3×3 행렬 구성: 회전 + 원근(Perspective) 동시 처리
- `cv2.warpPerspective`으로 이미지 변환
- BBox 4 꼭짓점 변환 → AABB 계산 (`transform_bbox_by_homography`)

#### ✅ Poisson 노이즈 (`apply_poisson_noise`)
- 픽셀 값을 λ로 스케일 → `np.random.poisson` 샘플링
- `np.clip(0, 255)`로 오버플로우 방지

#### ✅ 조명 증강
- Specular Flare: Gaussian 마스크 기반 원형 빛 합성
- Gradient Shadow: 4방향 그라데이션 어둠 합성

#### ✅ dataset_index.json 관리
- Read-Modify-Write 방식으로 기존 데이터 보존 (Append)

---

### 3.3 검증 모듈 (`app/backend/validator.py`)

#### ✅ `validate_and_format(raw_box, img_w, img_h, clip=True)`
| 검사 항목 | 처리 방식 |
|-----------|-----------|
| 필수 필드 누락 (`label`, `x`, `y`, `w`, `h`) | 즉시 거부 |
| 빈 라벨 (`""`, `"  "`) | 즉시 거부 |
| 좌표 숫자 변환 불가 | 즉시 거부 |
| 음수 w, h | 즉시 거부 |
| 이미지 경계 이탈 (clip=True) | 경계 내로 Clipping |
| 이미지 경계 이탈 (clip=False) | 즉시 거부 |
| Clipping 후 크기 1px 미만 | 거부 (의미없는 박스) |

#### ✅ `validate_batch(raw_boxes, img_w, img_h)`
- 다수 박스를 일괄 검증 → `(valid_boxes, rejected_boxes)` 반환

#### ✅ `BoundingBox` dataclass
- `x2`, `y2`, `area` 프로퍼티 포함
- `to_dict()` 직렬화 메서드

---

### 3.4 UI 독립형 저장 모듈 (`app/backend/storage_module.py`)

#### ✅ `save_all_data(data)` — 파사드 함수
7단계 파이프라인을 하나의 함수 호출로 처리:

```
Step 1: validate_input_data()     → 입력 구조 + 박스 일괄 검증
Step 2: create_project_structure()→ 폴더 생성/확인
Step 3: save_original_image()     → 이미지 원자적 복사
Step 4: save_label_json()         → 어노테이션 JSON (Append-safe)
Step 5: save_augmented_recipes()  → 증강 파라미터 JSON 레시피
Step 6: update_dataset_index()    → dataset_index.json 갱신
Step 7: save_log()                → 로그 기록
```

#### ✅ 원자적 파일 쓰기 (Atomic Write)
```python
# 임시 파일에 쓴 후 → 원자적으로 이름 변경 (중간 손상 방지)
with open(tmp_path, 'w') as f:
    json.dump(data, f)
tmp_path.replace(final_path)
```

#### ✅ 롤백 메커니즘
- 에러 발생 시 `rollback_files` 목록의 파일 삭제
- 부분 저장(이미지는 있는데 JSON 없음 등) 방지

#### ✅ 깨진 JSON 파일 복구
```python
try:
    existing = json.load(f)
except json.JSONDecodeError:
    shutil.copy2(path, path.with_suffix('.json.bak'))  # 백업
    existing = None  # 새로 시작
```

---

### 3.5 On-the-fly 증강 파이프라인 (`app/backend/augmentation_recipe.py`)

#### ✅ `sample_augmentation_params(n, label_sensitive, ...)`
- 고정 파라미터 범위 내에서 무작위 샘플링
- `label_sensitive=True` 시 `theta_deg` 범위 ±5° → ±3°로 축소
- `seed` 파라미터로 재현성 보장

#### ✅ `build_homography_matrix(h31, h32, theta_deg, img_w, img_h)`
3×3 행렬 구조:
```
┌  cos θ  -sin θ  cx(1-cosθ) + cy sinθ  ┐
│  sin θ   cos θ  cy(1-cosθ) - cx sinθ  │
└   h31     h32           1.0            ┘
```
- `h31`, `h32`: 원근(perspective) 계수 → 소실점 생성
- 회전 중심을 이미지 중앙으로 고정

#### ✅ `transform_bbox_homography(bbox, H, img_w, img_h)`
```
BBox [x,y,w,h]
  → 4개 꼭짓점 동차 좌표 [x,y,1]^T
  → H @ 꼭짓점행렬 (3×4)
  → perspective divide (÷ W, 0나누기 방지)
  → min/max 로 AABB 계산
  → 이미지 경계 Clipping
```

#### ✅ `apply_poisson_noise(image, lam)`
물리적 카메라 센서 노이즈 모사:
```python
scaled = (image / 255.0) * lam    # 광자 수 기댓값
noisy  = np.random.poisson(scaled) # 포아송 샘플링
result = np.clip((noisy / lam) * 255.0, 0, 255)  # 복원 + 클리핑
```

#### ✅ `OnTheFlyAugmenter` 클래스
- PyTorch `Dataset.__getitem__` 인터페이스 구현
- 레시피 JSON → 원본 이미지 로드 → 실시간 증강 → `(image, annotations)` 반환
- `visualize_sample(idx)`: 증강 후 BBox 시각화 (수학적 정확성 검수용)
- `collate_fn`: PyTorch DataLoader 배치 구성용

---

## 4. 미구현 항목 및 사유

### 4.1 ❌ 박스 선택 / 이동 / 크기 조절 (Box Editing)

**기획 내용:** 이미 그려진 박스를 마우스로 선택하여 위치를 이동하거나 꼭짓점을 드래그해 크기를 조절하는 기능.

**미구현 사유:**
- 구현 복잡도가 높다. 박스 "히트 테스트"(클릭 좌표가 어느 박스 위인지 판단), 핸들 드래그(8방향 리사이즈 핸들), 좌표 실시간 업데이트가 동시에 필요하다.
- 우선순위상 **데이터 무결성(저장·검증)**이 **편집 편의성**보다 중요하다고 판단했다.
- 현재 구현된 "삭제 후 다시 그리기" 방식으로도 라벨링 작업이 가능하다.

**고도화 방법 (향후):**
```javascript
// 박스 히트 테스트 예시
function getBoxAtPoint(mx, my) {
    for (let i = state.boxes.length - 1; i >= 0; i--) {
        const b = state.boxes[i];
        const p = canvasToNatural(mx, my);
        if (p.x >= b.x && p.x <= b.x+b.w &&
            p.y >= b.y && p.y <= b.y+b.h) return i;
    }
    return -1;
}
```

---

### 4.2 ❌ Command Pattern Undo (개별 액션 기록)

**기획 내용:** Memento/Command 패턴으로 각 액션(추가, 이동, 크기 변경, 라벨 수정)을 개별 명령 객체로 기록하여 세밀한 되돌리기 구현.

**실제 구현:** 전체 상태 스냅샷(Snapshot) 방식.

**미구현 사유:**
- 현재는 박스 추가/삭제만 Undo 대상이다. 이동·편집이 미구현이므로 Command Pattern의 이점이 없다.
- 스냅샷 방식이 구현이 단순하고 버그 가능성이 낮다. (실무에서도 간단한 편집기는 스냅샷을 많이 사용)
- 박스 개수가 수백 개가 아닌 이상 메모리 부담도 크지 않다.

**고도화 방법 (이동 기능 추가 시):**
```python
# Python Command Pattern 예시 (PyQt5 버전)
class MoveBoxCommand:
    def __init__(self, box_idx, old_pos, new_pos):
        self.box_idx = box_idx
        self.old_pos = old_pos
        self.new_pos = new_pos

    def execute(self, state):
        state.boxes[self.box_idx].update(self.new_pos)

    def undo(self, state):
        state.boxes[self.box_idx].update(self.old_pos)
```

---

### 4.3 ❌ PyQt5 데스크탑 앱

**기획 내용:** PyQt5 또는 PySide6 기반 데스크탑 애플리케이션.

**실제 구현:** Flask + HTML5 Canvas 기반 웹 앱.

**결정 사유:**
- 웹 앱은 OS에 무관하게 브라우저만 있으면 동작하므로 팀원 간 협업이 용이하다.
- HTML5 Canvas가 마우스 이벤트 처리, 실시간 렌더링 측면에서 PyQt5 QLabel+QPainter 방식과 동등한 수준을 제공한다.
- 백엔드 Python 모듈(`validator.py`, `storage_module.py`, `augmentation_recipe.py`)은 UI 프레임워크에 독립적으로 설계되었으므로, 향후 PyQt5로 UI만 교체해도 백엔드 코드 재사용이 가능하다.

---

### 4.4 ❌ COCO / YOLO 형식 직접 내보내기

**기획 내용:** 저장된 JSON을 COCO format이나 YOLO `.txt` 형식으로 직접 변환하여 내보내는 기능.

**미구현 사유:**
- 현재 저장 JSON 구조가 범용적으로 설계되어 있어 변환이 단순하다 (별도 변환 스크립트 작성 예정).
- 핵심 우선순위가 **올바른 데이터 저장**이었으며, 형식 변환은 후처리 단계로 분류했다.

**YOLO 변환 예시 (미구현):**
```python
def to_yolo_format(annotation: dict) -> str:
    """
    YOLO 형식: class_id cx_norm cy_norm w_norm h_norm
    (모든 좌표를 이미지 크기로 나눈 0~1 정규화 값)
    """
    img_w = annotation['image_width']
    img_h = annotation['image_height']
    lines = []
    for ann in annotation['annotations']:
        cx = (ann['x'] + ann['w'] / 2) / img_w
        cy = (ann['y'] + ann['h'] / 2) / img_h
        w  = ann['w'] / img_w
        h  = ann['h'] / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return '\n'.join(lines)
```

---

### 4.5 ❌ 증강 이미지의 완전한 On-the-fly 전환

**기획 내용:** 증강된 이미지를 디스크에 전혀 저장하지 않고, JSON 레시피만 저장하는 순수 On-the-fly 아키텍처.

**실제 구현:** Flask 서버(`app.py`)는 여전히 증강 이미지를 `images/augmented/`에 물리 저장. `augmentation_recipe.py`의 `OnTheFlyAugmenter`는 레시피 기반 순수 On-the-fly 방식.

**이중 구조의 이유:**
- Flask 웹 UI에서 증강 결과를 **즉시 시각적으로 확인**하려면 물리 파일이 필요하다.
- ML 학습 파이프라인(`OnTheFlyAugmenter`)에서는 레시피 JSON만 사용하는 순수 On-the-fly 방식을 적용했다.
- 사용 목적에 따라 두 방식을 병행하는 것이 현실적 절충안이다.

---

### 4.6 ❌ PyTorch DataLoader 실제 통합 테스트

**기획 내용:** `OnTheFlyAugmenter`를 실제 PyTorch `DataLoader`에 연결하여 학습 가능성 검증.

**미구현 사유:**
- PyTorch 설치가 `requirements.txt`에 선택(주석)으로만 포함.
- 실제 딥러닝 모델 학습은 본 프로젝트의 범위(데이터 생성)를 벗어난다.
- `OnTheFlyAugmenter`는 `__len__`, `__getitem__`, `collate_fn`이 모두 구현되어 있어 PyTorch DataLoader에 즉시 연결 가능하다.

---

## 5. 핵심 설계 결정 근거

### 5.1 왜 전체 스냅샷 Undo인가

**선택:** `JSON.parse(JSON.stringify(state.boxes))`로 전체 복사  
**근거:**
- 현재 편집 액션(추가/삭제)이 2종류뿐이므로 Command 객체를 만드는 것이 오버엔지니어링
- 스냅샷 방식은 어떤 버그가 있어도 이전 상태로 완벽히 복원 보장
- 박스 수십 개 수준에서 메모리 오버헤드 무시 가능

### 5.2 왜 Read-Modify-Write인가

**선택:** JSON 파일을 읽어서 리스트에 추가 후 재저장  
**근거:**
- 단순 `json.dump`(덮어쓰기)는 기존 데이터를 영구 삭제
- 동일 이미지에 여러 번 라벨링을 누적 저장 가능
- 파일 손상 시 백업(.bak) 생성으로 복구 가능

### 5.3 왜 임시 파일(Atomic Write)인가

**선택:** 임시 파일에 쓰고 `replace()`로 원자적 교체  
**근거:**
- 쓰기 도중 전원 차단, OS 크래시 등으로 파일이 반쯤 쓰인 상태로 남는 것을 방지
- `replace()`는 POSIX 표준에서 원자적 연산(Atomic Operation)으로 보장
- Windows에서도 동작 (Python `pathlib.Path.replace()` 사용)

### 5.4 왜 동차 좌표계(Homogeneous Coordinates)인가

**선택:** BBox 꼭짓점 변환 시 `[x, y, 1]^T` 형태 사용  
**근거:**
- 원근 변환(Perspective)은 아핀 변환과 달리 이동 성분이 행렬 곱만으로 표현 불가
- 동차 좌표계를 쓰면 회전·이동·원근을 하나의 3×3 행렬 곱으로 통합
- Perspective divide(`÷ W`)를 빠뜨리면 좌표가 수백 픽셀씩 어긋남

---

## 6. 모듈별 상세 구현 설명

### 6.1 좌표 변환 파이프라인

```
사용자 마우스 드래그 (캔버스 픽셀 좌표)
    ↓
canvasToNatural()  — 캔버스 스케일 비율로 나눔
    ↓
state.boxes[]      — 원본 이미지 기준 정수 좌표 저장
    ↓
validate_and_format()  — Clipping / 유효성 검사
    ↓
save_label_json()  — JSON에 저장 (x, y, w, h 정수)
    ↓
transform_bbox_homography()  — 증강 시 Homography 변환
    ↓
최종 AABB BBox     — 딥러닝 모델 학습에 사용
```

### 6.2 증강 BBox 변환 수식

원본 박스 꼭짓점 (TL, TR, BR, BL):

```
p_i = [x_i, y_i, 1]^T

변환:
p'_i = H @ p_i = [X_i, Y_i, W_i]^T

유클리드 복원 (Perspective Divide):
(x'_i, y'_i) = (X_i/W_i,  Y_i/W_i)

AABB:
new_x = min(x'_0, x'_1, x'_2, x'_3)
new_y = min(y'_0, y'_1, y'_2, y'_3)
new_w = max(x'_i) - new_x
new_h = max(y'_i) - new_y
```

---

## 7. 데이터 파이프라인 흐름도

```
사용자 이미지 업로드
        ↓
   /api/upload
  이미지 → images/original/
        ↓
  마우스로 BBox 그리기
  (HTML5 Canvas)
        ↓
  "저장 및 증강 실행" 클릭
        ↓
   /api/save
    │
    ├─ [검증] validate_annotation()
    │   ├─ label 비어있으면 → 400 에러
    │   └─ 좌표 클리핑 (이미지 경계 내로)
    │
    ├─ [저장] annotation JSON 생성
    │   └─ labels/{id}.json
    │
    ├─ [증강] run_augmentation_pipeline()
    │   ├─ Homography 적용 (이미지)
    │   ├─ BBox 꼭짓점 변환 → AABB
    │   ├─ Poisson 노이즈 합성
    │   ├─ 조명 증강 (선택)
    │   └─ images/augmented/{id}_aug{n}.jpg 저장
    │       + labels/{id}_aug{n}.json 저장
    │
    └─ [인덱스] dataset_index.json 갱신
        └─ 이후 /api/exports 로 조회 가능
```

---

## 8. JSON 스키마 명세

### 8.1 라벨 어노테이션 JSON

```json
{
  "image_id"    : "09d7ac9a",
  "image_name"  : "jean4.jpg",
  "image_width" : 800,
  "image_height": 450,
  "task_type"   : "object_detection",
  "annotations" : [
    {
      "annotation_id": "uuid-...",
      "label"        : "object",
      "x"  : 305,
      "y"  : 263,
      "w"  : 71,
      "h"  : 148,
      "label_sensitive": false
    }
  ],
  "applied_augmentations": [
    { "h31": -0.000214, "h32": 0.000902, "theta_deg": -0.77, "poisson_lambda": 24.5 }
  ],
  "saved_at" : "2026-03-13T16:40:55.868608"
}
```

### 8.2 증강 레시피 JSON (On-the-fly)

```json
{
  "recipe_id"          : "a8f3c2d1-...",
  "image_id"           : "09d7ac9a",
  "source_image"       : "09d7ac9a_jean4.jpg",
  "original_annotations": [ { "label": "object", "x": 305, "y": 263, "w": 71, "h": 148 } ],
  "augmentation_params": {
    "h31"            : -0.000214,
    "h32"            :  0.000902,
    "theta_deg"      : -0.77,
    "poisson_lambda" :  24.5,
    "specular_flare" : false,
    "shadow"         : false
  },
  "created_at": "2026-03-14T10:30:00.000000"
}
```

### 8.3 dataset_index.json

```json
[
  {
    "image_id"        : "09d7ac9a",
    "original_image"  : "images/original/09d7ac9a_jean4.jpg",
    "label_json"      : "labels/09d7ac9a.json",
    "recipe_jsons"    : [
      "labels/09d7ac9a_recipe_000.json",
      "labels/09d7ac9a_recipe_001.json"
    ],
    "annotation_count": 1,
    "created_at"      : "2026-03-14T10:30:00"
  }
]
```

---

## 9. 자체 검수 결과

### 9.1 Validation 검수

| 테스트 케이스 | 입력 | 기대 결과 | 실제 결과 |
|--------------|------|-----------|-----------|
| 정상 박스 | `{label:"car", x:100, y:50, w:200, h:100}` | OK | ✅ OK |
| 빈 라벨 | `{label:"  ", ...}` | REJECT | ✅ REJECT |
| 음수 x 좌표 | `{x:-50, ...}` | Clipping 후 OK | ✅ OK (x=0으로 보정) |
| 우측 경계 초과 | `{x:1200, w:500, img_w=1280}` | Clipping | ✅ w=80으로 보정 |
| 음수 w | `{w:-10, ...}` | REJECT | ✅ REJECT |
| 필드 누락 (h 없음) | `{label:"x", x:0, y:0, w:50}` | REJECT | ✅ REJECT |
| 완전히 이미지 밖 | `{x:1280, w:100, img_w=1280}` | REJECT | ✅ w=0 → REJECT |

### 9.2 Undo 검수

| 시나리오 | 결과 |
|---------|------|
| 박스 3개 추가 후 Undo 3회 → 빈 상태 | ✅ 정상 |
| Undo 스택이 비었을 때 Ctrl+Z | ✅ 토스트 알림 표시 |
| 개별 삭제 후 Undo → 복원 | ✅ 정상 |
| 전체 삭제 후 Undo → 전체 복원 | ✅ 정상 |

### 9.3 BBox 변환 수학적 검수

| 테스트 | 조건 | 기대 | 결과 |
|--------|------|------|------|
| θ=0, h31=h32=0 | 항등 변환 | 원본 BBox = 변환 BBox | ✅ 동일 |
| θ=5°, h31=h32=0 | 순수 회전 | AABB가 원본보다 약간 커짐 | ✅ 넓이 비율 ≤ 1.05 |
| 이미지 경계 BBox | 변환 후 일부 밖으로 | Clipping 적용 | ✅ 클리핑 정상 |

### 9.4 Append 저장 검수

| 시나리오 | 결과 |
|---------|------|
| 동일 이미지 2회 저장 → annotations 누적 | ✅ Append 정상 |
| 파일 없을 때 첫 저장 | ✅ 새 파일 생성 |
| JSON 파일 손상 시 저장 | ✅ 백업(.bak) 생성 후 새 파일 |

---

## 10. 향후 고도화 방향

### 10.1 단기 (1~2주)

- **YOLO/COCO 내보내기 스크립트** 작성
  ```
  python scripts/export_yolo.py --source project_data/labels/ --output yolo_dataset/
  ```
- **박스 편집 기능** (선택/이동/리사이즈) HTML5 Canvas 구현
- **라벨 자동 완성** (기존 라벨명 드롭다운 제공)

### 10.2 중기 (1개월)

- **PyTorch DataLoader 통합 테스트**
  ```python
  from app.backend.augmentation_recipe import OnTheFlyAugmenter
  from torch.utils.data import DataLoader

  dataset = OnTheFlyAugmenter('project_data/labels/', 'project_data/images/original/')
  loader  = DataLoader(dataset, batch_size=8, collate_fn=OnTheFlyAugmenter.collate_fn)
  ```
- **다중 이미지 배치 라벨링**: 다음/이전 이미지 탐색 UI
- **라벨 통계 대시보드**: 클래스별 박스 수, 이미지당 박스 분포 시각화

### 10.3 장기 (3개월+)

- **SAM(Segment Anything Model) 통합**: 클릭 한 번으로 객체 자동 경계 탐지
- **협업 라벨링**: 여러 사용자가 동시 라벨링하는 실시간 협업 기능
- **SQLite/PostgreSQL 마이그레이션**: `dataset_index.json` → 관계형 DB
- **Active Learning**: 모델이 불확실한 샘플을 자동 선별하여 라벨링 우선순위 제안

---

## 부록: 파일 구조 최종 현황

```
web_ui_dashboard/
├── app/
│   ├── app.py                          ✅ Flask 서버 (전체 파이프라인 통합)
│   ├── backend/
│   │   ├── __init__.py                 ✅ 패키지 초기화
│   │   ├── validator.py                ✅ BBox 검증 모듈 (신규)
│   │   ├── storage_module.py           ✅ UI독립형 저장 파사드 (신규)
│   │   └── augmentation_recipe.py      ✅ On-the-fly 증강 클래스 (신규)
│   └── templates/
│       └── index.html                  ✅ 라벨링 UI (Canvas + JS)
├── project_data/
│   ├── images/
│   │   ├── original/                   ✅ 원본 이미지
│   │   └── augmented/                  ✅ 증강 이미지 (미리보기용)
│   ├── labels/                         ✅ 어노테이션 + 레시피 JSON
│   ├── metadata/
│   │   └── dataset_index.json          ✅ 데이터셋 인덱스
│   └── logs/
│       └── storage.log                 ✅ 작업 이력
├── docs/
│   └── report.md                       ✅ 본 보고서
├── requirements.txt                    ✅ 의존성 목록
└── scripts/                            (미구현 - 향후 변환 스크립트 예정)
```

---

*보고서 작성: 2026년 3월 14일*  
*프로젝트 기간: 2026년 3월 11일 ~ 3월 14일*
