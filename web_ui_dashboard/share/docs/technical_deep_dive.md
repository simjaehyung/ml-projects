# 🔬 Object Detection 백엔드 모듈 — 초상세 기술 설명서

**작성자:** AI Assistant  
**작성일:** 2026년 3월 14일  
**대상:** 프로젝트 팀원 및 향후 유지보수자  
**목표:** 각 함수의 로직을 라인 단위로 이해할 수 있도록 설명

---

## 목차

1. [validator.py — 데이터 검증 모듈](#1-validatorpy--데이터-검증-모듈)
2. [storage_module.py — UI 독립형 저장 시스템](#2-storage_modulepy--ui-독립형-저장-시스템)
3. [augmentation_recipe.py — On-the-fly 증강 파이프라인](#3-augmentation_recipepy--on-the-fly-증강-파이프라인)
4. [데이터 흐름 통합 이해](#4-데이터-흐름-통합-이해)

---

## 1. validator.py — 데이터 검증 모듈

### 1.1 BoundingBox Dataclass

```python
@dataclass
class BoundingBox:
    label: str
    x: int
    y: int
    w: int
    h: int
    label_sensitive: bool = field(default=False)
```

#### 🎯 왜 dataclass를 썼나?

**선택지 1: 일반 dict**
```python
box = {'label': 'car', 'x': 100, 'y': 50, 'w': 200, 'h': 100}
# 장점: 자유로움
# 단점: 필드명 오타 시 에러 안 나고, IDE 자동완성 불가
```

**선택지 2: 일반 class**
```python
class BoundingBox:
    def __init__(self, label, x, y, w, h, label_sensitive=False):
        self.label = label
        # ... 반복되는 __init__ 코드
```

**선택지 3: dataclass ✅**
```python
@dataclass
class BoundingBox:  # 자동으로 __init__, __repr__, __eq__ 생성
    label: str
    x: int
    # ...
```

#### 💡 이점
1. **타입 안정성**: IDE가 `bbox.label` 입력시 자동 완성
2. **자동 생성**: `__init__`, `__repr__`, `__eq__` 자동 생성
3. **가독성**: 필드와 타입이 명확함
4. **JSON 변환**: `asdict()` 한 줄로 dict 변환

#### 📝 프로퍼티 설명

```python
@property
def x2(self) -> int:
    """우측 끝 좌표"""
    return self.x + self.w

@property
def y2(self) -> int:
    """하단 끝 좌표"""
    return self.y + self.h

@property
def area(self) -> int:
    """박스 넓이 (픽셀^2)"""
    return self.w * self.h
```

**사용 예:**
```python
box = BoundingBox(label='car', x=100, y=50, w=200, h=100)
print(box.x2)    # 100 + 200 = 300
print(box.y2)    # 50 + 100 = 150
print(box.area)  # 200 * 100 = 20000
```

---

### 1.2 validate_and_format() — 핵심 검증 함수

#### 함수 시그니처

```python
def validate_and_format(
    raw_box: dict,
    img_w: int,
    img_h: int,
    clip: bool = True
) -> Tuple[Optional[dict], Optional[str]]:
```

#### 입력값 설명

| 파라미터 | 타입 | 예시 | 설명 |
|---------|------|------|------|
| `raw_box` | dict | `{'label': 'car', 'x': 100, ...}` | 프론트엔드에서 받은 원시 데이터 |
| `img_w` | int | `1280` | 이미지 너비(픽셀) |
| `img_h` | int | `720` | 이미지 높이(픽셀) |
| `clip` | bool | `True` | True = 경계 내로 자르기, False = 범위 이탈 시 거부 |

#### 반환값

| 경우 | 반환값 |
|------|--------|
| 성공 | `({'label': 'car', 'x': 100, 'y': 50, 'w': 200, 'h': 100, 'label_sensitive': False}, None)` |
| 실패 | `(None, "error message")` |

#### 🔍 검증 단계별 상세 분석

##### **Step 1: 필수 필드 존재 확인**

```python
required_keys = ('label', 'x', 'y', 'w', 'h')
for key in required_keys:
    if key not in raw_box:
        return None, f"필수 필드 누락: '{key}'"
```

**로직:**
- 5개 필수 키를 순회하면서 하나라도 없으면 즉시 거부
- 예: `{'label': 'car', 'x': 100}`이 들어오면 `y, w, h` 누락 → 거부

**왜 이 단계가 첫 번째인가?**
→ 필드가 없으면 이후 단계에서 `KeyError` 발생. 미리 차단함.

---

##### **Step 2: 라벨 검증**

```python
label = raw_box['label']
if not isinstance(label, str):
    return None, f"label은 문자열이어야 합니다. (입력값: {type(label).__name__})"
label = label.strip()  # 공백 제거
if not label:
    return None, "label이 빈 문자열입니다. 저장을 거부합니다."
```

**상세 분석:**

| 입력 | 처리 |
|------|------|
| `label: 'car'` | ✅ 그대로 사용 |
| `label: '  car  '` | ✅ `'car'`로 정규화 |
| `label: '  '` (공백만) | ❌ `strip()` 후 빈 문자열 → 거부 |
| `label: ''` (빈 문자열) | ❌ 바로 거부 |
| `label: None` | ❌ `isinstance` 실패 → 거부 |
| `label: 123` (숫자) | ❌ `isinstance` 실패 → 거부 |

**왜 `strip()`을 쓰는가?**
→ 사용자가 실수로 공백을 포함해도 자동 보정. 예: 복사-붙여넣기 시 앞뒤 공백 포함되는 경우 방지.

---

##### **Step 3: 좌표 숫자 변환**

```python
try:
    x = float(raw_box['x'])
    y = float(raw_box['y'])
    w = float(raw_box['w'])
    h = float(raw_box['h'])
except (TypeError, ValueError) as e:
    return None, f"좌표 값을 숫자로 변환할 수 없습니다: {e}"
```

**왜 `float`로 변환하나?**

프론트엔드(JavaScript) → 백엔드(Python)로 데이터 이동 시:

| 상황 | 입력값 타입 | 예시 |
|------|-----------|------|
| JSON 파싱 후 | int 또는 float | `100`, `100.5` |
| 소수점 포함 (좌표 계산) | float | `100.49999...` |
| 문자열 실수 | str | `"100"` ← float()로 변환 필요 |

```python
# 예시: JSON에서 오는 데이터
import json
data = json.loads('{"x": 100.5}')  # JSON 파싱
x = float(data['x'])  # 100.5
```

**예외 처리:**
```python
try:
    x = float("abc")  # TypeError 발생
except (TypeError, ValueError):
    # 문자 "abc"를 float로 변환 불가 → 거부
```

---

##### **Step 4: 크기 양수 확인 (클리핑 전)**

```python
if w <= 0:
    return None, f"w(너비)는 양수여야 합니다. (입력값: {w})"
if h <= 0:
    return None, f"h(높이)는 양수여야 합니다. (입력값: {h})"
```

**왜 0을 거부하는가?**

| 크기 | 의미 | 처리 |
|------|------|------|
| `w=200, h=100` | 정상 박스 | ✅ 계속 진행 |
| `w=0, h=100` | 너비가 없는 선 | ❌ 거부 |
| `w=-50, h=100` | 음수 (불가능) | ❌ 거부 |

**왜 음수가 나올까?**
→ 프론트엔드에서 드래그 방향이 반대일 때. 예:
```javascript
// 우측 상단 → 좌측 하단으로 드래그
const w = Math.abs(mx - state.startX);  // abs()로 이미 처리됨
// 하지만 검증에서 한 번 더 확인 (방어 코드)
```

---

##### **Step 5: 이미지 해상도 유효성**

```python
if img_w <= 0 or img_h <= 0:
    return None, f"이미지 해상도가 유효하지 않습니다. (img_w={img_w}, img_h={img_h})"
```

**왜 필요한가?**
→ 이후 Clipping 계산에서 `img_w`, `img_h`로 나누기 때문. 0이면 오류 발생.

---

##### **Step 6: 경계 이탈 처리 (Clipping vs Reject)**

##### Case A: `clip=True` (경계 내로 자르기)

```python
if clip:
    x = max(0.0, min(x, float(img_w - 1)))
    y = max(0.0, min(y, float(img_h - 1)))
    w = max(0.0, min(w, float(img_w) - x))
    h = max(0.0, min(h, float(img_h) - y))
```

**상세 분석:**

```
x = max(0.0, min(x, float(img_w - 1)))
    ├─ min(x, img_w - 1)     : x를 img_w-1 이상으로 제한
    └─ max(0.0, ...)          : 결과를 0 이상으로 제한
    => x가 [0, img_w-1] 범위로 보정됨
```

**예시 (img_w=1280):**

| x 입력 | 중간 | 최종 |
|--------|------|------|
| -50 | min(-50, 1279) = -50 | max(0, -50) = **0** |
| 100 | min(100, 1279) = 100 | max(0, 100) = **100** |
| 1500 | min(1500, 1279) = 1279 | max(0, 1279) = **1279** |

**너비(w) Clipping:**

```
w = max(0.0, min(w, float(img_w) - x))
```

원본 박스가 우측으로 삐져나갔을 때:

| x | w | x+w | 결과 w |
|---|---|-----|---------|
| 1200 | 100 | 1300 | min(100, 1280-1200) = **80** |
| 100 | 200 | 300 | min(200, 1280-100) = **200** |

**왜 `img_w - 1`을 쓰나?**
→ 이미지 픽셀 인덱스: 0 ~ (width-1)
→ 1280×720 이미지: x는 0~1279, y는 0~719

---

##### Case B: `clip=False` (범위 이탈 시 거부)

```python
else:
    if x < 0 or y < 0:
        return None, f"x, y 좌표는 음수일 수 없습니다. (x={x}, y={y})"
    if x + w > img_w or y + h > img_h:
        return None, (
            f"박스가 이미지 범위({img_w}×{img_h})를 벗어납니다. "
            f"(x2={x+w}, y2={y+h})"
        )
```

**어떤 경우에 `clip=False`를 쓰는가?**
→ 데이터 품질 검증이 목표일 때 (클리핑하지 말고 잘못된 데이터를 명확히 드러내기)

---

##### **Step 7: 클리핑 후 크기 재검사**

```python
if w < 1.0:
    return None, f"클리핑 후 너비(w)가 1px 미만입니다. (w={w:.2f})"
if h < 1.0:
    return None, f"클리핑 후 높이(h)가 1px 미만입니다. (h={h:.2f})"
```

**왜 이 단계가 필요한가?**

예시 (img_w=1280):
```
원본 x=1280, w=100 → Clipping 후
x = max(0, min(1280, 1279)) = 1279
w = max(0, min(100, 1280-1279)) = max(0, 1) = 1
=> 1×1 픽셀 박스

아래는 Clipping 후 완전히 사라지는 경우:
x=1280, w=50 → Clipping 후
w = max(0, min(50, 1280-1280)) = max(0, 0) = 0
=> 거부!
```

**의미 있는 최소 크기:**
→ 1×1 픽셀 박스는 의미 없음 (쌀알만한 크기)
→ 거부하여 의미 있는 박스만 저장

---

##### **Step 8: 최종 표준화 포맷 반환**

```python
return {
    'label': label,
    'x': round(int(x)),
    'y': round(int(y)),
    'w': round(int(w)),
    'h': round(int(h)),
    'label_sensitive': bool(raw_box.get('label_sensitive', False))
}, None
```

**타입 변환:**
- `float` → `int` → `round()` : 부동소수점 오차 제거
- 예: `x=100.49999999` → `int(100)` → `round(100)` = `100`

**label_sensitive 안전한 추출:**
```python
bool(raw_box.get('label_sensitive', False))
    ├─ .get('label_sensitive', False)  : 없으면 False
    └─ bool(...)                        : 명시적 bool 변환
    => True/False 중 하나 보장
```

---

### 1.3 validate_batch() — 배치 검증

```python
def validate_batch(
    raw_boxes: list,
    img_w: int,
    img_h: int,
    clip: bool = True
) -> Tuple[List[dict], List[dict]]:
```

#### 로직

```python
valid_boxes = []
rejected_boxes = []

for i, raw in enumerate(raw_boxes):
    formatted, error = validate_and_format(raw, img_w, img_h, clip=clip)
    if formatted is not None:
        valid_boxes.append(formatted)  # 성공
    else:
        rejected_boxes.append({
            'index': i,
            'reason': error,
            'raw': raw
        })  # 실패 정보 저장

return valid_boxes, rejected_boxes
```

#### 예시

**입력:**
```python
raw_boxes = [
    {'label': 'car', 'x': 100, 'y': 50, 'w': 200, 'h': 100},     # OK
    {'label': '', 'x': 300, 'y': 100, 'w': 50, 'h': 50},          # 빈 라벨
    {'label': 'person', 'x': -10, 'y': 200, 'w': 80, 'h': 120},  # 음수 좌표 (Clipping)
]
img_w, img_h = 1280, 720
```

**출력:**
```python
valid_boxes = [
    {'label': 'car', 'x': 100, 'y': 50, 'w': 200, 'h': 100, 'label_sensitive': False},
    {'label': 'person', 'x': 0, 'y': 200, 'w': 80, 'h': 120, 'label_sensitive': False},  # x=-10 → 0
]

rejected_boxes = [
    {
        'index': 1,
        'reason': "label이 빈 문자열입니다. 저장을 거부합니다.",
        'raw': {'label': '', 'x': 300, 'y': 100, 'w': 50, 'h': 50}
    }
]
```

---

## 2. storage_module.py — UI 독립형 저장 시스템

### 2.1 전체 아키텍처

```
프론트엔드 데이터
    ↓
save_all_data()  ← 파사드 함수 (진입점)
    ├─ Step 1: validate_input_data()      [검증]
    ├─ Step 2: create_project_structure()  [폴더]
    ├─ Step 3: save_original_image()       [이미지]
    ├─ Step 4: save_label_json()           [라벨]
    ├─ Step 5: save_augmented_recipes()    [레시피]
    ├─ Step 6: update_dataset_index()      [인덱스]
    └─ Step 7: save_log()                  [로그]
    ↓
백엔드 완료
```

**핵심 원칙: 각 단계가 실패하면 이전 단계 파일을 모두 롤백**

---

### 2.2 validate_input_data() — 최상위 검증

```python
def validate_input_data(data: dict) -> Tuple[bool, str, dict]:
```

#### Step 1: 최상위 키 확인

```python
required = ('image_path', 'image_name', 'image_width', 'image_height', 'objects')
for key in required:
    if key not in data:
        return False, f"필수 키 누락: '{key}'", {}
```

**입력 데이터 형식 예시:**
```python
data = {
    'image_path'  : '/home/user/dataset/sample.jpg',  # ← 절대 경로
    'image_name'  : 'sample.jpg',                      # ← 파일명만
    'image_width' : 1280,
    'image_height': 720,
    'objects'     : [
        {'label': 'car', 'x': 100, 'y': 50, 'w': 200, 'h': 100},
        ...
    ],
    'augmentations': [...]  # 선택
}
```

---

#### Step 2: 이미지 파일 존재 확인

```python
image_path = Path(data['image_path'])
if not image_path.exists():
    return False, f"이미지 파일을 찾을 수 없습니다: {image_path}", {}
```

**왜 필요한가?**
→ 저장 진행 전에 원본 파일이 실제로 있는지 확인
→ "파일 없음" 오류로 중간에 작업이 실패하는 것을 방지

---

#### Step 3: 이미지 해상도 검증

```python
img_w = data['image_width']
img_h = data['image_height']
if not isinstance(img_w, int) or not isinstance(img_h, int):
    return False, "image_width, image_height는 정수여야 합니다.", {}
if img_w <= 0 or img_h <= 0:
    return False, f"이미지 해상도가 유효하지 않습니다. ({img_w}×{img_h})", {}
```

**타입 체크:**
```
isinstance(1280, int) = True
isinstance(1280.0, int) = False  ← float는 거부 (정수로 변환 불가)
isinstance("1280", int) = False  ← str는 거부
```

---

#### Step 4: 바운딩 박스 일괄 검증

```python
objects = data.get('objects', [])
if not isinstance(objects, list):
    return False, "'objects'는 리스트여야 합니다.", {}

# validator.py의 validate_batch() 호출
valid_boxes, rejected = validate_batch(objects, img_w, img_h, clip=True)
```

**반환:**
```python
return True, "OK", {
    'image_path'   : str(image_path),
    'image_name'   : str(data['image_name']),
    'image_width'  : img_w,
    'image_height' : img_h,
    'valid_boxes'  : valid_boxes,      # ✅ 통과한 박스
    'rejected_boxes': rejected,         # ❌ 거부된 박스 목록
    'augmentations': data.get('augmentations', [])
}
```

---

### 2.3 create_project_structure() — 폴더 생성

```python
def create_project_structure() -> Tuple[bool, str]:
    try:
        for name, path in DIRS.items():
            path.mkdir(parents=True, exist_ok=True)
        return True, "폴더 구조 생성/확인 완료"
    except PermissionError as e:
        return False, f"폴더 생성 권한 오류: {e}"
    except OSError as e:
        return False, f"폴더 생성 실패: {e}"
```

**DIRS 정의:**
```python
DIRS: Dict[str, Path] = {
    'original':  project_data / 'images' / 'original',
    'augmented': project_data / 'images' / 'augmented',
    'labels':    project_data / 'labels',
    'metadata':  project_data / 'metadata',
    'logs':      project_data / 'logs',
}
```

**`mkdir(parents=True, exist_ok=True)` 설명:**
- `parents=True`: 부모 디렉토리 없으면 자동 생성
- `exist_ok=True`: 이미 있으면 에러 안 냄

**예시:**
```python
# 이런 깊은 경로라도 한 줄에 생성됨
Path('project_data/images/original').mkdir(parents=True, exist_ok=True)
# 결과: project_data/
#       └─ images/
#          └─ original/
```

---

### 2.4 save_original_image() — 이미지 복사

```python
def save_original_image(
    src_path: str,
    image_name: str,
    image_id: str
) -> Tuple[bool, str, str]:
```

#### 로직

```python
ext = Path(image_name).suffix          # ".jpg"
base = Path(image_name).stem           # "sample"
dest_name = f"{image_id}_{base}{ext}"  # "a1b2c3d4_sample.jpg"
dest_path = DIRS['original'] / dest_name

try:
    shutil.copy2(src_path, str(dest_path))
    return True, "원본 이미지 저장 완료", dest_name
except FileNotFoundError:
    return False, f"원본 이미지 파일 없음: {src_path}", ""
except PermissionError as e:
    return False, f"이미지 저장 권한 오류: {e}", ""
except OSError as e:
    return False, f"이미지 저장 실패: {e}", ""
```

#### 파일명 전략

**왜 `image_id` 접두사를 붙이나?**

| 시나리오 | 접두사 X | 접두사 O |
|---------|---------|---------|
| sample.jpg 2번 업로드 | 같은 파일 | a1b2c3d4_sample.jpg, e5f6g7h8_sample.jpg |
| 충돌 | ❌ 덮어씀 | ✅ 서로 다른 파일 |

**`shutil.copy2()` vs `shutil.copy()`:**
```python
shutil.copy()   # 파일만 복사, 메타데이터 보존 X
shutil.copy2()  # 파일 + 생성시간, 수정시간 등 보존 ✅
                # 데이터셋 관리에서 시간 정보 중요
```

---

### 2.5 save_label_json() — Read-Modify-Write 저장

```python
def save_label_json(
    image_id: str,
    image_name: str,
    image_width: int,
    image_height: int,
    annotations: List[dict],
    saved_image_name: str
) -> Tuple[bool, str, str]:
```

#### 핵심: Append-safe Read-Modify-Write

##### Phase 1: 기존 파일 읽기 (데이터 보존)

```python
existing_data: Optional[dict] = None
if json_path.exists():
    try:
        with open(str(json_path), 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    except (json.JSONDecodeError, OSError):
        # 파일이 깨진 경우: 백업 + 새로 시작
        backup = json_path.with_suffix('.json.bak')
        try:
            shutil.copy2(str(json_path), str(backup))
        except OSError:
            pass
        existing_data = None
```

**예시:**

**상황 1: 파일이 없는 경우**
```python
# json_path.exists() = False
# → existing_data = None (새로 시작)
```

**상황 2: 파일이 정상인 경우**
```python
# json_path.exists() = True
# 파일 내용:
# {
#   "image_id": "a1b2c3d4",
#   "annotations": [{"label": "car", ...}]
# }
# → existing_data = {...}  (읽음)
```

**상황 3: 파일이 깨진 경우**
```python
# json_path.exists() = True
# 파일 내용: "{broken json data"
# → json.JSONDecodeError 발생
# → 01b2c3d4.json.bak 백업 생성
# → existing_data = None (새로 시작)
```

---

##### Phase 2: 어노테이션에 고유 ID 부여

```python
stamped_annotations = []
for ann in annotations:
    ann_copy = dict(ann)  # 원본 수정 방지
    ann_copy['annotation_id'] = str(uuid.uuid4())
    stamped_annotations.append(ann_copy)
```

**왜 annotation_id가 필요한가?**
→ 나중에 특정 박스를 수정/삭제할 때 ID로 추적 가능

**예시:**
```python
# 입력
annotations = [
    {'label': 'car', 'x': 100, 'y': 50, 'w': 200, 'h': 100},
    {'label': 'person', 'x': 500, 'y': 200, 'w': 80, 'h': 150}
]

# 출력
stamped_annotations = [
    {
        'annotation_id': 'f7a3e1c2-4d8b-11eb-ae93-0242ac120002',
        'label': 'car',
        'x': 100,
        'y': 50,
        'w': 200,
        'h': 100
    },
    {
        'annotation_id': 'e8c5d9b4-5e8c-12fb-be94-1353bd231113',
        'label': 'person',
        'x': 500,
        'y': 200,
        'w': 80,
        'h': 150
    }
]
```

---

##### Phase 3: 데이터 구성 (기존 + 신규)

```python
if existing_data and isinstance(existing_data.get('annotations'), list):
    existing_data['annotations'].extend(stamped_annotations)
    existing_data['updated_at'] = now_str
    final_data = existing_data
else:
    final_data = {
        'image_id'    : image_id,
        'image_name'  : saved_image_name,
        'image_width' : image_width,
        'image_height': image_height,
        'task_type'   : 'object_detection',
        'annotations' : stamped_annotations,
        'created_at'  : now_str,
        'updated_at'  : now_str
    }
```

**Append 로직:**

```
기존 파일 있음 + annotations 필드 유효함?
├─ YES: 기존 annotations 리스트에 신규 추가
│       existing_data['annotations'].extend([신규1, 신규2, ...])
└─ NO: 새로 시작
```

**예시:**

**첫 번째 저장:**
```python
final_data = {
    'image_id': 'a1b2c3d4',
    'image_name': 'a1b2c3d4_sample.jpg',
    'image_width': 1280,
    'image_height': 720,
    'task_type': 'object_detection',
    'annotations': [
        {
            'annotation_id': 'uuid1',
            'label': 'car',
            'x': 100,
            'y': 50,
            'w': 200,
            'h': 100
        }
    ],
    'created_at': '2026-03-14T10:30:00',
    'updated_at': '2026-03-14T10:30:00'
}
```

**두 번째 저장 (동일 이미지):**
```python
# 기존 파일에 annotations 추가
final_data = {
    'image_id': 'a1b2c3d4',
    'image_name': 'a1b2c3d4_sample.jpg',
    'image_width': 1280,
    'image_height': 720,
    'task_type': 'object_detection',
    'annotations': [
        {
            'annotation_id': 'uuid1',
            'label': 'car',
            'x': 100,
            'y': 50,
            'w': 200,
            'h': 100
        },
        {  # ← 신규 추가됨
            'annotation_id': 'uuid2',
            'label': 'person',
            'x': 500,
            'y': 200,
            'w': 80,
            'h': 150
        }
    ],
    'created_at': '2026-03-14T10:30:00',
    'updated_at': '2026-03-14T10:35:00'  # ← 갱신 시간
}
```

---

##### Phase 4: 안전한 쓰기 (Atomic Write)

```python
tmp_path = json_path.with_suffix('.json.tmp')
try:
    with open(str(tmp_path), 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    tmp_path.replace(json_path)  # 원자적 이름 변경
    return True, "라벨 JSON 저장 완료", json_name
except OSError as e:
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)  # 임시 파일 정리
    return False, f"JSON 저장 실패: {e}", ""
```

**원자적 쓰기의 중요성:**

```
❌ 나쁜 방식:
   json_path에 직접 쓰기 → 도중 전원 차단 → 파일 반쪽만 쓰임 (손상)

✅ 좋은 방식:
   1. tmp_path에 쓰기 (완료될 때까지)
   2. tmp_path → json_path로 원자적 이름 변경
      (이름 변경은 OS 수준에서 원자적 연산)
   → 파일이 완벽하게 쓰인 후에만 교체됨
```

**Visual:**
```
시간 →

❌ 직접 쓰기 (위험):
   |████░░░░░░| (도중 전원 차단!) → 손상된 파일

✅ 임시 파일 + 이름 변경 (안전):
   |████████████| (완료)
   a1b2c3d4.json.tmp 준비됨
   ↓
   rename → a1b2c3d4.json ✓ (원자적)
```

---

### 2.6 update_dataset_index() — 메타데이터 누적

```python
def update_dataset_index(entry: dict) -> Tuple[bool, str]:
```

#### Read-Modify-Write for JSON Array

```python
existing: list = []

if _INDEX_PATH.exists():
    try:
        with open(str(_INDEX_PATH), 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                existing = data
            else:
                existing = [data]  # 호환성
    except (json.JSONDecodeError, OSError):
        backup = _INDEX_PATH.with_suffix('.json.bak')
        try:
            shutil.copy2(str(_INDEX_PATH), str(backup))
        except OSError:
            pass
        existing = []

existing.append(entry)  # 신규 항목 추가

# 안전한 쓰기
tmp_path = _INDEX_PATH.with_suffix('.json.tmp')
try:
    with open(str(tmp_path), 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    tmp_path.replace(_INDEX_PATH)
    return True, "dataset_index.json 갱신 완료"
except OSError as e:
    if tmp_path.exists():
        tmp_path.unlink(missing_ok=True)
    return False, f"인덱스 갱신 실패: {e}"
```

**인덱스 구조:**
```python
# dataset_index.json
[
  {
    "image_id": "a1b2c3d4",
    "original_image": "images/original/a1b2c3d4_sample.jpg",
    "label_json": "labels/a1b2c3d4.json",
    "recipe_jsons": [
      "labels/a1b2c3d4_recipe_000.json",
      "labels/a1b2c3d4_recipe_001.json"
    ],
    "annotation_count": 2,
    "created_at": "2026-03-14T10:30:00"
  },
  {
    "image_id": "e5f6g7h8",
    # ... 두 번째 이미지
  }
]
```

**왜 이 인덱스가 필요한가?**
→ 전체 데이터셋의 "네비게이션"
→ 특정 이미지 찾기, 통계 조회 시 이 파일 하나만 읽으면 됨

---

### 2.7 save_all_data() — 통합 파사드

```python
def save_all_data(data: dict) -> dict:
```

#### 전체 흐름 상세 분석

```python
logger   = _get_logger()
image_id = str(uuid.uuid4())[:8]  # 고유 ID 생성
rollback_files: List[Path] = []   # 에러 시 삭제할 파일 목록

def _fail(msg: str, extra: Optional[dict] = None) -> dict:
    """에러 응답 + 롤백"""
    save_log(image_id, 'error', msg, extra)
    for fp in rollback_files:
        try:
            if fp.exists():
                fp.unlink()  # 파일 삭제
        except OSError:
            pass
    return {
        'success': False,
        'message': msg,
        'image_id': image_id,
        'saved_paths': {},
        'stats': {},
        'rejected_boxes': []
    }
```

---

#### Step 1: 입력 검증

```python
is_valid, msg, cleaned = validate_input_data(data)
if not is_valid:
    return _fail(f"입력 검증 실패: {msg}")

valid_boxes    = cleaned['valid_boxes']
rejected_boxes = cleaned['rejected_boxes']
augmentations  = cleaned['augmentations']

if rejected_boxes:
    save_log(image_id, 'warning',
             f"거부된 박스 {len(rejected_boxes)}개",
             {'rejected': rejected_boxes})
```

---

#### Step 2: 폴더 생성

```python
ok, msg = create_project_structure()
if not ok:
    return _fail(f"폴더 생성 실패: {msg}")
```

---

#### Step 3: 이미지 저장 + 롤백 등록

```python
ok, msg, saved_img_name = save_original_image(
    cleaned['image_path'],
    cleaned['image_name'],
    image_id
)
if not ok:
    return _fail(f"이미지 저장 실패: {msg}")

# 이후 에러 시 이 파일을 삭제
rollback_files.append(DIRS['original'] / saved_img_name)
```

---

#### Step 4: 라벨 JSON 저장 + 롤백 등록

```python
ok, msg, label_json_name = save_label_json(
    image_id,
    cleaned['image_name'],
    cleaned['image_width'],
    cleaned['image_height'],
    valid_boxes,
    saved_img_name
)
if not ok:
    return _fail(f"라벨 JSON 저장 실패: {msg}")

rollback_files.append(DIRS['labels'] / label_json_name)
```

---

#### Step 5: 레시피 JSON 저장 (On-the-fly)

```python
ok, msg, recipe_names = save_augmented_recipes(
    image_id,
    saved_img_name,
    valid_boxes,
    augmentations
)
if not ok:
    return _fail(f"레시피 저장 실패: {msg}")

for rn in recipe_names:
    rollback_files.append(DIRS['labels'] / rn)
```

---

#### Step 6: 인덱스 갱신

```python
index_entry = {
    'image_id'        : image_id,
    'original_image'  : f"images/original/{saved_img_name}",
    'label_json'      : f"labels/{label_json_name}",
    'recipe_jsons'    : [f"labels/{r}" for r in recipe_names],
    'annotation_count': len(valid_boxes),
    'created_at'      : datetime.now().isoformat()
}
ok, msg = update_dataset_index(index_entry)
if not ok:
    return _fail(f"인덱스 갱신 실패: {msg}")
```

---

#### Step 7: 성공 응답

```python
return {
    'success'   : True,
    'message'   : '저장 완료',
    'image_id'  : image_id,
    'saved_paths': {
        'original_image': f"images/original/{saved_img_name}",
        'label_json'    : f"labels/{label_json_name}",
        'recipe_jsons'  : [f"labels/{r}" for r in recipe_names]
    },
    'stats': {
        'total_boxes'   : len(data.get('objects', [])),
        'valid_boxes'   : len(valid_boxes),
        'rejected_boxes': len(rejected_boxes),
        'recipes_saved' : len(recipe_names)
    },
    'rejected_boxes': rejected_boxes
}
```

---

## 3. augmentation_recipe.py — On-the-fly 증강 파이프라인

### 3.1 sample_augmentation_params() — 레시피 생성

```python
def sample_augmentation_params(
    n: int = 5,
    label_sensitive: bool = False,
    enable_homography: bool = True,
    enable_noise: bool = True,
    enable_specular: bool = False,
    enable_shadow: bool = False,
    seed: Optional[int] = None
) -> List[dict]:
```

#### 파라미터 범위

```python
AUG_PARAM_RANGES = {
    'h31'            : (-0.001,  0.001),   # X축 원근
    'h32'            : (-0.001,  0.001),   # Y축 원근
    'theta_deg'      : (-5.0,    5.0),     # 회전각
    'poisson_lambda' : (20.0,    40.0),    # 노이즈
}
```

#### 샘플링 로직

```python
if seed is not None:
    np.random.seed(seed)  # 재현성 보장

params_list = []
for _ in range(n):
    p: Dict[str, Any] = {}

    if enable_homography:
        p['h31'] = round(float(np.random.uniform(*AUG_PARAM_RANGES['h31'])), 6)
        p['h32'] = round(float(np.random.uniform(*AUG_PARAM_RANGES['h32'])), 6)

        # label_sensitive 방어
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
```

**출력 예시:**
```python
[
    {
        'h31': -0.000214,
        'h32':  0.000902,
        'theta_deg': -0.77,
        'poisson_lambda': 24.5,
        'specular_flare': False,
        'shadow': False
    },
    {
        'h31': -0.00098,
        'h32': -0.000994,
        'theta_deg': -4.99,
        'poisson_lambda': 26.2,
        'specular_flare': False,
        'shadow': False
    }
]
```

---

### 3.2 build_homography_matrix() — 변환 행렬 구성

```python
def build_homography_matrix(
    h31: float, h32: float, theta_deg: float,
    img_w: int, img_h: int
) -> np.ndarray:
```

#### 수학적 배경

3×3 호모그래피 행렬의 구조:

```
┌                                      ┐
│  m00  m01  m02                       │
│  m10  m11  m12                       │
│  m20  m21  m22                       │
└                                      ┘

의미:
┌                                      ┐
│  cos θ  -sin θ  tx       (아핀: 회전+이동)  │
│  sin θ   cos θ  ty       (아핀: 회전+이동)  │
│   h31    h32     1       (원근: 소실점)    │
└                                      ┘
```

#### 구현

```python
cx, cy = img_w / 2.0, img_h / 2.0
rad = math.radians(theta_deg)
cos_a = math.cos(rad)
sin_a = math.sin(rad)

# 회전 중심을 이미지 중앙으로 하기 위한 이동값
tx = cx * (1.0 - cos_a) + cy * sin_a
ty = cy * (1.0 - cos_a) - cx * sin_a

H = np.array([
    [cos_a, -sin_a, tx],
    [sin_a,  cos_a, ty],
    [h31,    h32,   1.0]
], dtype=np.float64)

return H
```

#### 예시 계산

**Input:**
- 이미지: 640×480
- θ = 5°
- h31 = 0.0005
- h32 = -0.0003

**Step 1: 기본값 계산**
```
cx = 640 / 2 = 320
cy = 480 / 2 = 240
rad = 5 * π/180 ≈ 0.0873 라디안
cos_a ≈ 0.996
sin_a ≈ 0.0872

tx = 320 * (1 - 0.996) + 240 * 0.0872
   = 320 * 0.004 + 240 * 0.0872
   = 1.28 + 20.93
   ≈ 22.2

ty = 240 * (1 - 0.996) - 320 * 0.0872
   = 240 * 0.004 - 320 * 0.0872
   = 0.96 - 27.9
   ≈ -26.94
```

**Step 2: 행렬 구성**
```
H = [
  [ 0.996, -0.0872,  22.2  ],
  [ 0.0872, 0.996, -26.94  ],
  [ 0.0005, -0.0003,  1.0  ]
]
```

**의미:**
- 1, 2행: 5도 회전 + 중앙을 기준으로 이동
- 3행: 약간의 원근감 추가

---

### 3.3 transform_bbox_homography() — BBox 좌표 변환

```python
def transform_bbox_homography(
    bbox: List[float],
    H: np.ndarray,
    img_w: int,
    img_h: int
) -> List[int]:
```

#### 단계별 분석

##### Phase 1: 박스 → 4개 꼭짓점

```python
x, y, w, h = bbox
corners = np.array([
    [x,     y,     1],      # 좌상단 (TL)
    [x + w, y,     1],      # 우상단 (TR)
    [x + w, y + h, 1],      # 우하단 (BR)
    [x,     y + h, 1],      # 좌하단 (BL)
], dtype=np.float64).T      # (3, 4) 행렬로 변환
```

**Visual:**
```
(x, y) ─────────── (x+w, y)
  │                  │
  │      BBox        │
  │                  │
(x, y+h) ─────── (x+w, y+h)

동차 좌표로 표현:
[x,     y,     1]
[x + w, y,     1]
[x + w, y + h, 1]
[x,     y + h, 1]

행렬 형태 (3×4):
x     x+w   x+w   x
y     y     y+h   y+h
1     1     1     1
```

---

##### Phase 2: Homography 적용 (행렬 곱셈)

```python
transformed = H @ corners  # (3×3) @ (3×4) = (3×4)
```

**수학:**
```
H @ corners = 
[m00  m01  m02] [x1   x2   x3   x4]
[m10  m11  m12] [y1   y2   y3   y4]
[m20  m21  m22] [1    1    1    1 ]

= [m00*x1+m01*y1+m02*1   m00*x2+m01*y2+m02*1   ...]
  [m10*x1+m11*y1+m12*1   m10*x2+m11*y2+m12*1   ...]
  [m20*x1+m21*y1+m22*1   m20*x2+m21*y2+m22*1   ...]

= [X1  X2  X3  X4]
  [Y1  Y2  Y3  Y4]
  [W1  W2  W3  W4]
```

**결과:**
- `transformed[0]` = X 좌표들
- `transformed[1]` = Y 좌표들
- `transformed[2]` = 가중치 (W)

---

##### Phase 3: Perspective Divide (동차 좌표 → 유클리드)

```python
W = transformed[2]
W = np.where(np.abs(W) < 1e-10, 1e-10, W)  # 0 나누기 방지
xs = transformed[0] / W
ys = transformed[1] / W
```

**왜 필요한가?**

원근 변환은 선을 비선형으로 구부린다. 동차 좌표 [X, Y, W]를 유클리드 좌표 (x, y)로 복원하려면:

```
(x, y) = (X/W, Y/W)
```

**수치 안정성:**
```python
# 극단적 원근에서 W ≈ 0 가능
# 0으로 나누기 방지
W = np.where(np.abs(W) < 1e-10, 1e-10, W)
```

**예시:**
```
원본: [100, 50, 1]
변환: [105.2, 52.1, 1.001]
Divide: (105.2/1.001, 52.1/1.001) ≈ (105.1, 52.0)

극단적: [150, 80, 0.01]
Divide: (150/0.01, 80/0.01) = (15000, 8000)
        → 원근감이 강해서 좌표가 멀어짐
```

---

##### Phase 4: Min-Max → AABB

```python
new_x = float(np.clip(np.min(xs), 0, img_w))
new_y = float(np.clip(np.min(ys), 0, img_h))
new_x2 = float(np.clip(np.max(xs), 0, img_w))
new_y2 = float(np.clip(np.max(ys), 0, img_h))

new_w = new_x2 - new_x
new_h = new_y2 - new_y

return [round(int(new_x)), round(int(new_y)),
        round(int(new_w)), round(int(new_h))]
```

**Axis-Aligned BBox (AABB) 개념:**

```
원본 직사각형
┌─────────┐
│ BBox    │
└─────────┘

회전 후 (일반 사각형)
   ╱───────╲
  ╱         ╲
 │           │
  ╲         ╱
   ╲───────╱

AABB (축 정렬 직사각형)
┌─────────────┐
│ ╱───────╲   │
│╱         ╲  │
││           ││
│╲         ╱  │
│ ╲───────╱   │
└─────────────┘
```

**Clipping (이미지 경계 내로 제한):**
```python
new_x = np.clip(np.min(xs), 0, img_w)
```

| xs 값 | np.min(xs) | np.clip(..., 0, img_w) |
|-------|------------|----------------------|
| [-10, 100, 200, 150] | -10 | **0** |
| [50, 100, 200, 150] | 50 | **50** |
| [1300, 1350, 1400, 1350] | 1300 | **1280** (img_w=1280) |

---

### 3.4 apply_poisson_noise() — 물리적 노이즈 합성

```python
def apply_poisson_noise(image: np.ndarray, lam: float) -> np.ndarray:
```

#### 물리적 배경

실제 카메라 센서:
1. **광자 도달**: 밝은 픽셀에는 더 많은 광자 도달
2. **분산**: 광자 수의 분산 = 광자 수의 기댓값 (Poisson 분포)
3. **노이즈**: 상대 SNR(신호 대 잡음)은 밝기 증가에 따라 개선

**Poisson 분포:**
- 파라미터: λ (기댓값)
- 분산: λ
- 표준편차: √λ

#### 구현 로직

```python
# Step 1: 정규화 (0~1 범위)
img_f = image.astype(np.float64) / 255.0

# Step 2: λ 스케일링
scaled = img_f * lam

# Step 3: Poisson 샘플링
noisy = np.random.poisson(scaled).astype(np.float64)

# Step 4: λ로 나눠서 복원
noisy_norm = (noisy / lam) * 255.0

# Step 5: 클리핑 (0~255)
return np.clip(noisy_norm, 0, 255).astype(np.uint8)
```

#### 예시 계산

**Input:**
```
원본 픽셀값: [64, 128, 192, 255]  (0~255)
λ = 30
```

**Step 1: 정규화**
```
img_f = [64, 128, 192, 255] / 255.0
      = [0.251, 0.502, 0.753, 1.0]
```

**Step 2: λ 스케일**
```
scaled = [0.251, 0.502, 0.753, 1.0] * 30
       = [7.53, 15.06, 22.59, 30.0]
```

**Step 3: Poisson 샘플링**
```
np.random.poisson([7.53, 15.06, 22.59, 30.0])
= [8, 14, 23, 31]  (무작위, 평균이 λ 근처)

분산: √8 ≈ 2.83, √14 ≈ 3.74, √23 ≈ 4.80, √31 ≈ 5.57
→ 밝은 픽셀일수록 절대 노이즈 크지만, 상대 노이즈(σ/μ)는 작음
```

**Step 4: λ로 나눔**
```
noisy_norm = [8, 14, 23, 31] / 30 * 255
           = [68, 119, 195, 263]
```

**Step 5: 클리핑**
```
np.clip([68, 119, 195, 263], 0, 255)
= [68, 119, 195, 255]
```

**Output:**
```
원본:   [64,  128, 192, 255]
노이즈: [68,  119, 195, 255]
차이:   [+4,  -9,  +3,   0]
```

---

### 3.5 OnTheFlyAugmenter — PyTorch Dataset

```python
class OnTheFlyAugmenter:
    def __init__(
        self,
        recipe_dir: str,
        original_img_root: str = '',
        image_size: Optional[Tuple[int, int]] = None
    ):
```

#### 초기화

```python
self.recipe_dir       = Path(recipe_dir)
self.original_img_root = Path(original_img_root) if original_img_root else None
self.image_size       = image_size
self.recipes = self._load_recipes()
```

#### _load_recipes() — 메모리 적재

```python
def _load_recipes(self) -> List[dict]:
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
```

**왜 모든 레시피를 메모리에 로드하는가?**
→ 학습 중 빠른 접근 필요. JSON 읽기 오버헤드 제거.

---

#### __getitem__() — 실시간 증강

```python
def __getitem__(self, idx: int) -> dict:
    recipe = self.recipes[idx]
    params = recipe.get('augmentation_params', {})
    anns   = recipe.get('original_annotations', [])

    # Phase 1: 원본 이미지 로드
    source_img_name = recipe['source_image']
    if self.original_img_root:
        img_path = self.original_img_root / source_img_name
    else:
        img_path = Path(source_img_name)

    img_array = np.fromfile(str(img_path), dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"이미지를 로드할 수 없습니다: {img_path}")

    img_h, img_w = image.shape[:2]
    aug_anns = [dict(a) for a in anns]  # 깊은 복사

    # Phase 2: Homography 변환
    if 'h31' in params and 'h32' in params and 'theta_deg' in params:
        h31       = float(params['h31'])
        h32       = float(params['h32'])
        theta_deg = float(params['theta_deg'])

        image, H = apply_homography(image, h31, h32, theta_deg)

        for ann in aug_anns:
            new_bbox = transform_bbox_homography(
                [ann['x'], ann['y'], ann['w'], ann['h']],
                H, img_w, img_h
            )
            ann['x'], ann['y'], ann['w'], ann['h'] = new_bbox

    # Phase 3: Poisson 노이즈
    if 'poisson_lambda' in params:
        lam = float(params['poisson_lambda'])
        if lam > 0:
            image = apply_poisson_noise(image, lam)

    # Phase 4: 조명 증강
    if params.get('specular_flare', False):
        image = apply_specular_flare(image)

    if params.get('shadow', False):
        image = apply_shadow(image)

    # Phase 5: 리사이즈 (선택)
    if self.image_size is not None:
        target_w, target_h = self.image_size
        scale_x = target_w / img_w
        scale_y = target_h / img_h

        image = cv2.resize(image, (target_w, target_h),
                           interpolation=cv2.INTER_LINEAR)

        for ann in aug_anns:
            ann['x'] = round(int(ann['x'] * scale_x))
            ann['y'] = round(int(ann['y'] * scale_y))
            ann['w'] = round(int(ann['w'] * scale_x))
            ann['h'] = round(int(ann['h'] * scale_y))

    return {
        'image'      : image,
        'annotations': aug_anns,
        'image_id'   : recipe.get('image_id', ''),
        'recipe_id'  : recipe.get('recipe_id', '')
    }
```

**각 Phase의 목적:**

| Phase | 목적 | 복구 가능? |
|-------|------|-----------|
| 1: 이미지 로드 | 원본 파일 읽음 | ✅ 매번 새로 로드 |
| 2: Homography | 기하학적 변환 | ✅ 동일 H로 재현 |
| 3: Poisson | 물리 노이즈 | ✅ 동일 λ, seed로 재현 |
| 4: 조명 | 조명 효과 | ❌ 난수 기반 (매번 다름) |
| 5: 리사이즈 | 모델 입력 크기 | ✅ 결정적 |

**"매번 다른가?" vs "재현 가능한가?"**
→ Poisson과 조명은 난수 기반이지만 `seed` 고정 시 동일 재현 가능

---

## 4. 데이터 흐름 통합 이해

### 4.1 전체 파이프라인

```
사용자 (웹 브라우저)
    │
    ├─ 이미지 업로드 → /api/upload → images/original/
    │
    ├─ Canvas에서 박스 그리기 (JavaScript 상태 관리)
    │
    └─ "저장 및 증강 실행" 클릭 → /api/save
         │
         └─ Flask 서버
              │
              ├─ validate_annotation()        [flask 자체]
              │  └─ 각 박스 검증
              │
              ├─ run_augmentation_pipeline()  [flask 자체]
              │  ├─ Homography 적용
              │  ├─ BBox 변환
              │  ├─ Poisson 노이즈
              │  └─ 증강 이미지 저장 (이미지)
              │
              ├─ backend.save_all_data()      [새로 만든 것]
              │  ├─ validate_and_format()
              │  ├─ save_original_image()
              │  ├─ save_label_json()
              │  ├─ save_augmented_recipes()
              │  ├─ update_dataset_index()
              │  └─ save_log()
              │
              └─ JSON 응답 반환

저장된 파일
    ├─ project_data/images/original/      [이미지]
    ├─ project_data/images/augmented/     [미리보기 증강 이미지]
    ├─ project_data/labels/               [JSON: 라벨, 레시피]
    ├─ project_data/metadata/             [인덱스]
    └─ project_data/logs/                 [로그]
```

---

### 4.2 좌표의 여정

```
JavaScript (Canvas 픽셀)
    ↓
canvasToNatural()
    ↓ (축소 비율로 변환)
원본 이미지 좌표 (100, 50)
    ↓
validate_and_format()
    ↓ (Clipping, 정규화)
검증된 좌표 (100, 50)
    ↓
save_label_json()
    ↓
labels/{id}.json에 저장
    ↓
OnTheFlyAugmenter.__getitem__()
    ↓
transform_bbox_homography()
    ↓ (H @ 꼭짓점 → AABB)
증강 후 좌표 (105, 52)
    ↓
딥러닝 모델 학습
```

---

### 4.3 에러 시나리오

#### Scenario 1: 이미지 로드 실패

```
save_all_data()
    ├─ Step 1: 검증 OK
    ├─ Step 2: 폴더 생성 OK
    ├─ Step 3: 이미지 복사 ❌ (파일 없음)
    │  └─ _fail() 호출
    │     └─ rollback_files = [이미지 경로]
    │        (아무것도 없으므로 롤백 할 것 없음)
    └─ 에러 응답 반환
```

---

#### Scenario 2: JSON 저장 중 오류

```
save_all_data()
    ├─ Step 1: 검증 OK
    ├─ Step 2: 폴더 생성 OK
    ├─ Step 3: 이미지 복사 OK
    │  └─ rollback_files.append(이미지 경로)
    ├─ Step 4: 라벨 JSON 저장 ❌ (디스크 용량 부족)
    │  └─ _fail() 호출
    │     └─ for fp in rollback_files:
    │            fp.unlink()  # 이미지 삭제
    └─ 에러 응답 반환
```

→ 결과: 원본 이미지는 정리됨, 데이터 파편화 방지 ✅

---

## 최종 요약

| 모듈 | 역할 | 핵심 기법 |
|------|------|---------|
| **validator.py** | 데이터 검증 | 8단계 방어 로직, Clipping |
| **storage_module.py** | 안전한 저장 | Read-Modify-Write, 원자적 쓰기, 롤백 |
| **augmentation_recipe.py** | 수학적 증강 | Homography 행렬, Poisson 노이즈, AABB |

모든 모듈이 **UI 독립적**이며 **방어적 프로그래밍**을 따릅니다. ✨
