"""
==========================================================
storage_module.py — UI 독립형 백엔드 저장 모듈 (관제탑)
==========================================================
역할:
  이 모듈은 프론트엔드(UI)가 어떤 프레임워크(Flask, PyQt5,
  Streamlit, Tkinter)이든 상관없이, 단 하나의 함수
  save_all_data(...)만 호출하면 다음 모든 작업을 처리한다:

    [1] 입력 데이터 검증   (validate_input_data)
    [2] 폴더 구조 생성     (create_project_structure)
    [3] 원본 이미지 저장   (save_original_image)
    [4] 라벨 JSON 저장     (save_label_json)
    [5] 증강 레시피 저장   (save_augmented_recipes)
    [6] 메타데이터 갱신    (update_dataset_index)
    [7] 로그 기록          (save_log)

왜 이 순서인가 (안전성 보장):
  검증(1) → 구조(2) → 이미지(3) → 라벨(4) → 증강(5) → 인덱스(6) → 로그(7)
  * 검증이 실패하면 파일 시스템에 아무것도 기록하지 않는다.
  * 이미지가 먼저 저장된 후에야 라벨 JSON이 생성됨
    (라벨 JSON은 이미지 파일을 참조하므로).
  * dataset_index.json은 가장 마지막에 갱신
    (도중 에러 시 인덱스에 오염된 항목이 남지 않도록).

부분 저장 방지 전략:
  - 각 단계를 try/except로 감싸고, 실패 시 이미 저장된 파일을
    삭제(롤백)하여 데이터 파편화를 막는다.
  - 모든 오류는 로그 파일에 기록되고 호출자에게 반환된다.

On-the-fly 아키텍처:
  증강된 이미지를 디스크에 저장하지 않는다.
  대신 어떤 파라미터로 증강할지 적힌 JSON "레시피"만 저장한다.
  → 디스크 I/O 절약, 학습 시 DataLoader가 배치마다 실시간 적용.
"""

import os
import json
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .validator import validate_batch


# ──────────────────────────────────────────────────────────
# 기본 경로 설정
# ──────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parents[2]          # web_ui_dashboard/
_PROJECT_DATA = _BASE_DIR / 'project_data'

DIRS: Dict[str, Path] = {
    'original':  _PROJECT_DATA / 'images' / 'original',
    'augmented': _PROJECT_DATA / 'images' / 'augmented',
    'labels':    _PROJECT_DATA / 'labels',
    'metadata':  _PROJECT_DATA / 'metadata',
    'logs':      _PROJECT_DATA / 'logs',
}

_INDEX_PATH   = DIRS['metadata'] / 'dataset_index.json'
_LOG_PATH     = DIRS['logs']     / 'storage.log'


# ──────────────────────────────────────────────────────────
# 로거 초기화
# ──────────────────────────────────────────────────────────

def _get_logger() -> logging.Logger:
    logger = logging.getLogger('storage_module')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        DIRS['logs'].mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(_LOG_PATH), encoding='utf-8')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        logger.addHandler(fh)
    return logger


# ──────────────────────────────────────────────────────────
# Step 1: 입력 데이터 검증
# ──────────────────────────────────────────────────────────

def validate_input_data(data: dict) -> Tuple[bool, str, dict]:
    """
    프론트엔드에서 수신한 딕셔너리의 최상위 구조를 검증한다.

    Args:
        data: {
            'image_path'  : str   (원본 이미지 절대/상대 경로),
            'image_name'  : str   (파일명, 예: "sample.jpg"),
            'image_width' : int,
            'image_height': int,
            'objects'     : list  (바운딩 박스 목록),
            'augmentations': list (증강 레시피 목록, 선택)
        }

    Returns:
        (is_valid: bool, message: str, cleaned_data: dict)
    """
    # 필수 최상위 키 체크
    required = ('image_path', 'image_name', 'image_width', 'image_height', 'objects')
    for key in required:
        if key not in data:
            return False, f"필수 키 누락: '{key}'", {}

    # 이미지 파일 실제 존재 여부 확인
    image_path = Path(data['image_path'])
    if not image_path.exists():
        return False, f"이미지 파일을 찾을 수 없습니다: {image_path}", {}

    img_w = data['image_width']
    img_h = data['image_height']
    if not isinstance(img_w, int) or not isinstance(img_h, int):
        return False, "image_width, image_height는 정수여야 합니다.", {}
    if img_w <= 0 or img_h <= 0:
        return False, f"이미지 해상도가 유효하지 않습니다. ({img_w}×{img_h})", {}

    objects = data.get('objects', [])
    if not isinstance(objects, list):
        return False, "'objects'는 리스트여야 합니다.", {}

    # 바운딩 박스 일괄 검증 (validator.py 활용)
    valid_boxes, rejected = validate_batch(objects, img_w, img_h, clip=True)

    return True, "OK", {
        'image_path'   : str(image_path),
        'image_name'   : str(data['image_name']),
        'image_width'  : img_w,
        'image_height' : img_h,
        'valid_boxes'  : valid_boxes,
        'rejected_boxes': rejected,
        'augmentations': data.get('augmentations', [])
    }


# ──────────────────────────────────────────────────────────
# Step 2: 폴더 구조 생성
# ──────────────────────────────────────────────────────────

def create_project_structure() -> Tuple[bool, str]:
    """
    project_data/ 하위의 모든 폴더를 생성한다.
    이미 존재하면 무시한다 (exist_ok=True).

    왜 별도 함수로 분리하는가:
        - 권한 오류나 디스크 용량 부족 등을 조기에 감지하기 위함.
        - 데이터를 쓰기 전에 '경로가 실제로 쓸 수 있는 상태인지'
          먼저 확인하는 방어 로직.
    """
    try:
        for name, path in DIRS.items():
            path.mkdir(parents=True, exist_ok=True)
        return True, "폴더 구조 생성/확인 완료"
    except PermissionError as e:
        return False, f"폴더 생성 권한 오류: {e}"
    except OSError as e:
        return False, f"폴더 생성 실패: {e}"


# ──────────────────────────────────────────────────────────
# Step 3: 원본 이미지 저장
# ──────────────────────────────────────────────────────────

def save_original_image(
    src_path: str,
    image_name: str,
    image_id: str
) -> Tuple[bool, str, str]:
    """
    원본 이미지를 images/original/ 로 복사한다.

    파일명에 image_id를 접두사로 붙여 충돌을 방지한다.
    예: "sample.jpg" → "a1b2c3d4_sample.jpg"

    Returns:
        (success, message, saved_filename)
    """
    ext = Path(image_name).suffix
    base = Path(image_name).stem
    dest_name = f"{image_id}_{base}{ext}"
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


# ──────────────────────────────────────────────────────────
# Step 4: 라벨 JSON 저장 (Append-safe Read-Modify-Write)
# ──────────────────────────────────────────────────────────

def save_label_json(
    image_id: str,
    image_name: str,
    image_width: int,
    image_height: int,
    annotations: List[dict],
    saved_image_name: str
) -> Tuple[bool, str, str]:
    """
    바운딩 박스 어노테이션을 JSON 파일로 저장한다.

    왜 단순 json.dump가 아닌가:
        - 동일 이미지에 대한 라벨이 이미 존재할 수 있으므로
          기존 파일을 읽어서 annotations 리스트에 추가(Append)한다.
        - 파일이 깨져 있거나(JSONDecodeError) 없으면
          예외를 잡아 빈 구조로 새로 시작한다.

    JSON 스키마 (COCO/YOLO 변환에 적합한 범용 구조):
    {
        "image_id"    : "a1b2c3d4",
        "image_name"  : "a1b2c3d4_sample.jpg",
        "image_width" : 1280,
        "image_height": 720,
        "task_type"   : "object_detection",
        "annotations" : [
            {
                "annotation_id": "uuid-...",
                "label"        : "car",
                "x"  : 100,  "y": 50,
                "w"  : 200,  "h": 100,
                "label_sensitive": false
            }
        ],
        "created_at"  : "2026-03-14T...",
        "updated_at"  : "2026-03-14T..."
    }
    """
    json_name = f"{image_id}.json"
    json_path = DIRS['labels'] / json_name
    now_str   = datetime.now().isoformat()

    # ── 기존 파일 읽기 (Read-Modify-Write 패턴) ──
    existing_data: Optional[dict] = None
    if json_path.exists():
        try:
            with open(str(json_path), 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # 파일이 깨진 경우: 기존 데이터를 버리고 새로 시작
            # (깨진 파일을 백업)
            backup = json_path.with_suffix('.json.bak')
            try:
                shutil.copy2(str(json_path), str(backup))
            except OSError:
                pass
            existing_data = None

    # ── 어노테이션에 고유 ID 부여 ──
    stamped_annotations = []
    for ann in annotations:
        ann_copy = dict(ann)
        ann_copy['annotation_id'] = str(uuid.uuid4())
        stamped_annotations.append(ann_copy)

    # ── 데이터 구성 (기존 + 신규) ──
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

    # ── 안전 쓰기: 임시 파일 → 원자적 교체 ──
    # 직접 쓰다가 중단되면 파일이 손상되므로 임시 파일에 먼저 씀
    tmp_path = json_path.with_suffix('.json.tmp')
    try:
        with open(str(tmp_path), 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        # 임시 → 최종 (원자적 이동)
        tmp_path.replace(json_path)
        return True, "라벨 JSON 저장 완료", json_name
    except OSError as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False, f"JSON 저장 실패: {e}", ""


# ──────────────────────────────────────────────────────────
# Step 5: 증강 레시피 JSON 저장 (On-the-fly 아키텍처)
# ──────────────────────────────────────────────────────────

def save_augmented_recipes(
    image_id: str,
    original_image_name: str,
    original_annotations: List[dict],
    augmentations: List[dict]
) -> Tuple[bool, str, List[str]]:
    """
    증강된 이미지 파일을 저장하지 않는다.
    대신 "어떤 파라미터로 증강할지" 적힌 JSON 레시피만 저장한다.

    On-the-fly 아키텍처의 핵심 이유:
        - 이미지 5장 × 증강 10회 = 50장을 저장하면 디스크 용량 폭증.
        - 레시피 JSON은 수십 바이트에 불과, 학습 시 DataLoader가
          배치 생성 시마다 실시간으로 증강을 재현한다.
        - 재현성(Reproducibility): 동일한 레시피로 언제나 동일한
          증강 결과를 얻을 수 있다.

    레시피 JSON 스키마:
    {
        "recipe_id"       : "uuid-...",
        "image_id"        : "a1b2c3d4",
        "source_image"    : "a1b2c3d4_sample.jpg",
        "original_annotations": [...],
        "augmentation_params" : {
            "h31": -0.0005,
            "h32":  0.0003,
            "theta_deg": 2.1,
            "poisson_lambda": 28.5
        },
        "created_at": "2026-03-14T..."
    }

    Returns:
        (success, message, [recipe_json_names])
    """
    if not augmentations:
        return True, "증강 레시피 없음 (스킵)", []

    saved_recipes = []

    for idx, params in enumerate(augmentations):
        recipe_id = str(uuid.uuid4())
        recipe_name = f"{image_id}_recipe_{idx:03d}.json"
        recipe_path = DIRS['labels'] / recipe_name

        recipe_data = {
            'recipe_id'            : recipe_id,
            'image_id'             : image_id,
            'source_image'         : original_image_name,
            'original_annotations' : original_annotations,
            'augmentation_params'  : params,
            'created_at'           : datetime.now().isoformat()
        }

        tmp_path = recipe_path.with_suffix('.json.tmp')
        try:
            with open(str(tmp_path), 'w', encoding='utf-8') as f:
                json.dump(recipe_data, f, indent=2, ensure_ascii=False)
            tmp_path.replace(recipe_path)
            saved_recipes.append(recipe_name)
        except OSError as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return False, f"레시피[{idx}] 저장 실패: {e}", saved_recipes

    return True, f"레시피 {len(saved_recipes)}개 저장 완료", saved_recipes


# ──────────────────────────────────────────────────────────
# Step 6: dataset_index.json 갱신
# ──────────────────────────────────────────────────────────

def update_dataset_index(entry: dict) -> Tuple[bool, str]:
    """
    dataset_index.json에 새 항목을 추가(Append)한다.

    왜 파일 시스템이 DB를 대체할 수 있는가:
        - dataset_index.json은 각 이미지 항목에 대한 포인터(경로) 모음.
        - 이 파일 하나만 읽으면 전체 데이터셋의 구조를 파악 가능.
        - SQLite나 대형 DB 없이도 수천 장의 이미지 관리에 충분함.

    Read-Modify-Write 방식으로 기존 데이터를 보존한다.
    """
    existing: list = []

    if _INDEX_PATH.exists():
        try:
            with open(str(_INDEX_PATH), 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    existing = data
                else:
                    # 이전 버전과의 호환성: dict 형태이면 list로 감쌈
                    existing = [data]
        except (json.JSONDecodeError, OSError):
            backup = _INDEX_PATH.with_suffix('.json.bak')
            try:
                shutil.copy2(str(_INDEX_PATH), str(backup))
            except OSError:
                pass
            existing = []

    existing.append(entry)

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


# ──────────────────────────────────────────────────────────
# Step 7: 로그 기록
# ──────────────────────────────────────────────────────────

def save_log(
    image_id: str,
    level: str,
    message: str,
    extra: Optional[dict] = None
) -> None:
    """작업 결과를 logs/storage.log에 기록한다."""
    logger = _get_logger()
    log_msg = f"[{image_id}] {message}"
    if extra:
        log_msg += f" | {json.dumps(extra, ensure_ascii=False)}"

    if level == 'error':
        logger.error(log_msg)
    elif level == 'warning':
        logger.warning(log_msg)
    else:
        logger.info(log_msg)


# ──────────────────────────────────────────────────────────
# Step 5 (통합): 파사드(Facade) 함수 — save_all_data
# ──────────────────────────────────────────────────────────

def save_all_data(data: dict) -> dict:
    """
    UI 독립형 백엔드 저장 파사드(Facade) 함수.
    이 함수 하나를 호출하면 검증부터 로깅까지 모두 처리된다.

    입력 (data dict):
    {
        "image_path"   : "/abs/path/to/sample.jpg",
        "image_name"   : "sample.jpg",
        "image_width"  : 1280,
        "image_height" : 720,
        "objects"      : [
            {"label":"car", "x":100, "y":50, "w":200, "h":100,
             "label_sensitive": false},
            ...
        ],
        "augmentations": [
            {"h31": -0.0005, "h32": 0.0003, "theta_deg": 2.1,
             "poisson_lambda": 28.5},
            ...
        ]
    }

    출력 (result dict):
    {
        "success"       : true,
        "message"       : "저장 완료",
        "image_id"      : "a1b2c3d4",
        "saved_paths"   : {
            "original_image" : "images/original/a1b2c3d4_sample.jpg",
            "label_json"     : "labels/a1b2c3d4.json",
            "recipe_jsons"   : ["labels/a1b2c3d4_recipe_000.json", ...]
        },
        "stats"         : {
            "total_boxes"   : 3,
            "valid_boxes"   : 2,
            "rejected_boxes": 1,
            "recipes_saved" : 5
        },
        "rejected_boxes": [...]   # 거부된 박스 목록 (디버깅용)
    }
    """
    logger   = _get_logger()
    image_id = str(uuid.uuid4())[:8]
    rollback_files: List[Path] = []    # 에러 시 삭제할 파일 목록

    def _fail(msg: str, extra: Optional[dict] = None) -> dict:
        """에러 응답 생성 + 롤백"""
        save_log(image_id, 'error', msg, extra)
        for fp in rollback_files:
            try:
                if fp.exists():
                    fp.unlink()
            except OSError:
                pass
        return {
            'success'  : False,
            'message'  : msg,
            'image_id' : image_id,
            'saved_paths': {},
            'stats'    : {},
            'rejected_boxes': []
        }

    # ── Step 1: 입력 검증 ──
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

    # ── Step 2: 폴더 구조 생성 ──
    ok, msg = create_project_structure()
    if not ok:
        return _fail(f"폴더 생성 실패: {msg}")

    # ── Step 3: 원본 이미지 저장 ──
    ok, msg, saved_img_name = save_original_image(
        cleaned['image_path'],
        cleaned['image_name'],
        image_id
    )
    if not ok:
        return _fail(f"이미지 저장 실패: {msg}")
    rollback_files.append(DIRS['original'] / saved_img_name)

    # ── Step 4: 라벨 JSON 저장 ──
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

    # ── Step 5: 증강 레시피 JSON 저장 ──
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

    # ── Step 6: dataset_index.json 갱신 ──
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

    # ── Step 7: 로그 ──
    save_log(image_id, 'info',
             f"저장 완료 | 박스={len(valid_boxes)} | 레시피={len(recipe_names)}",
             {'image': saved_img_name})

    # ── 성공 응답 반환 ──
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


# ──────────────────────────────────────────────────────────
# UI 호출 예시 (이 파일 단독 실행 시 데모)
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    프론트엔드 개발자 사용 예시:

        from app.backend.storage_module import save_all_data

        result = save_all_data({
            'image_path'  : '/home/user/dataset/sample.jpg',
            'image_name'  : 'sample.jpg',
            'image_width' : 1280,
            'image_height': 720,
            'objects': [
                {'label': 'car',  'x': 100, 'y': 50, 'w': 200, 'h': 100},
                {'label': '',     'x': 300, 'y': 100, 'w': 50, 'h': 50},   # ← 빈 라벨, 거부됨
                {'label': 'tree', 'x': -20, 'y': 400, 'w': 100, 'h': 150}, # ← 음수 좌표, Clipping
            ],
            'augmentations': [
                {'h31': -0.0005, 'h32': 0.0003, 'theta_deg': 2.1, 'poisson_lambda': 28.5},
                {'h31':  0.0007, 'h32': -0.0002, 'theta_deg': -3.5, 'poisson_lambda': 35.0},
            ]
        })

        if result['success']:
            print("저장 완료:", result['image_id'])
            print("경로:", result['saved_paths'])
        else:
            print("오류:", result['message'])
    """
    import tempfile
    from PIL import Image as PILImage
    import numpy as np

    print("=" * 60)
    print("save_all_data() 통합 테스트")
    print("=" * 60)

    # 임시 테스트 이미지 생성
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_img_path = tmp.name
    dummy_img = PILImage.fromarray(
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    )
    dummy_img.save(tmp_img_path)

    payload = {
        'image_path'  : tmp_img_path,
        'image_name'  : 'test_sample.jpg',
        'image_width' : 640,
        'image_height': 480,
        'objects': [
            {'label': 'car',    'x': 100, 'y': 50, 'w': 150, 'h': 80},
            {'label': '',       'x': 300, 'y': 100, 'w': 50, 'h': 50},   # 거부 예상
            {'label': 'person', 'x': -10, 'y': 200, 'w': 80, 'h': 120},  # Clipping 예상
            {'label': 'truck',  'x': 500, 'y': 100, 'w': 500, 'h': 100}, # 우측 Clipping 예상
        ],
        'augmentations': [
            {'h31': -0.0005, 'h32': 0.0003, 'theta_deg': 2.1, 'poisson_lambda': 28.5},
            {'h31':  0.0007, 'h32': -0.0002, 'theta_deg': -3.5, 'poisson_lambda': 35.0},
        ]
    }

    result = save_all_data(payload)

    print(f"성공 여부: {result['success']}")
    print(f"메시지  : {result['message']}")
    if result['success']:
        print(f"이미지 ID: {result['image_id']}")
        print(f"저장 경로: {result['saved_paths']}")
        print(f"통계    : {result['stats']}")
        if result['rejected_boxes']:
            print(f"거부 박스:")
            for rb in result['rejected_boxes']:
                print(f"  [{rb['index']}] {rb['reason']}")

    # 임시 파일 정리
    os.unlink(tmp_img_path)
    print("=" * 60)
