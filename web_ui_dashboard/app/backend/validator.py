"""
==========================================================
validator.py — 바운딩 박스 데이터 검증 및 정규화 모듈
==========================================================
역할:
  - 프론트엔드(UI)에서 넘어온 원시(raw) 바운딩 박스 딕셔너리를
    검증하고, 표준 포맷으로 변환하는 순수 함수 모음.
  - Flask, PyQt5, Tkinter 등 어떤 UI에도 종속되지 않는 독립 모듈.

방어적 프로그래밍(Defensive Programming) 원칙:
  - 좌표 범위 이탈 → Clipping(경계 내로 잘라내기)
  - 빈 라벨, 음수 크기, 필드 누락 → 즉시 거부(reject)
  - 클리핑 후에도 크기가 1px 미만 → 무의미한 박스로 거부

데이터 흐름:
  raw dict (UI) → validate_and_format() → 표준화된 dict OR (None, 에러 메시지)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List

import numpy as np


# ──────────────────────────────────────────────────────────
# 데이터 모델 (dataclass): BoundingBox
# ──────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    """
    확정된 바운딩 박스를 표현하는 데이터 클래스.

    x, y: 박스 좌측 상단 좌표 (픽셀, 정수)
    w, h: 너비 / 높이 (픽셀, 양의 정수)
    label: 객체 클래스명 (비어있으면 안 됨)
    label_sensitive: True이면 방향 관련 증강(좌우 반전 등) 차단
    """
    label: str
    x: int
    y: int
    w: int
    h: int
    label_sensitive: bool = field(default=False)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h


# ──────────────────────────────────────────────────────────
# 핵심 함수: validate_and_format
# ──────────────────────────────────────────────────────────

def validate_and_format(
    raw_box: dict,
    img_w: int,
    img_h: int,
    clip: bool = True
) -> Tuple[Optional[dict], Optional[str]]:
    """
    단일 바운딩 박스 딕셔너리를 검증하고 표준 포맷으로 변환한다.

    Args:
        raw_box : 프론트엔드에서 전달된 원시 딕셔너리.
                  최소 {'label', 'x', 'y', 'w', 'h'} 키 필요.
        img_w   : 이미지 최대 너비 (픽셀).
        img_h   : 이미지 최대 높이 (픽셀).
        clip    : True이면 이미지 경계에 맞춰 Clipping.
                  False이면 경계 이탈 시 거부(drop).

    Returns:
        성공: (표준화된 dict, None)
        실패: (None, 에러 메시지 str)

    표준 출력 dict 형식:
        {
            "label": "car",
            "x": 10,  "y": 20,
            "w": 150, "h": 80,
            "label_sensitive": false
        }

    설계 이유 (왜 단순 dict 체크가 아닌가):
        - label이 공백 문자열인 경우("  ")도 빈 라벨로 취급해야 함.
        - 좌표가 float으로 넘어올 수 있으므로 명시적 형 변환 필요.
        - 클리핑 후 w나 h가 0이 되면 의미없는 박스이므로 거부해야 함.
    """

    # ── 1. 필수 필드 존재 여부 확인 ──
    required_keys = ('label', 'x', 'y', 'w', 'h')
    for key in required_keys:
        if key not in raw_box:
            return None, f"필수 필드 누락: '{key}'"

    # ── 2. 라벨 검증 ──
    label = raw_box['label']
    if not isinstance(label, str):
        return None, f"label은 문자열이어야 합니다. (입력값: {type(label).__name__})"
    label = label.strip()
    if not label:
        return None, "label이 빈 문자열입니다. 저장을 거부합니다."

    # ── 3. 좌표 숫자 변환 ──
    try:
        x = float(raw_box['x'])
        y = float(raw_box['y'])
        w = float(raw_box['w'])
        h = float(raw_box['h'])
    except (TypeError, ValueError) as e:
        return None, f"좌표 값을 숫자로 변환할 수 없습니다: {e}"

    # ── 4. 크기 음수/0 체크 (클리핑 전) ──
    if w <= 0:
        return None, f"w(너비)는 양수여야 합니다. (입력값: {w})"
    if h <= 0:
        return None, f"h(높이)는 양수여야 합니다. (입력값: {h})"

    # ── 5. 이미지 해상도 유효성 체크 ──
    if img_w <= 0 or img_h <= 0:
        return None, f"이미지 해상도가 유효하지 않습니다. (img_w={img_w}, img_h={img_h})"

    # ── 6. 좌표 Clipping 또는 범위 이탈 거부 ──
    if clip:
        # 좌측 상단 좌표를 이미지 경계 내로 고정
        x = max(0.0, min(x, float(img_w - 1)))
        y = max(0.0, min(y, float(img_h - 1)))
        # 너비/높이도 이미지 우측/하단 경계를 넘지 않도록 잘라냄
        w = max(0.0, min(w, float(img_w) - x))
        h = max(0.0, min(h, float(img_h) - y))
    else:
        if x < 0 or y < 0:
            return None, f"x, y 좌표는 음수일 수 없습니다. (x={x}, y={y})"
        if x + w > img_w or y + h > img_h:
            return None, (
                f"박스가 이미지 범위({img_w}×{img_h})를 벗어납니다. "
                f"(x2={x+w}, y2={y+h})"
            )

    # ── 7. 클리핑 후 크기 재검사 ──
    # 박스가 이미지 경계 밖에서 완전히 잘려나갔을 경우 방지
    if w < 1.0:
        return None, f"클리핑 후 너비(w)가 1px 미만입니다. (w={w:.2f})"
    if h < 1.0:
        return None, f"클리핑 후 높이(h)가 1px 미만입니다. (h={h:.2f})"

    # ── 8. 최종 표준화 포맷 반환 ──
    return {
        'label': label,
        'x': round(int(x)),
        'y': round(int(y)),
        'w': round(int(w)),
        'h': round(int(h)),
        'label_sensitive': bool(raw_box.get('label_sensitive', False))
    }, None


# ──────────────────────────────────────────────────────────
# 배치 검증 함수
# ──────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────
# [v2.0 신규] Polygon 전용 검증 함수
# ──────────────────────────────────────────────────────────

def validate_polygon(
    raw_ann: dict,
    img_w: int,
    img_h: int,
    clip: bool = True,
    min_points: int = 3
) -> Tuple[Optional[dict], Optional[str]]:
    """
    type='polygon' 라벨을 검증하고 표준 포맷으로 변환한다.

    검증 규칙:
      1. label 필드가 존재하고 비어있지 않을 것
      2. points 필드가 존재하고 리스트일 것
      3. 각 point가 [x, y] 형식(길이 2)의 숫자 쌍일 것
      4. 점 수가 min_points(기본 3) 이상일 것
      5. clip=True이면 이미지 경계 내로 좌표 클리핑

    Args:
        raw_ann   : 프론트엔드에서 전달된 원시 딕셔너리
        img_w     : 이미지 최대 너비
        img_h     : 이미지 최대 높이
        clip      : True이면 좌표를 이미지 경계 내로 클리핑
        min_points: 유효한 다각형의 최소 점 수 (기본 3)

    Returns:
        성공: (표준화된 dict, None)
        실패: (None, 에러 메시지 str)

    표준 출력 dict 형식:
        {
            "type": "polygon",
            "label": "crack",
            "points": [[10, 20], [50, 20], [50, 80]],
            "label_sensitive": false
        }
    """

    # ── 1. 라벨 검증 (bbox와 동일 규칙) ──
    label = raw_ann.get('label')
    if label is None:
        return None, "polygon 라벨에 'label' 필드가 없습니다."
    if not isinstance(label, str):
        return None, f"label은 문자열이어야 합니다. (입력: {type(label).__name__})"
    label = label.strip()
    if not label:
        return None, "label이 빈 문자열입니다."

    # ── 2. points 필드 존재 여부 ──
    raw_points = raw_ann.get('points')
    if raw_points is None:
        return None, "polygon 라벨에 'points' 필드가 없습니다."
    if not isinstance(raw_points, (list, tuple)):
        return None, f"'points'는 리스트여야 합니다. (입력: {type(raw_points).__name__})"

    # ── 3. 최소 점 수 검사 ──
    if len(raw_points) < min_points:
        return None, (
            f"다각형은 최소 {min_points}개의 점이 필요합니다. "
            f"(현재: {len(raw_points)}개)"
        )

    # ── 4. 이미지 해상도 유효성 ──
    if img_w <= 0 or img_h <= 0:
        return None, f"이미지 해상도가 유효하지 않습니다. ({img_w}×{img_h})"

    # ── 5. 각 점 좌표 검증 및 변환 ──
    validated_points = []
    for i, pt in enumerate(raw_points):
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None, (
                f"points[{i}]는 [x, y] 형식이어야 합니다. "
                f"(입력: {pt})"
            )
        try:
            px = float(pt[0])
            py = float(pt[1])
        except (TypeError, ValueError) as e:
            return None, f"points[{i}] 좌표를 숫자로 변환 불가: {e}"

        if clip:
            px = float(np.clip(px, 0.0, float(img_w)))
            py = float(np.clip(py, 0.0, float(img_h)))
        else:
            if not (0 <= px <= img_w and 0 <= py <= img_h):
                return None, (
                    f"points[{i}] = ({px:.1f}, {py:.1f})가 "
                    f"이미지 범위({img_w}×{img_h})를 벗어납니다."
                )

        validated_points.append([round(int(px)), round(int(py))])

    # ── 6. 최종 표준화 포맷 반환 ──
    return {
        'type'            : 'polygon',
        'label'           : label,
        'points'          : validated_points,
        'label_sensitive' : bool(raw_ann.get('label_sensitive', False))
    }, None


def validate_batch(
    raw_boxes: list,
    img_w: int,
    img_h: int,
    clip: bool = True
) -> Tuple[List[dict], List[dict]]:
    """
    여러 바운딩 박스를 일괄 검증한다.

    Returns:
        (valid_boxes, rejected_boxes)
        - valid_boxes  : 검증을 통과한 표준화 dict 목록
        - rejected_boxes: 거부된 항목 목록 [{'index': i, 'reason': '...', 'raw': {...}}, ...]
    """
    valid_boxes = []
    rejected_boxes = []

    for i, raw in enumerate(raw_boxes):
        # v2.0: type 필드로 polygon / bbox 라우팅
        if raw.get('type') == 'polygon':
            formatted, error = validate_polygon(raw, img_w, img_h, clip=clip)
        else:
            formatted, error = validate_and_format(raw, img_w, img_h, clip=clip)

        if formatted is not None:
            valid_boxes.append(formatted)
        else:
            rejected_boxes.append({
                'index': i,
                'reason': error,
                'raw': raw
            })

    return valid_boxes, rejected_boxes


# ──────────────────────────────────────────────────────────
# 단독 테스트 (python validator.py 로 실행 가능)
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("validate_and_format() 자체 검수 테스트")
    print("=" * 55)

    IMG_W, IMG_H = 1280, 720
    test_cases = [
        # (설명, raw_box, 기대 결과)
        ("정상 박스",
         {'label': 'car', 'x': 100, 'y': 50, 'w': 200, 'h': 100}, "OK"),

        ("빈 라벨 거부",
         {'label': '  ', 'x': 100, 'y': 50, 'w': 200, 'h': 100}, "REJECT"),

        ("이미지 밖 좌표 → Clipping",
         {'label': 'person', 'x': -50, 'y': 100, 'w': 300, 'h': 200}, "OK"),

        ("우측 경계 초과 → Clipping",
         {'label': 'truck', 'x': 1200, 'y': 100, 'w': 500, 'h': 100}, "OK"),

        ("음수 w → 거부",
         {'label': 'dog', 'x': 100, 'y': 100, 'w': -10, 'h': 50}, "REJECT"),

        ("필드 누락 (h 없음) → 거부",
         {'label': 'cat', 'x': 100, 'y': 100, 'w': 50}, "REJECT"),

        ("완전히 경계 밖 → Clipping 후 크기 0 → 거부",
         {'label': 'obj', 'x': 1280, 'y': 100, 'w': 100, 'h': 100}, "REJECT"),
    ]

    for desc, raw, expected in test_cases:
        result, err = validate_and_format(raw, IMG_W, IMG_H)
        status = "OK" if result is not None else "REJECT"
        icon = "✅" if status == expected else "❌"
        print(f"{icon} [{desc}] → {status}")
        if err:
            print(f"   사유: {err}")
        elif result:
            print(f"   결과: {result}")
    print("=" * 55)
