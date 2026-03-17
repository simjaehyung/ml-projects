# 🏭 1조 Object Detection 라벨링 & 증강 시스템

**버전:** v2.0 🚀 **[Instance Segmentation 지원!]**  
**이전 버전:** v1.1 (Mosaic, Cutout, Gamma, CLAHE)  
**작성일:** 2026년 3월 14일  
**개발팀:** AI 엔지니어 + Computer Vision Specialist

---

## 📦 패키지 구성

```
web_ui_dashboard_share/
├── README.md                      ← 이 파일
├── run.bat                        ← 🟢 Windows: 더블클릭으로 실행
├── run.sh                         ← 🟢 Mac/Linux: 터미널에서 실행
├── requirements.txt               ← 필요한 라이브러리 목록
│
├── app/                           ← 웹 애플리케이션
│   ├── app.py                     ← Flask 서버 (핵심)
│   ├── templates/
│   │   └── index.html             ← 웹 UI (Canvas 라벨링) [v1.1: UI 확장]
│   ├── static/                    ← CSS/JS 리소스
│   └── backend/                   ← 백엔드 모듈
│       ├── validator.py           ← 데이터 검증 [v2.0: Polygon 지원]
│       ├── storage_module.py      ← 안전한 저장
│       └── augmentation_recipe.py ← 증강 파이프라인 [v2.0: Copy-Paste + Polygon Homography]
│
├── project_data/                  ← 저장되는 데이터
│   ├── images/
│   │   ├── original/              ← 원본 이미지
│   │   └── augmented/             ← 증강 이미지
│   ├── labels/                    ← JSON 라벨/레시피
│   ├── metadata/                  ← 데이터셋 인덱스
│   └── logs/                      ← 작업 로그
│
└── docs/                          ← 📖 상세 문서
    ├── report.md                  ← 프로젝트 전체 보고서
    ├── technical_deep_dive.md     ← 초상세 기술 설명서
    ├── roadmap_advanced_features.md ← 고도화 로드맵
    ├── v1.1_update_report.md      ← v1.1 새 기능 설명서
    ├── INTEGRATION_REPORT.md      ← v1.0 + v1.1 통합 보고서
    ├── v2.0_IMPLEMENTATION_REPORT.md ← 🆕 v2.0 상세 구현 보고서
    └── v2.0_QUICKSTART.md         ← 🆕 v2.0 실행 가이드
```

---

## 🚀 빠른 시작 (3초)

### Windows 사용자
```
1. run.bat 더블클릭
2. 브라우저에서 http://localhost:5000 접속
3. 끝!
```

### Mac / Linux 사용자
```bash
bash run.sh
# 또는
python app/app.py
```

---

## ✅ 시스템 요구사항

| 항목 | 요구사항 |
|------|---------|
| Python | 3.8 이상 |
| 브라우저 | Chrome, Firefox, Safari (최신 버전) |
| 메모리 | 4GB 이상 권장 |
| 저장공간 | 1GB 이상 |

---

## 🆕 v2.0 신규 기능

### Instance Segmentation 지원 (자유형 다각형 라벨링)

**🔷 Polygon 모드 추가**
- 📍 Canvas에서 클릭으로 점 추가
- 🔗 Enter 키 또는 첫 점 근처 재클릭으로 도형 자동 완성
- 🎯 Esc 키로 취소
- ✅ 정규식 JSON: `{type:"polygon", points:[[x1,y1],...], label:...}`

**🔀 Polygon-aware 증강**
- Homography 변환 시 N개 모든 점 벡터화 처리 (bbox 4점만 아님)
- 원근 나눗셈(Perspective Divide) 적용

### Copy-Paste Augmentation (SOTA 기법)

**🎨 정밀한 객체 합성**
1. Polygon 마스크로 정확한 객체 경계 추출
2. 배경 픽셀 0인 "누끼" 이미지 생성
3. 무작위 배경 이미지에 합성
4. 객체 좌표 자동 동기화

**📊 효과**
- 배경 오염 완전 제거
- Instance Segmentation 학습 데이터 품질 ⬆️
- YOLO v8 등 최신 모델과 호환

---

## ✨ v1.1 기능 (v2.0에 통합)

### 고급 증강 기법 4가지
- 수식: `V_out = 255 × (V_in/255)^γ`
- 특징: **cv2.LUT로 극도로 최적화** (지수 연산 24,000× 감소)
- 용도: 어두운/밝은 이미지 자동 정규화

**2️⃣ CLAHE (히스토그램 평탄화)**
- BGR → YCrCb → Y 채널만 처리 → BGR 복원
- 색상 왜곡 없이 대비만 향상
- 효과: 세부 정보 강조 (도로 표지판, 번호판)

**3️⃣ Cutout / Random Erasing (오클루전)**
- 이미지의 무작위 영역을 마스킹 (2%~15% 면적)
- 부분 가려짐 상황 학습 강화
- BBox drop 옵션: 완전히 가려진 객체는 자동 제거

**4️⃣ Mosaic Augmentation (4-이미지 합성)**
- YOLOv4 도입 기법
- 4장의 이미지를 사분면에 배치 → 1장으로 합성
- 모든 BBox 좌표 자동 변환 + AABB 클리핑
- 소형 객체 탐지 성능 향상

### 웹 UI 개선
- 체크박스 4개 + 동적 슬라이더 5개 추가
- 각 증강 파라미터 실시간 조절 가능
- Mosaic 경고: 이미지 2장 이상 필요

---

## 📚 사용 방법

### 1️⃣ 이미지 업로드
- "📁 이미지 열기" 버튼 클릭
- JPG/PNG 파일 선택

### 2️⃣ 바운딩 박스 그리기
- Canvas 위에서 마우스 드래그
- 실시간 점선 박스 미리보기
- 좌표는 자동으로 원본 이미지 기준으로 변환됨

### 3️⃣ 라벨 설정
- 우측 패널에서 라벨명 입력 (기본값: "object")
- "방향 민감 라벨" 체크 (화살표, 6↔9 등)

### 4️⃣ 박스 관리
- "↩ Undo (Ctrl+Z)" - 마지막 작업 취소
- "🗑 전체 삭제" - 모든 박스 초기화
- ✕ 버튼 - 개별 박스 삭제

### 5️⃣ 증강 설정 [v1.1 확장]

**기본 증강 (v1.0)**
- ✅ 호모그래피 (기본: ON)
- ✅ Poisson 노이즈 (기본: ON)
- ☐ 빛 반사 / 그림자 (선택)

**신규 증강 (v1.1)**
- ☐ 감마 보정 (Gamma)
  - γ 최솟값: 0.20~1.00
  - γ 최댓값: 1.00~2.50
- ☐ CLAHE (히스토그램)
  - Clip Limit: 1.0~5.0
- ☐ Cutout (Random Erasing)
  - 최대 면적: 2%~15%
- ☐ Mosaic (4-이미지 합성)
  - 자동 모드 (이미지 풀 2장 이상 필요)

- 슬라이더: 생성할 증강 이미지 수 (1~20)

### 6️⃣ 저장 및 증강
- "💾 저장 및 증강 실행" 클릭
- 자동 처리:
  - ✅ 원본 이미지 저장
  - ✅ 라벨 JSON 저장
  - ✅ 증강 이미지 생성
  - ✅ 증강 파라미터 저장
  - ✅ 메타데이터 갱신

---

## 📊 저장되는 데이터 구조

### 파일 위치
```
project_data/
├── images/original/
│   └── {image_id}_{filename}.jpg         ← 업로드한 원본 이미지
├── images/augmented/
│   └── {image_id}_{base}_aug{n}.jpg      ← 미리보기용 증강 이미지
├── labels/
│   ├── {image_id}.json                   ← 원본 어노테이션
│   ├── {image_id}_recipe_000.json        ← 증강 레시피 #0
│   ├── {image_id}_recipe_001.json        ← 증강 레시피 #1
│   └── ...
├── metadata/
│   └── dataset_index.json                ← 전체 데이터셋 인덱스
└── logs/
    └── storage.log                       ← 작업 이력
```

### JSON 형식
```json
{
  "image_id": "a1b2c3d4",
  "image_name": "sample.jpg",
  "image_width": 1280,
  "image_height": 720,
  "annotations": [
    {
      "label": "car",
      "x": 100,
      "y": 50,
      "w": 200,
      "h": 100,
      "label_sensitive": false
    }
  ],
  "applied_augmentations": [
    {
      "h31": -0.0005,
      "h32": 0.0003,
      "theta_deg": 2.1,
      "poisson_lambda": 28.5,
      "gamma": 1.2,
      "clahe_clip_limit": 2.0,
      "cutout_area_max": 0.08,
      "mosaic": true
    }
  ]
}
```

---

## 🔧 주요 기능 설명

### 좌표 변환 시스템
```
JavaScript (Canvas) 픽셀 좌표
    ↓ (스케일 비율로 변환)
원본 이미지 기준 좌표 (저장됨)
    ↓ (증강 시)
Homography 변환 + AABB 계산
+ Gamma / CLAHE (좌표 불변)
+ Cutout (BBox 드롭 옵션)
+ Mosaic (scale + offset + clipping)
    ↓
딥러닝 모델 학습에 사용
```

### Undo (실행 취소)
- 전체 상태 스냅샷 방식
- `Ctrl+Z` 또는 "↩ Undo" 버튼
- 박스 추가/삭제 시마다 자동 저장

### 데이터 검증
- 빈 라벨 거부 ❌
- 음수/0 크기 거부 ❌
- 이미지 경계 초과 시 Clipping ✅
- 1px 미만 박스 거부 ❌

### v1.0 증강 파이프라인
1. **Homography** (원근 + 회전)
   - 3×3 변환 행렬
   - BBox 4개 꼭짓점 변환 → AABB

2. **Poisson Noise** (물리적 노이즈)
   - 카메라 센서 노이즈 모사
   - 밝기에 비례하는 분산

3. **Specular/Shadow** (조명 효과)
   - 반사 및 그림자 합성

### v1.1 신규 증강 파이프라인
- **Gamma Correction**: 비선형 밝기 조정
- **CLAHE**: 히스토그램 평탄화 (색상 보존)
- **Cutout**: 오클루전 시뮬레이션
- **Mosaic**: 4-이미지 합성 (맥락적 다양성)

---

## 📖 상세 문서

### 🔴 report.md
프로젝트 전체 개요:
- 구현 완료 항목 vs 미구현 항목
- 설계 결정 근거
- 향후 고도화 방향

### 🟡 technical_deep_dive.md
**100KB 초상세 기술 설명서:**
- validator.py 8단계 검증 로직
- storage_module.py Read-Modify-Write 패턴
- augmentation_recipe.py Homography 수학
- 각 함수 라인 단위 분석
- 실제 계산 예시 포함

### 🟢 roadmap_advanced_features.md
고도화 기능 추천:
- A축: 이미지 증강 (7가지)
- B축: UI 개선 (3가지)
- C축: 데이터 포맷 (3가지)
- D축: ML 기능 (2가지)
- 난이도, 효과, 소요시간 포함

### 🟣 v1.1_update_report.md [v1.1]
**v1.1 업데이트 상세 보고서:**
- 4개 신규 함수의 수학적 배경
- LUT 최적화, YCrCb 색공간, AABB 클리핑 설명
- BBox 좌표 변환 단계별 설명
- 파이프라인 연동 방식
- 단위 테스트 결과
- 검증 체크리스트

### 🆕 v2.0_QUICKSTART.md [v2.0]
**최초 사용자 필독:**
- 3단계 빠른 시작
- Polygon 모드 완벽 가이드
- Copy-Paste Augmentation 사용법
- FAQ 및 트러블슈팅

### 🆕 v2.0_IMPLEMENTATION_REPORT.md [v2.0]
**기술자 필독:**
- 전체 구현 사항 (파일별 변경사항)
- 신규 함수 상세 설명
- 수학적 정당성 및 설계 이유
- 통합 테스트 결과

---

## ⚠️ 문제 해결

### 문제: 포트 5000이 이미 사용 중
```bash
# app.py 마지막 줄 수정
app.run(debug=True, port=5001)  # 5000 → 5001
```

### 문제: 한글 경로에서 이미지 로드 안 됨
→ 이미 해결됨! `np.fromfile()` + `cv2.imdecode()` 사용

### 문제: 업로드 실패
→ 파일명에 특수문자 없는지 확인  
→ 파일 크기 32MB 미만인지 확인

### 문제: JSON 저장 오류
→ 디스크 여유공간 확인 (최소 1GB)  
→ 폴더 권한 확인

### 문제: Mosaic 작동 안 함
→ `project_data/images/original/`에 이미지 2장 이상 필요  
→ 첫 번째 이미지와 다른 이미지 사용 권장

---

## 🔑 기술 특징

### v1.0 (기초)
- ✨ **validator.py**: 데이터 검증 모듈
  - BoundingBox dataclass
  - 8단계 방어 로직
  - Clipping 처리

- ✨ **storage_module.py**: UI 독립형 저장 시스템
  - save_all_data() 파사드 함수
  - 원자적 파일 쓰기
  - 자동 롤백 메커니즘

- ✨ **augmentation_recipe.py**: On-the-fly 증강
  - Homography 행렬 변환
  - Poisson 노이즈
  - OnTheFlyAugmenter (PyTorch Dataset 호환)

### v1.1 (확장) [신규]
- ✨ **Gamma Correction**: LUT 기반 극도로 최적화
- ✨ **CLAHE**: 색상 왜곡 없는 히스토그램
- ✨ **Cutout**: 지능형 오클루전 (BBox drop)
- ✨ **Mosaic**: YOLOv4 스타일 4-이미지 합성
- ✨ **UI 확장**: 동적 슬라이더 5개 추가

---

## 📞 팀 협업

이 패키지는 다음 팀원들이 함께 개발했습니다:
- 프론트엔드 UI 개발
- Flask 서버 통합
- 백엔드 데이터 처리
- 증강 파이프라인 설계
- v1.1 고급 증강 기법 추가 ← **[신규]**

### 다음 기여자를 위해
백엔드 모듈이 완벽히 분리되어 있으므로:
- 라벨링 UI만 수정 가능
- 데이터 포맷 변경 가능
- 증강 기법 추가 가능 ← **v1.1에서 예시 제공됨**
- 다른 프레임워크(PyQt5, Streamlit)로 교체 가능

---

## 📅 버전 이력

### v1.0 (2026-03-15)
- ✅ 프론트엔드 UI (Canvas + Undo)
- ✅ Flask 서버 (REST API)
- ✅ 기본 증강 (Homography + Poisson)
- ✅ 백엔드 모듈 3개
- ✅ 초상세 문서 3개

### v1.1 (2026-03-15) [신규]
- ✅ Gamma Correction (감마 보정)
- ✅ CLAHE (히스토그램 평탄화)
- ✅ Cutout (Random Erasing)
- ✅ Mosaic (4-이미지 합성)
- ✅ UI 확장 (슬라이더 5개)
- ✅ v1.1 업데이트 보고서

### v1.2 (계획)
- Color Jitter 증강
- 박스 선택/이동 UI
- YOLO 형식 변환
- 통계 대시보드

---

## 🎯 지원 방법

문제가 생기면:
1. `project_data/logs/storage.log` 확인
2. 브라우저 개발자 도구 (F12) → Console 탭 확인
3. 터미널 메시지 확인

---

## 📄 라이선스

학교 프로젝트용 (재배포 금지)

---

**행운을 빕니다! 🚀**
