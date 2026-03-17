# ML Projects

머신러닝 및 컴퓨터 비전 관련 프로젝트 모음입니다.

## 📁 프로젝트 구조

```
projects/
├── 01_titanic_survival/    # 타이타닉 생존 예측 ML 파이프라인
├── image_processing/       # 이미지 처리 (RGB, HSV, YCbCr 도메인 분석)
├── web_ui_dashboard/       # Object Detection 라벨링 & 증강 웹 시스템
└── README.md
```

---

## 🚢 01_titanic_survival

타이타닉 데이터셋 기반 생존 예측 End-to-End ML 파이프라인

- **기술**: 데이터 전처리, AutoML(TPOT), 검증 테스트
- **실행**: `cd 01_titanic_survival && conda env create -f environment.yml && conda activate titanic_env && cd src && python main.py`
- **상세**: [01_titanic_survival/README.md](01_titanic_survival/README.md)

---

## 🖼️ image_processing

이미지 도메인 변환 및 분석 (RGB, HSV, YCbCr)

- **구성**: assets, data, docs, notebooks, presentations, src
- **데이터**: `data/raw/image_domain_dataset/` (실행 시 생성)

---

## 🏭 web_ui_dashboard

Object Detection 라벨링 & 증강 통합 웹 시스템 (1조 프로젝트)

- **기능**: BBox/Polygon 라벨링, Homography·Gamma·CLAHE·Cutout·Mosaic 증강, Copy-Paste Augmentation
- **실행**: `cd web_ui_dashboard/share && run.bat` (Windows) 또는 `bash run.sh` (Mac/Linux)
- **상세**: [web_ui_dashboard/share/README.md](web_ui_dashboard/share/README.md)

---

## 🛠️ 환경 설정

- **Python**: 3.8 이상 권장
- **01_titanic_survival**: Conda 환경 사용 (`environment.yml`)
- **web_ui_dashboard**: `pip install -r web_ui_dashboard/share/requirements.txt`

---

## 📄 라이선스

학교 프로젝트용 (재배포 금지)
