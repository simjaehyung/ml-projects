#!/bin/bash

# ===================================================
# 1조 Object Detection 라벨링 & 증강 시스템
# Mac/Linux용 시작 스크립트
# ===================================================

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")/.."

echo ""
echo "========================================"
echo "  🏭 Object Detection 라벨링 시스템"
echo "========================================"
echo ""

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3을 찾을 수 없습니다."
    echo "   Python 3.8 이상을 설치하세요."
    echo "   Mac: brew install python3"
    echo "   Linux: sudo apt-get install python3"
    exit 1
fi

echo "✅ Python 설치됨"
python3 --version

# 필요한 라이브러리 설치
echo ""
echo "📦 필요한 라이브러리 설치 중..."
pip3 install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ 라이브러리 설치 실패"
    exit 1
fi

echo "✅ 설치 완료"

# Flask 서버 실행
echo ""
echo "🚀 서버를 시작합니다..."
echo ""
echo "========================================"
echo "  💻 http://localhost:5000"
echo "  ⌨️  서버 중지: Ctrl+C"
echo "========================================"
echo ""

cd app
python3 app.py
