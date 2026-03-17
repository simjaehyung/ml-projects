#!/bin/bash

clear

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║   🚀  Object Detection 라벨링 시스템 v2.0                 ║"
echo "║   Instance Segmentation (Polygon) + Copy-Paste Aug        ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "⏳  서버를 시작하고 있습니다..."
echo ""

# 현재 스크립트 위치로 이동
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# app.py 존재 확인
if [ ! -f "app/app.py" ]; then
    echo "❌ 오류: app/app.py 파일을 찾을 수 없습니다."
    echo ""
    read -p "계속하려면 Enter 키를 누르세요..."
    exit 1
fi

echo "✅ 필요한 라이브러리 확인 중..."
python3 -c "import flask, cv2, numpy" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  필요한 라이브러리가 없습니다."
    echo ""
    echo "📥 설치 중..."
    pip3 install -r requirements.txt
    echo ""
fi

echo ""
echo "🌐 웹 서버 시작 중... (잠시만 기다리세요)"
echo ""

# 브라우저 자동 오픈 (2초 후)
sleep 2

# macOS와 Linux에서 다르게 처리
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open http://localhost:5000
else
    # Linux
    xdg-open http://localhost:5000 2>/dev/null || echo "🔗 브라우저에서 http://localhost:5000 을 열어주세요"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ 서버 실행 중!                                         ║"
echo "║                                                            ║"
echo "║  🔗 웹 주소: http://localhost:5000                        ║"
echo "║     (브라우저가 자동으로 열립니다)                          ║"
echo "║                                                            ║"
echo "║  📝 사용법:                                                ║"
echo "║     1️⃣  🎨 \"Box\" 또는 \"🔷 Polygon\" 모드 선택           ║"
echo "║     2️⃣  📷 이미지 업로드                                 ║"
echo "║     3️⃣  🖱️  Canvas에서 박스 또는 다각형 그리기           ║"
echo "║     4️⃣  💾 저장 및 증강 실행                             ║"
echo "║                                                            ║"
echo "║  📖 상세 가이드: docs/v2.0_QUICKSTART.md                 ║"
echo "║                                                            ║"
echo "║  🛑 종료: Ctrl+C 를 누르세요                             ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

python3 app/app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 서버 실행 중 오류가 발생했습니다."
    echo ""
    read -p "계속하려면 Enter 키를 누르세요..."
fi

exit 0
