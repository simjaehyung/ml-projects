@echo off
chcp 65001 > nul
cls

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║                                                            ║
echo ║   🚀  Object Detection 라벨링 시스템 v2.0                 ║
echo ║   Instance Segmentation (Polygon) + Copy-Paste Aug        ║
echo ║                                                            ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo ⏳  서버를 시작하고 있습니다...
echo.

cd /d "%~dp0"

if not exist "app\app.py" (
    echo ❌ 오류: app\app.py 파일을 찾을 수 없습니다.
    echo.
    pause
    exit /b 1
)

echo ✅ 필요한 라이브러리 확인 중...
python -c "import flask, cv2, numpy; print('   모든 라이브러리 설치됨')" 2>nul

if errorlevel 1 (
    echo ⚠️  필요한 라이브러리가 없습니다.
    echo.
    echo 📥 설치 중...
    pip install -r requirements.txt
    echo.
)

echo.
echo 🌐 웹 서버 시작 중... (잠시만 기다리세요)
echo.

REM 브라우저 자동 오픈 (3초 후)
timeout /t 2 /nobreak > nul
start http://localhost:5000

echo.
echo ╔════════════════════════════════════════════════════════════╗
echo ║  ✅ 서버 실행 중!                                         ║
echo ║                                                            ║
echo ║  🔗 웹 주소: http://localhost:5000                        ║
echo ║     (브라우저가 자동으로 열립니다)                          ║
echo ║                                                            ║
echo ║  📝 사용법:                                                ║
echo ║     1️⃣  🎨 "Box" 또는 "🔷 Polygon" 모드 선택             ║
echo ║     2️⃣  📷 이미지 업로드                                 ║
echo ║     3️⃣  🖱️  Canvas에서 박스 또는 다각형 그리기           ║
echo ║     4️⃣  💾 저장 및 증강 실행                             ║
echo ║                                                            ║
echo ║  📖 상세 가이드: docs/v2.0_QUICKSTART.md                 ║
echo ║                                                            ║
echo ║  🛑 종료: Ctrl+C 를 누르세요                             ║
echo ║                                                            ║
echo ╚════════════════════════════════════════════════════════════╝
echo.

python app/app.py

if errorlevel 1 (
    echo.
    echo ❌ 서버 실행 중 오류가 발생했습니다.
    echo.
    pause
)

exit /b 0
