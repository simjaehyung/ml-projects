@echo off
REM ===================================================
REM 1조 Object Detection 라벨링 & 증강 시스템
REM Windows용 시작 스크립트 (v1.1)
REM ===================================================
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 현재 디렉토리를 share 폴더로 설정
cd /d "%~dp0"

cls
echo.
echo ╔════════════════════════════════════════════════════╗
echo ║     🏭 Object Detection 라벨링 ^& 증강 시스템      ║
echo ║                      v1.1                          ║
echo ╚════════════════════════════════════════════════════╝
echo.

REM Python 버전 확인
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python을 찾을 수 없습니다.
    echo.
    echo    Python 3.8 이상을 설치하세요:
    echo    https://www.python.org/downloads/
    echo.
    echo    설치 시 반드시 "Add Python to PATH" 체크하세요!
    echo.
    pause
    exit /b 1
)

echo ✅ Python 설치됨
python --version
echo.

REM 필요한 라이브러리 설치
echo 📦 필요한 라이브러리 설치 중...
echo.
pip install -q -r requirements.txt

if errorlevel 1 (
    echo.
    echo ❌ 라이브러리 설치 실패
    echo.
    pause
    exit /b 1
)

echo ✅ 설치 완료
echo.

REM Flask 서버 실행
echo 🚀 Flask 서버를 시작합니다...
echo.
echo ═══════════════════════════════════════════════════
echo  💻 브라우저에서 열기: http://localhost:5000
echo  ⌨️  서버 중지: Ctrl+C 누르기
echo ═══════════════════════════════════════════════════
echo.

python app/app.py

pause
