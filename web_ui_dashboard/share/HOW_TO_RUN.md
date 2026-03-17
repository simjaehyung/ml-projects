# 🚀 run.bat 실행 가이드 (Windows)

## ✅ 빠른 시작 (3단계)

### 1️⃣ **run.bat 파일 찾기**
```
share/run.bat ← 이 파일
```

### 2️⃣ **더블클릭으로 실행**
- 마우스 왼쪽으로 `run.bat` **더블클릭**
- (또는 우클릭 → "실행" 선택)

### 3️⃣ **브라우저에서 접속**
```
http://localhost:5000
```

---

## 📋 run.bat가 자동으로 하는 일

```
1. ✅ Python 버전 확인 (3.8 이상 필요)
2. ✅ 필요한 라이브러리 설치 (pip install)
   - Flask
   - OpenCV
   - NumPy
   - Pillow
   - 등등...
3. ✅ Flask 서버 시작
4. ✅ http://localhost:5000 자동으로 준비됨
```

---

## ⚠️ 만약 실행이 안 되면

### 🔴 문제 1: "Python을 찾을 수 없습니다"
**해결책:**
1. Python 3.8 이상 설치
   - https://www.python.org/downloads/
   - 설치 시 **반드시** "Add Python to PATH" ☑️ 체크!
2. 컴퓨터 재부팅
3. run.bat 다시 더블클릭

### 🔴 문제 2: "라이브러리 설치 실패"
**해결책:**
```bash
# 터미널/PowerShell을 share 폴더에서 열고:
pip install -r requirements.txt
python app/app.py
```

### 🔴 문제 3: "포트 5000이 이미 사용 중"
**해결책:**
- 이미 다른 Flask 앱이 5000 포트를 사용 중
- 다른 프로그램을 먼저 종료하거나
- `app/app.py` 마지막 줄 수정:
  ```python
  app.run(debug=True, port=5001)  # 5000 → 5001
  ```

### 🔴 문제 4: CMD 창이 바로 닫혀버림
**해결책:**
- run.bat 내 마지막 `pause` 때문에 엔터 키를 누르면 닫힘
- 더블클릭이 아니라 **우클릭 → "편집"** 으로 열어서 내용 확인

---

## 📝 수동 실행 (터미널 사용)

더블클릭이 안 되면 수동으로:

### PowerShell (Windows 10+)
```powershell
cd "C:\Users\jhsim\Erica261\M.L\projects\web_ui_dashboard\share"
python app/app.py
```

### CMD
```cmd
cd C:\Users\jhsim\Erica261\M.L\projects\web_ui_dashboard\share
python app/app.py
```

---

## ✨ 준비 완료!

이제 브라우저에서:
```
http://localhost:5000
```

으로 접속하면 웹 UI가 열립니다! 🎉

---

**문제가 지속되면:**
1. `share\project_data\logs\storage.log` 파일 확인
2. 브라우저 F12 → Console 탭 에러 메시지 확인
3. 터미널 메시지 스크린샷 저장 후 공유
