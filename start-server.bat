@echo off
rem Khoi dong sach RAG Chatbot: tat server cu tren port 8000 (neu co) roi chay uvicorn.
rem Dung: double-click hoac chay start-server.bat trong cmd (giua Ctrl+C de dung).
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
  echo [LOI] Khong thay venv\Scripts\activate.bat - chay tu thu muc goc du an E:\rag
  pause
  exit /b 1
)
call "venv\Scripts\activate.bat"

for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:8000" ^| findstr "LISTENING"') do (
  echo [start-server] Tat server cu PID %%a ...
  taskkill /F /PID %%a >nul 2>&1
  timeout /t 2 /nobreak >nul
)

echo [start-server] Dang chay: uvicorn backend.api:app --host 127.0.0.1 --port 8000
echo [start-server] Mo trinh duyet: http://127.0.0.1:8000/
uvicorn backend.api:app --host 127.0.0.1 --port 8000
