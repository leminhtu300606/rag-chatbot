@echo off
rem Tat server RAG dang nghe tren port 8000 (neu co).
setlocal
set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:8000" ^| findstr "LISTENING"') do (
  set FOUND=1
  echo [stop-server] Tat PID %%a ...
  taskkill /F /PID %%a
)
if "%FOUND%"=="0" echo [stop-server] Khong co server nao tren port 8000.
