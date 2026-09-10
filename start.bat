@echo off
echo ========================================
echo    MsScope — Démarrage du système
echo ========================================

echo.
echo [1/3] Démarrage d'Ollama...
start /B "" "C:\Users\loulouu\AppData\Local\Programs\Ollama\ollama.exe" serve
timeout /t 3 /nobreak > nul

echo [2/3] Démarrage de MsScope avec Docker...
docker-compose up -d

echo [3/3] MsScope est prêt !
echo.
echo Ouvre ton navigateur sur : http://localhost:8501
echo.
pause
