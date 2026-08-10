@echo off
setlocal
cd /d "%~dp0"

if not exist "classificador_gui_v2.py" (
  echo [ERRO] classificador_gui_v2.py nao encontrado em %~dp0
  echo Verifique se este .bat esta na pasta src/ do projeto.
  pause
  exit /b 1
)

where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw "%~dp0classificador_gui_v2.py"
  exit /b 0
)

where python >nul 2>nul
if %errorlevel%==0 (
  python "%~dp0classificador_gui_v2.py"
  if errorlevel 1 (
    echo.
    echo [ERRO] O classificador fechou com um erro. Veja a mensagem acima.
    pause
  )
  exit /b 0
)

echo [ERRO] Python nao encontrado no PATH.
echo Instale o Python 3 (python.org) e marque "Add python.exe to PATH" na instalacao.
pause
exit /b 1
