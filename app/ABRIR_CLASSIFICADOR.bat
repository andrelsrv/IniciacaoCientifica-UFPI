@echo off
setlocal
cd /d "%~dp0"

if not exist "ClassificadorFaltasATP.exe" (
  echo [ERRO] ClassificadorFaltasATP.exe nao encontrado em %~dp0
  pause
  exit /b 1
)

start "" "ClassificadorFaltasATP.exe"
