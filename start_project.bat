@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py -3 "%SCRIPT_DIR%start_project.py" %*
    exit /b %errorlevel%
)

python "%SCRIPT_DIR%start_project.py" %*
exit /b %errorlevel%
