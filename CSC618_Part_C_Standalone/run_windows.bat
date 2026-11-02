@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py part_c.py
) else (
  python part_c.py
)
if errorlevel 1 (echo. & echo Experiment failed. See README.md for setup. & exit /b 1)
echo.
echo SUCCESS: Check results folder.
