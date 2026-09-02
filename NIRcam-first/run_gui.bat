@echo off
REM Launch the NIRcam-first GUI with the hybrid EfficientAD backend (Windows).
REM
REM   run_gui.bat
REM
REM Set ENV_PY first to use a different interpreter:
REM   set ENV_PY=C:\path\to\python.exe && run_gui.bat

setlocal
set "HERE=%~dp0"

if not defined ENV_PY set "ENV_PY=%USERPROFILE%\miniconda3\envs\nircam\python.exe"

if not exist "%ENV_PY%" (
    echo interpreter not found: %ENV_PY% 1>&2
    echo create it with:  conda create -n nircam python=3.10 -y 1>&2
    exit /b 1
)

REM Ignore the per-user site-packages directory. See run_gui.sh for why --
REM a `pip install --user` there shadows the env's pinned versions.
set PYTHONNOUSERSITE=1

cd /d "%HERE%"
"%ENV_PY%" BasicDemo.py %*
endlocal
