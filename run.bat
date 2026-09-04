@echo off
REM filter-inspection entry point (Windows).
REM
REM     conda activate filter-inspect
REM     run.bat            launch the camera GUI
REM     run.bat test       run selftest.py -- verify the install first
REM
REM Both forms go through NIRcam-first\run_gui.bat so there is ONE place that
REM resolves the interpreter and sanitises the Qt / MVS environment.

setlocal EnableExtensions

set "ARGS=%*"
if /i "%~1"=="test" (
    set "RUN_SCRIPT=%~dp0selftest.py"
    set "ARGS="
)

call "%~dp0NIRcam-first\run_gui.bat" %ARGS%
exit /b %ERRORLEVEL%
