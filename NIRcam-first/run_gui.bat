@echo off
REM Launch the NIRcam-first GUI (Windows). Mirror of run_gui.sh.
REM
REM Normal use -- activate the environment, then run:
REM
REM     conda activate filter-inspect
REM     run_gui.bat
REM
REM The launcher uses the ACTIVE conda env (%CONDA_PREFIX%). If none is
REM active it falls back to searching the usual conda roots for %ENV_NAME%.
REM Override either one explicitly:
REM
REM     set "ENV_NAME=some-other-env" && run_gui.bat
REM     set "ENV_PY=C:\path\to\python.exe" && run_gui.bat

setlocal EnableExtensions
set "HERE=%~dp0"

if not defined ENV_NAME set "ENV_NAME=filter-inspect"

REM --- interpreter resolution, most specific first -------------------------
if defined ENV_PY if exist "%ENV_PY%" goto :have_py

REM The env the caller activated. This is the intended path.
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
    set "ENV_PY=%CONDA_PREFIX%\python.exe"
    goto :have_py
)

REM No env active -- look for ENV_NAME under the usual conda roots so that
REM double-clicking this file still works.
for %%R in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "C:\ProgramData\anaconda3"
    "C:\ProgramData\miniconda3"
) do (
    if exist "%%~R\envs\%ENV_NAME%\python.exe" (
        set "ENV_PY=%%~R\envs\%ENV_NAME%\python.exe"
        goto :have_py
    )
)

echo. 1>&2
echo   no python found for environment "%ENV_NAME%". 1>&2
echo. 1>&2
echo   activate it first:      conda activate %ENV_NAME% 1>&2
echo   or create it:           conda create -n %ENV_NAME% python=3.10 -y 1>&2
echo   or name one directly:   set "ENV_PY=C:\path\to\python.exe" 1>&2
echo. 1>&2
exit /b 1

:have_py
echo interpreter: %ENV_PY%

REM Ignore the per-user site-packages directory. See run_gui.sh for why --
REM a `pip install --user` there shadows the env's pinned versions.
set PYTHONNOUSERSITE=1

REM --- Qt environment hygiene ----------------------------------------------
REM Do NOT inherit the caller's Qt variables. A QT_QPA_PLATFORM_PLUGIN_PATH
REM left over from a different conda env loads that env's plugins against
REM this env's Qt and aborts the GUI. Same failure class as the xcb abort
REM documented in run_gui.sh; on Windows it surfaces as
REM "could not find or load the Qt platform plugin windows".
REM Probed via a temp file rather than `for /f ... in (\`...\`)`: that form
REM re-parses the command through cmd and mangles a quoted interpreter path
REM combined with a quoted -c argument.
set "QT_PLUGIN_PATH="
set "QT_DEBUG_PLUGINS="
set "QT_QPA_PLATFORM_PLUGIN_PATH="
set "QT_PLUGINS="
set "QT_PROBE=%TEMP%\nircam_qt_%RANDOM%.txt"
"%ENV_PY%" -c "import PyQt5,os;print(os.path.join(os.path.dirname(PyQt5.__file__),'Qt5','plugins','platforms'))" > "%QT_PROBE%" 2>nul
if exist "%QT_PROBE%" set /p QT_PLUGINS=<"%QT_PROBE%"
del "%QT_PROBE%" 2>nul
if defined QT_PLUGINS if exist "%QT_PLUGINS%" set "QT_QPA_PLATFORM_PLUGIN_PATH=%QT_PLUGINS%"

REM --- Hikvision MVS runtime ------------------------------------------------
REM MvCameraControl.dll normally comes from the system PATH that the MVS
REM installer sets. Prepend it defensively so the GUI works from a shell that
REM has a trimmed PATH.
if exist "%CommonProgramFiles(x86)%\MVS\Runtime\Win64_x64" (
    set "PATH=%CommonProgramFiles(x86)%\MVS\Runtime\Win64_x64;%PATH%"
)

REM RUN_SCRIPT lets run.bat reuse this file's interpreter resolution and
REM environment hygiene for selftest.py. Defaults to the GUI.
if not defined RUN_SCRIPT set "RUN_SCRIPT=BasicDemo.py"

cd /d "%HERE%"
"%ENV_PY%" "%RUN_SCRIPT%" %*
endlocal
