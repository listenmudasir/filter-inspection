<#
.SYNOPSIS
    Build the FilterInspection Windows installer.

.DESCRIPTION
    1. Reads the version number from NIRcam-first/version.py
    2. Sanity-checks weights/supervised_global.pt is the real (not dummy) file
    3. Compiles nircam_launcher.pyw and BasicDemo.py with Nuitka
    4. Compiles the Inno Setup installer script
    5. Moves the resulting setup exe into dist/ with the version in its name

    Run from the repo root:
        .\build.ps1

    Requires (see PACKAGING.md section 3.1):
        - .venv-build: a venv made from OFFICIAL python.org CPython 3.10 with
          this project's dependencies + nuitka. NOT an Anaconda env -- Nuitka
          standalone builds from Anaconda Python on Windows crash on
          `import ctypes` (0xC0000409 in ucrtbase.dll), and the camera SDK
          needs ctypes.
        - MSVC Build Tools (C++ build tools workload)
        - Inno Setup 6
#>

param(
    [string]$PythonExe = (Join-Path $PSScriptRoot ".venv-build\Scripts\python.exe"),
    [string]$IsccPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    [int]$MinWeightsSizeMB = 100,
    [switch]$SkipLaunchTest
)

# Not "Stop": Nuitka writes progress to stderr, and when the caller redirects
# (.\build.ps1 *>&1 | Out-File ...) Windows PowerShell 5.1 turns each native
# stderr line into an ErrorRecord -- under "Stop" the first progress line
# aborts the build with no message. Every step checks $LASTEXITCODE / Test-Path
# and calls Fail explicitly instead.
$ErrorActionPreference = "Continue"
$RepoRoot = $PSScriptRoot
$NirDir = Join-Path $RepoRoot "NIRcam-first"
$BuildDir = Join-Path $RepoRoot "build"
$DistDir = Join-Path $RepoRoot "dist"
$WeightsFile = Join-Path $RepoRoot "weights\supervised_global.pt"

# Nuitka resolves imports from the current directory. supervised_detect.py
# reaches the repo-root `inspection` package (core algorithm) through a
# runtime sys.path hack Nuitka can't follow, so it is only compiled in when
# the build runs from the repo root. Don't depend on the caller for that.
Set-Location $RepoRoot

function Fail($msg) {
    Write-Host "BUILD FAILED: $msg" -ForegroundColor Red
    exit 1
}

# --- 1. Version -----------------------------------------------------------
$versionFile = Join-Path $NirDir "version.py"
if (-not (Test-Path $versionFile)) {
    Fail "NIRcam-first/version.py not found. See PACKAGING.md section 9."
}
$versionLine = Select-String -Path $versionFile -Pattern 'VERSION\s*=\s*"([^"]+)"'
if (-not $versionLine) {
    Fail "Could not parse VERSION from $versionFile"
}
$Version = $versionLine.Matches[0].Groups[1].Value
Write-Host "Building FilterInspection v$Version" -ForegroundColor Cyan

# --- 2. Weights sanity check ------------------------------------------------
if (-not (Test-Path $WeightsFile)) {
    Fail "weights/supervised_global.pt not found. Copy the real checkpoint in first (see weights/README.md)."
}
$weightsSizeMB = (Get-Item $WeightsFile).Length / 1MB
Write-Host ("weights/supervised_global.pt size: {0:N1} MB" -f $weightsSizeMB)
if ($weightsSizeMB -lt $MinWeightsSizeMB) {
    Fail "weights/supervised_global.pt is only $([math]::Round($weightsSizeMB,1)) MB (< $MinWeightsSizeMB MB). " + `
         "This looks like the dummy test checkpoint -- swap in the real 274MB file before building a release. See PACKAGING.md section 4."
}

# --- 2b. Toolchain preflight -------------------------------------------------
if (-not (Test-Path $PythonExe)) {
    Fail "Python not found at $PythonExe. Create .venv-build first (PACKAGING.md section 3.1)."
}
$flavor = & $PythonExe -c "import sys; print('anaconda' if ('conda' in sys.version.lower() or 'anaconda' in sys.version.lower()) else 'cpython')"
if ($flavor -eq "anaconda") {
    Fail "$PythonExe is an Anaconda Python. Nuitka builds from Anaconda crash at runtime on 'import ctypes'. Use .venv-build (official python.org CPython)."
}
if (-not (Test-Path $IsccPath)) {
    Fail "ISCC.exe not found at $IsccPath. Install Inno Setup 6 or pass -IsccPath."
}

# --- 2c. License signing key -------------------------------------------------
# license_check.py falls back to a public placeholder when license_secret.py
# is absent, so the dev tree runs -- but a build with the placeholder lets
# anyone who has seen this repo mint codes for any machine.
$secretFile = Join-Path $NirDir "license_secret.py"
if (-not (Test-Path $secretFile)) {
    Fail "NIRcam-first\license_secret.py not found. Create it once with: python tools\license\new_secret.py  (or copy it from the machine that made the existing codes)."
}
$usesPlaceholder = & $PythonExe -c "import sys; sys.path.insert(0, r'$NirDir'); import license_check as l; print(l.SECRET_KEY == l.PLACEHOLDER_SECRET)"
if ($usesPlaceholder -ne "False") {
    Fail "license_secret.py still holds the placeholder key (or failed to import: $usesPlaceholder)."
}

# --- 3. Clean previous build -------------------------------------------------
# cmd's rd, not Remove-Item: on Windows PowerShell 5.1, Remove-Item -Recurse
# over the ~11k-file torch build tree kills the whole PowerShell process with
# no error, so the script silently stops here.
if (Test-Path $BuildDir) { cmd /c "rd /s /q `"$BuildDir`"" }
if (Test-Path $BuildDir) { Fail "Could not delete $BuildDir -- a file in it is probably locked (a running exe from a previous build?)." }
New-Item -ItemType Directory -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Path $DistDir -Force | Out-Null

# --- 4. Nuitka: splash launcher (onefile) -----------------------------------
Write-Host "`n=== Building nircam_launcher.exe (onefile) ===" -ForegroundColor Cyan
& $PythonExe -m nuitka `
    --assume-yes-for-downloads `
    --onefile `
    --windows-console-mode=disable `
    --enable-plugin=tk-inter `
    --include-module=license_secret `
    --windows-icon-from-ico="$NirDir\assets\nircam_inspection.ico" `
    --include-data-file="$NirDir\assets\splash.png=assets/splash.png" `
    --output-dir="$BuildDir" `
    --output-filename=nircam_launcher.exe `
    "$NirDir\nircam_launcher.pyw"
if ($LASTEXITCODE -ne 0) { Fail "Nuitka failed building nircam_launcher.exe" }

# --- 5. Nuitka: main GUI (standalone folder) --------------------------------
Write-Host "`n=== Building BasicDemo.exe (standalone) ===" -ForegroundColor Cyan
# The Hikvision SDK in MvImport\ imports its siblings by bare name
# (`from CameraParams_const import *`) after its __init__ appends its own
# folder to sys.path at runtime. Nuitka can't see that, so the exe dies with
# "No module named 'CameraParams_const'". Put MvImport\ on PYTHONPATH and
# force those names in, reproducing the dev-tree import behaviour without
# editing the vendor SDK.
$env:PYTHONPATH = Join-Path $NirDir "MvImport"
& $PythonExe -m nuitka `
    --assume-yes-for-downloads `
    --standalone `
    --enable-plugin=pyqt5 `
    --include-module=license_secret `
    --include-module=CameraParams_const `
    --include-module=CameraParams_header `
    --include-module=PixelType_header `
    --include-module=MvErrorDefine_const `
    --include-module=MvCameraControl_class `
    --windows-icon-from-ico="$NirDir\assets\nircam_inspection.ico" `
    --include-data-file="$WeightsFile=weights/supervised_global.pt" `
    --output-dir="$BuildDir" `
    --output-filename=BasicDemo.exe `
    "$NirDir\BasicDemo.py"
if ($LASTEXITCODE -ne 0) { Fail "Nuitka failed building BasicDemo.exe" }
Remove-Item Env:\PYTHONPATH

# --- 5b. Nuitka: fingerprint tool (human-facing, ships to vendor for
# post-install re-activation -- e.g. hardware changed after install and a
# fresh license.key is needed without rerunning the whole installer) -------
# The tools/license scripts import license_check from NIRcam-first via a
# runtime sys.path.insert, which Nuitka's static import analysis can't see --
# without PYTHONPATH + --include-module the exe compiles fine and then dies
# with "No module named 'license_check'".
$env:PYTHONPATH = $NirDir

Write-Host "`n=== Building FingerprintTool.exe (onefile) ===" -ForegroundColor Cyan
& $PythonExe -m nuitka `
    --assume-yes-for-downloads `
    --onefile `
    --include-module=license_check `
    --include-module=license_secret `
    --output-dir="$BuildDir" `
    --output-filename=FingerprintTool.exe `
    "$RepoRoot\tools\license\print_fingerprint.py"
if ($LASTEXITCODE -ne 0) { Fail "Nuitka failed building FingerprintTool.exe" }

# --- 5c. Nuitka: installer-facing activation helper (no human interaction,
# called by installer/setup.iss's wizard page; see PACKAGING.md section 5) -
Write-Host "`n=== Building LicenseActivator.exe (onefile) ===" -ForegroundColor Cyan
& $PythonExe -m nuitka `
    --assume-yes-for-downloads `
    --onefile `
    --include-module=license_check `
    --include-module=license_secret `
    --output-dir="$BuildDir" `
    --output-filename=LicenseActivator.exe `
    "$RepoRoot\tools\license\license_installer_helper.py"
if ($LASTEXITCODE -ne 0) { Fail "Nuitka failed building LicenseActivator.exe" }
Remove-Item Env:\PYTHONPATH

# --- 5d. Smoke test: "Nuitka succeeded" does NOT mean the exe runs ----------
# Both failures hit on the first real build compiled cleanly and only crashed
# when executed. Also catches antivirus (Trend Micro Apex One on the dev
# machine) silently deleting a freshly built onefile exe.
Write-Host "`n=== Smoke-testing built executables ===" -ForegroundColor Cyan
foreach ($exe in @("nircam_launcher.exe", "FingerprintTool.exe", "LicenseActivator.exe", "BasicDemo.dist\BasicDemo.exe")) {
    if (-not (Test-Path (Join-Path $BuildDir $exe))) {
        Fail "$exe is missing from build\ -- if Nuitka reported success, antivirus probably deleted it. Add build\ to the AV exclusion list and rebuild."
    }
}
foreach ($mod in @("inspection.enhance", "inspection.gpu_preprocess", "supervised_detect", "license_check", "license_secret",
                   "CameraParams_const", "CameraParams_header", "MvCameraControl_class")) {
    if (-not (Test-Path (Join-Path $BuildDir "BasicDemo.build\module.$mod.c"))) {
        Fail "$mod was not compiled into BasicDemo.exe. Without inspection.* the app silently falls back to the old hybrid detector."
    }
}
$fpOut = Join-Path $BuildDir "smoke_fingerprint.txt"
& (Join-Path $BuildDir "LicenseActivator.exe") fingerprint $fpOut
if ($LASTEXITCODE -ne 0 -or -not (Test-Path $fpOut) -or (Get-Content $fpOut -Raw).Trim().Length -ne 64) {
    Fail "LicenseActivator.exe did not produce a fingerprint (exit $LASTEXITCODE). The installer's license page would show '讀取失敗'."
}
$fp = (Get-Content $fpOut -Raw).Trim()
$chkOut = Join-Path $BuildDir "smoke_check.txt"
& (Join-Path $BuildDir "LicenseActivator.exe") check "not-a-real-code" $chkOut
if ((Get-Content $chkOut -Raw).Trim() -ne "FAIL:MISMATCH") {
    Fail "LicenseActivator.exe check did not reject a bogus code."
}
& $PythonExe (Join-Path $RepoRoot "tools\license\generate_license.py") $fp (Join-Path $BuildDir "smoke_license.key") | Out-Null
& (Join-Path $BuildDir "LicenseActivator.exe") check (Get-Content (Join-Path $BuildDir "smoke_license.key") -Raw).Trim() $chkOut
if (-not (Get-Content $chkOut -Raw).Trim().StartsWith("OK:")) {
    Fail "LicenseActivator.exe rejected a valid code generated by generate_license.py -- SECRET_KEY mismatch between the build and the generator?"
}
Remove-Item $fpOut, $chkOut, (Join-Path $BuildDir "smoke_license.key") -ErrorAction SilentlyContinue
Write-Host "  LicenseActivator.exe: fingerprint OK, bogus code rejected, valid code accepted"

# Launch the real app and wait for its main window. Missing modules from
# runtime sys.path tricks (license_check, MvImport siblings) only show up here.
# Needs this machine's GPU and MVS runtime; pass -SkipLaunchTest where absent.
if (-not $SkipLaunchTest) {
    $appDir = Join-Path $BuildDir "BasicDemo.dist"
    & $PythonExe (Join-Path $RepoRoot "tools\license\generate_license.py") $fp (Join-Path $appDir "license.key") | Out-Null
    $proc = Start-Process -FilePath (Join-Path $appDir "BasicDemo.exe") -WorkingDirectory $appDir -PassThru
    $deadline = (Get-Date).AddSeconds(180)
    $title = ""
    while ((Get-Date) -lt $deadline -and -not $proc.HasExited) {
        $proc.Refresh()
        if ($proc.MainWindowTitle -like "*v$Version*") { $title = $proc.MainWindowTitle; break }
        Start-Sleep -Seconds 2
    }
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force; Start-Sleep -Seconds 3 }
    $appLog = Get-ChildItem (Join-Path $appDir "logs") -Filter *.log -ErrorAction SilentlyContinue | Select-Object -First 1
    $logTail = if ($appLog) { (Get-Content $appLog.FullName -Tail 15) -join "`n" } else { "(no log written)" }
    # The window still opens when the supervised model fails to load -- the app
    # silently falls back to the old hybrid detector (9 false alarms / 51 clean
    # frames vs 0). Treat that as a failed build.
    $modelLoaded = $appLog -and (Select-String -Path $appLog.FullName -Pattern "Supervised global segmenter loaded" -Quiet)
    # Never ship the smoke test's license or logs inside the installer.
    Remove-Item (Join-Path $appDir "license.key") -Force -ErrorAction SilentlyContinue
    cmd /c "rd /s /q `"$(Join-Path $appDir 'logs')`"" 2>$null
    if ((Test-Path (Join-Path $appDir "license.key")) -or (Test-Path (Join-Path $appDir "logs"))) {
        Fail "Could not remove the smoke test's license.key / logs from $appDir -- they would ship in the installer."
    }
    if (-not $title) {
        Fail "BasicDemo.exe never showed its main window (title containing v$Version). Last log lines:`n$logTail"
    }
    if (-not $modelLoaded) {
        Fail "BasicDemo.exe opened but the supervised model did not load (fell back to the old detector). Last log lines:`n$logTail"
    }
    Write-Host "  BasicDemo.exe: main window '$title' appeared, supervised model loaded"
}

# --- 6. Inno Setup -----------------------------------------------------------
Write-Host "`n=== Building installer with Inno Setup ===" -ForegroundColor Cyan
& $IsccPath "/DAppVersion=$Version" "/DBuildDir=$BuildDir" "$RepoRoot\installer\setup.iss"
if ($LASTEXITCODE -ne 0) { Fail "Inno Setup compilation failed" }

# --- 7. Done -----------------------------------------------------------------
$setupExe = Join-Path $DistDir "FilterInspection_Setup_v$Version.exe"
if (-not (Test-Path $setupExe)) {
    Fail "Expected installer not found at $setupExe"
}
$setupSizeMB = (Get-Item $setupExe).Length / 1MB
Write-Host "`nBUILD COMPLETE" -ForegroundColor Green
Write-Host ("  {0}" -f $setupExe)
Write-Host ("  {0:N1} MB" -f $setupSizeMB)
Write-Host "`nBefore handing this to the vendor: run the acceptance checklist in PACKAGING.md section 7 on a clean machine." -ForegroundColor Yellow
