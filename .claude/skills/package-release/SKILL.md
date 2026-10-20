---
name: package-release
description: Build the FilterInspection Windows installer (FilterInspection_Setup_vX.Y.Z.exe) after the app code changes. Use when the user asks to package, build, release, or make a new installer / 打包 / 出新版 / 做安裝檔.
---

# Package a FilterInspection release

Full design and background live in `PACKAGING.md`. This is the operating procedure.

## 1. Preflight (stop and tell the user if any fails)

- `.venv-build\Scripts\python.exe` exists. If not, build it with the commands in
  PACKAGING.md section 3.1. **Never build from a conda/Anaconda env** — the exe
  crashes on `import ctypes`. `build.ps1` refuses Anaconda anyway.
- `C:\Program Files (x86)\Inno Setup 6\ISCC.exe` exists.
- `weights\supervised_global.pt` is the real checkpoint (> 100 MB).
- `git status` is clean, or the user confirms the uncommitted changes belong in this release.
- `NIRcam-first\license_secret.py` exists (gitignored; build.ps1 refuses to run
  without it). If missing, **stop and ask the user** — they either copy it from the
  machine that issued the existing codes, or run `python tools\license\new_secret.py`
  themselves for a first-ever setup. Never create, read out, print, or commit it
  yourself: a new key invalidates every code already issued (PACKAGING.md 5.3).

## 2. Version

Ask the user which part to bump unless they said so, per PACKAGING.md section 2
(patch = fix/tuning, minor = new compatible feature, major = breaking config/model change).
Edit `VERSION` in `NIRcam-first\version.py`. That single value drives the window
title, the installer name, and the installer's AppVersion.

## 3. Build

From the repo root, in PowerShell, run in the background. Measured on the RTX 3080
dev machine: ~65 min on a cold compiler cache, ~35 min afterwards; about 11 min
of either is Inno Setup compressing ~4 GB. Don't poll — wait for completion.

```powershell
.\build.ps1
```

To keep a log, write it OUTSIDE the repo (a file the user has open in an editor
is locked and the pipe fails before the build even starts):

```powershell
.\build.ps1 *>&1 | Out-File "$env:TEMP\filterinspection_build.log" -Encoding utf8
```

It compiles four exes, then smoke-tests them before the slow Inno Setup step:
exe presence, core modules compiled in, LicenseActivator fingerprint/reject/accept
round trip, and it **launches BasicDemo.exe and waits for the main window and the
"Supervised global segmenter loaded" log line**. A window will briefly open on
screen — that's expected. On a machine without GPU/MVS use `-SkipLaunchTest`.
Output: `dist\FilterInspection_Setup_v<VERSION>.exe` (~1.7 GB).

A successful run prints, in order: four `Successfully created`, the
`LicenseActivator.exe: ...accepted` line, `BasicDemo.exe: main window ... supervised
model loaded`, `Successful compile`, `BUILD COMPLETE`. Anything less is a failure.

## 4. When it fails

| Symptom | Cause / fix |
|---|---|
| `... is missing from build\` after Nuitka said success | Antivirus (Trend Micro) deleted it. Rebuild; if it recurs, user must get `build\` excluded by IT. |
| Nuitka prompts for a download and dies | `--assume-yes-for-downloads` removed from build.ps1 — put it back. |
| exe exits `-1073740791` (0xC0000409) | Built from Anaconda Python. Use `.venv-build`. |
| `No module named 'license_check'` from a tools\license exe | build.ps1 lost the `PYTHONPATH` + `--include-module=license_check` for that step. |
| smoke test: valid code rejected | Built exe and `generate_license.py` see different keys — `license_secret.py` not compiled in (check the `--include-module=license_secret` flags). |
| `No module named 'CameraParams_const'` (or another MvImport sibling) | Hikvision SDK uses bare-name sibling imports. BasicDemo step needs `PYTHONPATH=NIRcam-first\MvImport` + the `--include-module=` list. |
| Any `No module named X` where X is imported after a runtime `sys.path.insert/append` | Same class of bug: Nuitka can't follow runtime sys.path edits. Put the folder on `PYTHONPATH` for that Nuitka call and add `--include-module=X`. |
| MVS "driver not found" box though MVS is installed | `ctypes.WinDLL(name)` without `winmode=0` skips PATH on Python 3.8+. |
| Log stops right after the weights line, no error | Old build.ps1 used `Remove-Item -Recurse` (kills PS 5.1 on the torch tree) or `$ErrorActionPreference="Stop"` (native stderr aborts when redirected). Current script uses `rd /s /q` and `Continue` — keep it that way. |
| `Out-File : ... user-mapped section open` | Log file is open elsewhere. Log to `$env:TEMP`. |
| "supervised model did not load" | Weights not bundled at `BasicDemo.dist\weights\`, or `inspection.*` not compiled in (build must run from repo root — the script `Set-Location`s itself). |

Paths inside the app: in the onefile `nircam_launcher.exe`, `__file__` is a temp
extraction folder. Anything that must be found in the **install folder**
(`license.key`, `BasicDemo.exe`, `logs\`) must use `sys.argv[0]`, not `__file__`.

Diagnose root cause before editing build.ps1; don't paper over a failing smoke test.

## 5. After a successful build

Report the installer path and size, and remind the user to run the acceptance
checklist in PACKAGING.md section 7 on a clean machine before handing it to the
vendor — this procedure cannot verify GPU inference speed or real camera behaviour.
Do not commit `build\` or `dist\` (both are gitignored).
