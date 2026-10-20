; Inno Setup script for FilterInspection.
; VERSION is passed in from build.ps1 via /DAppVersion=x.y.z
; BUILD_DIR is passed in from build.ps1 via /DBuildDir=<path to nuitka output root>

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef BuildDir
  #define BuildDir "..\build"
#endif

[Setup]
AppName=FilterInspection
AppVersion={#AppVersion}
DefaultDirName={autopf64}\FilterInspection
DefaultGroupName=FilterInspection
; torch/CUDA are 64-bit only -- refuse to install on a 32-bit OS and never
; fall back to Program Files (x86). x64compatible (not the deprecated x64
; identifier) is Inno Setup 6.3+'s recommended spelling for "needs a 64-bit
; x64 Windows"; it doesn't change what's actually required here -- the app
; still needs a real NVIDIA GPU, which an ARM64 machine running x64 under
; emulation wouldn't have anyway.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=FilterInspection_Setup_v{#AppVersion}
Compression=lzma2
SolidCompression=yes
DisableProgramGroupPage=yes
SetupIconFile=..\NIRcam-first\assets\nircam_inspection.ico

[Files]
; Splash launcher (onefile exe).
Source: "{#BuildDir}\nircam_launcher.exe"; DestDir: "{app}"; Flags: ignoreversion
; Main GUI, standalone Nuitka output folder -- everything inside it,
; including BasicDemo.exe, DLLs, and the bundled model weights.
Source: "{#BuildDir}\BasicDemo.dist\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Human-facing tool for RE-activation after install (e.g. hardware changed
; and license.key needs regenerating without rerunning the whole installer).
; The normal first-time activation happens inside this wizard instead --
; see [Code] below -- so this is a fallback, not the main path.
Source: "{#BuildDir}\FingerprintTool.exe"; DestDir: "{app}\Tools"; Flags: ignoreversion
; Installer-only activation helper. dontcopy: it is NOT installed to {app};
; the [Code] section below pulls it into {tmp} on demand via
; ExtractTemporaryFile and calls it from the wizard pages, before {app}'s
; files even exist on disk.
Source: "{#BuildDir}\LicenseActivator.exe"; DestDir: "{tmp}"; Flags: dontcopy

[Dirs]
; The launcher and BasicDemo.exe append to {app}\logs on every start, but
; ordinary users only get read access under Program Files -- a normal
; (non-admin) double-click of the shortcut would die before the splash shows.
; Grant write on logs\ only, so operators still can't replace the exes.
Name: "{app}\logs"; Permissions: users-modify

[Icons]
Name: "{group}\NIRcam Inspection"; Filename: "{app}\nircam_launcher.exe"
Name: "{autodesktop}\NIRcam Inspection"; Filename: "{app}\nircam_launcher.exe"

; No [Run] "launch now" checkbox: Inno starts a postinstall program as the
; original non-admin user through its spawn server, and on a vendor machine
; that handoff failed with "Internal error: CallSpawnServer: Unexpected
; response: $0". Operators start the app from the desktop shortcut instead.

[Messages]
FinishedLabel=安裝完成。%n%n若這台機器尚未安裝相機廠商的 Hikvision MVS 驅動程式，請先安裝 MVS Runtime 後再啟動本程式。

[Code]
var
  LicensePage: TInputQueryWizardPage;
  FingerprintValue: String;
  LicenseCodeValue: String;
  ActivatorExePath: String;

(* Runs LicenseActivator.exe (extracting it to the temp folder on first use)
   with the given command-line args, and returns the contents of the out
   file it was told to write its result to. Returns '' if the tool could
   not be run at all (missing/blocked), which callers treat as a hard
   failure. *)
function RunActivator(const Args: String; const OutFile: String): String;
var
  ResultCode: Integer;
  Output: AnsiString;
begin
  Result := '';
  if ActivatorExePath = '' then
  begin
    ExtractTemporaryFile('LicenseActivator.exe');
    ActivatorExePath := ExpandConstant('{tmp}\LicenseActivator.exe');
  end;

  if not Exec(ActivatorExePath, Args, '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    exit;

  (* license_installer_helper.py deliberately writes ASCII-only status codes
     here (OK:<hex>, FAIL:NO_FINGERPRINT, FAIL:MISMATCH), never free-text
     Chinese -- LoadStringFromFile reads into an AnsiString, which would
     mangle anything outside the system codepage. The Chinese messages the
     user actually sees live below in NextButtonClick, keyed off these
     codes. *)
  if LoadStringFromFile(OutFile, Output) then
    Result := Trim(String(Output));
end;

(* Silent installs (/SILENT, /VERYSILENT) take the code from
   /LICENSECODE=<code> and verify it before anything is copied. Without this,
   the license page's NextButtonClick rejects the empty field forever and
   the silent setup hangs instead of exiting. A bad or missing code exits
   with a non-zero code; the license stays machine-bound either way. *)
function InitializeSetup(): Boolean;
var
  Code, OutFile, Status: String;
begin
  Result := True;
  if not WizardSilent then
    exit;

  Code := Trim(ExpandConstant('{param:LICENSECODE|}'));
  if Code = '' then
  begin
    Log('Silent install requires /LICENSECODE=<code>; aborting.');
    Result := False;
    exit;
  end;

  OutFile := ExpandConstant('{tmp}\license_check.txt');
  Status := RunActivator('check "' + Code + '" "' + OutFile + '"', OutFile);
  if Copy(Status, 1, 3) = 'OK:' then
    LicenseCodeValue := Copy(Status, 4, MaxInt)
  else
  begin
    Log('License check failed (' + Status + '); aborting silent install.');
    Result := False;
  end;
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := WizardSilent and (PageID = LicensePage.ID);
end;

procedure InitializeWizard;
begin
  LicensePage := CreateInputQueryPage(wpSelectDir,
    '軟體授權', '本程式僅限授權機器使用',
    '下方「機器識別碼」是這台電腦專屬的一串代碼，請完整複製後透過 LINE/Email 傳給軟體提供者。'
    + #13#10
    + '收到提供者回傳的「授權碼」後，貼到下方欄位才能繼續安裝——沒有正確的授權碼，本程式無法在這台機器上執行。');
  LicensePage.Add('機器識別碼（複製給提供者，不要自行修改）：', False);
  LicensePage.Add('授權碼（貼上提供者回傳的內容）：', False);
end;

procedure CurPageChanged(CurPageID: Integer);
var
  OutFile: String;
begin
  if CurPageID = LicensePage.ID then
  begin
    if FingerprintValue = '' then
    begin
      OutFile := ExpandConstant('{tmp}\fingerprint.txt');
      FingerprintValue := RunActivator('fingerprint "' + OutFile + '"', OutFile);
      if FingerprintValue = '' then
        FingerprintValue := '(讀取失敗，請聯絡軟體提供者協助排除，暫時可略過此欄位)';
    end;
    LicensePage.Values[0] := FingerprintValue;
    (* Read-only: this value must reach the vendor unedited, or the license
       code they generate from it will never match. *)
    LicensePage.Edits[0].ReadOnly := True;
    LicensePage.Edits[0].TabStop := False;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Code, OutFile, Status: String;
begin
  Result := True;
  if CurPageID <> LicensePage.ID then
    exit;

  Code := Trim(LicensePage.Values[1]);
  if Code = '' then
  begin
    MsgBox('請輸入授權碼後再繼續。', mbError, MB_OK);
    Result := False;
    exit;
  end;

  OutFile := ExpandConstant('{tmp}\license_check.txt');
  Status := RunActivator('check "' + Code + '" "' + OutFile + '"', OutFile);
  if Status = '' then
  begin
    MsgBox('無法執行授權檢查工具，請聯絡軟體提供者協助排除。', mbError, MB_OK);
    Result := False;
    exit;
  end;

  if Copy(Status, 1, 3) = 'OK:' then
  begin
    (* Store the canonical value the checker computed, NOT the user's raw
       keystrokes -- guarantees an exact match with what license_check.py
       verifies at runtime even if the user pasted extra whitespace or a
       different letter case. *)
    LicenseCodeValue := Copy(Status, 4, MaxInt);
  end
  else if Status = 'FAIL:NO_FINGERPRINT' then
  begin
    MsgBox('無法讀取這台機器的識別資訊，請聯絡軟體提供者協助排除。', mbError, MB_OK);
    Result := False;
  end
  else if Status = 'FAIL:MISMATCH' then
  begin
    MsgBox('授權碼與這台機器不符，請確認輸入正確，或聯絡軟體提供者重新核發。', mbError, MB_OK);
    Result := False;
  end
  else
  begin
    MsgBox('授權驗證失敗，請聯絡軟體提供者協助排除。', mbError, MB_OK);
    Result := False;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and (LicenseCodeValue <> '') then
    SaveStringToFile(ExpandConstant('{app}\license.key'), LicenseCodeValue, False);
end;
