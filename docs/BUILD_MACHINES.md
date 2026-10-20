# 打包機紀錄

記錄每一台實際跑過 `build.ps1` 的電腦：環境、和 `PACKAGING.md` 不一樣的地方、
打包紀錄。換一台電腦打包之前先看這份，照「在另一台電腦打包」一節做。

> 這份檔案會進 git，**不要**寫入 `license_secret.py` 的內容、授權碼或任何密碼。

---

## 1. CITD2026-FT（產線旁的 RTX 5070 Ti 機器）

### 1.1 硬體 / 系統

| 項目 | 內容 |
|---|---|
| 電腦名稱 | `CITD2026-FT` |
| 作業系統 | Windows 11 專業版 10.0.26200 |
| CPU | Intel Core i7-14700（20 核 / 28 執行緒） |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 5070 Ti，16 GB，compute capability 12.0（**sm_120**） |
| 顯示卡驅動 | 591.86（`nvidia-smi` 顯示 CUDA 13.1） |
| 相機 SDK | Hikvision MVS Runtime 4.0.0.5，`C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64`（已在系統 PATH） |

### 1.2 打包工具

| 項目 | 內容 | 跟 `PACKAGING.md` 的差異 |
|---|---|---|
| Python | python.org 官方 CPython **3.10.11**，`%LOCALAPPDATA%\Programs\Python\Python310` | 同 3.1 節 |
| 打包環境 | repo 根目錄 `.venv-build\` | 同 3.1 節 |
| C 編譯器 | **沒有 MSVC**，Nuitka 自動使用 MinGW gcc 15.2.0（在 `%LOCALAPPDATA%\Nuitka\Nuitka\Cache\downloads\gcc`） | 3.1 節寫要 MSVC；實測用 MinGW 也能打包 |
| Inno Setup | **Inno Setup 7**，`C:\Program Files\Inno Setup 7\ISCC.exe` | 文件寫 6；`build.ps1` 要加 `-IsccPath` |
| git | 2.55.0，**沒有設定 user.name / user.email**，也沒有存 GitHub 登入 | push 要在自己的終端機登入 |

`.venv-build` 套件版本（2026-10-20）：

```
torch==2.7.1+cu128
torchvision==0.22.1+cu128
numpy==1.26.4
opencv-python-headless==4.11.0.86
PyQt5==5.15.11
scipy==1.15.3
PyYAML==6.0.3
Nuitka==4.3.1
ordered-set==4.1.0
zstandard==0.25.0
```

### 1.3 這台電腦的注意事項

- **torch 一定要 cu128。** 舊版安裝檔裡的 torch 不含 sm_120，`torch_cuda.dll` 裡只有到
  sm_90，程式在 RTX 5070 Ti 上開得起來、相機也正常，但一張缺陷都偵測不到。
  打包前跑 `python selftest.py`，「CUDA kernels run on this GPU」必須是 PASS。
- **不能用 conda 環境打包。** 實測：用 Anaconda 的 Python 3.10 以 Nuitka 打包一支只有
  `import ctypes` 的小程式，執行時沒有任何輸出，直接以 exit code 3 結束。
  conda 環境（`nircam`、`filter-inspect`、`This`）只拿來開發和跑 `run_gui.bat`。
- **Inno Setup 7**，打包指令要帶 `-IsccPath`（見 1.5 節）。
- **路徑不能太長。** 在很深的資料夾裡 `git clone` 會出現 `'$GIT_DIR' too big`。
  repo 放在 `C:\Users\User\Desktop\filter-inspection`。
- **GitHub 登入**：repo 是私有的，`git fetch` / `git push` 要在自己的終端機執行，
  第一次會跳出瀏覽器登入。
- **remote**：`origin` = `listenmudasir/filter-inspection`；
  `fork` = `RobertLee0522/filter-inspection`（公開，打包相關的 commit 最早只推到這裡）。
- `license_secret.py`、`weights\supervised_global.pt` 都是手動複製進來的，不在 git 裡。
  `license_secret.py` 已確認跟這台電腦原本的 `license.key` 相符（同一把金鑰）。

### 1.4 打包紀錄

| 日期 | 版本 | 分支 | 結果 |
|---|---|---|---|
| 2026-10-20 | 1.1.2 | `release/cu128-license` | ✅ `BUILD COMPLETE` |

**v1.1.2（2026-10-20）**

- 內容：`feature/hb-decoding-and-logging`（打包流程、授權、v1.1.1 的功能）
  合併 `master` 的 PR #4（GPU 不能跑時退回 CPU 並警告、修正 launcher 日誌無限遞迴），
  torch 換成 2.7.1+cu128。
- 耗時（28 執行緒，ccache 是空的）：

  | 步驟 | 時間 |
  |---|---|
  | `nircam_launcher.exe` | 12:27 → 12:29（約 2 分） |
  | `BasicDemo.exe`（3025 個 C 檔，含 torch） | 12:29 → 12:57（約 28 分） |
  | `FingerprintTool.exe`、`LicenseActivator.exe` | 約 1 分 |
  | 授權 smoke test + 啟動測試 | 約 1 分 |
  | Inno Setup | 504 秒（約 8.5 分） |
  | **合計** | **約 41 分** |

- 產出：`dist\FilterInspection_Setup_v1.1.2.exe`，2,313.7 MB
- SHA-256：`A49971C5B029A2AD116107E9B276639634AED310AAAB7BBD8830C79EF07CD1DD`
- `build.ps1` 自動檢查都通過：授權（指紋、拒絕錯誤授權碼、接受正確授權碼）、
  `BasicDemo.exe` 主視窗出現、監督式模型載入。
- 打包前 `selftest.py`（conda `nircam` 環境，torch 2.7.1+cu128）全部 PASS，
  空白影像推論 307 ms。
- 安裝方式：先備份 `license.key` → 從「設定 → 應用程式」解除安裝舊版 1.1.1 →
  安裝新版，授權頁貼上 `license.key` 的內容。

---

## 2. 在另一台電腦打包（例如主要的 RTX 3080 開發機）

那台電腦本來就有打包環境，但如果 `.venv-build` 是照舊版 `PACKAGING.md` 建的，
裡面會是 **torch 2.5.1+cu121**，打包出來的程式在 RTX 50 系列上會偵測不到缺陷。
先確認版本：

```bat
.venv-build\Scripts\python.exe -c "import torch; print(torch.__version__)"
```

印出的不是 `2.7.1+cu128` 的話，照下面步驟改：

**1. 取得這次的修改**（在 repo 資料夾）

```bat
git fetch origin
git checkout release/cu128-license
git pull
```

那台電腦原本的 `packaging/nuitka-license` / `feature/hb-decoding-and-logging`
**沒有** master 的 PR #4，請改用 `release/cu128-license`。

**2. 把 `.venv-build` 的 torch 換成 cu128**

```bat
.venv-build\Scripts\python.exe -m pip uninstall -y torch torchvision
.venv-build\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.venv-build\Scripts\python.exe -m pip install -r requirements.txt
```

確認有 sm_120（最後一個應該是 `sm_120`，`True` 代表那台的顯示卡能用）：

```bat
.venv-build\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.get_arch_list(), torch.cuda.is_available())"
```

cu128 版同時支援 RTX 30 / 40 / 50 系列，所以在 RTX 3080 上打包出來的安裝檔也能裝在 RTX 5070 Ti 上。

**3. 確認手動複製的檔案都在**

- `NIRcam-first\license_secret.py`（要跟發授權碼用的是同一把金鑰）
- `weights\supervised_global.pt`（274 MB，`build.ps1` 小於 100 MB 會中止）

**4. 打包**

```bat
powershell -ExecutionPolicy Bypass -File build.ps1
```

那台裝的若是 Inno Setup 7，加上 `-IsccPath "C:\Program Files\Inno Setup 7\ISCC.exe"`。

**5. 打包完把那台電腦也記到這份文件**：照第 1 節的格式新增一節，打包紀錄寫在該節的表格裡。
