# Image Shape Override + Remove Shared-Memory Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the `Image conversion error: cannot reshape array of size N into shape (H, W)` crash in the camera grab thread, let the operator manually override the capture width/height from the UI, and completely remove the shared-memory image-sharing feature (UI, logic, and standalone sender/receiver scripts).

**Architecture:** `CamOperation_class.py`'s `work_thread` currently reshapes the *entire* grab buffer (sized to the camera's `PayloadSize`, which can be larger than one frame) instead of just the frame's own bytes — that mismatch is the root cause of the crash. The fix extracts two pure, hardware-independent helpers into a new `image_shape.py` module (testable with plain pytest, since `CamOperation_class.py` itself can only be imported on a machine with the camera SDK DLL installed — its `MvImport` package calls `ctypes.cdll.LoadLibrary` at import time). `CamOperation_class.py` and `BasicDemo.py` (the PyQt5 GUI) are hardware/GUI-coupled with no existing test infrastructure in this repo, so those two tasks are verified by `py_compile` + `grep` sweeps for dangling references, not automated tests — call this out explicitly rather than faking test coverage. Shared-memory removal is a pure deletion across the same two files plus three standalone scripts.

**Tech Stack:** Python, PyQt5, numpy, opencv-python, pytest.

## Global Constraints

- Manual width/height override takes priority over the camera-reported frame size only when **both** values are set and positive; otherwise fall back to the camera's `st_frame_info.nHeight`/`nWidth`.
- The override is **not persisted** — it resets to "no override" every time the app restarts (in-memory only).
- Delete the shared-memory feature entirely: `NIRcam-first/shared_memory_sender.py`, `NIRcam-first/recive.py`, `NIRcam-first/image_receiver.py`, plus every reference to them in `CamOperation_class.py` and `BasicDemo.py`.
- Do not touch `tcp_server.py` or its UI — that's the unrelated LabVIEW trigger channel.

---

## Task 1: `image_shape.py` — pure shape-resolution and decode helpers

**Files:**
- Create: `NIRcam-first/image_shape.py`
- Test: `NIRcam-first/test_image_shape.py`

**Interfaces:**
- Produces: `resolve_capture_shape(frame_height: int, frame_width: int, manual_width: int | None, manual_height: int | None) -> tuple[int, int]` — returns `(height, width)` to reshape to.
- Produces: `decode_raw_frame(buffer, height: int, width: int) -> numpy.ndarray` — single-channel `uint8` array of shape `(height, width)`, reading only `height*width` bytes out of `buffer` regardless of the buffer's actual (possibly larger) size.

- [ ] **Step 1: Write the failing tests**

Create `NIRcam-first/test_image_shape.py`:

```python
import numpy as np
import pytest

from image_shape import resolve_capture_shape, decode_raw_frame


def test_resolve_capture_shape_uses_camera_size_when_no_override():
    assert resolve_capture_shape(2048, 2448, None, None) == (2048, 2448)


def test_resolve_capture_shape_uses_camera_size_when_override_partial():
    # Only one of width/height set -> override must not apply.
    assert resolve_capture_shape(2048, 2448, 2448, None) == (2048, 2448)
    assert resolve_capture_shape(2048, 2448, None, 2048) == (2048, 2448)


def test_resolve_capture_shape_uses_camera_size_when_override_zero():
    assert resolve_capture_shape(2048, 2448, 0, 0) == (2048, 2448)


def test_resolve_capture_shape_uses_manual_override_when_both_set():
    assert resolve_capture_shape(2048, 2448, 2200, 1944) == (1944, 2200)


def test_decode_raw_frame_reads_only_needed_bytes_from_oversized_buffer():
    height, width = 4, 3
    frame_bytes = bytes(range(height * width))
    # Buffer padded larger than one frame, like a PayloadSize-sized grab buffer.
    oversized_buffer = frame_bytes + bytes(50)

    result = decode_raw_frame(oversized_buffer, height, width)

    assert result.shape == (height, width)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.arange(12, dtype=np.uint8).reshape(4, 3))


def test_decode_raw_frame_raises_when_buffer_too_small():
    with pytest.raises(ValueError):
        decode_raw_frame(bytes(5), 4, 3)  # needs 12 bytes, only has 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd NIRcam-first && python -m pytest test_image_shape.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'image_shape'`

- [ ] **Step 3: Write the implementation**

Create `NIRcam-first/image_shape.py`:

```python
"""Pure helpers for resolving and decoding the raw camera frame buffer shape.

Kept dependency-free (numpy only) so these can be unit tested without the
camera SDK: CamOperation_class.py's own imports require the vendor DLL to be
loadable (MvImport/MvCameraControl_class.py calls ctypes.cdll.LoadLibrary at
module import time), so it can't be imported in a plain test environment.
"""
import numpy as np


def resolve_capture_shape(frame_height, frame_width, manual_width, manual_height):
    """Return the (height, width) to reshape the raw frame buffer to.

    The manual override wins only when both dimensions are set and positive;
    otherwise the camera-reported frame size is used.
    """
    if manual_width and manual_height and manual_width > 0 and manual_height > 0:
        return manual_height, manual_width
    return frame_height, frame_width


def decode_raw_frame(buffer, height, width):
    """Reshape a single-channel raw frame out of a possibly oversized buffer.

    Uses an explicit count=height*width so only the frame's own bytes are
    read, even when `buffer` is sized to the camera's PayloadSize rather than
    the actual frame length (the mismatch that causes
    "cannot reshape array of size N into shape (H, W)").
    """
    return np.frombuffer(buffer, dtype=np.uint8, count=height * width).reshape(height, width)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd NIRcam-first && python -m pytest test_image_shape.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add NIRcam-first/image_shape.py NIRcam-first/test_image_shape.py
git commit -m "$(cat <<'EOF'
Add pure helpers for camera frame shape override and safe decode

resolve_capture_shape lets the UI override the reshape target size;
decode_raw_frame reads only height*width bytes instead of reshaping
the whole (PayloadSize-sized) grab buffer, which is the root cause of
"cannot reshape array of size N into shape (H, W)".

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `CamOperation_class.py` — wire in the override, fix the reshape bug, remove shared memory

**Files:**
- Modify: `NIRcam-first/CamOperation_class.py`

**Interfaces:**
- Consumes: `resolve_capture_shape`, `decode_raw_frame` from `image_shape.py` (Task 1).
- Produces: `set_manual_image_shape(width: int | None, height: int | None) -> None` module-level function — `BasicDemo.py` (Task 3) calls this from the new UI fields.

- [ ] **Step 1: Remove the shared-memory import**

In `NIRcam-first/CamOperation_class.py`, remove line 24:

```python
from shared_memory_sender import SharedMemorySender
```

- [ ] **Step 2: Replace the shared-memory globals with the manual-shape-override globals**

Find (around line 57-59):

```python
# 在類的開頭添加全局變量引用
shared_memory_sender = None
auto_share_enabled = False
```

Replace with:

```python
# 手動覆寫取像時使用的影像寬高。兩者皆為正數時才生效，否則使用相機回報的
# st_frame_info.nWidth/nHeight（見 set_manual_image_shape）。
manual_image_width = None
manual_image_height = None
```

- [ ] **Step 3: Add the `image_shape` import**

Near the top imports (after the `cv2` import, where the now-removed `shared_memory_sender` import used to be), add:

```python
from image_shape import resolve_capture_shape, decode_raw_frame
```

- [ ] **Step 4: Replace `set_shared_memory_sender`/`set_auto_share` with `set_manual_image_shape`**

Find (around line 260-268):

```python
def set_shared_memory_sender(sender):
    """設置共享記憶體發送器"""
    global shared_memory_sender
    shared_memory_sender = sender

def set_auto_share(enabled):
    """設置是否自動分享圖像"""
    global auto_share_enabled
    auto_share_enabled = enabled
```

Replace with:

```python
def set_manual_image_shape(width, height):
    """設定手動覆寫的取像寬高。傳入 None 或 0 代表清除覆寫（改用相機回報值）。"""
    global manual_image_width, manual_image_height
    manual_image_width = width if width else None
    manual_image_height = height if height else None
```

- [ ] **Step 5: Apply the override and fix the raw-frame reshape in `work_thread`**

Find (around line 744-799):

```python
                    try:
                        nH = self.st_frame_info.nHeight
                        nW = self.st_frame_info.nWidth
                        enPT = self.st_frame_info.enPixelType

                        # RGB8/BGR8 Packed：相機端已經做完 debayer，每像素 3 bytes。
                        # 這種格式不能走下面的單通道 reshape（會直接丟
                        # "cannot reshape array of size N into shape (H, W)"），
                        # 也不能再 demosaic 一次。必須先擋在前面。
                        if enPT in (PixelType_Gvsp_RGB8_Packed,
                                    PixelType_Gvsp_BGR8_Packed):
                            packed = np.frombuffer(
                                self.buf_grab_image, dtype=np.uint8,
                                count=nH * nW * 3).reshape(nH, nW, 3)
                            if enPT == PixelType_Gvsp_BGR8_Packed:
                                image_rgb = cv2.cvtColor(packed, cv2.COLOR_BGR2RGB)
                            else:
                                image_rgb = packed.copy()

                        else:
                            raw_image = np.asarray(self.buf_grab_image).reshape(
                                (nH, nW)
                            )

                            # 根據像素格式進行轉換
                            if Is_color_data(self.st_frame_info.enPixelType):
                                # 彩色圖像 - 從 Bayer 格式直接轉換為 RGB
                                # 注意：嘗試使用 BG 格式來修正紅藍通道互換問題
                                if self.st_frame_info.enPixelType == PixelType_Gvsp_BayerRG8:
                                    # RG8 使用 BG2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_BG2RGB)
                                elif self.st_frame_info.enPixelType == PixelType_Gvsp_BayerGR8:
                                    # GR8 使用 GB2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GB2RGB)
                                elif self.st_frame_info.enPixelType == PixelType_Gvsp_BayerGB8:
                                    # GB8 使用 GR2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GR2RGB)
                                elif self.st_frame_info.enPixelType == PixelType_Gvsp_BayerBG8:
                                    # BG8 使用 RG2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_RG2RGB)
                                else:
                                    # 默認使用 BG8（而不是 RG8）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_BG2RGB)
                            elif Is_mono_data(self.st_frame_info.enPixelType):
                                # 單色影像轉換為 3 通道供後續處理
                                mono_array = Mono_numpy(
                                    self.buf_save_image, 
                                    self.st_frame_info.nWidth, 
                                    self.st_frame_info.nHeight
                                )
                                # 單色轉 RGB（三個通道相同）
                                image_rgb = cv2.cvtColor(mono_array.squeeze(), cv2.COLOR_GRAY2RGB)
                            else:
                                # 未知格式，跳過此幀
                                print(f"Unsupported pixel format: {self.st_frame_info.enPixelType}")
                                continue
                        
                    except Exception as e:
                        print(f"Image conversion error: {e}")
                        continue
```

Replace with:

```python
                    try:
                        nH, nW = resolve_capture_shape(
                            self.st_frame_info.nHeight,
                            self.st_frame_info.nWidth,
                            manual_image_width,
                            manual_image_height,
                        )
                        enPT = self.st_frame_info.enPixelType

                        # RGB8/BGR8 Packed：相機端已經做完 debayer，每像素 3 bytes。
                        # 這種格式不能走下面的單通道 reshape（會直接丟
                        # "cannot reshape array of size N into shape (H, W)"），
                        # 也不能再 demosaic 一次。必須先擋在前面。
                        if enPT in (PixelType_Gvsp_RGB8_Packed,
                                    PixelType_Gvsp_BGR8_Packed):
                            packed = np.frombuffer(
                                self.buf_grab_image, dtype=np.uint8,
                                count=nH * nW * 3).reshape(nH, nW, 3)
                            if enPT == PixelType_Gvsp_BGR8_Packed:
                                image_rgb = cv2.cvtColor(packed, cv2.COLOR_BGR2RGB)
                            else:
                                image_rgb = packed.copy()

                        else:
                            # decode_raw_frame 只讀取 nH*nW 個 bytes，不會像
                            # np.asarray(self.buf_grab_image) 那樣把整個依
                            # PayloadSize 配置、可能大於實際幀長度的緩衝區拿去
                            # reshape（那正是原本 reshape 崩潰的原因）。
                            raw_image = decode_raw_frame(self.buf_grab_image, nH, nW)

                            # 根據像素格式進行轉換
                            if Is_color_data(self.st_frame_info.enPixelType):
                                # 彩色圖像 - 從 Bayer 格式直接轉換為 RGB
                                # 注意：嘗試使用 BG 格式來修正紅藍通道互換問題
                                if self.st_frame_info.enPixelType == PixelType_Gvsp_BayerRG8:
                                    # RG8 使用 BG2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_BG2RGB)
                                elif self.st_frame_info.enPixelType == PixelType_Gvsp_BayerGR8:
                                    # GR8 使用 GB2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GB2RGB)
                                elif self.st_frame_info.enPixelType == PixelType_Gvsp_BayerGB8:
                                    # GB8 使用 GR2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_GR2RGB)
                                elif self.st_frame_info.enPixelType == PixelType_Gvsp_BayerBG8:
                                    # BG8 使用 RG2RGB 轉換（紅藍互換）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_RG2RGB)
                                else:
                                    # 默認使用 BG8（而不是 RG8）
                                    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BAYER_BG2RGB)
                            elif Is_mono_data(self.st_frame_info.enPixelType):
                                # 單色影像轉換為 3 通道供後續處理
                                mono_array = Mono_numpy(
                                    self.buf_save_image,
                                    nW,
                                    nH
                                )
                                # 單色轉 RGB（三個通道相同）
                                image_rgb = cv2.cvtColor(mono_array.squeeze(), cv2.COLOR_GRAY2RGB)
                            else:
                                # 未知格式，跳過此幀
                                print(f"Unsupported pixel format: {self.st_frame_info.enPixelType}")
                                continue
                        
                    except Exception as e:
                        print(f"Image conversion error: {e}")
                        continue
```

- [ ] **Step 6: Remove the shared-memory auto-send block from `work_thread`**

Find (immediately after the block replaced in Step 5, around line 805-830):

```python
                    # ========================================
                    # 第二步：共享記憶體自動發送（如果啟用）
                    # ========================================
                    if auto_share_enabled and shared_memory_sender is not None:
                        try:
                            # 複製圖像並轉換為 BGR 格式（共享記憶體可能需要 BGR）
                            image_for_sharing = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                            
                            # 執行您需要的預處理
                            # 目前不執行鏡像翻轉
                            image_for_sharing = image_for_sharing
                            
                            # 發送到共享記憶體
                            if hasattr(shared_memory_sender, 'trigger_count'):
                                shared_memory_sender.trigger_count += 1
                                trigger_count = shared_memory_sender.trigger_count
                            else:
                                shared_memory_sender.trigger_count = 1
                                trigger_count = 1
                            
                            shared_memory_sender.send_image(image_for_sharing, trigger_count)
                            
                            print(f"[共享記憶體] 已自動發送第 {trigger_count} 幀")
                            
                        except Exception as e:
                            print(f"[共享記憶體] 發送失敗: {e}")
    
                    # ========================================
                    # 第三步：AI 辨識處理（如果啟用）
                    # ========================================
```

Replace with just:

```python
                    # ========================================
                    # 第二步：AI 辨識處理（如果啟用）
                    # ========================================
```

- [ ] **Step 7: Verify the file compiles and no shared-memory references remain**

Run: `cd NIRcam-first && python -m py_compile CamOperation_class.py`
Expected: no output (success)

Run: `cd NIRcam-first && grep -n "shared_memory\|SharedMemorySender\|auto_share_enabled" CamOperation_class.py`
Expected: no matches

- [ ] **Step 8: Run the Task 1 tests again to confirm nothing broke**

Run: `cd NIRcam-first && python -m pytest test_image_shape.py -v`
Expected: PASS (6 passed)

- [ ] **Step 9: Commit**

```bash
git add NIRcam-first/CamOperation_class.py
git commit -m "$(cat <<'EOF'
Fix frame reshape crash, add manual shape override, drop shared memory

work_thread was reshaping the whole PayloadSize-sized grab buffer
instead of just the current frame's bytes, causing "cannot reshape
array of size N into shape (H, W)". It now goes through
resolve_capture_shape/decode_raw_frame from image_shape.py, which
also lets a UI-supplied width/height override the camera-reported
frame size. The shared-memory auto-send path and its globals are
removed entirely.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `BasicDemo.py` — replace the shared-memory panel with the shape-override panel

**Files:**
- Modify: `NIRcam-first/BasicDemo.py`

**Interfaces:**
- Consumes: `set_manual_image_shape(width, height)` from `CamOperation_class.py` (Task 2).

- [ ] **Step 1: Swap the shared-memory imports for the shape-override import**

Find (lines 17-19):

```python
from CamOperation_class import set_ai_model, set_ai_parameters_func
from shared_memory_sender import SharedMemorySender # 引入共享內存發送器
from CamOperation_class import set_shared_memory_sender, set_auto_share
```

Replace with:

```python
from CamOperation_class import set_ai_model, set_ai_parameters_func
from CamOperation_class import set_manual_image_shape
```

- [ ] **Step 2: Remove the shared-memory globals**

Find (lines 86-96):

```python
    signals = SignalEmitter() # 實例化訊號發射器
    # 新增全局變量用於共享內存發送器
    global global_sender
    global_sender = None
    global shared_trigger_count
    shared_trigger_count = 0
    # 配置共享內存的目標 (請根據您的接收端程式調整)
    global RECEIVER_HOST
    global RECEIVER_PORT
    RECEIVER_HOST = '127.0.0.1' 
    RECEIVER_PORT = 9999
```

Replace with:

```python
    signals = SignalEmitter() # 實例化訊號發射器
```

- [ ] **Step 3: Delete the shared-memory handler functions**

Find this entire contiguous block (`update_shared_memory_ui`, `start_shared_memory`, `stop_shared_memory`, `toggle_auto_share`, `manual_share_current_frame`, and `transfer_image_and_flip` — six functions):

```python
    def update_shared_memory_ui():
        """更新共享記憶體連接狀態"""
        global global_sender, shared_trigger_count
        
        if global_sender is not None:
            # 檢查 trigger_count 屬性
            try:
                count = getattr(global_sender, 'trigger_count', 0)
                ui.lblSharedMemStatus.setText(i18n.tr_fmt("共享記憶體: 已啟動 (已傳送 {count} 幀)", count=count))
                ui.lblSharedMemStatus.setStyleSheet("color: green; font-weight: bold;")
            except Exception as e:
                ui.lblSharedMemStatus.setText(i18n.tr("共享記憶體: 已啟動 (狀態未知)"))
                ui.lblSharedMemStatus.setStyleSheet("color: orange; font-weight: bold;")
        else:
            ui.lblSharedMemStatus.setText(i18n.tr("共享記憶體: 未啟動"))
            ui.lblSharedMemStatus.setStyleSheet("color: red; font-weight: bold;")

    def start_shared_memory():
       """啟動共享記憶體發送器"""
       global global_sender, shared_trigger_count

       try:
           host = ui.edtSharedMemHost.text() or '127.0.0.1'
           port = int(ui.edtSharedMemPort.text() or '9999')
           print(f"嘗試連接到 {host}:{port}") 
           if global_sender is None:
               # 創建發送器實例
               global_sender = SharedMemorySender(host, port)
               if not global_sender.is_connected():
                   QMessageBox.warning(mainWindow, "連接失敗", 
                        "無法連接到接收端！\n請確認：\n"
                        "1. 接收端程式 (receiver.py) 已啟動\n"
                        "2. IP 和端口設置正確")
                   global_sender = None
                   return
 
               # 初始化計數器
               if not hasattr(global_sender, 'trigger_count'):
                   global_sender.trigger_count = 0
               shared_trigger_count = 0

               # 將發送器設置到 CamOperation_class
               set_shared_memory_sender(global_sender)

               QMessageBox.information(mainWindow, "共享記憶體", 
                   f"共享記憶體已啟動！\n\n"
                   f"目標位址: {host}:{port}\n"
                   f"請確保接收端程式 (receiver.py) 已在運行\n\n"
                   f"提示: 可在「共享記憶體控制」頁籤中\n"
                   f"啟用「自動分享」或使用「手動分享」")

               ui.bnStartSharedMem.setEnabled(False)
               ui.bnStopSharedMem.setEnabled(True)
               ui.chkAutoShare.setEnabled(True)
               ui.bnManualShare.setEnabled(True)

               # 更新UI狀態
               update_shared_memory_ui()
           else:
               QMessageBox.warning(mainWindow, "共享記憶體", "發送器已在運行中！")

       except Exception as e:
           QMessageBox.critical(mainWindow, "共享記憶體", f"啟動失敗:\n{str(e)}")
           import traceback
           traceback.print_exc()

    def stop_shared_memory():
        """停止共享記憶體發送器"""
        global global_sender

        try:
            if global_sender is not None:
                # 停用自動分享
                set_auto_share(False)
                ui.chkAutoShare.setChecked(False)

                # 清除發送器引用
                set_shared_memory_sender(None)

                # 關閉連接
                try:
                    global_sender.close()
                except Exception as e:
                    print(f"關閉發送器時出錯: {e}")

                global_sender = None

                QMessageBox.information(mainWindow, "共享記憶體", "共享記憶體已停止")
                ui.bnStartSharedMem.setEnabled(True)
                ui.bnStopSharedMem.setEnabled(False)
                ui.chkAutoShare.setEnabled(False)
                ui.bnManualShare.setEnabled(False)

                # 更新UI狀態
                update_shared_memory_ui()
            else:
                QMessageBox.information(mainWindow, "共享記憶體", "共享記憶體未在運行")

        except Exception as e:
            QMessageBox.critical(mainWindow, "共享記憶體", f"停止失敗:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def toggle_auto_share():
        """切換自動分享模式"""
        global global_sender

        enabled = ui.chkAutoShare.isChecked()

        if enabled and global_sender is None:
            QMessageBox.warning(mainWindow, "自動分享", "請先啟動共享記憶體！")
            ui.chkAutoShare.setChecked(False)
            return

        if not isGrabbing:
            QMessageBox.warning(mainWindow, "自動分享", "請先開始取像！")
            ui.chkAutoShare.setChecked(False)
            return

        set_auto_share(enabled)

        if enabled:
            QMessageBox.information(mainWindow, "自動分享", 
                "✅ 已啟用自動分享\n\n"
                "每一幀圖像都會自動發送到共享記憶體\n"
                "接收端可實時接收圖像數據")
        else:
            QMessageBox.information(mainWindow, "自動分享", 
                "⏸ 已停用自動分享\n\n"
                "可使用「手動分享」按鈕手動發送圖像")

    def manual_share_current_frame():
        """手動分享當前幀"""
        global global_sender, shared_trigger_count

        if not isGrabbing:
            QMessageBox.warning(mainWindow, "錯誤", "請先開始取像！")
            return

        if global_sender is None:
            QMessageBox.warning(mainWindow, "錯誤", "請先啟動共享記憶體！")
            return

        try:
            # 調用原有的手動分享函數
            transfer_image_and_flip()

            # 獲取當前計數
            count = getattr(global_sender, 'trigger_count', shared_trigger_count)

            QMessageBox.information(mainWindow, "手動分享", 
                f"✅ 已發送圖像\n\n"
                f"觸發次數: {count}\n"
                f"接收端應已收到數據")

        except Exception as e:
            QMessageBox.critical(mainWindow, "手動分享", f"發送失敗:\n{str(e)}")
            import traceback
            traceback.print_exc()
    def transfer_image_and_flip():
        """
        手動觸發：獲取當前幀，處理並共享
        這個函數用於手動分享按鈕
        """
        global obj_cam_operation, global_sender, shared_trigger_count

        if not isGrabbing or obj_cam_operation.buf_save_image is None:
            raise Exception("相機未在取像或緩衝區為空")

        try:
            # 獲取緩衝區鎖，防止取圖線程同時寫入
            obj_cam_operation.buf_lock.acquire() 

            # 1. 從 C 緩衝區創建 NumPy 數組
            st_info = obj_cam_operation.st_frame_info

            if st_info is None:
                raise Exception("幀信息為空")

            # 創建原始圖像數組
            raw_data = np.ctypeslib.as_array(
                obj_cam_operation.buf_save_image, 
                shape=(st_info.nHeight, st_info.nWidth)
            )

            # 複製數據
            image_bayer = raw_data.copy()
            obj_cam_operation.buf_lock.release()

            # 2. 轉換圖像格式
            if st_info.enPixelType == PixelType_Gvsp_BayerRG8:
                image_bgr = cv2.cvtColor(image_bayer, cv2.COLOR_BAYER_RG2BGR)
            elif st_info.enPixelType == PixelType_Gvsp_BayerGR8:
                image_bgr = cv2.cvtColor(image_bayer, cv2.COLOR_BAYER_GR2BGR)
            elif st_info.enPixelType == PixelType_Gvsp_BayerGB8:
                image_bgr = cv2.cvtColor(image_bayer, cv2.COLOR_BAYER_GB2BGR)
            elif st_info.enPixelType == PixelType_Gvsp_BayerBG8:
                image_bgr = cv2.cvtColor(image_bayer, cv2.COLOR_BAYER_BG2BGR)
            elif st_info.enPixelType == PixelType_Gvsp_Mono8:
                image_bgr = cv2.cvtColor(image_bayer, cv2.COLOR_GRAY2BGR)
            else:
                raise Exception(f"不支援的像素格式: {st_info.enPixelType}")

        except Exception as e:
            # 確保在異常情況下釋放鎖
            if obj_cam_operation.buf_lock.locked():
                obj_cam_operation.buf_lock.release()
            raise e

        # 3. 圖像處理：
        # 不執行翻轉操作
        image_bgr = image_bgr

        # 4. 發送到共享記憶體
        shared_trigger_count += 1
        global_sender.send_image(image_bgr, shared_trigger_count)

        # 同步計數器
        if hasattr(global_sender, 'trigger_count'):
            global_sender.trigger_count = shared_trigger_count

        print(f"[手動分享] 已發送圖像，觸發次數: {shared_trigger_count}")
```

Replace with nothing (delete the whole block). Leave a single blank line before the `def open_device():` that follows it.

- [ ] **Step 4: Remove the leftover shared-memory globals inside `open_device()`**

Find:

```python
    def open_device():
        global deviceList, nSelCamIndex, obj_cam_operation, isOpen
        #共享記憶體全域變數
        global global_sender
        global shared_trigger_count # 確保可以訪問
        
        if isOpen:
```

Replace with:

```python
    def open_device():
        global deviceList, nSelCamIndex, obj_cam_operation, isOpen
        
        if isOpen:
```

- [ ] **Step 5: Remove the shared-memory cleanup call in `close_device`**

Find:

```python
    def close_device():
        global isOpen, isGrabbing, obj_cam_operation
        global obj_cam_operation
        global global_sender # 確保可以訪問
        if isGrabbing:
            stop_grabbing()
        if isOpen:
            obj_cam_operation.Close_device()
            isOpen = False
        # **【新增邏輯】** 清理共享內存資源
        if global_sender is not None:
            global_sender.close()
        isGrabbing = False
        enable_controls()
```

Replace with:

```python
    def close_device():
        global isOpen, isGrabbing, obj_cam_operation
        global obj_cam_operation
        if isGrabbing:
            stop_grabbing()
        if isOpen:
            obj_cam_operation.Close_device()
            isOpen = False
        isGrabbing = False
        enable_controls()
```

- [ ] **Step 6: Remove the shared-memory lines from `enable_controls`**

Find:

```python
        ui.bnSaveImage.setEnabled(isOpen and isGrabbing)
        # 共享記憶體控制
        # 只有在取像且共享記憶體已啟動時才能手動分享
        ui.bnManualShare.setEnabled(isOpen and isGrabbing and global_sender is not None)
            # 自動分享只有在共享記憶體啟動時才能勾選
        if global_sender is None:
            ui.chkAutoShare.setEnabled(False)
            ui.chkAutoShare.setChecked(False)
        elif not isGrabbing:
            # 如果停止取像，自動取消自動分享
            if ui.chkAutoShare.isChecked():
                ui.chkAutoShare.setChecked(False)
                set_auto_share(False)
```

Replace with:

```python
        ui.bnSaveImage.setEnabled(isOpen and isGrabbing)
```

- [ ] **Step 7: Replace the shared-memory group box with the shape-override group box**

Find (the whole "共享記憶體控制區塊"):

```python
    # === 共享記憶體控制區塊 (簡化版) ===
    shared_mem_group = QGroupBox("共享記憶體")
    shared_mem_layout = QVBoxLayout()
    
    # 連接設定
    host_conn_layout = QHBoxLayout()
    host_conn_layout.addWidget(QLabel("IP:"))
    ui.edtSharedMemHost = QLineEdit("127.0.0.1")
    host_conn_layout.addWidget(ui.edtSharedMemHost)
    shared_mem_layout.addLayout(host_conn_layout)
    
    port_conn_layout = QHBoxLayout()
    port_conn_layout.addWidget(QLabel("埠號:"))
    ui.edtSharedMemPort = QLineEdit("9999")
    port_conn_layout.addWidget(ui.edtSharedMemPort)
    shared_mem_layout.addLayout(port_conn_layout)
    
    # 控制按鈕
    shared_btn_layout = QHBoxLayout()
    ui.bnStartSharedMem = QPushButton("啟動")
    ui.bnStopSharedMem = QPushButton("停止")
    ui.bnStopSharedMem.setEnabled(False)
    shared_btn_layout.addWidget(ui.bnStartSharedMem)
    shared_btn_layout.addWidget(ui.bnStopSharedMem)
    shared_mem_layout.addLayout(shared_btn_layout)
    
    ui.bnManualShare = QPushButton("手動分享")
    ui.bnManualShare.setEnabled(False)
    shared_mem_layout.addWidget(ui.bnManualShare)
    
    # 自動分享選項
    ui.chkAutoShare = QCheckBox("自動分享")
    ui.chkAutoShare.setChecked(False)
    ui.chkAutoShare.setEnabled(False)
    shared_mem_layout.addWidget(ui.chkAutoShare)
    
    # 狀態顯示
    ui.lblSharedMemStatus = QLabel("未啟動")
    ui.lblSharedMemStatus.setStyleSheet("color: red; font-size: 10px;")
    shared_mem_layout.addWidget(ui.lblSharedMemStatus)
    
    shared_mem_group.setLayout(shared_mem_layout)
    control_layout.addWidget(shared_mem_group)
```

Replace with:

```python
    # === 影像尺寸覆寫區塊 ===
    # 手動指定取像時要 reshape 成的寬高，取代相機回報的 nWidth/nHeight。
    # 留空或 0 代表不覆寫。不會存檔，每次重開程式都要重新輸入。
    shape_override_group = QGroupBox("影像尺寸覆寫")
    shape_override_layout = QVBoxLayout()

    shape_width_layout = QHBoxLayout()
    shape_width_layout.addWidget(QLabel("寬度:"))
    ui.edtShapeWidth = QLineEdit()
    ui.edtShapeWidth.setPlaceholderText("留空 = 使用相機回報值")
    ui.edtShapeWidth.setValidator(QIntValidator(1, 100000, ui.edtShapeWidth))
    shape_width_layout.addWidget(ui.edtShapeWidth)
    shape_override_layout.addLayout(shape_width_layout)

    shape_height_layout = QHBoxLayout()
    shape_height_layout.addWidget(QLabel("高度:"))
    ui.edtShapeHeight = QLineEdit()
    ui.edtShapeHeight.setPlaceholderText("留空 = 使用相機回報值")
    ui.edtShapeHeight.setValidator(QIntValidator(1, 100000, ui.edtShapeHeight))
    shape_height_layout.addWidget(ui.edtShapeHeight)
    shape_override_layout.addLayout(shape_height_layout)

    ui.lblShapeOverrideStatus = QLabel("未覆寫（使用相機回報值）")
    ui.lblShapeOverrideStatus.setStyleSheet("color: gray; font-size: 10px;")
    shape_override_layout.addWidget(ui.lblShapeOverrideStatus)

    shape_override_group.setLayout(shape_override_layout)
    control_layout.addWidget(shape_override_group)
```

- [ ] **Step 8: Add the handler that applies the override, next to the other `def select_save_path():`-style helpers**

Find:

```python
    def select_save_path():
```

Insert immediately before it:

```python
    def apply_manual_image_shape():
        """套用手動輸入的影像寬高覆寫；留空任一欄位則清除覆寫。"""
        width_text = ui.edtShapeWidth.text().strip()
        height_text = ui.edtShapeHeight.text().strip()

        width = int(width_text) if width_text else None
        height = int(height_text) if height_text else None

        set_manual_image_shape(width, height)

        if width and height:
            ui.lblShapeOverrideStatus.setText(f"已覆寫: {width} x {height}")
            ui.lblShapeOverrideStatus.setStyleSheet("color: green; font-size: 10px;")
        else:
            ui.lblShapeOverrideStatus.setText("未覆寫（使用相機回報值）")
            ui.lblShapeOverrideStatus.setStyleSheet("color: gray; font-size: 10px;")

    def select_save_path():
```

- [ ] **Step 9: Remove the shared-memory signal connections and timer, and connect the new shape fields**

Find:

```python
    # === 連接共享記憶體相關信號 ===
    ui.bnStartSharedMem.clicked.connect(start_shared_memory)
    ui.bnStopSharedMem.clicked.connect(stop_shared_memory)
    ui.bnManualShare.clicked.connect(manual_share_current_frame)
    ui.chkAutoShare.stateChanged.connect(toggle_auto_share)
    
    # === 連接圖片儲存相關按鈕事件 ===
    ui.bnSelectSavePath.clicked.connect(select_save_path)
    ui.chkImageSaveEnabled.stateChanged.connect(toggle_image_save)
    shared_mem_timer = QTimer()
    shared_mem_timer.timeout.connect(update_shared_memory_ui)
    shared_mem_timer.start(1000)  # 每秒更新一次
```

Replace with:

```python
    # === 連接影像尺寸覆寫欄位 ===
    ui.edtShapeWidth.editingFinished.connect(apply_manual_image_shape)
    ui.edtShapeHeight.editingFinished.connect(apply_manual_image_shape)

    # === 連接圖片儲存相關按鈕事件 ===
    ui.bnSelectSavePath.clicked.connect(select_save_path)
    ui.chkImageSaveEnabled.stateChanged.connect(toggle_image_save)
```

- [ ] **Step 10: Remove the shared-memory stop call from `cleanup()`**

Find:

```python
    def cleanup():
        print("Cleaning up resources...")
        # 停止 TCP 服務器
        try:
            stop_tcp_server()
        except:
            pass
        
        # 停止共享記憶體
        try:
            stop_shared_memory()
        except:
            pass
        
        # 關閉相機
        try:
            close_device()
        except:
            pass
        print("Cleanup finished.")
```

Replace with:

```python
    def cleanup():
        print("Cleaning up resources...")
        # 停止 TCP 服務器
        try:
            stop_tcp_server()
        except:
            pass
        
        # 關閉相機
        try:
            close_device()
        except:
            pass
        print("Cleanup finished.")
```

- [ ] **Step 11: Add the `QIntValidator` import used in Step 7**

Find (line 5):

```python
from PyQt5.QtGui import QImage, QPixmap
```

Replace with:

```python
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
```

- [ ] **Step 12: Verify the file compiles and no shared-memory references remain**

Run: `cd NIRcam-first && python -m py_compile BasicDemo.py`
Expected: no output (success)

Run: `cd NIRcam-first && grep -in "shared_mem\|sharedmemory\|global_sender\|共享記憶體\|共享內存\|receiver_host\|receiver_port" BasicDemo.py`
Expected: no matches

- [ ] **Step 13: Commit**

```bash
git add NIRcam-first/BasicDemo.py
git commit -m "$(cat <<'EOF'
Replace shared-memory panel with manual image shape override UI

Removes every shared-memory control, handler, global and signal
connection from the main window and adds a small "影像尺寸覆寫" panel
(width/height fields) that calls CamOperation_class.set_manual_image_shape
on edit. Not persisted across restarts, by design.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Delete the shared-memory sender/receiver scripts and sweep for stragglers

**Files:**
- Delete: `NIRcam-first/shared_memory_sender.py`
- Delete: `NIRcam-first/recive.py`
- Delete: `NIRcam-first/image_receiver.py`

- [ ] **Step 1: Delete the three files**

```bash
git rm NIRcam-first/shared_memory_sender.py NIRcam-first/recive.py NIRcam-first/image_receiver.py
```

- [ ] **Step 2: Repo-wide sweep for any remaining reference**

Run: `grep -rniI "shared_memory_sender\|SharedMemorySender\|SharedMemoryReceiver\|image_receiver\|recive\.py" --include=*.py .`
Expected: no matches (aside from this plan file itself, if grepping the whole repo — scope the grep to `*.py` under the project root, excluding `docs/`)

- [ ] **Step 3: Confirm the two modified files still import and compile cleanly**

Run: `cd NIRcam-first && python -m py_compile CamOperation_class.py BasicDemo.py image_shape.py`
Expected: no output (success)

- [ ] **Step 4: Run the full pure-logic test suite one more time**

Run: `cd NIRcam-first && python -m pytest test_image_shape.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
Remove standalone shared-memory sender/receiver scripts

shared_memory_sender.py had no remaining caller after the previous
two commits removed the UI and CamOperation_class hooks; recive.py
and image_receiver.py were its only consumers, on the receiving
machine, and are now dead too.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Manual smoke test (requires the physical camera — cannot be automated here)**

This step can't be verified by an agent without the camera hardware attached. Whoever runs this on the production machine should:
1. Launch `BasicDemo.py` (or the desktop launcher) and confirm the window opens with the new "影像尺寸覆寫" panel where "共享記憶體" used to be, and that there is no trace of the old shared-memory controls.
2. Open the device and start grabbing with the width/height fields left blank — confirm frames display normally (this exercises the `decode_raw_frame` fix on the actual buffer sizes the camera produces).
3. Type a width/height into the new fields, tab out of the field, and confirm `ui.lblShapeOverrideStatus` shows "已覆寫: W x H" and the live image still displays without a reshape error in the console.
4. Clear one of the two fields and confirm the status label reverts to "未覆寫" and frames keep displaying using the camera's own reported size.
