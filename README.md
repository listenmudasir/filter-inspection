# filter-inspection

Water-filter cartridge surface AOI: Hikvision camera GUI + supervised defect
segmentation, in one repository.

Clone → copy weights → `selftest.py` → run.

---

## 1. Install

```bash
git clone https://github.com/listenmudasir/filter-inspection.git
cd filter-inspection
```

**Install torch first**, matched to your CUDA:

```bash
conda create -n nircam python=3.10 -y
conda activate nircam
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Copy the model weights.** `supervised_global.pt` is 262 MiB — above GitHub's
100 MB per-file limit — so it is not in git:

```bash
cp /path/to/supervised_global.pt weights/
```

Verify before trusting the line:

```bash
python selftest.py      # must print PASSED
```

## 2. Run

```bash
cd NIRcam-first
./run_gui.sh            # Linux    (run_gui.bat on Windows)
```

**Use `run_gui.sh`, not `python BasicDemo.py`.** The launcher sanitises two
environment problems that otherwise abort Qt — see §6.

In the GUI: **查找設備** → **打開設備** → **載入模型** → **開始採集**.

On success:

```
Supervised global segmenter loaded (epoch 17, threshold 0.3, 768px)
```

If you instead see a boxed **WARNING**, the supervised model did not load and
the old hybrid path was attempted. Stop and fix it.

## 3. What it does, measured

Locked 317-image test split, IoU 0.25 Hungarian matching, **single frame, no
track consensus**:

| | old hybrid | **this model** |
|---|---|---|
| F1 | 0.559 | **0.631** |
| precision | 0.484 | **0.706** |
| recall | 0.662 | 0.568 |
| **false alarms on 51 clean frames** | 9 | **0** |
| predicted/GT area ratio | 0.147 | **0.931** |
| Bug end-to-end recall | 0.255 | **0.734** |
| Bug type accuracy | 0.333 (chance) | **0.855** |

**Zero detections on all 51 clean frames** is the number that matters on a
line: no good cartridge is rejected.

### Known limitation — read before setting reject criteria

**Large diffuse staining (≥50k px): recall 0.030**, versus the old stack's
0.303.

Evidence indicates this is largely a *labelling* problem, not a model one: the
same model scores **0.917** on the validation split, and **23 of the 32** large
test instances are flagged in the label audit at median local contrast **1.30**
— below what a human can reliably see. But until those annotations are
adjudicated, **assume large stains are not detected.** If that class matters
for your accept/reject rule, do not rely on this model alone for it.

## 4. Timing

Measured on this line's own data (RTX A5000):

```
capture rate           2.10 FPS   (476 ms/frame)
frames per cartridge   10         over a 4.3 s burst
```

| budget | | this model |
|---|---|---|
| per frame | 476 ms | ~1000–1400 ms — **does not fit** |
| **per cartridge** | **4300 ms** | **~1000–1400 ms — fits, ~3× headroom** |

Run it **once per cartridge** (觸發模式), not per frame, and combine the burst
with track consensus. In 連續模式 at 14 FPS the frames will queue behind a
~1.2 s model.

Where the time goes:

```
crop + illumination   ~640 ms   65%   CPU
forward pass            45 ms    5%   GPU
everything else       ~300 ms
```

**The network is not the bottleneck.** Quantisation, fp16 or a smaller
architecture buy essentially nothing; the remaining speedup is a preprocessing
port, and it is not needed to meet the per-cartridge budget.

## 5. The method

Supervised 4-class semantic segmentation over the whole cartridge:

```
frame → crop_to_content → illumination_correct → pad square → resize 768
      → WideResNet-50-2 + FPN   (68.5M params, fine-tuned end to end)
      → per-pixel: background / Bug / Foreign_Body / Stain
      → threshold 0.30 → connected components → boxes
```

One forward pass. No anomaly scoring, no normality model, **no calibration file
and no threshold to fit on site**. Trained directly on 1,115 annotated defect
instances — replacing a hybrid in which 97% of the parameters were a frozen
one-class backbone that never saw a defect.

Geometry is defined at **model scale** (min area 20 px, border 49 px at 768),
not in camera pixels, so the detector works on any camera resolution without
recalibration. Verified identical behaviour at 4096×3650 and 2200×2048.

## 6. Troubleshooting

**`could not load the Qt platform plugin "xcb"` → `Aborted (core dumped)`**
Two independent causes, both from the shell, both handled by `run_gui.sh`:
- `/opt/MVS/bin` on `LD_LIBRARY_PATH`. The Hikvision SDK ships its own Qt5,
  which wins over PyQt5's and fails on
  `undefined symbol: _ZN23QPlatformVulkanInstance22presentAboutToBeQueuedEP7QWindow`.
  The camera libraries live in `/opt/MVS/lib/64` and do not need `bin`.
- `QT_QPA_PLATFORM_PLUGIN_PATH` left pointing at a *different* conda env.

**GUI shows 「AI 模型未載入」 although the model loaded**
`CamOperation_class` gates its whole AI branch on `detect_objects`, which is
`None` when `detect.py` fails to import. `detect.py` imports `ultralytics`
lazily here for exactly this reason — if you restore an eager import and
ultralytics is absent, the model loads and is then never called.

**`_ARRAY_API not found`** — numpy 2.x against an opencv built for 1.x.
`pip install "numpy<2"`.

**Weights not found** — copy `supervised_global.pt` into `weights/`.

## 7. Layout

```
filter-inspection/
├── NIRcam-first/                 camera GUI (PyQt5 + Hikvision MVS SDK)
│   ├── BasicDemo.py              entry point
│   ├── supervised_detect.py      THE METHOD: model + pre/post + GUI adapter
│   ├── hybrid_detect.py          load_model dispatch: supervised → hybrid → YOLO
│   ├── detect.py                 draw/detect helpers (lazy ultralytics)
│   ├── CamOperation_class.py     camera thread + per-frame AI call
│   └── run_gui.sh / run_gui.bat  launcher WITH environment sanitising
├── inspection/
│   └── enhance.py                crop_to_content + illumination_correct
├── weights/                      supervised_global.pt goes here (not in git)
├── requirements.txt
└── selftest.py
```

`inspection/enhance.py` is **not** generic preprocessing — it is the exact
transform the training data was built with. Replacing it changes what the model
sees.

## 8. What was changed in the GUI

Only two files, minimally:

- **`hybrid_detect.py`** — try the supervised model first; fall back **loudly**
  with the accuracy difference printed, so a silent downgrade is impossible.
- **`detect.py`** — `from ultralytics import YOLO` moved inside `load_model()`.
  Neither `detect_objects` nor `draw_custom_boxes` needs it.

Everything else — UI, TCP, tracking, boundary filter, image saving — is
untouched: the detector implements the same interface an ultralytics model
does (`model(img, conf=, imgsz=) → [Result]` with
`.boxes.xyxy/.conf/.cls` and `.names`).
