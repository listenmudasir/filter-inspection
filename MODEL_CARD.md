# Model card — supervised_global.pt

Deployed model for filter-cartridge surface AOI. Released as `v1.0`.

---

## 1. What it is

Supervised 4-class semantic segmentation over the whole cartridge, in one
forward pass.

```
frame → crop_to_content → illumination_correct → pad square → resize 768×768
      → WideResNet-50-2 encoder + FPN decoder
      → per-pixel logits: background / Bug / Foreign_Body / Stain
      → softmax → threshold 0.30 → connected components → boxes
```

| | |
|---|---|
| parameters | **68,576,633**, all trainable |
| encoder | `torchvision.wide_resnet50_2`, ImageNet init, **fine-tuned** (not frozen) |
| decoder | FPN — 1×1 laterals on the 4 stages → 128 ch, top-down add, smoothed, concatenated, 4-class head |
| input | 768×768 |
| checkpoint | 262 MiB fp32, epoch 17 |

**No anomaly scoring, no normality model, no calibration file, no threshold to
fit on site.** It replaces a hybrid in which 97% of the parameters were a
frozen one-class EfficientAD backbone that never saw a defect.

## 2. Training data

`wholeframe_data/`, built from `Unified_Dataset_seg`:

| split | frames | defect | normal |
|---|---|---|---|
| train | 963 | 819 | 144 |
| val | 186 | 158 | 28 |
| test | 317 | 266 | 51 |

1,115 polygon-annotated instances in train (Bug 284, Foreign_Body 311,
Stain 520). Splits are **burst-disjoint** — 100/17/35 distinct capture bursts,
zero shared. Frames inside a burst are milliseconds apart and near-identical,
so this matters: reshuffling them would put near-duplicates in train and test
and inflate every score.

Labels are SAM2 polygons prompted from human bounding boxes. **None are
human-verified** — see §6.

## 3. Training configuration

```
data                wholeframe_data        batch size          4
epochs              40 (stopped at 18)     steps/epoch         400
optimiser           AdamW                  weight decay        1e-4
schedule            OneCycle, pct_start 0.15
decoder LR          2e-4                   encoder LR          2e-5  (×0.1)
defect_fraction     0.8                    AMP                 on
loss                focal CE (γ=2, background weight 0.2) + Dice over defect classes
augmentation        h/v flip, rot90, photometric jitter (α ±0.16, β ±14)
```

**Discriminative learning rate is not optional.** A first run at a single
`lr=2e-4` for the whole network diverged: Bug IoU went 0.571 → 0.402 → **0.000**
and stayed there three epochs while training loss *rose* 0.549 → 0.706. An
ImageNet-pretrained encoder and a randomly-initialised decoder cannot share a
learning rate.

**Copy-paste augmentation was tried and rejected.** Pasting real instances at
`paste_prob 0.6` on top of `defect_fraction 0.8` meant nearly every sample
contained a defect; the class prior collapsed and F1 fell to **0.137** with 256
false alarms on 51 clean frames. Not used.

Loss design follows a measured ratio: defect pixels are **0.178%** of all pixels
(560:1). Plain CE converges to predicting background everywhere. Tversky
α=0.75/β=0.25 from the old pipeline is deliberately **not** used — it penalises
false positives 3× harder than false negatives and drove predicted/GT area to
0.147.

## 4. Training curve

```
 ep    loss  meanIoU    Bug     FB  Stain
  0  0.9944   0.0163  0.000  0.038  0.011
  2  0.8557   0.3057  0.391  0.378  0.148
  5  0.3403   0.3524  0.501  0.426  0.131
  7  0.3538   0.4507  0.602  0.395  0.355
 11  0.2235   0.5032  0.661  0.510  0.339
 15  0.2232   0.5118  0.601  0.604  0.330
 17  0.1876   0.5122  0.606  0.553  0.378   <-- selected
 18  0.1840   0.4038  0.464  0.550  0.198
```

**The run was interrupted at epoch 18 of 40.** Epoch 17 was the best validation
checkpoint at that point. Whether a full 40-epoch run does better is untested —
this is not a converged model.

Checkpoint selection used mean defect IoU, which averages in Stain. Stain
oscillates ±0.1 between adjacent epochs, so selection is partly noise-driven.

## 5. Results — locked 317-image test split

IoU 0.25, class-agnostic Hungarian matching, `prob_threshold 0.30`,
**single frame, no track consensus**.

| | old hybrid | **this model** |
|---|---|---|
| F1 | 0.559 | **0.631** |
| precision | 0.484 | **0.706** |
| recall | 0.662 | 0.568 |
| **false components on 51 clean frames** | 9 | **0** |
| predicted/GT area ratio | 0.147 | **0.931** |

By class (end-to-end recall = localised **and** correctly typed):

| class | old | this model | type accuracy |
|---|---|---|---|
| **Bug** | 0.255 | **0.734** | 0.333 → **0.855** |
| Foreign_Body | 0.641 | 0.451 | 0.835 → 0.759 |
| Stain | 0.351 | 0.298 | 0.660 → 0.933 |

By defect size (recall):

| size | old | this model |
|---|---|---|
| small <5k px | — | 0.640 |
| medium 5k–50k | — | 0.620 |
| **large ≥50k** | **0.303** | **0.030** |

Threshold 0.30 was selected on the **validation** split (F1 0.810 there) and
applied once to test. It was not tuned against test.

## 6. Limitations

**Large diffuse staining is not covered.** Recall 0.030 vs the old stack's
0.303. Evidence points at labelling rather than the model:

- the same model scores **0.917** on validation's large instances (11/12)
- **23 of the 32** large test instances are flagged in
  `label_audit/suspect_labels.csv` at median local contrast **1.30** — below
  the ~2.3 just-noticeable-difference threshold
- large-defect recall is **flat across every threshold 0.30–0.90**, meaning the
  model produces no weak evidence being thresholded away

Until those annotations are adjudicated, **assume large stains are not
detected.** Do not rely on this model alone for that class.

**Labels are unverified.** Every defect frame is `pending_review_sam2`; 11.5%
of instances are flagged `TOUCHES_CLIP` (mask reached the prompt-box edge),
18.4% flagged in `label_quality.json`, and 151 suspected *unlabelled* defects
are open in the test split. Reported precision is therefore a lower bound.

**Never validated on live camera frames.** All numbers come from stored images.

**Three capture sessions only** (2026-07-24, 08-11, 08-20). Generalisation to a
new lot, new lighting or a cleaned lens is unmeasured.

## 7. Runtime

RTX A5000, live camera resolution 2200×2048, measured through
`SupervisedDetector` over 80 frames:

| | mean | p95 | FPS |
|---|---|---|---|
| CPU preprocessing (original) | 311 ms | — | 3.22 |
| **GPU preprocessing (shipped)** | **38.6 ms** | 41.3 ms | **25.92** |

Against the line (2.10 FPS capture, 10 frames per cartridge over 4.3 s):

| budget | available | used | |
|---|---|---|---|
| **per frame** | 476 ms | **38.6 ms** | ✅ 12× headroom |
| per cartridge | 4300 ms | 386 ms (all 10 frames) | ✅ 11× headroom |

**The network was never the bottleneck — preprocessing was, by 13×.**
Measured stage costs before the port:

```
content_bbox      75 ms    numpy mean/std reductions over 4.5 MP
illum_correct    176 ms    cvtColor + float32 add/clip at 4.5 MP
pad + resize      24 ms
forward pass      22 ms    <-- 7% of the frame
```

Raising `cv2.setNumThreads` from 1 to 20 changes this by 0.5 ms: the hot
stages are numpy reductions and float32 array arithmetic and never enter
OpenCV's thread pool. The fix was to stop doing 4.5-megapixel work for a
768×768 model — see `inspection/gpu_preprocess.py`.

Equivalence was measured, not assumed (`scripts/validate_fast_preprocess.py`,
full 317-image test set): F1 0.6016 → 0.5974, clean-frame false alarms 0 → 0,
crop box identical on 317/317 frames, component count differing on 28/317.

**Precision variants** (all 317 frames, same protocol):

| variant | forward | FPS | TP | F1 |
|---|---|---|---|---|
| **fp32 (shipped)** | 22.3 ms | 2.94 | **188** | **0.6016** |
| autocast | 15.6 ms | 3.00 | 186 | 0.5971 |
| fp16 | 13.4 ms | 3.02 | 186 | 0.5971 |
| fp16 + channels_last | 13.2 ms | 3.02 | 186 | 0.5971 |

*(FPS column measured before the preprocessing port, so it reflects the old
311 ms frame; the forward-pass and accuracy columns are unaffected.)*

Half precision costs **2 true positives** — detections sitting near the 0.30
threshold move. With 12× frame headroom already, that trade is not worth
making, so fp32 is the default. int8 is unmeasured (TensorRT not installed).

**Edge extrapolation, not measured:** a Jetson Orin AGX has roughly ⅕ the
fp32 throughput of an A5000. Scaling the GPU-bound portion suggests ~150–200
ms/frame (5–7 FPS) — still above the 2.10 FPS line rate, but this is an
estimate from throughput ratios and has not been run on the hardware.

Geometry is defined at **model scale** (min area 20 px, border 49 px at 768),
not camera pixels, so the detector is resolution-independent. Verified
identical behaviour at 4096×3650 and 2200×2048.

## 8. Reproducing

```bash
python scripts/build_wholeframe_data.py --out wholeframe_data --size 768
python scripts/train_tile_segmenter.py \
    --data wholeframe_data --out runs/global_branch \
    --epochs 40 --steps-per-epoch 400 --batch-size 4 \
    --lr 2e-4 --encoder-lr-scale 0.1 --defect-fraction 0.8
python scripts/eval_tile_segmenter.py \
    --checkpoint runs/global_branch/best.pt --whole-frame --input-size 768 \
    --prob-threshold 0.30 --fast-postprocess
```

Those scripts live in the research repo, not in this deployment package.
