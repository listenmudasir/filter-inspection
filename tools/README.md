# tools/

Offline data-preparation scripts. **Not part of the runtime** — nothing under
`inspection/` or `NIRcam-first/` imports from here, and the production install
in `requirements.txt` does not cover these.

## build_unified_seg_dataset.py

Converts human-drawn YOLO detection boxes into YOLO-seg polygons by prompting
SAM2 with each box. This is where every training label in the project comes
from; no mask was drawn by hand.

Extra requirements beyond `requirements.txt`:

```
pip install ultralytics          # deliberately excluded from the runtime set
```

plus the SAM2 base checkpoint (`sam2_b.pt`, ~160 MB), not committed here.

Run:

```bash
python tools/build_unified_seg_dataset.py \
    --source  /path/to/Unified_Dataset \
    --output  /path/to/Unified_Dataset_seg \
    --sam-model /path/to/sam2_b.pt
```

The `--source`, `--output` and `--sam-model` defaults are absolute paths to the
original development machine and must be overridden elsewhere.

### Known limitation

When SAM2 returns no usable component the script falls back to the literal
bounding rectangle, flagged `SAM_EMPTY` in the report CSV. Loop-shaped hairs
also come back filled, flagged `HIGH_FILL`. Both produce labels much larger
than the defect, and both are recorded rather than corrected — review those
flags before trusting a fresh label set.
