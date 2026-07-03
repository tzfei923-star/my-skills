---
name: fluorescence-coexpression
description: >-
  Analyze SOX2/GFP (or any two-channel) fluorescence co-expression from confocal
  microscopy images. Use this skill whenever the user uploads fluorescence
  microscopy images (CZI, TIFF, JPG) and asks about co-expression,
  co-localization, overlap between two markers, or percentage of double-positive
  cells. Triggers on phrases like "算共表現", "co-expression", "GFP+ SOX2+",
  "兩個channel重疊", "共表現比例", or when the user uploads paired channel images.
---

# Fluorescence Co-expression Analysis Skill

Pixel-level co-expression analysis for two-channel fluorescence confocal images.
Supports CZI (Zeiss), TIFF (single-channel), and JPEG inputs.

---

## Step 1 — Identify Input Format

| Format | Bit depth | Approach |
|--------|-----------|----------|
| .czi (Zeiss raw) | 16-bit per channel | Parse ZISRAWSUBBLOCK segments directly |
| .tif single-channel | 8-bit (ZEN export) | PIL or tifffile; each file = one marker |
| .jpg paired | 8-bit RGB | Extract R channel (SOX2/magenta) and G channel (GFP/green) |

Ask the user to confirm channel assignment if ambiguous:
- SOX2 = magenta/purple → R channel of RGB
- GFP = green → G channel of RGB

---

## Step 2 — Read Image Data

### CZI (16-bit raw)

```python
import struct, numpy as np

def load_czi_channels(path):
    with open(path, "rb") as f:
        raw = f.read()
    segments = []
    pos = 0
    while pos < len(raw) - 32:
        seg_id = raw[pos:pos+16].rstrip(b"\x00").decode("ascii", errors="ignore")
        if seg_id in ["ZISRAWFILE","ZISRAWMETADATA","ZISRAWSUBBLOCK",
                      "ZISRAWDIRECTORY","ZISRAWATTDIR","ZISRAWATTACHMENT"]:
            alloc = struct.unpack_from("<q", raw, pos+16)[0]
            segments.append((pos, seg_id, alloc))
            pos += 32 + alloc
        else:
            pos += 1
    imgs = {}
    for pos, sid, alloc in segments:
        if sid != "ZISRAWSUBBLOCK":
            continue
        sb_start = pos + 32
        meta_size = struct.unpack_from("<i", raw, sb_start)[0]
        dir_start = sb_start + 16
        dir_size  = struct.unpack_from("<i", raw, dir_start)[0]
        dim_count = struct.unpack_from("<i", raw, dir_start+28)[0]
        ch_idx = 0; sx = 465; sy = 465
        for d in range(dim_count):
            doff = dir_start + 32 + d*20
            dn  = raw[doff:doff+4].rstrip(b"\x00").decode("ascii","ignore")
            val = struct.unpack_from("<i", raw, doff+4)[0]
            sz  = struct.unpack_from("<i", raw, doff+8)[0]
            if dn=="C": ch_idx=val
            if dn=="X": sx=sz
            if dn=="Y": sy=sz
        data_offset = dir_start + dir_size + meta_size
        avail = len(raw) - data_offset
        npx = sx * sy
        if avail >= npx*2:
            arr = np.frombuffer(raw[data_offset:data_offset+npx*2],
                                dtype=np.uint16).reshape(sy,sx).copy()
        else:
            buf = bytearray(npx*2)
            buf[:avail] = raw[data_offset:data_offset+avail]
            arr = np.frombuffer(bytes(buf), dtype=np.uint16).reshape(sy,sx).copy()
        imgs[ch_idx] = arr
    return imgs  # {0: array, 1: array, 2: array, 3: array}
```

### TIFF (8-bit ZEN export, single channel per file)

```python
from PIL import Image
import numpy as np

def load_tiff_channel(path, rgb_idx):
    """rgb_idx: 0=R(SOX2), 1=G(GFP), 2=B"""
    arr = np.array(Image.open(path).convert("RGB"))
    return arr[:,:,rgb_idx].astype(np.float32)
```

### JPEG (paired RGB images)

```python
def load_jpg_channels(sox2_path, gfp_path):
    sox2 = np.array(Image.open(sox2_path).convert("RGB"))[:,:,0].astype(np.float32)
    gfp  = np.array(Image.open(gfp_path).convert("RGB"))[:,:,1].astype(np.float32)
    return sox2, gfp
```

---

## Step 3 — Threshold Selection

Choose based on signal distribution shape and bit depth:

| Condition | Method | Rationale |
|-----------|--------|-----------|
| SOX2, any format | Otsu | Usually bimodal (background + signal peaks) |
| GFP, 16-bit CZI | mean + 2SD | Right-skewed, no clear bimodal |
| GFP, 8-bit TIFF (ZEN export) | Otsu | Already display-normalized |
| GFP, 8-bit JPEG (dark background) | mean + 2SD | Large zero-peak distorts Otsu |
| Both channels very bright (high background) | p80 percentile | Use consistent percentile for both |

```python
from skimage.filters import threshold_otsu

def get_threshold(arr, method="otsu"):
    if method == "otsu":
        return threshold_otsu(arr)
    elif method == "mean2sd":
        return arr.mean() + 2 * arr.std()
    elif method == "p80":
        return np.percentile(arr, 80)
```

Warning signs that threshold is wrong:
- SOX2+ or GFP+ exactly = 20% → likely using forced percentile, not real signal
- Co-exp > 80% → threshold too low, capturing background overlap
- Co-exp < 0.5% with clearly visible co-expressing cells → threshold too high

---

## Step 4 — Calculate Co-expression

```python
def coexpression_analysis(sox2, gfp, sox2_th, gfp_th):
    sox2_mask = sox2 > sox2_th
    gfp_mask  = gfp  > gfp_th
    coexp     = sox2_mask & gfp_mask
    total     = sox2.size
    results = {
        "total_px":    total,
        "sox2_px":     int(sox2_mask.sum()),
        "gfp_px":      int(gfp_mask.sum()),
        "coexp_px":    int(coexp.sum()),
        "sox2_pct":    round(100 * sox2_mask.sum() / total, 1),
        "gfp_pct":     round(100 * gfp_mask.sum()  / total, 1),
        "coexp_pct":   round(100 * coexp.sum()      / total, 1),
        "coexp_of_gfp":  round(100 * coexp.sum() / max(gfp_mask.sum(), 1), 1),
        "coexp_of_sox2": round(100 * coexp.sum() / max(sox2_mask.sum(), 1), 1),
    }
    return results, sox2_mask, gfp_mask, coexp
```

Primary metric = Co-exp / GFP+ (co-expressing pixels ÷ GFP+ pixels)
This answers: "Of all GFP-labeled neurons, what fraction also express SOX2?"

---

## Step 5 — Generate Output Images

```python
from PIL import Image

def save_output_images(sox2, gfp, sox2_mask, gfp_mask, coexp, out_dir):
    def autonorm(a, lo=1, hi=99):
        mn = np.percentile(a, lo); mx = np.percentile(a, hi)
        return np.clip((a-mn)/max(mx-mn,1)*255, 0, 255).astype(np.uint8)
    s = autonorm(sox2); g = autonorm(gfp)
    h, w = s.shape
    # Composite: SOX2=magenta(R+B), GFP=green(G)
    composite = np.stack([s, g, s], axis=2)
    # Mask overlay
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    mask[sox2_mask & ~gfp_mask] = [220, 0, 220]   # SOX2 only = magenta
    mask[gfp_mask & ~sox2_mask] = [0, 220, 0]     # GFP only = green
    mask[coexp]                  = [255, 255, 0]   # Co-expression = yellow
    blend = (composite.astype(float)*0.4 + mask.astype(float)*0.6).clip(0,255).astype(np.uint8)
    Image.fromarray(blend).save(f"{out_dir}/analysis.png")
    Image.fromarray(mask).save(f"{out_dir}/mask.png")
```

---

## Step 6 — Present Results

Use an interactive widget with these sections (per user preference):

1. 公式說明 (Formula box)
   - Show fraction: Co-exp px / GFP+ px = XX%
   - Show fraction: Co-exp px / SOX2+ px = XX%
   - Include exact pixel counts in numerator/denominator
2. 像素統計卡片 (Pixel stats cards)
   - GFP+ 面積 (% and px count)
   - SOX2+ 面積 (% and px count)
   - 共表現面積 (% and px count)
3. 長條圖 (Bar chart)
   - Co-exp/GFP+ bar with division shown inline
   - Co-exp/SOX2+ bar with division shown inline
4. 各 channel 詳細數據 (Per-channel detail cards)
   - Format, image size, mean intensity, SD, threshold value, positive pixels
5. 閾值說明 + 注意事項 (Threshold notes)
   - Which method used for each channel and why
   - JPEG compression warning if applicable
   - Truncation warning if CZI data was incomplete

---

## Step 7 — Multi-sample Summary

When multiple samples are ready, group by experimental condition and report:

```
Group A (e.g. sh26, ARID1b KD):
  - Sample 1: XX%
  - Sample 2: XX%
  mean ± SD = XX% ± X%

Group B (e.g. ctrl, TRC2):
  - Sample 1: XX%
  ...
  mean ± SD = XX% ± X%
```

Present in a widget with grouped bar charts, individual sample value cards,
and mean±SD summary cards per group.

---

## Key Caveats to Always Mention

- Pixel-area vs cell-count: This method counts pixels, not individual cells.
  Results approximate cell-level co-expression only when cell sizes are similar.
- JPEG compression: Introduces color mixing at boundaries; prefer TIFF or CZI.
- MAX projection: Inflates co-expression by stacking all Z-planes; single-plane
  is more conservative and closer to true cell-level overlap.
- 16-bit CZI with high background: Threshold choice strongly affects results;
  document the method used and apply consistently across samples.
- Truncated CZI: If file size mismatch detected, pad with zeros and warn user.
  Results for that channel are unreliable.
