# Model Weights (Baidu Pan)

Perception microservice checkpoints are large (**~24 GB**) and **not included in this repo**. Download from Baidu Pan and extract to `<repo>/models/`.

> **Maintainers:** After upload, fill in the share link and extraction code in the tables below.

---

## User download

### All-in-one package (recommended)

| Package | Contents | Size (approx.) | Baidu Pan link | Code |
|---------|----------|----------------|----------------|------|
| RSMeM-tools-weights-all | All 9 weight sets | ~24 GB | `TBD` | `TBD` |

### Per-model download

| Service | File / directory | Target path | Size (approx.) | Baidu Pan link | Code |
|---------|------------------|-------------|----------------|----------------|------|
| SAM2 | `sam2.1_hiera_small.pt` | `models/sam2.1_hiera_small.pt` | 176 MB | `TBD` | `TBD` |
| RemoteCLIP | `RemoteCLIP-ViT-L-14.pt` | `models/RemoteCLIP-ViT-L-14.pt` | 1.6 GB | `TBD` | `TBD` |
| RemoteSAM | `RemoteSAMv1.pth` | `models/RemoteSAMv1.pth` | 2.4 GB | `TBD` | `TBD` |
| ChangeOS | `changeos_r101.pt` | `models/changeos_r101.pt` | 223 MB | `TBD` | `TBD` |
| BRIGHT | `SiamCRNN_20250211_033615_1/` | `models/SiamCRNN_.../best_model.pth` | 127 MB | `TBD` | `TBD` |
| Strip-R-CNN | `strip_rcnn_s_fair1m.pth` | `models/strip_rcnn_s_fair1m.pth` | 358 MB | `TBD` | `TBD` |
| SM3Det | `iter_33468.pth` | `models/iter_33468.pth` | 3.1 GB | `TBD` | `TBD` |
| Qwen2.5-VL | `Qwen2_5_vl-7b/` | `models/Qwen2_5_vl-7b/` | 16 GB | `TBD` | `TBD` |
| InstructSAM | `sam2_hiera_large.pt` | `models/sam2_hiera_large.pt` | 857 MB | `TBD` | `TBD` |

### Install locally

```bash
# 1. Download RSMeM-tools-weights-all from Pan and extract
cd RSMeM-public
tar -xf RSMeM-tools-weights-all.tar   # or unzip
mv RSMeM-tools-weights-all/* models/

# 2. Verify
ls models/sam2.1_hiera_small.pt
ls models/Qwen2_5_vl-7b/config.json

# 3. Deploy
cd tools/deploy_tools && bash setup_models.sh && source setup_env.sh && bash start_all.sh
```

---

## Maintainers: pack and upload to Baidu Pan

Use PyPI **[bypy](https://pypi.org/project/bypy/)** (supports resume).

### Step 1: Install bypy

```bash
pip install bypy -i https://pypi.tuna.tsinghua.edu.cn/simple
# or
python3 -m pip install bypy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Step 2: Authorize (first time)

**Option A — Device code (recommended)**

```bash
bypy info
```

Follow the link, open `https://pan.baidu.com/device` in a browser, enter the code. Success creates `~/.bypy/token.json`.

**Option B — BDUSS cookie**

After logging in at pan.baidu.com, copy the `BDUSS` cookie value:

```bash
export BDUSS=your_bduss_value
```

### Step 3: Pack weights

Copy all 9 weight sets (resolve symlinks) into a release directory and create a tar archive:

```bash
RELEASE=RSMeM-public/models_release/RSMeM-tools-weights-all
mkdir -p "$RELEASE"
# Copy checkpoints from gkd-agent-models and Qwen2_5_vl-7b into $RELEASE/
tar -cf RSMeM-public/models_release/RSMeM-tools-weights-all.tar -C models_release RSMeM-tools-weights-all
```

### Step 4: Upload to Pan

```bash
pip install bypy -i https://pypi.tuna.tsinghua.edu.cn/simple
bypy info   # first-time auth
# Since 2025+, use whole-file upload to avoid Slice MD5 mismatch:
bypy --processes 1 -s 2000G upload models_release/RSMeM-tools-weights-all RSMeM-tools-weights-all
```

Remote path: `RSMeM-tools-weights-all` (maps to `/apps/bypy/RSMeM-tools-weights-all` under **My Apps → bypy** on Pan).

~24 GB; may take hours depending on bandwidth. Re-run the upload command to resume after interruption.

> **Slice MD5 mismatch:** Baidu PCS slice upload broke in 2025; bypy’s default 20 MB slices fail MD5 checks. Use `-s 2000G` for whole-file upload ([bypy#741](https://github.com/houtianze/bypy/issues/741)).

### Step 5: Create a share link

1. Open [Baidu Pan](https://pan.baidu.com)
2. Go to **My Apps → bypy → RSMeM-tools-weights-all**
3. Select the folder → **Share** → copy link and extraction code
4. Update the tables at the top of this document

---

## Pan directory layout

```text
RSMeM-tools-weights-all/
├── MANIFEST.txt
├── sam2.1_hiera_small.pt
├── RemoteCLIP-ViT-L-14.pt
├── RemoteSAMv1.pth
├── changeos_r101.pt
├── SiamCRNN_20250211_033615_1/
│   └── best_model.pth
├── strip_rcnn_s_fair1m.pth
├── iter_33468.pth
├── sam2_hiera_large.pt
└── Qwen2_5_vl-7b/
    ├── config.json
    └── …
```

---

## Related docs

- [`README.md`](README.md) — Weight directory overview
- [`../tools/deploy_tools/README.md`](../tools/deploy_tools/README.md) — Deployment guide
