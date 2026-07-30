# Model weights

Checkpoints for all 9 Perception microservices live here. Download instructions: **[`WEIGHTS_BAIDU.md`](WEIGHTS_BAIDU.md)**.

| File | Service |
|------|---------|
| `sam2.1_hiera_small.pt` | SAM2 |
| `RemoteCLIP-ViT-L-14.pt` | RemoteCLIP |
| `RemoteSAMv1.pth` | RemoteSAM |
| `changeos_r101.pt` | ChangeOS |
| `SiamCRNN_20250211_033615_1/best_model.pth` | BRIGHT |
| `strip_rcnn_s_fair1m.pth` | Strip-R-CNN |
| `iter_33468.pth` | SM3Det |
| `Qwen2_5_vl-7b/` | Qwen2.5-VL |
| `sam2_hiera_large.pt` | InstructSAM |

```bash
cd tools/deploy_tools
export RSMEM_MODELS_SRC=/path/to/checkpoints   # optional
bash setup_models.sh
```

Custom directory: `export RSMEM_MODELS_DIR=/your/path`

Deployment: [`../tools/deploy_tools/README.md`](../tools/deploy_tools/README.md)
