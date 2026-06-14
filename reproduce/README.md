# AttSiOff Reproduction: ENsiRNA Dataset 5-Fold Training

This directory contains scripts to reproduce training AttSiOff using the ENsiRNA dataset.

## Prerequisites

- Pixi (https://pixi.sh)
- Python 3.10+
- CUDA-capable GPU (optional, for faster training on Linux)
- Apple Silicon (MPS) for macOS training
- ENsiRNA dataset (train_*.csv, valid_*.csv in ENsiRNA/ENsiRNA/dataset/)

## Steps

### 1. Setup Environment

```bash
cd reproduce
pixi install
```

### 2. Download RNA-FM Pretrained Weights

```bash
bash download_weights.sh
```

This downloads RNA-FM pretrained weights from `cuhkaih/rnafm` on HuggingFace
to `~/.cache/torch/hub/checkpoints/RNA-FM_pretrained.pth`.

### 3. Prepare Data

Edit `prepare_data.py` to set the correct `DATA_DIR` pointing to your ENsiRNA dataset, then:

```bash
pixi run python prepare_data.py
```

This generates AttSiOff-compatible CSV files and RNA-FM embeddings for all 5 folds
under `reproduce/data/fold_{1-5}/`.

### 4. Train (5-Fold Cross-Validation)

```bash
pixi run python train_5fold.py
```

Trained models saved to `reproduce/output/fold_{1-5}/best_model.pth.tar`.

## Notes

- RNA-FM embeddings are generated on-the-fly during data preparation using the
  `fm` package (RNA-FM model, ~105M parameters, RoBERTa base architecture).
- The antisense sequences are padded from 19nt to 21nt with 'AA' at the 3' end
  to match AttSiOff's expected input dimensions.
- The mRNA window is 59nt centered on the siRNA binding site.
- Missing features (s-Biopredsi, DSIR, i-score) are set to 0.
