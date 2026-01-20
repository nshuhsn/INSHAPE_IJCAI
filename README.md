# INSHAPE: Instance-Level Shapelets for Interpretable Time-Series Classification

This repository contains the official implementation of **INSHAPE** for IJCAI.

## Installation
```bash
pip install -r requirements.txt
```

For PyTorch, install separately based on your CUDA version:
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Dataset Setup

Place your datasets in the following structure:

```
datasets/
├── UCRConverted/                    # UCR Time Series Archive
│   ├── CBF/
│   │   ├── CBF_TRAIN.ts
│   │   └── CBF_TEST.ts
│   └── ...
└── Multivariate_ts/                    # UEA Multivariate Archive
    ├── CharacterTrajectories/
    │   ├── CharacterTrajectories_TRAIN.ts
    │   └── CharacterTrajectories_TEST.ts
    └── ...
```

## Usage

### Training on UCR datasets
```bash
python scripts/run_UCR.py --ablation
```

### Training on UEA datasets
```bash
python scripts/run_UEA.py --ablation
```

### Shapelet Extraction
```bash
python scripts/Global_full_pipeline_UCR.py --dataset CBF
python scripts/Global_full_pipeline_UEA.py --dataset CharacterTrajectories
```

### Filter Similar Shapelets
```bash
python scripts/filter_similar_shapelets.py --dataset ECG5000 --threshold 0.8
```

## Project Structure

```
INSHAPE_IJCAI/
├── models/          # Core model architecture
├── layers/          # Network building blocks  
├── data_provider/   # Data loading utilities
├── scripts/         # Training and evaluation scripts
├── configs/         # Configuration files
├── Visualization/        # Visualization notebooks
└── utils/           # Helper functions
```
