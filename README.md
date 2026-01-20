# INSHAPE: Instance-Level Shapelets for Interpretable Time-Series Classification
This repository contains the official implementation of **INSHAPE** for IJCAI.

![INSHAPE_FIG](https://github.com/user-attachments/assets/827ec743-942d-44d0-bce8-7e4ef171d9be)
Overview of our INSHAPE framework.
    (Top) During training, the transition point algorithm segments the input time series, and the shared stochastic selector learns a Bernoulli parameter for each segment to identify discriminative regions. 
    The gate vector masks non-selected segments with zeros (while preserving positional information), and the predictor performs TSC based on the masked time series.
    To make gradient flow through the non-differentiable discrete Bernoulli sampling process, we employ a gradient estimator (i.e., ReinMax), allowing end-to-end training. 
    (Bottom) For population-level shapelet discovery, the pre-trained (frozen) selector extracts instance-level shapelets across the dataset, which are then clustered using FastDTW to obtain representative cluster centroids.

<img width="15780" height="12718" alt="fig1_motivation_ver2" src="https://github.com/user-attachments/assets/c3ded632-9666-4256-a522-7270dbff3e56" />



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

<img width="10000" height="9758" alt="Group 283" src="https://github.com/user-attachments/assets/29cdfabe-9471-4ec0-859e-8026c94c6c23" />


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
