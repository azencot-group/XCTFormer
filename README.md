# XCTFormer

Official repository for [XCTFormer: Cross-Variate-Temporal Transformer with Learnable Causality Masks for Multivariate Time Series Forecasting](https://openreview.net/forum?id=TEfyR4t0Tw).

## Setup

```bash
conda create -n xctformer python=3.9.21
conda activate xctformer
pip install -r requirements.txt
```

## Data

Download datasets and place them in the `datasets/` directory.

## Generating the Synthetic Dataset

```bash
python generate_synthetic_data.py
```

Output: `datasets/Synthetic/synthetic.csv`

## Reproducing Results

```bash
# Long-term forecasting
bash scripts/long_term_forecast/XCTFormer_ETTm1.sh

# Imputation
bash scripts/imputation/XCTFormer_ETTm1.sh

# Anomaly detection
bash scripts/anomaly_detection/XCTFormer_PSM.sh
```

