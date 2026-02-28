# VN-Quant Colab Training Guide

## Architecture
```
┌────────────────────┐         ┌────────────────────┐
│   Google Colab     │         │   Host i7-14700    │
│   (GPU Training)   │  SYNC   │   (Inference)      │
│                    │ ◄─────► │                    │
│  • Train 102 models│         │  • Paper trading   │
│  • A100/V100/T4    │         │  • API server      │
│  • 15-45 hours     │         │  • Real-time       │
└────────────────────┘         └────────────────────┘
```

## Quick Start

### Step 1: Upload Data to Google Drive
From host server, copy `data/historical/*.parquet` to Google Drive:
```
Google Drive/
└── VNQuant/
    └── data/
        ├── ACB.parquet
        ├── FPT.parquet
        └── ... (289 files)
```

### Step 2: Run Colab Notebook
1. Open `VNQuant_Stockformer_Training.ipynb` in Colab
2. Runtime > Change runtime type > **GPU** (A100 recommended)
3. Run all cells
4. Wait 15-45 hours depending on GPU

### Step 3: Download Models to Host
After training completes, copy from Google Drive to host:
```
Google Drive/VNQuant/models/*.pt → D:\testpapertr\models\
```

## Estimated Training Time

| GPU | Time (102 stocks) | Cost |
|-----|-------------------|------|
| T4  | 35-45 hours | $10/mo (Pro) |
| V100 | 25-35 hours | $10/mo (Pro) |
| A100 | 15-20 hours | $50/mo (Pro+) |

## Files

- `VNQuant_Stockformer_Training.ipynb` - Main training notebook
- `../config/dual_machine_config.json` - Configuration
- `../scripts/local_cpu_training.py` - i7 quick training (3-5 stocks)
