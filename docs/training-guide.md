# VN-Quant Model Training Guide

**Training 102 Stockformer Models for Autonomous Trading**

**Version:** 1.0
**Last Updated:** 2026-02-28
**Scope:** Complete training setup from local CPU to Google Colab GPU

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [Google Colab Setup](#google-colab-setup)
4. [Local CPU Training](#local-cpu-training)
5. [Hybrid Workflow](#hybrid-workflow)
6. [Training Configuration](#training-configuration)
7. [Validation & Deployment](#validation--deployment)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Training Strategy Matrix

| Task | Primary | Secondary | Rationale |
|------|---------|-----------|-----------|
| **Full retraining** (102 stocks) | Colab A100 | Local CPU | Heavy computation (~12-15h on A100) |
| **Weekly incremental updates** | Colab A100 | Local CPU | New weekly data, benefit from GPU |
| **Quick inference tests** | Local CPU | N/A | Fast <1min per model, no training |
| **Data preprocessing** | Local CPU | Colab | Lightweight, can handle locally |
| **Feature engineering** | Local CPU | Colab | CPU-only task, no GPU needed |
| **Model validation/backtest** | Local CPU | Colab | Can run overnight on local |

### Weekly Training Timeline

```
Week Structure:
└── Sunday 2:00 AM (UTC+7)
    ├── [Local] Data collection from CafeF (OHLCV for 289 stocks)
    ├── [Local] Feature engineering (60-day windows → 50 features per day)
    ├── [Colab] Sync training data to Google Drive
    ├── [Colab] Train Stockformer for 102 stocks (batch training)
    ├── [Colab] Save checkpoints to Google Drive (every 2 hours)
    ├── [Local] Download trained models from Drive
    ├── [Local] Run validation backtest
    └── [Production] Deploy validated models by Monday 9:15 AM
```

---

## Quick Start

### Manual Training (5 minutes)

```bash
# 1. Prepare environment
cd D:\testpapertr
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/macOS

# 2. Run training (all 102 stocks) - ~30-60 minutes
python -m quantum_stock.ml.training_pipeline

# Or train specific symbols
python -m quantum_stock.ml.training_pipeline --symbols "HPG,VCB,FPT,MWG"

# Or resume interrupted training
python -m quantum_stock.ml.training_pipeline --resume

# 3. Monitor progress
# Output shows:
# [2025-01-12 02:15:30] Training Stockformer for FPT
# Epoch 1/100: Loss=0.0245, Val Loss=0.0187
# ...
# [2025-01-12 03:45:22] ✅ FPT: Accuracy=56.2%, Saved to models/FPT_stockformer_simple_best.pt

# 4. Validate models
python -c "
from quantum_stock.ml.training_pipeline import validate_model
validate_model('FPT', show_predictions=True)
"
```

---

## Google Colab Setup

### GPU Tier Comparison

| Feature | Free | Pro ($9.99/mo) | Pro+ ($49.99/mo) |
|---------|------|---|---|
| GPU Type | T4 (mostly) | V100, T4 | A100, V100, A100-80GB |
| GPU Memory | 16GB | 16GB | 40-80GB |
| CPU Cores | 2 | 4 | 8 |
| RAM | 12GB | 32GB | 52GB |
| A100 Priority | None | Rare | High |
| Idle Timeout | 30 min | 90 min | 90 min |

**Recommendation:** Pro+ for production training (A100 is 13x faster than T4). Pro acceptable for development.

### Colab Session Management

#### Detect GPU at Runtime

```python
import torch

def check_gpu_setup():
    """Detect available GPU and configure training"""

    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected. Falling back to CPU training.")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return {
        'device': device,
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'cuda_available': torch.cuda.is_available(),
    }

device_info = check_gpu_setup()
device = device_info['device']
```

#### Prevent Colab Timeout (12-hour limit)

```python
from IPython import display
import time

def prevent_colab_disconnect():
    """Keep Colab session alive during training (run in separate cell)"""
    while True:
        try:
            display.clear_output(wait=True)
            print(f"Session alive at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            time.sleep(300)  # Check every 5 minutes
        except KeyboardInterrupt:
            print("Session monitor stopped")
            break

# Run in background (in separate code cell)
# prevent_colab_disconnect()
```

#### Google Drive Integration

```python
from google.colab import drive
import os

def setup_google_drive():
    """Mount Google Drive and create necessary directories"""

    # Mount Drive
    drive.mount('/content/drive', force_remount=True)

    # Create directory structure
    base_dir = '/content/drive/My Drive/VN-Quant-Training'
    dirs = {
        'data': f'{base_dir}/data',
        'models': f'{base_dir}/models',
        'checkpoints': f'{base_dir}/checkpoints',
        'logs': f'{base_dir}/logs',
        'configs': f'{base_dir}/configs'
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Directory ready: {dir_path}")

    return dirs

dirs = setup_google_drive()
```

#### Checkpoint Management

```python
import torch
import time
from datetime import datetime

class CheckpointManager:
    """Manage training checkpoints to Google Drive"""

    def __init__(self, checkpoint_dir: str, interval_minutes: int = 20):
        self.checkpoint_dir = checkpoint_dir
        self.interval_minutes = interval_minutes
        self.last_checkpoint = time.time()
        self.checkpoint_count = 0

    def should_checkpoint(self) -> bool:
        """Check if enough time passed since last checkpoint"""
        return (time.time() - self.last_checkpoint) > (self.interval_minutes * 60)

    def save_checkpoint(self, model, optimizer, epoch, symbol, metrics):
        """Save training checkpoint"""

        if not self.should_checkpoint():
            return False

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_epoch{epoch}_{timestamp}.pt'
        filepath = f'{self.checkpoint_dir}/{filename}'

        torch.save(checkpoint, filepath)
        self.last_checkpoint = time.time()
        self.checkpoint_count += 1

        print(f"✅ Checkpoint {self.checkpoint_count}: {filepath}")
        print(f"   Epoch: {epoch}, Loss: {metrics.get('val_loss', 'N/A'):.4f}")

        return True

# Usage in training loop
checkpoint_mgr = CheckpointManager('/content/drive/My Drive/VN-Quant-Training/checkpoints')

for epoch in range(100):
    # ... training code ...
    checkpoint_mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        symbol='FPT',
        metrics={'train_loss': train_loss, 'val_loss': val_loss}
    )
```

#### Session Recovery (Auto-Resume)

```python
def setup_session_recovery(checkpoint_dir: str):
    """Auto-resume training if session disconnects"""

    import json

    state_file = f'{checkpoint_dir}/training_state.json'

    def save_state(symbol: str, epoch: int, stocks_completed: list):
        """Save training progress to JSON"""
        state = {
            'current_symbol': symbol,
            'current_epoch': epoch,
            'stocks_completed': stocks_completed,
            'last_update': datetime.now().isoformat()
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state():
        """Restore training from last known state"""
        if os.path.exists(state_file):
            with open(state_file) as f:
                return json.load(f)
        return None

    return save_state, load_state

# Resume if interrupted
save_state, load_state = setup_session_recovery(dirs['checkpoints'])
state = load_state()

if state:
    print(f"Resuming from {state['current_symbol']} epoch {state['current_epoch']}")
    start_symbol_idx = STOCK_SYMBOLS.index(state['current_symbol'])
else:
    start_symbol_idx = 0
```

### Complete Colab Notebook Template

```python
# ============================================================
# VN-Quant Stockformer Training - Colab Pro/Pro+
# ============================================================

# 1. SETUP
!pip install -q torch torchvision torchaudio

# 2. MOUNT DRIVE
from google.colab import drive
drive.mount('/content/drive')

# 3. CLONE REPO
!git clone https://github.com/username/vnquant.git /content/vnquant
import sys
sys.path.insert(0, '/content/vnquant')

# 4. SETUP TRAINING
from quantum_stock.ml.training_pipeline import train_stockformer
from quantum_stock.models.stockformer import StockformerEnsemble

# 5. LOAD DATA FROM DRIVE
import shutil
shutil.copy('/content/drive/My Drive/VN-Quant-Training/data/training_data.pkl',
           '/content/training_data.pkl')

# 6. TRAIN MODELS
training_config = {
    'symbols': 'ALL',  # or ['FPT', 'VCB', 'HPG'] for subset
    'epochs': 100,
    'batch_size': 32,
    'num_workers': 4,
    'checkpoint_interval': 20,  # minutes
    'checkpoint_dir': '/content/drive/My Drive/VN-Quant-Training/checkpoints'
}

results = train_stockformer(**training_config)

# 7. SAVE TO DRIVE
!rclone copy /content/models gdrive:VN-Quant-Training/models --progress

# 8. SUMMARY
print(f"✅ Training complete: {len(results)} models trained")
print(f"Average accuracy: {sum([r['accuracy'] for r in results.values()]) / len(results):.2%}")
```

---

## Local CPU Training

### CPU Specifications & Optimization

**Hardware:** Intel i7-14700
- P-cores: 8 (3.4-5.6 GHz)
- E-cores: 12 (2.5-4.2 GHz)
- L3 Cache: 33MB
- TDP: 125W

#### CPU Optimization

```python
import os
import torch

def optimize_cpu_training():
    """Configure CPU for maximum PyTorch training performance"""

    config = {
        'num_threads': min(16, os.cpu_count()),
        'num_interop_threads': 2,
        'enable_mkldnn': True,
        'cache_aligned': True,
        'use_numa': os.cpu_count() > 16,
        'num_workers': 4,
        'pin_memory': False,
        'prefetch_factor': 2,
        'batch_size': 8,
        'gradient_accumulation': 4,
        'mixed_precision': False,
    }

    torch.set_num_threads(config['num_threads'])
    torch.set_num_interop_threads(config['num_interop_threads'])

    print("✅ CPU Optimization Applied:")
    print(f"   Threads: {torch.get_num_threads()}")
    print(f"   Workers: {config['num_workers']}")
    print(f"   Batch size: {config['batch_size']}")

    return config

cpu_config = optimize_cpu_training()
```

### CPU-Suitable Tasks

**Fast Tasks (CPU-OK):**
- Data loading/caching (<1s)
- Feature engineering (2-5s)
- Data normalization (1s)
- Model inference single (<100ms)
- Validation backtest (5-10m)
- Model comparison (1-2m)

**Slow Tasks (Use GPU):**
- Training 1 Stockformer: 2-4 hours (CPU) vs 15-30 min (GPU) → 8-16x slower
- Batch training (102 stocks): ~200 hours (CPU) vs 15 hours (GPU) → Days vs 1 day
- Hyperparameter tuning: >7 days (CPU) vs 12 hours (GPU)

### CPU Training Recipe

```python
def cpu_training_recipe():
    """Minimal training suitable for CPU"""

    config = {
        'models': ['FPT', 'VCB', 'HPG'],  # Top 3 only
        'epochs': 50,  # Half of normal (100)
        'batch_size': 4,  # Very small
        'lookback': 30,  # Shorter sequences
        'gradient_accumulation': 8,
        'checkpoint_interval': 100,
        'num_workers': 2
    }

    # Expected timeline:
    # - Single model: 30-45 min
    # - 3 models: 2-3 hours
    # - Limited value for production (reduced accuracy)

    print("⚠️  CPU Training Profile:")
    print(f"   Models: {config['models']}")
    print(f"   Epochs: {config['epochs']} (reduced from 100)")
    print(f"   Batch: {config['batch_size']} (reduced from 32)")
    print(f"   Est. Time: 2-3 hours for {len(config['models'])} models")
    print("   ℹ️  Best used for: validation, fine-tuning, emergency fallback")

    return config
```

### Memory Monitoring

```python
def monitor_cpu_memory(symbol: str = ''):
    """Monitor CPU memory during training"""

    import psutil

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_mb = memory_info.rss / 1024 / 1024

    system_mem = psutil.virtual_memory()
    available_gb = system_mem.available / 1024 / 1024 / 1024
    used_pct = system_mem.percent

    print(f"Memory (CPU Training {symbol}):")
    print(f"  Process: {rss_mb:.1f} MB")
    print(f"  System:  {used_pct:.1f}% used ({available_gb:.1f} GB free)")

    if used_pct > 85:
        print("  ⚠️  WARNING: System memory >85%, may cause slowdown")

    return {
        'process_mb': rss_mb,
        'system_percent': used_pct,
        'available_gb': available_gb
    }

# Call regularly during training
for epoch in range(epochs):
    # ... training ...
    if epoch % 10 == 0:
        monitor_cpu_memory(symbol='FPT')
```

### CPU Inference Optimization

```python
def optimize_cpu_inference(model: torch.nn.Module):
    """Optimize model for CPU inference"""

    model = torch.jit.script(model)
    model.eval()

    # Quantization (int8) for 2-4x speedup
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # Benchmark
    with torch.no_grad():
        for _ in range(10):
            dummy_input = torch.randn(1, 60, 50)
            _ = quantized_model(dummy_input)

        import time
        start = time.perf_counter()
        for _ in range(100):
            _ = quantized_model(dummy_input)
        elapsed = (time.perf_counter() - start) * 1000 / 100

        print(f"✅ Optimized Model Inference: {elapsed:.1f}ms per forward pass")

    return quantized_model
```

---

## Hybrid Workflow

### Data Sync with Rclone

```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive (one-time)
rclone config

# Test connection
rclone lsd gdrive:

# Sync commands
rclone sync ~/vnquant/models gdrive:VN-Quant-Training/models --progress
rclone copy gdrive:VN-Quant-Training/models ~/vnquant/models --progress
rclone bisync ~/vnquant/models gdrive:VN-Quant-Training/models --progress
```

### Rclone Sync Manager (Python)

```python
import subprocess
import os
from pathlib import Path
from datetime import datetime

class RcloneSyncManager:
    """Manage bidirectional sync between local and Google Drive"""

    def __init__(self, local_path: str, remote_path: str, remote_name: str = 'gdrive'):
        self.local_path = Path(local_path)
        self.remote_path = remote_path
        self.remote_name = remote_name

    def upload_to_drive(self, verbose: bool = True) -> bool:
        """Upload local files to Google Drive"""

        cmd = [
            'rclone', 'copy',
            str(self.local_path),
            f'{self.remote_name}:{self.remote_path}',
            '--progress' if verbose else '--no-progress',
            '--transfers', '4',
            '--fast-list',
        ]

        print(f"Uploading {self.local_path} → {self.remote_path}...")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            print(f"✅ Upload complete at {datetime.now().isoformat()}")
            return True
        else:
            print(f"❌ Upload failed with code {result.returncode}")
            return False

    def download_from_drive(self, verbose: bool = True) -> bool:
        """Download files from Google Drive to local"""

        cmd = [
            'rclone', 'copy',
            f'{self.remote_name}:{self.remote_path}',
            str(self.local_path),
            '--progress' if verbose else '--no-progress',
            '--transfers', '4',
            '--fast-list',
        ]

        print(f"Downloading {self.remote_path} → {self.local_path}...")
        result = subprocess.run(cmd, capture_output=False)

        if result.returncode == 0:
            print(f"✅ Download complete at {datetime.now().isoformat()}")
            return True
        else:
            print(f"❌ Download failed with code {result.returncode}")
            return False

# Usage
sync_mgr = RcloneSyncManager(
    local_path='/home/user/vnquant/models',
    remote_path='VN-Quant-Training/models'
)

# After Colab training, download models
sync_mgr.download_from_drive()

# Check sync status
status = sync_mgr.get_file_count()
print(f"Local files: {status['local']}, Remote files: {status['remote']}")
```

---

## Training Configuration

### Stockformer Architecture

```python
STOCKFORMER_CONFIG = {
    'input_size': 50,              # Feature dimensions
    'sequence_length': 60,         # Lookback days
    'forecast_horizon': 5,         # Predict 5 days ahead
    'd_model': 128,                # Transformer dim
    'n_heads': 8,                  # Attention heads
    'n_layers': 4,                 # Encoder layers
    'd_ff': 512,                   # Feedforward dim
    'dropout': 0.1,
    'n_models': 3,                 # Ensemble size
}

TRAINING_DATA = {
    'num_stocks': 102,
    'total_daily_points': 3240,
    'train_samples': ~180000,
    'validation_split': 0.2,
}
```

### Hyperparameters

#### Colab (GPU) - Production

```python
COLAB_TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'optimizer': 'Adam',
    'scheduler': 'CosineAnnealingLR',
    'warmup_epochs': 5,
    'early_stopping_patience': 15,
    'gradient_clip': 1.0,
    'num_workers': 4,
    'pin_memory': True,
    'prefetch_factor': 2,
    'checkpoint_interval': 20,  # minutes
    'save_last_model': True,
    'keep_top_k': 3,
    'time_per_model': '15-20 min',
    'time_all_102': '25-30 hours',
}

ACCURACY_TARGETS = {
    'minimum': 0.52,
    'acceptable': 0.54,
    'good': 0.56,
    'excellent': 0.58,
}
```

#### Local (CPU) - Validation/Development

```python
LOCAL_TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 4,
    'learning_rate': 0.001,
    'gradient_accumulation': 8,
    'num_workers': 2,
    'pin_memory': False,
    'checkpoint_interval': 100,
    'symbols_to_train': ['FPT', 'VCB', 'HPG'],
    'time_per_model': '30-45 min',
    'time_3_models': '2-3 hours',
    'purpose': 'Validation, fine-tuning, emergency fallback',
}
```

### Data Requirements

For each stock to train, need minimum:
- **100+ trading days** of historical data
- **5 OHLCV columns**: open, high, low, close, volume
- **Data quality**: No gaps >5 trading days
- **Recent data**: Latest data within 1 week

Check data availability:

```python
from quantum_stock.dataconnector.data_loader import DataLoader

loader = DataLoader()
df = loader.load_stock_data('FPT', days=365)

print(f"Data points: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing values: {df.isnull().sum()}")
```

---

## Validation & Deployment

### Model Validation Metrics

After training, evaluate models on:

```python
Accuracy:         56.2%    # Directional accuracy
RMSE:            0.0187    # Root mean squared error
MAE:             0.0142    # Mean absolute error
Sharpe Ratio:     1.45     # Risk-adjusted returns
Max Drawdown:    -8.2%     # Maximum decline
```

### Acceptance Criteria

Models are accepted for production if:
- ✅ **Accuracy > 52%** (better than random)
- ✅ **RMSE < 0.03** (reasonable error)
- ✅ **Sharpe Ratio > 0.8** (decent risk-return)
- ⚠️ Stocks with **< 52% accuracy** flagged for review

### Backtesting

```bash
# Test trained models on historical data (past year)
python -c "
from quantum_stock.ml.backtest_models import backtest_all_models
results = backtest_all_models(
    start_date='2024-01-12',
    end_date='2025-01-12',
    position_size=1000,
    slippage=0.002
)
print(results.summary())
"
```

### Automatic Deployment

Once training completes:

```python
# Models saved to: models/{SYMBOL}_stockformer_simple_best.pt
# Each model:
# - ~2MB file size
# - Includes model weights and architecture
# - Can be loaded immediately by scanner

# Scanner automatically uses latest models
# (checks modification time, reloads on startup)
```

### Manual Deployment

If training completed but models not auto-deployed:

```bash
# 1. Verify model quality
python docs/scripts/validate_models.py

# 2. Backup current models (optional)
cp -r models models.backup.$(date +%Y%m%d)

# 3. Deploy new models
cp models/FPT_stockformer_simple_best.pt models/FPT_stockformer_simple_best.pt.prod

# 4. Restart scanning service
docker restart vnquant-autonomous

# 5. Verify new predictions in logs
docker logs vnquant-autonomous | grep "Model:"
```

### Rollback Procedure

If new models perform poorly:

```bash
# 1. Restore from backup
rm -r models
cp -r models.backup.latest models

# 2. Restart service
docker restart vnquant-autonomous

# 3. Investigate issue
docker logs vnquant-autonomous
```

### Automated Weekly Training

#### Option 1: Docker Cron Service (Recommended)

```yaml
# docker-compose.yml includes:
services:
  model-trainer:
    build:
      context: .
      dockerfile: Dockerfile.training
    container_name: vnquant-trainer
    command: python train_scheduler.py
    environment:
      - TRAINING_SCHEDULE=0 2 * * 0  # 2 AM Sunday
      - ENVIRONMENT=production
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
```

**Start training service:**
```bash
docker-compose up -d model-trainer
```

**Check training status:**
```bash
docker logs vnquant-trainer -f
```

#### Option 2: System Cron Job (Linux/macOS)

```bash
# Edit crontab
crontab -e

# Add this line (runs Sunday 2:00 AM)
0 2 * * 0 cd /opt/vnquant && python -m quantum_stock.ml.training_pipeline >> /var/log/vnquant-training.log 2>&1
```

#### Option 3: Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task: "VN-Quant Weekly Training"
3. Trigger: Weekly, Sunday, 2:00 AM
4. Action: `python D:\testpapertr\quantum_stock\ml\training_pipeline.py`
5. Set working directory: `D:\testpapertr`

---

## Troubleshooting

### Colab: Training Hangs/Crashes

**Solution:**
```bash
# 1. Check logs for errors
# 2. Look for OOM (Out of Memory) - reduce batch_size from 32 to 16
# 3. Check disk space: df -h
# 4. Restart training with resume capability
```

### Colab: Session Disconnects

**Solution:** Implement auto-resume training (see Checkpoint Manager above)

### Colab: Out of Memory

**Solution:**
```python
'batch_size': 16,  # from 32
'gradient_accumulation': 2,  # from 4
```

### Models: Accuracy Decreases

**Solution:**
```bash
# 1. Check data quality
python docs/scripts/check_data_quality.py

# 2. Analyze new market regime
# Market may have changed - retune parameters

# 3. Check for data gaps
# May need to reload historical data

# 4. Fallback to previous models
# Use models.backup if accuracy drops >10%
```

### CPU Training: Too Slow

**Solution:** Use Colab for full training, CPU only for validation

### Sync: Models Not Syncing

**Solution:**
```bash
rclone config show  # List remotes
rclone lsd gdrive:VN-Quant-Training/models  # Test access
!rclone copy /content/models gdrive:VN-Quant-Training/models --progress
```

---

## Performance Benchmarks

### Training Speed

| Configuration | Time/Model | Total 102 | Cost |
|---|---|---|---|
| Colab A100 (ideal) | 15-20 min | 26-34 hours | $50/mo |
| Colab V100 | 25-35 min | 42-60 hours | $10/mo |
| Colab T4 | 45-60 min | 76-102 hours | Free |
| Local i7-14700 CPU | 30-45 min | 51-77 hours | $0 |

### Resource Usage

| System | Peak RAM | VRAM | Notes |
|---|---|---|---|
| Colab A100 | 52GB | 40GB | No memory pressure |
| Colab T4 | 12GB | 16GB | Batch size limited to 16 |
| i7-14700 | 32GB system | N/A | Batch size 4, ~3GB per model |

### Expected Training Duration

- **Single stock**: 5-10 minutes
- **50 stocks**: 4-8 hours
- **102 stocks (full)**: 10-15 hours
- **Total with validation**: 12-18 hours

---

## Maintenance Schedule

| Task | Frequency | Time | Owner |
|------|-----------|------|-------|
| Full retraining | Weekly (Sunday 2 AM) | 26-30h | Automated (Colab) |
| Data sync test | Daily | 5 min | Automated script |
| Model validation | After training | 30 min | Automated |
| CPU validation training | Monthly | 2-4h | Manual (optional) |
| Checkpoint cleanup | Weekly | 10 min | Automated |

---

## References

- `quantum_stock/models/stockformer.py` - Model architecture
- `quantum_stock/ml/training_pipeline.py` - Current training pipeline
- `scripts/colab_training_setup.py` - Colab initialization
- `scripts/local_cpu_training.py` - CPU training optimization
- `scripts/hybrid_training_orchestrator.py` - Workflow automation

---

**Last Updated:** 2026-02-28
**Status:** Production-Ready
**Maintainer:** VN-Quant Development Team
