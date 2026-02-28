# Weekly Model Training Guide

## Overview

VN-Quant uses **Stockformer deep learning models** trained on historical Vietnamese stock data to predict price movements. Models are retrained weekly to incorporate new market data and improve prediction accuracy.

This guide explains:
- How to set up weekly training
- How to monitor training progress
- How to validate trained models
- How to deploy new models to production
- How to troubleshoot training issues

---

## Training Architecture

### Current Setup

```
Training Pipeline:
├── Data Collection (daily via CafeF API)
├── Feature Calculation (15 technical indicators)
├── Data Preparation (train/validation splits)
├── Model Training (Stockformer for 102 stocks)
├── Validation & Testing
├── Model Deployment
└── Performance Monitoring
```

### Models Trained

- **102 Vietnamese stocks** from HOSE, HNX, UPCOM exchanges
- **Stockformer architecture** (attention-based time series forecasting)
- **5-day ahead predictions** (using 60-day historical window)
- **Trained weekly** on Sundays at 2:00 AM (after market week closes)

---

## Quick Start: Manual Training

### 1. Prepare Environment

```bash
# SSH into Docker container (if running in production)
docker exec -it vnquant-autonomous bash

# Or run locally
cd D:\testpapertr

# Activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/macOS
```

### 2. Run Training

```bash
# Full training (all 102 stocks) - ~30-60 minutes
python -m quantum_stock.ml.training_pipeline

# Training with specific symbols
python -m quantum_stock.ml.training_pipeline --symbols "HPG,VCB,FPT,MWG"

# Resume interrupted training
python -m quantum_stock.ml.training_pipeline --resume
```

### 3. Monitor Progress

Training output shows:
```
[2025-01-12 02:15:30] Training Stockformer for FPT
Epoch 1/100: Loss=0.0245, Val Loss=0.0187
Epoch 2/100: Loss=0.0198, Val Loss=0.0165
...
[2025-01-12 03:45:22] ✅ FPT: Accuracy=56.2%, Saved to models/FPT_stockformer_simple_best.pt
```

### 4. Validate Models

```bash
# Test model predictions on recent data
python -c "
from quantum_stock.ml.training_pipeline import validate_model
validate_model('FPT', show_predictions=True)
"
```

---

## Automated Weekly Training

### Option 1: Docker Cron Service (Recommended)

A dedicated training service runs in Docker on schedule:

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
      - DATABASE_URL=postgresql://...
    volumes:
      - ./models:/app/models
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - vn-quant-network
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

### Option 2: System Cron Job (Linux/macOS)

```bash
# Edit crontab
crontab -e

# Add this line (runs Sunday 2:00 AM)
0 2 * * 0 cd /opt/vnquant && python -m quantum_stock.ml.training_pipeline >> /var/log/vnquant-training.log 2>&1
```

### Option 3: Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task: "VN-Quant Weekly Training"
3. Trigger: Weekly, Sunday, 2:00 AM
4. Action: `python D:\testpapertr\quantum_stock\ml\training_pipeline.py`
5. Set working directory: `D:\testpapertr`

---

## Training Configuration

### Model Parameters

Edit `quantum_stock/ml/training_pipeline.py`:

```python
# Training configuration
TRAINING_CONFIG = {
    'epochs': 100,                    # Number of training epochs
    'batch_size': 32,                 # Batch size for training
    'd_model': 64,                    # Transformer dimension
    'n_heads': 4,                     # Number of attention heads
    'n_layers': 2,                    # Number of transformer layers
    'dropout': 0.5,                   # Dropout rate
    'learning_rate': 0.001,           # Initial learning rate
    'val_split': 0.2,                 # Validation split ratio
    'patience': 15,                   # Early stopping patience
}

# Data configuration
DATA_CONFIG = {
    'window_size': 60,                # Input sequence length (days)
    'horizon': 5,                     # Prediction horizon (days)
    'min_data_points': 100,           # Minimum data points required
    'normalization': 'standard',      # 'standard' or 'minmax'
}

# Stock selection
STOCKS_TO_TRAIN = [
    # Core stocks (always train)
    'HPG', 'VCB', 'FPT', 'MWG',      # Top 4 by liquidity
    'SAB', 'MSN', 'VNM', 'VJC',      # Large cap banks/stocks
    # Add more stocks as needed
]
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

## Model Validation & Testing

### Validation Metrics

After training, evaluate models on:

```python
# In validation output:
Accuracy:         56.2%    # Directional accuracy (UP/DOWN correct)
RMSE:            0.0187    # Root mean squared error
MAE:             0.0142    # Mean absolute error
Sharpe Ratio:     1.45     # Risk-adjusted returns
Max Drawdown:    -8.2%     # Maximum peak-to-trough decline
```

### Acceptance Criteria

Models are accepted for production if:
- ✅ **Accuracy > 52%** (better than random 50%)
- ✅ **RMSE < 0.03** (reasonable prediction error)
- ✅ **Sharpe Ratio > 0.8** (decent risk-return)
- ⚠️ Stocks with **< 52% accuracy** are flagged for review

### Backtesting New Models

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

---

## Model Deployment

### Automatic Deployment

Once training completes successfully:

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

---

## Monitoring Training Progress

### Real-time Dashboard

```bash
# Watch training in progress
docker logs vnquant-trainer -f --tail=100

# Output example:
[2025-01-12 02:15:30] Starting weekly training cycle
[2025-01-12 02:16:15] Loading data for 102 stocks...
[2025-01-12 02:18:45] ✅ Data loaded: 3,240 features
[2025-01-12 02:19:00] Training FPT (1/102)
Epoch 1/100: Loss=0.0245, Val Loss=0.0187 [████░░░░░░] 10%
...
[2025-01-12 03:45:22] ✅ FPT: Accuracy=56.2%, Saved
[2025-01-12 03:46:10] Training VCB (2/102)
...
[2025-01-12 14:30:00] ✅ Training complete: 98/102 successful
[2025-01-12 14:31:00] Deploying models...
[2025-01-12 14:31:30] ✅ Deployment complete
```

### Training History

```bash
# View past training results
cat models/stockformer/training_history.json | python -m json.tool

# Output:
{
  "2025-01-12": {
    "stocks_trained": 102,
    "avg_accuracy": 54.8,
    "failed_stocks": [],
    "duration_hours": 12.5,
    "model_sizes_mb": {...}
  },
  "2025-01-05": {...}
}
```

---

## Troubleshooting

### Issue: Training Hangs/Crashes

**Solution:**
```bash
# 1. Check logs
docker logs vnquant-trainer -f

# 2. Look for OOM (Out of Memory)
# If found: reduce batch_size in config from 32 to 16

# 3. Check disk space
docker exec vnquant-trainer df -h

# 4. Restart training
docker restart vnquant-trainer
```

### Issue: Accuracy Decreases

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

### Issue: Out of Memory During Training

**Solution:**
```bash
# Reduce model size in docker-compose.yml:
environment:
  - BATCH_SIZE=16        # Reduce from 32
  - N_LAYERS=1           # Reduce from 2
  - D_MODEL=32           # Reduce from 64
```

### Issue: Models Not Loading in Scanner

**Solution:**
```bash
# 1. Check model files exist
ls -la models/*_stockformer_simple_best.pt | wc -l

# 2. Verify model format
python -c "
import torch
model = torch.load('models/FPT_stockformer_simple_best.pt', map_location='cpu')
print(f'Model keys: {list(model.keys())}')
"

# 3. Restart scanner
docker restart vnquant-autonomous
```

---

## Performance Benchmarks

### Expected Training Duration

- **Single stock**: 5-10 minutes (depends on data size)
- **50 stocks**: 4-8 hours
- **102 stocks (full)**: 10-15 hours
- **Total with validation**: 12-18 hours

### Resource Usage

```
CPU:     40-60% (multi-core)
Memory:  2-4 GB
Disk:    200-500 MB (new models)
Network: Minimal (data already cached)
```

### Model Accuracy Trends (Historical)

```
Week 1:  52.1% accuracy
Week 2:  53.8% accuracy
Week 3:  54.2% accuracy
Week 4:  54.8% accuracy  <- Current week target
Week 5:  55.1% accuracy
```

Target: **55%+ accuracy** on directional predictions

---

## Advanced Topics

### Custom Training Features

```python
# Use different feature sets
features = [
    'close_returns',     # Basic returns
    'rsi_14',            # Momentum
    'macd',              # Trend
    'bb_position',       # Volatility
    'volume_ma_ratio',   # Volume confirmation
]

# Use different architectures
models = ['stockformer', 'lstm', 'gru', 'transformer']

# Ensemble multiple models
ensemble = EnsemblePredictor(
    models=['stockformer', 'lstm'],
    weights=[0.6, 0.4]  # Weight newer Stockformer higher
)
```

### Distributed Training

For large numbers of stocks, run training in parallel:

```bash
# Use multiple worker processes
export TRAINING_WORKERS=4  # Use 4 CPU cores

python -m quantum_stock.ml.training_pipeline --parallel
```

### Transfer Learning

Reuse weights from strong performers:

```python
# Train new stock using weights from similar stock
from quantum_stock.ml.transfer_learning import TransferTrainer

trainer = TransferTrainer()
trainer.train_from_similar(
    new_symbol='ABC',
    reference_symbol='HPG',  # Similar stock to learn from
    epochs=50  # Fewer epochs, faster training
)
```

---

## Integration with Autonomous Trading

### How Models Are Used

```
1. Model Prediction Scanner (runs every 3 minutes)
   ↓
2. Load latest trained models
   ↓
3. Predict next 5 days for all 102 stocks
   ↓
4. Filter: expected return > 3%, confidence > 0.7
   ↓
5. Trigger Scout signal
   ↓
6. Agents analyze and vote
   ↓
7. Execute trades if consensus reached
```

### Model Performance Impact

- **Better models** → Better predictions → More profitable trades
- **Weekly retraining** → Adapts to market changes
- **Ensemble models** → More robust predictions
- **Transfer learning** → Faster training on new stocks

---

## Maintenance Schedule

| Task | Frequency | Time | Owner |
|------|-----------|------|-------|
| Weekly training | Every Sunday | 2 AM | Automated |
| Validation | Weekly | 3 PM | Automated |
| Accuracy review | Weekly | Daily reports | Dashboard |
| Model backup | Every training | Auto | System |
| Historical analysis | Monthly | 1st Sunday | Analytics |
| Parameter tuning | Quarterly | As needed | Engineer |
| Architecture review | Quarterly | Review meetings | Team |

---

## References

- **Training Pipeline**: `quantum_stock/ml/training_pipeline.py`
- **Training Scheduler**: `train_scheduler.py` (created for Docker)
- **Model Architecture**: `quantum_stock/models/stockformer.py`
- **Backtest Module**: `quantum_stock/ml/backtest_models.py`
- **Configuration**: `.env` file (TRAINING_* variables)

---

## Support & Questions

For issues or questions about training:

1. Check logs: `docker logs vnquant-trainer`
2. Review this guide's troubleshooting section
3. Check Discord/Telegram community (if available)
4. Create GitHub issue with logs and error messages

**Last Updated**: 2025-01-12
**Version**: 1.0
