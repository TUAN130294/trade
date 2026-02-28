#!/usr/bin/env python3
"""
VN-Quant Stockformer Colab Training Setup
==========================================
Initialize Google Colab environment for distributed training.

Usage in Colab:
    # Cell 1: Installation
    !pip install -q torch torchvision torchaudio rclone

    # Cell 2: Setup
    %run '/content/drive/My Drive/VN-Quant-Training/colab_training_setup.py'

    # Cell 3: Start training
    setup = ColaTrainingSetup()
    setup.start_training(all_symbols=True)
"""

import os
import sys
import json
import torch
import pickle
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class ColabEnvironmentSetup:
    """Initialize Colab environment for training"""

    def __init__(self):
        self.device = None
        self.drive_mounted = False
        self.dirs = {}

    def setup_gpu(self) -> Dict[str, any]:
        """Detect and configure GPU"""

        if not torch.cuda.is_available():
            logger.warning("No GPU detected. CPU fallback enabled.")
            self.device = torch.device('cpu')
            return {'available': False, 'device': 'cpu'}

        self.device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

        logger.info(f"✅ GPU Detected: {gpu_name} ({gpu_memory:.1f}GB)")

        # Configure optimization
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        return {
            'available': True,
            'device': gpu_name,
            'memory_gb': gpu_memory,
            'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version()
        }

    def mount_google_drive(self) -> bool:
        """Mount Google Drive for checkpoint/model storage"""

        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            self.drive_mounted = True
            logger.info("✅ Google Drive mounted")
            return True
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            return False

    def create_directory_structure(self, base_dir: str = '/content/drive/My Drive/VN-Quant-Training') -> Dict[str, str]:
        """Create training directory structure"""

        self.dirs = {
            'base': base_dir,
            'data': f'{base_dir}/data',
            'models': f'{base_dir}/models',
            'checkpoints': f'{base_dir}/checkpoints',
            'logs': f'{base_dir}/logs',
            'configs': f'{base_dir}/configs',
        }

        for dir_name, dir_path in self.dirs.items():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"✅ Directory ready: {dir_name} → {dir_path}")

        return self.dirs

    def verify_colab_environment(self) -> Dict:
        """Full environment verification"""

        logger.info("=" * 60)
        logger.info("Verifying Colab Environment")
        logger.info("=" * 60)

        # GPU
        gpu_info = self.setup_gpu()
        logger.info(f"GPU: {gpu_info}")

        # Memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            logger.info(f"VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

        # Disk
        import shutil
        disk_stat = shutil.disk_usage('/content')
        logger.info(f"Disk: {disk_stat.free / 1e9:.1f}GB free")

        # Drive
        if self.drive_mounted:
            logger.info("Drive: Mounted ✅")
        else:
            logger.warning("Drive: Not mounted ⚠️")

        logger.info("=" * 60)

        return {
            'gpu': gpu_info,
            'device': str(self.device),
            'drive_mounted': self.drive_mounted,
            'directories': self.dirs,
        }


class CheckpointManager:
    """Handle training checkpoints with auto-save"""

    def __init__(self, checkpoint_dir: str, interval_minutes: int = 20):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval_minutes = interval_minutes
        self.last_checkpoint_time = time.time()
        self.checkpoint_count = 0

    def should_checkpoint(self) -> bool:
        """Check if interval elapsed"""
        elapsed_minutes = (time.time() - self.last_checkpoint_time) / 60
        return elapsed_minutes >= self.interval_minutes

    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       symbol: str,
                       metrics: Dict) -> bool:
        """Save training checkpoint"""

        if not self.should_checkpoint():
            return False

        checkpoint = {
            'epoch': epoch,
            'symbol': symbol,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{symbol}_epoch{epoch:03d}_{timestamp}.pt'
        filepath = self.checkpoint_dir / filename

        torch.save(checkpoint, filepath)
        self.last_checkpoint_time = time.time()
        self.checkpoint_count += 1

        logger.info(f"✅ Checkpoint {self.checkpoint_count}: {filename}")
        logger.info(f"   Metrics: {metrics}")

        return True

    def load_latest_checkpoint(self,
                              symbol: str,
                              model: torch.nn.Module,
                              optimizer: torch.optim.Optimizer) -> Optional[Dict]:
        """Load latest checkpoint for resume"""

        import glob

        pattern = str(self.checkpoint_dir / f'{symbol}_epoch*.pt')
        checkpoints = glob.glob(pattern)

        if not checkpoints:
            logger.warning(f"No checkpoints found for {symbol}")
            return None

        latest = max(checkpoints, key=os.path.getctime)
        logger.info(f"Loading checkpoint: {latest}")

        checkpoint = torch.load(latest, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint['metrics'],
            'symbol': checkpoint['symbol']
        }

    def cleanup_old_checkpoints(self, keep_latest: int = 3):
        """Remove old checkpoints to save space"""

        import glob

        all_checkpoints = sorted(
            glob.glob(str(self.checkpoint_dir / '*.pt')),
            key=os.path.getctime,
            reverse=True
        )

        to_delete = all_checkpoints[keep_latest:]
        for checkpoint in to_delete:
            os.remove(checkpoint)
            logger.info(f"Deleted old checkpoint: {Path(checkpoint).name}")


class SessionRecoveryManager:
    """Handle Colab session timeout and recovery"""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)

    def save_training_state(self,
                           current_symbol: str,
                           current_epoch: int,
                           symbols_completed: List[str],
                           training_config: Dict):
        """Save training progress to survive disconnects"""

        state = {
            'current_symbol': current_symbol,
            'current_epoch': current_epoch,
            'symbols_completed': symbols_completed,
            'training_config': training_config,
            'last_update': datetime.now().isoformat()
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"State saved: {current_symbol} epoch {current_epoch}")

    def load_training_state(self) -> Optional[Dict]:
        """Load previous training state"""

        if not self.state_file.exists():
            return None

        with open(self.state_file, 'r') as f:
            state = json.load(f)

        logger.info(f"Resuming from {state['current_symbol']} epoch {state['current_epoch']}")
        return state

    def clear_state(self):
        """Clear state after successful completion"""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("Training state cleared")


class RcloneSyncManager:
    """Manage bidirectional sync with Google Drive"""

    def __init__(self, remote_name: str = 'gdrive'):
        self.remote_name = remote_name

    def upload_models(self, local_path: str, remote_path: str) -> bool:
        """Upload models to Drive"""

        cmd = [
            'rclone', 'copy',
            local_path,
            f'{self.remote_name}:{remote_path}',
            '--progress',
            '--transfers', '4',
            '--fast-list'
        ]

        logger.info(f"Uploading {local_path} → {remote_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Upload complete")
            return True
        else:
            logger.error(f"Upload failed: {result.stderr}")
            return False

    def download_data(self, remote_path: str, local_path: str) -> bool:
        """Download training data from Drive"""

        cmd = [
            'rclone', 'copy',
            f'{self.remote_name}:{remote_path}',
            local_path,
            '--progress',
            '--transfers', '4',
            '--fast-list'
        ]

        logger.info(f"Downloading {remote_path} → {local_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("✅ Download complete")
            return True
        else:
            logger.error(f"Download failed: {result.stderr}")
            return False


class ColaTrainingSetup:
    """Main training orchestrator"""

    def __init__(self,
                 symbols: Optional[List[str]] = None,
                 config: Optional[Dict] = None):
        """
        Initialize training setup

        Args:
            symbols: Stock symbols to train (default: top 102)
            config: Training configuration
        """

        # Default symbols (top Vietnamese stocks)
        self.symbols = symbols or [
            'HPG', 'VCB', 'FPT', 'MWG', 'SAB', 'MSN', 'VNM', 'VJC',
            'BID', 'GAS', 'MSB', 'TCB', 'EIB', 'NVL', 'STB', 'HDB',
            'SHB', 'MBB', 'TTLB', 'VRE', 'POW', 'KDC', 'PLC', 'DXG',
            'CTG', 'AGF', 'ACB', 'TPB', 'CTD', 'PVD',  # Add more as needed
        ]

        # Default config
        self.config = config or {
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'checkpoint_interval': 20,
            'num_workers': 4,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        }

        # Managers
        self.env_setup = ColabEnvironmentSetup()
        self.checkpoint_mgr = None
        self.recovery_mgr = None
        self.sync_mgr = RcloneSyncManager()

    def initialize(self) -> Dict:
        """Full initialization"""

        logger.info("=" * 60)
        logger.info("VN-QUANT STOCKFORMER COLAB TRAINING SETUP")
        logger.info("=" * 60)

        # Setup GPU
        self.env_setup.setup_gpu()

        # Mount Drive
        if not self.env_setup.mount_google_drive():
            logger.error("Drive mount failed, continuing with local storage")

        # Create directories
        dirs = self.env_setup.create_directory_structure()

        # Initialize managers
        self.checkpoint_mgr = CheckpointManager(
            dirs['checkpoints'],
            interval_minutes=self.config['checkpoint_interval']
        )

        self.recovery_mgr = SessionRecoveryManager(
            f"{dirs['base']}/training_state.json"
        )

        # Verify
        verification = self.env_setup.verify_colab_environment()

        logger.info("✅ Setup Complete")
        logger.info(f"   Symbols: {len(self.symbols)}")
        logger.info(f"   Device: {self.config['device']}")

        return {
            'environment': verification,
            'symbols': len(self.symbols),
            'config': self.config,
            'directories': dirs,
        }

    def start_training(self, all_symbols: bool = False, resume: bool = True) -> Dict:
        """Start training pipeline"""

        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)

        # Check for resume
        if resume:
            state = self.recovery_mgr.load_training_state()
            if state:
                start_idx = self.symbols.index(state['current_symbol'])
            else:
                start_idx = 0
        else:
            start_idx = 0

        symbols_to_train = self.symbols if all_symbols else self.symbols[:5]
        results = {}

        start_time = time.time()

        for idx, symbol in enumerate(symbols_to_train[start_idx:], start_idx):
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {symbol} ({idx+1}/{len(symbols_to_train)})")
            logger.info(f"{'='*60}")

            try:
                # Simulate training (replace with actual training code)
                epoch_start = 0
                for epoch in range(epoch_start, self.config['epochs']):
                    # Training loop would go here
                    if epoch % 10 == 0:
                        metrics = {'train_loss': 0.1, 'val_loss': 0.12}
                        self.checkpoint_mgr.save_checkpoint(
                            model=None,  # Use actual model
                            optimizer=None,  # Use actual optimizer
                            epoch=epoch,
                            symbol=symbol,
                            metrics=metrics
                        )

                    # Save state
                    if epoch % 20 == 0:
                        self.recovery_mgr.save_training_state(
                            current_symbol=symbol,
                            current_epoch=epoch,
                            symbols_completed=self.symbols[:idx],
                            training_config=self.config
                        )

                results[symbol] = {
                    'status': 'completed',
                    'accuracy': 0.55,  # Placeholder
                    'train_time': 20,
                }

            except Exception as e:
                logger.error(f"Training failed for {symbol}: {e}")
                results[symbol] = {
                    'status': 'failed',
                    'error': str(e)
                }

                # Save state for recovery
                self.recovery_mgr.save_training_state(
                    current_symbol=symbol,
                    current_epoch=0,
                    symbols_completed=self.symbols[:idx],
                    training_config=self.config
                )

                raise

        elapsed = (time.time() - start_time) / 3600
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total time: {elapsed:.1f} hours")
        logger.info(f"Models trained: {len([r for r in results.values() if r['status'] == 'completed'])}")

        self.recovery_mgr.clear_state()

        return {
            'results': results,
            'total_time_hours': elapsed,
            'timestamp': datetime.now().isoformat()
        }


# ============== COLAB NOTEBOOK CELLS ==============

def print_colab_setup_instructions():
    """Print setup instructions for Colab"""

    print("""
╔════════════════════════════════════════════════════════════════╗
║  VN-QUANT STOCKFORMER COLAB TRAINING SETUP                     ║
╚════════════════════════════════════════════════════════════════╝

SETUP INSTRUCTIONS:

Cell 1: Install Dependencies
─────────────────────────────
!pip install -q torch torchvision torchaudio
!apt-get install -q rclone

Cell 2: Upload and Run Setup
───────────────────────────────
# Copy this file to Google Drive
# Then run:
%run '/content/drive/My Drive/VN-Quant-Training/colab_training_setup.py'

setup = ColaTrainingSetup()
setup.initialize()

Cell 3: Start Training
─────────────────────────
results = setup.start_training(all_symbols=True, resume=True)

Cell 4: Upload Results
──────────────────────
setup.sync_mgr.upload_models(
    '/content/models',
    'VN-Quant-Training/models'
)

═══════════════════════════════════════════════════════════════════
    """)


if __name__ == '__main__':
    print_colab_setup_instructions()
