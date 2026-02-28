#!/usr/bin/env python3
"""
Sync data to Google Drive for Colab training
=============================================
Options:
1. Manual: Copy files to Google Drive folder
2. Rclone: Auto sync (requires setup)
3. PyDrive: Python API (requires OAuth)

This script helps prepare data for upload.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import zipfile

DATA_DIR = Path("data/historical")
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("sync_package")

def create_upload_package():
    """Create a zip package for easy upload to Google Drive"""

    print("="*60)
    print("Creating upload package for Google Drive")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Zip data files
    data_zip = OUTPUT_DIR / "vnquant_data.zip"
    print(f"\n1. Zipping data files...")

    with zipfile.ZipFile(data_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        parquet_files = list(DATA_DIR.glob("*.parquet"))
        for f in parquet_files:
            zf.write(f, f"data/{f.name}")
        print(f"   Added {len(parquet_files)} parquet files")

    data_size = data_zip.stat().st_size / 1024 / 1024
    print(f"   Data package: {data_size:.1f} MB")

    # Copy notebook
    print(f"\n2. Copying notebook...")
    notebook_src = Path("notebooks/VNQuant_Stockformer_Training.ipynb")
    if notebook_src.exists():
        shutil.copy(notebook_src, OUTPUT_DIR / "VNQuant_Stockformer_Training.ipynb")
        print(f"   Copied: {notebook_src.name}")

    # Create instructions
    print(f"\n3. Creating instructions...")
    instructions = f"""
VN-Quant Colab Training Package
================================
Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Contents:
- vnquant_data.zip: Historical data ({len(list(DATA_DIR.glob('*.parquet')))} stocks)
- VNQuant_Stockformer_Training.ipynb: Training notebook

Instructions:
1. Upload vnquant_data.zip to Google Drive
2. Extract to: Google Drive/VNQuant/data/
3. Open notebook in Colab
4. Runtime > Change runtime type > A100 GPU
5. Run all cells
6. Download trained models from Drive/VNQuant/models/

Estimated training time (A100): 15-20 hours for 100 stocks
"""

    with open(OUTPUT_DIR / "README.txt", 'w') as f:
        f.write(instructions)

    print(f"\n" + "="*60)
    print("PACKAGE READY!")
    print("="*60)
    print(f"\nUpload these files to Google Drive:")
    print(f"  📁 {OUTPUT_DIR.absolute()}")
    for f in OUTPUT_DIR.iterdir():
        size = f.stat().st_size / 1024 / 1024
        print(f"     - {f.name} ({size:.1f} MB)")

    print(f"\nThen open Colab and run the notebook!")

    return str(OUTPUT_DIR.absolute())


def download_models_from_gdrive():
    """Instructions for downloading models after training"""
    print("""
After Colab training completes:
================================

1. Go to Google Drive > VNQuant > models

2. Download all *_stockformer_simple_best.pt files

3. Copy to: D:\\testpapertr\\models\\

Or use this command (if rclone configured):
   rclone copy gdrive:VNQuant/models/ models/ --include "*.pt"
""")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--download':
        download_models_from_gdrive()
    else:
        create_upload_package()
