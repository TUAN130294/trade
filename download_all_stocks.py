# -*- coding: utf-8 -*-
"""
Download All Vietnam Stock Historical Data from CafeF
=====================================================
Downloads OHLCV data for all stocks and saves to parquet files.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import sys

# CafeF API
CAFEF_API = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Output directory
DATA_DIR = Path("data/historical")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# List of all VN stock symbols (top stocks first, then others)
VN_STOCKS = [
    # VN30 Blue chips
    "ACB", "BCM", "BID", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SHB", "SSB", "SSI", "STB",
    "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE",
    # HNX30
    "CEO", "DDG", "DGC", "DHT", "HUT", "IDC", "L14", "MBS", "NDN", "NTP",
    "NVB", "PVB", "PVC", "PVS", "S99", "SHS", "SLS", "TIG", "TNG", "VC3",
    "VCS", "VGC", "VND", "VNR",
    # Other popular stocks
    "AAA", "AGG", "ANV", "ASM", "BWE", "CII", "CMG", "CTD", "DBC", "DCM",
    "DGW", "DIG", "DPM", "DXG", "EIB", "EVF", "FCN", "GEX", "GMD", "HAG",
    "HAH", "HCM", "HDC", "HDG", "HNG", "HSG", "HT1", "HTN", "HVN", "IJC",
    "IMP", "ITA", "KBC", "KDC", "KDH", "KSB", "LCG", "LPB", "MIG", "MSH",
    "NAB", "NKG", "NLG", "NT2", "NVL", "OCB", "PAN", "PC1", "PDR", "PHR",
    "PNJ", "PPC", "PTB", "PVD", "PVT", "REE", "SBT", "SCR", "SCS", "SIP",
    "SJS", "SKG", "SMC", "SSC", "SZC", "TDM", "TLG", "TLH", "TNH", "VCG",
    "VCI", "VDS", "VGI", "VHC", "VIP", "VOS", "VPI", "VRC", "VSC", "VSH",
    "VTO", "YEG",
    # Additional stocks
    "AAM", "ABT", "ACC", "ACL", "ACV", "ADG", "ADS", "AGM", "AGR", "APC",
    "APG", "APH", "AST", "BAF", "BBC", "BCC", "BCE", "BFC", "BGM", "BHN",
    "BIC", "BKG", "BMP", "BRC", "BSI", "BTT", "BVS", "C32", "C47", "CAV",
    "CCI", "CCL", "CDC", "CDN", "CHP", "CIG", "CLC", "CLG", "CLL", "CMV",
    "CMX", "CNG", "COM", "CRC", "CRE", "CSM", "CSV", "CTF", "CTI", "CTR",
    "CTS", "CVT", "D2D", "DAG", "DAH", "DAT", "DBC", "DBD", "DBT", "DC4",
    "DHA", "DHC", "DHG", "DHM", "DIC", "DLG", "DMC", "DPG", "DPR", "DQC",
    "DRC", "DRH", "DRL", "DSN", "DTA", "DTD", "DTL", "DTT", "DVP", "DXS",
    "ELC", "EMC", "EVE", "EVG", "FDC", "FIR", "FIT", "FLC", "FMC", "FRT",
    "FTS", "GDT", "GIL", "GLT", "GSP", "GTA", "GTN", "HAD", "HAP", "HAR",
    "HAS", "HAX", "HBC", "HCD", "HHP", "HHS", "HID", "HII", "HMC", "HNA",
    "HND", "HNF", "HOT", "HPX", "HQC", "HRC", "HTV", "HU1", "HU3", "HUB",
    "HVH", "HVT", "ICT", "IDI", "IDJ", "IDV", "ILB", "INC", "ITD", "ITS",
    "JVC", "KAC", "KHA", "KHG", "KHP", "KMR", "KOS", "KPF", "KSD", "KTL",
    "L10", "L18", "LAF", "LAS", "LBM", "LDG", "LEC", "LGC", "LGL", "LHG",
    "LIG", "LIX", "LM8", "LSS", "MAC", "MCG", "MCP", "MDG", "MEL", "MHC",
]


def download_stock(symbol: str, days: int = 365) -> bool:
    """Download historical data for a single stock"""
    try:
        url = f"{CAFEF_API}?Symbol={symbol}&StartDate=&EndDate=&PageIndex=1&PageSize={days}"
        resp = requests.get(url, headers=HEADERS, timeout=15)

        if resp.status_code != 200:
            return False

        data = resp.json()
        if not data.get('Success') or not data.get('Data', {}).get('Data'):
            return False

        items = data['Data']['Data']
        if len(items) < 10:
            return False

        # Convert to DataFrame
        records = []
        for item in reversed(items):
            try:
                records.append({
                    'date': item.get('Ngay', ''),
                    'open': float(item.get('GiaMoCua', 0)) * 1000,
                    'high': float(item.get('GiaCaoNhat', 0)) * 1000,
                    'low': float(item.get('GiaThapNhat', 0)) * 1000,
                    'close': float(item.get('GiaDongCua', 0)) * 1000,
                    'volume': int(item.get('KhoiLuongKhopLenh', 0))
                })
            except:
                continue

        if len(records) < 10:
            return False

        df = pd.DataFrame(records)

        # Save to parquet
        output_path = DATA_DIR / f"{symbol}.parquet"
        df.to_parquet(output_path, index=False)

        return True

    except Exception as e:
        return False


def main():
    print("=" * 60)
    print("VN Stock Data Downloader - CafeF")
    print("=" * 60)
    print(f"Total stocks to download: {len(VN_STOCKS)}")
    print(f"Output directory: {DATA_DIR.absolute()}")
    print()

    success = 0
    failed = 0
    failed_symbols = []

    for i, symbol in enumerate(VN_STOCKS, 1):
        sys.stdout.write(f"\r[{i}/{len(VN_STOCKS)}] Downloading {symbol}... ")
        sys.stdout.flush()

        if download_stock(symbol):
            success += 1
            sys.stdout.write(f"OK ({success} downloaded)")
        else:
            failed += 1
            failed_symbols.append(symbol)
            sys.stdout.write(f"FAILED")

        sys.stdout.flush()

        # Rate limiting - don't hammer the API
        time.sleep(0.3)

    print("\n")
    print("=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Success: {success}")
    print(f"Failed: {failed}")

    if failed_symbols:
        print(f"Failed symbols: {', '.join(failed_symbols[:20])}")
        if len(failed_symbols) > 20:
            print(f"  ... and {len(failed_symbols) - 20} more")

    # Create summary file
    summary = {
        "last_update": datetime.now().isoformat(),
        "total_downloaded": success,
        "total_failed": failed,
        "coverage_pct": round(success / len(VN_STOCKS) * 100, 1)
    }

    import json
    with open(DATA_DIR / "data_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nData saved to: {DATA_DIR.absolute()}")


if __name__ == "__main__":
    main()
