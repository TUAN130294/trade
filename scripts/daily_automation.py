#!/usr/bin/env python3
"""
Daily Automation Script for i7-14700 Host
==========================================
Chạy tự động hàng ngày trên host server.

Tasks:
- 15:30: Download new market data
- 16:00: Evaluate model performance
- 18:00: Generate daily report
- 08:45: Start trading server (if not running)

Usage:
    python daily_automation.py              # Run all daily tasks
    python daily_automation.py --download   # Only download data
    python daily_automation.py --evaluate   # Only evaluate models
    python daily_automation.py --report     # Only generate report
    python daily_automation.py --schedule   # Run as scheduled service
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime, time
import schedule
import time as time_module

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/daily_automation.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
Path('logs').mkdir(exist_ok=True)


class DailyAutomation:
    """Daily automation tasks for VN-Quant"""

    def __init__(self):
        self.data_dir = Path("data/historical")
        self.models_dir = Path("models")
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)

    def download_market_data(self):
        """Download latest market data from CafeF (run after 15:00)"""
        logger.info("="*50)
        logger.info("📥 Downloading market data...")
        logger.info("="*50)

        try:
            # Import and run download script
            from download_all_stocks import download_stock, VN_STOCKS, DATA_DIR

            success = 0
            failed = 0

            for symbol in VN_STOCKS[:100]:  # Top 100 stocks
                if download_stock(symbol):
                    success += 1
                else:
                    failed += 1

                # Rate limit
                time_module.sleep(0.3)

            logger.info(f"✅ Downloaded: {success}, Failed: {failed}")
            return {'success': success, 'failed': failed}

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            return {'error': str(e)}

    def evaluate_models(self):
        """Evaluate model performance"""
        logger.info("="*50)
        logger.info("📊 Evaluating models...")
        logger.info("="*50)

        try:
            from scripts.auto_model_evaluator import ModelEvaluator, RetrainDecisionMaker

            evaluator = ModelEvaluator()
            results = evaluator.evaluate_all_models()

            decision_maker = RetrainDecisionMaker(results)
            should_retrain, reason = decision_maker.should_retrain()

            logger.info(f"Healthy: {results['healthy_models']}/{results['total_models']}")
            logger.info(f"Avg score: {results['average_health_score']:.1f}/100")

            if should_retrain:
                logger.warning(f"⚠️ RETRAIN NEEDED: {reason}")

            return results

        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return {'error': str(e)}

    def generate_daily_report(self):
        """Generate daily performance report"""
        logger.info("="*50)
        logger.info("📝 Generating daily report...")
        logger.info("="*50)

        try:
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'generated_at': datetime.now().isoformat(),
                'sections': {}
            }

            # 1. Trading summary
            try:
                import requests
                status = requests.get('http://localhost:8001/api/status', timeout=5).json()
                positions = requests.get('http://localhost:8001/api/positions', timeout=5).json()

                report['sections']['trading'] = {
                    'balance': status.get('balance', 0),
                    'active_positions': status.get('active_positions', 0),
                    'opportunities_detected': status.get('statistics', {}).get('opportunities_detected', 0),
                    'positions': positions.get('positions', [])
                }
            except:
                report['sections']['trading'] = {'error': 'Server not running'}

            # 2. Model health
            eval_file = Path('models/evaluation_results.json')
            if eval_file.exists():
                with open(eval_file) as f:
                    eval_data = json.load(f)
                report['sections']['models'] = {
                    'total': eval_data.get('total_models', 0),
                    'healthy': eval_data.get('healthy_models', 0),
                    'needs_retrain': eval_data.get('needs_retrain', 0),
                    'avg_score': eval_data.get('average_health_score', 0)
                }

            # 3. Data status
            parquet_files = list(self.data_dir.glob('*.parquet'))
            report['sections']['data'] = {
                'total_stocks': len(parquet_files),
                'data_dir': str(self.data_dir)
            }

            # Save report
            report_file = self.reports_dir / f"daily_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"✅ Report saved: {report_file}")

            # Print summary
            self._print_report_summary(report)

            return report

        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return {'error': str(e)}

    def _print_report_summary(self, report):
        """Print human-readable report summary"""
        print("\n" + "="*60)
        print(f"📊 DAILY REPORT - {report['date']}")
        print("="*60)

        trading = report['sections'].get('trading', {})
        if 'error' not in trading:
            print(f"\n💰 Trading:")
            print(f"   Balance: {trading.get('balance', 0):,.0f} VND")
            print(f"   Positions: {trading.get('active_positions', 0)}")
            print(f"   Opportunities: {trading.get('opportunities_detected', 0)}")

        models = report['sections'].get('models', {})
        if models:
            print(f"\n🤖 Models:")
            print(f"   Healthy: {models.get('healthy', 0)}/{models.get('total', 0)}")
            print(f"   Avg Score: {models.get('avg_score', 0):.1f}/100")
            if models.get('needs_retrain', 0) > 20:
                print(f"   ⚠️ {models.get('needs_retrain', 0)} models need retrain!")

        data = report['sections'].get('data', {})
        print(f"\n📁 Data:")
        print(f"   Stocks: {data.get('total_stocks', 0)}")

        print("\n" + "="*60)

    def run_all_daily_tasks(self):
        """Run all daily tasks in sequence"""
        logger.info("🚀 Starting daily automation...")

        current_hour = datetime.now().hour

        # After market close (15:00+)
        if current_hour >= 15:
            self.download_market_data()

        # Evaluate models
        self.evaluate_models()

        # Generate report
        self.generate_daily_report()

        logger.info("✅ Daily automation complete!")

    def run_scheduled(self):
        """Run as scheduled service"""
        logger.info("📅 Starting scheduled automation service...")

        # Schedule tasks
        schedule.every().day.at("15:30").do(self.download_market_data)
        schedule.every().day.at("16:00").do(self.evaluate_models)
        schedule.every().day.at("18:00").do(self.generate_daily_report)

        logger.info("Scheduled tasks:")
        logger.info("  15:30 - Download market data")
        logger.info("  16:00 - Evaluate models")
        logger.info("  18:00 - Generate report")

        while True:
            schedule.run_pending()
            time_module.sleep(60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Daily Automation')
    parser.add_argument('--download', action='store_true', help='Download data only')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate models only')
    parser.add_argument('--report', action='store_true', help='Generate report only')
    parser.add_argument('--schedule', action='store_true', help='Run as scheduled service')
    args = parser.parse_args()

    automation = DailyAutomation()

    if args.download:
        automation.download_market_data()
    elif args.evaluate:
        automation.evaluate_models()
    elif args.report:
        automation.generate_daily_report()
    elif args.schedule:
        automation.run_scheduled()
    else:
        automation.run_all_daily_tasks()


if __name__ == '__main__':
    main()
