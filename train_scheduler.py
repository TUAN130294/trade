#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly Model Training Scheduler
================================

Automatically runs model training on schedule (default: Sundays 2 AM).

Features:
- Cron-like scheduling with APScheduler
- Email/Slack notifications on completion
- Automatic model validation
- Training status monitoring
- Error handling and retry logic
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

# Training configuration
TRAINING_SCHEDULE = os.getenv('TRAINING_SCHEDULE', '0 2 * * 0')  # 2 AM Sunday
TRAINING_SYMBOLS = os.getenv('TRAINING_SYMBOLS', '').split(',') if os.getenv('TRAINING_SYMBOLS') else None
ENABLE_NOTIFICATIONS = os.getenv('ENABLE_NOTIFICATIONS', 'true').lower() == 'true'
SLACK_WEBHOOK = os.getenv('SLACK_WEBHOOK_URL')
EMAIL_RECIPIENTS = os.getenv('EMAIL_RECIPIENTS', '').split(',') if os.getenv('EMAIL_RECIPIENTS') else None
TIMEZONE = os.getenv('TIMEZONE', 'Asia/Ho_Chi_Minh')


class ModelTrainingScheduler:
    """Manages automated model training schedule"""

    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone(TIMEZONE))
        self.is_running = False
        self.training_history = self._load_history()

    def _load_history(self) -> dict:
        """Load previous training history"""
        history_file = Path('models/stockformer/training_history.json')
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load training history: {e}")
        return {}

    def _save_history(self, entry: dict):
        """Save training history"""
        history_file = Path('models/stockformer/training_history.json')
        history_file.parent.mkdir(parents=True, exist_ok=True)

        self.training_history[datetime.now().strftime('%Y-%m-%d')] = entry

        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

    async def run_training(self):
        """Execute model training"""
        logger.info("=" * 60)
        logger.info("🚀 Starting weekly model training cycle")
        logger.info("=" * 60)

        start_time = datetime.now()
        training_result = {
            'start_time': start_time.isoformat(),
            'stocks_trained': 0,
            'stocks_failed': [],
            'avg_accuracy': 0.0,
            'duration_hours': 0.0,
        }

        try:
            # Import training pipeline
            from quantum_stock.ml.training_pipeline import TrainingPipeline

            pipeline = TrainingPipeline()

            # Run training
            if TRAINING_SYMBOLS:
                logger.info(f"Training specific stocks: {TRAINING_SYMBOLS}")
                results = await pipeline.train_symbols(TRAINING_SYMBOLS)
            else:
                logger.info("Training all available stocks (102)")
                results = await pipeline.train_all()

            # Analyze results
            successful = [r for r in results if r.get('success', False)]
            training_result['stocks_trained'] = len(successful)
            training_result['stocks_failed'] = [r['symbol'] for r in results if not r.get('success', False)]

            if successful:
                accuracies = [r.get('accuracy', 0) for r in successful]
                training_result['avg_accuracy'] = sum(accuracies) / len(accuracies)

            logger.info(f"✅ Training complete")
            logger.info(f"   Successful: {len(successful)}/{len(results)}")
            logger.info(f"   Average accuracy: {training_result['avg_accuracy']:.2%}")

            if training_result['stocks_failed']:
                logger.warning(f"   Failed: {len(training_result['stocks_failed'])}")
                logger.warning(f"   Symbols: {', '.join(training_result['stocks_failed'][:5])}")

            # Validate models
            logger.info("🔍 Validating trained models...")
            validation_results = await self._validate_models()
            training_result['validation_passed'] = validation_results['passed']
            training_result['validation_warnings'] = validation_results['warnings']

            # Deploy models
            logger.info("🚀 Deploying models to production...")
            await self._deploy_models()

            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds() / 3600
            training_result['duration_hours'] = duration
            training_result['end_time'] = datetime.now().isoformat()
            training_result['status'] = 'success'

            logger.info(f"⏱️  Training took {duration:.1f} hours")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            training_result['status'] = 'failed'
            training_result['error'] = str(e)

        finally:
            # Save history
            self._save_history(training_result)

            # Send notifications
            if ENABLE_NOTIFICATIONS:
                await self._send_notifications(training_result)

            logger.info("=" * 60)

    async def _validate_models(self) -> dict:
        """Validate trained models"""
        try:
            from quantum_stock.ml.backtest_models import validate_all_models

            results = await validate_all_models()
            return {
                'passed': len([r for r in results if r.get('valid', False)]),
                'warnings': [r for r in results if not r.get('valid', False)]
            }
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {'passed': 0, 'warnings': [str(e)]}

    async def _deploy_models(self):
        """Deploy trained models to production"""
        try:
            # Models are auto-loaded by scanner
            # Just verify they exist and are readable
            models_dir = Path('models')
            model_files = list(models_dir.glob('*_stockformer_simple_best.pt'))
            logger.info(f"✅ Found {len(model_files)} trained models ready for deployment")
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

    async def _send_notifications(self, result: dict):
        """Send training completion notifications"""
        status = result.get('status', 'unknown')
        status_emoji = '✅' if status == 'success' else '❌'

        message = f"""
{status_emoji} **VN-Quant Weekly Training Report**

Status: {status.upper()}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Duration: {result.get('duration_hours', 0):.1f} hours

📊 Results:
- Stocks Trained: {result.get('stocks_trained', 0)}/102
- Average Accuracy: {result.get('avg_accuracy', 0):.2%}
- Failed: {len(result.get('stocks_failed', []))}

{"⚠️ Failed stocks: " + ", ".join(result.get('stocks_failed', [])[:5]) if result.get('stocks_failed') else ""}
        """

        # Slack notification
        if SLACK_WEBHOOK:
            await self._notify_slack(message)

        # Email notification
        if EMAIL_RECIPIENTS:
            await self._notify_email(message, result)

    async def _notify_slack(self, message: str):
        """Send Slack notification"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    SLACK_WEBHOOK,
                    json={'text': message}
                ) as resp:
                    if resp.status == 200:
                        logger.info("✅ Slack notification sent")
                    else:
                        logger.warning(f"Slack notification failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Slack notification error: {e}")

    async def _notify_email(self, message: str, result: dict):
        """Send email notification"""
        try:
            import aiosmtplib
            from email.mime.text import MIMEText

            smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASSWORD')

            if not all([smtp_user, smtp_password]):
                logger.warning("Email config incomplete, skipping")
                return

            subject = f"[VN-Quant] Training Report - {result.get('status', 'unknown').upper()}"
            body = message

            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = smtp_user
            msg['To'] = ', '.join(EMAIL_RECIPIENTS)

            async with aiosmtplib.SMTP(hostname=smtp_host, port=smtp_port) as smtp:
                await smtp.login(smtp_user, smtp_password)
                await smtp.send_message(msg)

            logger.info("✅ Email notification sent")

        except Exception as e:
            logger.warning(f"Email notification error: {e}")

    def schedule_training(self):
        """Schedule model training to run periodically"""
        try:
            # Parse cron schedule
            parts = TRAINING_SCHEDULE.split()
            if len(parts) != 5:
                logger.error(f"Invalid schedule format: {TRAINING_SCHEDULE}")
                logger.info("Expected format: minute hour day month day_of_week")
                return

            minute, hour, day, month, day_of_week = parts

            trigger = CronTrigger(
                minute=minute,
                hour=hour,
                day=day if day != '*' else None,
                month=month,
                day_of_week=day_of_week if day_of_week != '*' else None,
                timezone=pytz.timezone(TIMEZONE)
            )

            self.scheduler.add_job(
                self._schedule_async_training,
                trigger=trigger,
                id='model_training',
                name='Weekly Model Training',
                replace_existing=True
            )

            logger.info(f"✅ Training scheduled: {TRAINING_SCHEDULE} ({TIMEZONE})")

        except Exception as e:
            logger.error(f"Failed to schedule training: {e}")

    def _schedule_async_training(self):
        """Wrapper to run async training from scheduler"""
        asyncio.run(self.run_training())

    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler already running")
            return

        self.schedule_training()
        self.scheduler.start()
        self.is_running = True

        logger.info("=" * 60)
        logger.info("🎯 Training Scheduler Started")
        logger.info("=" * 60)
        logger.info(f"Schedule: {TRAINING_SCHEDULE}")
        logger.info(f"Timezone: {TIMEZONE}")
        logger.info(f"Notifications: {ENABLE_NOTIFICATIONS}")

        # Keep scheduler running
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down scheduler...")
            self.stop()

    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("✅ Scheduler stopped")

    def run_now(self):
        """Run training immediately (for testing)"""
        logger.info("Running training immediately...")
        asyncio.run(self.run_training())


if __name__ == '__main__':
    scheduler = ModelTrainingScheduler()

    # Check for command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'run-now':
            scheduler.run_now()
        elif command == 'schedule':
            scheduler.start()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python train_scheduler.py [run-now|schedule]")
            sys.exit(1)
    else:
        # Default: start scheduler
        scheduler.start()
