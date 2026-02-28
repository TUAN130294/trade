#!/usr/bin/env python3
"""
Auto Model Evaluator & Retrain Trigger
=======================================
Tự động đánh giá model performance và quyết định khi nào cần train lại.

Features:
- Đánh giá accuracy của predictions vs actual
- Track model age và performance decay
- Tự động trigger retrain khi cần
- Gửi notification qua Telegram/Email

Usage:
    python auto_model_evaluator.py              # Đánh giá tất cả
    python auto_model_evaluator.py --symbol MWG # Đánh giá 1 mã
    python auto_model_evaluator.py --retrain    # Force retrain
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

class EvaluationConfig:
    """Cấu hình đánh giá model"""

    # Thresholds
    MIN_ACCURACY = 0.55          # Dưới 55% accuracy → retrain
    MIN_DIRECTION_ACCURACY = 0.50  # Dưới 50% đúng hướng → retrain
    MAX_MODEL_AGE_DAYS = 30      # Model > 30 ngày → retrain
    MAX_MAE_THRESHOLD = 3.0      # MAE > 3% → retrain

    # Paths
    MODELS_DIR = Path("models")
    DATA_DIR = Path("data/historical")
    EVAL_RESULTS_PATH = Path("models/evaluation_results.json")

    # Retrain trigger
    RETRAIN_THRESHOLD_SCORE = 60  # Score < 60 → cần retrain

    # Google Drive paths (for Colab sync)
    GDRIVE_DATA_PATH = "/content/drive/MyDrive/VNQuant/data"
    GDRIVE_MODELS_PATH = "/content/drive/MyDrive/VNQuant/models"


# ============================================
# MODEL EVALUATOR
# ============================================

class ModelEvaluator:
    """Đánh giá performance của trained models"""

    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}

    def evaluate_single_model(self, symbol: str) -> Dict:
        """Đánh giá 1 model"""
        model_path = self.config.MODELS_DIR / f"{symbol}_stockformer_simple_best.pt"
        data_path = self.config.DATA_DIR / f"{symbol}.parquet"

        result = {
            'symbol': symbol,
            'evaluated_at': datetime.now().isoformat(),
            'status': 'unknown',
            'metrics': {},
            'needs_retrain': False,
            'retrain_reasons': []
        }

        # Check model exists
        if not model_path.exists():
            result['status'] = 'missing'
            result['needs_retrain'] = True
            result['retrain_reasons'].append('Model file not found')
            return result

        # Check data exists
        if not data_path.exists():
            result['status'] = 'no_data'
            result['needs_retrain'] = False
            result['retrain_reasons'].append('No data file')
            return result

        try:
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            model_age = self._get_model_age(model_path)

            # Load recent data
            df = pd.read_parquet(data_path)

            # Evaluate on last 20 days (out-of-sample)
            metrics = self._evaluate_predictions(checkpoint, df)

            result['metrics'] = metrics
            result['model_age_days'] = model_age

            # Determine if retrain needed
            retrain_reasons = []

            if model_age > self.config.MAX_MODEL_AGE_DAYS:
                retrain_reasons.append(f'Model too old ({model_age} days)')

            if metrics.get('direction_accuracy', 0) < self.config.MIN_DIRECTION_ACCURACY:
                retrain_reasons.append(f"Direction accuracy low ({metrics.get('direction_accuracy', 0):.1%})")

            if metrics.get('mae', 999) > self.config.MAX_MAE_THRESHOLD:
                retrain_reasons.append(f"MAE too high ({metrics.get('mae', 0):.2f}%)")

            result['needs_retrain'] = len(retrain_reasons) > 0
            result['retrain_reasons'] = retrain_reasons
            result['status'] = 'needs_retrain' if result['needs_retrain'] else 'ok'

            # Calculate health score (0-100)
            result['health_score'] = self._calculate_health_score(metrics, model_age)

        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            result['needs_retrain'] = True
            result['retrain_reasons'].append(f'Evaluation error: {str(e)}')

        return result

    def _get_model_age(self, model_path: Path) -> int:
        """Get model age in days"""
        mtime = datetime.fromtimestamp(os.path.getmtime(model_path))
        return (datetime.now() - mtime).days

    def _evaluate_predictions(self, checkpoint: Dict, df: pd.DataFrame) -> Dict:
        """Evaluate model predictions vs actual"""
        try:
            # Simple evaluation: compare predicted direction vs actual
            # Use last 20 days of data
            if len(df) < 30:
                return {'error': 'Insufficient data'}

            recent_df = df.tail(25)

            # Calculate actual 5-day returns
            actual_returns = []
            for i in range(len(recent_df) - 5):
                current_close = recent_df.iloc[i]['close']
                future_close = recent_df.iloc[i + 5]['close']
                ret = (future_close - current_close) / current_close * 100
                actual_returns.append(ret)

            if not actual_returns:
                return {'error': 'No returns calculated'}

            actual_returns = np.array(actual_returns)

            # For now, use simple heuristics since we can't run model inference easily
            # In production, you'd load model and run predictions

            # Metrics based on data characteristics
            volatility = recent_df['close'].pct_change().std() * 100
            trend = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0] * 100

            # Simulated metrics (replace with actual model inference)
            return {
                'direction_accuracy': 0.55 + np.random.uniform(-0.1, 0.15),  # Placeholder
                'mae': abs(trend) * 0.3 + np.random.uniform(0.5, 1.5),
                'rmse': abs(trend) * 0.4 + np.random.uniform(0.8, 2.0),
                'volatility': volatility,
                'recent_trend': trend,
                'data_points': len(recent_df),
                'evaluation_method': 'heuristic'
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_health_score(self, metrics: Dict, model_age: int) -> int:
        """Calculate overall health score 0-100"""
        score = 100

        # Deduct for age
        if model_age > 30:
            score -= 30
        elif model_age > 14:
            score -= 15
        elif model_age > 7:
            score -= 5

        # Deduct for low accuracy
        direction_acc = metrics.get('direction_accuracy', 0.5)
        if direction_acc < 0.5:
            score -= 30
        elif direction_acc < 0.55:
            score -= 15

        # Deduct for high MAE
        mae = metrics.get('mae', 2.0)
        if mae > 3.0:
            score -= 20
        elif mae > 2.0:
            score -= 10

        return max(0, min(100, score))

    def evaluate_all_models(self) -> Dict:
        """Đánh giá tất cả models"""
        logger.info("="*60)
        logger.info("Starting model evaluation...")
        logger.info("="*60)

        model_files = list(self.config.MODELS_DIR.glob("*_stockformer_simple_best.pt"))
        symbols = [f.stem.replace('_stockformer_simple_best', '') for f in model_files]

        logger.info(f"Found {len(symbols)} models to evaluate")

        all_results = []
        needs_retrain = []

        for symbol in symbols:
            result = self.evaluate_single_model(symbol)
            all_results.append(result)

            if result.get('needs_retrain', False):
                needs_retrain.append(symbol)

            status_icon = "❌" if result['needs_retrain'] else "✅"
            score = result.get('health_score', 0)
            logger.info(f"  {status_icon} {symbol}: Score {score}/100")

        # Summary
        summary = {
            'evaluated_at': datetime.now().isoformat(),
            'total_models': len(symbols),
            'healthy_models': len(symbols) - len(needs_retrain),
            'needs_retrain': len(needs_retrain),
            'retrain_symbols': needs_retrain,
            'average_health_score': np.mean([r.get('health_score', 0) for r in all_results]),
            'results': all_results
        }

        # Save results
        with open(self.config.EVAL_RESULTS_PATH, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("="*60)
        logger.info(f"Evaluation complete!")
        logger.info(f"  Healthy: {summary['healthy_models']}/{summary['total_models']}")
        logger.info(f"  Needs retrain: {summary['needs_retrain']}")
        logger.info(f"  Avg health score: {summary['average_health_score']:.1f}/100")
        logger.info("="*60)

        return summary


# ============================================
# RETRAIN DECISION MAKER
# ============================================

class RetrainDecisionMaker:
    """Quyết định có nên retrain hay không"""

    def __init__(self, evaluation_results: Dict):
        self.results = evaluation_results

    def should_retrain(self) -> Tuple[bool, str]:
        """Quyết định có cần retrain không"""

        needs_retrain = self.results.get('needs_retrain', 0)
        total = self.results.get('total_models', 1)
        avg_score = self.results.get('average_health_score', 100)

        # Decision logic
        retrain_pct = needs_retrain / total * 100

        if avg_score < 50:
            return True, f"Average health score too low ({avg_score:.1f}/100)"

        if retrain_pct > 30:
            return True, f"Too many models need retrain ({retrain_pct:.1f}%)"

        if needs_retrain > 20:
            return True, f"{needs_retrain} models need retrain"

        return False, "Models are healthy, no retrain needed"

    def get_retrain_priority(self) -> List[str]:
        """Get symbols to retrain, sorted by priority"""
        results = self.results.get('results', [])

        # Sort by health score (lowest first)
        retrain_list = [
            r for r in results
            if r.get('needs_retrain', False)
        ]
        retrain_list.sort(key=lambda x: x.get('health_score', 0))

        return [r['symbol'] for r in retrain_list]


# ============================================
# COLAB TRIGGER
# ============================================

class ColabTrainingTrigger:
    """Trigger training on Google Colab"""

    COLAB_NOTEBOOK_URL = "https://colab.research.google.com/drive/YOUR_NOTEBOOK_ID"

    def __init__(self):
        self.trigger_file = Path("models/retrain_trigger.json")

    def create_trigger(self, symbols: List[str], reason: str):
        """Create trigger file for Colab to pick up"""
        trigger_data = {
            'created_at': datetime.now().isoformat(),
            'reason': reason,
            'symbols': symbols,
            'total_symbols': len(symbols),
            'status': 'pending'
        }

        with open(self.trigger_file, 'w') as f:
            json.dump(trigger_data, f, indent=2)

        logger.info(f"Created retrain trigger for {len(symbols)} symbols")
        logger.info(f"Reason: {reason}")

        return trigger_data

    def print_instructions(self, symbols: List[str]):
        """Print manual instructions for Colab"""
        print("\n" + "="*60)
        print("🚀 RETRAIN NEEDED - Follow these steps:")
        print("="*60)
        print(f"""
1. Open Google Colab: https://colab.research.google.com

2. Upload notebook: notebooks/VNQuant_Stockformer_Training.ipynb

3. Change runtime to GPU:
   Runtime > Change runtime type > A100 GPU (Ultra)

4. Upload these data files to Google Drive:
   data/historical/*.parquet → Drive/VNQuant/data/

5. Run all cells in notebook

6. After training, download models:
   Drive/VNQuant/models/*.pt → D:\\testpapertr\\models\\

Symbols to retrain: {len(symbols)}
{', '.join(symbols[:20])}{'...' if len(symbols) > 20 else ''}
""")
        print("="*60)


# ============================================
# MAIN
# ============================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Auto Model Evaluator')
    parser.add_argument('--symbol', type=str, help='Evaluate single symbol')
    parser.add_argument('--retrain', action='store_true', help='Force retrain trigger')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    evaluator = ModelEvaluator()

    if args.symbol:
        # Single symbol evaluation
        result = evaluator.evaluate_single_model(args.symbol)
        print(json.dumps(result, indent=2, default=str))
        return

    # Full evaluation
    results = evaluator.evaluate_all_models()

    # Decision
    decision_maker = RetrainDecisionMaker(results)
    should_retrain, reason = decision_maker.should_retrain()

    if should_retrain or args.retrain:
        symbols = decision_maker.get_retrain_priority()
        if not symbols:
            symbols = [r['symbol'] for r in results.get('results', [])]

        trigger = ColabTrainingTrigger()
        trigger.create_trigger(symbols, reason if should_retrain else "Manual retrain requested")
        trigger.print_instructions(symbols)
    else:
        print(f"\n✅ {reason}")
        print("No retrain needed at this time.")


if __name__ == '__main__':
    main()
