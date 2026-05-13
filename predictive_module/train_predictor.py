"""
Training Pipeline for Predictive Models

Trains both LSTM failure prediction and ARIMA delay forecasting models.

Usage:
    python train_predictor.py
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_all_models():
    """Main training pipeline"""
    
    # Import after logging setup
    from predictive_model import FailurePredictorLSTM, DelayForecaster
    from data_preparation import DataLoader
    from risk_scorer import RiskScorer
    from disruption_handler import DisruptionHandler
    
    logger.info("="*80)
    logger.info("PREDICTIVE DISRUPTION MODEL TRAINING PIPELINE")
    logger.info("="*80)
    
    # 1. Load and prepare data
    logger.info("\n[STEP 1/6] Loading data...")
    loader = DataLoader("data/")
    loader.load_all()
    loader.explore_data()
    
    # 2. Prepare LSTM features
    logger.info("\n[STEP 2/6] Preparing LSTM features...")
    lstm_data = loader.prepare_lstm_features(sequence_length=30, forecast_horizon=5)
    logger.info(f"LSTM Training set size: {lstm_data['X_train'].shape}")
    
    # 3. Train LSTM model
    logger.info("\n[STEP 3/6] Training LSTM failure prediction model...")
    lstm_model = FailurePredictorLSTM(sequence_length=30, forecast_horizon=5)
    lstm_model.build_model(n_features=lstm_data['X_train'].shape[2])
    
    history = lstm_model.train(
        lstm_data['X_train'], lstm_data['y_train'],
        lstm_data['X_val'], lstm_data['y_val'],
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate LSTM
    logger.info("\nEvaluating LSTM on test set...")
    test_metrics = lstm_model.evaluate(lstm_data['X_test'], lstm_data['y_test'])
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1-Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # Save LSTM model
    os.makedirs("models", exist_ok=True)
    lstm_model.save("models/lstm_failure_model.h5")
    
    with open("models/lstm_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # 4. Prepare ARIMA features
    logger.info("\n[STEP 4/6] Preparing ARIMA features for delay forecasting...")
    arima_data = loader.prepare_arima_features()
    
    # 5. Train ARIMA models
    logger.info("\n[STEP 5/6] Training ARIMA delay forecasting models...")
    delay_model = DelayForecaster()
    arima_results = delay_model.fit(arima_data['delays_by_machine'])
    
    logger.info(f"Fitted ARIMA models for {len(arima_results)} machines")
    for machine_id, results in list(arima_results.items())[:5]:  # Log first 5
        logger.info(f"  Machine {machine_id}: ARIMA{results['order']} AIC={results['aic']:.2f}")
    
    delay_model.save("models/arima_models")
    
    with open("models/arima_results.json", "w") as f:
        # Convert to JSON serializable format
        json_results = {
            str(k): {
                'order': v['order'],
                'aic': float(v['aic']),
                'rmse': float(v['rmse'])
            }
            for k, v in arima_results.items()
        }
        json.dump(json_results, f, indent=2)
    
    # 6. Create and save RiskScorer
    logger.info("\n[STEP 6/6] Creating risk scorer...")
    machine_stats = loader.get_machine_statistics()
    machine_stats_dict = machine_stats.to_dict('index')
    
    risk_scorer = RiskScorer(failure_model=lstm_model, delay_model=delay_model)
    risk_scorer.set_machine_statistics(machine_stats_dict)
    
    with open("models/risk_scorer_config.pkl", "wb") as f:
        pickle.dump(risk_scorer.to_dict(), f)
    
    # 7. Create disruption handler
    handler = DisruptionHandler(
        failure_model=lstm_model,
        delay_model=delay_model,
        risk_scorer=risk_scorer
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE")
    logger.info("="*80)
    logger.info("\nSaved models:")
    logger.info("  - models/lstm_failure_model.h5")
    logger.info("  - models/lstm_metrics.json")
    logger.info("  - models/arima_models/ (directory)")
    logger.info("  - models/arima_results.json")
    logger.info("  - models/risk_scorer_config.pkl")
    logger.info("  - outputs/machine_statistics.csv")
    
    logger.info("\nNext steps:")
    logger.info("  1. Review model performance metrics")
    logger.info("  2. Run evaluate_predictor.py for detailed comparison")
    logger.info("  3. Integrate DisruptionHandler with RL scheduler")
    logger.info("  4. Test on sample scheduling scenarios")
    
    return {
        'lstm_model': lstm_model,
        'delay_model': delay_model,
        'risk_scorer': risk_scorer,
        'handler': handler,
        'lstm_metrics': test_metrics,
        'arima_results': arima_results
    }


if __name__ == "__main__":
    try:
        results = train_all_models()
        logger.info("\n✓ Training pipeline executed successfully")
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        exit(1)
