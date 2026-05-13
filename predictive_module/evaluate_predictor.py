"""
Evaluation & Comparison of Predictive Models

Compares LSTM vs ARIMA performance and generates visualization reports.

Usage:
    python evaluate_predictor.py
"""

import os
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def evaluate_models():
    """Main evaluation pipeline"""
    
    from predictive_model import FailurePredictorLSTM, DelayForecaster
    from data_preparation import DataLoader
    
    logger.info("="*80)
    logger.info("MODEL EVALUATION & COMPARISON")
    logger.info("="*80)
    
    # Load trained models
    logger.info("\n[STEP 1/4] Loading trained models...")
    try:
        lstm_model = FailurePredictorLSTM.load("models/lstm_failure_model.h5")
        delay_model = DelayForecaster.load("models/arima_models")
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return
    
    # Load test data
    logger.info("\n[STEP 2/4] Loading test data...")
    loader = DataLoader("data/")
    loader.load_all()
    
    lstm_data = loader.prepare_lstm_features(sequence_length=30, forecast_horizon=5)
    X_test = lstm_data['X_test']
    y_test = lstm_data['y_test']
    
    # Evaluate LSTM
    logger.info("\n[STEP 3/4] Evaluating LSTM model...")
    eval_results = lstm_model.evaluate(X_test, y_test)
    
    logger.info("\n📊 LSTM FAILURE PREDICTION RESULTS")
    logger.info("-" * 50)
    logger.info(f"Accuracy:  {eval_results['accuracy']:.4f}")
    logger.info(f"Precision: {eval_results['precision']:.4f}")
    logger.info(f"Recall:    {eval_results['recall']:.4f}")
    logger.info(f"F1-Score:  {eval_results['f1_score']:.4f}")
    logger.info(f"AUC:       {eval_results['auc']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(eval_results['confusion_matrix'])
    
    # Generate predictions for visualization
    predictions = lstm_model.predict_failure_probability(X_test)
    pred_binary = (predictions > 0.5).astype(int)
    
    # Handle both 1D and 2D y_test arrays
    if y_test.ndim == 1:
        y_binary = (y_test > 0.5).astype(int)
    else:
        y_binary = (y_test.max(axis=1) > 0.5).astype(int)
    
    # 1. Confusion Matrix Plot
    logger.info("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Confusion Matrix
    cm = confusion_matrix(y_binary, pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('LSTM Confusion Matrix')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_binary, predictions)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve - LSTM Failure Prediction')
    axes[0, 1].legend(loc="lower right")
    
    # Probability Distribution
    axes[1, 0].hist(predictions[y_binary == 0], bins=30, alpha=0.6, label='No Failure', color='blue')
    axes[1, 0].hist(predictions[y_binary == 1], bins=30, alpha=0.6, label='Failure', color='red')
    axes[1, 0].axvline(0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Failure Probability Distribution')
    axes[1, 0].legend()
    
    # Performance metrics bar chart
    metrics = {
        'Accuracy': eval_results['accuracy'],
        'Precision': eval_results['precision'],
        'Recall': eval_results['recall'],
        'F1-Score': eval_results['f1_score'],
        'AUC': eval_results['auc']
    }
    axes[1, 1].bar(metrics.keys(), metrics.values(), color='skyblue', edgecolor='navy')
    axes[1, 1].set_ylim([0, 1.0])
    axes[1, 1].set_title('LSTM Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    for i, (k, v) in enumerate(metrics.items()):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/lstm_evaluation.png", dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: outputs/lstm_evaluation.png")
    plt.close()
    
    # Evaluate ARIMA
    logger.info("\n[STEP 4/4] Evaluating ARIMA model...")
    
    arima_data = loader.prepare_arima_features()
    
    # Split delays into train/test
    test_size = 0.2
    X_test_arima = {}
    for machine_id, delays in arima_data['delays_by_machine'].items():
        split_idx = int(len(delays) * (1 - test_size))
        X_test_arima[machine_id] = delays[split_idx:]
    
    arima_eval = delay_model.evaluate(X_test_arima)
    
    logger.info("\n📊 ARIMA DELAY FORECASTING RESULTS")
    logger.info("-" * 50)
    
    rmse_values = [v['rmse'] for v in arima_eval.values()]
    mae_values = [v['mae'] for v in arima_eval.values()]
    
    logger.info(f"Number of machines evaluated: {len(arima_eval)}")
    logger.info(f"Average RMSE: {np.mean(rmse_values):.4f}")
    logger.info(f"Average MAE:  {np.mean(mae_values):.4f}")
    logger.info(f"RMSE range:   [{min(rmse_values):.4f}, {max(rmse_values):.4f}]")
    logger.info(f"MAE range:    [{min(mae_values):.4f}, {max(mae_values):.4f}]")
    
    # ARIMA evaluation plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    machines_list = list(arima_eval.keys())[:20]  # Top 20 machines
    rmse_list = [arima_eval[m]['rmse'] for m in machines_list]
    mae_list = [arima_eval[m]['mae'] for m in machines_list]
    
    axes[0].bar(range(len(machines_list)), rmse_list, color='coral', edgecolor='darkred')
    axes[0].set_xlabel('Machine ID')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('ARIMA Delay Forecast RMSE by Machine')
    axes[0].set_xticks(range(len(machines_list)))
    axes[0].set_xticklabels(machines_list, rotation=45)
    
    axes[1].bar(range(len(machines_list)), mae_list, color='lightgreen', edgecolor='darkgreen')
    axes[1].set_xlabel('Machine ID')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('ARIMA Delay Forecast MAE by Machine')
    axes[1].set_xticks(range(len(machines_list)))
    axes[1].set_xticklabels(machines_list, rotation=45)
    
    plt.tight_layout()
    plt.savefig("outputs/arima_evaluation.png", dpi=300, bbox_inches='tight')
    logger.info("✓ Saved: outputs/arima_evaluation.png")
    plt.close()
    
    # Generate comparison report
    logger.info("\n📝 Generating comparison report...")
    
    report = {
        'timestamp': str(pd.Timestamp.now()),
        'lstm_results': {
            'accuracy': float(eval_results['accuracy']),
            'precision': float(eval_results['precision']),
            'recall': float(eval_results['recall']),
            'f1_score': float(eval_results['f1_score']),
            'auc': float(eval_results['auc']),
            'test_samples': len(X_test)
        },
        'arima_results': {
            'avg_rmse': float(np.mean(rmse_values)),
            'avg_mae': float(np.mean(mae_values)),
            'machines_evaluated': len(arima_eval),
            'rmse_range': [float(min(rmse_values)), float(max(rmse_values))],
            'mae_range': [float(min(mae_values)), float(max(mae_values))]
        },
        'comparison': {
            'lstm_strengths': [
                'Captures temporal dependencies',
                'Multi-step ahead predictions',
                'Handles nonlinear patterns'
            ],
            'lstm_weaknesses': [
                'Requires sufficient training data',
                'Slower inference',
                'Black-box model'
            ],
            'arima_strengths': [
                'Fast inference',
                'Interpretable parameters',
                'Good for stationary series'
            ],
            'arima_weaknesses': [
                'Assumes stationarity',
                'Separate model per machine',
                'Single-step forecasting'
            ]
        },
        'recommendations': [
            'Use LSTM for failure prediction (better for multivariate patterns)',
            'Use ARIMA for delay forecasting (simpler, faster)',
            'Ensemble approach: combine both for robust predictions',
            'Monitor both models in production',
            'Retrain monthly with new data'
        ]
    }
    
    with open("outputs/model_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info("✓ Saved: outputs/model_comparison_report.json")
    
    logger.info("\n" + "="*80)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info("\nGenerated files:")
    logger.info("  - outputs/lstm_evaluation.png")
    logger.info("  - outputs/arima_evaluation.png")
    logger.info("  - outputs/model_comparison_report.json")
    
    return report


if __name__ == "__main__":
    try:
        report = evaluate_models()
        logger.info("\n✓ Evaluation completed successfully")
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}", exc_info=True)
        exit(1)
