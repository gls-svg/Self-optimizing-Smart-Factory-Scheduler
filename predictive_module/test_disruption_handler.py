"""
Test DisruptionHandler integration with trained models

Usage:
    python test_disruption_handler.py
"""

import logging
import numpy as np
from disruption_handler import DisruptionHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_disruption_handler():
    """Test DisruptionHandler functionality"""
    
    logger.info("=" * 80)
    logger.info("DISRUPTION HANDLER INTEGRATION TEST")
    logger.info("=" * 80)
    
    # Initialize DisruptionHandler
    logger.info("\n[TEST 1] Initializing DisruptionHandler...")
    try:
        handler = DisruptionHandler()
        logger.info("✓ DisruptionHandler initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize: {e}")
        return
    
    # Test prediction for a single machine
    logger.info("\n[TEST 2] Testing prediction_for_machine()...")
    try:
        machine_id = 0
        current_time = 100
        current_telemetry = {
            'utilization': 85.5,
            'temperature': 75.0,
            'vibration': 0.55
        }
        
        prediction = handler.predict_for_machine(
            machine_id, current_time, current_telemetry
        )
        
        logger.info(f"✓ Prediction for Machine {machine_id}:")
        logger.info(f"  - Failure Probability: {prediction['failure_probability']:.4f}")
        logger.info(f"  - Risk Score: {prediction['risk_score']:.4f}")
        logger.info(f"  - Risk Category: {prediction['risk_category']}")
        logger.info(f"  - Recommendation: {prediction['recommendation']}")
        logger.info(f"  - Predicted Delay: {prediction['predicted_delay']:.2f}")
        logger.info(f"  - Estimated Downtime: {prediction['estimated_downtime']:.2f}")
        logger.info(f"  - Confidence: {prediction['confidence']:.4f}")
        
    except Exception as e:
        logger.error(f"✗ Prediction failed: {e}")
        return
    
    # Test filtering safe machines
    logger.info("\n[TEST 3] Testing get_safe_machines()...")
    try:
        machine_list = [0, 1, 2, 3, 4]
        safe_machines = handler.get_safe_machines(machine_list, allow_caution=False)
        logger.info(f"✓ Safe machines (strict): {safe_machines}")
        
        safe_machines_caution = handler.get_safe_machines(machine_list, allow_caution=True)
        logger.info(f"✓ Safe machines (with caution): {safe_machines_caution}")
        
    except Exception as e:
        logger.error(f"✗ Safe machines filtering failed: {e}")
        return
    
    # Test comparing machines
    logger.info("\n[TEST 4] Testing compare_machines()...")
    try:
        machine_list = [0, 1, 2, 3, 4]
        comparison = handler.compare_machines(machine_list)
        logger.info("✓ Machine ranking by risk (best to worst):")
        for idx, row in comparison.iterrows():
            logger.info(f"  Machine {int(row['machine_id'])}: Risk={row['risk_score']:.4f}, "
                       f"Category={row['risk_category']}, Recommendation={row['recommendation']}")
        
    except Exception as e:
        logger.error(f"✗ Machine comparison failed: {e}")
        return
    
    # Test batch predictions
    logger.info("\n[TEST 5] Testing batch predictions for all machines...")
    try:
        all_machines = list(range(15))
        predictions = {}
        
        for mid in all_machines:
            pred = handler.predict_for_machine(
                mid, 100,
                {
                    'utilization': 85.5,
                    'temperature': 75.0,
                    'vibration': 0.55
                }
            )
            predictions[mid] = pred
        
        logger.info(f"✓ Generated predictions for {len(predictions)} machines")
        
        # Summary statistics
        failure_probs = [p['failure_probability'] for p in predictions.values()]
        risk_scores = [p['risk_score'] for p in predictions.values()]
        
        logger.info(f"  - Avg Failure Probability: {np.mean(failure_probs):.4f}")
        logger.info(f"  - Avg Risk Score: {np.mean(risk_scores):.4f}")
        logger.info(f"  - Failure Prob Range: [{np.min(failure_probs):.4f}, {np.max(failure_probs):.4f}]")
        logger.info(f"  - Risk Score Range: [{np.min(risk_scores):.4f}, {np.max(risk_scores):.4f}]")
        
    except Exception as e:
        logger.error(f"✗ Batch prediction failed: {e}")
        return
    
    # Test alerts
    logger.info("\n[TEST 6] Testing alert system...")
    try:
        # Trigger some alerts
        for mid in [0, 2, 5]:
            pred = handler.predict_for_machine(
                mid, 100,
                {
                    'utilization': 85.5,
                    'temperature': 75.0,
                    'vibration': 0.55
                }
            )
            handler.trigger_predictive_alert(mid, pred)
        
        alert_summary = handler.get_alert_summary()
        logger.info("✓ Alert Summary:")
        logger.info(f"  - Total Alerts: {alert_summary['total_alerts']}")
        logger.info(f"  - High Risk Alerts: {alert_summary['high_risk_count']}")
        logger.info(f"  - Medium Risk Alerts: {alert_summary['medium_risk_count']}")
        logger.info(f"  - Most Recent Alert Time: {alert_summary['last_alert_time']}")
        
    except Exception as e:
        logger.error(f"✗ Alert system failed: {e}")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL INTEGRATION TESTS PASSED")
    logger.info("=" * 80)


if __name__ == '__main__':
    test_disruption_handler()
