"""
Disruption Handler - Main Interface

Coordinates LSTM failure prediction and ARIMA delay forecasting
to provide actionable disruption alerts to the scheduler.

This is the primary interface that Member 2 (RL Scheduler) will use.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class DisruptionHandler:
    """
    Central coordinator for predictive disruption handling.
    
    Provides unified interface to:
    - Predict machine failures before they happen
    - Forecast processing delays
    - Generate risk scores
    - Recommend safe/unsafe machines for job assignment
    - Trigger proactive rescheduling
    """
    
    def __init__(self, failure_model=None, delay_model=None, risk_scorer=None):
        """
        Initialize disruption handler.
        
        Args:
            failure_model: Trained LSTM failure prediction model
            delay_model: Trained ARIMA delay forecasting model
            risk_scorer: Configured RiskScorer instance
        """
        # If no models provided, try to load from disk
        if failure_model is None or delay_model is None or risk_scorer is None:
            self._load_trained_models()
        
        self.failure_model = failure_model or self.failure_model
        self.delay_model = delay_model or self.delay_model
        self.risk_scorer = risk_scorer or self.risk_scorer
        
        # Alert history for tracking
        self.alert_history = []
        self.prediction_cache = {}
    
    def _load_trained_models(self):
        """Load trained models from disk if available"""
        try:
            from predictive_model import FailurePredictorLSTM, DelayForecaster
            from risk_scorer import RiskScorer
            import pickle
            
            # Try multiple possible paths
            possible_paths = [
                Path('../models'),          # From predictive_module
                Path('./models'),           # From project root
                Path('models'),             # Current directory
            ]
            
            models_dir = None
            for path in possible_paths:
                if path.exists():
                    models_dir = path
                    break
            
            if models_dir is None:
                logger.warning("Models directory not found. Predictions will be unavailable.")
                self.failure_model = None
                self.delay_model = None
                self.risk_scorer = None
                return
            
            # Try to load LSTM model
            lstm_path = models_dir / 'lstm_failure_model.h5'
            if lstm_path.exists():
                self.failure_model = FailurePredictorLSTM.load(str(lstm_path))
                logger.info("Loaded LSTM failure model")
            else:
                self.failure_model = None
                logger.warning(f"LSTM model not found at {lstm_path}")
            
            # Try to load ARIMA models
            arima_dir = models_dir / 'arima_models'
            if arima_dir.exists():
                self.delay_model = DelayForecaster.load(str(arima_dir))
                logger.info("Loaded ARIMA delay models")
            else:
                self.delay_model = None
                logger.warning(f"ARIMA models not found at {arima_dir}")
            
            # Try to load RiskScorer config
            risk_scorer_path = models_dir / 'risk_scorer_config.pkl'
            if risk_scorer_path.exists():
                with open(risk_scorer_path, 'rb') as f:
                    config = pickle.load(f)
                
                # If it's a dict, use from_dict; otherwise use directly
                if isinstance(config, dict):
                    self.risk_scorer = RiskScorer.from_dict(config)
                else:
                    self.risk_scorer = config
                logger.info("Loaded RiskScorer configuration")
            else:
                self.risk_scorer = None
                logger.warning(f"RiskScorer config not found at {risk_scorer_path}")
                
        except Exception as e:
            logger.warning(f"Could not load trained models: {e}")
            self.failure_model = None
            self.delay_model = None
            self.risk_scorer = None
        
    def predict_for_machine(self, machine_id: int, current_time: datetime = None,
                           current_telemetry: Dict = None) -> Dict:
        """
        Get complete prediction for a machine.
        
        This is the PRIMARY METHOD called by the RL scheduler before assigning jobs.
        
        Args:
            machine_id: Machine to evaluate
            current_time: Current timestamp (for logging)
            current_telemetry: Dict with keys:
                - temperature (°C)
                - vibration (m/s)
                - utilization_percent (%)
                
        Returns:
            Dict with keys:
            {
                'machine_id': int,
                'failure_probability': float (0-1),
                'risk_score': float (0-1),
                'risk_category': str ('LOW', 'MEDIUM', 'HIGH'),
                'predicted_delay': float (minutes),
                'estimated_downtime': float (minutes),
                'recommendation': str ('SAFE', 'CAUTION', 'AVOID'),
                'confidence': float (0-1),
                'timestamp': datetime,
                'details': dict
            }
        """
        current_time = current_time or datetime.now()
        
        # Start with empty prediction
        prediction = {
            'machine_id': machine_id,
            'timestamp': current_time,
            'failure_probability': 0.0,
            'risk_score': 0.0,
            'risk_category': 'UNKNOWN',
            'predicted_delay': 0.0,
            'estimated_downtime': 0.0,
            'recommendation': 'SAFE',
            'confidence': 0.0,
            'details': {}
        }
        
        # 1. Get failure probability from LSTM
        if self.failure_model is not None and current_telemetry is not None:
            try:
                # Assume current_telemetry contains a sequence-length window of data
                failure_prob = self._get_failure_probability(machine_id, current_telemetry)
                prediction['failure_probability'] = failure_prob
            except Exception as e:
                logger.warning(f"Failed to get failure probability for machine {machine_id}: {e}")
        
        # 2. Get risk score from scorer
        if self.risk_scorer is not None:
            try:
                risk_score, details = self.risk_scorer.calculate_machine_risk(
                    machine_id=machine_id,
                    temperature=current_telemetry.get('temperature') if current_telemetry else None,
                    vibration=current_telemetry.get('vibration') if current_telemetry else None,
                    utilization=current_telemetry.get('utilization_percent') if current_telemetry else None,
                    failure_probability=prediction['failure_probability']
                )
                
                prediction['risk_score'] = risk_score
                prediction['risk_category'] = self.risk_scorer.get_risk_category(risk_score)
                prediction['recommendation'] = self.risk_scorer.get_machine_recommendation(
                    risk_score, prediction['failure_probability']
                )
                prediction['details'] = details
                prediction['confidence'] = 0.8  # Moderate confidence when using both models
                
            except Exception as e:
                logger.warning(f"Failed to calculate risk score for machine {machine_id}: {e}")
        
        # 3. Get delay forecast from ARIMA
        if self.delay_model is not None:
            try:
                delay_forecast = self.delay_model.forecast(machine_id, steps=1)
                if len(delay_forecast) > 0:
                    prediction['predicted_delay'] = float(delay_forecast[0])
            except Exception as e:
                logger.warning(f"Failed to forecast delay for machine {machine_id}: {e}")
        
        # 4. Estimate downtime if failure is likely
        if prediction['failure_probability'] > 0.6:
            if self.risk_scorer and machine_id in self.risk_scorer.machine_stats:
                stats = self.risk_scorer.machine_stats[machine_id]
                prediction['estimated_downtime'] = stats.get('avg_downtime', 50)
        
        return prediction
    
    def _get_failure_probability(self, machine_id: int, current_telemetry: Dict) -> float:
        """
        Extract failure probability from LSTM model.
        
        Args:
            machine_id: Machine ID
            current_telemetry: Telemetry dict or sequence
            
        Returns:
            float: Failure probability (0-1)
        """
        if self.failure_model is None:
            return 0.0
        
        # If telemetry is already a sequence, use it; otherwise wrap it
        if isinstance(current_telemetry, dict):
            # Try to extract sequence from dict
            if 'sequence' in current_telemetry:
                X = current_telemetry['sequence']
            else:
                # Single point - cannot use LSTM without sequence
                logger.warning("Single telemetry point cannot be used for LSTM prediction")
                return 0.0
        else:
            X = current_telemetry
        
        # Ensure proper shape
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=0)
        
        # Predict
        prob = self.failure_model.predict_failure_probability(X)[0]
        return float(prob)
    
    def get_safe_machines(self, machine_list: List[int], 
                         current_telemetry_df: pd.DataFrame = None,
                         allow_caution: bool = False) -> List[int]:
        """
        Filter machines to only safe ones.
        
        Useful for constraining job assignment to only safe machines.
        
        Args:
            machine_list: List of candidate machine IDs
            current_telemetry_df: DataFrame with current telemetry (optional)
            allow_caution: If True, include CAUTION machines; if False, only SAFE
            
        Returns:
            List[int]: Filtered list of safe machines
        """
        safe_machines = []
        
        for machine_id in machine_list:
            prediction = self.predict_for_machine(machine_id, current_telemetry={} if current_telemetry_df is None 
                                                  else current_telemetry_df.get(machine_id, {}))
            
            category = prediction['risk_category']
            
            if category == 'LOW':
                safe_machines.append(machine_id)
            elif category == 'MEDIUM' and allow_caution:
                safe_machines.append(machine_id)
            # else: skip HIGH risk machines
        
        if len(safe_machines) == 0:
            # Fallback: return least risky machine
            logger.warning(f"No safe machines found for {machine_list}. Returning least risky.")
            predictions = [(mid, self.predict_for_machine(mid)) for mid in machine_list]
            safe_machines = [min(predictions, key=lambda x: x[1]['risk_score'])[0]]
        
        return safe_machines
    
    def trigger_predictive_alert(self, machine_id: int, prediction: Dict) -> bool:
        """
        Trigger alert for high-risk machine.
        
        Called when risk exceeds thresholds. Can trigger rescheduling.
        
        Args:
            machine_id: Machine ID
            prediction: Prediction dict from predict_for_machine()
            
        Returns:
            bool: True if alert should trigger rescheduling
        """
        risk_category = prediction['risk_category']
        failure_prob = prediction['failure_probability']
        
        should_reschedule = False
        alert_level = 'INFO'
        
        if risk_category == 'HIGH' and failure_prob > 0.7:
            alert_level = 'CRITICAL'
            should_reschedule = True
        elif risk_category == 'HIGH':
            alert_level = 'WARNING'
            should_reschedule = True
        elif risk_category == 'MEDIUM' and failure_prob > 0.8:
            alert_level = 'WARNING'
            should_reschedule = True
        
        # Log alert
        alert = {
            'timestamp': prediction['timestamp'],
            'machine_id': machine_id,
            'alert_level': alert_level,
            'risk_category': risk_category,
            'failure_probability': failure_prob,
            'risk_score': prediction['risk_score'],
            'reschedule_recommended': should_reschedule
        }
        
        self.alert_history.append(alert)
        
        msg = f"[{alert_level}] Machine {machine_id}: Risk={risk_category}, P(failure)={failure_prob:.2f}, Score={prediction['risk_score']:.2f}"
        if should_reschedule:
            msg += " | RESCHEDULING RECOMMENDED"
            logger.warning(msg)
        else:
            logger.info(msg)
        
        return should_reschedule
    
    def compare_machines(self, machine_list: List[int],
                        current_telemetry_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare risk across multiple machines.
        
        Useful for scheduler to evaluate alternatives.
        
        Args:
            machine_list: List of machine IDs to compare
            current_telemetry_df: Current telemetry data
            
        Returns:
            pd.DataFrame: Comparison table with risk scores, delays, recommendations
        """
        predictions = []
        
        for machine_id in machine_list:
            pred = self.predict_for_machine(machine_id)
            predictions.append({
                'machine_id': machine_id,
                'failure_prob': pred['failure_probability'],
                'risk_score': pred['risk_score'],
                'risk_category': pred['risk_category'],
                'predicted_delay': pred['predicted_delay'],
                'downtime_est': pred['estimated_downtime'],
                'recommendation': pred['recommendation']
            })
        
        return pd.DataFrame(predictions).sort_values('risk_score')
    
    def get_alert_summary(self) -> Dict:
        """
        Get summary of recent alerts.
        
        Returns:
            dict: Alert statistics
        """
        if not self.alert_history:
            return {
                'total_alerts': 0,
                'critical_alerts': 0,
                'warning_alerts': 0,
                'reschedule_triggers': 0
            }
        
        alerts_df = pd.DataFrame(self.alert_history)
        
        return {
            'total_alerts': len(alerts_df),
            'critical_alerts': (alerts_df['alert_level'] == 'CRITICAL').sum(),
            'warning_alerts': (alerts_df['alert_level'] == 'WARNING').sum(),
            'reschedule_triggers': alerts_df['reschedule_recommended'].sum(),
            'machines_with_alerts': alerts_df['machine_id'].nunique(),
            'recent_alert': alerts_df.iloc[-1].to_dict() if len(alerts_df) > 0 else None
        }
    
    def save_predictions_log(self, filepath: str):
        """Save alert history to CSV"""
        if self.alert_history:
            df = pd.DataFrame(self.alert_history)
            df.to_csv(filepath, index=False)
            logger.info(f"✓ Saved {len(df)} alerts to {filepath}")
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("✓ Prediction cache cleared")


# Example integration with Member 2's scheduler
def example_integration():
    """
    Example of how Member 2 (RL Scheduler) would use this handler.
    """
    # Initialize
    handler = DisruptionHandler(failure_model=None, delay_model=None, risk_scorer=None)
    
    # Before assigning job to machines [0, 1, 2]
    candidate_machines = [0, 1, 2]
    
    # Get predictions for each candidate
    for machine_id in candidate_machines:
        pred = handler.predict_for_machine(machine_id)
        print(f"Machine {machine_id}: {pred['recommendation']}")
    
    # Filter to safe machines only
    safe_machines = handler.get_safe_machines(candidate_machines)
    print(f"Safe machines for assignment: {safe_machines}")
    
    # If no safe machines, compare and pick least risky
    comparison = handler.compare_machines(candidate_machines)
    print(comparison)
    
    # Scheduler can then assign to best machine from safe_machines
    # If any failures predicted, trigger rescheduling
    for machine_id in candidate_machines:
        pred = handler.predict_for_machine(machine_id)
        should_reschedule = handler.trigger_predictive_alert(machine_id, pred)
        if should_reschedule:
            print(f"Reschedule triggered for machine {machine_id}")


if __name__ == "__main__":
    example_integration()
