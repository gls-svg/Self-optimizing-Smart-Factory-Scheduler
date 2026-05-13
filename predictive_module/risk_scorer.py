"""
Risk Scoring System for Machine Health

Combines multiple signals to compute machine risk scores:
- Failure probability (from LSTM model)
- Current telemetry (temperature, vibration, utilization)
- Historical downtime
- Expected delays
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RiskScorer:
    """
    Multi-factor risk scorer for machines.
    
    Computes risk scores (0-1) based on:
    - Failure prediction (50% weight)
    - Telemetry thresholds (30% weight)
    - Historical reliability (20% weight)
    """
    
    def __init__(self, failure_model=None, delay_model=None):
        """
        Initialize risk scorer.
        
        Args:
            failure_model: Trained LSTM failure prediction model
            delay_model: Trained ARIMA delay forecasting model
        """
        self.failure_model = failure_model
        self.delay_model = delay_model
        
        # Thresholds for risk assessment
        self.telemetry_thresholds = {
            'temperature_high': 90,      # °C
            'temperature_critical': 95,  # °C
            'vibration_high': 0.8,       # m/s
            'vibration_critical': 0.95,  # m/s
            'utilization_high': 90,      # %
        }
        
        # Historical machine statistics
        self.machine_stats = {}
        
    def set_machine_statistics(self, stats: Dict):
        """
        Set historical machine statistics for context.
        
        Args:
            stats: Dict mapping machine_id -> {
                'failure_count': int,
                'avg_downtime': float,
                'avg_delay': float,
                'avg_temperature': float,
                'avg_vibration': float,
                'avg_utilization': float
            }
        """
        self.machine_stats = stats
        logger.info(f"✓ Set statistics for {len(stats)} machines")
    
    def calculate_telemetry_risk(self, machine_id: int, 
                                 temperature: float, vibration: float, 
                                 utilization: float) -> float:
        """
        Calculate risk from current telemetry readings.
        
        Args:
            machine_id: Machine ID
            temperature: Current temperature (°C)
            vibration: Current vibration level (m/s)
            utilization: Current utilization (%)
            
        Returns:
            float: Risk score (0-1)
        """
        risk = 0.0
        
        # Temperature risk (up to 0.4)
        if temperature > self.telemetry_thresholds['temperature_critical']:
            risk += 0.4
        elif temperature > self.telemetry_thresholds['temperature_high']:
            risk += 0.2
        elif machine_id in self.machine_stats:
            avg_temp = self.machine_stats[machine_id].get('avg_temperature', 0)
            if temperature > avg_temp + 15:  # 15°C above average
                risk += 0.1
        
        # Vibration risk (up to 0.4)
        if vibration > self.telemetry_thresholds['vibration_critical']:
            risk += 0.4
        elif vibration > self.telemetry_thresholds['vibration_high']:
            risk += 0.2
        elif machine_id in self.machine_stats:
            avg_vib = self.machine_stats[machine_id].get('avg_vibration', 0)
            if vibration > avg_vib + 0.3:  # 0.3 above average
                risk += 0.1
        
        # Utilization risk (up to 0.2)
        if utilization > self.telemetry_thresholds['utilization_high']:
            risk += 0.15
        
        return min(risk, 1.0)
    
    def calculate_historical_risk(self, machine_id: int) -> float:
        """
        Calculate risk based on historical reliability.
        
        Args:
            machine_id: Machine ID
            
        Returns:
            float: Risk score (0-1)
        """
        if machine_id not in self.machine_stats:
            return 0.0
        
        stats = self.machine_stats[machine_id]
        
        risk = 0.0
        
        # Failure frequency risk
        failure_count = stats.get('failure_count', 0)
        if failure_count > 10:
            risk += 0.6
        elif failure_count > 5:
            risk += 0.3
        elif failure_count > 2:
            risk += 0.1
        
        # Average downtime risk
        avg_downtime = stats.get('avg_downtime', 0)
        if avg_downtime > 100:
            risk += 0.3
        elif avg_downtime > 50:
            risk += 0.15
        
        return min(risk, 1.0)
    
    def calculate_machine_risk(self, machine_id: int,
                               temperature: float = None,
                               vibration: float = None,
                               utilization: float = None,
                               failure_probability: float = None) -> Tuple[float, Dict]:
        """
        Calculate comprehensive risk score for a machine.
        
        Args:
            machine_id: Machine ID
            temperature: Current temperature (°C)
            vibration: Current vibration level (m/s)
            utilization: Current utilization (%)
            failure_probability: Predicted failure probability from model
            
        Returns:
            tuple: (risk_score, details_dict)
        """
        # Weights: failure (50%), telemetry (30%), historical (20%)
        components = {}
        
        # 1. Failure prediction component (50%)
        if failure_probability is not None:
            components['failure_risk'] = float(failure_probability) * 0.5
        else:
            components['failure_risk'] = 0.0
        
        # 2. Telemetry component (30%)
        if temperature is not None and vibration is not None and utilization is not None:
            telemetry_risk = self.calculate_telemetry_risk(
                machine_id, temperature, vibration, utilization
            )
            components['telemetry_risk'] = telemetry_risk * 0.3
        else:
            components['telemetry_risk'] = 0.0
        
        # 3. Historical component (20%)
        historical_risk = self.calculate_historical_risk(machine_id)
        components['historical_risk'] = historical_risk * 0.2
        
        # Total risk
        total_risk = sum(components.values())
        total_risk = min(total_risk, 1.0)
        
        details = {
            'machine_id': machine_id,
            'risk_score': float(total_risk),
            'components': {k: float(v) for k, v in components.items()},
            'temperature': temperature,
            'vibration': vibration,
            'utilization': utilization,
            'failure_probability': failure_probability
        }
        
        return total_risk, details
    
    def get_risk_category(self, risk_score: float) -> str:
        """
        Categorize risk score.
        
        Args:
            risk_score: Score (0-1)
            
        Returns:
            str: 'LOW', 'MEDIUM', or 'HIGH'
        """
        if risk_score < 0.3:
            return 'LOW'
        elif risk_score < 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def forecast_delay(self, machine_id: int, steps: int = 1) -> np.ndarray:
        """
        Forecast expected delay using delay model.
        
        Args:
            machine_id: Machine ID
            steps: Number of operations ahead to forecast
            
        Returns:
            np.ndarray: Forecasted delays
        """
        if self.delay_model is None:
            return np.zeros(steps)
        
        return self.delay_model.forecast(machine_id, steps=steps)
    
    def get_machine_recommendation(self, risk_score: float, 
                                  failure_probability: float = None) -> str:
        """
        Get scheduling recommendation for a machine.
        
        Args:
            risk_score: Calculated risk score
            failure_probability: Failure probability
            
        Returns:
            str: 'SAFE', 'CAUTION', or 'AVOID'
        """
        category = self.get_risk_category(risk_score)
        
        if category == 'LOW':
            return 'SAFE'
        elif category == 'MEDIUM':
            # Additional check on failure probability
            if failure_probability is not None and failure_probability > 0.7:
                return 'AVOID'
            return 'CAUTION'
        else:  # HIGH
            return 'AVOID'
    
    def score_batch(self, machines: list, telemetry_df: pd.DataFrame) -> pd.DataFrame:
        """
        Score multiple machines at once.
        
        Args:
            machines: List of machine IDs
            telemetry_df: DataFrame with columns:
                machine_id, temperature, vibration, utilization
                
        Returns:
            pd.DataFrame: Risk scores and details for each machine
        """
        results = []
        
        for machine_id in machines:
            # Get latest telemetry
            machine_data = telemetry_df[telemetry_df['machine_id'] == machine_id]
            if len(machine_data) == 0:
                continue
            
            latest = machine_data.iloc[-1]
            
            risk_score, details = self.calculate_machine_risk(
                machine_id=machine_id,
                temperature=latest.get('temperature'),
                vibration=latest.get('vibration'),
                utilization=latest.get('utilization_percent'),
                failure_probability=latest.get('failure_probability', None)
            )
            
            details['risk_category'] = self.get_risk_category(risk_score)
            details['recommendation'] = self.get_machine_recommendation(
                risk_score, latest.get('failure_probability', None)
            )
            
            results.append(details)
        
        return pd.DataFrame(results)
    
    def to_dict(self) -> Dict:
        """Serialize scorer configuration"""
        return {
            'thresholds': self.telemetry_thresholds,
            'machine_stats': self.machine_stats
        }
    
    @staticmethod
    def from_dict(config: Dict):
        """Deserialize scorer from configuration"""
        scorer = RiskScorer()
        scorer.telemetry_thresholds = config.get('thresholds', scorer.telemetry_thresholds)
        scorer.machine_stats = config.get('machine_stats', {})
        return scorer
