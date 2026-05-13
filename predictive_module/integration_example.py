"""
Integration Guide: Using DisruptionHandler in RL Scheduler
===========================================================

This module shows how Member 2 (RL Scheduler) can integrate the 
Predictive Disruption Module into the JSSP scheduling pipeline.

Author: Member 4
Purpose: Enable safer scheduling decisions with failure/delay predictions
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add predictive module to path
sys.path.insert(0, str(Path(__file__).parent / 'predictive_module'))

from disruption_handler import DisruptionHandler


class EnhancedJSSPScheduler:
    """
    Enhanced JSSP Scheduler with predictive disruption awareness.
    
    Integrates the DisruptionHandler to:
    1. Predict failures before assignment
    2. Forecast delays
    3. Score risks
    4. Make safer scheduling decisions
    """
    
    def __init__(self, strict_safety=True):
        """
        Initialize enhanced scheduler.
        
        Args:
            strict_safety: If True, avoid MEDIUM/HIGH risk machines
        """
        self.disruption_handler = DisruptionHandler()
        self.strict_safety = strict_safety
        
    def get_machine_prediction(self, machine_id: int):
        """
        Get failure/delay predictions for a machine.
        
        Args:
            machine_id: Machine to evaluate
            
        Returns:
            dict: Complete prediction with failure_prob, risk_score, etc.
        """
        # DisruptionHandler returns all predictions in one call
        prediction = self.disruption_handler.predict_for_machine(machine_id)
        return prediction
    
    def filter_safe_machines(self, available_machines: list):
        """
        Filter machines to only safe candidates.
        
        Args:
            available_machines: List of machine IDs that could process job
            
        Returns:
            list: Filtered list of safe machines, ranked by risk
        """
        if not available_machines:
            return []
        
        # Get predictions for all candidates
        predictions = {}
        for mid in available_machines:
            pred = self.disruption_handler.predict_for_machine(mid)
            predictions[mid] = pred
        
        # Filter based on risk category
        if self.strict_safety:
            # Only allow LOW risk or UNKNOWN (no MEDIUM/HIGH)
            safe_machines = [
                mid for mid, pred in predictions.items()
                if pred['risk_category'] in ['LOW', 'UNKNOWN']
            ]
        else:
            # Allow LOW and MEDIUM (avoid only HIGH)
            safe_machines = [
                mid for mid, pred in predictions.items()
                if pred['risk_category'] != 'HIGH'
            ]
        
        # If no safe machines, return all sorted by risk
        if not safe_machines:
            return sorted(
                available_machines,
                key=lambda m: predictions[m]['risk_score']
            )
        
        # Sort safe machines by risk score (best first)
        return sorted(
            safe_machines,
            key=lambda m: predictions[m]['risk_score']
        )
    
    def schedule_job(self, job_id: int, operation_id: int, 
                    available_machines: list):
        """
        Schedule a job operation on best available machine.
        
        Args:
            job_id: Job to schedule
            operation_id: Operation number within job
            available_machines: Candidate machines
            
        Returns:
            dict: Selected machine and prediction details
        """
        # Get safe machines ranked by risk
        safe_machines = self.filter_safe_machines(available_machines)
        
        if not safe_machines:
            return {
                'success': False,
                'message': 'No machines available'
            }
        
        # Select best machine (lowest risk)
        selected_machine = safe_machines[0]
        prediction = self.disruption_handler.predict_for_machine(selected_machine)
        
        return {
            'success': True,
            'selected_machine': selected_machine,
            'failure_probability': prediction['failure_probability'],
            'risk_category': prediction['risk_category'],
            'predicted_delay': prediction['predicted_delay'],
            'confidence': prediction['confidence'],
            'recommendation': prediction['recommendation']
        }
    
    def compare_alternative_schedules(self, machine_options: dict):
        """
        Compare risk across different scheduling options.
        
        Useful for batch scheduling decisions.
        
        Args:
            machine_options: Dict mapping job_id -> list of machines
            
        Returns:
            dict: Comparison with risk scores
        """
        comparison = {}
        
        for job_id, machines in machine_options.items():
            safe_machines = self.filter_safe_machines(machines)
            
            if not safe_machines:
                comparison[job_id] = {
                    'recommended': None,
                    'reason': 'No safe machines available'
                }
            else:
                best_machine = safe_machines[0]
                pred = self.disruption_handler.predict_for_machine(best_machine)
                
                comparison[job_id] = {
                    'recommended': best_machine,
                    'risk_score': pred['risk_score'],
                    'failure_prob': pred['failure_probability'],
                    'rank_in_pool': safe_machines.index(best_machine) + 1,
                    'pool_size': len(machines)
                }
        
        return comparison


# Example Usage
if __name__ == '__main__':
    print("=" * 80)
    print("MEMBER 4 PREDICTIVE MODULE - INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # Initialize enhanced scheduler
    scheduler = EnhancedJSSPScheduler(strict_safety=True)
    print("[OK] Enhanced scheduler initialized with strict safety mode")
    
    # Example 1: Get machine prediction
    print("\n[EXAMPLE 1] Get prediction for a specific machine")
    print("-" * 50)
    machine_id = 3
    pred = scheduler.get_machine_prediction(machine_id)
    print(f"Machine {machine_id}:")
    print(f"  Failure Probability: {pred['failure_probability']:.4f}")
    print(f"  Risk Score: {pred['risk_score']:.4f}")
    print(f"  Risk Category: {pred['risk_category']}")
    print(f"  Recommendation: {pred['recommendation']}")
    
    # Example 2: Filter safe machines for job scheduling
    print("\n[EXAMPLE 2] Filter safe machines from pool of candidates")
    print("-" * 50)
    available_machines = [0, 3, 5, 7, 10]
    safe_machines = scheduler.filter_safe_machines(available_machines)
    print(f"Available machines: {available_machines}")
    print(f"Safe machines (ranked): {safe_machines}")
    
    # Example 3: Schedule a job operation
    print("\n[EXAMPLE 3] Schedule job operation on best machine")
    print("-" * 50)
    result = scheduler.schedule_job(
        job_id=1,
        operation_id=1,
        available_machines=available_machines
    )
    if result['success']:
        print(f"[OK] Job scheduled on Machine {result['selected_machine']}")
        print(f"  Risk Category: {result['risk_category']}")
        print(f"  Failure Prob: {result['failure_probability']:.4f}")
        print(f"  Predicted Delay: {result['predicted_delay']:.2f} minutes")
    
    # Example 4: Compare multiple scheduling options
    print("\n[EXAMPLE 4] Compare risk across multiple scheduling options")
    print("-" * 50)
    options = {
        'job_1': [0, 1, 2],
        'job_2': [3, 4, 5],
        'job_3': [6, 7, 8]
    }
    comparison = scheduler.compare_alternative_schedules(options)
    for job_id, result in comparison.items():
        if result['recommended'] is not None:
            print(f"{job_id}: Recommend Machine {result['recommended']} "
                  f"(Risk={result['risk_score']:.4f})")
        else:
            print(f"{job_id}: {result['reason']}")
    
    print("\n" + "=" * 80)
    print("[OK] Integration example completed successfully")
    print("=" * 80)
    print("\nNOTE: In production, integrate this with jssp_env.py step() method")
    print("See enhanced_jssp_env.py for full implementation example")
