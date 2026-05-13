"""
Data Preparation Module for Predictive Disruption Handling

This module loads, explores, and prepares data for LSTM and ARIMA models.

Usage:
    python data_preparation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and explore datasets from Member 1"""
    
    def __init__(self, data_dir="data/"):
        self.data_dir = Path(data_dir)
        self.machine_history = None
        self.machine_failures = None
        self.delays = None
        self.master_schedule = None
        
    def load_all(self):
        """Load all required datasets"""
        logger.info("Loading datasets...")
        
        try:
            self.machine_history = pd.read_csv(self.data_dir / "machine_history_dataset.csv")
            logger.info(f"✓ Machine History: {len(self.machine_history)} rows")
            
            self.machine_failures = pd.read_csv(self.data_dir / "machine_failure_dataset.csv")
            logger.info(f"✓ Machine Failures: {len(self.machine_failures)} rows")
            
            self.delays = pd.read_csv(self.data_dir / "delay_dataset.csv")
            logger.info(f"✓ Delays: {len(self.delays)} rows")
            
            self.master_schedule = pd.read_csv(self.data_dir / "master_schedule_dataset.csv")
            logger.info(f"✓ Master Schedule: {len(self.master_schedule)} rows")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        
        return self
    
    def explore_data(self):
        """Print exploratory data analysis"""
        logger.info("\n" + "="*80)
        logger.info("DATA EXPLORATION REPORT")
        logger.info("="*80)
        
        # Machine History
        logger.info("\n📊 MACHINE HISTORY DATASET")
        logger.info(f"Shape: {self.machine_history.shape}")
        logger.info(f"Columns: {list(self.machine_history.columns)}")
        logger.info(f"Date Range: {self.machine_history['timestamp'].min()} to {self.machine_history['timestamp'].max()}")
        logger.info(f"Unique Machines: {self.machine_history['machine_id'].nunique()}")
        logger.info(f"Missing Values:\n{self.machine_history.isnull().sum()}")
        logger.info(f"\nSample:\n{self.machine_history.head()}")
        
        # Machine Failures
        logger.info("\n📊 MACHINE FAILURES DATASET")
        logger.info(f"Shape: {self.machine_failures.shape}")
        logger.info(f"Columns: {list(self.machine_failures.columns)}")
        logger.info(f"Failure Types: {self.machine_failures['failure_type'].unique()}")
        logger.info(f"Avg Downtime: {self.machine_failures['downtime_minutes'].mean():.2f} minutes")
        logger.info(f"Missing Values:\n{self.machine_failures.isnull().sum()}")
        
        # Delays
        logger.info("\n📊 DELAYS DATASET")
        logger.info(f"Shape: {self.delays.shape}")
        logger.info(f"Columns: {list(self.delays.columns)}")
        logger.info(f"Avg Delay: {self.delays['delay'].mean():.2f}")
        logger.info(f"Max Delay: {self.delays['delay'].max():.2f}")
        logger.info(f"Min Delay: {self.delays['delay'].min():.2f}")
        logger.info(f"Delay Range by Machine:\n{self.delays.groupby('machine_id')['delay'].agg(['mean', 'max', 'min']).head(10)}")
        
        logger.info("\n" + "="*80)
    
    def prepare_lstm_features(self, sequence_length=30, forecast_horizon=5):
        """
        Prepare data for LSTM training.
        
        Creates sequences from machine history telemetry.
        
        Args:
            sequence_length: Number of timesteps to look back
            forecast_horizon: Number of steps ahead to predict failures
            
        Returns:
            dict: {
                'X_train': np.array of shape (samples, sequence_length, features),
                'y_train': np.array of shape (samples, forecast_horizon),
                'X_val': validation data,
                'y_val': validation labels,
                'machines': list of machine IDs,
                'feature_names': list of feature names
            }
        """
        logger.info(f"\n🔧 Preparing LSTM features (seq_len={sequence_length}, horizon={forecast_horizon})")
        
        # Convert timestamp to datetime
        self.machine_history['timestamp'] = pd.to_datetime(self.machine_history['timestamp'])
        
        # Sort by machine and timestamp
        self.machine_history = self.machine_history.sort_values(['machine_id', 'timestamp'])
        
        # Create failure labels
        self.machine_failures['timestamp'] = pd.to_datetime(self.machine_failures['timestamp'])
        failure_machines = set(self.machine_failures['machine_id'].unique())
        
        X_list = []
        y_list = []
        machines_list = []
        
        feature_cols = ['utilization_percent', 'temperature', 'vibration']
        
        for machine_id in self.machine_history['machine_id'].unique():
            machine_data = self.machine_history[self.machine_history['machine_id'] == machine_id].copy()
            
            if len(machine_data) < sequence_length + forecast_horizon:
                continue
            
            # Normalize features
            for col in feature_cols:
                machine_data[col] = (machine_data[col] - machine_data[col].mean()) / (machine_data[col].std() + 1e-5)
            
            features = machine_data[feature_cols].values
            
            # Create sliding windows
            for i in range(len(features) - sequence_length - forecast_horizon + 1):
                X = features[i:i+sequence_length]
                
                # Label: Did this machine fail in the next forecast_horizon steps?
                future_time_start = machine_data.iloc[i+sequence_length]['timestamp']
                future_time_end = machine_data.iloc[i+sequence_length+forecast_horizon-1]['timestamp']
                
                failures_in_range = self.machine_failures[
                    (self.machine_failures['machine_id'] == machine_id) &
                    (self.machine_failures['timestamp'] >= future_time_start) &
                    (self.machine_failures['timestamp'] <= future_time_end)
                ]
                
                y = 1.0 if len(failures_in_range) > 0 else 0.0
                
                X_list.append(X)
                y_list.append(y)
                machines_list.append(machine_id)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Split 70/15/15
        n = len(X)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        indices = np.random.permutation(n)
        
        X_train = X[indices[:train_size]]
        y_train = y[indices[:train_size]]
        X_val = X[indices[train_size:train_size+val_size]]
        y_val = y[indices[train_size:train_size+val_size]]
        X_test = X[indices[train_size+val_size:]]
        y_test = y[indices[train_size+val_size:]]
        
        logger.info(f"✓ LSTM Features prepared")
        logger.info(f"  X_train: {X_train.shape}, Positive: {(y_train==1).sum()}/{len(y_train)}")
        logger.info(f"  X_val: {X_val.shape}, Positive: {(y_val==1).sum()}/{len(y_val)}")
        logger.info(f"  X_test: {X_test.shape}, Positive: {(y_test==1).sum()}/{len(y_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_cols,
            'machines': list(self.machine_history['machine_id'].unique())
        }
    
    def prepare_arima_features(self):
        """
        Prepare delay data for ARIMA model training.
        
        Returns:
            dict: {
                'delays_by_machine': dict mapping machine_id to delay series,
                'machines': list of machine IDs
            }
        """
        logger.info("\n🔧 Preparing ARIMA features for delay forecasting")
        
        delays_by_machine = {}
        
        for machine_id in self.delays['machine_id'].unique():
            machine_delays = self.delays[self.delays['machine_id'] == machine_id]['delay'].values
            
            if len(machine_delays) > 10:  # Need sufficient data for ARIMA
                delays_by_machine[machine_id] = machine_delays
        
        logger.info(f"✓ ARIMA Features prepared for {len(delays_by_machine)} machines")
        logger.info(f"  Average samples per machine: {np.mean([len(v) for v in delays_by_machine.values()]):.0f}")
        
        return {
            'delays_by_machine': delays_by_machine,
            'machines': list(delays_by_machine.keys())
        }
    
    def get_machine_statistics(self):
        """Get per-machine statistics for risk scoring"""
        logger.info("\n📈 Computing machine-level statistics")
        
        stats = {}
        
        for machine_id in self.machine_history['machine_id'].unique():
            machine_history = self.machine_history[self.machine_history['machine_id'] == machine_id]
            machine_failures = self.machine_failures[self.machine_failures['machine_id'] == machine_id]
            machine_delays = self.delays[self.delays['machine_id'] == machine_id]
            
            stats[machine_id] = {
                'failure_count': len(machine_failures),
                'avg_downtime': machine_failures['downtime_minutes'].mean() if len(machine_failures) > 0 else 0,
                'avg_delay': machine_delays['delay'].mean() if len(machine_delays) > 0 else 0,
                'max_delay': machine_delays['delay'].max() if len(machine_delays) > 0 else 0,
                'avg_utilization': machine_history['utilization_percent'].mean(),
                'avg_temperature': machine_history['temperature'].mean(),
                'avg_vibration': machine_history['vibration'].mean(),
            }
        
        stats_df = pd.DataFrame(stats).T
        logger.info(f"\n{stats_df}")
        
        return stats_df


def main():
    """Example usage"""
    loader = DataLoader("data/")
    loader.load_all()
    loader.explore_data()
    
    # Prepare LSTM features
    lstm_data = loader.prepare_lstm_features(sequence_length=30, forecast_horizon=5)
    
    # Prepare ARIMA features
    arima_data = loader.prepare_arima_features()
    
    # Get machine statistics
    stats = loader.get_machine_statistics()
    stats.to_csv("outputs/machine_statistics.csv")
    logger.info("✓ Machine statistics saved to outputs/machine_statistics.csv")


if __name__ == "__main__":
    main()
