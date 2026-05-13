"""
Predictive Models for Machine Failures and Delays

Implements:
- LSTM model for failure prediction
- ARIMA model for delay forecasting
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Try to import deep learning libraries
TF_AVAILABLE = False
STATSMODELS_AVAILABLE = False

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception as e:
    logger.warning(f"TensorFlow not available. LSTM model will not be available. Error: {e}")

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except Exception as e:
    logger.warning(f"statsmodels not available. ARIMA model will not be available. Error: {e}")


class FailurePredictorLSTM:
    """
    LSTM-based predictor for machine failures.
    
    Predicts probability of machine failure in the next N timesteps
    based on historical telemetry data (utilization, temperature, vibration).
    """
    
    def __init__(self, sequence_length: int = 30, forecast_horizon: int = 5):
        """
        Args:
            sequence_length: Number of timesteps to look back
            forecast_horizon: Number of timesteps ahead to predict
        """
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for LSTM model. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.model = None
        self.is_trained = False
        
    def build_model(self, n_features: int = 3):
        """
        Build LSTM architecture.
        
        Args:
            n_features: Number of input features (default: 3 for utilization, temp, vibration)
        """
        self.model = keras.Sequential([
            layers.LSTM(64, activation='relu', input_shape=(self.sequence_length, n_features), return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC()]
        )
        
        logger.info(f"✓ LSTM model built with input shape: ({self.sequence_length}, {n_features})")
        return self
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              verbose: int = 1) -> Dict:
        """
        Train LSTM model.
        
        Args:
            X_train: Training sequences of shape (samples, sequence_length, features)
            y_train: Training labels of shape (samples, forecast_horizon)
            X_val: Validation sequences
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Logging verbosity
            
        Returns:
            dict: Training history
        """
        if self.model is None:
            self.build_model(n_features=X_train.shape[2])
        
        logger.info(f"Training LSTM model for {epochs} epochs...")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
        )
        
        self.is_trained = True
        logger.info("✓ LSTM training completed")
        
        return history.history
    
    def predict_failure_probability(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict failure probabilities.
        
        Args:
            X: Input sequences of shape (samples, sequence_length, features)
            threshold: Classification threshold
            
        Returns:
            np.ndarray: Failure probabilities (max across forecast horizon)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        probs = self.model.predict(X, verbose=0)  # Shape: (samples, forecast_horizon)
        
        # Return max probability across forecast horizon
        return probs.max(axis=1)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance on test set.
        
        Returns:
            dict: Evaluation metrics
        """
        loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        predictions = self.predict_failure_probability(X_test)
        pred_binary = (predictions > 0.5).astype(int)
        
        # Handle both 1D and 2D y_test arrays
        if y_test.ndim == 1:
            y_binary = (y_test > 0.5).astype(int)
        else:
            y_binary = (y_test.max(axis=1) > 0.5).astype(int)
        
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
        
        cm = confusion_matrix(y_binary, pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(y_binary, pred_binary, average='binary')
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
    
    def save(self, filepath: str):
        """Save model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        self.model.save(filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Load model from disk"""
        model = keras.models.load_model(filepath)
        logger.info(f"✓ Model loaded from {filepath}")
        
        # Wrap in FailurePredictorLSTM instance
        predictor = FailurePredictorLSTM()
        predictor.model = model
        predictor.is_trained = True
        return predictor


class DelayForecaster:
    """
    ARIMA-based predictor for processing delays.
    
    Maintains separate ARIMA model for each machine to predict
    expected delays for upcoming operations.
    """
    
    def __init__(self):
        """Initialize delay forecaster"""
        if not STATSMODELS_AVAILABLE:
            raise RuntimeError("statsmodels is required for ARIMA. Install with: pip install statsmodels")
        
        self.models = {}  # Dict mapping machine_id -> ARIMA model
        self.is_fitted = False
    
    def fit(self, delays_by_machine: Dict[int, np.ndarray], 
            p_range: Tuple = (0, 1, 2), d_range: Tuple = (0, 1),
            q_range: Tuple = (0, 1, 2)) -> Dict:
        """
        Fit ARIMA models for each machine.
        
        Automatically selects best (p, d, q) parameters using AIC.
        
        Args:
            delays_by_machine: Dict mapping machine_id to delay series (np.ndarray)
            p_range: Range of p values to try
            d_range: Range of d values to try
            q_range: Range of q values to try
            
        Returns:
            dict: Model information and AIC scores
        """
        logger.info(f"Fitting ARIMA models for {len(delays_by_machine)} machines...")
        
        results = {}
        
        for machine_id, delays in delays_by_machine.items():
            best_aic = float('inf')
            best_order = None
            best_model = None
            
            # Grid search for best ARIMA parameters
            for p in p_range:
                for d in d_range:
                    for q in q_range:
                        try:
                            model = ARIMA(delays, order=(p, d, q))
                            fitted = model.fit()
                            
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_model = fitted
                        except:
                            continue
            
            if best_model is not None:
                self.models[machine_id] = best_model
                results[machine_id] = {
                    'order': best_order,
                    'aic': float(best_aic),
                    'rmse': float(best_model.resid.std())
                }
                logger.debug(f"  Machine {machine_id}: ARIMA{best_order} AIC={best_aic:.2f}")
            else:
                logger.warning(f"  Machine {machine_id}: Could not fit ARIMA model")
        
        self.is_fitted = True
        logger.info(f"✓ Fitted ARIMA models for {len(self.models)} machines")
        
        return results
    
    def forecast(self, machine_id: int, steps: int = 5) -> np.ndarray:
        """
        Forecast delays for given machine.
        
        Args:
            machine_id: Machine to forecast
            steps: Number of steps ahead to forecast
            
        Returns:
            np.ndarray: Forecasted delays
        """
        if machine_id not in self.models:
            logger.warning(f"Machine {machine_id} not in fitted models. Returning zeros.")
            return np.zeros(steps)
        
        forecast_result = self.models[machine_id].get_forecast(steps=steps)
        predicted_mean = forecast_result.predicted_mean
        
        # Handle both pandas Series and numpy arrays
        if hasattr(predicted_mean, 'values'):
            return predicted_mean.values
        else:
            return np.array(predicted_mean)
    
    def forecast_confidence(self, machine_id: int, steps: int = 5, alpha: float = 0.05):
        """
        Forecast with confidence intervals.
        
        Args:
            machine_id: Machine to forecast
            steps: Number of steps ahead
            alpha: Significance level for confidence interval
            
        Returns:
            dict: forecast, lower_ci, upper_ci
        """
        if machine_id not in self.models:
            return {
                'forecast': np.zeros(steps),
                'lower_ci': np.zeros(steps),
                'upper_ci': np.zeros(steps)
            }
        
        forecast_result = self.models[machine_id].get_forecast(steps=steps)
        ci = forecast_result.conf_int(alpha=alpha)
        
        # Handle both pandas Series/DataFrames and numpy arrays
        predicted_mean = forecast_result.predicted_mean
        if hasattr(predicted_mean, 'values'):
            forecast_array = predicted_mean.values
        else:
            forecast_array = np.array(predicted_mean)
            
        if hasattr(ci, 'iloc'):  # pandas DataFrame
            lower_array = ci.iloc[:, 0].values
            upper_array = ci.iloc[:, 1].values
        else:  # numpy array
            lower_array = ci[:, 0]
            upper_array = ci[:, 1]
        
        return {
            'forecast': forecast_array,
            'lower_ci': lower_array,
            'upper_ci': upper_array
        }
    
    def evaluate(self, X_test: Dict) -> Dict:
        """
        Evaluate forecast accuracy on test data.
        
        Args:
            X_test: Dict mapping machine_id to test delay series
            
        Returns:
            dict: RMSE, MAE metrics per machine
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        results = {}
        
        for machine_id, test_delays in X_test.items():
            if machine_id not in self.models or len(test_delays) < 2:
                continue
            
            forecast = self.forecast(machine_id, steps=len(test_delays))
            
            rmse = np.sqrt(mean_squared_error(test_delays, forecast))
            mae = mean_absolute_error(test_delays, forecast)
            
            results[machine_id] = {'rmse': float(rmse), 'mae': float(mae)}
        
        return results
    
    def save(self, dirpath: str):
        """Save all ARIMA models to directory"""
        import pickle
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        for machine_id, model in self.models.items():
            with open(dirpath / f"arima_machine_{machine_id}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"✓ Saved {len(self.models)} ARIMA models to {dirpath}")
    
    @staticmethod
    def load(dirpath: str):
        """Load ARIMA models from directory"""
        import pickle
        forecaster = DelayForecaster()
        dirpath = Path(dirpath)
        
        for model_file in dirpath.glob("arima_machine_*.pkl"):
            machine_id = int(model_file.stem.split('_')[-1])
            with open(model_file, 'rb') as f:
                forecaster.models[machine_id] = pickle.load(f)
        
        forecaster.is_fitted = True
        logger.info(f"✓ Loaded {len(forecaster.models)} ARIMA models from {dirpath}")
        
        return forecaster

