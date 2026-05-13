# Predictive Module - README

**Predictive Disruption Handling for Smart Factory Scheduling**

## Overview

This module provides **machine failure prediction** and **processing delay forecasting** to enable the RL scheduler to make safer, more reliable scheduling decisions.

### Key Features

✅ **LSTM-based Failure Prediction** - Learns from telemetry time-series
✅ **ARIMA-based Delay Forecasting** - Predicts processing delays per machine  
✅ **Multi-factor Risk Scoring** - Combines failure probability + current state + history
✅ **Simple Integration** - One-method interface for scheduler: `predict_for_machine()`
✅ **Comprehensive Evaluation** - Metrics, visualizations, comparison reports

---

## Files & Their Purpose

| File | Purpose | Key Class/Function |
|------|---------|-------------------|
| `data_preparation.py` | Load, explore, prepare data | `DataLoader` |
| `predictive_model.py` | LSTM & ARIMA models | `FailurePredictorLSTM`, `DelayForecaster` |
| `risk_scorer.py` | Multi-factor risk calculation | `RiskScorer` |
| `disruption_handler.py` | Main interface for scheduler | `DisruptionHandler` |
| `train_predictor.py` | Training pipeline | `train_all_models()` |
| `evaluate_predictor.py` | Evaluation & comparison | `evaluate_models()` |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Explore Data
```bash
python data_preparation.py
```

### 3. Train Models
```bash
python train_predictor.py
```
Expected time: 10-15 minutes

### 4. Evaluate Models
```bash
python evaluate_predictor.py
```
Outputs: plots and metrics

### 5. Use in Code
```python
from disruption_handler import DisruptionHandler
from predictive_model import FailurePredictorLSTM, DelayForecaster

# Load models
lstm = FailurePredictorLSTM.load("../models/lstm_failure_model.h5")
arima = DelayForecaster.load("../models/arima_models")

# Initialize handler
handler = DisruptionHandler(lstm, arima)

# Predict for a machine
prediction = handler.predict_for_machine(machine_id=0)
print(prediction)

# Get safe machines only
safe_machines = handler.get_safe_machines([0, 1, 2, 3])
```

---

## Module Architecture

```
┌──────────────────────────┐
│  Scheduler (Member 2)    │
│  Needs predictions       │
└────────────┬─────────────┘
             │
             ↓
┌──────────────────────────────────┐
│   DisruptionHandler              │
│   (Main Interface)               │
│   predict_for_machine(id)        │
│   get_safe_machines(list)        │
│   trigger_predictive_alert()     │
└──────┬─────────────────┬──────────┘
       │                 │
       ↓                 ↓
┌─────────────────┐  ┌──────────────┐
│ RiskScorer      │  │ LSTM/ARIMA   │
│ 3-factor score  │  │ Predictions  │
│ Risk categories │  │              │
└─────────────────┘  └──────────────┘
       ↑                      ↑
       └──────────┬───────────┘
                  │
                  ↓
         ┌─────────────────┐
         │  DataLoader     │
         │  Historical     │
         │  Telemetry      │
         │  Failures       │
         │  Delays         │
         └─────────────────┘
```

---

## Data Flow

### Training Time

```
Raw Data (Member 1)
    ↓
DataLoader.load_all()
    ├→ Load machine_history_dataset.csv
    ├→ Load machine_failure_dataset.csv
    ├→ Load delay_dataset.csv
    └→ Load master_schedule_dataset.csv
    ↓
Prepare Features
    ├→ prepare_lstm_features() → Sequences
    └→ prepare_arima_features() → Series
    ↓
Train Models
    ├→ FailurePredictorLSTM.train()
    └→ DelayForecaster.fit()
    ↓
Save Models
    ├→ models/lstm_failure_model.h5
    ├→ models/arima_models/
    └→ models/risk_scorer_config.pkl
```

### Inference Time (Scheduler Usage)

```
Scheduler has job to assign to machine(s)
    ↓
DisruptionHandler.predict_for_machine(machine_id)
    ├→ LSTM.predict_failure_probability()
    ├→ RiskScorer.calculate_machine_risk()
    ├→ DelayForecaster.forecast_delay()
    └→ Return {failure_prob, risk_score, recommendation}
    ↓
Scheduler filters/chooses safe machine
    ├→ IF risk == 'HIGH' → AVOID
    ├→ IF risk == 'MEDIUM' → CAUTION
    └→ IF risk == 'LOW' → SAFE
    ↓
Assign job and execute schedule
```

---

## API Reference

### DisruptionHandler (Main Class)

```python
class DisruptionHandler:
    def predict_for_machine(machine_id, current_time=None, current_telemetry=None):
        """
        Returns:
        {
            'machine_id': int,
            'failure_probability': float (0-1),
            'risk_score': float (0-1),
            'risk_category': str ('LOW', 'MEDIUM', 'HIGH'),
            'predicted_delay': float,
            'estimated_downtime': float,
            'recommendation': str ('SAFE', 'CAUTION', 'AVOID'),
            'confidence': float,
            'timestamp': datetime,
            'details': dict
        }
        """
    
    def get_safe_machines(machine_list, allow_caution=False):
        """Filter to machines with risk < HIGH"""
        
    def compare_machines(machine_list):
        """Rank machines by risk score"""
        
    def trigger_predictive_alert(machine_id, prediction):
        """Log alert, may trigger rescheduling"""
        
    def get_alert_summary():
        """Get statistics on recent alerts"""
```

### FailurePredictorLSTM

```python
class FailurePredictorLSTM:
    def build_model(n_features=3):
        """Create LSTM architecture"""
        
    def train(X_train, y_train, X_val, y_val, epochs=50):
        """Train on sequences"""
        
    def predict_failure_probability(X):
        """Get failure probability"""
        
    def evaluate(X_test, y_test):
        """Get metrics"""
        
    def save(filepath):
    def load(filepath):  # static method
```

### DelayForecaster

```python
class DelayForecaster:
    def fit(delays_by_machine):
        """Train ARIMA per machine"""
        
    def forecast(machine_id, steps=1):
        """Predict future delays"""
        
    def forecast_confidence(machine_id, steps=1, alpha=0.05):
        """Forecast with confidence intervals"""
        
    def evaluate(X_test):
        """Get RMSE, MAE metrics"""
        
    def save(dirpath):
    def load(dirpath):  # static method
```

### RiskScorer

```python
class RiskScorer:
    def set_machine_statistics(stats):
        """Set historical context"""
        
    def calculate_machine_risk(machine_id, temperature, vibration, utilization, failure_probability):
        """Returns (risk_score, details)"""
        
    def get_risk_category(risk_score):
        """Returns 'LOW', 'MEDIUM', or 'HIGH'"""
        
    def get_machine_recommendation(risk_score, failure_probability):
        """Returns 'SAFE', 'CAUTION', or 'AVOID'"""
        
    def forecast_delay(machine_id, steps=1):
        """Get expected delays"""
```

---

## Configuration & Thresholds

### Risk Score Weights (in RiskScorer)
```python
risk_score = 0.5 × failure_probability    # Prediction (50%)
           + 0.3 × telemetry_risk         # Current state (30%)
           + 0.2 × historical_risk        # History (20%)
```

### Risk Categories
- **LOW** (0.0-0.3): Safe to assign
- **MEDIUM** (0.3-0.6): Caution, may delay
- **HIGH** (0.6-1.0): Avoid if possible

### Telemetry Thresholds (default)
```python
temperature_high: 90°C
temperature_critical: 95°C
vibration_high: 0.8 m/s
vibration_critical: 0.95 m/s
utilization_high: 90%
```

---

## Model Details

### LSTM Architecture (Default)
```
Input: (batch, sequence_length=30, features=3)
  ↓
LSTM(64, return_sequences=True)
  ↓
Dropout(0.2)
  ↓
LSTM(32)
  ↓
Dropout(0.2)
  ↓
Dense(16, activation='relu')
  ↓
Dense(5, activation='sigmoid')
Output: (batch, forecast_horizon=5) - Probabilities
```

**Loss**: Binary Cross-Entropy
**Optimizer**: Adam (lr=0.001)
**Metrics**: Accuracy, AUC

### ARIMA Configuration (Auto-selected)
- **Parameter Search**: p ∈ {0,1,2}, d ∈ {0,1}, q ∈ {0,1,2}
- **Selection**: Best AIC per machine
- **Type**: Individual model per machine
- **Forecast**: Single-step ahead (configurable)

---

## Performance Targets

| Metric | Target | How to Check |
|--------|--------|-------------|
| LSTM Accuracy | >75% | `outputs/lstm_evaluation.png` |
| LSTM AUC-ROC | >0.80 | `outputs/model_comparison_report.json` |
| ARIMA RMSE | ±10% | `outputs/arima_evaluation.png` |
| Prediction Latency | <100ms | `time` 100 predictions |
| Model Size | <50MB | `du -sh models/` |

---

## Troubleshooting

### Training Issues

**Q: LSTM accuracy < 75%?**
- Try: Increase epochs, add more data, adjust learning rate
- Check: Data quality, feature scaling, class imbalance

**Q: ARIMA errors for some machines?**
- Try: Use fallback model, combine machines
- Check: Sparse data, non-stationary series

**Q: Training too slow?**
- Try: Use GPU, reduce batch size, smaller sequences
- Check: Data size, model complexity

### Prediction Issues

**Q: Predictions always HIGH risk?**
- Try: Calibrate thresholds on historical data
- Check: Feature scaling, threshold values

**Q: Different predictions each time?**
- Try: Set random seed, use deterministic mode
- Check: Dropout during inference, batch normalization

**Q: Slow predictions?**
- Try: Batch predictions, add caching
- Check: Model size, input preprocessing

---

## Integration Example

```python
# In jssp_env.py
from predictive_module import DisruptionHandler

class JSSPEnv(gym.Env):
    def __init__(self):
        # ... existing init ...
        
        # Load disruption handler
        lstm = FailurePredictorLSTM.load('models/lstm_failure_model.h5')
        delay = DelayForecaster.load('models/arima_models')
        self.handler = DisruptionHandler(lstm, delay)
    
    def step(self, action):
        machine_id = self.get_machine_from_action(action)
        
        # Check if safe
        pred = self.handler.predict_for_machine(machine_id)
        
        if pred['risk_category'] == 'HIGH':
            # Option 1: Penalize
            reward -= 50
            
            # Option 2: Trigger alert
            self.handler.trigger_predictive_alert(machine_id, pred)
        
        # Continue scheduling
        # ...
        return obs, reward, done, truncated, info
```

---

## Output Files Generated

### After Training
```
models/
├── lstm_failure_model.h5          # LSTM model weights
├── lstm_metrics.json              # LSTM performance
├── arima_models/
│   ├── arima_machine_0.pkl
│   ├── arima_machine_1.pkl
│   └── ...
├── arima_results.json             # ARIMA configs & AIC
└── risk_scorer_config.pkl         # Risk thresholds & stats

outputs/
├── machine_statistics.csv         # Per-machine stats
└── (after evaluation):
    ├── lstm_evaluation.png        # Plots & metrics
    ├── arima_evaluation.png       # Forecast errors
    └── model_comparison_report.json
```

---

## Citation & References

This module implements:
- **LSTM for Predictive Maintenance**: Based on Hochreiter & Schmidhuber (1997)
- **ARIMA for Time Series Forecasting**: Box & Jenkins (1970)
- **Multi-criteria Risk Assessment**: Hwang & Yoon (1981)

---

## License & Credit

Part of **Self-Optimizing Smart Factory Scheduler** project
- Member 4 Responsibility: Predictive Disruption Handling
- Contributors: [Your Name]

---

## Next Steps

1. ✅ Review this README
2. ✅ Run `python data_preparation.py`
3. ✅ Run `python train_predictor.py`
4. ✅ Run `python evaluate_predictor.py`
5. ✅ Test integration with scheduler
6. ✅ Hand off to Member 5

---

**Questions?** Check the documentation files or source code docstrings.

**Ready to predict failures?** Let's go! 🚀
