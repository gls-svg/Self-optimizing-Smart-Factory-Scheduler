# 🚀 MEMBER 4 QUICK START GUIDE

## Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r predictive_module/requirements.txt
```

### 2. Verify Data Files
Ensure these files exist in `data/`:
- `master_schedule_dataset.csv`
- `machine_history_dataset.csv`
- `machine_failure_dataset.csv`
- `delay_dataset.csv`
- `job_cost_dataset.csv`
- `machine_cost_dataset.csv`

---

## Day 1: Data Exploration

### Task 1.1: Understand Your Data
```bash
cd predictive_module
python data_preparation.py
```

This will:
- Load all datasets
- Print summary statistics
- Check for missing values
- Save machine statistics to `outputs/machine_statistics.csv`

**Expected Output**:
```
Machine History: 5000+ rows
Machine Failures: 100+ rows  
Delays: 200+ rows
Unique Machines: 14-20
```

### Task 1.2: Review Statistics
```bash
cat outputs/machine_statistics.csv
```

Look for patterns:
- Machines with high failure counts
- Machines with high average delays
- Temperature/vibration outliers

---

## Days 2-3: Train LSTM Model

### Task 2.1: Train All Models
```bash
python train_predictor.py
```

This will:
1. Load data
2. Create LSTM sequences (sliding windows)
3. Train LSTM model for 50 epochs
4. Train ARIMA models for each machine
5. Save models to `models/`
6. Create risk scorer

**Expected Time**: 10-15 minutes
**Expected Output**:
```
✓ LSTM Training completed
  Accuracy: 0.75+ (target: >75%)
  AUC: 0.80+
✓ ARIMA models fitted for 15+ machines
✓ Risk scorer created
```

---

## Days 4-5: Evaluate & Integrate

### Task 3.1: Evaluate Models
```bash
python evaluate_predictor.py
```

Generates:
- `outputs/lstm_evaluation.png` - Confusion matrix, ROC curve
- `outputs/arima_evaluation.png` - Forecast accuracy
- `outputs/model_comparison_report.json` - Detailed comparison

### Task 3.2: Check Integration
```python
from disruption_handler import DisruptionHandler

# Load models
from predictive_model import FailurePredictorLSTM, DelayForecaster
from risk_scorer import RiskScorer

lstm = FailurePredictorLSTM.load("models/lstm_failure_model.h5")
arima = DelayForecaster.load("models/arima_models")

# Initialize handler
handler = DisruptionHandler(failure_model=lstm, 
                           delay_model=arima,
                           risk_scorer=risk_scorer)

# Test prediction
pred = handler.predict_for_machine(machine_id=0)
print(pred)  # Should show risk_score, failure_probability, recommendation
```

---

## ⚙️ Module Architecture

```
predictive_module/
├── data_preparation.py        → Load & prepare data
├── predictive_model.py        → LSTM & ARIMA models
├── risk_scorer.py             → Multi-factor risk scoring
├── disruption_handler.py      → Main interface for scheduler
├── train_predictor.py         → Training pipeline
└── evaluate_predictor.py      → Evaluation & comparison
```

---

## 🔗 Integration with Member 2 (RL Scheduler)

### In `jssp_env.py` or wrapper:

```python
from predictive_module import DisruptionHandler

# Initialize in env
self.disruption_handler = DisruptionHandler(...)

# Before assigning job to machine:
def is_machine_safe(machine_id, current_telemetry):
    pred = self.disruption_handler.predict_for_machine(machine_id, current_telemetry)
    return pred['recommendation'] != 'AVOID'

# Use in step function:
def step(self, action):
    machine_id = self.job_ops[action]
    
    # Check if machine is safe
    if not is_machine_safe(machine_id):
        # Penalty for unsafe assignment
        reward -= 50
        # Trigger rescheduling
        self.disruption_handler.trigger_predictive_alert(...)
    
    # Continue with normal scheduling
    ...
```

---

## 📊 Key Metrics to Monitor

| Metric | Target | How to Check |
|--------|--------|-------------|
| LSTM Accuracy | >75% | `outputs/lstm_evaluation.png` |
| LSTM AUC | >0.80 | `outputs/model_comparison_report.json` |
| ARIMA RMSE | ±10% of mean delay | `outputs/arima_evaluation.png` |
| Risk-Failure Correlation | >0.7 | Compare predictions with actual |
| Integration Latency | <100ms | Benchmark in production |

---

## ✅ Checklist

- [ ] Dependencies installed
- [ ] Data files present in `data/`
- [ ] `data_preparation.py` runs successfully
- [ ] Machine statistics reviewed
- [ ] LSTM model trained (Acc >75%)
- [ ] ARIMA models fitted for all machines
- [ ] Risk scorer configured
- [ ] Models saved in `models/`
- [ ] Evaluation reports generated
- [ ] Integration tested with sample data
- [ ] Ready for Member 5 backend integration

---

## 🆘 Troubleshooting

### Issue: "TensorFlow not found"
```bash
pip install tensorflow keras
```

### Issue: "statsmodels not found"
```bash
pip install statsmodels
```

### Issue: LSTM accuracy <75%
- Increase training epochs in `train_predictor.py`
- Check data quality in `outputs/machine_statistics.csv`
- Adjust sequence_length or features

### Issue: ARIMA errors
- Ensure delay data is not all zeros
- Check that each machine has enough historical delays
- Verify `data/delay_dataset.csv` is loaded correctly

---

## 📞 Need Help?

Check these files:
1. `MEMBER4_IMPLEMENTATION_GUIDE.md` - Detailed design guide
2. `MEMBER4_TASK_TRACKING.md` - Task checklist
3. Model docstrings in source files

---

## 🎯 Success = Shipping Member 4 to Member 5

When complete, Member 5 will integrate this module into the backend API:
- Scheduler calls `handler.predict_for_machine()`
- Gets risk scores and recommendations
- Makes safer scheduling decisions
- Triggers rescheduling when needed

Your disruption handler is the brain preventing failures! 🧠
