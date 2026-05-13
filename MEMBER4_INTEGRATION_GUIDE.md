# MEMBER 4 PROJECT MAP & INTEGRATION GUIDE

## 📊 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 SELF-OPTIMIZING SMART FACTORY SCHEDULER                 │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│   MEMBER 1               │     │   MEMBER 2               │     │   MEMBER 3               │
│   DATASET COLLECTION     │     │   RL SCHEDULER           │     │   SAFETY VALIDATOR       │
│                          │     │                          │     │                          │
│ • Dataset preparation    │     │ • JSSP Environment       │     │ • Constraint checking    │
│ • Machine history logs   │────→│ • PPO RL Agent           │────→│ • Schedule validation    │
│ • Failure simulation     │     │ • Action: job→machine    │     │ • Safety reports         │
│ • Delay patterns         │     │                          │     │                          │
└──────────────────────────┘     └──────────────────────────┘     └──────────────────────────┘
         ↓                                  ↓ (calls prediction)           ↓ (validates)
   Machine History             ┌─────────────────────────────────────────┐
   Failure Dataset             │ MEMBER 4 (YOU!)                         │
   Delay Dataset               │ PREDICTIVE DISRUPTION HANDLING          │
                              │                                         │
                              │ 1. LSTM Failure Predictor              │
                              │ 2. ARIMA Delay Forecaster              │
                              │ 3. Risk Scorer (Multi-factor)          │
                              │ 4. DisruptionHandler (Main Interface)  │
                              │                                         │
                              │ Returns:                                │
                              │ • failure_probability                  │
                              │ • risk_score                           │
                              │ • risk_category (LOW/MED/HIGH)         │
                              │ • recommendation (SAFE/CAUTION/AVOID)  │
                              │ • predicted_delay                      │
                              └─────────────────────────────────────────┘
                                         ↓ (feeds back to scheduler)

┌──────────────────────────────────────────────────────────────────────────┐
│ MEMBER 5: BACKEND INTEGRATION & API                                      │
│ • FastAPI endpoints                                                      │
│ • Integrate all modules (Member 1-4)                                    │
│ • Database storage                                                       │
│ • User interface / Gantt charts                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow: How Member 4 Fits In

### Scheduling Loop:

```
1. Scheduler (Member 2) selects candidate machines for a job
   ↓
2. FOR EACH CANDIDATE MACHINE:
   ├─ Call: disruption_handler.predict_for_machine(machine_id)
   ├─ LSTM predicts failure probability based on telemetry
   ├─ RiskScorer combines failure_prob + current_state + history
   ├─ ARIMA forecasts delay
   └─ Returns: risk_score, recommendation
   ↓
3. Scheduler filters to safe machines using:
   safe_machines = handler.get_safe_machines(candidates)
   ↓
4. Scheduler assigns job to best safe machine
   ↓
5. Safety Validator (Member 3) checks schedule feasibility
   ↓
6. If high risk predicted:
   ├─ DisruptionHandler.trigger_predictive_alert()
   ├─ May trigger proactive rescheduling
   └─ Logs alert for monitoring
   ↓
7. Schedule executed, predictions logged
   ↓
8. Actual outcomes compared with predictions
   ├─ Used to improve future models
   └─ Feedback loop for continuous learning
```

---

## 📂 File Organization After Completion

```
Self-optimizing-Smart-Factory-Scheduler/
│
├── predictive_module/                 ← YOUR MODULE
│   ├── __init__.py
│   ├── data_preparation.py            (Step 1: Load & explore data)
│   ├── predictive_model.py            (Step 2-3: LSTM & ARIMA)
│   ├── risk_scorer.py                 (Step 4: Risk computation)
│   ├── disruption_handler.py          (Step 5: Main interface)
│   ├── train_predictor.py             (Step 6: Training pipeline)
│   ├── evaluate_predictor.py          (Step 7: Evaluation)
│   ├── requirements.txt
│   ├── QUICKSTART.md
│   └── MEMBER4_INTEGRATION_GUIDE.md   (This file)
│
├── models/
│   ├── jssp_ppo_ft06.zip              (from Member 2)
│   ├── lstm_failure_model.h5          ← YOUR MODEL
│   ├── lstm_metrics.json
│   ├── arima_models/                  ← YOUR MODELS
│   │   ├── arima_machine_0.pkl
│   │   ├── arima_machine_1.pkl
│   │   └── ... (one per machine)
│   ├── arima_results.json
│   └── risk_scorer_config.pkl
│
├── outputs/
│   ├── lstm_evaluation.png            ← YOUR OUTPUTS
│   ├── arima_evaluation.png
│   ├── model_comparison_report.json
│   ├── machine_statistics.csv
│   ├── prediction_results.csv         (runtime predictions)
│   ├── risk_scores.csv                (runtime scores)
│   └── ... (existing Member 2-3 outputs)
│
├── data/
│   ├── machine_history_dataset.csv    (input for YOUR module)
│   ├── machine_failure_dataset.csv
│   ├── delay_dataset.csv
│   └── ... (other files from Member 1)
│
├── safety_validator/                  (from Member 3)
│   ├── validator.py
│   ├── constraints.py
│   └── ...
│
├── jssp_env.py                        (from Member 2)
├── train_rl.py                        (from Member 2)
├── evaluate.py                        (from Member 2)
│
├── MEMBER4_IMPLEMENTATION_GUIDE.md    (reference guide)
├── MEMBER4_TASK_TRACKING.md           (checklist)
├── MEMBER4_PROJECT_MAP.md             (this file)
└── README.md
```

---

## 🔌 Integration Points: How Your Code Connects

### 1️⃣ With RL Scheduler (Member 2)

**File**: `jssp_env.py` (modifications needed)

```python
# Add to imports
from predictive_module import DisruptionHandler
import pickle

# Add to JSSPEnv.__init__()
def __init__(self, ...):
    # ... existing code ...
    
    # Load your predictive models
    from tensorflow.keras.models import load_model
    lstm_model = load_model('models/lstm_failure_model.h5')
    delay_model = DelayForecaster.load('models/arima_models')
    
    with open('models/risk_scorer_config.pkl', 'rb') as f:
        scorer_config = pickle.load(f)
    
    risk_scorer = RiskScorer.from_dict(scorer_config)
    
    # Initialize disruption handler
    self.disruption_handler = DisruptionHandler(
        failure_model=lstm_model,
        delay_model=delay_model,
        risk_scorer=risk_scorer
    )

# Modify step() function
def step(self, action):
    machine_id, op_idx = self.decode_action(action)
    
    # ✨ NEW: Check if machine is safe
    prediction = self.disruption_handler.predict_for_machine(
        machine_id=machine_id,
        current_telemetry=self.get_latest_telemetry(machine_id)
    )
    
    if prediction['risk_category'] == 'HIGH':
        # Penalize unsafe assignment
        risk_penalty = 100 * prediction['risk_score']
        # Option: trigger rescheduling
        self.disruption_handler.trigger_predictive_alert(machine_id, prediction)
    
    # Continue with normal RL logic
    # ... existing step logic ...
    
    return obs, reward, done, truncated, info
```

### 2️⃣ With Safety Validator (Member 3)

**File**: `safety_validator/validator.py` (no changes needed)

Your module complements Member 3:
- Member 3: Validates **feasibility** (constraints)
- Member 4: Validates **safety** (predictive risk)

Both should pass before execution.

### 3️⃣ With Backend API (Member 5)

**Expected API Endpoints** (Member 5 will create):

```python
from fastapi import FastAPI
from predictive_module import DisruptionHandler

app = FastAPI()
handler = DisruptionHandler(...)

@app.post("/api/predict/machine/{machine_id}")
async def predict_machine(machine_id: int, telemetry: dict):
    """Get prediction for a machine"""
    prediction = handler.predict_for_machine(machine_id, telemetry)
    return prediction

@app.post("/api/predict/safe-machines")
async def get_safe_machines(machine_list: list):
    """Filter to safe machines"""
    safe = handler.get_safe_machines(machine_list)
    return {"safe_machines": safe}

@app.post("/api/predict/compare")
async def compare_machines(machine_list: list):
    """Compare risk across machines"""
    comparison = handler.compare_machines(machine_list)
    return comparison.to_dict()

@app.get("/api/alerts/summary")
async def get_alert_summary():
    """Get alert summary"""
    return handler.get_alert_summary()
```

---

## 🧪 Integration Testing Checklist

Before handing off to Member 5:

### Test 1: Model Loading
```python
# Can all models be loaded?
from predictive_model import FailurePredictorLSTM, DelayForecaster
from risk_scorer import RiskScorer
from disruption_handler import DisruptionHandler

lstm = FailurePredictorLSTM.load('models/lstm_failure_model.h5')
arima = DelayForecaster.load('models/arima_models')
# ✓ Models load without error
```

### Test 2: Predictions Work
```python
handler = DisruptionHandler(lstm, arima, risk_scorer)

# Test 1: Single prediction
pred = handler.predict_for_machine(machine_id=0)
assert 0 <= pred['risk_score'] <= 1
assert pred['recommendation'] in ['SAFE', 'CAUTION', 'AVOID']
# ✓ Predictions have correct format

# Test 2: Batch predictions
preds = [handler.predict_for_machine(i) for i in range(5)]
assert len(preds) == 5
# ✓ Batch predictions work
```

### Test 3: Integration with Scheduler
```python
# Can scheduler use your predictions?
from jssp_env import JSSPEnv

env = JSSPEnv()
obs, info = env.reset()

for _ in range(10):
    # This should call your handler internally
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    # Check that disruption alerts are logged if needed
    if 'disruption_alerts' in info:
        print(f"Alert triggered: {info['disruption_alerts']}")
    
    if done:
        break
# ✓ Scheduler successfully uses handler
```

### Test 4: Performance
```python
import time

# Measure prediction latency
start = time.time()
for i in range(100):
    handler.predict_for_machine(i % 14)
elapsed = (time.time() - start) / 100

print(f"Avg prediction time: {elapsed*1000:.2f}ms")
assert elapsed < 0.1, "Prediction too slow!"
# ✓ Performance acceptable
```

---

## 📋 Handoff Checklist for Member 5

Create a summary file `MEMBER4_HANDOFF.md`:

- [ ] All models trained and saved
- [ ] Performance metrics meet targets
- [ ] Integration tests pass
- [ ] API specification documented
- [ ] Example code provided
- [ ] Quick start guide created
- [ ] Error handling in place
- [ ] Logging configured
- [ ] Model update procedure documented
- [ ] Performance monitoring setup

---

## ⚠️ Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Model accuracy too low | Insufficient data | Simulate more data, increase epochs |
| ARIMA fails for machine | Sparse delay data | Use fallback forecaster |
| Predictions inconsistent | Non-deterministic TensorFlow | Set random seeds |
| Scheduler crashes on prediction | Model not loaded | Add error handling |
| Latency too high | Large batch prediction | Use caching, vectorization |
| Risk scores always HIGH | Thresholds too sensitive | Calibrate on historical data |

---

## 🎯 Member 4 Success Criteria

✅ **Minimum Requirements**:
- [ ] LSTM model: >75% accuracy on failure prediction
- [ ] ARIMA models: RMSE within ±10% of average delay
- [ ] Risk scorer: Combines at least 3 factors
- [ ] DisruptionHandler: Integrated with scheduler without breaking it
- [ ] All models saved and loadable

✅ **Nice to Have**:
- [ ] Risk-failure correlation > 0.7
- [ ] Predictions < 50ms average latency
- [ ] Visualization dashboards
- [ ] Confidence intervals for ARIMA
- [ ] Automated model retraining pipeline

✅ **Excellent**:
- [ ] Ensemble predictions (LSTM + ARIMA + baseline)
- [ ] Online learning capability
- [ ] Model explainability
- [ ] Advanced metrics (Precision-Recall, Matthews correlation)
- [ ] A/B testing framework

---

## 🚀 Next Steps

1. **Start**: Run `python predictive_module/data_preparation.py`
2. **Explore**: Review outputs and statistics
3. **Train**: Run `python predictive_module/train_predictor.py` (15 mins)
4. **Evaluate**: Run `python predictive_module/evaluate_predictor.py`
5. **Integrate**: Test with `jssp_env.py`
6. **Optimize**: Adjust hyperparameters if needed
7. **Hand Off**: Create handoff documentation for Member 5

---

## 💡 Pro Tips

1. **Start Simple**: Test with one machine before batch predictions
2. **Monitor Training**: Watch for overfitting; use validation set
3. **Document Decisions**: Why you chose LSTM over ARIMA, thresholds, etc.
4. **Version Models**: Save with timestamps (e.g., `lstm_failure_2024_01_15.h5`)
5. **Test Edge Cases**: What happens with new machines? Zero historical data?

---

## 🤝 Communication Strategy

**With Member 2 (RL)**:
- "Your scheduler can call `handler.predict_for_machine()`"
- "Use `get_safe_machines()` to filter candidates"
- "Integrate risk score into reward function"

**With Member 5 (Backend)**:
- "All models are serialized and ready to load"
- "Here's the API contract: predict(), compare(), alerts()"
- "Performance target: <100ms per prediction"

---

## 📞 Questions to Answer Before Handoff

1. **Can Member 2 easily integrate your module?** → Simple API ✓
2. **What happens if a machine has no historical data?** → Fallback strategy
3. **How often should models be retrained?** → Every month / on demand
4. **What's the memory footprint?** → Total model size
5. **Is there a graceful degradation if predictions fail?** → Yes, returns neutral

---

Congratulations! You're building the brain of the factory. 🧠

Good luck, Member 4! 🚀
