# evaluate.py
import pandas as pd
import numpy as np
from jssp_env import JSSPEnv
from stable_baselines3 import PPO

def run_rl(dataset_name="FT06"):
    env = JSSPEnv(dataset_name=dataset_name, data_dir="data/")
    model = PPO.load("models/jssp_ppo_ft06")
    obs, _ = env.reset()
    done = False
    step_count = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        step_count += 1
        if step_count % 100 == 0:
            print(f"  Step {step_count}, done={done}")
        if step_count > 10000:  # safety exit
            print("WARNING: Too many steps, breaking")
            break
    return env.get_makespan(), env.get_schedule_df()
    

def run_fcfs(dataset_name="FT06"):
    """First Come First Served — always pick job 0, 1, 2... in order."""
    env = JSSPEnv(dataset_name=dataset_name, data_dir="data/")
    obs, _ = env.reset()
    done = False
    job_idx = 0
    while not done:
        action = job_idx % env.n_jobs
        obs, _, done, _, _ = env.step(action)
        job_idx += 1
    return env.get_makespan(), env.get_schedule_df()

def run_spt(dataset_name="FT06"):
    """Shortest Processing Time — always pick job whose next op is shortest."""
    env = JSSPEnv(dataset_name=dataset_name, data_dir="data/")
    obs, _ = env.reset()
    done = False
    while not done:
        # Find job with shortest next processing time
        min_time = float('inf')
        best_action = 0
        for i, job in enumerate(env.jobs):
            op_idx = env.job_op_idx[job]
            if op_idx < len(env.job_ops[job]):
                _, pt = env.job_ops[job][op_idx]
                if pt < min_time:
                    min_time = pt
                    best_action = i
        obs, _, done, _, _ = env.step(best_action)
    return env.get_makespan(), env.get_schedule_df()

# --- Run comparison ---
print("Running RL agent...")
rl_makespan, rl_schedule = run_rl()

print("Running FCFS baseline...")
fcfs_makespan, fcfs_schedule = run_fcfs()

print("Running SPT baseline...")
spt_makespan, spt_schedule = run_spt()

print("\n========== RESULTS ==========")
print(f"RL (PPO)  makespan: {rl_makespan:.2f}")
print(f"FCFS      makespan: {fcfs_makespan:.2f}")
print(f"SPT       makespan: {spt_makespan:.2f}")
print(f"\nRL improvement over FCFS: {((fcfs_makespan - rl_makespan)/fcfs_makespan)*100:.1f}%")
print(f"RL improvement over SPT:  {((spt_makespan - rl_makespan)/spt_makespan)*100:.1f}%")

# Save schedules for Member 5 (Gantt chart)
import os
os.makedirs("outputs", exist_ok=True)
rl_schedule.to_csv("outputs/rl_schedule.csv", index=False)
fcfs_schedule.to_csv("outputs/fcfs_schedule.csv", index=False)
spt_schedule.to_csv("outputs/spt_schedule.csv", index=False)
print("\nSchedules saved to outputs/")

# Safety validation for Member 3
try:
    from safety_validator.report_generator import validate_and_report

    failure_file = "data/machine_failure_dataset.csv"
    if not os.path.exists(failure_file):
        failure_file = None

    validation_results, report_files = validate_and_report(
        [
            "outputs/fcfs_schedule.csv",
            "outputs/spt_schedule.csv",
            "outputs/rl_schedule.csv",
        ],
        failure_path=failure_file,
        output_dir="outputs",
    )

    print("\n========== SAFETY VALIDATION ==========")
    for schedule_name, result in validation_results.items():
        status = "SAFE" if result["is_safe"] else "UNSAFE"
        total_violations = result["summary"]["total_violations"]
        print(f"{schedule_name}: {status} ({total_violations} violations)")

    print(f"Safety report saved to {report_files[0]}")
    print(f"Violation CSV saved to {report_files[1]}")
except Exception as exc:
    print(f"\nSafety validation skipped because of an error: {exc}")
