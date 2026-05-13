# explore_data.py
import pandas as pd

schedule = pd.read_csv("data/master_schedule_dataset.csv")
delays   = pd.read_csv("data/delay_dataset.csv")
failures = pd.read_csv("data/machine_failure_dataset.csv")
job_cost = pd.read_csv("data/job_cost_dataset.csv")
mac_cost = pd.read_csv("data/machine_cost_dataset.csv")

print("=== SCHEDULE ===")
print(schedule.head(10))
print("\nDatasets:", schedule['dataset'].unique())
print("Total operations:", len(schedule))

print("\n=== DELAYS ===")
print(delays.describe())

print("\n=== JOB PRIORITIES ===")
print(job_cost['priority'].value_counts())

print("\n=== MACHINE COSTS ===")
print(mac_cost)