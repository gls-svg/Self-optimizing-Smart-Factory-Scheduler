# jssp_env.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class JSSPEnv(gym.Env):
    def __init__(self, dataset_name="FT06", data_dir="data/"):
        super().__init__()

        schedule_df = pd.read_csv(f"{data_dir}master_schedule_dataset.csv")
        delay_df    = pd.read_csv(f"{data_dir}delay_dataset.csv")
        job_cost_df = pd.read_csv(f"{data_dir}job_cost_dataset.csv")
        mac_cost_df = pd.read_csv(f"{data_dir}machine_cost_dataset.csv")
        failure_df  = pd.read_csv(f"{data_dir}machine_failure_dataset.csv")

        self.schedule_df = schedule_df[schedule_df['dataset'] == dataset_name].copy()
        self.jobs     = self.schedule_df['global_job_id'].unique().tolist()
        self.machines = sorted(self.schedule_df['machine_id'].unique().tolist())
        self.n_jobs     = len(self.jobs)
        self.n_machines = len(self.machines)

        self.job_ops = {}
        for job in self.jobs:
            ops = self.schedule_df[self.schedule_df['global_job_id'] == job].sort_values('operation_id')
            self.job_ops[job] = list(zip(ops['machine_id'], ops['processing_time']))

        self.total_ops = sum(len(v) for v in self.job_ops.values())

        self.delay_lookup = {}
        for _, row in delay_df[delay_df['global_job_id'].isin(self.jobs)].iterrows():
            self.delay_lookup[(row['global_job_id'], int(row['operation_id']))] = float(row['delay'])

        priority_map = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        self.job_priority = {}
        self.job_penalty  = {}
        for _, row in job_cost_df[job_cost_df['global_job_id'].isin(self.jobs)].iterrows():
            self.job_priority[row['global_job_id']] = priority_map.get(row['priority'], 1)
            self.job_penalty[row['global_job_id']]  = float(row['delay_penalty'])

        self.machine_failure_risk = {}
        for mid in self.machines:
            rows = failure_df[failure_df['machine_id'] == mid]
            self.machine_failure_risk[mid] = rows['downtime_minutes'].mean() if len(rows) > 0 else 0.0

        self.machine_op_cost = {}
        for _, row in mac_cost_df.iterrows():
            self.machine_op_cost[int(row['machine_id'])] = float(row['operating_cost_per_hour'])

        # Normalize constants
        all_times = [pt for ops in self.job_ops.values() for _, pt in ops]
        self.max_time = sum(all_times)  # worst possible makespan

        obs_size = self.n_jobs * 2 + self.n_machines
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_jobs)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.job_op_idx    = {j: 0 for j in self.jobs}
        self.job_avail     = {j: 0.0 for j in self.jobs}
        self.machine_avail = {m: 0.0 for m in self.machines}
        self.schedule_log  = []
        self.ops_done      = 0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for j in self.jobs:
            obs.append(self.job_op_idx[j] / len(self.job_ops[j]))  # progress 0-1
            obs.append(self.job_avail[j] / self.max_time)           # normalized time
        for m in self.machines:
            obs.append(self.machine_avail[m] / self.max_time)
        return np.array(obs, dtype=np.float32)

    def _job_done(self, job):
        return self.job_op_idx[job] >= len(self.job_ops[job])

    def _all_done(self):
        return all(self._job_done(j) for j in self.jobs)

    def step(self, action):
        job = self.jobs[action]

        # Redirect if this job is finished
        if self._job_done(job):
            for i in range(self.n_jobs):
                if not self._job_done(self.jobs[i]):
                    action = i
                    job = self.jobs[i]
                    break
            else:
                return self._get_obs(), 0.0, True, False, {}

        op_idx = self.job_op_idx[job]
        machine, proc_time = self.job_ops[job][op_idx]

        start_time = max(self.job_avail[job], self.machine_avail[machine])
        delay = self.delay_lookup.get((job, op_idx + 1), 0.0)
        actual_proc = proc_time + delay
        end_time = start_time + actual_proc

        self.machine_avail[machine] = end_time
        self.job_avail[job] = end_time
        self.job_op_idx[job] += 1
        self.ops_done += 1

        self.schedule_log.append({
            'job': job, 'operation': op_idx + 1,
            'machine': machine, 'start': start_time,
            'end': end_time, 'delay': delay
        })

        # Clean normalized reward: minimize idle time (machine wait)
        idle_time = start_time - self.machine_avail.get(machine, start_time)
        idle_time = max(0, idle_time)
        reward = -(idle_time / self.max_time)  # penalize idle gaps

        done = self._all_done()
        if done:
            makespan = max(self.machine_avail.values())
            # Final reward: how much better than worst case
            reward += 1.0 - (makespan / self.max_time)

        return self._get_obs(), reward, done, False, {}

    def get_makespan(self):
        return max(self.machine_avail.values()) if self.machine_avail else 0

    def get_schedule_df(self):
        return pd.DataFrame(self.schedule_log)