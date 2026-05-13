# train_rl.py
from jssp_env import JSSPEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import os

env = JSSPEnv(dataset_name="FT06", data_dir="data/")
check_env(env, warn=True)
print(f"Jobs: {env.n_jobs}, Machines: {env.n_machines}, Total ops: {env.total_ops}")

def make_env():
    return JSSPEnv(dataset_name="FT06", data_dir="data/")

vec_env = DummyVecEnv([make_env] * 4)  # 4 parallel envs = faster training

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=1e-3,
    n_steps=512,
    batch_size=64,
    n_epochs=20,
    gamma=0.99,
    ent_coef=0.01,       # encourages exploration
    clip_range=0.2,
    policy_kwargs=dict(net_arch=[128, 128]),
    tensorboard_log="./logs/"
)

print("\nTraining (this takes ~10-15 min)...")
model.learn(total_timesteps=500_000)

os.makedirs("models", exist_ok=True)
model.save("models/jssp_ppo_ft06")
print("Saved: models/jssp_ppo_ft06.zip")