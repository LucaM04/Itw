from stable_baselines3 import PPO
from environment import AXLPrisonersDilemmaEnv, device
import numpy as np
from dqn import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Name muss groß sein ("Random", "Tit For Tat", "Grudger")
opponent_name = "Tit For Tat"
#opponent_name = "Defector"


# Environment erstellen
env = AXLPrisonersDilemmaEnv(opponent_name=opponent_name, random_opponent=False, max_rounds=100)

print("🔄 Lade PPO Modell...")
model = PPO.load("ppo_agent")

#model = DQN(2, 2).to(device)
#model.load_state_dict(torch.load("dqn_agent.pth"))
# --- FIX: ERST Reset, DANN Name abfragen ---
# Durch reset() wird der Gegner erst erstellt.
obs, _ = env.reset()

# Jetzt existiert 'env.opponent' und wir können den Namen lesen
real_opponent = env.unwrapped.opponent.name if hasattr(env, "unwrapped") else env.opponent.name
print(f"\n--- Test gegen: {real_opponent} ---")

total_reward = 0

for i in range(50):
    action, _ = model.predict(obs, deterministic=True)
    
    # Wichtig: int() Umwandlung
    action_int = int(action)
    
    obs, reward, terminated, truncated, info = env.step(action_int)
    total_reward += reward
    
    my_move = "😇 Coop" if action_int == 0 else "😈 Defect"
    opp_move = "😇 Coop" if info["opponent_move"] == 0 else "😈 Defect"
    
    print(f"Runde {i+1:02d}: Ich: {my_move} | Gegner: {opp_move} | Reward: {reward}")

print(f"Gesamtpunkte: {total_reward}")
