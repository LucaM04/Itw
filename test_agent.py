from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from environment import AXLPrisonersDilemmaEnv, device
import numpy as np
from dqn import DQN
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

my_moves = []
opp_moves = []
my_scores = []
opp_scores = []
rounds = 50

opponent_name = "Tit For Tat"
#opponent_name = "Defector"


# Environment erstellen
env = AXLPrisonersDilemmaEnv(opponent_name=opponent_name, random_opponent=False, max_rounds=100)

print(" Lade PPO Modell...")
model = PPO.load("ppo_agent_cc1")


obs, _ = env.reset()

real_opponent = env.unwrapped.opponent.name if hasattr(env, "unwrapped") else env.opponent.name
print(f"\n--- Test gegen: {real_opponent} ---")

total_reward = 0

for i in range(rounds): #50 Runden gegen einen Gegner spielen
    action, _ = model.predict(
        obs, 
        deterministic=True
    )
    
    action_int = int(action)
    
    obs, reward, terminated, truncated, info = env.step(action_int)
    
    my_moves.append(action_int)
    opp_moves.append(info["opponent_move"])
    episode_start = np.zeros((1,), dtype=bool)

    
    my_move = "😇 Coop" if action_int == 0 else "😈 Defect"
    opp_move = "😇 Coop" if info["opponent_move"] == 0 else "😈 Defect"
    
    print(f"Runde {i+1:02d}: Ich: {my_move} | Gegner: {opp_move} | Reward: {reward}")

    my_scores.append(reward)
    
    opp_payoff_map = {(0,0): 3, (0,1): 5, (1,0): 0, (1,1): 1} # (Ich, Er) -> Seine Pkt
    
    opp_reward = opp_payoff_map[(action_int, info["opponent_move"])]
    opp_scores.append(opp_reward)

print(f"Gesamtpunkte Mein Agent: {sum(my_scores)} | Gegner: {sum(opp_scores)}")

plt.figure(figsize=(12, 3)) 
x = range(rounds)

# Meine Züge 
colors_me = ['green' if m == 0 else 'red' for m in my_moves]
plt.scatter(x, [1]*rounds, c=colors_me, s=100, marker='s', label="Ich")

# Gegner Züge 
colors_opp = ['green' if m == 0 else 'red' for m in opp_moves]
plt.scatter(x, [0]*rounds, c=colors_opp, s=100, marker='s', label="Gegner")

# Beschriftung
plt.yticks([0, 1])
plt.ylabel(f"{real_opponent} Mein Agent")
plt.ylim(-0.5, 1.5)
plt.title(f"Mein Agent gegen {real_opponent}")
plt.xlabel("Runde")
# Legende
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Kooperation', markersize=10),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Verrat', markersize=10)]
plt.legend(handles=legend_elements, loc='right')

plt.tight_layout()
plt.show()