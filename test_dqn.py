import torch
import torch.nn as nn
import numpy as np
from environment import AXLPrisonersDilemmaEnv, device
from dqn import DQN
from cc_learn_advanced import DuelingDQN
import matplotlib.pyplot as plt

my_moves = []
opp_moves = []
my_scores = []
opp_scores = []
rounds = 50

opponent_name = "Tit For Tat" 
#opponent_name = "Defector"
#opponent_name = "Cooperator"
# Environment erstellen
env = AXLPrisonersDilemmaEnv(opponent_name=opponent_name, random_opponent=False, max_rounds=100)

print("Lade DQN Modell...")

# Leeres Netz erstellen
#model = DQN(20, 2).to(device)
model = DuelingDQN(20, 2).to(device)

try:
    # Gewichte laden
    model.load_state_dict(torch.load("dqn_agent.pth"))
    # In den Eval-Modus schalten, keine Anpassung moeglich
    model.eval()
    print("Modell erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden: {e}")
    exit()

state, _ = env.reset()

real_opponent = env.unwrapped.opponent.name if hasattr(env, "unwrapped") else env.opponent.name
print(f"\n--- Test gegen: {real_opponent} ---")

total_reward = 0

for i in range(rounds): #50 Runden spielen
    
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    
    with torch.no_grad(): #kein Gradient, da kein Training
        q_values = model(state_tensor)
        
        #Aktion mit dem höchsten Q-Wert wählen 
        action = q_values.argmax().item()

    state, reward, terminated, truncated, info = env.step(action)
    
    my_move = "😇 Coop" if action == 0 else "😈 Defect"
    opp_move = "😇 Coop" if info["opponent_move"] == 0 else "😈 Defect"
    
    
    q_val_list = q_values.cpu().numpy()[0]
    print(f"Runde {i+1:02d}: Ich: {my_move} | Gegner: {opp_move} | Reward: {reward} | Q-Werte: {q_val_list}")

    my_moves.append(action)
    opp_moves.append(info["opponent_move"])
    episode_start = np.zeros((1,), dtype=bool)

    my_scores.append(reward)
    
    opp_payoff_map = {(0,0): 3, (0,1): 5, (1,0): 0, (1,1): 1} # (Ich, Er) -> Seine Pkt
    
    opp_reward = opp_payoff_map[(action, info["opponent_move"])]
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
plt.ylabel(f"{real_opponent}    Mein Agent")
plt.ylim(-0.5, 1.5)
plt.title(f"Mein Agent gegen {real_opponent}")
plt.xlabel("Runde")
# Legende
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='green', label='Kooperation', markersize=10),
                   Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Verrat', markersize=10)]
plt.legend(handles=legend_elements, loc='right')

plt.tight_layout()
plt.show()