import torch
import torch.nn as nn
import numpy as np
from environment import AXLPrisonersDilemmaEnv, device
from dqn import DQN

#opponent_name = "Tit For Tat" # Test-Gegner (Versuch auch "Random" oder "Grudger")
#opponent_name = "Defector"
opponent_name = "Cooperator"
# Environment erstellen
env = AXLPrisonersDilemmaEnv(opponent_name=opponent_name, random_opponent=False, max_rounds=100)

print("🔄 Lade DQN Modell...")

# Leeres Netz erstellen
model = DQN(20, 2).to(device)

try:
    # Gewichte laden
    model.load_state_dict(torch.load("dqn_agent.pth"))
    # WICHTIG: In den Eval-Modus schalten (deaktiviert Dropout etc., falls vorhanden)
    model.eval()
    print("✅ Modell erfolgreich geladen.")
except Exception as e:
    print(f"❌ Fehler beim Laden: {e}")
    exit()

# --- 3. Der Test-Loop ---
# Resetten, damit der Gegner erstellt wird
state, _ = env.reset()

# Echten Gegner-Namen holen
real_opponent = env.unwrapped.opponent.name if hasattr(env, "unwrapped") else env.opponent.name
print(f"\n--- Test gegen: {real_opponent} ---")

total_reward = 0

for i in range(50):
    # A) Daten vorbereiten
    # State ist np.array([0, 0]) -> Wir brauchen Tensor [[0., 0.]] (Batch Size 1)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # B) Netz fragen
    with torch.no_grad(): # WICHTIG: Wir lernen hier nicht, also keine Gradienten speichern
        q_values = model(state_tensor)
        
        # Die Aktion mit dem höchsten Q-Wert wählen (Argmax)
        action = q_values.argmax().item()

    # C) Schritt ausführen
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # D) Ausgabe
    my_move = "😇 Coop" if action == 0 else "😈 Defect"
    opp_move = "😇 Coop" if info["opponent_move"] == 0 else "😈 Defect"
    
    # Optional: Q-Werte anzeigen, um zu sehen, wie sicher sich das Netz ist
    q_val_list = q_values.cpu().numpy()[0]
    print(f"Runde {i+1:02d}: Ich: {my_move} | Gegner: {opp_move} | Reward: {reward} | Q-Werte: {q_val_list}")

print(f"\nGesamtpunkte: {total_reward}")