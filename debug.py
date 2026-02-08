from environment import AXLPrisonersDilemmaEnv
import numpy as np

print("🔍 DEBUG STARTET...")

# 1. Environment erstellen
env = AXLPrisonersDilemmaEnv()

# 2. Was behauptet das Environment, wie groß es ist?
planned_size = env.observation_space.shape[0]
print(f"📋 Observation Space sagt: {planned_size}")

# 3. Was kommt WIRKLICH raus?
real_obs, _ = env.reset()
real_size = real_obs.shape[0]
print(f"🎲 Reset() liefert tatsächlich: {real_size}")

if planned_size != real_size:
    print("\n🚨 ALARM: Die beiden Zahlen stimmen nicht überein!")
    print("Das bedeutet, 'HISTORY_LENGTH' wird in __init__ und reset() unterschiedlich benutzt.")
else:
    print("\n✅ Alles synchron. Das Problem muss woanders liegen.")