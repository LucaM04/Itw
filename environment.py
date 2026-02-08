import gymnasium as gym
import numpy as np
from gymnasium import spaces
import axelrod as ax
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from stable_baselines3 import PPO


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


HISTORY_LENGTH = 10  # 10 Züge * 2 Spieler = 20 Inputs
RL_TO_AX = {0: ax.Action.C, 1: ax.Action.D} #Uebersetzung zwischen Training und Axelrod Turnier
AX_TO_RL = {'C': 0, 'D': 1}

class AXLPrisonersDilemmaEnv(gym.Env):
    def __init__(self, opponent_name="Tit For Tat", random_opponent=False, max_rounds=100):
        super(AXLPrisonersDilemmaEnv, self).__init__()
        self.max_rounds = max_rounds
        self.random_opponent = random_opponent
        
        self.all_strategies = []
        for s in ax.strategies: #Axelrod Strategien laden
            try:
                _ = s()
                self.all_strategies.append(s)
            except: continue

        if self.random_opponent:
            self.learning_pool = [
                ax.TitForTat,        # Lehrt Kooperation
                ax.TitFor2Tats,      # Doppelt, damit er es öfter sieht
                ax.Cooperator,       # Ausbeutung 
                ax.Random,           # Robustheit
                ax.Defector,         # Selbstschutz 
                ax.WinStayLoseShift, # Verhalten je nach Erfolg anpassen
                ax.Grumpy            # Gut sein lohnt sich
            ]
            # nur definierte Gegner verwenden
            self.all_strategies = self.learning_pool
        
        self.opponent_class = ax.TitForTat
        if not random_opponent and opponent_name:
            class_map = {s.name: s for s in ax.strategies}
            if opponent_name in class_map:
                self.opponent_class = class_map[opponent_name]

        self.action_space = spaces.Discrete(2)  #Coop oder Defect
        self.observation_space = spaces.MultiDiscrete([2] * (2 * HISTORY_LENGTH))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_round = 0
        
        if self.random_opponent:
            self.opponent_class = random.choice(self.all_strategies)

        try:
            self.opponent = self.opponent_class()
        except:
            self.opponent = ax.TitForTat()
            
        self.opponent.reset()
        self.mock_opponent = ax.Player() 
        
        # Startzustand: 10 Runden beide Coop
        return np.zeros(2 * HISTORY_LENGTH, dtype=np.int32), {}

    def step(self, action):
        try:
            raw_move = self.opponent.strategy(self.mock_opponent)
        except:
            raw_move = ax.Action.C
            
        my_move_ax = RL_TO_AX[action]
        
        try:
            self.opponent.update_history(raw_move, my_move_ax)
            self.mock_opponent.update_history(my_move_ax, raw_move)
        except: pass

        # Observation holen 
        observation = self._get_observation()

        try:
            opponent_action = AX_TO_RL[str(raw_move)]
        except:
            opponent_action = 0

        self.current_round += 1
        reward = self._calculate_payoff(action, opponent_action) # Reward berechnen
        terminated = self.current_round >= self.max_rounds
        
        info = {"opponent_move": opponent_action, "opponent_name": getattr(self.opponent, "name", "Unknown")}
        
        return observation, reward, terminated, False, info

    def _get_observation(self):
        # Rohdaten holen
        my_hist = [AX_TO_RL[str(m)] for m in self.mock_opponent.history]
        opp_hist = [AX_TO_RL[str(m)] for m in self.opponent.history]
        
        # Padding falls noch nicht genug Runden gespielt wurden
        if len(my_hist) < HISTORY_LENGTH:
            needed = HISTORY_LENGTH - len(my_hist)
            padding = [0] * needed  #beide haben Kooperiert
            my_hist = padding + my_hist
            opp_hist = padding + opp_hist
            
        # Nur die letzten 10 Runden sind für die beiden Spieler sichtbar
        my_hist = my_hist[-HISTORY_LENGTH:]
        opp_hist = opp_hist[-HISTORY_LENGTH:]
        
        # Zusammenfügen
        obs = np.array(my_hist + opp_hist, dtype=np.int32)
        
        
        # Check, dass Dimension passt
        expected_size = 2 * HISTORY_LENGTH
        if obs.shape[0] != expected_size:
            # Fallback: Einfach Nullen zurückgeben, damit das Training nicht abstürzt
            
            return np.zeros(expected_size, dtype=np.int32)
            
        return obs

    def _calculate_payoff(self, action, opponent_action):
        if action == 0 and opponent_action == 0: return 3
        elif action == 1 and opponent_action == 1: return 1
        elif action == 1 and opponent_action == 0: return 5
        elif action == 0 and opponent_action == 1: return 0
        return 0