import gymnasium as gym
import numpy as np
from gymnasium import spaces
import axelrod as ax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import seaborn as sns
from environment import AXLPrisonersDilemmaEnv, device, RL_TO_AX, AX_TO_RL, HISTORY_LENGTH
from dqn import ReplayBuffer, run_training_loop
import tensorboard

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        
        # Durchschnitt abziehen ab, damit das System mathematisch stabil bleibt.
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_vals
    
def train_dueling_dqn_curriculum():
    
    print("\n PHASE 1: Schule (Lerne Kooperation gegen TitForTat)")
    
    
    env_school = AXLPrisonersDilemmaEnv(opponent_name="Tit For Tat", random_opponent=False, max_rounds=200)
    input_dim = env_school.observation_space.shape[0]
    
    
    policy_net = DuelingDQN(input_dim, 2).to(device)
    target_net = DuelingDQN(input_dim, 2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    memory = ReplayBuffer(10000)
    
    
    run_training_loop(
        env=env_school, 
        policy_net=policy_net, 
        target_net=target_net, 
        optimizer=optimizer, 
        memory=memory, 
        episodes=3000,       #  Phase 1
        epsilon_start=1.0,   
        epsilon_end=0.05, 
        gamma=0.995          
    )
    
    print(" Phase 1 abgeschlossen. Agent kann jetzt kooperieren.")
    torch.save(policy_net.state_dict(), "dqn_school_dueling.pth") # Zwischenspeichern

    
    print("\n PHASE 2: Universität (Gemischte Gegner)")
    
    
    env_uni = AXLPrisonersDilemmaEnv(random_opponent=True, max_rounds=200)
    
    
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001
        
    run_training_loop(
        env=env_uni, 
        policy_net=policy_net, 
        target_net=target_net, 
        optimizer=optimizer, 
        memory=memory,       # Alter Speicher hilft gegen Vergessen
        episodes=5000,       # Länger trainieren für die komplexeren Gegner
        epsilon_start=0.2,   # Nicht bei 1.0 starten
        epsilon_end=0.01,    # Ganz runter gehen
        gamma=0.995
    )
    
    print(" DQN Training komplett beendet.")
    torch.save(policy_net.state_dict(), "dqn_agent.pth")
    return policy_net

def train_recurrent_ppo_curriculum():
    print("\n PHASE 1: Lerne Kooperation")
    
    
    env_school = AXLPrisonersDilemmaEnv(opponent_name="Tit For Tat", random_opponent=False, max_rounds=200)
    
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env_school, 
        learning_rate=0.0003, 
        ent_coef=0.01, 
        n_steps=2048,
        batch_size=128,
        verbose=0, 
        tensorboard_log="./ppo_tensorboard/", 
        device="cpu")
    
    model.learn(total_timesteps=200000, tb_log_name="PPO_Curriculum_Phase1")
    model.save("ppo_school_rec")
    print("✅ PPO Phase 1 abgeschlossen.")

    # ==========================================
    
    print("\n PHASE 2: PPO Universität (Harte Realität)")
    
    
    env_uni = AXLPrisonersDilemmaEnv(random_opponent=True, max_rounds=200)
    
    model = RecurrentPPO.load("ppo_school_rec", env=env_uni)
    
    
    new_lr = 0.0001
    model.learning_rate = new_lr
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = new_lr
        
    print(f"  Lernrate auf {new_lr} gesenkt.")
    
    # 6. Weiterlernen
    model.learn(total_timesteps=600000, tb_log_name="PPO_Curriculum_Phase2")
    model.save("ppo_agent")
    print(" PPO Training komplett beendet.")
    
    return model