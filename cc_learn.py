import gymnasium as gym
import numpy as np
from gymnasium import spaces
import axelrod as ax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
from collections import deque
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
from environment import AXLPrisonersDilemmaEnv, device, RL_TO_AX, AX_TO_RL, HISTORY_LENGTH
from dqn import DQN, ReplayBuffer, run_training_loop
import tensorboard

def train_dqn_curriculum():
    
    print("\n PHASE 1: Schule (Lerne Kooperation gegen TitForTat)")
    
    #  Nur Tit For Tat
    env_school = AXLPrisonersDilemmaEnv(opponent_name="Tit For Tat", random_opponent=False, max_rounds=200)
    input_dim = env_school.observation_space.shape[0]
    
    # Netz erstellen
    policy_net = DQN(input_dim, 2).to(device)
    target_net = DQN(input_dim, 2).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    memory = ReplayBuffer(10000)
    
    
    run_training_loop(
        env=env_school, 
        policy_net=policy_net, 
        target_net=target_net, 
        optimizer=optimizer, 
        memory=memory, 
        episodes=3000,       
        epsilon_start=1.0,   
        epsilon_end=0.05, 
        gamma=0.995          # Weitsichtig
    )
    
    print(" Phase 1 abgeschlossen. Agent kann jetzt kooperieren.")
    torch.save(policy_net.state_dict(), "dqn_school.pth") # Zwischenspeichern

    
    print("\n PHASE 2: Universität (Gemischte Gegner)")
    
    
    env_uni = AXLPrisonersDilemmaEnv(random_opponent=True, max_rounds=200)
    
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001
        
    run_training_loop(
        env=env_uni, 
        policy_net=policy_net, 
        target_net=target_net, 
        optimizer=optimizer, 
        memory=memory,       # alten ReplayBuffer verwenden
        episodes=5000,       # Länger trainieren für die komplexeren Gegner
        epsilon_start=0.2,   # Wissen aus Phase 1 nutzen
        epsilon_end=0.01,    # Ganz runter gehen
        gamma=0.995
    )
    
    print(" DQN Training komplett beendet.")
    torch.save(policy_net.state_dict(), "dqn_agent.pth")
    return policy_net


def train_ppo_curriculum():
    print("\n PHASE 1: Lerne Kooperation")
    
    
    env_school = AXLPrisonersDilemmaEnv(opponent_name="Tit For Tat", random_opponent=False, max_rounds=200)
    
    model = PPO("MlpPolicy", env_school, learning_rate=0.001, ent_coef=0.01, verbose=1, tensorboard_log="./ppo_tensorboard/", device="cpu")
    
    model.learn(total_timesteps=200000, tb_log_name="PPO_Curriculum_Phase1")
    model.save("ppo_school")
    print(" PPO Phase 1 abgeschlossen.")

    
    print("\n PHASE 2: Gemischte Gegner")
    
    env_uni = AXLPrisonersDilemmaEnv(random_opponent=True, max_rounds=200)
    
    model = PPO.load("ppo_school", env=env_uni)
    
    
    new_lr = 0.0001 #lernrate senken für Feintuning
    model.learning_rate = new_lr
    for param_group in model.policy.optimizer.param_groups:
        param_group['lr'] = new_lr
        
    print(f"    Lernrate auf {new_lr} gesenkt.")
    
    
    model.learn(total_timesteps=600000, tb_log_name="PPO_Curriculum_Phase2")
    model.save("ppo_agent")
    print(" PPO Training komplett beendet.")
    
    return model