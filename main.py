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
from sb3_contrib import RecurrentPPO
import matplotlib.pyplot as plt
import seaborn as sns
from environment import AXLPrisonersDilemmaEnv, RL_TO_AX, AX_TO_RL
from dqn import DQN, ReplayBuffer, train_dqn, train_ppo, DQNPlayer,DuelingDQNPlayer, SB3Player,SB3RecurrentPlayer, visualize_results, train_ppo_curriculum1
from cc_learn import train_dqn_curriculum, run_training_loop, train_ppo_curriculum
from cc_learn_advanced import train_dueling_dqn_curriculum, train_recurrent_ppo_curriculum, DuelingDQN

if __name__ == "__main__":
    # Trainieren (oder auskommentieren, wenn schon trainiert)
    #train_dqn()
    #train_ppo()
    #train_ppo_curriculum()
    #train_dqn_curriculum()
    #train_recurrent_ppo_curriculum()
    #train_dueling_dqn_curriculum()


    # Turnier Vorbereitung
    print("\n Bereite das große Turnier vor...")
    device = torch.device("cpu") #turnier läuft auf CPU schneller als auf GPU, da Datentransport kleiner
    
    # Gegner laden
    #players = [s() for s in ax.demo_strategies]
    #players = [s() for s in ax.strategies]
    players = [s() for s in ax.axelrod_first_strategies]
    ##players.append(ax.Cooperator())

    # PPO Laden & Hinzufügen
    print(" Lade PPO Modell...")
    try:
        
        ppo_model = RecurrentPPO.load("ppo_agent", device="cpu") 
        my_ppo_rec_player = SB3RecurrentPlayer(ppo_model)
        players.append(my_ppo_rec_player)
        ppo_model = PPO.load("ppo_agent_cc1", device="cpu") 
        my_ppo_cc_player = SB3Player(ppo_model)
        players.append(my_ppo_cc_player)
    except Exception as e:
        print(f" PPO Fehler: {e}")

    # DQN Laden & Hinzufügen
    print(" Lade DQN Modell...")
    temp_env = AXLPrisonersDilemmaEnv() 
    input_dim = temp_env.observation_space.shape[0]

    duel_dqn_net = DuelingDQN(input_dim, 2).to(device)
    try:
        duel_dqn_net.load_state_dict(torch.load("dqn_agent.pth", map_location=device, weights_only=True))
        # map_location=device stellt sicher, dass Gewichte auf die CPU geladen werden
        my_dueling_dqn_player = DuelingDQNPlayer(duel_dqn_net, device)
        players.append(my_dueling_dqn_player)
    except Exception as e:
        print(f" DQN Fehler: {e}")

    dqn_net = DQN(input_dim, 2).to(device)
    try:
        dqn_net.load_state_dict(torch.load("dqn_agent_cc1.pth", map_location=device, weights_only=True))
        # map_location=device stellt sicher, dass Gewichte auf die CPU geladen werden
        my_cc_dqn_player = DQNPlayer(dqn_net, device)
        players.append(my_cc_dqn_player)
    except Exception as e:
        print(f" DQN Fehler: {e}")

    # Turnier Start
    print(f" Starte Kämpfe zwischen {len(players)} Strategien...")
    tournament = ax.Tournament(players, turns=200, repetitions=5)
    results = tournament.play(processes=1, progress_bar=True)

    # Auswertung
    print("\n" + "="*30)
    print("       LEADERBOARD       ")
    print("="*30)

    ranked_names = results.ranked_names
    for rank, name in enumerate(ranked_names):
        marker = ""
        if "PPO" in name: marker = "👈 (PPO)"
        if "DQN" in name: marker = "👈 (DQN)"
        print(f"Platz {rank+1:02d}: {name} {marker}")

    print("\n" + "-"*30)
    # Scores sicher ausgeben
    def safe_print_score(agent):
        try:
            if agent in players:
                idx = players.index(agent)
                avg = sum(results.scores[idx]) / len(results.scores[idx]) / 100
                print(f" > {agent.name}: {avg:.4f} Punkte/Runde")
        except: pass

    #if 'my_ppo_player' in locals(): safe_print_score(my_ppo_player)
    #if 'my_dqn_player' in locals(): safe_print_score(my_dqn_player)


    visualize_results(results, players)

