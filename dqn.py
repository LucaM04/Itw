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

##Variante 1  mit selbstgebautem DQN
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer: #speichert zustand, aktion, reward, nächsten Zustand für das Training
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

##training Variante 1
def train_dqn():
    print("\n START: DQN Training...")
    # Hyperparameter 
    LEARNING_RATE = 0.0005
    GAMMA = 0.99
    EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.01, 0.9995
    BATCH_SIZE = 128
    TOTAL_EPISODES = 5000 

    env = AXLPrisonersDilemmaEnv(random_opponent=True, max_rounds=200)
    input_dim = env.observation_space.shape[0] 
    print(f"ℹ Erkanntes Input-Format: {input_dim}") # Sollte 20 sein

    policy_net = DQN(input_dim=input_dim, output_dim=2).to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayBuffer(10000) 
    epsilon = EPSILON_START

    for episode in range(TOTAL_EPISODES): #Training
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        done = False
        if((episode%100)==0):
            print(f"{(episode/TOTAL_EPISODES*100):.0f} %")  #Fortschritt ausgeben
        
        while not done: #Zuerst zufällig Aktion, dann immer oefter Aktion aus DQN
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()
            
            next_state_np, reward, terminated, truncated, _ = env.step(action)  #Aktion ausfuehren
            done = terminated or truncated
            
            next_state = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0).to(device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            action_tensor = torch.tensor([action], dtype=torch.long).to(device)
            done_tensor = torch.tensor([done], dtype=torch.float32).to(device)
            
            memory.push(state, action_tensor, reward_tensor, next_state, done_tensor)   #in ReplayBuffer speichern
            state = next_state

            # lernen
            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
            
            #cat fuegt einzelne Werte korrekt zusammen [64, 20]  
                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
            
            #fuer Actions, Rewards, Dones ist stack okay, weil sie [1] oder scalar sind
                actions = torch.stack(actions).to(device)
                rewards = torch.stack(rewards).to(device).view(-1, 1)
                dones = torch.stack(dones).to(device).view(-1, 1)

                current_q = policy_net(states).gather(1, actions) #Q-wert berechen
            
                with torch.no_grad():
                    max_next_q = policy_net(next_states).max(1)[0].view(-1, 1)  #aus nachfolgenden zustaenden beste Aktionen berechnen
                    target_q = rewards + (GAMMA * max_next_q * (1 - dones)) #daraus und je nach Gewichtung der Zukunft, den besten
                                                                            #Q-Wert berechen
            
                loss = nn.MSELoss()(current_q, target_q)    #daruas Loss berechnen
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    #Anpassung
        
                epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    print("✅ DQN Training beendet. Speichere...")
    torch.save(policy_net.state_dict(), "dqn_agent.pth") 
    return policy_net

#Traing loop fuer weitere Trainings
def run_training_loop(env, policy_net, target_net, optimizer, memory, episodes, epsilon_start, epsilon_end, gamma):
    epsilon = epsilon_start
    epsilon_decay = (epsilon_end / epsilon_start) ** (1 / episodes) if epsilon_start > 0 else 0 #decay automatisch anpassen
    BATCH_SIZE = 128
    writer = SummaryWriter(log_dir="./dqn_tensorboard") #bessere Auswertung des Trainingsverlaufs
    steps_done = 0
    TARGET_UPDATE_FREQ = 10 

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        done = False
        total_reward = 0
        
        if((episode % 100) == 0):
            print(f"{(episode / episodes * 100):.0f} %")
        
        while not done:
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state).argmax().item()
            
            
            next_state_np, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            next_state = torch.tensor(next_state_np, dtype=torch.float32).unsqueeze(0).to(device)
            reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
            action_tensor = torch.tensor([action], dtype=torch.long).to(device)
            done_tensor = torch.tensor([done], dtype=torch.float32).to(device)
            
            memory.push(state, action_tensor, reward_tensor, next_state, done_tensor)
            state = next_state

            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                
                states = torch.cat(states).to(device)
                next_states = torch.cat(next_states).to(device)
                actions = torch.stack(actions).to(device)
                rewards = torch.stack(rewards).to(device).view(-1, 1)
                dones = torch.stack(dones).to(device).view(-1, 1)

                current_q = policy_net(states).gather(1, actions)
                
                with torch.no_grad():
                    #Berchnung der optimalen Werte mit stabilem Netz, um diese nicht durch aktuelles training zu veraendern
                    max_next_q = target_net(next_states).max(1)[0].view(-1, 1)
                    
                    target_q = rewards + (gamma * max_next_q * (1 - dones))
                
                
                loss = nn.MSELoss()(current_q, target_q)
                if loss is not None:
                    writer.add_scalar("train/loss", loss.item(), steps_done)    #loss und steps an logger senden
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0) #stabileres Training durch Gradient Clipping
                optimizer.step()
            steps_done+=1
        
        
        writer.add_scalar("rollout/ep_rew_mean", total_reward, steps_done) 
        writer.add_scalar("train/epsilon", epsilon, steps_done)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict()) #target net synchronisieren

        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    writer.close()

##Version 2 normales training
def train_ppo():    
    print("\n START: PPO Training...")
    env = AXLPrisonersDilemmaEnv(random_opponent=True, max_rounds=200)
    model = PPO("MlpPolicy", env, gamma=0.99, ent_coef=0.05, learning_rate=0.001, verbose=0) # verbose=0 für weniger Spam
    model.learn(total_timesteps=1000000) 
    print("✅ PPO Training beendet. Speichere...")
    model.save("ppo_agent") # Speichert als ppo_agent.zip
    return model

##Version 2 curriculum  training unnötig nach cc_learn
def train_ppo_curriculum1():    #erste Version des Curriculum learning
    env_start = AXLPrisonersDilemmaEnv(opponent_name="Tit For Tat", random_opponent=False) #nur gegen TitForTat trainieren
    model = PPO("MlpPolicy", env_start, learning_rate=0.001)
    model.learn(total_timesteps=100000)
    model.save("ppo_start")
    model = PPO.load("ppo_start")
    env_real = AXLPrisonersDilemmaEnv(random_opponent=True) #gegen mehrere Gegner trainieren
    model.learning_rate = 0.0001 
    model.set_env(env_real)
    model.learn(total_timesteps=200000)
    model.save("ppo_agent")

##Version 1 Spieler 
class DQNPlayer(ax.Player):
    name = "Mein Smart DQN"
    
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.model.eval()

    def strategy(self, opponent):
        # Historie aufbereiten 
        my_hist = [AX_TO_RL[str(m)] for m in self.history]
        opp_hist = [AX_TO_RL[str(m)] for m in opponent.history]
        
        # Nur letzte 10
        my_hist = my_hist[-HISTORY_LENGTH:]
        opp_hist = opp_hist[-HISTORY_LENGTH:]
        
        # Auffüllen mit 0
        if len(my_hist) < HISTORY_LENGTH:
            padding = [0] * (HISTORY_LENGTH - len(my_hist))
            my_hist = padding + my_hist
            opp_hist = padding + opp_hist
            
        # Input Tensor bauen (Größe 20)
        obs = np.array(my_hist + opp_hist, dtype=np.float32)
        state_tensor = torch.tensor(obs).unsqueeze(0).to(self.device)

        # 3. Vorhersage
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action_index = q_values.argmax().item()

        return RL_TO_AX[int(action_index)]
    

##Version 3 Spieler
class DuelingDQNPlayer(ax.Player):
    name = "Mein Dueling DQN"
    
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.model.eval()

    def strategy(self, opponent):
        # Historie aufbereiten 
        my_hist = [AX_TO_RL[str(m)] for m in self.history]
        opp_hist = [AX_TO_RL[str(m)] for m in opponent.history]
        
        # Nur letzte 10
        my_hist = my_hist[-HISTORY_LENGTH:]
        opp_hist = opp_hist[-HISTORY_LENGTH:]
        
        # Auffüllen mit 0
        if len(my_hist) < HISTORY_LENGTH:
            padding = [0] * (HISTORY_LENGTH - len(my_hist))
            my_hist = padding + my_hist
            opp_hist = padding + opp_hist
            
        # Input Tensor bauen (Größe 20)
        obs = np.array(my_hist + opp_hist, dtype=np.float32)
        state_tensor = torch.tensor(obs).unsqueeze(0).to(self.device)

        # 3. Vorhersage
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action_index = q_values.argmax().item()

        return RL_TO_AX[int(action_index)]


##Version 2 Spieler 
class SB3Player(ax.Player):
    name = "Mein Smart PPO"

    def __init__(self, model):
        super().__init__()
        self.model = model

    def strategy(self, opponent):
        # gleiche Logik wie beim DQN Player
        my_hist = [AX_TO_RL[str(m)] for m in self.history]
        opp_hist = [AX_TO_RL[str(m)] for m in opponent.history]
        
        my_hist = my_hist[-HISTORY_LENGTH:]
        opp_hist = opp_hist[-HISTORY_LENGTH:]
        
        if len(my_hist) < HISTORY_LENGTH:
            padding = [0] * (HISTORY_LENGTH - len(my_hist))
            my_hist = padding + my_hist
            opp_hist = padding + opp_hist
            
        obs = np.array(my_hist + opp_hist) 
            
        action, _ = self.model.predict(obs, deterministic=True)
        action=action.item()
        return RL_TO_AX[int(action)]
    
## Version 4 Spieler
class SB3RecurrentPlayer(ax.Player):    #Recurrent PPO Player für Axelrod Turnier
    name = "Mein Smart LSTM"

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Speicher für das LSTM
        self.lstm_states = None
        # Flag: Ist das der Start eines Spiels?
        self.episode_start = True

    def reset(self):
        """Wird von axelrod automatisch vor jedem neuen Match aufgerufen."""
        super().reset()
        # Gedächtnis löschen, damit neuer Gegner erkannt werden kann
        self.lstm_states = None
        self.episode_start = True

    def strategy(self, opponent):   #diese Funktion muss jeder Speiler in Axelrod besitzen
        # wie im Environment
        my_hist = [AX_TO_RL[str(m)] for m in self.history]
        opp_hist = [AX_TO_RL[str(m)] for m in opponent.history]
        
        my_hist = my_hist[-HISTORY_LENGTH:]
        opp_hist = opp_hist[-HISTORY_LENGTH:]
        
        if len(my_hist) < HISTORY_LENGTH:
            padding = [0] * (HISTORY_LENGTH - len(my_hist))
            my_hist = padding + my_hist
            opp_hist = padding + opp_hist
            
        obs = np.array(my_hist + opp_hist)
        
        # SB3 Modell erwartet (Batch_Size, Input_Dim), also (1, 20)
        obs = obs.reshape(1, -1)

        #Vorhersage
        action, self.lstm_states = self.model.predict(
            obs, 
            state=self.lstm_states, 
            episode_start=self.episode_start, 
            deterministic=True
        )
        
        
        self.episode_start = False  #Spiel hat begonnen
        action=action.item()
        
        return RL_TO_AX[int(action)]

def visualize_results(results, players):    #Verschiedene Diagramme zur Auswertung des Turniers
    
    sns.set_theme(style="whitegrid")
    
    # Namen der Spieler holen
    names = [p.name for p in players]

    
    print(" Erstelle Heatmap aller Matches...") 
    
    raw_matrix = np.array(results.payoff_matrix, dtype=float)
    
    # falls Dimension nicht passt, hier lagen viele Probleme
    if raw_matrix.ndim == 3:
        payoff_matrix = np.mean(raw_matrix, axis=2)
    else:
        payoff_matrix = raw_matrix
        
    
    plt.figure(figsize=(20, 14))    #Heatmap erzeugen, die durchschnittliche Punkte jeder Paarung zeigt
    sns.heatmap(
        payoff_matrix, 
        annot=True, 
        fmt=".2f", 
        xticklabels=names, 
        yticklabels=names,
        cmap="RdYlGn", 
        vmin=0, vmax=5 
    )
    plt.title("Payoff Matrix (Punkte pro Zug)")
    plt.tight_layout()
    plt.show()

    #Boxplot des Gessamtrankings
    print(" Erstelle Ranking-Boxplot...")
    
    fig, ax_plot = plt.subplots(figsize=(30, 12))
    
    # in Axelrod bib enthalten
    plot = ax.Plot(results)
    plot.boxplot(ax=ax_plot)
    
    plt.title("Verteilung der Scores")
    plt.xlabel("Strategie")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Auswertung für die trainierten Modelle anzeigen
    my_agents = [p for p in players if "Mein" in p.name or "DQN" in p.name or "PPO" in p.name]  #Modelle suchen
    
    for agent in my_agents:
        print(f" Analyse für: {agent.name}...")
        
        try:
            idx = players.index(agent)
            scores = payoff_matrix[idx]
            fig_height = max(6, len(names) * 0.4)
            
            # Farben
            cols = ['red' if x < 2.0 else 'orange' if x < 2.8 else 'green' for x in scores]
            
            plt.figure(figsize=(12, fig_height))
            
        
            sns.barplot(y=names, x=scores, hue=names, legend=False, palette=cols)
            
            
            plt.axvline(3.0, color='blue', linestyle='--', label="Kooperation ")
            plt.axvline(1.0, color='black', linestyle='--', label="Defect ")
            
            plt.xlim(0, 5.5) 
            plt.title(f"Performance von {agent.name}")
            plt.xlabel("Durchschnittlicher Score")
            plt.ylabel("Gegner")
            
            plt.tight_layout()
            plt.show()
        except: pass
            

    plot = ax.Plot(results) #Evolutionssimulation aus der Axelrod bib
    eco = ax.Ecosystem(results)

    eco.reproduce(10000) 

    fig, ax_obj = plt.subplots(figsize=(12, 8))
    plot.stackplot(eco, ax=ax_obj)  

    plt.title("Evolutionäre Entwicklung")
    plt.xlabel("Generationen")
    plt.ylabel("Marktanteil der Strategie")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()


    print(" Erstelle Ranking-Tabelle...")
    
    # 1. Daten sammeln und MANUELL sortieren
    original_names = [str(p) for p in players]
    score_data = []
    
    # Wir berechnen erst alle Punkte
    for idx, name in enumerate(original_names):
        total_score = int(np.sum(results.scores[idx]))
        clean_name = players[idx].name
        score_data.append((clean_name, total_score))
        
    # HIER IST DER FIX: Wir sortieren die Liste mathematisch absteigend nach Punkten!
    score_data.sort(key=lambda x: x[1], reverse=True)
    
    # 2. Tabelle vorbereiten
    table_data = []
    
    for rank, (clean_name, total_score) in enumerate(score_data):
        
        # Ränge formatieren (ohne Emojis wegen Windows-Font-Problemen)
        rank_str = f"{rank + 1}."
        
        # Deine Agenten erkennen
        is_my_agent = "Mein" in clean_name or "DQN" in clean_name or "PPO" in clean_name or "LSTM" in clean_name
        marker = "► " if is_my_agent else ""
        
        # Tausendertrennzeichen hinzufügen (z.B. 12.345)
        formatted_score = f"{total_score:,}".replace(',', '.')
        
        table_data.append([
            rank_str, 
            f"{marker}{clean_name}", 
            formatted_score
        ])
        
    # 3. Plot für die Tabelle aufbauen
    fig_height = max(5, len(table_data) * 0.35 + 1.5) 
    fig_table, ax_table = plt.subplots(figsize=(10, fig_height))
    
    ax_table.axis('tight')
    ax_table.axis('off')
    
    # Spaltenüberschriften
    columns = ["Platz", "Strategie", "Gesamtpunkte"]
    
    # Tabelle zeichnen
    table = ax_table.table(
        cellText=table_data, 
        colLabels=columns, 
        loc='center', 
        cellLoc='center'
    )
    
    # 4. Styling der Tabelle (Premium Optik)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.0) 
    
    # Spaltenbreiten automatisch anpassen
    table.auto_set_column_width([0, 1, 2])
    
    # Farben anpassen
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd') 
        
        if row == 0:
            # Header-Zeile: Edles, dunkles Blau-Grau
            cell.set_text_props(weight='bold', color='white', fontsize=13)
            cell.set_facecolor('#2c3e50') 
        else:
            # Abwechselnde Zeilenfarben (Zebra-Muster)
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')
                
            # Deine Agenten extrem edel hervorheben
            cell_text = table_data[row-1][1]
            if "►" in cell_text:
                cell.set_facecolor('#d4edda') 
                cell.set_edgecolor('#c3e6cb') 
                
                if col == 1: 
                    cell.set_text_props(weight='bold', color='#155724')
                if col == 2: 
                    cell.set_text_props(weight='bold')

    plt.title("Finales Turnier-Ranking", fontsize=18, pad=20, weight='bold', color='#2c3e50')
    plt.tight_layout()
    plt.show()