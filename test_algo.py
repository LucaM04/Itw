import axelrod as ax
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1. SETUP & SPIELER
rounds = 50
my_moves = []
opp_moves = []
my_scores = []
opp_scores = []

# Hier wählst du die beiden Kämpfer
player1 = ax.TitForTat()
player2 = ax.FirstByTidemanAndChieruzzi()
##First by Tideman and Chieruzzi
##First by Rapo

print(f"\n--- Duell: {player1.name} vs {player2.name} ---")

# 2. MATCH STARTEN
# Das Match wird erstellt und direkt gespielt
match = ax.Match([player1, player2], turns=rounds)
results = match.play() 


# 3. DATEN AUFBEREITEN (Die fehlende Schleife)
# Übersetzungstabelle: 'C' wird zu 0, 'D' wird zu 1
move_map = {'C': 0, 'D': 1}
# Punkteverteilung (Ich, Gegner) -> (Meine Pkt, Seine Pkt)
payoff_map = {(0, 0): (3, 3), (0, 1): (0, 5), (1, 0): (5, 0), (1, 1): (1, 1)}

for i, (m1_str, m2_str) in enumerate(results):
    m1 = move_map[str(m1_str)]
    m2 = move_map[str(m2_str)]
    
    my_moves.append(m1)
    opp_moves.append(m2)
    
    score1, score2 = payoff_map[(m1, m2)]
    my_scores.append(score1)
    opp_scores.append(score2)
    
    # Konsolenausgabe für den perfekten Überblick
    p1_text = "😇 Coop" if m1 == 0 else "😈 Defect"
    p2_text = "😇 Coop" if m2 == 0 else "😈 Defect"
    print(f"Runde {i+1:02d}: {player1.name}: {p1_text} | {player2.name}: {p2_text} | Punkte: {score1} : {score2}")

print(f"\n🏆 Gesamtpunkte -> {player1.name}: {sum(my_scores)} | {player2.name}: {sum(opp_scores)}")

# ==========================================
# 4. VISUALISIERUNG
# ==========================================
name_p1 = player1.name
name_p2 = player2.name

plt.figure(figsize=(12, 3)) 
x = range(rounds)

# Züge Spieler 1
colors_p1 = ['green' if m == 0 else 'red' for m in my_moves]
plt.scatter(x, [1]*rounds, c=colors_p1, s=100, marker='s', label=name_p1)

# Züge Spieler 2
colors_p2 = ['green' if m == 0 else 'red' for m in opp_moves]
plt.scatter(x, [0]*rounds, c=colors_p2, s=100, marker='s', label=name_p2)

# Beschriftung
plt.yticks([0, 1], [name_p2, name_p1], fontweight='bold') 
plt.ylim(-0.5, 1.5)
plt.title(f"Duell: {name_p1} vs {name_p2}", fontsize=14, pad=15)
plt.xlabel("Runde")

# Legende
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', label='Kooperation', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Verrat', markersize=10)
]
plt.legend(handles=legend_elements, loc='right', bbox_to_anchor=(1.15, 0.5))

plt.tight_layout()
plt.show()