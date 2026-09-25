import pandas as pd
import pulp

def optimize_squad():
    print("Lade KI-berechnete Spielerdaten...")
    # Wir laden jetzt die AI-Daten statt der Basis-Daten
    df = pd.read_csv('data/ai_players.csv')
    
    # Der Optimizer nutzt die KI-Prediction als Zielwert
    df['expected_points'] = df['AI_xP'] 
    
    df = df[df['expected_points'] > 0].reset_index(drop=True)

    prob = pulp.LpProblem("FPL_AI_Optimal_Squad", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("players", df.index, cat='Binary')

    prob += pulp.lpSum([df.loc[i, 'expected_points'] * player_vars[i] for i in df.index]), "Total_Expected_Points"
    prob += pulp.lpSum([df.loc[i, 'now_cost'] * player_vars[i] for i in df.index]) <= 100.0, "Budget"
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 15, "Total_Players"
    
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'GK']) == 2, "GK_Count"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'DEF']) == 5, "DEF_Count"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'MID']) == 5, "MID_Count"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'FWD']) == 3, "FWD_Count"

    teams = df['team_name'].unique()
    for team in teams:
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'team_name'] == team]) <= 3, f"Team_Limit_{team}"

    print("Berechne optimales KI-Team...")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != 'Optimal':
        print("Fehler: Keine optimale Lösung gefunden.")
        return

    chosen_indices = [i for i in df.index if player_vars[i].varValue == 1.0]
    squad = df.loc[chosen_indices].sort_values(by='position', ascending=True)

    print("\n--- OPTIMALES KI-FPL TEAM ---")
    total_cost = squad['now_cost'].sum()
    total_xp = squad['expected_points'].sum()

    for _, player in squad.iterrows():
        print(f"[{player['position']}] {player['web_name']} ({player['team_name']}) - £{player['now_cost']}m | KI-xP: {player['expected_points']:.1f}")

    print("-" * 30)
    print(f"Total Cost: £{total_cost:.1f}m")
    print(f"Expected Points (AI): {total_xp:.1f}")

if __name__ == "__main__":
    optimize_squad()