import pandas as pd
import pulp
import json

def optimize_transfers(free_transfers=1, max_transfers=2):
    print("Lade KI-Daten, Team und Budget...")
    df = pd.read_csv('data/ai_players.csv')
    df['expected_points'] = df['AI_xP']
    df = df[df['expected_points'] >= 0].reset_index(drop=True)

    my_team = pd.read_csv('data/my_team.csv')
    my_player_ids = my_team['id'].tolist()
    
    # NEU: Lade das gespeicherte Budget
    with open('data/manager_info.json', 'r') as f:
        manager_info = json.load(f)
        
    current_team_indices = df[df['id'].isin(my_player_ids)].index.tolist()
    
    # Dein echtes Transfer-Budget
    current_team_cost = df.loc[current_team_indices, 'now_cost'].sum()
    max_budget = current_team_cost + manager_info['bank']

    prob = pulp.LpProblem("FPL_Transfer_Optimizer", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("players", df.index, cat='Binary')
    hits = pulp.LpVariable("hits", lowBound=0, cat='Integer')

    prob += pulp.lpSum([df.loc[i, 'expected_points'] * player_vars[i] for i in df.index]) - (4 * hits), "Objective"

    kept_players = pulp.lpSum([player_vars[i] for i in current_team_indices])
    transfers_made = 15 - kept_players
    
    prob += transfers_made <= max_transfers, "Max_Transfers_Limit"
    prob += hits >= transfers_made - free_transfers, "Hit_Calculation"

    # NEU: Das dynamische Budget-Limit statt 100.0m
    prob += pulp.lpSum([df.loc[i, 'now_cost'] * player_vars[i] for i in df.index]) <= max_budget, "Budget"
    
    prob += pulp.lpSum([player_vars[i] for i in df.index]) == 15, "Total_Players"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'GK']) == 2, "GK"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'DEF']) == 5, "DEF"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'MID']) == 5, "MID"
    prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'position'] == 'FWD']) == 3, "FWD"

    for team in df['team_name'].unique():
        prob += pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'team_name'] == team]) <= 3, f"Team_{team}"

    print(f"Transfer Budget: £{max_budget:.1f}m")
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    chosen_indices = [i for i in df.index if player_vars[i].varValue == 1.0]
    players_out = [df.loc[i, 'web_name'] for i in current_team_indices if i not in chosen_indices]
    players_in = [df.loc[i, 'web_name'] for i in chosen_indices if i not in current_team_indices]

    print("\n--- TRANSFER EMPFEHLUNG ---")
    if not players_out:
        print("Keine Transfers empfohlen. Dein Team ist mathematisch optimal!")
    else:
        print(f"Verkaufen ({len(players_out)}): {', '.join(players_out)}")
        print(f"Kaufen    ({len(players_in)}): {', '.join(players_in)}")
        print(f"Transfer Hits (Kosten): {int(hits.varValue)} (-{int(hits.varValue * 4)} Punkte)")
        
    new_team_cost = df.loc[chosen_indices, 'now_cost'].sum()
    print(f"\nNeues Restbudget auf der Bank: £{max_budget - new_team_cost:.1f}m")

if __name__ == "__main__":
    optimize_transfers(free_transfers=1, max_transfers=2)