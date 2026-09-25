import requests
import pandas as pd
import json

def fetch_my_team(manager_id):
    print(f"Verbinde mit FPL Servern für Manager ID: {manager_id}...")
    
    bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    boot_res = requests.get(bootstrap_url).json()
    
    current_gw = 1
    for event in boot_res['events']:
        if event['is_current'] or event['is_next']:
            current_gw = event['id']
            if event['is_current']:
                break
                
    print(f"Lade Kader für Gameweek {current_gw}...")
    picks_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{current_gw}/picks/"
    picks_res = requests.get(picks_url)
    
    if picks_res.status_code != 200:
        print(f"Fehler: Team nicht gefunden. Status Code: {picks_res.status_code}")
        return
        
    picks_data = picks_res.json()
    
    # --- NEU: Kontostand (Bank) aus der API lesen ---
    # FPL gibt Geldwerte in 0.1m an (also 15 = 1.5m)
    bank_money = picks_data['entry_history']['bank'] / 10.0
    team_value = picks_data['entry_history']['value'] / 10.0
    
    manager_info = {
        "manager_id": manager_id,
        "bank": bank_money,
        "team_value": team_value
    }
    
    with open('data/manager_info.json', 'w') as f:
        json.dump(manager_info, f)
    # ------------------------------------------------
    
    my_picks = pd.DataFrame(picks_data['picks'])
    my_picks = my_picks.rename(columns={'position': 'squad_position'})
    clean_players = pd.read_csv('data/clean_players.csv')
    
    my_team = pd.merge(my_picks, clean_players, left_on='element', right_on='id')
    my_team['is_benched'] = my_team['multiplier'] == 0
    my_team.to_csv('data/my_team.csv', index=False)
    
    print(f"Erfolg! Team & Budget gespeichert.")
    print("-" * 30)
    print(f"DEIN KADER (Geld auf der Bank: £{bank_money}m | Gesamtwert: £{team_value}m)")
    for _, player in my_team.sort_values(by='position').iterrows():
        status = "(BANK)" if player['is_benched'] else ""
        print(f"[{player['position']}] {player['web_name']} - £{player['now_cost']}m {status}")

if __name__ == "__main__":
    MANAGER_ID = 1 
    fetch_my_team(MANAGER_ID)