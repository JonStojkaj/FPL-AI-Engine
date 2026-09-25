import pandas as pd

def calculate_fdr():
    print("Berechne den Schwierigkeitsgrad der nächsten 3 Spiele...")
    
    fixtures = pd.read_csv('data/upcoming_fixtures.csv')
    teams = pd.read_csv('data/raw_teams.csv')
    
    team_fdr = []
    
    for team_id in teams['id']:
        # Finde alle Spiele, bei denen das Team Heim oder Auswärts spielt
        team_games = fixtures[(fixtures['team_h'] == team_id) | (fixtures['team_a'] == team_id)].copy()
        
        # Sortiere chronologisch nach Gameweek und nimm nur die nächsten 3
        team_games = team_games.sort_values(by='gameweek').head(3)
        
        # Berechne die Gegnerstärke
        total_difficulty = 0
        for _, game in team_games.iterrows():
            if game['team_h'] == team_id:
                # Wenn das Team zu Hause spielt, ist die eigene Schwierigkeit relevant (aus Sicht des Heimteams)
                total_difficulty += game['team_h_difficulty']
            else:
                total_difficulty += game['team_a_difficulty']
                
        # Durchschnitt bilden
        avg_fdr = total_difficulty / len(team_games) if not team_games.empty else 0
        team_name = teams[teams['id'] == team_id]['name'].values[0]
        
        team_fdr.append({
            'team': team_id,
            'team_name': team_name,
            'next_3_fdr': avg_fdr
        })
        
    fdr_df = pd.DataFrame(team_fdr)
    
    # Speichern für das KI-Training
    fdr_df.to_csv('data/team_fdr.csv', index=False)
    print("Erfolg! Schwierigkeitsgrade in 'data/team_fdr.csv' gespeichert.")
    
    print("\nDie 5 Teams mit dem LEICHTESTEN Programm:")
    print(fdr_df.sort_values(by='next_3_fdr').head(5)[['team_name', 'next_3_fdr']])
    
    print("\nDie 5 Teams mit dem SCHWERSTEN Programm:")
    print(fdr_df.sort_values(by='next_3_fdr', ascending=False).head(5)[['team_name', 'next_3_fdr']])

if __name__ == "__main__":
    calculate_fdr()