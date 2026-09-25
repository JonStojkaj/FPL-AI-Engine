import requests
import pandas as pd

def fetch_fixtures():
    print("Lade den offiziellen Premier League Spielplan...")
    
    # Der API-Endpunkt für alle Spiele der Saison
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Fehler beim Abrufen der Fixtures.")
        return
        
    fixtures = pd.DataFrame(response.json())
    
    # Wir filtern nur Spiele heraus, die noch nicht gespielt wurden (Future Fixtures)
    upcoming = fixtures[fixtures['finished'] == False].copy()
    
    # Wir brauchen nur die wichtigsten Infos: Wer (Home/Away), Wann (GW), und wie schwer (FDR)
    cols_to_keep = ['event', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty']
    upcoming = upcoming[cols_to_keep]
    upcoming = upcoming.rename(columns={'event': 'gameweek'})
    
    # Speichern für das nächste KI-Training
    upcoming.to_csv('data/upcoming_fixtures.csv', index=False)
    print(f"Erfolg! {len(upcoming)} kommende Spiele in 'data/upcoming_fixtures.csv' gespeichert.")
    
    print("\nDie nächsten 5 Spiele (mit Gegnerstärke 1-5):")
    print(upcoming.head(5))

if __name__ == "__main__":
    fetch_fixtures()