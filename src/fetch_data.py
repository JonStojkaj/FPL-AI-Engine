import requests
import pandas as pd


FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def get_current_gameweek(events):
    """Return the next GW, or the current GW at the end of a season."""
    next_events = [event for event in events if event.get("is_next")]
    if next_events:
        return int(next_events[0]["id"])

    current_events = [event for event in events if event.get("is_current")]
    if current_events:
        return int(current_events[0]["id"])

    raise ValueError("Official FPL API returned no next or current gameweek")


def fetch_fpl_data():
    """Fetch the current player and team data from the official FPL API."""
    print("Fetching live data from the official FPL API...")
    response = requests.get(FPL_BOOTSTRAP_URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    current_gw = get_current_gameweek(data.get("events", []))

    if not isinstance(data.get("elements"), list) or not isinstance(data.get("teams"), list):
        raise ValueError("Official FPL API response has no valid elements or teams payload")

    players_df = pd.DataFrame(data["elements"])
    teams_df = pd.DataFrame(data["teams"])
    official_team_ids = set(teams_df["id"].astype(int))
    players_df["team"] = pd.to_numeric(players_df["team"], errors="coerce")
    active_players = (
        players_df["status"].astype(str).str.lower().ne("u")
        & ~players_df["removed"].astype(bool)
    )
    players_df = players_df[
        active_players & players_df["team"].isin(official_team_ids)
    ].copy()

    players_df.to_csv("data/raw_players.csv", index=False)
    teams_df.to_csv("data/raw_teams.csv", index=False)
    with open("data/season_state.json", "w", encoding="utf-8") as file:
        import json

        json.dump({"current_gw": current_gw}, file, indent=2)

    print(
        f"Saved {len(players_df)} active Premier League players and {len(teams_df)} teams. "
        f"Next/current GW: {current_gw}."
    )

if __name__ == "__main__":
    fetch_fpl_data()