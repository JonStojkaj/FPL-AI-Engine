import pandas as pd


def clean_fpl_data():
    print("Loading official FPL data...")
    players = pd.read_csv("data/raw_players.csv")
    teams = pd.read_csv("data/raw_teams.csv")

    official_team_ids = set(teams["id"].astype(int))
    players["team"] = pd.to_numeric(players["team"], errors="coerce")
    active_players = (
        players["status"].astype(str).str.strip().str.lower().ne("u")
        & ~players["removed"].astype(str).str.strip().str.lower().eq("true")
    )
    players = players[
        active_players & players["team"].isin(official_team_ids)
    ].copy()

    columns = [
        "id", "web_name", "element_type", "team",
        "now_cost", "total_points", "form", "minutes",
        "expected_goals", "expected_assists", "expected_goal_involvements", "ict_index",
        "chance_of_playing_next_round",
    ]

    missing_columns = set(columns) - set(players.columns)
    if missing_columns:
        raise ValueError(f"Official player data is missing columns: {sorted(missing_columns)}")
    available_cols = columns
    df = players[available_cols].copy()

    df["now_cost"] = df["now_cost"] / 10.0
    for col in [
        "form",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "ict_index",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["chance_of_playing_next_round"] = pd.to_numeric(
        df["chance_of_playing_next_round"], errors="coerce"
    ).fillna(100.0)

    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    df["position"] = df["element_type"].map(pos_map)
    df = df[df["position"].notna()].copy()

    team_map = dict(zip(teams["id"], teams["name"]))
    df["team_name"] = df["team"].map(team_map)
    df = df[df["team_name"].notna()].copy()

    df = df.drop(columns=["element_type", "team"])
    df = df.reset_index(drop=True)

    df.to_csv("data/clean_players.csv", index=False)
    print(f"Saved {len(df)} active Premier League players to data/clean_players.csv.\n")

if __name__ == "__main__":
    clean_fpl_data()