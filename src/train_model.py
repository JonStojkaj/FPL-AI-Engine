import numpy as np
import pandas as pd


def train_multi_week_model():
    """Generate season-long deterministic Bayesian FPL xP projections."""
    print("Generating season-long Bayesian xP projections...")
    df = pd.read_csv("data/clean_players.csv")
    fixtures = pd.read_csv("data/upcoming_fixtures.csv")
    teams = pd.read_csv("data/raw_teams.csv")

    required_columns = {
        "now_cost",
        "minutes",
        "expected_goals",
        "expected_assists",
        "chance_of_playing_next_round",
        "team_name",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Player data is missing columns: {sorted(missing_columns)}")

    numeric_columns = [
        "now_cost",
        "minutes",
        "expected_goals",
        "expected_assists",
        "chance_of_playing_next_round",
    ]
    for column in numeric_columns:
        fill_value = 100.0 if column == "chance_of_playing_next_round" else 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fill_value)

    # The official API stores now_cost in tenths; clean_players.csv stores
    # the same value in millions. Support both representations autonomously.
    cost = df["now_cost"] / 10.0 if df["now_cost"].max() > 20.0 else df["now_cost"]
    total_mins = df["minutes"].astype(float)
    xG = df["expected_goals"].astype(float)
    xA = df["expected_assists"].astype(float)

    expected_mins_ratio = (total_mins / 360.0).clip(lower=0.0, upper=1.0)
    matches_played = total_mins / 90.0
    normalized_games = matches_played.clip(lower=1.0)
    xG_90 = xG / normalized_games
    xA_90 = xA / normalized_games

    credibility_threshold = 900.0
    stats_weight = total_mins / (total_mins + credibility_threshold)
    price_xp = (cost - 4.0) * 0.40 + 2.5
    stats_xp = 2.0 + (xG_90 * 4.5) + (xA_90 * 3.0)
    stats_xp = stats_xp.clip(upper=10.0)
    minutes_confidence = matches_played / (matches_played + 2.0)
    stats_xp = price_xp + (stats_xp - price_xp) * minutes_confidence
    conservative_base_xp = ((1.0 - stats_weight) * price_xp) + (stats_weight * stats_xp)
    overperformer = (total_mins > 0.0) & (stats_xp > price_xp)
    base_xp = np.where(
        overperformer,
        (0.3 * price_xp) + (0.7 * stats_xp),
        conservative_base_xp,
    )
    availability = (df["chance_of_playing_next_round"] / 100.0).fillna(1.0).clip(0.0, 1.0)
    df["base_xP"] = np.maximum(0.0, base_xp * expected_mins_ratio * availability)

    next_gws = list(range(1, 39))
    print(f"Planning gameweeks: GW{next_gws[0]}-GW{next_gws[-1]}")
    team_ids = dict(zip(teams["name"], teams["id"]))

    for gw in next_gws:
        gw_fixtures = fixtures[fixtures["gameweek"] == gw]
        team_fdr_map = {}
        for _, game in gw_fixtures.iterrows():
            team_fdr_map[game["team_h"]] = game["team_h_difficulty"]
            team_fdr_map[game["team_a"]] = game["team_a_difficulty"]

        col_name = f"AI_xP_gw{gw}"

        def get_gw_xp(row):
            player_team_id = team_ids.get(row["team_name"])
            fdr = team_fdr_map.get(player_team_id, 3.0)
            fdr_mult = np.clip(3.0 / fdr, 0.75, 1.25)
            return row["base_xP"] * fdr_mult

        df[col_name] = df.apply(get_gw_xp, axis=1)

    df.to_csv("data/ai_players_multiweek.csv", index=False)
    print("Saved season-long xP data to data/ai_players_multiweek.csv")


if __name__ == "__main__":
    train_multi_week_model()