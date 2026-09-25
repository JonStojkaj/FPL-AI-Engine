"""Season-wide FPL chip planner based on fixture congestion and FDR heuristics."""

from pathlib import Path
import json

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
GAMEWEEKS = range(1, 39)
TEAM_COUNT = 20


FALLBACKS = {
    "wildcard_1": 7,
    "free_hit": 29,
    "bench_boost": 37,
}


def _fixture_frame_from_api() -> pd.DataFrame:
    """Load future fixtures from the official FPL fixtures endpoint."""
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()
    fixtures = pd.DataFrame(payload)
    if fixtures.empty:
        raise ValueError("The official fixtures API returned no fixtures")

    required = {"event", "team_h", "team_a", "finished"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"Official fixtures data is missing columns: {sorted(missing)}")

    fixtures = fixtures[
        ~fixtures["finished"].astype(bool)
        & fixtures["event"].notna()
    ].copy()
    return fixtures.rename(columns={"event": "gameweek"})


def _fixture_frame_from_csv(data_dir: Path) -> pd.DataFrame:
    """Load the repository's upcoming fixture export."""
    fixtures = pd.read_csv(data_dir / "upcoming_fixtures.csv")
    required = {"gameweek", "team_h", "team_a"}
    missing = required - set(fixtures.columns)
    if missing:
        raise ValueError(f"Fixture CSV is missing columns: {sorted(missing)}")
    return fixtures.copy()


def load_fixtures(data_dir: Path = DATA_DIR, use_api: bool = True) -> tuple[pd.DataFrame, str]:
    """Prefer official API data and fall back to the local fixture export."""
    if use_api:
        try:
            return _fixture_frame_from_api(), "official FPL API"
        except (requests.RequestException, ValueError, KeyError) as error:
            print(f"Fixture API unavailable ({error}); using upcoming_fixtures.csv.")

    return _fixture_frame_from_csv(data_dir), "upcoming_fixtures.csv"


def _normalise_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    fixtures = fixtures.copy()
    fixtures["gameweek"] = pd.to_numeric(fixtures["gameweek"], errors="coerce")
    fixtures["team_h"] = pd.to_numeric(fixtures["team_h"], errors="coerce")
    fixtures["team_a"] = pd.to_numeric(fixtures["team_a"], errors="coerce")
    fixtures = fixtures[
        fixtures["gameweek"].between(1, 38)
        & fixtures["team_h"].notna()
        & fixtures["team_a"].notna()
    ].copy()
    fixtures["gameweek"] = fixtures["gameweek"].astype(int)
    fixtures["team_h"] = fixtures["team_h"].astype(int)
    fixtures["team_a"] = fixtures["team_a"].astype(int)
    return fixtures


def aggregate_schedule(fixtures: pd.DataFrame, team_count: int = TEAM_COUNT) -> dict:
    """Return per-GW totals, per-team fixture counts, and FDR details."""
    fixtures = _normalise_fixtures(fixtures)
    team_ids = sorted(set(fixtures["team_h"]) | set(fixtures["team_a"]))
    if len(team_ids) < team_count:
        team_ids = list(range(1, team_count + 1))

    team_counts = {
        gameweek: {
            team_id: int(
                ((fixtures["gameweek"] == gameweek) & (fixtures["team_h"] == team_id)).sum()
                + ((fixtures["gameweek"] == gameweek) & (fixtures["team_a"] == team_id)).sum()
            )
            for team_id in team_ids
        }
        for gameweek in GAMEWEEKS
    }
    total_fixtures = {
        gameweek: int(sum(team_counts[gameweek].values()) // 2)
        for gameweek in GAMEWEEKS
    }

    fdr_by_team_week = {}
    for gameweek in GAMEWEEKS:
        for team_id in team_ids:
            home = fixtures[(fixtures["gameweek"] == gameweek) & (fixtures["team_h"] == team_id)]
            away = fixtures[(fixtures["gameweek"] == gameweek) & (fixtures["team_a"] == team_id)]
            entries = []
            for _, fixture in home.iterrows():
                entries.append((True, float(fixture.get("team_h_difficulty", 3.0))))
            for _, fixture in away.iterrows():
                entries.append((False, float(fixture.get("team_a_difficulty", 3.0))))
            fdr_by_team_week[(team_id, gameweek)] = entries

    return {
        "team_ids": team_ids,
        "team_counts": team_counts,
        "total_fixtures": total_fixtures,
        "fdr_by_team_week": fdr_by_team_week,
        "observed_gameweeks": set(fixtures["gameweek"].unique()),
    }


def _confirmed_double_gameweeks(schedule: dict) -> list[int]:
    return [
        gameweek
        for gameweek in GAMEWEEKS
        if gameweek in schedule["observed_gameweeks"]
        if any(count >= 2 for count in schedule["team_counts"][gameweek].values())
    ]


def _confirmed_blank_gameweeks(schedule: dict) -> list[int]:
    return [
        gameweek
        for gameweek in GAMEWEEKS
        if gameweek in schedule["observed_gameweeks"]
        if any(count == 0 for count in schedule["team_counts"][gameweek].values())
    ]


def choose_chip_weeks(schedule: dict) -> tuple[dict, dict[str, str]]:
    """Choose chip weeks and return a human-readable reason for each choice."""
    totals = schedule["total_fixtures"]
    doubles = _confirmed_double_gameweeks(schedule)
    blanks = _confirmed_blank_gameweeks(schedule)
    reasons = {}

    if doubles:
        bench_boost = max(doubles, key=lambda gameweek: (totals[gameweek], -gameweek))
        reasons["bench_boost"] = (
            f"GW{bench_boost} has the highest confirmed fixture total among DGWs "
            f"({totals[bench_boost]} fixtures)."
        )
    else:
        bench_boost = FALLBACKS["bench_boost"]
        reasons["bench_boost"] = "No DGW is confirmed yet; using the historical late-season GW37 fallback."

    if blanks:
        free_hit = min(blanks, key=lambda gameweek: (totals[gameweek], gameweek))
        reasons["free_hit"] = (
            f"GW{free_hit} has the lowest confirmed fixture total and is the largest "
            f"scheduled blank ({totals[free_hit]} fixtures)."
        )
    elif doubles:
        free_hit = min(doubles, key=lambda gameweek: (totals[gameweek], gameweek))
        reasons["free_hit"] = (
            f"No blank is confirmed; GW{free_hit} is the smallest available DGW/fixture "
            f"anomaly ({totals[free_hit]} fixtures)."
        )
    else:
        free_hit = FALLBACKS["free_hit"]
        reasons["free_hit"] = "No BGW is confirmed yet; using the historical FA Cup-period GW29 fallback."

    tc_candidates = []
    for gameweek in doubles:
        for team_id in schedule["team_ids"]:
            entries = schedule["fdr_by_team_week"].get((team_id, gameweek), [])
            if len(entries) == 2:
                average_fdr = sum(fdr for _, fdr in entries) / 2
                home_games = sum(is_home for is_home, _ in entries)
                if average_fdr < 2.5:
                    tc_candidates.append((average_fdr, -home_games, gameweek, team_id))
    if tc_candidates:
        average_fdr, negative_home_games, triple_captain, team_id = min(tc_candidates)
        reasons["triple_captain"] = (
            f"GW{triple_captain} has a DGW team ({team_id}) with FDR average "
            f"{average_fdr:.1f} and {-negative_home_games} home fixture(s)."
        )
    else:
        triple_captain = None
        reasons["triple_captain"] = "No confirmed DGW has a top-team FDR profile yet; chip remains pending."

    wildcard_1 = next(
        (gameweek for gameweek in range(6, 13) if gameweek in doubles or gameweek in blanks),
        FALLBACKS["wildcard_1"],
    )
    reasons["wildcard_1"] = (
        f"GW{wildcard_1} is in the first-half fixture-change window; "
        "using the GW6-8 meta fallback when no major swing is confirmed."
    )

    wildcard_2 = max(1, bench_boost - 2)
    reasons["wildcard_2"] = (
        f"GW{wildcard_2} is two weeks before the Bench Boost target GW{bench_boost}, "
        "allowing the squad to be prepared for the double-gameweek."
    )

    return {
        "wildcard_1": wildcard_1,
        "wildcard_2": wildcard_2,
        "free_hit": free_hit,
        "bench_boost": bench_boost,
        "triple_captain": triple_captain,
    }, reasons


def main(data_dir: Path = DATA_DIR) -> dict:
    fixtures, source = load_fixtures(data_dir)
    schedule = aggregate_schedule(fixtures)
    plan, reasons = choose_chip_weeks(schedule)

    print(f"\n--- FPL SEASON CHIP PLAN ({source}) ---")
    for chip, gameweek in plan.items():
        label = f"GW{gameweek}" if gameweek is not None else "PENDING"
        print(f"{chip}: {label}")
        print(f"  Why: {reasons[chip]}")

    print("\nSchedule scan:")
    print(f"  Confirmed BGWs: {_confirmed_blank_gameweeks(schedule) or 'none'}")
    print(f"  Confirmed DGWs: {_confirmed_double_gameweeks(schedule) or 'none'}")

    output_path = data_dir / "macro_plan.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(plan, file, indent=2)
        file.write("\n")
    print(f"\nSaved macro plan to {output_path}")
    return plan


if __name__ == "__main__":
    main()
