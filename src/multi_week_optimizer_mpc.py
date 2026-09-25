"""Four-gameweek model-predictive-control optimizer for FPL transfers.

The model optimizes the squad held at the end of each gameweek.  Transfers
therefore change the state for the following week's expected points instead
of selecting one static squad for the whole horizon.
"""

from pathlib import Path
import json
import re

import pandas as pd
import pulp
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
HORIZON = 4
GAMMA = 0.70
LIQUIDITY_PREMIUM = 0.005
HIT_COST = 4.0
HIT_BUNCHING_COST = 0.75
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def _detect_current_gameweek(data_dir: Path) -> int:
    """Detect the next planning GW from FPL, with local fallbacks."""
    try:
        response = requests.get(FPL_BOOTSTRAP_URL, timeout=15)
        response.raise_for_status()
        events = response.json().get("events", [])
        next_events = [event for event in events if event.get("is_next")]
        if next_events:
            return int(next_events[0]["id"])
        current_events = [event for event in events if event.get("is_current")]
        if current_events:
            return int(current_events[0]["id"])
    except (requests.RequestException, ValueError, KeyError, TypeError):
        pass

    state_path = data_dir / "season_state.json"
    if state_path.exists():
        with state_path.open(encoding="utf-8") as file:
            state = json.load(file)
        if state.get("current_gw") is not None:
            return int(state["current_gw"])

    fixture_path = data_dir / "upcoming_fixtures.csv"
    if fixture_path.exists():
        fixtures = pd.read_csv(fixture_path)
        if not fixtures.empty and "gameweek" in fixtures:
            return int(fixtures["gameweek"].min())
    raise ValueError("Unable to detect the current FPL gameweek")


def _load_inputs(
    data_dir: Path,
    horizon: int,
    scratch: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[int]]:
    """Load and validate the player, team, and manager inputs."""
    players = pd.read_csv(data_dir / "ai_players_multiweek.csv")
    my_team_path = data_dir / "my_team.csv"
    if scratch:
        my_team = pd.DataFrame({"id": []})
    else:
        my_team = pd.read_csv(my_team_path) if my_team_path.exists() else pd.DataFrame({"id": []})
    manager_info_path = data_dir / "manager_info.json"
    if manager_info_path.exists():
        with manager_info_path.open(encoding="utf-8") as file:
            manager_info = json.load(file)
    elif scratch:
        manager_info = {"bank": 0.0, "team_value": 100.0}
    else:
        raise FileNotFoundError(f"Missing manager data: {manager_info_path}")

    required_columns = {"id", "web_name", "now_cost", "position", "team_name"}
    missing_columns = required_columns - set(players.columns)
    if missing_columns:
        raise ValueError(f"Player data is missing columns: {sorted(missing_columns)}")

    xp_columns = []
    for column in players.columns:
        match = re.fullmatch(r"AI_xP_gw(\d+)", column)
        if match:
            xp_columns.append((int(match.group(1)), column))
    xp_columns.sort()
    current_gw = _detect_current_gameweek(data_dir)
    future_xp_columns = [
        (gameweek, column)
        for gameweek, column in xp_columns
        if gameweek >= current_gw
    ]
    if len(future_xp_columns) < horizon:
        found = [gameweek for gameweek, _ in xp_columns]
        raise ValueError(
            f"The MPC horizon from GW{current_gw} requires {horizon} xP columns; found {len(found)}: {found}. "
            "Run train_model.py with at least four upcoming gameweeks."
        )

    selected_gameweeks = [gameweek for gameweek, _ in future_xp_columns[:horizon]]
    selected_columns = [column for _, column in future_xp_columns[:horizon]]
    players = players.copy()
    players["id"] = players["id"].astype(int)
    players["now_cost"] = pd.to_numeric(players["now_cost"], errors="raise")
    for column in selected_columns:
        players[column] = pd.to_numeric(players[column], errors="raise").fillna(0.0)

    team_ids = set(my_team["id"].astype(int))
    available_ids = set(players["id"])
    missing_team_players = team_ids - available_ids
    if missing_team_players:
        raise ValueError(f"Current team players are absent from xP data: {sorted(missing_team_players)}")
    if not scratch and len(team_ids) != 15:
        raise ValueError(f"Expected 15 current players, found {len(team_ids)}")
    if players["id"].duplicated().any():
        raise ValueError("Player IDs must be unique in ai_players_multiweek.csv")

    players = players.set_index("id", drop=False)
    players.attrs["xp_columns"] = selected_columns
    players.attrs["current_gw"] = current_gw
    return players, my_team, manager_info, selected_gameweeks


def _load_macro_plan(data_dir: Path) -> dict:
    """Load the season-level chip plan, treating a missing plan as empty."""
    macro_plan_path = data_dir / "macro_plan.json"
    if not macro_plan_path.exists():
        return {}
    with macro_plan_path.open(encoding="utf-8") as file:
        macro_plan = json.load(file)
    if not isinstance(macro_plan, dict):
        raise ValueError("macro_plan.json must contain a JSON object")
    return macro_plan


def build_mpc_model(
    players: pd.DataFrame,
    current_player_ids: set[int],
    manager_info: dict,
    horizon: int = HORIZON,
    gamma: float = GAMMA,
    gameweeks: list[int] | None = None,
    macro_plan: dict | None = None,
    scratch: bool = False,
) -> tuple[pulp.LpProblem, dict]:
    """Build the sequential FPL MILP and return it with its variables."""
    planned_gameweeks = gameweeks or list(range(1, horizon + 1))
    macro_plan = macro_plan or {}
    wildcard_weeks = {
        int(macro_plan[key])
        for key in ("wildcard_1", "wildcard_2")
        if macro_plan.get(key) is not None
    }
    wildcard_gameweek = next(
        (gameweek for gameweek in planned_gameweeks if gameweek in wildcard_weeks),
        None,
    )
    wildcard_endgame = wildcard_gameweek is not None
    effective_horizon = (
        planned_gameweeks.index(wildcard_gameweek)
        if wildcard_endgame
        else min(horizon, len(planned_gameweeks))
    )
    active_gameweeks = planned_gameweeks[:effective_horizon]
    weeks = list(range(1, effective_horizon + 1))
    player_ids = list(players.index)
    xp_columns = players.attrs["xp_columns"][:effective_horizon]

    current_team_cost = (
        0.0 if scratch else float(players.loc[list(current_player_ids), "now_cost"].sum())
    )
    bank = 100.0 if scratch else float(manager_info.get("bank", 0.0))
    initial_free_transfers = int(manager_info.get("free_transfers", 1))
    initial_free_transfers = max(1, min(5, initial_free_transfers))

    problem = pulp.LpProblem("FPL_MultiWeek_MPC", pulp.LpMaximize)

    squad = pulp.LpVariable.dicts("squad", (player_ids, range(0, horizon + 1)), cat="Binary")
    lineup = pulp.LpVariable.dicts("lineup", (player_ids, weeks), cat="Binary")
    transfer_in = pulp.LpVariable.dicts("transfer_in", (player_ids, weeks), cat="Binary")
    transfer_out = pulp.LpVariable.dicts("transfer_out", (player_ids, weeks), cat="Binary")
    ft_available = pulp.LpVariable.dicts("ft_available", weeks, lowBound=1, upBound=5, cat="Integer")
    rolled_ft = pulp.LpVariable.dicts("rolled_ft", weeks, lowBound=0, upBound=5, cat="Integer")
    ft_cap_reached = pulp.LpVariable.dicts(
        "ft_cap_reached", weeks[:-1], cat="Binary"
    )
    hits = pulp.LpVariable.dicts("hits", weeks, lowBound=0, cat="Integer")
    # Threshold binaries linearize a convex cost for taking several hits in
    # one week. The range covers every transfer count possible from the
    # binary squad inventory; hits themselves remain unbounded above.
    hit_thresholds = pulp.LpVariable.dicts(
        "hit_threshold", (weeks, range(2, len(player_ids) + 1)), cat="Binary"
    )
    itb = pulp.LpVariable.dicts("itb", range(0, horizon + 1), lowBound=0, cat="Continuous")
    transfers_made_by_week = {}

    # The initial state is the actual squad and the initial bank balance.
    for player_id in player_ids:
        problem += squad[player_id][0] == int(player_id in current_player_ids), f"Initial_Squad_{player_id}"
    problem += itb[0] == bank, "Initial_Bank"

    for week in weeks:
        previous_week = week - 1
        for player_id in player_ids:
            problem += (
                squad[player_id][week]
                == squad[player_id][previous_week]
                + transfer_in[player_id][week]
                - transfer_out[player_id][week]
            ), f"State_Transition_{player_id}_{week}"
            problem += (
                transfer_in[player_id][week] + transfer_out[player_id][week] <= 1
            ), f"One_Action_Per_Player_{player_id}_{week}"
            problem += (
                transfer_in[player_id][week] <= 1 - squad[player_id][previous_week]
            ), f"Only_Buy_Outside_Squad_{player_id}_{week}"
            problem += (
                transfer_out[player_id][week] <= squad[player_id][previous_week]
            ), f"Only_Sell_Current_Player_{player_id}_{week}"
            problem += lineup[player_id][week] <= squad[player_id][week], f"Lineup_In_Squad_{player_id}_{week}"

        problem += pulp.lpSum(squad[player_id][week] for player_id in player_ids) == 15, f"Total_Players_{week}"
        for position, required_count in {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}.items():
            problem += pulp.lpSum(
                squad[player_id][week]
                for player_id in player_ids
                if players.loc[player_id, "position"] == position
            ) == required_count, f"{position}_Count_{week}"
        problem += pulp.lpSum(lineup[player_id][week] for player_id in player_ids) == 11, f"Lineup_Players_{week}"
        problem += pulp.lpSum(
            lineup[player_id][week]
            for player_id in player_ids
            if players.loc[player_id, "position"] == "GK"
        ) == 1, f"Lineup_GK_{week}"
        problem += pulp.lpSum(
            lineup[player_id][week]
            for player_id in player_ids
            if players.loc[player_id, "position"] == "DEF"
        ) >= 3, f"Lineup_DEF_Minimum_{week}"
        problem += pulp.lpSum(
            lineup[player_id][week]
            for player_id in player_ids
            if players.loc[player_id, "position"] == "MID"
        ) >= 2, f"Lineup_MID_Minimum_{week}"
        problem += pulp.lpSum(
            lineup[player_id][week]
            for player_id in player_ids
            if players.loc[player_id, "position"] == "FWD"
        ) >= 1, f"Lineup_FWD_Minimum_{week}"
        for team in players["team_name"].dropna().unique():
            problem += pulp.lpSum(
                squad[player_id][week]
                for player_id in player_ids
                if players.loc[player_id, "team_name"] == team
            ) <= 3, f"Team_Limit_{team}_{week}"

        # Use the manager's actual squad value as the budget base.  ITB is
        # carried forward, so the constraint is equivalent to the usual
        # "budget base + previous bank" rule without a nonlinear cash flow.
        squad_cost = pulp.lpSum(players.loc[player_id, "now_cost"] * squad[player_id][week] for player_id in player_ids)
        if scratch:
            problem += squad_cost <= 100.0, f"Budget_{week}"
            problem += itb[week] == 100.0 - squad_cost, f"Bank_Balance_{week}"
        else:
            problem += squad_cost <= current_team_cost + itb[previous_week], f"Budget_{week}"
            problem += itb[week] == current_team_cost + itb[previous_week] - squad_cost, f"Bank_Balance_{week}"

        transfers_made = pulp.lpSum(transfer_in[player_id][week] for player_id in player_ids)
        transfers_made_by_week[week] = transfers_made
        problem += (
            transfers_made == ft_available[week] - rolled_ft[week] + hits[week]
        ), f"Transfer_Budget_{week}"
        for threshold in range(2, len(player_ids) + 1):
            threshold_binary = hit_thresholds[week][threshold]
            problem += (
                hits[week] >= threshold * threshold_binary
            ), f"Hit_Threshold_Lower_{week}_{threshold}"
            problem += (
                hits[week] <= threshold - 1 + len(player_ids) * threshold_binary
            ), f"Hit_Threshold_Upper_{week}_{threshold}"
        problem += rolled_ft[week] <= ft_available[week], f"Rolled_FT_Limit_{week}"
        if week == 1:
            problem += ft_available[week] == initial_free_transfers, "Initial_Free_Transfers"
        else:
            problem += (
                ft_available[week]
                == rolled_ft[previous_week] + 1 - ft_cap_reached[previous_week]
            ), f"Next_Week_Free_Transfers_{week}"
            problem += (
                rolled_ft[previous_week] <= 4 + ft_cap_reached[previous_week]
            ), f"FT_Cap_Upper_{previous_week}"
            problem += (
                rolled_ft[previous_week] >= 5 * ft_cap_reached[previous_week]
            ), f"FT_Cap_Indicator_{previous_week}"

    objective_points = pulp.lpSum(
        (gamma ** (week - 1))
        * players.loc[player_id, xp_columns[week - 1]]
        * lineup[player_id][week]
        for week in weeks
        for player_id in player_ids
    )
    # Carry the final lineup's value one additional week so a GW1 hit is not
    # evaluated only against the truncated four-week reward window. The last
    # available xP column is the conservative continuation estimate.
    terminal_continuation = (
        pulp.lpSum(
            (gamma ** effective_horizon)
            * players.loc[player_id, xp_columns[-1]]
            * lineup[player_id][weeks[-1]]
            for player_id in player_ids
        )
        if weeks and not wildcard_endgame
        else 0
    )
    liquidity_value = pulp.lpSum(LIQUIDITY_PREMIUM * itb[week] for week in weeks)
    hit_penalty = HIT_COST * pulp.lpSum(hits[week] for week in weeks)
    hit_bunching_penalty = pulp.lpSum(
        HIT_BUNCHING_COST
        * (threshold - 1)
        * hit_thresholds[week][threshold]
        for week in weeks
        for threshold in range(2, len(player_ids) + 1)
    )
    ft_opportunity_value = 1.5 * ft_available[weeks[-1]] if weeks and not wildcard_endgame else 0
    transfer_friction = 0.5 * pulp.lpSum(transfers_made_by_week[week] for week in weeks)
    problem += (
        objective_points
        + terminal_continuation
        + liquidity_value
        + ft_opportunity_value
        - transfer_friction
        - hit_penalty
        - hit_bunching_penalty
    ), "Risk_Adjusted_MPC_Objective"

    variables = {
        "squad": squad,
        "lineup": lineup,
        "transfer_in": transfer_in,
        "transfer_out": transfer_out,
        "ft_available": ft_available,
        "rolled_ft": rolled_ft,
        "hits": hits,
        "hit_thresholds": hit_thresholds,
        "itb": itb,
        "weeks": weeks,
        "gameweeks": active_gameweeks,
        "wildcard_endgame": wildcard_endgame,
        "wildcard_gameweek": wildcard_gameweek,
        "scratch": scratch,
    }
    return problem, variables


def optimize_multi_week_mpc(
    data_dir: Path = DATA_DIR,
    horizon: int = HORIZON,
    solver: pulp.LpSolver | None = None,
    scratch: bool = False,
) -> dict:
    """Solve and print the optimal week-by-week transfer sequence."""
    players, my_team, manager_info, gameweeks = _load_inputs(data_dir, horizon, scratch=scratch)
    macro_plan = _load_macro_plan(data_dir)
    current_player_ids = set(my_team["id"].astype(int))
    problem, variables = build_mpc_model(
        players,
        current_player_ids,
        manager_info,
        horizon,
        gameweeks=gameweeks,
        macro_plan=macro_plan,
        scratch=scratch,
    )

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    problem.solve(solver)
    status = pulp.LpStatus[problem.status]
    if status != "Optimal":
        raise RuntimeError(f"MPC solver did not find an optimal solution: {status}")

    plan = []
    previous_ids = current_player_ids
    for week, gameweek in zip(variables["weeks"], variables["gameweeks"]):
        squad_ids = {
            player_id
            for player_id in players.index
            if pulp.value(variables["squad"][player_id][week]) > 0.5
        }
        transfers_out = sorted(previous_ids - squad_ids)
        transfers_in = sorted(squad_ids - previous_ids)
        plan.append(
            {
                "week": week,
                "gameweek": gameweek,
                "players_out": [players.loc[player_id, "web_name"] for player_id in transfers_out],
                "players_in": [players.loc[player_id, "web_name"] for player_id in transfers_in],
                "free_transfers": int(round(pulp.value(variables["ft_available"][week]))),
                "rolled_free_transfers": int(round(pulp.value(variables["rolled_ft"][week]))),
                "hits": int(round(pulp.value(variables["hits"][week]))),
                "itb": float(pulp.value(variables["itb"][week])),
            }
        )
        previous_ids = squad_ids

    print("\n--- FPL MPC TRANSFER PLAN ---")
    if variables["wildcard_endgame"]:
        print(
            f"Wildcard endgame: optimizing through GW{variables['gameweeks'][-1]} "
            f"before the GW{variables['wildcard_gameweek']} wildcard."
        )
    for recommendation in plan:
        print(f"GW{recommendation['gameweek']}: ")
        print(f"  Out: {', '.join(recommendation['players_out']) or 'None'}")
        print(f"  In:  {', '.join(recommendation['players_in']) or 'None'}")
        print(
            f"  FTs: {recommendation['free_transfers']} "
            f"(rolled {recommendation['rolled_free_transfers']}), "
            f"hits: {recommendation['hits']}, ITB: £{recommendation['itb']:.1f}m"
        )
    print(f"Objective value: {pulp.value(problem.objective):.2f}")

    current_gw = variables["gameweeks"][0]
    current_week = variables["weeks"][0]
    current_xp_column = f"AI_xP_gw{current_gw}"
    current_squad_ids = [
        player_id
        for player_id in players.index
        if pulp.value(variables["squad"][player_id][current_week]) > 0.5
    ]
    lineup_ids = {
        player_id
        for player_id in current_squad_ids
        if pulp.value(variables["lineup"][player_id][current_week]) > 0.5
    }

    def alignment_record(player_id: int) -> dict:
        player = players.loc[player_id]
        return {
            "name": player["web_name"],
            "position": player["position"],
            "team": player["team_name"],
            "xp": float(player[current_xp_column]),
        }

    lineup_players = [alignment_record(player_id) for player_id in lineup_ids]
    bench_players = [
        alignment_record(player_id)
        for player_id in current_squad_ids
        if player_id not in lineup_ids
    ]
    captaincy_order = sorted(lineup_players, key=lambda player: player["xp"], reverse=True)
    bench_goalkeeper = next(
        player for player in bench_players if player["position"] == "GK"
    )
    bench_outfield = sorted(
        [player for player in bench_players if player["position"] != "GK"],
        key=lambda player: player["xp"],
        reverse=True,
    )

    print(f"\n--- GW{current_gw} SQUAD ALIGNMENT ---")
    print("STARTING 11:")
    for position in ("GK", "DEF", "MID", "FWD"):
        position_players = [
            player for player in lineup_players if player["position"] == position
        ]
        formatted_players = ", ".join(
            f"{player['name']} ({player['xp']:.2f})" for player in position_players
        )
        print(f"{position}: {formatted_players}")

    print(f"\nCAPTAIN: {captaincy_order[0]['name']} ({captaincy_order[0]['xp']:.2f})")
    print(f"VICE-CAPTAIN: {captaincy_order[1]['name']} ({captaincy_order[1]['xp']:.2f})")
    print("\nBENCH:")
    print(f"GK: {bench_goalkeeper['name']} ({bench_goalkeeper['xp']:.2f})")
    for bench_number, player in enumerate(bench_outfield, start=1):
        print(
            f"{bench_number}: {player['name']} ({player['xp']:.2f}) "
            f"- [{player['position']}]"
        )

    return {
        "status": status,
        "gameweeks": variables["gameweeks"],
        "plan": plan,
        "objective": pulp.value(problem.objective),
        "alignment": {
            "gameweek": current_gw,
            "lineup": lineup_players,
            "captain": captaincy_order[0],
            "vice_captain": captaincy_order[1],
            "bench_goalkeeper": bench_goalkeeper,
            "bench_outfield": bench_outfield,
        },
        "scratch": scratch,
    }


if __name__ == "__main__":
    optimize_multi_week_mpc()