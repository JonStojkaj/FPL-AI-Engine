"""Single-command executive runner for the FPL AI Engine."""

from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _format_player(player: dict) -> str:
    return f"{player['name']} ({player['xp']:.2f} xP)"


def run_engine() -> int:
    """Refresh inputs, solve the MPC plan, and print one executive report."""
    try:
        from fetch_data import fetch_fpl_data
        from fetch_fixtures import fetch_fixtures
        from process_data import clean_fpl_data
        from season_planner import main as build_macro_plan
        from train_model import train_multi_week_model
        from multi_week_optimizer_mpc import optimize_multi_week_mpc

        _print_header("FPL AI ENGINE | EXECUTIVE RUN")
        print("Refreshing official FPL players and fixtures...")
        fetch_fpl_data()
        clean_fpl_data()
        fetch_fixtures()

        print("\nBuilding season macro plan...")
        macro_plan = build_macro_plan(PROJECT_ROOT / "data")

        print("\nGenerating season-long xP projections...")
        train_multi_week_model()

        print("\nSolving the MPC transfer plan...")
        result = optimize_multi_week_mpc(data_dir=PROJECT_ROOT / "data")
        alignment = result["alignment"]

        _print_header("THE LONG-TERM CHIP ROADMAP")
        chip_labels = {
            "wildcard_1": "Wildcard 1",
            "wildcard_2": "Wildcard 2",
            "free_hit": "Free Hit",
            "bench_boost": "Bench Boost",
            "triple_captain": "Triple Captain",
        }
        for key, label in chip_labels.items():
            gameweek = macro_plan.get(key)
            print(f"{label:<18} GW{gameweek}" if gameweek else f"{label:<18} PENDING")

        _print_header("IMMEDIATE NEXT GAMEWEEK ACTIONS")
        first_plan = result["plan"][0]
        print(f"Gameweek: GW{first_plan['gameweek']}")
        print(f"Out: {', '.join(first_plan['players_out']) or 'None'}")
        print(f"In:  {', '.join(first_plan['players_in']) or 'None'}")
        print(f"Hits: {first_plan['hits']} | ITB: £{first_plan['itb']:.1f}m")
        print("\nStarting 11:")
        for position in ("GK", "DEF", "MID", "FWD"):
            players = [player for player in alignment["lineup"] if player["position"] == position]
            print(f"{position}: {', '.join(_format_player(player) for player in players)}")
        print(f"Captain: { _format_player(alignment['captain']) }")
        print(f"Vice-Captain: { _format_player(alignment['vice_captain']) }")
        print(f"Bench GK: {_format_player(alignment['bench_goalkeeper'])}")
        for number, player in enumerate(alignment["bench_outfield"], start=1):
            print(f"Bench {number}: {_format_player(player)} [{player['position']}]")

        _print_header("LONG-TERM HORIZON PLAN")
        for recommendation in result["plan"][1:]:
            transfers = ", ".join(
                f"{player} in" for player in recommendation["players_in"]
            ) or "No planned buys"
            sales = ", ".join(
                f"{player} out" for player in recommendation["players_out"]
            ) or "No planned sales"
            print(
                f"GW{recommendation['gameweek']}: {sales}; {transfers}; "
                f"hits {recommendation['hits']}"
            )
        print(f"\nMPC objective: {result['objective']:.2f}")
        print("\nPipeline completed successfully.")
        return 0
    except Exception as error:
        print(f"\nFPL AI Engine failed: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(run_engine())