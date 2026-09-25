import streamlit as st
import pandas as pd
import os
import json
import sys
from pathlib import Path
import subprocess
import requests

try:
    import pulp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pulp"])
    import pulp

PROJECT_ROOT = Path.cwd()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(page_title="FPL AI Engine", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #121214; color: #f0f2f6; }
    .player-card { background-color: #1e1e24; border: 1px solid #2d2d38; padding: 12px; border-radius: 8px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); margin-bottom: 8px; }
    .player-name { font-weight: bold; font-size: 14px; color: #ffffff; }
    .player-meta { font-size: 11px; color: #a0a0b0; margin-top: 4px; }
    .player-xp { color: #00ffcc; font-weight: bold; font-size: 13px; margin-top: 4px; }
    .pitch-divider { text-align: center; color: #a0a0b0; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; margin: 15px 0 10px 0; border-bottom: 1px solid #2d2d38; padding-bottom: 5px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ FPL AI Engine")
st.markdown("Autonomous MPC Transfer Solver & Squad Intelligence")

def fetch_manager_team(team_id: str, data_dir: Path):
    """Holt das aktuelle Team, erkennt Free Hits und speichert das Original-Layout."""
    boot_req = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/", timeout=10)
    boot_req.raise_for_status()
    events = boot_req.json().get("events", [])
    
    current_gw = 1
    for ev in events:
        if ev.get("is_current") or ev.get("is_previous"):
            current_gw = ev["id"]
            
    picks_req = requests.get(f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{current_gw}/picks/", timeout=10)
    if picks_req.status_code != 200:
        raise ValueError(f"Team ID {team_id} konnte nicht gefunden werden.")
        
    picks_data = picks_req.json()
    
    # FREE HIT DETEKTOR
    if picks_data.get("active_chip") == "freehit":
        fallback_gw = max(1, current_gw - 1)
        st.warning(f"⚠️ Free Hit in GW{current_gw} erkannt! Lade den echten permanenten Kader aus GW{fallback_gw} für die Optimierung.")
        picks_req = requests.get(f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{fallback_gw}/picks/", timeout=10)
        picks_data = picks_req.json()
    
    # 1. Basis-CSV für den Solver
    player_ids = [p["element"] for p in picks_data.get("picks", [])]
    if not player_ids:
        raise ValueError("Keine Spieler im Kader gefunden.")
    pd.DataFrame({"id": player_ids}).to_csv(data_dir / "my_team.csv", index=False)
    
    # 2. Original-Layout für das UI speichern (damit wir exakt sehen, was Sache ist)
    with open(data_dir / "current_squad.json", "w") as f:
        json.dump(picks_data.get("picks", []), f)
    
    # 3. Manager Info
    bank_raw = picks_data.get("entry_history", {}).get("bank", 0)
    with open(data_dir / "manager_info.json", "w") as f:
        json.dump({"bank": bank_raw / 10.0, "free_transfers": 1}, f)

st.sidebar.header("Control room")
no_team = st.sidebar.checkbox("Generate optimal squad from scratch (£100.0m)", value=True)

team_id = ""
if not no_team:
    team_id = st.sidebar.text_input("FPL Team ID (z.B. 830622)", value="")

run_button = st.sidebar.button("Run AI Optimization Pipeline", type="primary")

if run_button:
    with st.spinner("Fetching global data and your active FPL squad..."):
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        
        try:
            if not no_team:
                for stale_file in ["my_team.csv", "manager_info.json", "current_squad.json"]:
                    stale_path = data_dir / stale_file
                    if stale_path.exists():
                        stale_path.unlink()
                        
            if not no_team and team_id:
                fetch_manager_team(team_id, data_dir)
                
            from fetch_data import fetch_fpl_data
            from fetch_fixtures import fetch_fixtures
            from process_data import clean_fpl_data
            from season_planner import main as build_macro_plan
            from train_model import train_multi_week_model
            from multi_week_optimizer_mpc import optimize_multi_week_mpc

            fetch_fpl_data()
            clean_fpl_data()
            fetch_fixtures()
            build_macro_plan(data_dir)
            train_multi_week_model()
            
            result = optimize_multi_week_mpc(data_dir=data_dir, scratch=no_team)
            st.session_state['mpc_result'] = result
            st.success("Optimization completed successfully!")
            
        except Exception as e:
            import traceback
            st.error(f"Pipeline failed: {e}")
            st.code(traceback.format_exc())

tab1, tab2, tab3, tab4 = st.tabs(["🏟️ Squad & Lineup", "🔄 Next GW Actions", "⚡ Chip Roadmap", "📊 Analytics"])

with tab1:
    if 'mpc_result' in st.session_state:
        res = st.session_state['mpc_result']
        target_gw = res['gameweeks'][0]
        data_dir = PROJECT_ROOT / "data"
        
        # LOGIK TRENNUNG: Scratch (Post-Transfer) vs. Echtes Team (Pre-Transfer)
        if res.get('scratch') or not (data_dir / "current_squad.json").exists():
            # SCRATCH MODUS: Zeige das vom MPC neu gebaute Team
            st.subheader(f"Optimized Scratch Squad - Planning for GW{target_gw}")
            align = res['alignment']
            lineup = align['lineup']
            cap = align['captain']['name']
            vcap = align['vice_captain']['name']
            
            fwds = [p for p in lineup if p['position'] == 'FWD']
            mids = [p for p in lineup if p['position'] == 'MID']
            defs = [p for p in lineup if p['position'] == 'DEF']
            gks = [p for p in lineup if p['position'] == 'GK']
            
            bgk = align['bench_goalkeeper']
            b_outfield = align['bench_outfield']
            
        else:
            # ECHTES TEAM MODUS: Zeige exakt das, was der User aktuell hat
            st.subheader(f"Current Squad (Pre-Transfers) - ID: {team_id}")
            with open(data_dir / "current_squad.json") as f:
                current_picks = json.load(f)
                
            df_players = pd.read_csv(data_dir / "ai_players_multiweek.csv")
            xp_col = f"AI_xP_gw{target_gw}"
            
            fwds, mids, defs, gks = [], [], [], []
            bgk = None
            b_outfield = []
            cap, vcap = "", ""
            
            for p in current_picks:
                row = df_players[df_players["id"] == p["element"]]
                if row.empty: continue
                
                # Holt sich sicher den xP Wert für die kommende Woche
                xp_val = row.iloc[0].get(xp_col, row.iloc[0].get("base_xP", 0))
                
                p_data = {
                    "name": row.iloc[0]["web_name"],
                    "team": row.iloc[0]["team_name"],
                    "position": row.iloc[0]["position"],
                    "xp": float(xp_val)
                }
                
                if p.get("is_captain"): cap = p_data["name"]
                if p.get("is_vice_captain"): vcap = p_data["name"]
                
                # FPL multiplier > 0 bedeutet, der Spieler ist in der Startelf
                if p.get("multiplier", 0) > 0:
                    if p_data["position"] == "GK": gks.append(p_data)
                    elif p_data["position"] == "DEF": defs.append(p_data)
                    elif p_data["position"] == "MID": mids.append(p_data)
                    elif p_data["position"] == "FWD": fwds.append(p_data)
                else:
                    if p_data["position"] == "GK": bgk = p_data
                    else: b_outfield.append(p_data)
        
        # GEMEINSAMES RENDERING (Startelf)
        formations = [("FORWARDS", fwds), ("MIDFIELDERS", mids), ("DEFENDERS", defs), ("GOALKEEPER", gks)]
        for label, group in formations:
            if group:
                st.markdown(f'<div class="pitch-divider">{label}</div>', unsafe_allow_html=True)
                cols = st.columns(len(group))
                for i, p in enumerate(group):
                    with cols[i]:
                        badge = " 👑 (C)" if p['name'] == cap else (" Ⓥ (VC)" if p['name'] == vcap else "")
                        st.markdown(f"""
                            <div class="player-card">
                                <div class="player-name">{p['name']}{badge}</div>
                                <div class="player-meta">{p['team']}</div>
                                <div class="player-xp">{p['xp']:.2f} xP</div>
                            </div>
                        """, unsafe_allow_html=True)
                        
        # GEMEINSAMES RENDERING (Bank)
        st.markdown("### 🛋️ Bench")
        b_cols = st.columns(4)
        if bgk:
            with b_cols[0]:
                st.markdown(f"""
                    <div class="player-card" style="opacity: 0.5;">
                        <div class="player-name">{bgk['name']}</div>
                        <div class="player-meta">{bgk['position']} | {bgk['team']}</div>
                        <div class="player-xp" style="color: #a0a0b0;">{bgk['xp']:.2f} xP</div>
                    </div>
                """, unsafe_allow_html=True)
            
        for i, p in enumerate(b_outfield):
            with b_cols[i + 1]:
                st.markdown(f"""
                    <div class="player-card" style="opacity: 0.5;">
                        <div class="player-name">{p['name']}</div>
                        <div class="player-meta">{p['position']} | {p['team']}</div>
                        <div class="player-xp" style="color: #a0a0b0;">{p['xp']:.2f} xP</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Click 'Run AI Optimization Pipeline' to generate your squad overview.")

with tab2:
    if 'mpc_result' in st.session_state:
        plan = st.session_state['mpc_result']['plan'][0]
        outs = ", ".join(plan['players_out']) or "None"
        ins = ", ".join(plan['players_in']) or "None"
        
        st.subheader(f"🎯 MPC Transfer Recommendation (GW{plan['gameweek']})")
        st.markdown(f"""
            <div style="background-color: #1a1a24; border-left: 5px solid #00ffcc; padding: 20px; border-radius: 4px;">
                <h3>Action Plan</h3>
                <p><b>Transfers Out:</b> <code>{outs}</code></p>
                <p><b>Transfers In:</b> <code>{ins}</code></p>
                <p><b>Free Transfers Used:</b> {plan['free_transfers']} (Rolled: {plan['rolled_free_transfers']}) | <b>Hits:</b> {plan['hits']}</p>
                <p><b>In The Bank (ITB):</b> £{plan['itb']:.1f}m</p>
                <hr style="border-color: #2d2d38;">
                <p style="color: #a0a0b0; font-size: 13px; margin: 0;">Make these transfers on the official FPL site. The MPC model has calculated this as the mathematically optimal move.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Run the pipeline to load the action plan.")

with tab3:
    st.subheader("Season Chip Roadmap")
    macro_path = PROJECT_ROOT / "data" / "macro_plan.json"
    if macro_path.exists():
        macro_data = json.load(open(macro_path))
        c1, c2, c3, c4, c5 = st.columns(5)
        chips = [
            ("Wildcard 1", macro_data.get("wildcard_1", {})),
            ("Wildcard 2", macro_data.get("wildcard_2", {})),
            ("Free Hit", macro_data.get("free_hit", {})),
            ("Bench Boost", macro_data.get("bench_boost", {})),
            ("Triple Captain", macro_data.get("triple_captain", {}))
        ]
        for col, (name, info) in zip([c1, c2, c3, c4, c5], chips):
            with col:
                gw = info.get('gw', 'PENDING') if isinstance(info, dict) else info
                st.markdown(f"""
                    <div style="background-color: #1e1e24; border: 1px solid #2d2d38; padding: 15px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 12px; color: #a0a0b0;">{name}</div>
                        <div style="font-size: 20px; font-weight: bold; color: #00ffcc; margin: 5px 0;">GW {gw}</div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Run pipeline to load roadmap.")

with tab4:
    st.subheader("Full Analytics Engine Output")
    players_path = PROJECT_ROOT / "data" / "ai_players_multiweek.csv"
    if players_path.exists():
        df_players = pd.read_csv(players_path)
        search_query = st.text_input("🔍 Search player", "")
        filtered_df = df_players[df_players['web_name'].str.contains(search_query, case=False, na=False)] if search_query else df_players
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
    else:
        st.info("No analytics available.")