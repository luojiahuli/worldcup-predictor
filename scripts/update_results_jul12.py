"""
update_results_jul12.py - 更新7月2日-12日全部淘汰赛赛果
包括：剩余6场R32 + 全部8场R16 + 全部4场QF
半决赛对阵已填入但保留TIMED状态
"""
import pickle, json, os, logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
with open(path, "rb") as f:
    wc = pickle.load(f)

# === 1. R32 剩余6场（idx 82-87）===
r32_results = {
    # idx: (home_goals, away_goals)
    82: (3, 0),   # Spain 3-0 Austria
    83: (1, 2),   # Croatia 1-2 Portugal (Portugal wins)
    84: (2, 0),   # Switzerland 2-0 Algeria
    85: (1, 1),   # Australia 1-1 Egypt (Egypt 4-2 pens)
    86: (3, 2),   # Argentina 3-2 Cape Verde Islands
    87: (1, 0),   # Colombia 1-0 Ghana
}

for idx, (hg, ag) in r32_results.items():
    wc.at[idx, "HomeGoals"] = float(hg)
    wc.at[idx, "AwayGoals"] = float(ag)
    wc.at[idx, "status"] = "FINISHED"
    log.info(f"  R32 idx={idx}: {wc.at[idx, 'Home']} {hg}-{ag} {wc.at[idx, 'Away']}")

# === 2. R16（idx 88-95）填入对阵 + 赛果 ===
# Bracket mapping based on R32 winners:
#   88: Morocco (R32_FC) vs Canada (R32_AB) → 3-0
#   89: France (R32_I3F) vs Paraguay (R32_E3D) → 1-0
#   90: Norway (R32_EI) vs Brazil (R32_CF) → 2-1
#   91: England (R32_L3X) vs Mexico (R32_A3X) → 3-2
#   92: Spain (R32_HJ) vs Portugal (R32_LK) → 1-0
#   93: Belgium (R32_G3X) vs USA (R32_DB) → 4-1
#   94: Argentina (R32_JH) vs Egypt (R32_DG) → 3-2
#   95: Switzerland (R32_B3X) vs Colombia (R32_K3X) → 0-0 (4-3 pens)
r16_fixtures = {
    88: ("Morocco", "Canada", 3, 0),
    89: ("France", "Paraguay", 1, 0),
    90: ("Norway", "Brazil", 2, 1),
    91: ("England", "Mexico", 3, 2),
    92: ("Spain", "Portugal", 1, 0),
    93: ("Belgium", "United States", 4, 1),
    94: ("Argentina", "Egypt", 3, 2),
    95: ("Switzerland", "Colombia", 0, 0),  # 0-0 AET, Switzerland 4-3 pens
}

r16_winners = {}
for idx, (home, away, hg, ag) in r16_fixtures.items():
    wc.at[idx, "Home"] = home
    wc.at[idx, "Away"] = away
    wc.at[idx, "HomeGoals"] = float(hg)
    wc.at[idx, "AwayGoals"] = float(ag)
    wc.at[idx, "stage"] = "LAST_16"
    wc.at[idx, "status"] = "FINISHED"
    if hg > ag:
        r16_winners[idx] = home
    elif ag > hg:
        r16_winners[idx] = away
    else:
        r16_winners[idx] = home  # Switzerland won on pens vs Colombia
    log.info(f"  R16 idx={idx}: {home} {hg}-{ag} {away}")

# === 3. QF（idx 96-99）填入对阵 + 赛果 ===
#   96: France vs Morocco → 2-0
#   97: Spain vs Belgium → 2-1
#   98: England vs Norway → 2-1 (AET)
#   99: Argentina vs Switzerland → 3-1 (AET)
qf_fixtures = {
    96: ("France", "Morocco", 2, 0),
    97: ("Spain", "Belgium", 2, 1),
    98: ("England", "Norway", 2, 1),
    99: ("Argentina", "Switzerland", 3, 1),
}

qf_winners = {}
for idx, (home, away, hg, ag) in qf_fixtures.items():
    wc.at[idx, "Home"] = home
    wc.at[idx, "Away"] = away
    wc.at[idx, "HomeGoals"] = float(hg)
    wc.at[idx, "AwayGoals"] = float(ag)
    wc.at[idx, "stage"] = "QUARTER_FINALS"
    wc.at[idx, "status"] = "FINISHED"
    qf_winners[idx] = home if hg >= ag else away
    log.info(f"  QF idx={idx}: {home} {hg}-{ag} {away}")

# === 4. SF（idx 100-101）填入对阵，保留TIMED ===
#   100: France vs Spain
#   101: England vs Argentina
sf_fixtures = {
    100: ("France", "Spain"),
    101: ("England", "Argentina"),
}

for idx, (home, away) in sf_fixtures.items():
    wc.at[idx, "Home"] = home
    wc.at[idx, "Away"] = away
    wc.at[idx, "stage"] = "SEMI_FINALS"
    wc.at[idx, "status"] = "TIMED"
    wc.at[idx, "HomeGoals"] = -1.0
    wc.at[idx, "AwayGoals"] = -1.0
    log.info(f"  SF idx={idx}: {home} vs {away} (TIMED)")

with open(path, "wb") as f:
    pickle.dump(wc, f)
log.info("\nall_wc_matches.pkl 更新完成")

# === 5. 更新 finished_matches.json ===
finished_path = os.path.join(DATA_DIR, "finished_matches.json")
with open(finished_path) as f:
    finished = json.load(f)

# Collect all finished matches (R32 remaining + R16 + QF)
all_new_results = {}

# R32 remaining
for idx, (hg, ag) in r32_results.items():
    h = wc.at[idx, "Home"]
    a = wc.at[idx, "Away"]
    all_new_results[idx] = (h, a, hg, ag, "LAST_32")

# R16
for idx, (home, away, hg, ag) in r16_fixtures.items():
    all_new_results[idx] = (home, away, hg, ag, "LAST_16")

# QF
for idx, (home, away, hg, ag) in qf_fixtures.items():
    all_new_results[idx] = (home, away, hg, ag, "QUARTER_FINALS")

for idx, (h, a, hg, ag, stage) in all_new_results.items():
    result = "H" if hg > ag else ("D" if hg == ag else "A")
    match_key = f"{stage}_{h}_{a}".lower().replace(" ", "_").replace("-", "_")
    if match_key not in finished:
        finished[match_key] = {
            "home_team": h,
            "away_team": a,
            "actual_home_score": hg,
            "actual_away_score": ag,
            "actual_result": result,
            "stage": stage,
            "date": str(wc.iloc[idx]["Date"])[:10] if "Date" in wc.columns else "",
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

with open(finished_path, "w") as f:
    json.dump(finished, f, indent=2)
log.info(f"finished_matches.json 更新: {len(finished)} 场")

# === 6. 更新 qualification_status.json ===
qual_path = os.path.join(DATA_DIR, "qualification_status.json")
with open(qual_path) as f:
    qual = json.load(f)

qual["_updated"] = "2026-07-12"
qual["knockout_results"] = {
    "r32_complete": True,
    "r16_complete": True,
    "qf_complete": True,
}
qual["sf_matchups"] = [
    {"match": 100, "home": "France", "away": "Spain", "date": "2026-07-14", "status": "TIMED"},
    {"match": 101, "home": "England", "away": "Argentina", "date": "2026-07-15", "status": "TIMED"},
]

with open(qual_path, "w") as f:
    json.dump(qual, f, indent=2)
log.info("qualification_status.json 已更新")

log.info("\n✅ 全部更新完成!")
print(f"  R32补赛: {len(r32_results)} 场")
print(f"  R16完赛: {len(r16_fixtures)} 场")
print(f"  QF完赛:  {len(qf_fixtures)} 场")
print(f"  SF待赛:  2 场 (France vs Spain, England vs Argentina)")
