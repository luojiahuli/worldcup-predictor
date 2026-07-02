"""
update_results_jul02.py - 更新7月2日前所有已确定的赛果
包括：小组赛补赛(12场) + 淘汰赛R32(截至7月2日已完赛场次)
"""
import pickle, json, os, logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

# === 1. 更新 all_wc_matches.pkl ===
path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
with open(path, "rb") as f:
    wc = pickle.load(f)

# ---- 小组赛补赛（12场）- ----
group_results = {
    # idx: (home_goals, away_goals)
    35: (0, 4),   # Tunisia 0-4 Japan (Jun 21)
    36: (4, 0),   # Spain 4-0 Saudi Arabia (Jun 21)
    37: (0, 0),   # Belgium 0-0 Iran (Jun 21)
    38: (2, 2),   # Uruguay 2-2 Cape Verde (Jun 21)
    40: (2, 0),   # Argentina 2-0 Austria (Jun 22)
    41: (3, 0),   # France 3-0 Iraq (Jun 22)
    42: (3, 2),   # Norway 3-2 Senegal (Jun 23)
    43: (1, 2),   # Jordan 1-2 Algeria (Jun 23)
    44: (5, 0),   # Portugal 5-0 Uzbekistan (Jun 23)
    45: (0, 0),   # England 0-0 Ghana (Jun 23)
    46: (0, 1),   # Panama 0-1 Croatia (Jun 23)
    47: (1, 0),   # Colombia 1-0 Congo DR (Jun 24)
}

for idx, (hg, ag) in group_results.items():
    wc.at[idx, "HomeGoals"] = float(hg)
    wc.at[idx, "AwayGoals"] = float(ag)
    wc.at[idx, "status"] = "FINISHED"
    log.info(f"  小组赛 idx={idx}: {wc.at[idx, 'Home']} {hg}-{ag} {wc.at[idx, 'Away']}")

# ---- R32 淘汰赛完赛（9场）- ----
# 填写TBD对手 + 赛果
r32_updates = {
    # idx: (home, away, hg, ag)
    72: ("South Africa", "Canada", 0, 1),           # South Africa 0-1 Canada
    73: ("Germany", "Paraguay", 1, 1),              # 1-1, Paraguay win on pens
    74: ("Netherlands", "Morocco", 1, 1),            # 1-1, Morocco win on pens
    75: ("Brazil", "Japan", 2, 1),                   # already done
    76: ("France", "Sweden", 3, 0),                  # France 3-0 Sweden
    77: ("Ivory Coast", "Norway", 1, 2),             # Ivory Coast 1-2 Norway
    78: ("Mexico", "Ecuador", 2, 0),                 # Mexico 2-0 Ecuador
    79: ("England", "Congo DR", 2, 1),               # England 2-1 DR Congo
    80: ("Belgium", "Senegal", 3, 2),                # Belgium 3-2 Senegal (aet)
    81: ("United States", "Bosnia-Herzegovina", 2, 0), # USA 2-0 Bosnia
}

for idx, (home, away, hg, ag) in r32_updates.items():
    wc.at[idx, "Home"] = home
    wc.at[idx, "Away"] = away
    wc.at[idx, "HomeGoals"] = float(hg)
    wc.at[idx, "AwayGoals"] = float(ag)
    wc.at[idx, "status"] = "FINISHED"
    log.info(f"  R32 idx={idx}: {home} {hg}-{ag} {away}")

with open(path, "wb") as f:
    pickle.dump(wc, f)
log.info(f"\nall_wc_matches.pkl 更新完成")

# === 2. 更新 finished_matches.json ===
finished_path = os.path.join(DATA_DIR, "finished_matches.json")
with open(finished_path) as f:
    finished = json.load(f)

all_updates = {**group_results, **{k: (v[2], v[3]) for k, v in r32_updates.items()}}

# 匹配表: idx -> (home_team, away_team, date)
idx_to_match = {}
for idx in all_updates:
    if idx in r32_updates:
        h, a, *_ = r32_updates[idx]
        date_str = str(wc.at[idx, "Date"])[:10] if "Date" in wc.columns else "2026-06-30"
        stage = "LAST_32"
    else:
        h = wc.at[idx, "Home"]
        a = wc.at[idx, "Away"]
        date_str = str(wc.at[idx, "Date"])[:10] if "Date" in wc.columns else "2026-06-21"
        stage = "GROUP_STAGE"
    idx_to_match[idx] = (h, a, date_str, stage)

for idx, (hg, ag) in all_updates.items():
    h, a, date_str, stage = idx_to_match[idx]
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
            "date": date_str,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

with open(finished_path, "w") as f:
    json.dump(finished, f, indent=2)
log.info(f"finished_matches.json 更新: {len(finished)} 场")

# === 3. 更新 qualification_status.json - R32 matchup info ===
qual_path = os.path.join(DATA_DIR, "qualification_status.json")
with open(qual_path) as f:
    qual = json.load(f)

qual["_updated"] = "2026-07-02"
# 更新 R32 matchups with confirmed opponents
qual["r32_matchups"] = [
    {"match": 73, "home": "South Africa", "away": "Canada", "date": "2026-06-28", "venue": "Los Angeles", "result": "0-1"},
    {"match": 76, "home": "Brazil", "away": "Japan", "date": "2026-06-29", "venue": "Houston", "result": "2-1"},
    {"match": 74, "home": "Germany", "away": "Paraguay", "date": "2026-06-29", "venue": "Boston", "result": "1-1 (3-4 pens)"},
    {"match": 75, "home": "Netherlands", "away": "Morocco", "date": "2026-06-30", "venue": "Monterrey", "result": "1-1 (2-3 pens)"},
    {"match": 78, "home": "Ivory Coast", "away": "Norway", "date": "2026-06-30", "venue": "Dallas", "result": "1-2"},
    {"match": 77, "home": "France", "away": "Sweden", "date": "2026-06-30", "venue": "New Jersey", "result": "3-0"},
    {"match": 79, "home": "Mexico", "away": "Ecuador", "date": "2026-07-01", "venue": "Mexico City", "result": "2-0"},
    {"match": 80, "home": "England", "away": "Congo DR", "date": "2026-07-01", "venue": "Atlanta", "result": "2-1"},
    {"match": 82, "home": "Belgium", "away": "Senegal", "date": "2026-07-01", "venue": "Seattle", "result": "3-2"},
    {"match": 81, "home": "United States", "away": "Bosnia-Herzegovina", "date": "2026-07-02", "venue": "San Francisco", "result": "2-0"},
    {"match": 84, "home": "Spain", "away": "Austria", "date": "2026-07-02", "venue": "Los Angeles", "result": "TBD"},
    {"match": 83, "home": "Croatia", "away": "Portugal", "date": "2026-07-03", "venue": "Toronto", "result": "TBD"},
    {"match": 85, "home": "Switzerland", "away": "Algeria", "date": "2026-07-03", "venue": "Vancouver", "result": "TBD"},
    {"match": 88, "home": "Australia", "away": "Egypt", "date": "2026-07-03", "venue": "Dallas", "result": "TBD"},
    {"match": 86, "home": "Argentina", "away": "Cape Verde Islands", "date": "2026-07-03", "venue": "Miami", "result": "TBD"},
    {"match": 87, "home": "Colombia", "away": "Ghana", "date": "2026-07-04", "venue": "Kansas City", "result": "TBD"},
]

with open(qual_path, "w") as f:
    json.dump(qual, f, indent=2)
log.info(f"qualification_status.json 已更新")

log.info("\n✅ 全部更新完成!")
print(f"  小组赛补赛: {len(group_results)} 场")
print(f"  R32完赛: {len(r32_updates)} 场")
print(f"  R32待赛: 6 场(Spain-Austria, Croatia-Portugal, Switzerland-Algeria, Australia-Egypt, Argentina-Cape Verde, Colombia-Ghana)")
