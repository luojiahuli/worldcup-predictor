"""
fix_90min_scores.py - 将加时赛比分修正为90分钟常规时间比分（足彩口径）
修正3场加时赛绝杀的比赛：
  Belgium 3-2 Senegal → 2-2
  England 2-1 Norway → 1-1
  Argentina 3-1 Switzerland → 1-1
"""
import pickle, json, os, logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

# === 1. 修正 all_wc_matches.pkl ===
path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
with open(path, "rb") as f:
    wc = pickle.load(f)

corrections = {
    # idx: (home_goals_90, away_goals_90, result_90)
    80: (2, 2),   # Belgium vs Senegal: 3-2 → 2-2
    98: (1, 1),   # England vs Norway: 2-1 → 1-1
    99: (1, 1),   # Argentina vs Switzerland: 3-1 → 1-1
}

for idx, (hg, ag) in corrections.items():
    old_hg = wc.at[idx, "HomeGoals"]
    old_ag = wc.at[idx, "AwayGoals"]
    wc.at[idx, "HomeGoals"] = float(hg)
    wc.at[idx, "AwayGoals"] = float(ag)
    log.info(f"  修正 idx={idx}: {wc.at[idx,'Home']} {int(old_hg)}-{int(old_ag)} → {hg}-{ag}")

with open(path, "wb") as f:
    pickle.dump(wc, f)
log.info("all_wc_matches.pkl 已修正")

# === 2. 修正 finished_matches.json（更新比分+结果）===
finished_path = os.path.join(DATA_DIR, "finished_matches.json")
with open(finished_path) as f:
    finished = json.load(f)

# Build key mappings
key_map = {
    "last_32_belgium_senegal": (2, 2, "D"),
    "quarter_finals_england_norway": (1, 1, "D"),
    "quarter_finals_argentina_switzerland": (1, 1, "D"),
}

for key, (hg, ag, result) in key_map.items():
    if key in finished:
        entry = finished[key]
        old = (entry["actual_home_score"], entry["actual_away_score"], entry["actual_result"])
        entry["actual_home_score"] = hg
        entry["actual_away_score"] = ag
        entry["actual_result"] = result
        log.info(f"  finished修正 {key}: {old} → ({hg}, {ag}, {result})")

with open(finished_path, "w") as f:
    json.dump(finished, f, indent=2)
log.info(f"finished_matches.json 已修正")

# === 3. 修正 qualification_status.json 中的赛果 ===
qual_path = os.path.join(DATA_DIR, "qualification_status.json")
with open(qual_path) as f:
    qual = json.load(f)

# Update R32 matchups
for m in qual.get("r32_matchups", []):
    if m.get("match") == 80:
        m["result"] = "2-2"
        log.info("qualification R32 match 80: set to 2-2")

# Update knockout_results
if "qf_results" not in qual:
    qual["qf_results"] = {}
qual["qf_results"]["england_norway"] = "1-1 (1-1 AET, 2-1 FT)"
qual["qf_results"]["argentina_switzerland"] = "1-1 (1-1 AET, 3-1 FT)"

with open(qual_path, "w") as f:
    json.dump(qual, f, indent=2)
log.info("qualification_status.json 已修正")

log.info("\n✅ 修正完成！共 3 场")
