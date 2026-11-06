"""
Update France 0-2 Spain (SF1) result in all data files
"""
import pickle, json, os, logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

# === 1. all_wc_matches.pkl ===
path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
with open(path, "rb") as f:
    wc = pickle.load(f)

idx = 100
old_hg = int(wc.at[idx, "HomeGoals"])
old_ag = int(wc.at[idx, "AwayGoals"])
wc.at[idx, "HomeGoals"] = 0.0
wc.at[idx, "AwayGoals"] = 2.0
log.info(f"  all_wc_matches idx={idx}: {wc.at[idx,'Home']} {old_hg}-{old_ag} → 0-2 (Spain wins)")

with open(path, "wb") as f:
    pickle.dump(wc, f)
log.info("  all_wc_matches.pkl ✓")

# === 2. finished_matches.json ===
finished_path = os.path.join(DATA_DIR, "finished_matches.json")
with open(finished_path) as f:
    finished = json.load(f)

key = "semi_finals_france_spain"
finished[key] = {
    "home_team": "France",
    "away_team": "Spain",
    "date": "2026-07-14",
    "actual_result": "A",
    "actual_home_score": 0,
    "actual_away_score": 2,
    "stage": "SEMI_FINALS"
}
log.info(f"  finished_matches.json: added {key}")

with open(finished_path, "w") as f:
    json.dump(finished, f, indent=2)
log.info("  finished_matches.json ✓")

# === 3. qualification_status.json ===
qual_path = os.path.join(DATA_DIR, "qualification_status.json")
with open(qual_path) as f:
    qual = json.load(f)

if "sf_results" not in qual:
    qual["sf_results"] = {}
qual["sf_results"]["france_spain"] = "0-2"
qual["knockout_results"]["sf_complete"] = True

log.info("  qualification_status.json: sf_results.france_spain = 0-2, sf_complete = True")

with open(qual_path, "w") as f:
    json.dump(qual, f, indent=2)
log.info("  qualification_status.json ✓")

log.info("\n✅ France 0-2 Spain updated")
