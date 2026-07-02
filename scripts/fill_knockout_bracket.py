"""
fill_knockout_bracket.py - 填入淘汰赛对阵（R32 已确认的 11 场, 剩余 5 场待 3rd place 分配）
"""
import pickle, logging, os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

# Confirmed R32 matchups
# DataFrame index 72-87 correspond to LAST_32 matches
# Each entry: (df_index, Home, Away, Date, group_name)
R32_FIXTURES = [
    (72, "South Africa",   "Canada",              "2026-06-28T19:00:00Z", "R32_AB"),
    (73, "Germany",        "Paraguay",            "2026-06-29T17:00:00Z", "R32_E3D"),
    (74, "Netherlands",    "Morocco",             "2026-06-29T20:30:00Z", "R32_FC"),
    (75, "Brazil",         "Japan",               "2026-06-30T01:00:00Z", "R32_CF"),
    (76, "France",         "Sweden",              "2026-06-30T17:00:00Z", "R32_I3F"),
    (77, "Ivory Coast",    "Norway",              "2026-06-30T21:00:00Z", "R32_EI"),
    (78, "Mexico",         None,                  "2026-07-01T16:00:00Z", "R32_A3X"),   # vs 3rd place TBD
    (79, "England",        None,                  "2026-07-01T20:00:00Z", "R32_L3X"),   # vs 3rd place TBD
    (80, "Belgium",        None,                  "2026-07-02T00:00:00Z", "R32_G3X"),   # vs 3rd place TBD
    (81, "United States",  "Bosnia-Herzegovina",  "2026-07-02T19:00:00Z", "R32_DB"),
    (82, "Spain",          "Austria",             "2026-07-02T23:00:00Z", "R32_HJ"),
    (83, "Croatia",        "Portugal",            "2026-07-03T03:00:00Z", "R32_LK"),
    (84, "Switzerland",    None,                  "2026-07-03T18:00:00Z", "R32_B3X"),   # vs 3rd place TBD
    (85, "Australia",      "Egypt",               "2026-07-03T22:00:00Z", "R32_DG"),
    (86, "Argentina",      "Cape Verde Islands",  "2026-07-04T01:30:00Z", "R32_JH"),
    (87, "Colombia",       None,                  "2026-07-04T17:00:00Z", "R32_K3X"),   # vs 3rd place TBD
]


def fill_bracket():
    path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
    with open(path, "rb") as f:
        wc = pickle.load(f)

    filled = 0
    for idx, home, away, date_str, group in R32_FIXTURES:
        wc.at[idx, "Home"] = home
        wc.at[idx, "Away"] = away
        wc.at[idx, "Date"] = date_str
        wc.at[idx, "stage"] = "LAST_32"
        wc.at[idx, "group"] = group
        wc.at[idx, "赛季"] = "2026"
        wc.at[idx, "比赛日"] = 4  # knockout round 1
        wc.at[idx, "status"] = "TIMED"
        wc.at[idx, "HomeGoals"] = -1.0
        wc.at[idx, "AwayGoals"] = -1.0
        if away is None:
            log.info(f"  R32 #{idx}: {home} vs TBD ({group})")
        else:
            log.info(f"  R32 #{idx}: {home} vs {away} ({group})")
            filled += 1

    with open(path, "wb") as f:
        pickle.dump(wc, f)

    log.info(f"\n完成: 填写 {filled} 场确认对阵, 剩余 {16-filled} 场待 3rd place 分配")


if __name__ == "__main__":
    fill_bracket()
