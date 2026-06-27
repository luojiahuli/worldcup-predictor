"""
update_results.py - 回填小组赛第三轮赛果到 all_wc_matches.pkl
"""
import pickle, logging, os, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

# Matchday 3 results (confirmed from web search, June 24-27 2026)
# Format: (Home, Away, HomeGoals, AwayGoals, group)
MD3_RESULTS = [
    # June 24 - Groups A, B, C
    ("Switzerland", "Canada", 2, 1, "GROUP_B"),
    ("Bosnia-Herzegovina", "Qatar", 3, 1, "GROUP_B"),
    ("Scotland", "Brazil", 0, 3, "GROUP_C"),
    ("Morocco", "Haiti", 4, 2, "GROUP_C"),
    ("Czechia", "Mexico", 0, 3, "GROUP_A"),
    ("South Africa", "South Korea", 1, 0, "GROUP_A"),
    # June 25 - Groups E, F
    ("Ecuador", "Germany", 2, 1, "GROUP_E"),
    ("Curaçao", "Ivory Coast", 0, 2, "GROUP_E"),
    ("Tunisia", "Netherlands", 1, 3, "GROUP_F"),
    ("Japan", "Sweden", 1, 1, "GROUP_F"),
    # June 26 - Groups D, I
    ("Turkey", "United States", 3, 2, "GROUP_D"),
    ("Paraguay", "Australia", 0, 0, "GROUP_D"),
    ("Norway", "France", 1, 4, "GROUP_I"),
    ("Senegal", "Iraq", 5, 0, "GROUP_I"),
    # June 27 - Groups G, H
    ("Uruguay", "Spain", 0, 1, "GROUP_H"),
    ("Cape Verde Islands", "Saudi Arabia", 0, 0, "GROUP_H"),
    ("New Zealand", "Belgium", 1, 5, "GROUP_G"),
    ("Egypt", "Iran", 1, 1, "GROUP_G"),
]


def update_results():
    path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
    with open(path, "rb") as f:
        wc = pickle.load(f)

    updated = 0
    not_found = []
    for home, away, hg, ag, group in MD3_RESULTS:
        mask = (wc["Home"] == home) & (wc["Away"] == away) & (wc["group"] == group)
        match = wc[mask]
        if len(match) == 0:
            # try without group filter
            mask2 = (wc["Home"] == home) & (wc["Away"] == away)
            match2 = wc[mask2]
            if len(match2) == 0:
                not_found.append(f"{home} vs {away}")
                continue
            idx = match2.index[0]
        else:
            idx = match.index[0]

        old_hg = int(wc.at[idx, "HomeGoals"]) if not pd.isna(wc.at[idx, "HomeGoals"]) else -1
        old_ag = int(wc.at[idx, "AwayGoals"]) if not pd.isna(wc.at[idx, "AwayGoals"]) else -1
        if old_hg >= 0 and old_ag >= 0:
            log.info(f"  已存在: {home} {old_hg}-{old_ag} {away}, 跳过")
            continue

        wc.at[idx, "HomeGoals"] = hg
        wc.at[idx, "AwayGoals"] = ag
        updated += 1
        log.info(f"  更新: {home} {hg}-{ag} {away}")

    if not_found:
        log.warning(f"未找到: {not_found}")

    with open(path, "wb") as f:
        pickle.dump(wc, f)

    log.info(f"\n完成: 更新 {updated} 场, 未找到 {len(not_found)}")
    log.info(f"总比赛: {len(wc)}, 有比分: {len(wc[wc['HomeGoals'] >= 0])}")
    return updated


if __name__ == "__main__":
    update_results()
