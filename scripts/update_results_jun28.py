"""
update_results_jun28.py - 回填小组赛第三轮最后6场赛果（6月28日 J/K/L组）
"""
import pickle, logging, os, pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

# June 28 results (the remaining 6 matchday 3 games)
MD3_RESULTS = [
    # Group L - June 28
    ("Croatia", "Ghana", 2, 1, "GROUP_L"),
    ("Panama", "England", 0, 2, "GROUP_L"),
    # Group K - June 28
    ("Colombia", "Portugal", 0, 0, "GROUP_K"),
    ("Congo DR", "Uzbekistan", 3, 1, "GROUP_K"),
    # Group J - June 28
    ("Algeria", "Austria", 3, 3, "GROUP_J"),
    ("Jordan", "Argentina", 1, 3, "GROUP_J"),
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
        wc.at[idx, "status"] = "FINISHED"
        updated += 1
        log.info(f"  更新: {home} {hg}-{ag} {away}")

    if not_found:
        log.warning(f"未找到: {not_found}")

    with open(path, "wb") as f:
        pickle.dump(wc, f)

    total_finished = len(wc[wc['HomeGoals'] >= 0])
    log.info(f"\n完成: 更新 {updated} 场, 未找到 {len(not_found)}")
    log.info(f"总比赛: {len(wc)}, 有比分: {total_finished}")
    return updated


if __name__ == "__main__":
    update_results()
