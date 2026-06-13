"""
feature_engineer.py - 世界杯特征工程
为国家级球队适配特征：FIFA排名、历史世界杯表现、近期状态
"""
import pandas as pd
import numpy as np
import os, pickle, logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_fifa_rankings():
    path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


TEAM_NAME_MAP = {
    "United States": "USA",
    "Czechia": "Czech Republic",
}

def get_team_rank_points(team, rankings_df):
    row = rankings_df[rankings_df["Team"] == team]
    if len(row) > 0:
        return int(row["RankPoints"].values[0])
    mapped = TEAM_NAME_MAP.get(team)
    if mapped:
        row = rankings_df[rankings_df["Team"] == mapped]
        if len(row) > 0:
            return int(row["RankPoints"].values[0])
    for t in rankings_df["Team"]:
        if t.lower() == team.lower():
            return int(rankings_df[rankings_df["Team"] == t]["RankPoints"].values[0])
    return 1500


# 五大联赛球员密度缓存
_TOP5_CACHE = None

def load_top5_data():
    global _TOP5_CACHE
    if _TOP5_CACHE is not None:
        return _TOP5_CACHE
    path = os.path.join(DATA_DIR, "team_top5.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            _TOP5_CACHE = json.load(f)
        return _TOP5_CACHE
    return {}

def get_team_top5_ratio(team, top5_data):
    return top5_data.get(team, 0.10)


def compute_team_wc_history(all_matches, team, window=2):
    """计算球队在历史世界杯中的数据"""
    team_matches = all_matches[
        ((all_matches["Home"] == team) | (all_matches["Away"] == team)) &
        (all_matches["HomeGoals"] >= 0)
    ].sort_values("Date") if "Date" in all_matches.columns else all_matches[
        (all_matches["Home"] == team) | (all_matches["Away"] == team)
    ]

    if len(team_matches) == 0:
        return {
            "wc_avg_gf": 1.0, "wc_avg_ga": 1.2, "wc_win_rate": 0.35,
            "wc_matches": 0, "wc_avg_pts": 1.1,
        }

    gf_list, ga_list, pts_list = [], [], []
    for _, row in team_matches.iterrows():
        sh, sa = int(row.get("HomeGoals", -1)), int(row.get("AwayGoals", -1))
        if sh < 0 or sa < 0:
            continue
        if row["Home"] == team:
            gf_list.append(sh); ga_list.append(sa)
        else:
            gf_list.append(sa); ga_list.append(sh)
        if gf_list[-1] > ga_list[-1]:
            pts_list.append(3)
        elif gf_list[-1] == ga_list[-1]:
            pts_list.append(1)
        else:
            pts_list.append(0)

    n = len(gf_list)
    if n == 0:
        return {"wc_avg_gf": 1.0, "wc_avg_ga": 1.2, "wc_win_rate": 0.35, "wc_matches": 0, "wc_avg_pts": 1.1}

    return {
        "wc_avg_gf": np.mean(gf_list),
        "wc_avg_ga": np.mean(ga_list),
        "wc_win_rate": sum(1 for p in pts_list if p == 3) / n,
        "wc_matches": n,
        "wc_avg_pts": np.mean(pts_list),
        "wc_recent_gf": np.mean(gf_list[-min(5, len(gf_list)):]),
        "wc_recent_ga": np.mean(ga_list[-min(5, len(ga_list)):]),
    }


def compute_recent_form(all_matches, team, match_date=None, window=5):
    """计算球队近期状态（从所有国际比赛中）"""
    # 过滤到比赛日期之前
    hist = all_matches.copy()
    if match_date is not None and "Date" in hist.columns:
        try:
            cut = pd.to_datetime(match_date)
            hist = hist[hist["Date"].apply(
                lambda x: pd.notna(x) and pd.to_datetime(x) < cut if isinstance(x, (datetime, pd.Timestamp, str)) else False
            )]
        except:
            pass

    team_matches = hist[
        ((hist["Home"] == team) | (hist["Away"] == team)) &
        (hist["HomeGoals"] >= 0)
    ].sort_values("Date") if "Date" in hist.columns else hist[
        (hist["Home"] == team) | (hist["Away"] == team)
    ]

    recent = team_matches.tail(window)
    gf_list, ga_list, pts_list = [], [], []
    for _, row in recent.iterrows():
        sh, sa = int(row.get("HomeGoals", -1)), int(row.get("AwayGoals", -1))
        if sh < 0 or sa < 0:
            continue
        if row["Home"] == team:
            gf_list.append(sh); ga_list.append(sa)
        else:
            gf_list.append(sa); ga_list.append(sh)
        if gf_list[-1] > ga_list[-1]:
            pts_list.append(3)
        elif gf_list[-1] == ga_list[-1]:
            pts_list.append(1)
        else:
            pts_list.append(0)

    n = len(gf_list)
    if n == 0:
        return {"form_avg_gf": 1.1, "form_avg_ga": 1.1, "form_pts": 1.2, "form_n": 0, "form_streak": 0}

    return {
        "form_avg_gf": np.mean(gf_list),
        "form_avg_ga": np.mean(ga_list),
        "form_pts": np.mean(pts_list),
        "form_n": n,
        "form_streak": sum(1 for p in pts_list[-3:] if p == 3) / 3 if len(pts_list) >= 3 else 0.33,
    }


def build_match_features(match_row, all_wc_matches, rankings_df):
    """为单场世界杯比赛构建特征向量"""
    features = {}
    home = match_row["Home"]
    away = match_row["Away"]

    # 1. FIFA排名特征（核心特征）
    home_rank = get_team_rank_points(home, rankings_df)
    away_rank = get_team_rank_points(away, rankings_df)
    features["home_rank_pts"] = home_rank
    features["away_rank_pts"] = away_rank
    features["rank_diff"] = home_rank - away_rank
    features["rank_ratio"] = home_rank / max(away_rank, 1)
    features["rank_relative"] = (home_rank - away_rank) / (home_rank + away_rank + 1)

    # 1b. 五大联赛球员密度特征
    top5_data = load_top5_data()
    home_top5 = get_team_top5_ratio(home, top5_data)
    away_top5 = get_team_top5_ratio(away, top5_data)
    features["home_top5_ratio"] = home_top5
    features["away_top5_ratio"] = away_top5
    features["top5_diff"] = home_top5 - away_top5
    features["top5_relative"] = (home_top5 - away_top5) / (home_top5 + away_top5 + 0.01)

    # 2. 历史世界杯表现（如果数据存在）
    home_wc = compute_team_wc_history(all_wc_matches, home)
    away_wc = compute_team_wc_history(all_wc_matches, away)
    for k, v in home_wc.items():
        features[f"home_{k}"] = v
    for k, v in away_wc.items():
        features[f"away_{k}"] = v
    features["wc_gf_diff"] = home_wc["wc_avg_gf"] - away_wc["wc_avg_gf"]
    features["wc_ga_diff"] = home_wc["wc_avg_ga"] - away_wc["wc_avg_ga"]
    features["wc_win_rate_diff"] = home_wc["wc_win_rate"] - away_wc["wc_win_rate"]

    # 3. 近期状态（如果数据存在）
    match_date = match_row.get("Date", None)
    home_form = compute_recent_form(all_wc_matches, home, match_date)
    away_form = compute_recent_form(all_wc_matches, away, match_date)
    for k, v in home_form.items():
        features[f"home_{k}"] = v
    for k, v in away_form.items():
        features[f"away_{k}"] = v
    features["form_pts_diff"] = home_form["form_pts"] - away_form["form_pts"]
    features["form_gf_diff"] = home_form["form_avg_gf"] - away_form["form_avg_gf"]
    features["form_ga_diff"] = home_form["form_avg_ga"] - away_form["form_avg_ga"]

    # 4. 历史交锋 (H2H)，仅用已完赛数据
    finished = all_wc_matches[all_wc_matches["HomeGoals"] >= 0] if "HomeGoals" in all_wc_matches.columns else all_wc_matches
    h2h = finished[
        ((finished["Home"] == home) & (finished["Away"] == away)) |
        ((finished["Home"] == away) & (finished["Away"] == home))
    ]
    if len(h2h) > 0:
        home_wins = sum(1 for _, r in h2h.iterrows()
                        if (r["Home"] == home and int(r["HomeGoals"]) > int(r["AwayGoals"])) or
                           (r["Home"] == away and int(r["AwayGoals"]) > int(r["HomeGoals"])))
        features["h2h_home_win_rate"] = home_wins / len(h2h)
        features["h2h_n"] = len(h2h)
    else:
        # 无H2H时基于排名估算
        rank_ratio = home_rank / max(home_rank + away_rank, 1)
        features["h2h_home_win_rate"] = max(0.3, min(0.7, rank_ratio))
        features["h2h_n"] = 0

    # 5. 赔率特征（如果有）
    for col in ["odds_home", "odds_draw", "odds_away"]:
        try:
            val = match_row.get(col)
            if pd.notna(val) and val is not None and float(val) > 0:
                odd = float(val)
                features[col.replace("odds_", "") + "_odds"] = odd
                features[col.replace("odds_", "") + "_implied"] = 1.0 / odd
            else:
                features[col.replace("odds_", "") + "_odds"] = 2.0
                features[col.replace("odds_", "") + "_implied"] = 0.33
        except:
            features[col.replace("odds_", "") + "_odds"] = 2.0
            features[col.replace("odds_", "") + "_implied"] = 0.33

    # 6. 目标变量
    try:
        hg, ag = int(match_row.get("HomeGoals", -1)), int(match_row.get("AwayGoals", -1))
        if hg >= 0 and ag >= 0:
            features["result"] = "H" if hg > ag else ("D" if hg == ag else "A")
            features["is_finished"] = True
            features["home_goals"] = hg
            features["away_goals"] = ag
        else:
            features["result"] = "U"
            features["is_finished"] = False
    except:
        features["result"] = "U"
        features["is_finished"] = False

    features["home_team"] = home
    features["away_team"] = away
    features["date"] = str(match_row.get("Date", ""))[:10]
    match_date = match_row.get("Date", "")
    features["match_id"] = f"{home}_{away}_{str(match_date)[:10]}"

    return features


def build_feature_matrix(all_wc_matches, rankings_df):
    """构建完整特征矩阵"""
    log.info("构建世界杯特征矩阵...")

    if all_wc_matches is None or len(all_wc_matches) == 0:
        log.error("无比赛数据")
        return None

    df = all_wc_matches.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date") if "Date" in df.columns else df

    all_features = []
    total = len(df)
    for idx, (_, row) in enumerate(df.iterrows()):
        # 跳过参赛队未定的比赛（淘汰赛场次）
        home_name = row.get("Home")
        away_name = row.get("Away")
        if pd.isna(home_name) or pd.isna(away_name) or not str(home_name).strip() or not str(away_name).strip():
            continue
        try:
            feats = build_match_features(row, df, rankings_df)
            all_features.append(feats)
        except Exception as e:
            log.warning(f"#{idx} 特征构建失败: {e}")
        if (idx + 1) % 50 == 0:
            log.info(f"  进度: {idx+1}/{total}")

    result = pd.DataFrame(all_features)
    path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    with open(path, "wb") as f:
        pickle.dump(result, f)

    finished = result[result["is_finished"] == True]
    log.info(f"特征矩阵: {result.shape[0]} 行 x {result.shape[1]} 列")
    log.info(f"  已完赛: {len(finished)}")
    log.info(f"  待预测: {len(result[result['is_finished'] == False])}")
    return result


def load_feature_matrix():
    path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


if __name__ == "__main__":
    from data_collector import load_data, load_rankings
    wc = load_data()
    rankings = load_rankings()
    if wc is not None and rankings is not None:
        build_feature_matrix(wc, rankings)
