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


# ─── 社交媒体数据 ───────────────────────────────────────
_SOCIAL_CACHE = None
_MEDIA_CACHE = None

def load_social_data():
    global _SOCIAL_CACHE
    if _SOCIAL_CACHE is not None:
        return _SOCIAL_CACHE
    path = os.path.join(DATA_DIR, "social_heat.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        _SOCIAL_CACHE = df.set_index("team").to_dict("index")
        return _SOCIAL_CACHE
    return {}

def load_media_data():
    global _MEDIA_CACHE
    if _MEDIA_CACHE is not None:
        return _MEDIA_CACHE
    path = os.path.join(DATA_DIR, "media_sentiment.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        _MEDIA_CACHE = df.set_index("team").to_dict("index")
        return _MEDIA_CACHE
    return {}

def get_social_features(team, social_data, media_data):
    """获取球队社交媒体特征"""
    s = social_data.get(team, {})
    m = media_data.get(team, {})
    return {
        "heat_score": s.get("heat_score", 0.5),
        "heat_change": s.get("heat_change", 0),
        "sentiment_ratio": s.get("sentiment_ratio", 0.5),
        "mention_count": s.get("mention_count", 0),
        "media_sentiment": m.get("sentiment_score", 0.5),
        "positive_ratio": m.get("positive_ratio", 0.5),
        "news_volume": m.get("news_volume", 10),
    }


# ─── 英超球员密度 ──────────────────────────────────────
_EPL_CACHE = None

def load_epl_data():
    global _EPL_CACHE
    if _EPL_CACHE is not None:
        return _EPL_CACHE
    path = os.path.join(DATA_DIR, "epl_density.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            _EPL_CACHE = json.load(f)
        return _EPL_CACHE
    return {}

def get_team_epl_ratio(team, epl_data):
    return epl_data.get(team, 0.10)


# ─── 主帅特征 ──────────────────────────────────────────
_COACH_CACHE = None

def load_coach_data():
    global _COACH_CACHE
    if _COACH_CACHE is not None:
        return _COACH_CACHE
    path = os.path.join(DATA_DIR, "coaches.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            _COACH_CACHE = json.load(f)
        return _COACH_CACHE
    return {}

def get_coach_features(team, coach_data):
    info = coach_data.get(team, {})
    return {
        "coach_reputation": info.get("reputation", 0.55),
        "coach_tactical": info.get("tactical_flex", 0.55),
        "coach_big_match": info.get("big_match", 0.50),
    }


# ─── 球员数据库（身价/明星/阵容深度） ──────────────────
_PLAYER_DB_CACHE = None

def load_player_database():
    """加载完整球员数据库"""
    global _PLAYER_DB_CACHE
    if _PLAYER_DB_CACHE is not None:
        return _PLAYER_DB_CACHE
    path = os.path.join(DATA_DIR, "player_database.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            data = json.load(f)
        _PLAYER_DB_CACHE = {k: v for k, v in data.items() if not k.startswith("_")}
        return _PLAYER_DB_CACHE
    return {}

def get_player_db_features(team, player_db):
    """从球员数据库提取团队级特征"""
    info = player_db.get(team, {})
    star_players = info.get("star_players", [])

    if star_players:
        ratings = [p.get("rating", 75) for p in star_players]
        top_rating = max(ratings) if ratings else 75
        avg_top3 = sum(ratings) / len(ratings) if ratings else 75
        total_goals = sum(p.get("goals_intl", 0) for p in star_players)
    else:
        top_rating, avg_top3, total_goals = 75, 75, 0

    return {
        "top_player_rating": top_rating,
        "top3_avg_rating": avg_top3,
        "star_goals": total_goals,
        "squad_value": info.get("squad_value_euros", 100_000_000),
        "avg_age": info.get("avg_age", 27.0),
        "avg_rating": info.get("avg_rating", 78.0),
        "key_player_form": info.get("key_player_form", 7.5),
        "attack_strength": info.get("attack_strength", 0.65),
        "defense_strength": info.get("defense_strength", 0.65),
        "midfield_strength": info.get("midfield_strength", 0.65),
    }


# ─── 球员联赛熟悉度特征 ────────────────────────────────
def compute_familiarity_features(home_team, away_team, player_db):
    """计算两队球员的联赛熟悉程度：同联赛交锋过 = 熟悉对手"""
    home_info = player_db.get(home_team, {})
    away_info = player_db.get(away_team, {})
    home_players = home_info.get("star_players", [])
    away_players = away_info.get("star_players", [])

    if not home_players or not away_players:
        return {"home_def_vs_away_fwd_fam": 0, "away_def_vs_home_fwd_fam": 0,
                "def_fwd_familiarity": 0, "league_overlap": 0, "same_club_connections": 0}

    # 按位置分组
    def _by_pos(players):
        groups = {"DF": [], "FW": [], "MF": [], "GK": []}
        for p in players:
            groups.get(p.get("pos", "MF"), groups["MF"]).append(p)
        return groups

    home_pos = _by_pos(home_players)
    away_pos = _by_pos(away_players)

    def _league_overlap_rating(players_a, players_b):
        """计算两组球员的联赛重叠（按评分加权）"""
        if not players_a or not players_b:
            return 0.0
        total, count = 0.0, 0
        for pa in players_a:
            for pb in players_b:
                if pa.get("league") and pa["league"] == pb.get("league"):
                    avg_r = (pa.get("rating", 75) + pb.get("rating", 75)) / 2.0
                    total += avg_r / 100.0
                count += 1
        return total / max(count, 1)

    # 1) 主队后卫 vs 客队前锋（主队后卫熟悉客队前锋）
    home_df_vs_away_fw = _league_overlap_rating(home_pos["DF"], away_pos["FW"])

    # 2) 客队后卫 vs 主队前锋（客队后卫熟悉主队前锋）
    away_df_vs_home_fw = _league_overlap_rating(away_pos["DF"], home_pos["FW"])

    # 3) 所有球员的通用联赛重叠
    all_overlap = _league_overlap_rating(home_players, away_players)

    # 4) 同俱乐部连接（不同国家队但同一俱乐部，如皇马队友）
    club_pairs, total_club_pairs = 0.0, 0
    for hp in home_players:
        for ap in away_players:
            total_club_pairs += 1
            if hp.get("club") and hp["club"] == ap.get("club"):
                avg_r = (hp.get("rating", 75) + ap.get("rating", 75)) / 2.0
                club_pairs += avg_r / 100.0
    same_club = club_pairs / max(total_club_pairs, 1)

    return {
        "home_def_vs_away_fwd_fam": round(home_df_vs_away_fw, 4),
        "away_def_vs_home_fwd_fam": round(away_df_vs_home_fw, 4),
        "def_fwd_familiarity": round(home_df_vs_away_fw - away_df_vs_home_fw, 4),
        "league_overlap": round(all_overlap, 4),
        "same_club_connections": round(same_club, 4),
    }


# ─── 关键球员状态 ──────────────────────────────────────
_PLAYER_STATUS_CACHE = None

def load_player_status():
    global _PLAYER_STATUS_CACHE
    if _PLAYER_STATUS_CACHE is not None:
        return _PLAYER_STATUS_CACHE
    path = os.path.join(DATA_DIR, "player_status.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            _PLAYER_STATUS_CACHE = json.load(f)
        return _PLAYER_STATUS_CACHE
    return {}

def get_player_features(team, player_data):
    """获取球队关键球员状态特征"""
    info = player_data.get(team, {})
    return {
        "key_player_missing": info.get("missing_count", 0),
        "injury_severity": info.get("severity", 0.0),
        "star_player_available": info.get("star_available", 1.0),
    }


# ─── 真实赔率特征 ──────────────────────────────────────
_REAL_ODDS_CACHE = None

def load_real_odds():
    global _REAL_ODDS_CACHE
    if _REAL_ODDS_CACHE is not None:
        return _REAL_ODDS_CACHE
    path = os.path.join(DATA_DIR, "real_odds.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            _REAL_ODDS_CACHE = json.load(f)
        return _REAL_ODDS_CACHE
    return {}


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

    # 5b. 社交媒体特征
    social_data = load_social_data()
    media_data = load_media_data()
    home_social = get_social_features(home, social_data, media_data)
    away_social = get_social_features(away, social_data, media_data)
    for k, v in home_social.items():
        features[f"home_{k}"] = v
    for k, v in away_social.items():
        features[f"away_{k}"] = v
    features["heat_diff"] = home_social["heat_score"] - away_social["heat_score"]
    features["sentiment_diff"] = home_social["sentiment_ratio"] - away_social["sentiment_ratio"]
    features["media_sentiment_diff"] = home_social["media_sentiment"] - away_social["media_sentiment"]

    # 5c. 关键球员状态特征
    player_data = load_player_status()
    home_player = get_player_features(home, player_data)
    away_player = get_player_features(away, player_data)
    for k, v in home_player.items():
        features[f"home_{k}"] = v
    for k, v in away_player.items():
        features[f"away_{k}"] = v
    features["injury_diff"] = home_player["injury_severity"] - away_player["injury_severity"]
    features["star_diff"] = home_player["star_player_available"] - away_player["star_player_available"]

    # 5d-bis. 英超球员密度特征
    epl_data = load_epl_data()
    home_epl = get_team_epl_ratio(home, epl_data)
    away_epl = get_team_epl_ratio(away, epl_data)
    features["home_epl_ratio"] = home_epl
    features["away_epl_ratio"] = away_epl
    features["epl_diff"] = home_epl - away_epl
    features["epl_relative"] = (home_epl - away_epl) / (home_epl + away_epl + 0.01)

    # 5d-ter. 主帅特征
    coach_data = load_coach_data()
    home_coach = get_coach_features(home, coach_data)
    away_coach = get_coach_features(away, coach_data)
    for k, v in home_coach.items():
        features[f"home_{k}"] = v
    for k, v in away_coach.items():
        features[f"away_{k}"] = v
    features["coach_reputation_diff"] = home_coach["coach_reputation"] - away_coach["coach_reputation"]
    features["coach_tactical_diff"] = home_coach["coach_tactical"] - away_coach["coach_tactical"]
    features["coach_big_match_diff"] = home_coach["coach_big_match"] - away_coach["coach_big_match"]

    # 5d-quater. 球员数据库特征（身价/头号/阵容深度）
    player_db = load_player_database()
    home_pdb = get_player_db_features(home, player_db)
    away_pdb = get_player_db_features(away, player_db)
    for k, v in home_pdb.items():
        features[f"home_{k}"] = v
    for k, v in away_pdb.items():
        features[f"away_{k}"] = v
    features["top_player_diff"] = home_pdb["top_player_rating"] - away_pdb["top_player_rating"]
    features["top3_rating_diff"] = home_pdb["top3_avg_rating"] - away_pdb["top3_avg_rating"]
    features["squad_value_diff"] = home_pdb["squad_value"] - away_pdb["squad_value"]
    features["squad_value_log_diff"] = np.log10(home_pdb["squad_value"] + 1) - np.log10(away_pdb["squad_value"] + 1)
    features["attack_strength_diff"] = home_pdb["attack_strength"] - away_pdb["attack_strength"]
    features["defense_strength_diff"] = home_pdb["defense_strength"] - away_pdb["defense_strength"]
    features["midfield_strength_diff"] = home_pdb["midfield_strength"] - away_pdb["midfield_strength"]
    features["key_form_diff"] = home_pdb["key_player_form"] - away_pdb["key_player_form"]
    features["avg_age_diff"] = home_pdb["avg_age"] - away_pdb["avg_age"]
    features["avg_rating_diff"] = home_pdb["avg_rating"] - away_pdb["avg_rating"]

    # 5d-quinquies. 关键比赛因子（基于实际比赛分析得出）
    rank_diff = features.get("rank_diff", 0)
    coach_rep_diff = features["coach_reputation_diff"]
    coach_big_diff = features["coach_big_match_diff"]

    # 爆冷预警：强队+弱教练 vs 弱队+强教练
    # 当主队排名占优但教练声誉反向时，倾向平局/客胜
    rank_advantage = 1 if rank_diff > 30 else 0
    coach_under_signal = -1 if coach_rep_diff < 0 else 1
    features["upset_warning"] = rank_advantage * coach_under_signal if rank_advantage else 0
    features["upset_warning_x_value"] = features["upset_warning"] * features["squad_value_log_diff"]

    # 压制力指数：身价差 × 教练优势（综合指标）
    features["dominance_index"] = features["squad_value_log_diff"] * coach_rep_diff

    # 主帅大赛能力交互：教练大赛差 × 排名差
    features["big_match_pressure"] = coach_big_diff * (rank_diff / 100.0)

    # 头号状态差 × 排名差（爆冷预测：状态差距+排名接近）
    features["form_pressure"] = features["key_form_diff"] * (rank_diff / 200.0)

    # 5d-sexties. 球员联赛熟悉度特征（后卫vs前锋交锋经验）
    familiarity = compute_familiarity_features(home, away, player_db)
    for k, v in familiarity.items():
        features[k] = v

    # 5e. 真实赔率特征（与模型隐含赔率对比）
    real_odds = load_real_odds()
    match_key = f"{home}_vs_{away}"
    match_key_rev = f"{away}_vs_{home}"
    odds_entry = real_odds.get(match_key, real_odds.get(match_key_rev, {}))
    if odds_entry:
        for k in ["home_odds", "draw_odds", "away_odds"]:
            val = odds_entry.get(k, 0)
            if val > 0:
                features[f"real_{k}"] = val
                features[f"real_{k.replace('_odds', '')}_implied"] = 1.0 / val
            else:
                features[f"real_{k}"] = 2.0
                features[f"real_{k.replace('_odds', '')}_implied"] = 0.33
        # 计算真实赔率与模型赔率的偏差
        for suffix in ["home", "draw", "away"]:
            model_key = f"{suffix}_implied"
            real_key = f"real_{suffix}_implied"
            if model_key in features and real_key in features:
                features[f"odds_value_{suffix}"] = features[real_key] - features[model_key]
    else:
        features["real_home_odds"] = 2.0
        features["real_draw_odds"] = 2.0
        features["real_away_odds"] = 2.0
        features["real_home_implied"] = 0.33
        features["real_draw_implied"] = 0.33
        features["real_away_implied"] = 0.33

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
