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

_ODDS_HISTORY_CACHE = None


def _load_odds_history():
    global _ODDS_HISTORY_CACHE
    if _ODDS_HISTORY_CACHE is not None:
        return _ODDS_HISTORY_CACHE
    path = os.path.join(DATA_DIR, "odds_history.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            _ODDS_HISTORY_CACHE = json.load(f)
        return _ODDS_HISTORY_CACHE
    return None


def _set_odds_movement_defaults(features):
    features["odds_h_volatility"] = 0.0
    features["odds_d_volatility"] = 0.0
    features["odds_a_volatility"] = 0.0
    features["odds_h_trend"] = 0.0
    features["odds_d_trend"] = 0.0
    features["odds_a_trend"] = 0.0
    features["odds_max_swing"] = 0.0
    features["odds_momentum"] = 0.0


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


# ─── 动量特征辅助函数 ──────────────────────────────────

def _get_team_matches_sorted(all_matches, team, match_date=None):
    """获取球队已完赛比赛（按日期倒序），返回 [{gf, ga, date}]"""
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
    ]
    if "Date" in team_matches.columns:
        team_matches = team_matches.sort_values("Date", ascending=False)
    results = []
    for _, row in team_matches.iterrows():
        sh, sa = int(row.get("HomeGoals", -1)), int(row.get("AwayGoals", -1))
        if sh < 0 or sa < 0:
            continue
        if row["Home"] == team:
            results.append({"gf": sh, "ga": sa, "date": row.get("Date", "")})
        else:
            results.append({"gf": sa, "ga": sh, "date": row.get("Date", "")})
    return results


def _unbeaten_streak(matches):
    """计算连续不败场次"""
    streak = 0
    for m in matches:
        if m["gf"] >= m["ga"]:
            streak += 1
        else:
            break
    return streak


def _win_streak(matches):
    """计算连续胜利场次"""
    streak = 0
    for m in matches:
        if m["gf"] > m["ga"]:
            streak += 1
        else:
            break
    return streak


def _avg_opponent_rank(all_matches, team, match_date, rankings_df):
    """最近2个对手的平均FIFA排名分"""
    matches = _get_team_matches_sorted(all_matches, team, match_date)[:2]
    if not matches:
        return 1500
    total_rank = 0
    n = 0
    for m in matches:
        opp = _get_opponent_from_match(all_matches, team, m["date"])
        if opp:
            total_rank += get_team_rank_points(opp, rankings_df)
            n += 1
    return total_rank / n if n > 0 else 1500


def _get_opponent_from_match(all_matches, team, match_date):
    """从比赛记录中获取对手"""
    try:
        date_str = str(match_date)[:10] if match_date else ""
        for _, row in all_matches.iterrows():
            rdate = str(row.get("Date", ""))[:10]
            if rdate == date_str or (match_date and abs((pd.to_datetime(match_date) - pd.to_datetime(rdate)).days) < 2):
                if row.get("Home") == team:
                    return row.get("Away")
                elif row.get("Away") == team:
                    return row.get("Home")
    except:
        pass
    return None


def _rest_days(all_matches, team, match_date):
    """距离上一场比赛的天数"""
    if match_date is None:
        return 4
    try:
        current = pd.to_datetime(match_date)
        prev = _get_team_matches_sorted(all_matches, team, match_date)
        if len(prev) > 0 and prev[0].get("date"):
            prev_date = pd.to_datetime(prev[0]["date"])
            return (current - prev_date).days
    except:
        pass
    return 4  # default


# ─── 小组积分榜 & 淘汰赛落位 ─────────────────────────────

# 2026世界杯16强配对分组（相邻两组进入同一半区）
# A/B配对 → A1vsB2/B1vsA2胜者 在16强相遇
# C/D、E/F、G/H、I/J、K/L 同理
_GROUP_PAIRINGS = {
    "GROUP_A": "GROUP_B", "GROUP_B": "GROUP_A",
    "GROUP_C": "GROUP_D", "GROUP_D": "GROUP_C",
    "GROUP_E": "GROUP_F", "GROUP_F": "GROUP_E",
    "GROUP_G": "GROUP_H", "GROUP_H": "GROUP_G",
    "GROUP_I": "GROUP_J", "GROUP_J": "GROUP_I",
    "GROUP_K": "GROUP_L", "GROUP_L": "GROUP_K",
}


def _get_group_info(all_matches):
    """从已完赛比赛中计算各队当前小组积分榜"""
    standings = {}
    for _, row in all_matches.iterrows():
        group = row.get("group")
        if pd.isna(group):
            continue
        h, a = row["Home"], row["Away"]
        hg = row.get("HomeGoals")
        ag = row.get("AwayGoals")
        if pd.isna(hg) or hg is None or float(hg) < 0:
            continue
        hg, ag = int(float(hg)), int(float(ag))
        for t in [h, a]:
            if t not in standings:
                standings[t] = {"pts": 0, "gf": 0, "ga": 0, "gd": 0, "group": group, "played": 0}
        standings[h]["gf"] += hg; standings[h]["ga"] += ag
        standings[a]["gf"] += ag; standings[a]["ga"] += hg
        if hg > ag:
            standings[h]["pts"] += 3
        elif hg == ag:
            standings[h]["pts"] += 1; standings[a]["pts"] += 1
        else:
            standings[a]["pts"] += 3
        standings[h]["played"] += 1; standings[a]["played"] += 1

    # 计算各组内排名
    group_positions = {}
    for t, info in standings.items():
        g = info["group"]
        if g not in group_positions:
            group_positions[g] = []
        group_positions[g].append((t, info["pts"], info["gd"], info["gf"]))

    for g, teams in group_positions.items():
        teams.sort(key=lambda x: (-x[1], -x[2], -x[3]))
        for pos, (t, _, _, _) in enumerate(teams, 1):
            standings[t]["pos"] = pos

    for t in standings:
        standings[t]["gd"] = standings[t]["gf"] - standings[t]["ga"]

    return standings


def _get_team_group(team, all_matches):
    """查找球队所在小组"""
    for _, row in all_matches.iterrows():
        group = row.get("group")
        if pd.isna(group):
            continue
        if row["Home"] == team or row["Away"] == team:
            return group
    return None


def _get_strongest_team_in_group(group, standings, rankings_df):
    """找到指定小组中排位最靠前的球队"""
    group_teams = {t for t, info in standings.items() if info.get("group") == group}
    if not group_teams:
        return None
    return max(group_teams, key=lambda t: get_team_rank_points(t, rankings_df))


def _positioning_incentive(match_row, team, standings, all_wc_matches):
    """
    计算球队在小组赛最后一轮的淘汰赛落位压力
    返回 0-3 分数，越高表示落位考虑越重要
    """
    group = _get_team_group(team, all_wc_matches)
    if group is None:
        return 0

    info = standings.get(team)
    if info is None:
        return 0

    pts = info["pts"]
    pos = info["pos"]
    played = info.get("played", 0)

    # 只对已经踢了2场的球队计算（小组赛第二轮之后/第三轮之前）
    if played < 2:
        return 0

    # 基本压力：出线压力（还需要积分）
    # 12组每组4队，前2名出线 + 8个最好小组第3
    if pos <= 2:
        qual_pressure = 0.5  # 在出线区，但未锁定
    elif pos == 3:
        qual_pressure = 2.0  # 小组第3，需要抢分
    else:
        qual_pressure = 2.5  # 小组第4，形势危急

    # 小组头名之争：第1和第2积分接近
    paired = _GROUP_PAIRINGS.get(group)
    if paired and pos == 1:
        # 小组第1在32强碰上对面小组第2，16强碰对面第1
        # 如果对面小组第1非常强，小组第1的落位有一定压力
        qual_pressure += 0.3  # 锁定头名的额外动力
    elif paired and pos == 2:
        # 小组第2可能在32强碰上对面小组第1（更强），也有压力
        qual_pressure += 0.5  # 争取头名的动力

    return round(min(3.0, qual_pressure), 2)


def compute_positioning_note(home_team, away_team, match_row, all_wc_matches, standings=None):
    """
    生成小组赛末轮的淘汰赛落位分析文字说明
    返回 dict: {home_note, away_note, positioning_battle}
    """
    if standings is None:
        standings = _get_group_info(all_wc_matches)
    matchday = match_row.get("比赛日", 0)
    if matchday != 3:
        return {"home_note": "", "away_note": "", "positioning_battle": ""}

    home_info = standings.get(home_team)
    away_info = standings.get(away_team)
    if not home_info or not away_info:
        return {"home_note": "", "away_note": "", "positioning_battle": ""}

    home_pts, home_pos = home_info["pts"], home_info["pos"]
    away_pts, away_pos = away_info["pts"], away_info["pos"]
    home_gd, away_gd = home_info["gd"], away_info["gd"]
    group = _get_team_group(home_team, all_wc_matches)

    notes = {}
    for team, pts, pos, gd_ in [
        (home_team, home_pts, home_pos, home_gd),
        (away_team, away_pts, away_pos, away_gd),
    ]:
        note_parts = []
        # 判定是否已出局（0分 + 垫底 → 数学上已无法出线）
        is_eliminated = (pts == 0 and pos >= 3) or (pts == 0 and pos == 4)
        if is_eliminated:
            note_parts.append(f"暂列第{pos}({pts}分)，已提前出局")
        else:
            # 出线形势
            if pos == 1:
                if pts >= 6:
                    note_parts.append("已基本锁定出线")
                else:
                    note_parts.append(f"暂列第1({pts}分)，争头名")
            elif pos == 2:
                if pts >= 4:
                    note_parts.append(f"暂列第2({pts}分)，保平争胜")
                else:
                    note_parts.append(f"暂列第2({pts}分)，出线不稳")
            elif pos == 3:
                note_parts.append(f"暂列第3({pts}分)，背水一战")
            else:
                note_parts.append(f"暂列第4({pts}分)，形势危急")

        # 淘汰赛落位
        if group and pos <= 2:
            paired = _GROUP_PAIRINGS.get(group)
            if paired:
                note_parts.append(f"若出线将进入{paired.replace('GROUP_', '')}组半区")

        notes[team] = "；".join(note_parts)

    # 对决性质（需要已出局判定）
    h_elim = (home_pts == 0 and home_pos >= 3)
    a_elim = (away_pts == 0 and away_pos >= 3)
    battle = ""
    if h_elim and a_elim:
        battle = "荣誉之战，双方均已提前出局"
    elif h_elim:
        battle = f"{home_team}已提前出局，{away_team}全力争胜"
    elif a_elim:
        battle = f"{away_team}已提前出局，{home_team}全力争胜"
    elif home_pos <= 2 and away_pos <= 2:
        battle = "出线关键战，双方都输不起"
    elif home_pos <= 2 and away_pos > 2:
        battle = f"{home_team}力争晋级，{away_team}为荣誉而战"
    elif home_pos > 2 and away_pos <= 2:
        battle = f"{away_team}力争晋级，{home_team}为荣誉而战"
    else:
        battle = "荣誉之战，双方都需一场胜利"

    return {
        "home_note": notes.get(home_team, ""),
        "away_note": notes.get(away_team, ""),
        "positioning_battle": battle,
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

    # 5e2. 赔率时间序列变化特征（从 odds_history.json 提取）
    odds_hist = _load_odds_history()
    if odds_hist is not None and match_key in odds_hist:
        movement = odds_hist[match_key].get("movement", [])
        if len(movement) >= 2:
            odds_h_vals = [m.get("odds_H", 2.0) for m in movement]
            odds_d_vals = [m.get("odds_D", 3.0) for m in movement]
            odds_a_vals = [m.get("odds_A", 3.0) for m in movement]
            probs_h = [m.get("prob_H", 0.33) for m in movement]

            features["odds_h_volatility"] = float(np.std(odds_h_vals))
            features["odds_d_volatility"] = float(np.std(odds_d_vals))
            features["odds_a_volatility"] = float(np.std(odds_a_vals))
            features["odds_h_trend"] = odds_h_vals[-1] - odds_h_vals[0]
            features["odds_d_trend"] = odds_d_vals[-1] - odds_d_vals[0]
            features["odds_a_trend"] = odds_a_vals[-1] - odds_a_vals[0]
            swings = [abs(odds_h_vals[i+1] - odds_h_vals[i]) for i in range(len(odds_h_vals)-1)]
            features["odds_max_swing"] = max(swings) if swings else 0.0
            if len(probs_h) >= 3:
                features["odds_momentum"] = probs_h[-1] - probs_h[-4]
            else:
                features["odds_momentum"] = probs_h[-1] - probs_h[0]
        else:
            _set_odds_movement_defaults(features)
    else:
        _set_odds_movement_defaults(features)

    # 5f. 动量 + 赛前动态特征
    h_form = compute_recent_form(all_wc_matches, home, match_date)
    a_form = compute_recent_form(all_wc_matches, away, match_date)
    h_matches = _get_team_matches_sorted(all_wc_matches, home, match_date)
    a_matches = _get_team_matches_sorted(all_wc_matches, away, match_date)

    # 5f1. 最近一场结果与进球
    if len(h_matches) > 0:
        last = h_matches[0]
        features["home_last_gf"] = last["gf"]
        features["home_last_ga"] = last["ga"]
        features["home_last_result"] = 1 if last["gf"] > last["ga"] else (0 if last["gf"] == last["ga"] else -1)
    else:
        features["home_last_gf"] = 0
        features["home_last_ga"] = 0
        features["home_last_result"] = 0

    if len(a_matches) > 0:
        last = a_matches[0]
        features["away_last_gf"] = last["gf"]
        features["away_last_ga"] = last["ga"]
        features["away_last_result"] = 1 if last["gf"] > last["ga"] else (0 if last["gf"] == last["ga"] else -1)
    else:
        features["away_last_gf"] = 0
        features["away_last_ga"] = 0
        features["away_last_result"] = 0

    # 5f2. 连续不败/连胜场次
    features["home_unbeaten_streak"] = _unbeaten_streak(h_matches)
    features["away_unbeaten_streak"] = _unbeaten_streak(a_matches)
    features["home_win_streak"] = _win_streak(h_matches)
    features["away_win_streak"] = _win_streak(a_matches)

    # 5f3. 对手强度调整（最近对手的平均FIFA排名分）
    features["home_opponent_strength"] = _avg_opponent_rank(all_wc_matches, home, match_date, rankings_df)
    features["away_opponent_strength"] = _avg_opponent_rank(all_wc_matches, away, match_date, rankings_df)
    features["opponent_strength_diff"] = features["home_opponent_strength"] - features["away_opponent_strength"]

    # 5f4. 休息天数（距离上一场比赛）
    features["home_rest_days"] = _rest_days(all_wc_matches, home, match_date)
    features["away_rest_days"] = _rest_days(all_wc_matches, away, match_date)
    features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]

    # 5f5. 近期总进球趋势（近2场总进球和）
    h_total_gf = sum(m["gf"] for m in h_matches[:2])
    a_total_gf = sum(m["gf"] for m in a_matches[:2])
    h_total_ga = sum(m["ga"] for m in h_matches[:2])
    a_total_ga = sum(m["ga"] for m in a_matches[:2])
    features["home_recent_2gf"] = h_total_gf
    features["away_recent_2gf"] = a_total_gf
    features["home_recent_2ga"] = h_total_ga
    features["away_recent_2ga"] = a_total_ga
    features["recent_gf_diff"] = h_total_gf - a_total_gf
    features["recent_ga_diff"] = h_total_ga - a_total_ga

    # 5f7. 淘汰赛落位特征（小组赛最后一轮专用）
    # 计算当前小组积分榜
    group_info = _get_group_info(all_wc_matches)
    h_group_info = group_info.get(home, {})
    a_group_info = group_info.get(away, {})
    features["home_group_pts"] = h_group_info.get("pts", 0)
    features["away_group_pts"] = a_group_info.get("pts", 0)
    features["home_group_gd"] = h_group_info.get("gd", 0)
    features["away_group_gd"] = a_group_info.get("gd", 0)
    features["home_group_pos"] = h_group_info.get("pos", 3)
    features["away_group_pos"] = a_group_info.get("pos", 3)
    features["home_group_gf"] = h_group_info.get("gf", 0)
    features["away_group_gf"] = a_group_info.get("gf", 0)

    # 判断是否为小组赛最后一轮
    matchday = match_row.get("比赛日", 0)
    features["is_final_group_match"] = 1 if matchday == 3 else 0

    # 计算淘汰赛落位压力（小组第1vs第2在16强对手差异）
    # 2026年扩军48队/12组，16强配对应考虑相邻组落位
    h_incentive = _positioning_incentive(match_row, home, group_info, all_wc_matches)
    a_incentive = _positioning_incentive(match_row, away, group_info, all_wc_matches)
    features["home_positioning_incentive"] = h_incentive
    features["away_positioning_incentive"] = a_incentive
    features["positioning_incentive_diff"] = h_incentive - a_incentive

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
