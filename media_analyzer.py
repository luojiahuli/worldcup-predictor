"""
media_analyzer.py - 媒体情感分析（基于实际比赛表现）
分析各参赛球队的新闻热度和情感倾向，使用实际比赛数据而非随机模拟
"""
import pandas as pd
import numpy as np
import os, logging, pickle

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def _load_feature_matrix():
    """加载特征矩阵获取比赛数据"""
    path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def _load_rankings():
    """加载FIFA排名"""
    path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def _team_recent_performance(team_name, feature_df):
    """计算球队最近2场表现指标"""
    finished = feature_df[feature_df["is_finished"] == True]
    as_home = finished[finished["home_team"] == team_name]
    as_away = finished[finished["away_team"] == team_name]

    matches = []
    for _, r in as_home.iterrows():
        matches.append({
            "gf": r["home_goals"], "ga": r["away_goals"],
            "result": "W" if r["home_goals"] > r["away_goals"] else ("D" if r["home_goals"] == r["away_goals"] else "L"),
            "opponent": r["away_team"],
        })
    for _, r in as_away.iterrows():
        matches.append({
            "gf": r["away_goals"], "ga": r["home_goals"],
            "result": "W" if r["away_goals"] > r["home_goals"] else ("D" if r["away_goals"] == r["home_goals"] else "L"),
            "opponent": r["home_team"],
        })
    # Sort by date if available
    matches = matches[-3:] if len(matches) > 3 else matches  # last 3

    if not matches:
        return {"avg_gf": 0, "avg_ga": 0, "win_rate": 0, "n": 0}

    return {
        "avg_gf": sum(m["gf"] for m in matches) / len(matches),
        "avg_ga": sum(m["ga"] for m in matches) / len(matches),
        "win_rate": sum(1 for m in matches if m["result"] == "W") / len(matches),
        "n": len(matches),
    }


def analyze_news_sentiment(team_name):
    """基于实际比赛表现计算球队媒体情感"""
    feature_df = _load_feature_matrix()
    rankings_df = _load_rankings()

    # 默认值
    base_sentiment = 0.50
    base_volume = 15
    positive_ratio = 0.55

    # 基于排名的基线
    if rankings_df is not None:
        rank_row = rankings_df[rankings_df["Team"] == team_name]
        if len(rank_row) > 0:
            rank_pos = rank_row.index[0] + 1
            rank_score = max(0, 1 - rank_pos / 60)  # 0-1 归一化
            base_sentiment = 0.45 + rank_score * 0.30
            base_volume = max(8, int(10 + rank_score * 30))

    # 基于实际比赛表现调整
    if feature_df is not None:
        perf = _team_recent_performance(team_name, feature_df)
        if perf["n"] > 0:
            # 赢球提升情感，进球多提升热度
            perf_boost = perf["win_rate"] * 0.20 - (perf["avg_ga"] - perf["avg_gf"]) * 0.03
            base_sentiment = min(0.92, max(0.15, base_sentiment + perf_boost))
            # 进球多→新闻多，失球多→新闻也多（负面新闻）
            volume_boost = perf["avg_gf"] * 2 + perf["avg_ga"] * 1.5
            base_volume = max(5, min(50, int(base_volume + volume_boost)))
            # 正负面比例取决于近期战绩
            positive_ratio = min(0.85, max(0.20, 0.50 + perf["win_rate"] * 0.30 - perf["avg_ga"] * 0.02))

    hotness = min(1.0, base_volume / 40)

    return {
        "team": team_name,
        "sentiment_score": round(base_sentiment, 3),
        "news_volume": base_volume,
        "positive_ratio": round(positive_ratio, 3),
        "negative_ratio": round(max(0.05, min(0.50, 1.0 - positive_ratio - 0.10)), 3),
        "trending": "up" if base_sentiment > 0.55 else ("down" if base_sentiment < 0.40 else "stable"),
        "hotness": round(hotness, 3),
        "sample_count": base_volume,
    }


def analyze_all_teams(teams_list):
    """分析所有球队的媒体情感"""
    log.info("=" * 50)
    log.info("媒体情感分析（基于实际表现）")
    log.info("=" * 50)

    results = []
    for team in teams_list:
        result = analyze_news_sentiment(team)
        results.append(result)

    df = pd.DataFrame(results)
    path = os.path.join(DATA_DIR, "media_sentiment.csv")
    df.to_csv(path, index=False)

    avg_sentiment = df["sentiment_score"].mean()
    log.info(f"分析完成: {len(results)} 支球队")
    log.info(f"平均情感得分: {avg_sentiment:.3f}")

    top_positive = df.nlargest(5, "sentiment_score")
    log.info("\n最正面媒体形象 Top5:")
    for _, row in top_positive.iterrows():
        log.info(f"  {row['team']}: {row['sentiment_score']:.3f} (热度:{row['hotness']:.2f})")

    top_negative = df.nsmallest(5, "sentiment_score")
    log.info("\n最负面媒体形象 Top5:")
    for _, row in top_negative.iterrows():
        log.info(f"  {row['team']}: {row['sentiment_score']:.3f}")

    return df


def load_media_data():
    path = os.path.join(DATA_DIR, "media_sentiment.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


if __name__ == "__main__":
    teams_df = pd.read_csv(os.path.join(DATA_DIR, "worldcup_teams.csv"))
    teams = teams_df["Team"].tolist()
    analyze_all_teams(teams)
