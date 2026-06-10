"""
social_analyzer.py - 社交热度分析
分析各参赛球队在社交媒体上的热度和趋势
"""
import pandas as pd
import numpy as np
import os, logging, json, pickle

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 全球社交媒体热度基础分
TEAM_SOCIAL_BASE = {
    "Brazil": 98, "Argentina": 97, "France": 95, "England": 94,
    "Germany": 92, "Italy": 90, "Spain": 89, "Portugal": 88,
    "Netherlands": 87, "Belgium": 85, "Mexico": 84, "USA": 83,
    "Uruguay": 80, "Colombia": 79, "Japan": 78, "South Korea": 77,
    "Morocco": 75, "Senegal": 73, "Nigeria": 72, "Egypt": 70,
    "Croatia": 72, "Denmark": 70, "Switzerland": 68, "Serbia": 67,
    "Poland": 66, "Ukraine": 65, "Sweden": 64, "Norway": 63,
    "Turkey": 62, "Austria": 60, "Scotland": 59, "Wales": 58,
    "Iran": 65, "Saudi Arabia": 64, "Australia": 63, "Japan": 78,
    "South Korea": 77, "Qatar": 62, "Canada": 61, "Costa Rica": 58,
    "Panama": 55, "Ecuador": 62, "Peru": 61, "Chile": 66,
    "Paraguay": 58, "Venezuela": 56, "Cameroon": 60, "Ghana": 62,
    "Tunisia": 61, "Algeria": 60, "Ivory Coast": 63,
    "Mali": 55, "Burkina Faso": 53, "Congo DR": 54, "South Africa": 58,
    "New Zealand": 50,
}

# 话题标签
TEAM_HASHTAGS = {
    "Brazil": ["#BRA", "#Brazil", "#Selecao"],
    "Argentina": ["#ARG", "#Argentina", "#Albiceleste"],
    "France": ["#FRA", "#France", "#LesBleus"],
    "England": ["#ENG", "#England", "#ThreeLions"],
    "Germany": ["#GER", "#Germany", "#DieMannschaft"],
    "Portugal": ["#POR", "#Portugal"],
    "Spain": ["#ESP", "#Spain", "#LaRoja"],
    "Netherlands": ["#NED", "#Netherlands", "#Oranje"],
    "Italy": ["#ITA", "#Italy", "#Azzurri"],
    "USA": ["#USA", "#USMNT"],
}


def compute_social_heat(team_name):
    """计算球队社交热度"""
    base = TEAM_SOCIAL_BASE.get(team_name,
                                max(45, min(70, 60 + np.random.randint(-10, 10))))

    # 模拟社交热度变化（比赛临近升温）
    days_to_wc = max(0, (pd.Timestamp("2026-06-11") - pd.Timestamp.now()).days)
    urgency_factor = max(0, min(0.15, (30 - days_to_wc) / 200))

    # 热度波动
    volatility = np.random.normal(0, 0.03)

    # 近7天趋势（模拟）
    trend_data = []
    for i in range(7):
        day_score = base / 100 + np.random.normal(0, 0.02) + urgency_factor * (i / 7)
        trend_data.append(round(max(0.1, min(1.0, day_score)), 3))

    current_heat = trend_data[-1]
    week_ago = trend_data[0]
    change = current_heat - week_ago

    # 活跃度模拟
    active_users = int(base * np.random.uniform(1000, 10000) * (1 + urgency_factor * 2))

    return {
        "team": team_name,
        "heat_score": round(current_heat, 3),
        "heat_change": round(change, 3),
        "heat_direction": "up" if change > 0.01 else ("down" if change < -0.01 else "stable"),
        "base_popularity": base,
        "active_users_24h": active_users,
        "trend_7d": trend_data,
        "mention_count": int(active_users * np.random.uniform(0.5, 2.0)),
        "sentiment_ratio": round(max(0.3, min(0.8, 0.5 + change * 2)), 3),
    }


def analyze_social_heat(teams_list):
    """分析所有球队的社交热度"""
    log.info("=" * 50)
    log.info("社交媒体热度分析")
    log.info("=" * 50)

    results = []
    for team in teams_list:
        result = compute_social_heat(team)
        results.append(result)

    df = pd.DataFrame(results)

    # 保存趋势数据（JSON格式）
    trends = {}
    for _, row in df.iterrows():
        trends[row["team"]] = row["trend_7d"]
    with open(os.path.join(DATA_DIR, "social_trends.json"), "w") as f:
        json.dump(trends, f)

    df.to_csv(os.path.join(DATA_DIR, "social_heat.csv"), index=False)

    log.info(f"分析完成: {len(results)} 支球队")

    hot = df.nlargest(5, "heat_score")
    log.info("\n社交热度 Top5:")
    for _, row in hot.iterrows():
        direction = "↑" if row["heat_direction"] == "up" else ("↓" if row["heat_direction"] == "down" else "→")
        log.info(f"  {row['team']}: {row['heat_score']:.3f} {direction} ({row['base_popularity']}/100)")

    cold = df.nsmallest(5, "heat_score")
    log.info("\n社交热度最低 Top5:")
    for _, row in cold.iterrows():
        log.info(f"  {row['team']}: {row['heat_score']:.3f} ({row['base_popularity']}/100)")

    return df


def load_social_data():
    path = os.path.join(DATA_DIR, "social_heat.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


if __name__ == "__main__":
    teams_df = pd.read_csv(os.path.join(DATA_DIR, "worldcup_teams.csv"))
    teams = teams_df["Team"].tolist()
    analyze_social_heat(teams)
