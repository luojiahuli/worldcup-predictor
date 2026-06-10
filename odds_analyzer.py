"""
odds_analyzer.py - 赔率分析
分析各场比赛的赔率变化和隐含概率
"""
import pandas as pd
import numpy as np
import os, json, logging, pickle
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_wc_2026_matches():
    path = os.path.join(DATA_DIR, "wc_2026.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def compute_match_odds(match_row):
    """为比赛计算赔率和隐含概率（基于DeepSeek预测概率）"""
    # 尝试从数据中读取真实赔率
    for col in ["odds_home", "odds_draw", "odds_away"]:
        val = match_row.get(col, None)
        if pd.notna(val) and val is not None and val > 0:
            return {
                "odds_H": float(val),
                "odds_D": float(match_row.get("odds_draw", 2.5)),
                "odds_A": float(match_row.get("odds_away", 2.5)),
                "source": "football-data.org",
            }

    # 基于排名估算基准赔率
    home = match_row.get("Home", "")
    away = match_row.get("Away", "")

    rankings_df = None
    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    if os.path.exists(rank_path):
        rankings_df = pd.read_csv(rank_path)

    home_pts, away_pts = 1500, 1500
    if rankings_df is not None:
        h = rankings_df[rankings_df["Team"] == home]
        a = rankings_df[rankings_df["Team"] == away]
        if len(h) > 0: home_pts = int(h["RankPoints"].values[0])
        if len(a) > 0: away_pts = int(a["RankPoints"].values[0])

    # 用ELO公式估算胜率
    expected_h = 1.0 / (1.0 + 10 ** ((away_pts - home_pts) / 400.0))
    expected_a = 1.0 / (1.0 + 10 ** ((home_pts - away_pts) / 400.0))
    expected_d = 1.0 - expected_h - expected_a

    # 调整：确保平局概率在20-35%之间
    expected_d = max(0.20, min(0.35, expected_d))
    remaining = 1.0 - expected_d
    expected_h = expected_h / (expected_h + expected_a) * remaining
    expected_a = expected_a / (expected_h + expected_a) * remaining

    # 加庄家抽水(约6%)
    margin = 1.06
    odds_h = 1.0 / expected_h * margin
    odds_d = 1.0 / expected_d * margin
    odds_a = 1.0 / expected_a * margin

    return {
        "odds_H": round(odds_h, 2),
        "odds_D": round(odds_d, 2),
        "odds_A": round(odds_a, 2),
        "source": "estimated (FIFA ranking based)",
        "implied_H": round(expected_h, 3),
        "implied_D": round(expected_d, 3),
        "implied_A": round(expected_a, 3),
    }


def simulate_odds_movement(odds_H, odds_D, odds_A, days=14):
    """模拟赔率在比赛前的变化趋势"""
    n = min(days, 30)
    base_H = 1.0 / odds_H
    base_D = 1.0 / odds_D
    base_A = 1.0 / odds_A

    np.random.seed(42)
    history = []
    for i in range(n):
        noise_h = np.random.normal(0, 0.005)
        noise_d = np.random.normal(0, 0.003)
        noise_a = np.random.normal(0, 0.005)

        # 比赛临近，赔率逐渐收敛
        proximity = i / n
        drift = proximity * 0.01

        prob_h = max(0.05, base_H + noise_h + drift * (0.5 - base_H))
        prob_d = max(0.05, base_D + noise_d)
        prob_a = max(0.05, base_A + noise_a - drift * (0.5 - base_H))

        total = prob_h + prob_d + prob_a
        prob_h /= total
        prob_d /= total
        prob_a /= total

        # 加抽水
        margin = 1.06
        history.append({
            "day": i,
            "odds_H": round(1.0 / prob_h * margin, 2),
            "odds_D": round(1.0 / prob_d * margin, 2),
            "odds_A": round(1.0 / prob_a * margin, 2),
            "prob_H": round(prob_h, 3),
            "prob_D": round(prob_d, 3),
            "prob_A": round(prob_a, 3),
        })

    return history


def analyze_all_odds():
    """分析所有世界杯比赛的赔率"""
    log.info("=" * 50)
    log.info("赔率分析")
    log.info("=" * 50)

    matches = load_wc_2026_matches()
    if matches is None:
        log.error("无2026世界杯数据")
        return None

    results = []
    odds_history = {}

    for _, row in matches.iterrows():
        home = row["Home"]
        away = row["Away"]
        if pd.isna(home) or pd.isna(away) or not str(home).strip() or not str(away).strip():
            continue
        match_key = f"{home}_vs_{away}"

        odds = compute_match_odds(row)
        odds["home"] = home
        odds["away"] = away
        odds["date"] = str(row.get("Date", ""))[:10]
        results.append(odds)

        movement = simulate_odds_movement(odds["odds_H"], odds["odds_D"], odds["odds_A"])
        odds_history[match_key] = {
            "home": home,
            "away": away,
            "movement": movement,
            "current": movement[-1] if movement else None,
        }

    df = pd.DataFrame(results)
    df["odds_spread"] = abs(df["odds_H"] - df["odds_A"])
    df.to_csv(os.path.join(DATA_DIR, "odds_analysis.csv"), index=False)

    with open(os.path.join(DATA_DIR, "odds_history.json"), "w") as f:
        json.dump(odds_history, f, ensure_ascii=False, indent=2)

    log.info(f"分析完成: {len(results)} 场比赛")

    # 最大赔率差异比赛
    df["odds_spread"] = abs(df["odds_H"] - df["odds_A"])
    biggest_mismatch = df.nlargest(5, "odds_spread")
    log.info("\n实力最悬殊比赛（预计赔率）:")
    for _, row in biggest_mismatch.iterrows():
        log.info(f"  {row['home']} vs {row['away']}: {row['odds_H']:.2f} - {row['odds_D']:.2f} - {row['odds_A']:.2f}")

    closest = df.nsmallest(5, "odds_spread")
    log.info("\n实力最接近比赛:")
    for _, row in closest.iterrows():
        log.info(f"  {row['home']} vs {row['away']}: {row['odds_H']:.2f} - {row['odds_D']:.2f} - {row['odds_A']:.2f}")

    return df


def load_odds_data():
    path = os.path.join(DATA_DIR, "odds_analysis.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_odds_history():
    path = os.path.join(DATA_DIR, "odds_history.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


if __name__ == "__main__":
    df = analyze_all_odds()
    if df is not None:
        print(f"\n赔率分析完成: {len(df)} 场比赛")
