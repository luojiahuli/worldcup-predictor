"""
match_review.py - 已完赛比赛全面复盘
分析所有已完赛比赛的预测准确性、关键因子、爆冷原因
"""
import os, json, pickle, logging
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

UPSET_CONFIDENCE_THRESHOLD = 0.35


def load_all():
    """加载所有相关数据"""
    pred_path = os.path.join(DATA_DIR, "wc_predictions.pkl")
    df = pickle.load(open(pred_path, "rb")) if os.path.exists(pred_path) else None

    finished_path = os.path.join(DATA_DIR, "finished_matches.json")
    finished = json.load(open(finished_path)) if os.path.exists(finished_path) else {}

    agent_path = os.path.join(DATA_DIR, "agent_predictions.json")
    agent_data = json.load(open(agent_path)) if os.path.exists(agent_path) else {}

    weights_path = os.path.join(DATA_DIR, "agent_weights.json")
    weights_data = json.load(open(weights_path)) if os.path.exists(weights_path) else {}

    top5_path = os.path.join(DATA_DIR, "team_top5.json")
    top5_data = json.load(open(top5_path)) if os.path.exists(top5_path) else {}

    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    rankings = pd.read_csv(rank_path) if os.path.exists(rank_path) else None

    return {
        "df": df, "finished": finished,
        "agent_data": agent_data, "weights_data": weights_data,
        "top5_data": top5_data, "rankings": rankings,
    }


def get_finished_matches(data):
    """提取所有已完赛比赛"""
    df = data["df"]
    if df is None:
        return pd.DataFrame()
    return df[df["is_finished"] == True].copy()


def analyze_predictions(f_df):
    """分析预测准确率"""
    if len(f_df) == 0:
        return {}

    n = len(f_df)
    result_correct = int((f_df["actual_result"] == f_df["pred_result"]).sum())
    score_correct = int(((f_df.get("pred_home_score", 0) == f_df["actual_home_score"]) &
                         (f_df.get("pred_away_score", 0) == f_df["actual_away_score"])).sum())

    predictions = []
    for _, r in f_df.iterrows():
        predictions.append({
            "date": str(r["date"])[:10],
            "home": r["home_team"], "away": r["away_team"],
            "pred_result": r["pred_result"], "actual_result": r["actual_result"],
            "pred_home": int(r.get("pred_home_score", 0)),
            "pred_away": int(r.get("pred_away_score", 0)),
            "actual_home": int(r["actual_home_score"]),
            "actual_away": int(r["actual_away_score"]),
            "result_correct": bool(r["actual_result"] == r["pred_result"]),
            "score_correct": bool(
                r.get("pred_home_score", 0) == r["actual_home_score"]
                and r.get("pred_away_score", 0) == r["actual_away_score"]
            ),
            "confidence": float(r.get("confidence", 0)),
        })

    # 按轮次统计
    from apply_finished import ROUND_DATES
    round_stats = {}
    for p in predictions:
        d = p["date"]
        rd = "其他"
        for rn, (rs, re) in ROUND_DATES.items():
            if rs <= d <= re:
                rd = rn
                break
        if rd not in round_stats:
            round_stats[rd] = {"total": 0, "correct": 0, "matches": []}
        round_stats[rd]["total"] += 1
        if p["result_correct"]:
            round_stats[rd]["correct"] += 1
        round_stats[rd]["matches"].append(p)

    for rd, rs in round_stats.items():
        rs["accuracy"] = rs["correct"] / rs["total"] if rs["total"] > 0 else 0

    return {
        "total_matches": n,
        "accuracy": result_correct / n,
        "exact_score_accuracy": score_correct / n,
        "result_correct": result_correct,
        "score_correct": int(score_correct),
        "predictions": predictions,
        "round_stats": round_stats,
    }


def get_team_rank_points(team, rankings):
    if rankings is None:
        return 1500
    row = rankings[rankings["Team"] == team]
    if len(row) > 0:
        return int(row["RankPoints"].values[0])
    name_map = {"United States": "USA", "Czechia": "Czech Republic",
                "Cape Verde Islands": "Cape Verde"}
    mapped = name_map.get(team)
    if mapped:
        row = rankings[rankings["Team"] == mapped]
        if len(row) > 0:
            return int(row["RankPoints"].values[0])
    for t in rankings["Team"]:
        if t.lower() == team.lower():
            return int(rankings[rankings["Team"] == t]["RankPoints"].values[0])
    return 1500


def analyze_key_factors(f_df, data):
    """分析关键因子"""
    rankings = data["rankings"]
    top5_data = data["top5_data"]

    factors = []
    for _, r in f_df.iterrows():
        home, away = r["home_team"], r["away_team"]
        hp = get_team_rank_points(home, rankings)
        ap = get_team_rank_points(away, rankings)
        rank_diff = hp - ap
        home_top5 = top5_data.get(home, 0.10)
        away_top5 = top5_data.get(away, 0.10)
        is_correct = r["actual_result"] == r["pred_result"]

        factors.append({
            "home": home, "away": away,
            "rank_diff": int(rank_diff),
            "abs_rank_diff": int(abs(rank_diff)),
            "home_top5": home_top5, "away_top5": away_top5,
            "top5_gap": round(home_top5 - away_top5, 3),
            "actual_result": r["actual_result"],
            "pred_result": r["pred_result"],
            "is_correct": bool(is_correct),
            "home_score": int(r["actual_home_score"]),
            "away_score": int(r["actual_away_score"]),
        })

    df_f = pd.DataFrame(factors)
    correct = df_f[df_f["is_correct"] == True]
    wrong = df_f[df_f["is_correct"] == False]

    return {
        "matches": factors,
        "summary": {
            "avg_rank_diff_correct": float(correct["abs_rank_diff"].mean()) if len(correct) > 0 else 0,
            "avg_rank_diff_wrong": float(wrong["abs_rank_diff"].mean()) if len(wrong) > 0 else 0,
            "avg_top5_gap_correct": float(correct["top5_gap"].mean()) if len(correct) > 0 else 0,
            "avg_top5_gap_wrong": float(wrong["top5_gap"].mean()) if len(wrong) > 0 else 0,
        },
    }


def identify_upsets(analysis, factor_data):
    """识别爆冷"""
    predictions = analysis.get("predictions", [])
    upsets = []
    for p in predictions:
        if not p["result_correct"]:
            f = None
            for m in factor_data.get("matches", []):
                if m["home"] == p["home"] and m["away"] == p["away"]:
                    f = m
                    break
            rank_diff = f["abs_rank_diff"] if f else 0
            top5_gap = f["top5_gap"] if f else 0
            reasons = []
            if rank_diff > 100:
                reasons.append(f"排名差距悬殊(差{rank_diff}分)")
            if top5_gap > 0.2:
                reasons.append(f"五大联赛含量优势(+{top5_gap:.0%})但未取胜")
            if p["confidence"] < UPSET_CONFIDENCE_THRESHOLD:
                reasons.append(f"模型本身低置信度({p['confidence']:.0%})")
            if not reasons:
                reasons.append("预测方向偏差")

            upsets.append({
                "home": p["home"], "away": p["away"],
                "score": f"{p['actual_home']}-{p['actual_away']}",
                "result": p["actual_result"], "predicted": p["pred_result"],
                "confidence": p["confidence"],
                "rank_diff": rank_diff, "top5_gap": top5_gap,
                "reasons": reasons,
            })

    return {
        "total_upsets": len(upsets),
        "upset_rate": len(upsets) / max(analysis.get("total_matches", 1), 1),
        "upsets": upsets,
        "common_factors": _summarize_upset_factors(upsets),
    }


def _summarize_upset_factors(upsets):
    factor_counts = {}
    for u in upsets:
        for r in u["reasons"]:
            if "排名差距" in r:
                factor = "排名差距大但未体现"
            elif "五大联赛" in r:
                factor = "五大联赛优势未转化"
            elif "置信度" in r:
                factor = "低置信度预测"
            else:
                factor = "其他预测偏差"
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    sorted_factors = sorted(factor_counts.items(), key=lambda x: -x[1])
    return [{"factor": k, "count": v} for k, v in sorted_factors]


def run_review():
    """运行全面复盘"""
    log.info("=" * 60)
    log.info("已完赛比赛全面复盘")
    log.info("=" * 60)

    data = load_all()
    f_df = get_finished_matches(data)

    if len(f_df) == 0:
        log.warning("无已完赛比赛数据")
        return None

    log.info(f"共 {len(f_df)} 场已完赛比赛")

    # 1. 预测准确性
    analysis = analyze_predictions(f_df)
    log.info(f"方向准确率: {analysis['accuracy']:.1%} ({analysis['result_correct']}/{analysis['total_matches']})")
    log.info(f"精确比分准确率: {analysis['exact_score_accuracy']:.1%}")

    # 2. 因子分析
    factor_data = analyze_key_factors(f_df, data)
    fs = factor_data.get("summary", {})
    log.info(f"正确预测平均排名差: {fs.get('avg_rank_diff_correct', 0):.0f}")
    log.info(f"错误预测平均排名差: {fs.get('avg_rank_diff_wrong', 0):.0f}")

    # 3. 爆冷
    upset_report = identify_upsets(analysis, factor_data)
    log.info(f"爆冷比赛: {upset_report['total_upsets']} 场 ({upset_report['upset_rate']:.1%})")

    # 4. 轮次统计
    round_stats = analysis.get("round_stats", {})
    log.info(f"\n各轮次准确率:")
    for rn in ["group_1", "group_2", "group_3", "r16", "quarter", "semi", "final"]:
        if rn in round_stats:
            rs = round_stats[rn]
            log.info(f"  {rn}: {rs['accuracy']:.1%} ({rs['correct']}/{rs['total']})")

    result = {
        "analysis": analysis,
        "key_factors": factor_data,
        "upsets": upset_report,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    path = os.path.join(DATA_DIR, "round1_review.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info(f"复盘报告已保存: {path}")
    return result


if __name__ == "__main__":
    run_review()
