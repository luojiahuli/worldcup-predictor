"""
round_1_review.py - 第一轮小组赛（June 11-14）全面复盘
分析关键因子、爆冷原因，为下一轮预测提供修正信号
"""
import os, json, pickle, logging
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# 第一轮日期范围
GROUP1_START = "2026-06-11"
GROUP1_END = "2026-06-14"

# 爆冷阈值：预测准确率<0.35 且实际结果与预测不符
UPSET_CONFIDENCE_THRESHOLD = 0.35


def load_all():
    """加载所有相关数据"""
    # 比赛预测
    pred_path = os.path.join(DATA_DIR, "wc_predictions.pkl")
    df = pickle.load(open(pred_path, "rb")) if os.path.exists(pred_path) else None

    # 完赛数据
    finished_path = os.path.join(DATA_DIR, "finished_matches.json")
    finished = json.load(open(finished_path)) if os.path.exists(finished_path) else {}

    # 智能体预测
    agent_path = os.path.join(DATA_DIR, "agent_predictions.json")
    agent_data = json.load(open(agent_path)) if os.path.exists(agent_path) else {}

    # 权重数据
    weights_path = os.path.join(DATA_DIR, "agent_weights.json")
    weights_data = json.load(open(weights_path)) if os.path.exists(weights_path) else {}

    # 五大联赛数据
    top5_path = os.path.join(DATA_DIR, "team_top5.json")
    top5_data = json.load(open(top5_path)) if os.path.exists(top5_path) else {}

    # 排名数据
    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    rankings = pd.read_csv(rank_path) if os.path.exists(rank_path) else None

    return {
        "df": df,
        "finished": finished,
        "agent_data": agent_data,
        "weights_data": weights_data,
        "top5_data": top5_data,
        "rankings": rankings,
    }


def get_group1_matches(data):
    """提取第一轮小组赛"""
    df = data["df"]
    if df is None:
        return pd.DataFrame()
    mask = df["date"].between(GROUP1_START, GROUP1_END)
    return df[mask].copy()


def analyze_predictions(g1_df):
    """分析预测准确率"""
    if len(g1_df) == 0:
        return {}
    n_total = len(g1_df)
    finished = g1_df[g1_df["is_finished"] == True]
    n_finished = len(finished)

    if n_finished == 0:
        return {
            "total_matches": n_total,
            "finished_matches": 0,
            "accuracy": 0,
            "exact_score_accuracy": 0,
            "result_correct": 0,
            "score_correct": 0,
            "predictions": [],
        }

    result_correct = int((finished["actual_result"] == finished["pred_result"]).sum())
    score_correct = (
        (finished["pred_home_score"] == finished["actual_home_score"])
        & (finished["pred_away_score"] == finished["actual_away_score"])
    ).sum()

    predictions = []
    for _, r in finished.iterrows():
        is_result_correct = r["actual_result"] == r["pred_result"]
        is_score_correct = (
            r.get("pred_home_score", 0) == r["actual_home_score"]
            and r.get("pred_away_score", 0) == r["actual_away_score"]
        )
        predictions.append({
            "date": str(r["date"])[:10],
            "home": r["home_team"],
            "away": r["away_team"],
            "pred_result": r["pred_result"],
            "actual_result": r["actual_result"],
            "pred_home": int(r.get("pred_home_score", 0)),
            "pred_away": int(r.get("pred_away_score", 0)),
            "actual_home": int(r["actual_home_score"]),
            "actual_away": int(r["actual_away_score"]),
            "result_correct": bool(is_result_correct),
            "score_correct": bool(is_score_correct),
            "confidence": float(r.get("confidence", 0)),
        })

    return {
        "total_matches": n_total,
        "finished_matches": n_finished,
        "accuracy": result_correct / n_finished if n_finished > 0 else 0,
        "exact_score_accuracy": score_correct / n_finished if n_finished > 0 else 0,
        "result_correct": int(result_correct),
        "score_correct": int(score_correct),
        "predictions": predictions,
    }


def get_team_rank_points(team, rankings):
    """获取球队排名积分"""
    if rankings is None:
        return 1500
    row = rankings[rankings["Team"] == team]
    if len(row) > 0:
        return int(row["RankPoints"].values[0])
    # 尝试名称映射
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


def analyze_key_factors(g1_df, data):
    """分析关键影响因子：排名差距、五大联赛密度、主客场"""
    if len(g1_df) == 0:
        return {}

    rankings = data["rankings"]
    top5_data = data["top5_data"]
    finished = g1_df[g1_df["is_finished"] == True]

    factors = []
    for _, r in finished.iterrows():
        home = r["home_team"]
        away = r["away_team"]
        hp = get_team_rank_points(home, rankings)
        ap = get_team_rank_points(away, rankings)
        rank_diff = hp - ap
        home_top5 = top5_data.get(home, 0.10)
        away_top5 = top5_data.get(away, 0.10)
        top5_gap = home_top5 - away_top5
        is_correct = r["actual_result"] == r["pred_result"]

        factors.append({
            "home": home,
            "away": away,
            "rank_diff": int(rank_diff),
            "abs_rank_diff": int(abs(rank_diff)),
            "home_top5": home_top5,
            "away_top5": away_top5,
            "top5_gap": round(top5_gap, 3),
            "actual_result": r["actual_result"],
            "pred_result": r["pred_result"],
            "is_correct": bool(is_correct),
            "home_score": int(r["actual_home_score"]),
            "away_score": int(r["actual_away_score"]),
        })

    # 因子分析
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
    """识别爆冷比赛"""
    predictions = analysis.get("predictions", [])
    upsets = []
    for p in predictions:
        # 爆冷定义：预测错误 + 模型置信度<0.35 或 排名差距大但结果相反
        if not p["result_correct"]:
            is_upset = True
            reasons = []
            # 找对应因子
            f = None
            for m in factor_data.get("matches", []):
                if m["home"] == p["home"] and m["away"] == p["away"]:
                    f = m
                    break
            rank_diff = f["abs_rank_diff"] if f else 0
            top5_gap = f["top5_gap"] if f else 0

            # 高排名差爆冷
            if rank_diff > 100:
                reasons.append(f"排名差距悬殊(差{rank_diff}分)")
            # 五大联赛优势方未赢
            if top5_gap > 0.2:
                reasons.append(f"五大联赛含量优势(+{top5_gap:.0%})但未取胜")
            if p["confidence"] < UPSET_CONFIDENCE_THRESHOLD:
                reasons.append(f"模型本身低置信度({p['confidence']:.0%})")
            if not reasons:
                reasons.append("预测方向错误")

            upsets.append({
                "home": p["home"],
                "away": p["away"],
                "score": f"{p['actual_home']}-{p['actual_away']}",
                "result": p["actual_result"],
                "predicted": p["pred_result"],
                "confidence": p["confidence"],
                "rank_diff": rank_diff,
                "top5_gap": top5_gap,
                "reasons": reasons,
            })

    # 总结爆冷模式
    upset_summary = {
        "total_upsets": len(upsets),
        "upset_rate": len(upsets) / max(analysis.get("finished_matches", 1), 1),
        "upsets": upsets,
        "common_factors": _summarize_upset_factors(upsets),
    }
    return upset_summary


def _summarize_upset_factors(upsets):
    """总结爆冷常见因素"""
    factor_counts = {}
    for u in upsets:
        for r in u["reasons"]:
            key = r.split("(")[0]  # 去掉具体数值
            if "排名差距" in key:
                factor = "排名差距大但未体现"
            elif "五大联赛" in key:
                factor = "五大联赛优势未转化"
            elif "置信度" in key:
                factor = "低置信度预测"
            else:
                factor = "其他预测偏差"
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
    sorted_factors = sorted(factor_counts.items(), key=lambda x: -x[1])
    return [{"factor": k, "count": v} for k, v in sorted_factors]


def generate_agent_round1_report(data):
    """生成智能体在第一轮的表现报告"""
    agent_data = data.get("agent_data", {})
    weights_data = data.get("weights_data", {})
    matches = agent_data.get("matches", {})

    round_acc = weights_data.get("agent_round_accuracy", {})
    round_history = weights_data.get("round_history", {})
    group1_acc = round_acc.get("group_1", {})

    # group_1 accuracy might be in round_history if transition happened
    if not any(a.get("total", 0) > 0 for a in group1_acc.values()):
        history_g1 = round_history.get("group_1", {})
        group1_acc = history_g1.get("accuracy", group1_acc)

    agent_names = ["稳健派", "激进派", "价值派", "防守派", "数据派", "爆冷派"]
    report = []
    for name in agent_names:
        info = group1_acc.get(name, {})
        report.append({
            "name": name,
            "correct": info.get("correct", 0),
            "total": info.get("total", 0),
            "accuracy": info.get("accuracy", 0),
        })
    return sorted(report, key=lambda x: -x["accuracy"])


def run_review():
    """运行第一轮全面复盘"""
    log.info("=" * 60)
    log.info("第一轮小组赛全面复盘 (June 11-14)")
    log.info("=" * 60)

    data = load_all()
    g1_df = get_group1_matches(data)

    if len(g1_df) == 0:
        log.warning("无第一轮比赛数据")
        return None

    log.info(f"第一轮共 {len(g1_df)} 场比赛")

    # 1. 预测准确性分析
    analysis = analyze_predictions(g1_df)
    log.info(f"已完赛: {analysis['finished_matches']}/{analysis['total_matches']}")
    log.info(f"方向准确率: {analysis['accuracy']:.1%} ({analysis['result_correct']}/{analysis['finished_matches']})")
    log.info(f"精确比分准确率: {analysis['exact_score_accuracy']:.1%}")

    # 2. 关键因子分析
    factor_data = analyze_key_factors(g1_df, data)
    fs = factor_data.get("summary", {})
    log.info(f"关键因子:")
    log.info(f"  正确预测平均排名差: {fs.get('avg_rank_diff_correct', 0):.0f}")
    log.info(f"  错误预测平均排名差: {fs.get('avg_rank_diff_wrong', 0):.0f}")

    # 3. 爆冷分析
    upset_report = identify_upsets(analysis, factor_data)
    log.info(f"\n爆冷比赛: {upset_report['total_upsets']} 场 ({upset_report['upset_rate']:.1%})")
    for u in upset_report.get("upsets", []):
        log.info(f"  {u['home']} {u['score']} {u['away']} | 预测:{u['predicted']} 置信度:{u['confidence']:.0%}")
        for r in u["reasons"]:
            log.info(f"    → {r}")

    # 4. 智能体表现
    agent_report = generate_agent_round1_report(data)
    log.info(f"\n智能体第一轮表现:")
    for a in agent_report:
        log.info(f"  {a['name']}: {a['accuracy']:.1%} ({a['correct']}/{a['total']})")

    result = {
        "analysis": analysis,
        "key_factors": factor_data,
        "upsets": upset_report,
        "agent_round1": agent_report,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    # 保存
    path = os.path.join(DATA_DIR, "round1_review.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info(f"\n复盘报告已保存: {path}")
    return result


if __name__ == "__main__":
    run_review()
