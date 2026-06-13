"""apply_finished.py - 将已完成比赛结果写入预测数据文件并更新智能体权重"""
import os, json, pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

FINISHED_PATH = os.path.join(DATA_DIR, "finished_matches.json")
PRED_PATH = os.path.join(DATA_DIR, "wc_predictions.pkl")
AGENT_PATH = os.path.join(DATA_DIR, "agent_predictions.json")
WEIGHTS_PATH = os.path.join(DATA_DIR, "agent_weights.json")

AGENT_NAMES = ["稳健派", "激进派", "价值派", "防守派", "数据派", "爆冷派"]

# 轮次定义（基于赛程日期）
ROUND_DATES = {
    "group_1": ("2026-06-11", "2026-06-14"),
    "group_2": ("2026-06-15", "2026-06-19"),
    "group_3": ("2026-06-20", "2026-06-27"),
    "r16":     ("2026-06-28", "2026-06-30"),
    "quarter": ("2026-07-03", "2026-07-04"),
    "semi":    ("2026-07-08", "2026-07-09"),
    "final":   ("2026-07-12", "2026-07-12"),
}


def load_finished():
    if not os.path.exists(FINISHED_PATH):
        return {}
    with open(FINISHED_PATH) as f:
        return json.load(f)


def apply_to_pickle(finished):
    if not os.path.exists(PRED_PATH):
        print("  wc_predictions.pkl not found, skipping")
        return
    with open(PRED_PATH, "rb") as f:
        df = pickle.load(f)
    updated = 0
    for mid, info in finished.items():
        mask = (
            (df["home_team"] == info["home_team"])
            & (df["away_team"] == info["away_team"])
            & (df["is_finished"] == False)
        )
        if mask.any():
            df.loc[mask, "actual_home_score"] = info["actual_home_score"]
            df.loc[mask, "actual_away_score"] = info["actual_away_score"]
            df.loc[mask, "actual_result"] = info["actual_result"]
            df.loc[mask, "is_finished"] = True
            pred_h = df.loc[mask, "pred_home_score"].values[0]
            pred_a = df.loc[mask, "pred_away_score"].values[0]
            correct = int(pred_h == info["actual_home_score"] and pred_a == info["actual_away_score"])
            if "correct" not in df.columns:
                df["correct"] = False
            df.loc[mask, "correct"] = bool(correct)
            updated += 1
    if updated > 0:
        with open(PRED_PATH, "wb") as f:
            pickle.dump(df, f)
    print(f"  wc_predictions.pkl: {updated} matches updated")


def apply_to_agent_json(finished):
    if not os.path.exists(AGENT_PATH):
        print("  agent_predictions.json not found, skipping")
        return
    with open(AGENT_PATH) as f:
        data = json.load(f)
    matches = data.get("matches", {})
    updated = 0
    for mid, info in finished.items():
        if mid in matches:
            m = matches[mid]
            m["actual_home_score"] = info["actual_home_score"]
            m["actual_away_score"] = info["actual_away_score"]
            m["actual_result"] = info["actual_result"]
            m["is_finished"] = True
            for aname, p in m.get("agents", {}).items():
                if isinstance(p, dict) and "home_score" in p:
                    ph = p.get("home_score")
                    pa = p.get("away_score")
                    p["correct"] = (ph == info["actual_home_score"] and pa == info["actual_away_score"])
            updated += 1
    if updated > 0:
        with open(AGENT_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  agent_predictions.json: {updated} matches updated")


# ─── 权重系统 ─────────────────────────────────────────────


def load_agent_weights():
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    default = {
        "current_round": "group_1",
        "weights": {name: 1.0 for name in AGENT_NAMES},
        "round_history": {},
        "agent_round_accuracy": {},
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(WEIGHTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(default, f, ensure_ascii=False, indent=2)
    return default


def save_agent_weights(weights_data):
    with open(WEIGHTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(weights_data, f, ensure_ascii=False, indent=2)


def get_round_for_date(date_str):
    """根据日期判断所属轮次"""
    for round_name, (start, end) in ROUND_DATES.items():
        if start <= date_str <= end:
            return round_name
    return "group_1"


def detect_round_transition(weights_data, latest_date_str):
    """检测轮次切换，需要时重置权重"""
    new_round = get_round_for_date(latest_date_str)
    old_round = weights_data.get("current_round", "group_1")

    if new_round == old_round:
        return weights_data, False

    # 保存当前轮次权重历史
    if "round_history" not in weights_data:
        weights_data["round_history"] = {}
    weights_data["round_history"][old_round] = {
        "weights": dict(weights_data["weights"]),
        "accuracy": weights_data.get("agent_round_accuracy", {}).get(old_round, {}),
    }

    # 新轮次：80%重置为1.0 + 20%记忆上一轮
    prev_weights = weights_data["weights"]
    new_weights = {}
    for agent_name in AGENT_NAMES:
        prev = prev_weights.get(agent_name, 1.0)
        blended = 0.8 * 1.0 + 0.2 * max(0.5, min(2.0, prev))
        new_weights[agent_name] = round(blended, 4)

    weights_data["current_round"] = new_round
    weights_data["weights"] = new_weights

    # 初始化新轮次准确率追踪
    if "agent_round_accuracy" not in weights_data:
        weights_data["agent_round_accuracy"] = {}
    weights_data["agent_round_accuracy"][new_round] = {
        name: {"correct": 0, "total": 0, "accuracy": 0} for name in AGENT_NAMES
    }

    return weights_data, True


def update_agent_weights(finished):
    """核心权重更新：根据实际结果调整每个智能体的权重

    公式:
      猜对: weight *= (1 + confidence * 0.15)
      猜错: weight *= (1 - confidence * 0.10)
      钳制: [0.3, 3.0]

    以 result 方向为准（H/D/A），不要求精确比分。
    """
    if not os.path.exists(AGENT_PATH):
        print("  agent_predictions.json not found, skipping weight update")
        return

    weights_data = load_agent_weights()

    with open(AGENT_PATH, 'r', encoding='utf-8') as f:
        agent_data = json.load(f)

    matches = agent_data.get("matches", {})
    updated_count = 0
    round_acc = weights_data.get("agent_round_accuracy", {}).get(
        weights_data["current_round"], {}
    )
    if not round_acc:
        round_acc = {name: {"correct": 0, "total": 0, "accuracy": 0} for name in AGENT_NAMES}

    for mid, info in finished.items():
        if mid not in matches:
            continue
        m = matches[mid]
        if m.get("weights_applied", False):
            continue

        agents_preds = m.get("agents", {})
        if not agents_preds:
            continue

        actual_result = info.get("actual_result", "")
        if not actual_result:
            continue

        for agent_name, pred in agents_preds.items():
            if not isinstance(pred, dict) or "result" not in pred:
                continue

            predicted = pred["result"]
            confidence = pred.get("confidence", 0.5)
            correct = (predicted == actual_result)

            current_weight = weights_data["weights"].get(agent_name, 1.0)
            if correct:
                reward = confidence * 0.15
                new_weight = current_weight * (1 + reward)
            else:
                penalty = confidence * 0.10
                new_weight = current_weight * (1 - penalty)

            new_weight = max(0.3, min(3.0, new_weight))
            weights_data["weights"][agent_name] = round(new_weight, 4)

            # 更新轮次准确率
            if agent_name in round_acc:
                round_acc[agent_name]["total"] += 1
                if correct:
                    round_acc[agent_name]["correct"] += 1
                t = round_acc[agent_name]["total"]
                c = round_acc[agent_name]["correct"]
                round_acc[agent_name]["accuracy"] = round(c / t, 4) if t > 0 else 0

        m["weights_applied"] = True
        updated_count += 1

    if updated_count > 0:
        if "agent_round_accuracy" not in weights_data:
            weights_data["agent_round_accuracy"] = {}
        weights_data["agent_round_accuracy"][weights_data["current_round"]] = round_acc

        with open(AGENT_PATH, 'w', encoding='utf-8') as f:
            json.dump(agent_data, f, indent=2, ensure_ascii=False)

        save_agent_weights(weights_data)

        print(f"  Agent weights updated: {updated_count} matches")
        for name in AGENT_NAMES:
            w = weights_data["weights"].get(name, 1.0)
            acc = round_acc.get(name, {}).get("accuracy", 0)
            print(f"    {name}: weight={w:.3f} round_acc={acc:.1%}")
    else:
        print("  No new matches to apply weights for")


def reoptimize_hybrid_params():
    """每次有新完赛比赛时，重新优化混合模型参数（在线学习）"""
    feat_path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    finished_path = FINISHED_PATH
    top5_path = os.path.join(DATA_DIR, "team_top5.json")
    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")

    if not all(os.path.exists(p) for p in [feat_path, finished_path, top5_path, rank_path]):
        print("  [hybrid] 缺少数据文件，跳过")
        return

    try:
        from model_trainer import optimize_hybrid_params
        with open(finished_path) as f:
            finished_matches = json.load(f)
        with open(top5_path) as f:
            top5_data = json.load(f)
        rankings_df = pd.read_csv(rank_path)

        if len(finished_matches) == 0:
            print("  [hybrid] 无已完赛比赛，跳过")
            return

        result = optimize_hybrid_params(finished_matches, top5_data, rankings_df)
        best = result["best_params"]
        print(f"  [hybrid] 优化完成! 参数: strength={best['top5_strength']}, "
              f"threshold={best['close_threshold']}, min_weight={best['elo_min_weight']}")
        print(f"  [hybrid] 准确率={result['history'][0]['accuracy']:.1%} "
              f"(基于{len(finished_matches)}场完赛比赛)")
    except Exception as e:
        print(f"  [hybrid] 重优化失败: {e}")


def main():
    finished = load_finished()
    if not finished:
        print("No finished matches data")
        return

    print(f"Applying {len(finished)} finished match(es)...")
    apply_to_pickle(finished)
    apply_to_agent_json(finished)

    # 权重更新
    print("\n--- Agent Weight Update ---")
    update_agent_weights(finished)

    # 混合模型参数重优化（每次有新完赛比赛时触发）
    print("\n--- Hybrid Model Re-optimization ---")
    reoptimize_hybrid_params()

    # 轮次切换检测
    latest_date = max(
        (info.get("date", "2026-06-11") for info in finished.values()),
        default="2026-06-11",
    )
    weights_data = load_agent_weights()
    weights_data, transitioned = detect_round_transition(weights_data, latest_date)
    if transitioned:
        print(f"  Round transition: {weights_data['current_round']}")
        for name, w in weights_data["weights"].items():
            print(f"    {name}: {w:.3f}")
    save_agent_weights(weights_data)


if __name__ == "__main__":
    main()
