"""apply_finished.py - 将已完成比赛结果写入预测数据文件"""
import os, json, pickle, sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

FINISHED_PATH = os.path.join(DATA_DIR, "finished_matches.json")
PRED_PATH = os.path.join(DATA_DIR, "wc_predictions.pkl")
AGENT_PATH = os.path.join(DATA_DIR, "agent_predictions.json")


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
            # correct flag: exact score match
            pred_h = df.loc[mask, "pred_home_score"].values[0]
            pred_a = df.loc[mask, "pred_away_score"].values[0]
            correct = int(pred_h == info["actual_home_score"] and pred_a == info["actual_away_score"])
            df.loc[mask, "correct"] = correct
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
            # Mark each agent's correctness
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


def main():
    finished = load_finished()
    if not finished:
        print("No finished matches data")
        return
    print(f"Applying {len(finished)} finished match(es)...")
    apply_to_pickle(finished)
    apply_to_agent_json(finished)


if __name__ == "__main__":
    main()
