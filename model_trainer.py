"""
model_trainer.py - 世界杯预测模型训练
使用: RandomForest + GradientBoosting + ExtraTrees + Logistic 集成
"""
import pandas as pd
import numpy as np
import os, pickle, logging
from datetime import datetime
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_MAP = {"H": 0, "D": 1, "A": 2}
RESULT_INV = {0: "H", 1: "D", 2: "A"}
RESULT_NAMES = {0: "主胜", 1: "平局", 2: "客胜"}
CONFIDENCE_THRESHOLD = 0.42


def elo_prob(pts_home, pts_away, draw_neutral=True):
    """基于FIFA排名点数的Elo概率估算"""
    rating_diff = pts_home - pts_away
    expected_h = 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))
    expected_a = 1.0 / (1.0 + 10 ** (rating_diff / 400.0))
    # 平局概率从排名差距估算：差距越小，平局概率越高
    draw_base = 0.22 + 0.20 * (1.0 - abs(rating_diff) / 800.0)
    draw_base = max(0.18, min(0.35, draw_base))
    # 调整
    remaining = 1.0 - draw_base
    prob_h = expected_h / (expected_h + expected_a) * remaining
    prob_a = expected_a / (expected_h + expected_a) * remaining
    return prob_h, draw_base, prob_a


class RankingModel:
    """基于FIFA排名的基准模型"""
    def __init__(self, rankings_df):
        self.rankings = {}
        if rankings_df is not None:
            for _, row in rankings_df.iterrows():
                self.rankings[row["Team"]] = int(row["RankPoints"])

    def predict_proba(self, feature_df):
        probs = []
        for _, row in feature_df.iterrows():
            home = row.get("home_team", "")
            away = row.get("away_team", "")
            hp = self.rankings.get(home, 1500)
            ap = self.rankings.get(away, 1500)
            ph, pd_val, pa = elo_prob(hp, ap)
            probs.append([ph, pd_val, pa])
        return np.array(probs)

    def predict(self, feature_df):
        probs = self.predict_proba(feature_df)
        return np.argmax(probs, axis=1)


def prepare_data(feature_df):
    df = feature_df[feature_df["is_finished"] == True].copy()
    if len(df) == 0:
        return None, None, None, None

    y = df["result"].map(RESULT_MAP).values
    exclude = [
        "result", "home_goals", "away_goals", "is_finished",
        "home_team", "away_team", "date", "match_id",
    ]
    feat_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]
    X = df[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log.info(f"训练数据: {len(X)}样本, {len(feat_cols)}特征")
    log.info(f"分布: H={sum(y==0)} D={sum(y==1)} A={sum(y==2)}")
    return X_scaled, y, feat_cols, scaler


def time_series_split(df, n=4):
    df = df[df["is_finished"] == True].copy()
    df = df.sort_values("date") if "date" in df.columns else df
    df = df.reset_index(drop=True)
    n_total = len(df)
    size = n_total // n
    folds = []
    for i in range(n):
        val_start = i * size
        val_end = n_total if i == n - 1 else (i + 1) * size
        if val_start >= 20:
            folds.append((list(range(val_start)), list(range(val_start, val_end))))
    if not folds:
        from sklearn.model_selection import KFold
        folds = [(tr.tolist(), va.tolist()) for tr, va in KFold(n, shuffle=True, random_state=42).split(df)]
    return folds, df


def train_rf(X_train, y_train, X_val, y_val, params=None):
    from sklearn.ensemble import RandomForestClassifier
    p = {"n_estimators": 400, "max_depth": 12, "min_samples_split": 4, "min_samples_leaf": 1,
         "class_weight": "balanced", "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
    if params: p.update(params)
    m = RandomForestClassifier(**p)
    m.fit(X_train, y_train)
    p_val = m.predict_proba(X_val)
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val))}


def train_gb(X_train, y_train, X_val, y_val, params=None):
    from sklearn.ensemble import GradientBoostingClassifier
    p = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8,
         "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42}
    if params: p.update(params)
    m = GradientBoostingClassifier(**p)
    m.fit(X_train, y_train)
    p_val = m.predict_proba(X_val)
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val))}


def train_et(X_train, y_train, X_val, y_val, params=None):
    from sklearn.ensemble import ExtraTreesClassifier
    p = {"n_estimators": 400, "max_depth": 12, "min_samples_split": 3, "min_samples_leaf": 1,
         "class_weight": "balanced", "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
    if params: p.update(params)
    m = ExtraTreesClassifier(**p)
    m.fit(X_train, y_train)
    p_val = m.predict_proba(X_val)
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val))}


def train_lr(X_train, y_train, X_val, y_val, params=None):
    from sklearn.linear_model import LogisticRegression
    p = {"C": 5.0, "max_iter": 5000, "solver": "lbfgs", "random_state": 42, "multi_class": "multinomial"}
    if params: p.update(params)
    m = LogisticRegression(**p)
    m.fit(X_train, y_train)
    p_val = m.predict_proba(X_val)
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val))}


def weighted_ensemble(probs_list, weights):
    w = np.array(weights) / sum(weights)
    return sum(p * wi for p, wi in zip(probs_list, w))


def notify_no_training_data(feature_df):
    """无训练数据时发布警告并提供排名模型替代方案"""
    import warnings
    msg = (
        "\n" + "!" * 50 +
        "\n! 无历史完赛数据，无法训练ML模型" +
        "\n! 将使用基于FIFA排名的Elo概率模型" +
        "\n! 世界杯开赛后，完赛数据会被自动用于增量学习" +
        "\n" + "!" * 50
    )
    log.warning(msg)

    # 加载排名数据构建排名模型
    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    rankings_df = None
    if os.path.exists(rank_path):
        rankings_df = pd.read_csv(rank_path)
        log.info(f"加载FIFA排名: {len(rankings_df)} 支球队")
    model = RankingModel(rankings_df)
    return model


def run_backtest(feature_df):
    log.info("=" * 50)
    log.info("世界杯模型回测")
    log.info("=" * 50)

    df_finished = feature_df[feature_df["is_finished"] == True]

    # 无训练数据时使用排名模型
    if len(df_finished) < 10:
        model = notify_no_training_data(feature_df)
        # 构建伪结果集
        dummy_results = []
        result = {
            "models": {"final": model},
            "summary": {
                "avg_accuracy": 0.0, "avg_log_loss": 0.0,
                "n_matches": 0, "n_folds": 0,
                "feature_cols": [c for c in feature_df.columns
                                 if feature_df[c].dtype in [np.float64, np.int64, float, int]
                                 and c not in ["result", "is_finished", "home_goals", "away_goals",
                                               "home_team", "away_team", "date", "match_id"]],
            },
        }
        pkg_path = os.path.join(MODEL_DIR, "wc_model_package.pkl")
        with open(pkg_path, "wb") as f:
            pickle.dump(result, f)
        log.info(f"✅ 排名模型已保存: {pkg_path}")
        return None, result["summary"]

    folds, df_sorted = time_series_split(feature_df)
    X, y, feat_cols, scaler = prepare_data(df_sorted)
    if X is None:
        return None, None

    results, models_dict = [], {}
    rf_ll, gb_ll, et_ll, lr_ll = [], [], [], []

    rf_configs = [
        {"n_estimators": 300, "max_depth": 10, "min_samples_split": 5},
        {"n_estimators": 400, "max_depth": 12, "min_samples_split": 4},
        {"n_estimators": 500, "max_depth": 8, "min_samples_split": 8},
    ]
    gb_configs = [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.04},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03},
        {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.02},
    ]

    for fi, (tr, va) in enumerate(folds):
        log.info(f"\n折 {fi+1}/{len(folds)}: train={len(tr)} val={len(va)}")
        X_tr, X_va = X[tr], X[va]
        y_tr, y_va = y[tr], y[va]

        best_rf, best_gb = None, None
        best_rf_ll, best_gb_ll = 999, 999

        for cfg in rf_configs:
            m, met = train_rf(X_tr, y_tr, X_va, y_va, cfg)
            if met["logloss"] < best_rf_ll:
                best_rf_ll = met["logloss"]
                best_rf = (m, met)
        rf_ll.append(best_rf_ll)

        for cfg in gb_configs:
            m, met = train_gb(X_tr, y_tr, X_va, y_va, cfg)
            if met["logloss"] < best_gb_ll:
                best_gb_ll = met["logloss"]
                best_gb = (m, met)
        gb_ll.append(best_gb_ll)

        m, met = train_et(X_tr, y_tr, X_va, y_va)
        best_et = (m, met)
        et_ll.append(met["logloss"])

        m, met = train_lr(X_tr, y_tr, X_va, y_va)
        best_lr = (m, met)
        lr_ll.append(met["logloss"])

        rf_m, rf_met = best_rf
        gb_m, gb_met = best_gb
        et_m, et_met = best_et
        lr_m, lr_met = best_lr

        rf_p = rf_m.predict_proba(X_va)
        gb_p = gb_m.predict_proba(X_va)
        et_p = et_m.predict_proba(X_va)
        lr_p = lr_m.predict_proba(X_va)

        w = [1 / max(rf_met["logloss"], 0.01), 1 / max(gb_met["logloss"], 0.01),
             1 / max(et_met["logloss"], 0.01), 1 / max(lr_met["logloss"], 0.01)]
        en_p = weighted_ensemble([rf_p, gb_p, et_p, lr_p], w)
        en_pred = np.argmax(en_p, axis=1)
        en_acc = accuracy_score(y_va, en_pred)
        en_ll = log_loss(y_va, en_p)

        log.info(f"  RF={rf_met['acc']:.3f}(ll={rf_met['logloss']:.3f}) GB={gb_met['acc']:.3f}(ll={gb_met['logloss']:.3f})")
        log.info(f"  ET={et_met['acc']:.3f}(ll={et_met['logloss']:.3f}) LR={lr_met['acc']:.3f}(ll={lr_met['logloss']:.3f})")
        log.info(f"  Ensemble: ACC={en_acc:.3f} LogLoss={en_ll:.3f}")

        vdf = df_sorted.iloc[va].copy()
        vdf["pred_H"] = en_p[:, 0]
        vdf["pred_D"] = en_p[:, 1]
        vdf["pred_A"] = en_p[:, 2]
        vdf["pred_r"] = [RESULT_INV[p] for p in en_pred]
        vdf["correct"] = (en_pred == y_va).astype(int)

        results.append({
            "fold": fi,
            "predictions": vdf[["date", "home_team", "away_team", "result",
                                "pred_H", "pred_D", "pred_A", "pred_r", "correct"]].copy(),
            "metrics": {
                "rf": rf_met, "gb": gb_met, "et": et_met, "lr": lr_met,
                "ensemble": {"accuracy": float(en_acc), "log_loss": float(en_ll)},
            },
        })
        models_dict[f"fold_{fi}"] = {"rf": rf_m, "gb": gb_m, "et": et_m, "lr": lr_m, "feature_cols": feat_cols}

    avg_en = np.mean([r["metrics"]["ensemble"]["accuracy"] for r in results])
    avg_en_ll = np.mean([r["metrics"]["ensemble"]["log_loss"] for r in results])

    log.info(f"\n{'='*50}")
    log.info("回测汇总:")
    log.info(f"  RandomForest:     {np.mean([r['metrics']['rf']['acc'] for r in results]):.3f}")
    log.info(f"  GradBoost:        {np.mean([r['metrics']['gb']['acc'] for r in results]):.3f}")
    log.info(f"  ExtraTrees:       {np.mean([r['metrics']['et']['acc'] for r in results]):.3f}")
    log.info(f"  Logistic:         {np.mean([r['metrics']['lr']['acc'] for r in results]):.3f}")
    log.info(f"  Ensemble(加权):   {avg_en:.3f} | LL={avg_en_ll:.3f}")
    log.info(f"{'='*50}")

    # 全量训练最终模型
    log.info("\n训练最终模型 (全量)...")
    final = {}
    for name, fn, cfg in [
        ("rf", train_rf, {"n_estimators": 400, "max_depth": 12}),
        ("gb", train_gb, {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03}),
        ("et", train_et, {}),
        ("lr", train_lr, {"C": 1.0}),
    ]:
        m, _ = fn(X, y, X[:5], y[:5], cfg)
        final[name] = m
    final["feature_cols"] = feat_cols

    models_dict["final"] = final

    summary = {
        "avg_accuracy": float(avg_en),
        "avg_log_loss": float(avg_en_ll),
        "n_matches": sum(len(r["predictions"]) for r in results),
        "n_folds": len(folds),
        "feature_cols": feat_cols,
    }

    pkg_path = os.path.join(MODEL_DIR, "wc_model_package.pkl")
    with open(pkg_path, "wb") as f:
        pickle.dump({"models": models_dict, "summary": summary, "scaler": scaler}, f)
    bt_path = os.path.join(DATA_DIR, "wc_backtest_results.pkl")
    with open(bt_path, "wb") as f:
        pickle.dump(results, f)
    log.info(f"✅ 模型已保存: {pkg_path}")
    return results, summary


def predict_matches(feature_df):
    """使用训练好的模型预测比赛"""
    model_path = os.path.join(MODEL_DIR, "wc_model_package.pkl")
    if not os.path.exists(model_path):
        log.error("模型不存在，请先训练")
        return None

    with open(model_path, "rb") as f:
        pkg = pickle.load(f)

    final = pkg["models"].get("final")
    if final is None:
        log.error("无最终模型")
        return None

    # 检测是否是排名模型（有rankings属性）
    if hasattr(final, "rankings"):
        log.info("使用FIFA排名Elo模型进行预测")
        ensemble_prob = final.predict_proba(feature_df)
        ensemble_pred = np.argmax(ensemble_prob, axis=1)
    else:
        feat_cols = final["feature_cols"]
        X_raw = feature_df[feat_cols].fillna(0).values
        scaler = pkg.get("scaler")
        X = scaler.transform(X_raw) if scaler else X_raw

        rf_p = final["rf"].predict_proba(X)
        gb_p = final["gb"].predict_proba(X)
        et_p = final["et"].predict_proba(X)
        lr_p = final["lr"].predict_proba(X)
        ensemble_prob = (rf_p + gb_p + et_p + lr_p) / 4
        ensemble_pred = np.argmax(ensemble_prob, axis=1)

    results = feature_df[["date", "home_team", "away_team"]].copy()
    results["pred_H"] = ensemble_prob[:, 0]
    results["pred_D"] = ensemble_prob[:, 1]
    results["pred_A"] = ensemble_prob[:, 2]
    results["pred_result"] = [RESULT_INV[p] for p in ensemble_pred]

    # 智能标签处理
    smart_labels = []
    for i in range(len(results)):
        probs = ensemble_prob[i]
        sorted_p = sorted(probs, reverse=True)
        if sorted_p[0] - sorted_p[1] < 0.05 and probs[1] > 0.30:
            smart_labels.append("D")
        elif sorted_p[0] - sorted_p[1] < 0.03:
            smart_labels.append("D")
        else:
            smart_labels.append(RESULT_INV[ensemble_pred[i]])

    results["pred_label"] = [{"H": "主胜", "D": "平局", "A": "客胜"}[r] for r in smart_labels]
    results["confidence"] = np.max(ensemble_prob, axis=1)
    results["is_finished"] = feature_df["is_finished"].values if "is_finished" in feature_df.columns else False
    results["fair_odds_H"] = 1.0 / np.clip(results["pred_H"], 0.01, 0.99)
    results["fair_odds_D"] = 1.0 / np.clip(results["pred_D"], 0.01, 0.99)
    results["fair_odds_A"] = 1.0 / np.clip(results["pred_A"], 0.01, 0.99)

    # 价值评分
    results["value_score"] = 0.0
    for i in range(len(results)):
        max_prob = results["confidence"].iloc[i]
        if results["pred_result"].iloc[i] == "H":
            results.loc[i, "value_score"] = max_prob * 0.92 / results["fair_odds_H"].iloc[i]
        elif results["pred_result"].iloc[i] == "D":
            results.loc[i, "value_score"] = max_prob * 0.92 / results["fair_odds_D"].iloc[i]
        else:
            results.loc[i, "value_score"] = max_prob * 0.92 / results["fair_odds_A"].iloc[i]

    if "result" in feature_df.columns:
        results["actual_result"] = feature_df["result"].values
        results["correct"] = results["pred_result"].values == feature_df["result"].values
    else:
        results["actual_result"] = "U"
        results["correct"] = False

    return results


if __name__ == "__main__":
    feat_path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            feature_df = pickle.load(f)
        log.info(f"加载特征矩阵: {feature_df.shape}")
        run_backtest(feature_df)
    else:
        log.error("请先运行特征工程")
