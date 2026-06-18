"""
model_trainer.py - 世界杯预测模型训练
使用: RandomForest + GradientBoosting + ExtraTrees + Logistic 集成
"""
import pandas as pd
import numpy as np
import os, json, math, pickle, logging
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
    """基于FIFA排名点数的Elo概率估算（降低平局概率以增加区分度）"""
    rating_diff = pts_home - pts_away
    expected_h = 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))
    expected_a = 1.0 / (1.0 + 10 ** (rating_diff / 400.0))
    # 降低平局基础概率以增加主/客胜区分度
    draw_base = 0.18 + 0.14 * (1.0 - abs(rating_diff) / 800.0)
    draw_base = max(0.12, min(0.28, draw_base))
    remaining = 1.0 - draw_base
    prob_h = expected_h / (expected_h + expected_a) * remaining
    prob_a = expected_a / (expected_h + expected_a) * remaining
    return prob_h, draw_base, prob_a


def exp_goals_from_elo(pts_home, pts_away):
    """基于ELO差值估算期望进球数"""
    diff = (pts_home - pts_away) / 400.0
    exp_h = 1.4 + diff * 0.8
    exp_a = 1.4 - diff * 0.8
    return max(0.3, min(3.5, exp_h)), max(0.3, min(3.5, exp_a))


def _normalize_probs(p_h, p_d, p_a, min_prob=0.05):
    """钳制并重新归一化概率"""
    probs = [max(min_prob, p) for p in [p_h, p_d, p_a]]
    total = sum(probs)
    return probs[0] / total, probs[1] / total, probs[2] / total


def hybrid_elo_prob(pts_home, pts_away, top5_home, top5_away,
                    top5_strength=0.15, close_threshold=100, elo_min_weight=0.4):
    """混合ElO概率：排名相近时降低排名权重，增加五大联赛球员含量权重

    参数:
        pts_home, pts_away: FIFA排名积分
        top5_home, top5_away: 五大联赛球员比例 (0.0~1.0)
        top5_strength: top5差异对概率的偏移强度
        close_threshold: 排名分差阈值，超过此值则纯用Elo
        elo_min_weight: 排名分差为0时Elo的最小权重
    返回:
        (prob_h, prob_d, prob_a)
    """
    # 1. 纯Elo概率
    prob_h, prob_d, prob_a = elo_prob(pts_home, pts_away)

    # 2. Top-5调整后的概率
    top5_diff = top5_home - top5_away
    t5_h, t5_d, t5_a = prob_h, prob_d, prob_a
    if abs(top5_diff) > 0.05:
        shift = top5_diff * top5_strength
        t5_h += shift
        t5_a -= shift * 0.6
        t5_d -= shift * 0.4
        t5_h, t5_d, t5_a = _normalize_probs(t5_h, t5_d, t5_a)

    # 3. 计算混合权重alpha
    #    rank_diff越大 → alpha越接近1.0（纯Elo）
    #    rank_diff越小 → alpha越接近elo_min_weight（Top5占更大比重）
    rank_diff = abs(pts_home - pts_away)
    alpha_ratio = min(rank_diff / close_threshold, 1.0)
    alpha = elo_min_weight + (1.0 - elo_min_weight) * alpha_ratio

    # 4. 混合
    hybrid_h = alpha * prob_h + (1.0 - alpha) * t5_h
    hybrid_d = alpha * prob_d + (1.0 - alpha) * t5_d
    hybrid_a = alpha * prob_a + (1.0 - alpha) * t5_a

    return _normalize_probs(hybrid_h, hybrid_d, hybrid_a)


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


class HybridRankingModel(RankingModel):
    """增强排名模型：混合Elo + 五大联赛球员密度

    当排名接近时，降低排名权重，增加五大联赛球员含量权重。
    参数通过网格搜索从已完赛比赛中学习。
    """
    def __init__(self, rankings_df, top5_data=None, params=None):
        super().__init__(rankings_df)
        self.top5_data = top5_data or {}
        self.params = params or {
            "top5_strength": 0.15,
            "close_threshold": 100,
            "elo_min_weight": 0.4,
        }

    def get_top5(self, team):
        """获取球队五大联赛比例，默认0.10"""
        return self.top5_data.get(team, 0.10)

    def predict_proba(self, feature_df):
        probs = []
        for _, row in feature_df.iterrows():
            home = str(row.get("home_team", ""))
            away = str(row.get("away_team", ""))
            hp = self.rankings.get(home, 1500)
            ap = self.rankings.get(away, 1500)
            t5h = self.get_top5(home)
            t5a = self.get_top5(away)
            ph, pd_val, pa = hybrid_elo_prob(
                hp, ap, t5h, t5a,
                top5_strength=self.params["top5_strength"],
                close_threshold=self.params["close_threshold"],
                elo_min_weight=self.params["elo_min_weight"],
            )
            probs.append([ph, pd_val, pa])
        return np.array(probs)


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
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val, labels=[0,1,2]))}


def train_gb(X_train, y_train, X_val, y_val, params=None):
    from sklearn.ensemble import GradientBoostingClassifier
    p = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8,
         "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "sqrt", "random_state": 42}
    if params: p.update(params)
    m = GradientBoostingClassifier(**p)
    m.fit(X_train, y_train)
    p_val = m.predict_proba(X_val)
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val, labels=[0,1,2]))}


def train_et(X_train, y_train, X_val, y_val, params=None):
    from sklearn.ensemble import ExtraTreesClassifier
    p = {"n_estimators": 400, "max_depth": 12, "min_samples_split": 3, "min_samples_leaf": 1,
         "class_weight": "balanced", "max_features": "sqrt", "random_state": 42, "n_jobs": -1}
    if params: p.update(params)
    m = ExtraTreesClassifier(**p)
    m.fit(X_train, y_train)
    p_val = m.predict_proba(X_val)
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val, labels=[0,1,2]))}


def train_lr(X_train, y_train, X_val, y_val, params=None):
    from sklearn.linear_model import LogisticRegression
    p = {"C": 5.0, "max_iter": 5000, "solver": "lbfgs", "random_state": 42}
    if params: p.update(params)
    m = LogisticRegression(**p)
    m.fit(X_train, y_train)
    p_val = m.predict_proba(X_val)
    return m, {"acc": float(accuracy_score(y_val, m.predict(X_val))), "logloss": float(log_loss(y_val, p_val, labels=[0,1,2]))}


def train_mlp(X_train, y_train, X_val, y_val, params=None):
    """MLP神经网络分类器"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    # MLP需要独立缩放
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_train)
    X_va_sc = sc.transform(X_val)
    p = {"hidden_layer_sizes": (64, 32), "activation": "relu", "solver": "adam",
         "max_iter": 2000, "random_state": 42, "early_stopping": True,
         "validation_fraction": 0.15, "n_iter_no_change": 20, "batch_size": 32,
         "alpha": 0.001}
    if params: p.update(params)
    m = MLPClassifier(**p)
    m.fit(X_tr_sc, y_train)
    p_val = m.predict_proba(X_va_sc)
    acc = float(accuracy_score(y_val, m.predict(X_va_sc)))
    ll = float(log_loss(y_val, p_val, labels=[0,1,2]))
    return m, {"acc": acc, "logloss": ll, "scaler": sc}


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
        "\n! 混合模式：排名接近时降低排名权重，增加五大联赛含量权重" +
        "\n" + "!" * 50
    )
    log.warning(msg)

    # 加载排名数据构建排名模型
    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    rankings_df = None
    if os.path.exists(rank_path):
        rankings_df = pd.read_csv(rank_path)
        log.info(f"加载FIFA排名: {len(rankings_df)} 支球队")

    # 使用混合模型（Elo + Top5）
    top5_path = os.path.join(DATA_DIR, "team_top5.json")
    top5_data = {}
    if os.path.exists(top5_path):
        with open(top5_path) as f:
            top5_data = json.load(f)

    params = load_hybrid_params()
    model = HybridRankingModel(rankings_df, top5_data=top5_data, params=params)
    log.info(f"混合模型参数: strength={params['top5_strength']}, "
             f"threshold={params['close_threshold']}, min_weight={params['elo_min_weight']}")
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
    rf_ll, gb_ll, et_ll, lr_ll, mlp_ll = [], [], [], [], []

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

        m, met = train_mlp(X_tr, y_tr, X_va, y_va)
        best_mlp = (m, met)
        mlp_ll.append(met["logloss"])

        rf_m, rf_met = best_rf
        gb_m, gb_met = best_gb
        et_m, et_met = best_et
        lr_m, lr_met = best_lr
        mlp_m, mlp_met = best_mlp

        rf_p = rf_m.predict_proba(X_va)
        gb_p = gb_m.predict_proba(X_va)
        et_p = et_m.predict_proba(X_va)
        lr_p = lr_m.predict_proba(X_va)
        mlp_p = mlp_m.predict_proba(X_va)

        w = [1 / max(rf_met["logloss"], 0.01), 1 / max(gb_met["logloss"], 0.01),
             1 / max(et_met["logloss"], 0.01), 1 / max(lr_met["logloss"], 0.01),
             1 / max(mlp_met["logloss"], 0.01)]
        en_p = weighted_ensemble([rf_p, gb_p, et_p, lr_p, mlp_p], w)
        en_pred = np.argmax(en_p, axis=1)
        en_acc = accuracy_score(y_va, en_pred)
        en_ll = log_loss(y_va, en_p, labels=[0,1,2])

        log.info(f"  RF={rf_met['acc']:.3f}(ll={rf_met['logloss']:.3f}) GB={gb_met['acc']:.3f}(ll={gb_met['logloss']:.3f})")
        log.info(f"  ET={et_met['acc']:.3f}(ll={et_met['logloss']:.3f}) LR={lr_met['acc']:.3f}(ll={lr_met['logloss']:.3f}) MLP={mlp_met['acc']:.3f}(ll={mlp_met['logloss']:.3f})")
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
                "rf": rf_met, "gb": gb_met, "et": et_met, "lr": lr_met, "mlp": mlp_met,
                "ensemble": {"accuracy": float(en_acc), "log_loss": float(en_ll)},
            },
        })
        models_dict[f"fold_{fi}"] = {"rf": rf_m, "gb": gb_m, "et": et_m, "lr": lr_m, "mlp": mlp_m, "mlp_scaler": mlp_met.get("scaler"), "feature_cols": feat_cols}

    avg_en = np.mean([r["metrics"]["ensemble"]["accuracy"] for r in results])
    avg_en_ll = np.mean([r["metrics"]["ensemble"]["log_loss"] for r in results])

    log.info(f"\n{'='*50}")
    log.info("回测汇总:")
    log.info(f"  RandomForest:     {np.mean([r['metrics']['rf']['acc'] for r in results]):.3f}")
    log.info(f"  GradBoost:        {np.mean([r['metrics']['gb']['acc'] for r in results]):.3f}")
    log.info(f"  ExtraTrees:       {np.mean([r['metrics']['et']['acc'] for r in results]):.3f}")
    log.info(f"  Logistic:         {np.mean([r['metrics']['lr']['acc'] for r in results]):.3f}")
    log.info(f"  MLP(神经网络):    {np.mean([r['metrics']['mlp']['acc'] for r in results]):.3f}")
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
        ("mlp", train_mlp, {}),
    ]:
        m, met_info = fn(X, y, X[:5], y[:5], cfg)
        final[name] = m
        if name == "mlp" and "scaler" in met_info:
            final["mlp_scaler"] = met_info["scaler"]
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

        # 构建混合模型作为抗过拟合后备
        from data_collector import load_data, load_rankings
        import json
        rankings_df = load_rankings()
        top5_data = {}
        top5_path = os.path.join(DATA_DIR, "team_top5.json")
        if os.path.exists(top5_path):
            with open(top5_path) as f:
                top5_data = json.load(f)
        params = load_hybrid_params()
        hybrid_model = HybridRankingModel(rankings_df, top5_data=top5_data, params=params)
        hybrid_prob = hybrid_model.predict_proba(feature_df)

        X = scaler.transform(X_raw) if scaler else X_raw

        rf_p = final["rf"].predict_proba(X)
        gb_p = final["gb"].predict_proba(X)
        et_p = final["et"].predict_proba(X)
        lr_p = final["lr"].predict_proba(X)
        # MLP needs separate scaling
        mlp_scaler = final.get("mlp_scaler")
        if mlp_scaler is not None and "mlp" in final:
            X_mlp = mlp_scaler.transform(X)
            mlp_p = final["mlp"].predict_proba(X_mlp)
        else:
            mlp_p = (rf_p + gb_p + et_p + lr_p) / 4  # fallback

        # 混合模型权重：训练样本少时降低ML权重
        n_train = len(feature_df[feature_df["is_finished"] == True]) if "is_finished" in feature_df.columns else 24
        ml_weight = min(n_train / 80.0, 0.25)  # 最多25%权重给ML
        log.info(f"混合预测: ML权重={ml_weight:.2f}, Hybrid权重={1-ml_weight:.2f} (训练样本={n_train})")

        ensemble_prob = ml_weight * (rf_p + gb_p + et_p + lr_p + mlp_p) / 5 + (1 - ml_weight) * hybrid_prob

        # 平局概率校正：排名差距大时降低平局概率
        for i in range(len(ensemble_prob)):
            home = str(feature_df.iloc[i].get("home_team", ""))
            away = str(feature_df.iloc[i].get("away_team", ""))
            hp = feature_df.iloc[i].get("home_rank_pts", 1500)
            ap = feature_df.iloc[i].get("away_rank_pts", 1500)
            rank_gap = abs(hp - ap)
            # 获取伤病信息调整
            player_data = {}
            player_path = os.path.join(DATA_DIR, "player_status.json")
            if os.path.exists(player_path):
                with open(player_path) as f:
                    player_data = json.load(f)
            home_injury = player_data.get(home, {}).get("severity", 0)
            away_injury = player_data.get(away, {}).get("severity", 0)
            injury_shift = (away_injury - home_injury) * 0.05  # 对方伤病越多，主队胜率越高

            if rank_gap > 150:
                # 大排名差距降低平局概率
                draw_reduction = min((rank_gap - 150) / 500.0 * 0.15, 0.15)
                new_d = max(ensemble_prob[i][1] - draw_reduction, 0.08)
                shift = ensemble_prob[i][1] - new_d
                ensemble_prob[i][1] = new_d
                if ensemble_prob[i][0] >= ensemble_prob[i][2]:
                    ensemble_prob[i][0] += shift * 0.7 + injury_shift
                    ensemble_prob[i][2] += shift * 0.3 - injury_shift
                else:
                    ensemble_prob[i][0] += shift * 0.3 + injury_shift
                    ensemble_prob[i][2] += shift * 0.7 - injury_shift
            elif rank_gap > 50 and injury_shift != 0:
                ensemble_prob[i][0] += injury_shift
                ensemble_prob[i][2] -= injury_shift

            # 重新归一化
            total = sum(ensemble_prob[i])
            ensemble_prob[i] = ensemble_prob[i] / total

        ensemble_pred = np.argmax(ensemble_prob, axis=1)

    results = feature_df[["date", "home_team", "away_team"]].copy()
    results["pred_H"] = ensemble_prob[:, 0]
    results["pred_D"] = ensemble_prob[:, 1]
    results["pred_A"] = ensemble_prob[:, 2]
    results["pred_result"] = [RESULT_INV[p] for p in ensemble_pred]

    # 智能标签处理：只有当最高概率和第二概率非常接近时才标平局
    smart_labels = []
    for i in range(len(results)):
        probs = ensemble_prob[i]
        sorted_p = sorted(probs, reverse=True)
        # 只有在前两名差异极小且平局概率本身足够高时才标平局
        if sorted_p[0] - sorted_p[1] < 0.02 and probs[1] > 0.30:
            smart_labels.append("D")
        else:
            smart_labels.append(RESULT_INV[ensemble_pred[i]])

    results["pred_label"] = [{"H": "主胜", "D": "平局", "A": "客胜"}[r] for r in smart_labels]
    results["confidence"] = np.max(ensemble_prob, axis=1)
    results["is_finished"] = feature_df["is_finished"].values if "is_finished" in feature_df.columns else False
    results["fair_odds_H"] = 1.0 / np.clip(results["pred_H"], 0.01, 0.99)
    results["fair_odds_D"] = 1.0 / np.clip(results["pred_D"], 0.01, 0.99)
    results["fair_odds_A"] = 1.0 / np.clip(results["pred_A"], 0.01, 0.99)

    # 比分预测：优先使用真实赔率 O/U 数据 + Poisson 分布
    # 后备使用 ELO 期望进球
    h2h_keys = ["pred_H", "pred_D", "pred_A"]
    has_h2h = all(c in results.columns for c in h2h_keys)
    try:
        from data_collector import get_score_prediction
    except ImportError:
        get_score_prediction = None
    home_scores, away_scores = [], []
    for i, row in results.iterrows():
        h2h = (row["pred_H"], row["pred_D"], row["pred_A"]) if has_h2h else None
        hg, ag, sp, exph, expa = (1, 0, 0, 1.4, 0.8)
        if get_score_prediction is not None:
            try:
                hg, ag, sp, exph, expa = get_score_prediction(row["home_team"], row["away_team"], h2h, result=row["pred_result"])
            except:
                pass
        else:
            # 后备：用 ELO
            hp = feature_df["home_rank_pts"].iloc[i] if "home_rank_pts" in feature_df.columns else 1500
            ap = feature_df["away_rank_pts"].iloc[i] if "away_rank_pts" in feature_df.columns else 1500
            exp_h, exp_a = exp_goals_from_elo(int(hp), int(ap))
            from data_collector import predict_score_poisson
            (hg, ag), sp = predict_score_poisson(exp_h, exp_a)
        home_scores.append(hg)
        away_scores.append(ag)
    results["pred_home_score"] = home_scores
    results["pred_away_score"] = away_scores

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


# ─── 混合模型参数优化（RL式网格搜索） ─────────────────────────


def load_hybrid_params():
    """加载学习到的混合模型参数，不存在时返回默认值"""
    params_path = os.path.join(DATA_DIR, "hybrid_params.json")
    if os.path.exists(params_path):
        with open(params_path) as f:
            return json.load(f)
    return {"top5_strength": 0.15, "close_threshold": 100, "elo_min_weight": 0.4}


def optimize_hybrid_params(finished_matches, top5_data, rankings_df, param_grid=None, save=True):
    """用已完赛比赛进行网格搜索，找到最优混合模型参数

    这是"强化学习"的核心：每个参数组合是一个"动作"，
    在已完赛比赛上的预测准确率是"奖励"。
    随着比赛增多，参数会持续优化。

    参数:
        finished_matches: dict from finished_matches.json
        top5_data: dict of team -> top5 ratio
        rankings_df: 排名DataFrame
        param_grid: 自定义参数网格
        save: 是否保存结果
    返回:
        {"best_params": ..., "best_score": ..., "history": [...]}
    """
    if param_grid is None:
        param_grid = {
            "top5_strength": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "close_threshold": [50, 75, 100, 125, 150, 175, 200],
            "elo_min_weight": [0.3, 0.4, 0.5, 0.6, 0.7],
        }

    # 构建排名查询（含队名映射）
    TEAM_MAP = {"United States": "USA", "Czechia": "Czech Republic"}
    rank_lookup = {}
    if rankings_df is not None:
        for _, row in rankings_df.iterrows():
            rank_lookup[row["Team"]] = int(row["RankPoints"])
        for match_name, mapped_name in TEAM_MAP.items():
            if mapped_name in rank_lookup:
                rank_lookup[match_name] = rank_lookup[mapped_name]

    # 准备已完赛比赛数据
    match_data = []
    for mid, info in finished_matches.items():
        home = info["home_team"]
        away = info["away_team"]
        actual = info.get("actual_result", "")
        if not actual:
            continue
        hp = rank_lookup.get(home, 1500)
        ap = rank_lookup.get(away, 1500)
        t5h = top5_data.get(home, 0.10)
        t5a = top5_data.get(away, 0.10)
        match_data.append({
            "home": home, "away": away,
            "hp": hp, "ap": ap,
            "t5h": t5h, "t5a": t5a,
            "actual": actual,
        })

    if not match_data:
        log.warning("无已完赛比赛可供优化，使用默认参数")
        return {"best_params": {"top5_strength": 0.15, "close_threshold": 100, "elo_min_weight": 0.4},
                "best_score": 0, "history": []}

    # 为每个匹配写日志
    log.info(f"混合模型优化: {len(match_data)} 场已完赛比赛")
    for md in match_data:
        log.info(f"  {md['home']} vs {md['away']}: rank_diff={md['hp']-md['ap']:+d}, "
                 f"top5_diff={md['t5h']-md['t5a']:+.2f}, actual={md['actual']}")

    best_score = -999
    best_params = None
    history = []

    # 全网格搜索
    for ts in param_grid["top5_strength"]:
        for ct in param_grid["close_threshold"]:
            for emw in param_grid["elo_min_weight"]:
                correct = 0
                logloss_sum = 0.0
                for md in match_data:
                    ph, pd_val, pa = hybrid_elo_prob(
                        md["hp"], md["ap"], md["t5h"], md["t5a"],
                        top5_strength=ts, close_threshold=ct, elo_min_weight=emw,
                    )
                    actual_map = {"H": 0, "D": 1, "A": 2}
                    pred_idx = 0 if ph > pd_val and ph > pa else (1 if pd_val > pa else 2)
                    actual_idx = actual_map.get(md["actual"], 1)
                    if pred_idx == actual_idx:
                        correct += 1
                    probs = [ph, pd_val, pa]
                    actual_one_hot = [0, 0, 0]
                    actual_one_hot[actual_idx] = 1.0
                    clipped = np.clip(probs, 1e-10, 1 - 1e-10)
                    logloss_sum += -sum(actual_one_hot[i] * math.log(clipped[i]) for i in range(3))

                avg_acc = correct / len(match_data)
                avg_ll = logloss_sum / len(match_data)
                combined = avg_acc - avg_ll * 0.1

                history.append({
                    "params": {"top5_strength": ts, "close_threshold": ct, "elo_min_weight": emw},
                    "accuracy": round(avg_acc, 4),
                    "log_loss": round(avg_ll, 4),
                    "combined_score": round(combined, 4),
                })

                if combined > best_score:
                    best_score = combined
                    best_params = {"top5_strength": ts, "close_threshold": ct, "elo_min_weight": emw}

    best_params["score"] = round(best_score, 4)
    history_sorted = sorted(history, key=lambda x: -x["combined_score"])

    log.info(f"最优混合参数: strength={best_params['top5_strength']}, "
             f"threshold={best_params['close_threshold']}, "
             f"min_weight={best_params['elo_min_weight']} "
             f"(score={best_score:.4f})")
    log.info(f"  准确率={history_sorted[0]['accuracy']:.1%}, "
             f"log_loss={history_sorted[0]['log_loss']:.4f}")

    if save:
        params_path = os.path.join(DATA_DIR, "hybrid_params.json")
        with open(params_path, "w") as f:
            json.dump(best_params, f, indent=2)

        history_path = os.path.join(MODEL_DIR, "hybrid_optimization_history.json")
        with open(history_path, "w") as f:
            json.dump({
                "n_matches": len(match_data),
                "n_param_combinations": len(history),
                "best_params": best_params,
                "top_results": history_sorted[:20],
                "all_results": history_sorted,
                "match_data": [{"home": md["home"], "away": md["away"],
                                "actual": md["actual"]} for md in match_data],
                "optimized_at": datetime.now().isoformat(),
            }, f, indent=2)
        log.info(f"混合参数已保存: {params_path}")

    return {"best_params": best_params, "best_score": best_score, "history": history}


if __name__ == "__main__":
    feat_path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            feature_df = pickle.load(f)
        log.info(f"加载特征矩阵: {feature_df.shape}")
        run_backtest(feature_df)
    else:
        log.error("请先运行特征工程")
