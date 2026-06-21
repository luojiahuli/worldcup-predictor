"""
main.py - 世界杯2026预测系统主入口
一键完成：数据采集 → 特征工程 → 模型训练 → 预测 → 多元分析 → 可视化
"""
import os, sys, logging, subprocess, pickle, json
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
VIZ_DIR = os.path.join(BASE_DIR, "..", "viz_output")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)


def step_data_collection():
    log.info("\n" + "=" * 60)
    log.info("步骤 1/6: 数据采集")
    log.info("=" * 60)

    wc_path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
    if os.path.exists(wc_path):
        log.info("数据缓存已存在")
        # 确保参赛队列表存在
        teams_path = os.path.join(DATA_DIR, "worldcup_teams.csv")
        if not os.path.exists(teams_path):
            with open(wc_path, "rb") as f:
                wc = pickle.load(f)
            valid = wc[wc["Home"].notna() & wc["Away"].notna()]
            actual_teams = sorted(set(valid["Home"].unique()) | set(valid["Away"].unique()))
            pd.DataFrame({"Team": actual_teams}).to_csv(teams_path, index=False)
            log.info(f"生成参赛队列表: {len(actual_teams)} 支")
        return True

    from data_collector import collect_all
    wc_data, rankings = collect_all()
    return wc_data is not None


def step_feature_engineering():
    log.info("\n" + "=" * 60)
    log.info("步骤 2/6: 特征工程")
    log.info("=" * 60)

    feat_path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    if os.path.exists(feat_path):
        log.info("特征矩阵已缓存")
        return True

    from data_collector import load_data, load_rankings
    from feature_engineer import build_feature_matrix

    wc = load_data()
    rankings = load_rankings()
    if wc is None or rankings is None:
        log.error("数据不完整")
        return False

    df = build_feature_matrix(wc, rankings)
    return df is not None


def step_model_training():
    log.info("\n" + "=" * 60)
    log.info("步骤 3/6: 模型训练与回测")
    log.info("=" * 60)

    model_path = os.path.join(MODEL_DIR, "wc_model_package.pkl")
    if os.path.exists(model_path):
        log.info("模型已存在 (如需重新训练请删除 models/ 目录)")

    feat_path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    if not os.path.exists(feat_path):
        log.error("特征矩阵不存在")
        return False

    with open(feat_path, "rb") as f:
        feature_df = pickle.load(f)

    from model_trainer import run_backtest
    results, summary = run_backtest(feature_df)
    # 使用排名模型时results为None, summary有值
    return summary is not None


def step_prediction():
    log.info("\n" + "=" * 60)
    log.info("步骤 4/6: 预测2026世界杯")
    log.info("=" * 60)

    import pandas as pd
    from model_trainer import predict_matches

    feat_path = os.path.join(DATA_DIR, "wc_feature_matrix.pkl")
    model_path = os.path.join(MODEL_DIR, "wc_model_package.pkl")

    if not os.path.exists(feat_path) or not os.path.exists(model_path):
        log.warning("特征矩阵或模型不存在")
        return None

    with open(feat_path, "rb") as f:
        feature_df = pickle.load(f)

    results = predict_matches(feature_df)
    if results is None:
        return None

    # 保护已完赛比赛的旧预测结果（避免pipeline重跑时覆盖用户的预测/手动修正）
    # 优先从备份文件恢复，备份包含原始OLD模型预测
    pred_path = os.path.join(DATA_DIR, "wc_predictions.pkl")
    backup_path = os.path.join(DATA_DIR, "wc_predictions_old_backup.pkl")
    source_path = backup_path if os.path.exists(backup_path) else pred_path
    if os.path.exists(source_path):
        try:
            with open(source_path, "rb") as f:
                old_preds = pickle.load(f)
            if isinstance(old_preds, pd.DataFrame) and "is_finished" in old_preds.columns:
                # 确保 results 包含所有需要的列
                for col in ['actual_home_score', 'actual_away_score', 'score_correct', 'correct']:
                    if col not in results.columns:
                        results[col] = None if col != 'score_correct' and col != 'correct' else False
                preserved = 0
                preserve_cols = ['pred_H', 'pred_D', 'pred_A', 'pred_result', 'pred_label',
                                 'confidence', 'fair_odds_H', 'fair_odds_D', 'fair_odds_A',
                                 'pred_home_score', 'pred_away_score', 'value_score',
                                 'actual_home_score', 'actual_away_score', 'actual_result',
                                 'correct', 'score_correct']
                for idx, row in results.iterrows():
                    if row.get("is_finished") == True or row.get("is_finished") == 1:
                        old_match = old_preds[
                            (old_preds["home_team"] == row["home_team"]) &
                            (old_preds["away_team"] == row["away_team"])
                        ]
                        if len(old_match) > 0:
                            old_row = old_match.iloc[0]
                            for col in preserve_cols:
                                if col in old_row.index and pd.notna(old_row[col]):
                                    results.at[idx, col] = old_row[col]
                            preserved += 1
                if preserved > 0:
                    source_name = "备份" if source_path == backup_path else "当前"
                    log.info(f"已从{source_name}文件保留 {preserved} 场已完赛比赛的旧预测结果")
        except Exception as e:
            log.warning(f"保留旧预测失败: {e}")

    # 从历史完赛数据补全 actual_home_score/actual_away_score（备份可能缺失）
    all_wc_path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
    if os.path.exists(all_wc_path):
        try:
            with open(all_wc_path, "rb") as f:
                all_wc = pickle.load(f)
            for col in ['actual_home_score', 'actual_away_score', 'score_correct']:
                if col not in results.columns:
                    results[col] = None if col != 'score_correct' else False
            filled = 0
            for idx, row in results.iterrows():
                if row.get("is_finished") == True and (pd.isna(row.get("actual_home_score")) or pd.isna(row.get("actual_away_score"))):
                    h, a = row["home_team"], row["away_team"]
                    actual = all_wc[(all_wc["Home"] == h) & (all_wc["Away"] == a) & (all_wc["HomeGoals"] >= 0)]
                    if len(actual) > 0:
                        ah = int(actual.iloc[0]["HomeGoals"])
                        aa = int(actual.iloc[0]["AwayGoals"])
                        results.at[idx, "actual_home_score"] = ah
                        results.at[idx, "actual_away_score"] = aa
                        actual_r = "H" if ah > aa else ("D" if ah == aa else "A")
                        results.at[idx, "actual_result"] = actual_r
                        results.at[idx, "correct"] = (row["pred_result"] == actual_r)
                        results.at[idx, "score_correct"] = (row["pred_home_score"] == ah and row["pred_away_score"] == aa)
                        filled += 1
            if filled > 0:
                log.info(f"从历史数据补全 {filled} 场已完赛比赛的实际比分")
        except Exception as e:
            log.warning(f"补全实际比分失败: {e}")

    # 合并历史完赛数据（防止 pipeline 覆盖手动更新的比赛结果）
    finished_path = os.path.join(DATA_DIR, "finished_matches.json")
    if os.path.exists(finished_path):
        with open(finished_path) as f:
            finished_data = json.load(f)
        merged = 0
        for mid, info in finished_data.items():
            mask = (
                (results["home_team"] == info["home_team"])
                & (results["away_team"] == info["away_team"])
                & (results["is_finished"] == False)
            )
            if mask.any():
                results.loc[mask, "actual_home_score"] = info["actual_home_score"]
                results.loc[mask, "actual_away_score"] = info["actual_away_score"]
                results.loc[mask, "actual_result"] = info["actual_result"]
                results.loc[mask, "is_finished"] = True
                ph = results.loc[mask, "pred_home_score"].values[0]
                pa = results.loc[mask, "pred_away_score"].values[0]
                correct = int(ph == info["actual_home_score"] and pa == info["actual_away_score"])
                if "correct" not in results.columns:
                    results["correct"] = False
                results.loc[mask, "correct"] = bool(correct)
                merged += 1
        if merged > 0:
            log.info(f"已合并 {merged} 场历史完赛数据")

    # 保存预测结果
    with open(os.path.join(DATA_DIR, "wc_predictions.pkl"), "wb") as f:
        pickle.dump(results, f)

    # 输出预测
    unfinished = results[results["is_finished"] == False]
    if len(unfinished) > 0:
        log.info(f"\n2026世界杯预测 ({len(unfinished)} 场比赛):")
        for _, row in unfinished.iterrows():
            log.info(f"  {str(row['home_team'])[:25]:25s} vs {str(row['away_team'])[:25]:25s} "
                    f"→ {row['pred_label']} "
                    f"(H:{float(row['pred_H'])*100:.0f}% D:{float(row['pred_D'])*100:.0f}% A:{float(row['pred_A'])*100:.0f}%) "
                    f"置信度{float(row['confidence'])*100:.0f}%")

        # 高置信度
        high_conf = unfinished[unfinished["confidence"] >= 0.50]
        if len(high_conf) > 0:
            log.info(f"\n高置信度预测 ({len(high_conf)} 场, 置信度≥50%):")
            for _, row in high_conf.iterrows():
                log.info(f"  {str(row['home_team'])[:25]:25s} vs {str(row['away_team'])[:25]:25s} → {row['pred_label']} "
                        f"({float(row['confidence'])*100:.0f}%)")
    else:
        log.info("当前无未完成比赛")

    return results


def step_media_analysis():
    log.info("\n" + "=" * 60)
    log.info("步骤 5/6: 多元分析")
    log.info("=" * 60)

    # Ensure teams CSV exists
    teams_csv_path = os.path.join(DATA_DIR, "worldcup_teams.csv")
    if not os.path.exists(teams_csv_path):
        import pickle
        wc_path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
        if os.path.exists(wc_path):
            with open(wc_path, "rb") as f:
                wc = pickle.load(f)
            valid = wc[wc["Home"].notna() & wc["Away"].notna()]
            actual_teams = sorted(set(valid["Home"].unique()) | set(valid["Away"].unique()))
            pd.DataFrame({"Team": actual_teams}).to_csv(teams_csv_path, index=False)
            log.info(f"生成参赛队列表: {len(actual_teams)} 支")

    # 5.1 媒体情感
    log.info("\n--- 5.1 媒体情感分析 ---")
    media_path = os.path.join(DATA_DIR, "media_sentiment.csv")
    if not os.path.exists(media_path):
        from media_analyzer import analyze_all_teams
        teams_csv = pd.read_csv(os.path.join(DATA_DIR, "worldcup_teams.csv"))
        analyze_all_teams(teams_csv["Team"].tolist())
    else:
        log.info("媒体情感数据已存在")

    # 5.2 社交热度
    log.info("\n--- 5.2 社交热度分析 ---")
    social_path = os.path.join(DATA_DIR, "social_heat.csv")
    if not os.path.exists(social_path):
        from social_analyzer import analyze_social_heat
        teams_csv = pd.read_csv(os.path.join(DATA_DIR, "worldcup_teams.csv"))
        analyze_social_heat(teams_csv["Team"].tolist())
    else:
        log.info("社交热度数据已存在")

    # 5.3 赔率分析
    log.info("\n--- 5.3 赔率分析 ---")
    odds_path = os.path.join(DATA_DIR, "odds_analysis.csv")
    if not os.path.exists(odds_path):
        from odds_analyzer import analyze_all_odds
        analyze_all_odds()
    else:
        log.info("赔率数据已存在")


def step_visualization():
    log.info("\n" + "=" * 60)
    log.info("步骤 6/6: 生成可视化仪表盘")
    log.info("=" * 60)

    from visualizer import load_all_data, generate_dashboard
    data = load_all_data()
    if not data:
        log.error("无数据")
        return None

    html_path = generate_dashboard(data)

    # 复制到 viz_output
    import shutil
    viz_path = os.path.join(VIZ_DIR, os.path.basename(html_path))
    shutil.copy2(html_path, viz_path)
    log.info(f"已复制到: {viz_path}")

    return html_path


def step_hybrid_optimization():
    """步骤3.5: 用已完赛比赛优化混合模型参数（RL式网格搜索）"""
    log.info("\n" + "=" * 60)
    log.info("步骤 3.5/7: 混合模型参数优化（RL网格搜索）")
    log.info("=" * 60)

    from model_trainer import optimize_hybrid_params, load_hybrid_params

    finished_path = os.path.join(DATA_DIR, "finished_matches.json")
    top5_path = os.path.join(DATA_DIR, "team_top5.json")
    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")

    if not all(os.path.exists(p) for p in [finished_path, top5_path, rank_path]):
        log.warning("缺少数据文件，跳过混合优化")
        return None

    with open(finished_path) as f:
        finished_matches = json.load(f)
    with open(top5_path) as f:
        top5_data = json.load(f)
    rankings_df = pd.read_csv(rank_path)

    if len(finished_matches) == 0:
        log.info("暂无已完赛比赛，使用默认混合参数")
        params = load_hybrid_params()
        log.info(f"  默认参数: strength={params['top5_strength']}, "
                 f"threshold={params['close_threshold']}, min_weight={params['elo_min_weight']}")
        return params

    result = optimize_hybrid_params(finished_matches, top5_data, rankings_df)
    return result


def run_pipeline():
    log.info("🏆" * 30)
    log.info("🏆  世界杯2026预测系统 v1.0")
    log.info(f"🏆  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("🏆" * 30)

    if not step_data_collection():
        log.error("数据采集失败")
        return

    if not step_feature_engineering():
        log.error("特征工程失败")
        return

    if not step_model_training():
        log.error("模型训练失败")
        return

    step_hybrid_optimization()

    predictions = step_prediction()
    step_media_analysis()

    html_path = step_visualization()

    if html_path:
        log.info(f"\n{'='*60}")
        log.info(f"✅ 完成! 仪表盘: {html_path}")
        log.info(f"{'='*60}")
        try:
            subprocess.run(["open", html_path], check=True)
        except:
            log.info(f"请在浏览器中打开: {html_path}")
    else:
        log.error("可视化生成失败")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--collect-only":
        step_data_collection()
    elif len(sys.argv) > 1 and sys.argv[1] == "--predict-only":
        step_prediction()
        step_visualization()
    else:
        run_pipeline()
