"""
main.py - 世界杯2026预测系统主入口
一键完成：数据采集 → 特征工程 → 模型训练 → 预测 → 多元分析 → 可视化
"""
import os, sys, logging, subprocess, pickle
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
