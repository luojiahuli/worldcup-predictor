"""
visualizer.py - 世界杯2026预测可视化仪表盘
包含：预测结果、媒体情感、社交热度、赔率变化
"""
import pandas as pd
import numpy as np
import os, json, logging, pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULT_COLORS = {"H": "#ef4444", "D": "#f59e0b", "A": "#3b82f6"}
RESULT_NAMES = {"H": "主胜", "D": "平局", "A": "客胜"}
AGENT_COLORS = {"稳健派": "#22c55e", "激进派": "#ef4444", "价值派": "#3b82f6", "防守派": "#8b5cf6", "数据派": "#f59e0b", "爆冷派": "#ec4899"}


def load_all_data():
    """加载所有分析数据"""
    data = {}

    # 预测结果
    pred_path = os.path.join(DATA_DIR, "wc_predictions.pkl")
    if os.path.exists(pred_path):
        with open(pred_path, "rb") as f:
            data["predictions"] = pickle.load(f)

    # 回测结果
    bt_path = os.path.join(DATA_DIR, "wc_backtest_results.pkl")
    if os.path.exists(bt_path):
        with open(bt_path, "rb") as f:
            data["backtest"] = pickle.load(f)

    # 媒体情感
    media_path = os.path.join(DATA_DIR, "media_sentiment.csv")
    if os.path.exists(media_path):
        data["media"] = pd.read_csv(media_path)

    # 社交热度
    social_path = os.path.join(DATA_DIR, "social_heat.csv")
    if os.path.exists(social_path):
        data["social"] = pd.read_csv(social_path)

    # 赔率分析
    odds_path = os.path.join(DATA_DIR, "odds_analysis.csv")
    if os.path.exists(odds_path):
        data["odds"] = pd.read_csv(odds_path)

    # 赔率历史
    history_path = os.path.join(DATA_DIR, "odds_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            data["odds_history"] = json.load(f)

    # 社交趋势
    trend_path = os.path.join(DATA_DIR, "social_trends.json")
    if os.path.exists(trend_path):
        with open(trend_path, "r") as f:
            data["social_trends"] = json.load(f)

    # 智能体预测
    agent_path = os.path.join(DATA_DIR, "agent_predictions.json")
    if os.path.exists(agent_path):
        with open(agent_path, "r") as f:
            data["agent_predictions"] = json.load(f)

    # 智能体排行榜
    lb_path = os.path.join(DATA_DIR, "agent_leaderboard.json")
    if os.path.exists(lb_path):
        with open(lb_path, "r") as f:
            data["agent_leaderboard"] = json.load(f)

    # 智能体权重
    weights_path = os.path.join(DATA_DIR, "agent_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            data["agent_weights"] = json.load(f)

    # 真实赔率
    real_odds_path = os.path.join(DATA_DIR, "real_odds.json")
    if os.path.exists(real_odds_path):
        with open(real_odds_path, "r") as f:
            data["real_odds"] = json.load(f)

    return data


def build_prediction_rows(predictions):
    """构建预测数据表格行"""
    if predictions is None or len(predictions) == 0:
        return "[]"

    unfinished = predictions[predictions["is_finished"] == False]
    if len(unfinished) == 0:
        unfinished = predictions

    rows = []
    for _, row in unfinished.iterrows():
        rows.append(json.dumps({
            "date": str(row.get("date", ""))[:10],
            "home": row["home_team"],
            "away": row["away_team"],
            "pred": row.get("pred_label", ""),
            "pred_r": row.get("pred_result", ""),
            "ph": round(float(row["pred_H"]) * 100, 1),
            "pd": round(float(row["pred_D"]) * 100, 1),
            "pa": round(float(row["pred_A"]) * 100, 1),
            "conf": round(float(row["confidence"]) * 100, 1),
            "oh": round(float(row.get("fair_odds_H", 2.0)), 2),
            "od": round(float(row.get("fair_odds_D", 3.0)), 2),
            "oa": round(float(row.get("fair_odds_A", 3.0)), 2),
            "hs": int(row.get("pred_home_score", 0)),
            "as": int(row.get("pred_away_score", 0)),
        }, ensure_ascii=False))
    return ",\n".join(rows)


def build_backtest_rows(backtest_results):
    """构建回测数据行"""
    if not backtest_results:
        return "[]"

    all_preds = []
    for r in backtest_results:
        all_preds.append(r["predictions"])
    df = pd.concat(all_preds, ignore_index=True)
    if "date" in df.columns:
        df = df.sort_values("date").tail(100)

    rows = []
    for _, row in df.iterrows():
        rows.append(json.dumps({
            "date": str(row.get("date", ""))[:10],
            "home": row["home_team"],
            "away": row["away_team"],
            "result": row["result"],
            "pred": row["pred_r"],
            "ph": round(float(row["pred_H"]) * 100, 1),
            "pd": round(float(row["pred_D"]) * 100, 1),
            "pa": round(float(row["pred_A"]) * 100, 1),
        }, ensure_ascii=False))
    return ",\n".join(rows)


def generate_dashboard(data):
    """生成世界杯预测仪表盘HTML"""
    log.info("生成世界杯2026预测仪表盘...")

    today_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    predictions = data.get("predictions")
    backtest = data.get("backtest")
    media_df = data.get("media")
    social_df = data.get("social")
    odds_df = data.get("odds")
    odds_history = data.get("odds_history", {})
    social_trends = data.get("social_trends", {})
    real_odds_data = data.get("real_odds", {})
    real_odds_matches = real_odds_data.get("matches", {}) if real_odds_data else {}

    # === 计算指标 ===
    avg_acc = 0
    n_matches = 0
    if backtest:
        avg_acc = np.mean([r["metrics"]["ensemble"]["accuracy"] for r in backtest])
        n_matches = sum(len(r["predictions"]) for r in backtest)

    # 预测数据
    high_conf_picks = []
    if predictions is not None and len(predictions) > 0:
        unfinished = predictions[predictions["is_finished"] == False]
        if len(unfinished) > 0:
            top = unfinished.nlargest(min(20, len(unfinished)), "confidence")
            for _, row in top.iterrows():
                high_conf_picks.append({
                    "date": str(row.get("date", ""))[:10],
                    "home": row["home_team"],
                    "away": row["away_team"],
                    "pred": row.get("pred_label", ""),
                    "pred_r": row.get("pred_result", ""),
                    "pH": f"{float(row['pred_H'])*100:.1f}",
                    "pD": f"{float(row['pred_D'])*100:.1f}",
                    "pA": f"{float(row['pred_A'])*100:.1f}",
                    "conf": f"{float(row['confidence'])*100:.1f}",
                    "oH": f"{float(row.get('fair_odds_H', 2.0)):.2f}",
                    "oD": f"{float(row.get('fair_odds_D', 3.0)):.2f}",
                    "oA": f"{float(row.get('fair_odds_A', 3.0)):.2f}",
                    "hs": int(row.get("pred_home_score", 0)),
                    "as": int(row.get("pred_away_score", 0)),
                    "color": RESULT_COLORS.get(row.get("pred_result", ""), "#888"),
                })

    # === 已完成比赛复盘 ===
    finished_review_rows = ""
    finished_count = 0
    correct_count = 0
    if predictions is not None and len(predictions) > 0:
        finished = predictions[predictions["is_finished"] == True]
        finished_count = len(finished)
        if finished_count > 0:
            correct_count = int(finished["correct"].sum())
            review_rows = []
            # Sort by date descending so latest match is first
            finished_sorted = finished.sort_values("date", ascending=False)
            for idx, (_, row) in enumerate(finished_sorted.iterrows()):
                hs = int(row.get("pred_home_score", 0))
                as_ = int(row.get("pred_away_score", 0))
                ah = int(row.get("actual_home_score", 0))
                aa = int(row.get("actual_away_score", 0))
                is_correct = row.get("correct", False)
                is_result_correct = row.get("actual_result", "") == row.get("pred_result", "")
                mark = "✅" if is_correct else ("⚠️" if is_result_correct else "❌")
                date_str = str(row["date"])[:10]
                home = row["home_team"]
                away = row["away_team"]
                pred_label = row.get("pred_label", "")
                dir_color = "var(--green)" if is_result_correct else "#ef4444"
                result_label = {'H': '主胜', 'D': '平局', 'A': '客胜'}.get(row.get("actual_result", ""), "-")
                # Latest match: always expanded; older matches: collapsible
                if idx == 0:
                    review_rows.append(f'''<div class="fmc" style="border:1px solid var(--border);border-left:3px solid {dir_color};border-radius:8px;margin-bottom:8px;overflow:hidden;background:var(--card)">
  <div class="fmc-h" onclick="var n=this.nextElementSibling;n.style.display=n.style.display==='none'?'block':'none'" style="display:flex;align-items:center;gap:10px;padding:10px 14px;cursor:pointer;user-select:none">
    <span style="font-weight:600;font-size:13px">{home}</span>
    <span style="color:var(--text3);font-size:11px">vs</span>
    <span style="font-weight:600;font-size:13px">{away}</span>
    <span style="font-size:10px;color:var(--text3);background:var(--bg2);padding:1px 6px;border-radius:4px">{date_str}</span>
    <span style="margin-left:auto;display:flex;align-items:center;gap:10px">
      <span style="font-size:16px;font-weight:700;color:var(--green)">{ah}-{aa}</span>
      <span>{mark}</span>
      <span style="font-size:10px;color:var(--text3)">▼</span>
    </span>
  </div>
  <div class="fmc-b" style="display:none;padding:10px 14px;border-top:1px solid var(--border);font-size:12px">
    <table style="width:100%">
      <tr><td style="color:var(--text3);padding:3px 8px">预测结果</td><td style="padding:3px 8px;color:{dir_color};font-weight:600">{pred_label}</td>
          <td style="color:var(--text3);padding:3px 8px">实际结果</td><td style="padding:3px 8px;font-weight:600">{result_label}</td></tr>
      <tr><td style="color:var(--text3);padding:3px 8px">预测比分</td><td style="padding:3px 8px;font-weight:600">{hs}-{as_}</td>
          <td style="color:var(--text3);padding:3px 8px">实际比分</td><td style="padding:3px 8px;font-weight:600;color:var(--green)">{ah}-{aa}</td></tr>
    </table>
  </div>
</div>''')
                else:
                    review_rows.append(f'''<details style="margin-bottom:6px" {"open" if idx == 1 else ""}>
  <summary style="display:flex;align-items:center;gap:10px;padding:8px 14px;background:var(--card);border:1px solid var(--border);border-radius:8px;cursor:pointer;font-size:12px;list-style:none">
    <span style="font-weight:500;font-size:13px">{home}</span>
    <span style="color:var(--text3)">vs</span>
    <span style="font-weight:500;font-size:13px">{away}</span>
    <span style="font-size:10px;color:var(--text3)">{date_str}</span>
    <span style="margin-left:auto;display:flex;align-items:center;gap:10px">
      <span style="font-size:14px;font-weight:600;color:var(--green)">{ah}-{aa}</span>
      <span>{mark}</span>
    </span>
  </summary>
  <div style="padding:10px 14px;border:1px solid var(--border);border-top:none;border-radius:0 0 8px 8px;font-size:12px;margin-top:-1px">
    <table style="width:100%">
      <tr><td style="color:var(--text3);padding:3px 8px">预测结果</td><td style="padding:3px 8px;color:{dir_color};font-weight:600">{pred_label}</td>
          <td style="color:var(--text3);padding:3px 8px">实际结果</td><td style="padding:3px 8px;font-weight:600">{result_label}</td></tr>
      <tr><td style="color:var(--text3);padding:3px 8px">预测比分</td><td style="padding:3px 8px;font-weight:600">{hs}-{as_}</td>
          <td style="color:var(--text3);padding:3px 8px">实际比分</td><td style="padding:3px 8px;font-weight:600;color:var(--green)">{ah}-{aa}</td></tr>
    </table>
  </div>
</details>''')
            finished_review_rows = "\n".join(review_rows)

    # === 最新比赛日分析 ===
    matchday_date_str = ""
    matchday_html_rows = ""
    matchday_dates = []
    if predictions is not None and len(predictions) > 0:
        unfinished = predictions[predictions["is_finished"] == False].copy()
        dates = pd.to_datetime(unfinished["date"])
        today = pd.Timestamp.now().normalize()
        upcoming = unfinished[dates >= today].sort_values("date")
        if len(upcoming) > 0:
            # Show matches from today through the next 2 matchdays
            md_date = upcoming["date"].iloc[0]
            cutoff = pd.Timestamp(md_date) + pd.Timedelta(days=2)
            md_matches = upcoming[pd.to_datetime(upcoming["date"]) < cutoff]
            matchday_dates = md_matches["date"].unique()
            # Use first date for matching agent data; display all dates
            matchday_date_str = str(md_date)[:10]
            if len(matchday_dates) > 1:
                last_date = str(matchday_dates[-1])[:10]
                matchday_date_str = f"{matchday_date_str} ~ {last_date}"
            rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
            rankings_df = None
            if os.path.exists(rank_path):
                rankings_df = pd.read_csv(rank_path)
            rows_html = []
            for _, row in md_matches.iterrows():
                home, away = row["home_team"], row["away_team"]
                pred_r = row["pred_result"]
                conf = float(row["confidence"])
                pred_h, pred_d, pred_a = float(row["pred_H"]), float(row["pred_D"]), float(row["pred_A"])
                hs = int(row.get("pred_home_score", 0))
                as_ = int(row.get("pred_away_score", 0))
                oH = float(row.get("fair_odds_H", 2.0))
                oD = float(row.get("fair_odds_D", 3.0))
                oA = float(row.get("fair_odds_A", 3.0))
                reasons = []
                if conf >= 0.50: reasons.append(f"高置信度({conf*100:.0f}%)")
                elif conf >= 0.40: reasons.append(f"模型倾向({conf*100:.0f}%)")
                elif conf >= 0.35: reasons.append("置信度一般")
                else: reasons.append("不确定性强")
                if rankings_df is not None:
                    h_rank = rankings_df[rankings_df["Team"] == home]
                    a_rank = rankings_df[rankings_df["Team"] == away]
                    if len(h_rank) > 0 and len(a_rank) > 0:
                        hr, ar = int(h_rank["Rank"].values[0]), int(a_rank["Rank"].values[0])
                        diff = abs(hr - ar)
                        if diff >= 30: reasons.append(f"排名差距{diff}位")
                        better = home if hr < ar else away
                        reasons.append(f"{better}排名占优")
                if pred_r == "H" and oH > 1:
                    if pred_h > 1.0/oH + 0.05: reasons.append("主胜赔率有空间")
                elif pred_r == "A" and oA > 1:
                    if pred_a > 1.0/oA + 0.05: reasons.append("客胜赔率有空间")
                if pred_r == "H": kelly = (pred_h * oH - 1) / (oH - 1) if oH > 1 else 0
                elif pred_r == "A": kelly = (pred_a * oA - 1) / (oA - 1) if oA > 1 else 0
                else: kelly = (pred_d * oD - 1) / (oD - 1) if oD > 1 else 0
                kelly = max(0, kelly)
                if kelly > 0.15: bet = f"推荐{row['pred_label']}，仓位10-15%"
                elif kelly > 0.08: bet = f"可投{row['pred_label']}，仓位5-10%"
                elif kelly > 0.03: bet = f"小注{row['pred_label']}，仓位3-5%"
                elif conf >= 0.40: bet = f"小额娱乐{row['pred_label']}"
                else: bet = "不推荐，置信度偏低"
                color = RESULT_COLORS.get(pred_r, "#888")
                bet_color = "#22c55e" if kelly > 0.08 else "#f59e0b" if kelly > 0 else "#5b6380"
                # 真实赔率对比
                match_key = f"{home}_vs_{away}"
                real = real_odds_matches.get(match_key)
                odds_display = f"{oH:.2f}/{oD:.2f}/{oA:.2f}"
                if real:
                    rH, rD, rA = real["odds_H"], real["odds_D"], real["odds_A"]
                    odds_display = f"模型{oH:.2f}/{oD:.2f}/{oA:.2f}<br><span style=font-size:9px;color:var(--accent2)>市场{rH:.2f}/{rD:.2f}/{rA:.2f}</span>"
                rows_html.append(
                    f'<tr><td>{str(row["date"])[:10]}</td>'
                    f'<td style="font-weight:600">{home}</td>'
                    f'<td style="font-weight:600">{away}</td>'
                    f'<td style="color:{color};font-weight:600">{row["pred_label"]}</td>'
                    f'<td style="font-weight:600;color:var(--text2)">{hs}-{as_}</td>'
                    f'<td>{conf*100:.0f}%</td>'
                    f'<td>{odds_display}</td>'
                    f'<td class="reason-cell">{"；".join(reasons)}</td>'
                    f'<td class="bet-cell" style="color:{bet_color}">{bet}</td></tr>'
                )
            matchday_html_rows = "\n".join(rows_html)

    # === 智能体PK辩论 ===
    agent_pk_html = ""
    agent_leaderboard_rows = ""
    agent_predictions = data.get("agent_predictions")
    agent_leaderboard = data.get("agent_leaderboard")
    agent_weights_data = data.get("agent_weights")
    current_weights = agent_weights_data.get("weights", {}) if agent_weights_data else {}
    round_history = agent_weights_data.get("round_history", {}) if agent_weights_data else {}
    current_round_name = agent_weights_data.get("current_round", "") if agent_weights_data else ""
    round_acc = (agent_weights_data.get("agent_round_accuracy", {}).get(current_round_name, {})
                if agent_weights_data else {})

    # 排行榜
    if agent_leaderboard:
        lb_rows = []
        for entry in agent_leaderboard:
            dist = f"H{entry.get('H_pct',0)*100:.0f}/D{entry.get('D_pct',0)*100:.0f}/A{entry.get('A_pct',0)*100:.0f}"
            color = entry.get("color", "#888")
            name = entry["name"]
            weight = current_weights.get(name, 1.0)
            ra = round_acc.get(name, {})
            real_acc = ra.get("accuracy", 0)
            real_acc_str = f'{real_acc:.0%}' if real_acc > 0 and ra.get("total", 0) > 0 else "-"
            lb_rows.append(
                f'<tr>'
                f'<td style="color:{color};font-weight:600">#{entry["rank"]}</td>'
                f'<td style="color:{color};font-weight:600">{name}</td>'
                f'<td>{dist}</td>'
                f'<td>{entry.get("avg_confidence",0)*100:.0f}%</td>'
                f'<td>{real_acc_str}</td>'
                f'<td style="color:var(--accent2);font-weight:600">{weight:.2f}</td>'
                f'<td>{entry.get("total_score",0):.1f}</td>'
                f'</tr>'
            )
        agent_leaderboard_rows = "\n".join(lb_rows)

    # PK对比
    if agent_predictions and matchday_date_str:
        matches = agent_predictions.get("matches", {})
        matchday_dates_list = [str(d)[:10] for d in matchday_dates]
        md_matches_agent = [
            m for m in matches.values()
            if m.get("date", "")[:10] in matchday_dates_list
        ]
        if md_matches_agent:
            sections = []
            for m in md_matches_agent:
                agents = m.get("agents", {})
                consensus = m.get("consensus", {})
                consensus_result = consensus.get("consensus", "N/A")
                agreement = consensus.get("agreement_pct", 0)

                agent_rows = ""
                for agent_name in ["稳健派", "激进派", "价值派", "防守派", "数据派", "爆冷派"]:
                    if agent_name not in agents:
                        continue
                    a = agents[agent_name]
                    a_color = AGENT_COLORS.get(agent_name, "#888")
                    is_consensus = "✓" if a.get("result") == consensus_result else ""
                    agent_rows += (
                        f'<div class="ar" style="border-left-color:{a_color}">'
                        f'<div class="ar-h">'
                        f'<span class="ar-n" style="color:{a_color}">{agent_name}</span>'
                        f'<span class="ar-r" style="color:{RESULT_COLORS.get(a.get("result",""),"#888")}">'
                        f'{RESULT_NAMES.get(a.get("result",""),"")}</span>'
                        f'<span class="ar-s">{a.get("home_score",0)}-{a.get("away_score",0)}</span>'
                        f'<span class="ar-c">{a.get("confidence",0)*100:.0f}%</span>'
                        f'{f"<span class=ar-cons>共识</span>" if is_consensus else ""}'
                        f'</div>'
                        f'<div class="ar-reason">{a.get("reasoning","")}</div>'
                        f'</div>'
                    )

                # Consensus bar (with weighted display)
                is_weighted = consensus.get("weighted", False)
                weight_label = '<span style="font-size:10px;color:var(--text3);margin-left:6px">(加权)</span>' if is_weighted else ""
                h_c = consensus.get("H", 0)
                d_c = consensus.get("D", 0)
                a_c = consensus.get("A", 0)
                total_c = h_c + d_c + a_c or 1
                sections.append(
                    f'<div class="am">'
                    f'<div class="am-h">{m["home"]} <span style="color:var(--text3);font-weight:400">vs</span> {m["away"]}'
                    f'{weight_label}'
                    f'</div>'
                    f'<div class="am-cb">'
                    f'<span class="am-cb-s" style="width:{h_c/total_c*100}%;background:var(--home)">{h_c if h_c>0 else ""}</span>'
                    f'<span class="am-cb-s" style="width:{d_c/total_c*100}%;background:var(--draw)">{d_c if d_c>0 else ""}</span>'
                    f'<span class="am-cb-s" style="width:{a_c/total_c*100}%;background:var(--away)">{a_c if a_c>0 else ""}</span>'
                    f'</div>'
                    f'<div class="am-agents">{agent_rows}</div>'
                    f'</div>'
                )
            agent_pk_html = "\n".join(sections)

    # Weight history for ECharts
    weight_history_json = "[]"
    if round_history:
        rounds = sorted(round_history.keys())
        traces = []
        for agent_name in ["稳健派", "激进派", "价值派", "防守派", "数据派", "爆冷派"]:
            values = []
            for r in rounds:
                w = round_history[r].get("weights", {}).get(agent_name, 1.0)
                values.append(w)
            values.append(current_weights.get(agent_name, 1.0))
            all_rounds = list(rounds) + ([current_round_name] if current_round_name else [])
            traces.append({
                "name": agent_name,
                "color": AGENT_COLORS.get(agent_name, "#888"),
                "values": values,
                "rounds": all_rounds,
            })
        weight_history_json = json.dumps(traces, ensure_ascii=False)

    # 媒体情感
    media_json = []
    if media_df is not None and len(media_df) > 0:
        media_json = media_df.to_dict("records")
    media_str = json.dumps(media_json, ensure_ascii=False)

    # 社交热度
    social_json = []
    if social_df is not None and len(social_df) > 0:
        social_json = social_df.sort_values("heat_score", ascending=False).head(20).to_dict("records")
    social_str = json.dumps(social_json, ensure_ascii=False)

    # 赔率对比
    odds_json = []
    if odds_df is not None and len(odds_df) > 0:
        top_odds = odds_df.nlargest(min(15, len(odds_df)), "odds_spread")
        for _, row in top_odds.iterrows():
            odds_json.append({
                "home": row["home"], "away": row["away"],
                "oH": row["odds_H"], "oD": row["odds_D"], "oA": row["odds_A"],
            })
    odds_str = json.dumps(odds_json, ensure_ascii=False)

    # 预测数据行
    pred_rows = build_prediction_rows(predictions)
    bt_rows = build_backtest_rows(backtest)

    # 回测累计收益数据
    cum_ret = []
    if backtest:
        all_preds = []
        for r in backtest:
            all_preds.append(r["predictions"])
        if all_preds:
            pdf = pd.concat(all_preds, ignore_index=True)
            if "date" in pdf.columns:
                pdf = pdf.sort_values("date")
            bank = 0
            cum = []
            for _, row in pdf.iterrows():
                max_p = max(row["pred_H"], row["pred_D"], row["pred_A"])
                if max_p >= 0.50:
                    if row["correct"]:
                        bank += 0.92 / max_p - 1
                    else:
                        bank -= 1
                cum.append(bank)
            cum_ret = cum
    cum_ret_str = json.dumps(cum_ret)

    # 比赛复盘section
    if finished_review_rows:
        finished_review_section = f'''<div class="cd cd-md" style="border-left:3px solid var(--green)">
  <div class="cd-h">
    <span class="cd-h-dot" style="background:var(--green)"></span>
    <h2>已完成比赛复盘</h2>
    <span class="cd-h-b" style="background:var(--green)">{finished_count} MATCHES &middot; {correct_count}/{finished_count} CORRECT</span>
  </div>
  <div style="padding:4px 0">
    {finished_review_rows}
  </div>
</div>'''
    else:
        finished_review_section = ""

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>世界杯2026预测系统 | {today_str}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root{{
  --bg:#07080e;--bg2:#0b0d18;--card:rgba(12,15,28,.82);--card-hover:rgba(18,22,38,.88);
  --border:rgba(255,255,255,.04);--border-hover:rgba(255,255,255,.09);
  --text:#e2e6ef;--text2:#858ca5;--text3:#3d4560;
  --accent:#f5a623;--accent2:#f7c948;
  --home:#ef4444;--draw:#f59e0b;--away:#3b82f6;--green:#22c55e;
  --r:14px;--rs:10px;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:var(--bg);color:var(--text);font-family:Inter,'PingFang SC','Microsoft YaHei',system-ui,sans-serif;padding:28px;max-width:1380px;margin:0 auto;line-height:1.7;font-size:14px;-webkit-font-smoothing:antialiased;font-feature-settings:'cv02','cv03','cv04','cv11'}}
h1,h2,h3,h4,.hd-t,.st-v,.pc-pr,.am-h,.ar-r{{font-family:Outfit,Inter,'PingFang SC','Microsoft YaHei',system-ui,sans-serif;letter-spacing:-.02em}}
table,.num{{font-variant-numeric:tabular-nums}}

/* Header */
.hd{{display:flex;align-items:center;justify-content:space-between;padding:20px 0 16px;margin-bottom:24px;border-bottom:1px solid var(--border)}}
.hd-l{{display:flex;align-items:center;gap:14px}}
.hd-icon{{width:42px;height:42px;background:linear-gradient(135deg,var(--accent),#c6841a);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;box-shadow:0 0 0 1px rgba(245,166,35,.15)}}
.hd-t{{font-size:22px;font-weight:700;background:linear-gradient(135deg,#f7c948,#e8951a);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}

/* Stats */
.st{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.st-c{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:20px 16px;text-align:center;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);transition:all .3s cubic-bezier(.16,1,.3,1);position:relative;overflow:hidden}}
.st-c:hover{{border-color:var(--border-hover);transform:translateY(-3px);box-shadow:0 8px 32px rgba(0,0,0,.3)}}
.st-c::before{{content:'';position:absolute;top:0;left:20%;right:20%;height:1px;background:linear-gradient(90deg,transparent,var(--accent),transparent);opacity:0;transition:opacity .4s}}
.st-c:hover::before{{opacity:1}}
.st-v{{font-size:28px;font-weight:700;line-height:1.15;letter-spacing:-.04em;text-wrap:balance}}
.st-l{{font-size:10px;color:var(--text3);text-transform:uppercase;letter-spacing:.08em;margin-top:6px;font-weight:500}}

/* Cards */
.cd{{background:var(--card);border:1px solid var(--border);border-radius:var(--r);padding:24px;margin-bottom:20px;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);transition:border-color .3s cubic-bezier(.16,1,.3,1)}}
.cd:hover{{border-color:var(--border-hover)}}
.cd-h{{display:flex;align-items:center;gap:12px;margin-bottom:16px;padding-bottom:14px;border-bottom:1px solid var(--border)}}
.cd-h-dot{{width:6px;height:6px;border-radius:50%;background:var(--accent);flex-shrink:0}}
.cd-h h2{{font-size:10px;font-weight:600;color:var(--text3);letter-spacing:.06em;text-transform:uppercase}}
.cd-h-b{{margin-left:auto;font-size:9px;padding:3px 10px;border-radius:6px;background:rgba(255,255,255,.03);color:var(--text3);letter-spacing:.04em;font-weight:500;border:1px solid rgba(255,255,255,.04)}}

/* Matchday card */
.cd-md{{border-color:rgba(245,166,35,.15) !important}}
.cd-md:hover{{border-color:rgba(245,166,35,.3) !important}}
.cd-md .cd-h{{border-bottom-color:rgba(245,166,35,.08)}}
.cd-md .cd-h-dot{{background:#22c55e}}

/* Predictions Grid */
.pg{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:12px}}
.pc{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--rs);padding:18px;position:relative;transition:all .3s cubic-bezier(.16,1,.3,1);display:flex;flex-direction:column}}
.pc:hover{{border-color:var(--border-hover);transform:translateY(-3px);box-shadow:0 8px 24px rgba(0,0,0,.25)}}
.pc-d{{position:absolute;top:10px;right:12px;font-size:9px;padding:2px 8px;border-radius:5px;background:var(--border);color:var(--text3);letter-spacing:.03em}}
.pc-t{{font-size:15px;font-weight:600;color:var(--text);margin-bottom:6px;letter-spacing:-.02em}}
.pc-pr{{font-size:24px;font-weight:700;margin:8px 0 12px;text-wrap:balance}}
.pc-pr span{{font-size:11px;font-weight:400;color:var(--text3);margin-left:6px}}
.pc-prb{{display:flex;gap:8px;margin-bottom:10px}}
.pc-prb-b{{flex:1;text-align:center;padding:8px 4px 6px;border-radius:8px;background:var(--card)}}
.pc-prb-b .p{{font-size:16px;font-weight:600}}
.pc-prb-b .l{{font-size:9px;color:var(--text3);margin-top:2px}}
.pc-prb-b .o{{font-size:9px;color:var(--text3);margin-top:2px}}
.pc-cb{{height:4px;background:var(--border);border-radius:3px;overflow:hidden;margin-top:auto}}
.pc-cb .f{{height:100%;border-radius:3px;transition:width .8s cubic-bezier(.16,1,.3,1)}}

/* Tables */
.tw{{overflow-x:auto;border-radius:var(--rs);border:1px solid var(--border)}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th{{padding:10px 12px;text-align:left;font-weight:500;color:var(--text3);font-size:9px;text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--border);background:var(--bg2);position:sticky;top:0;z-index:1}}
td{{padding:9px 12px;border-bottom:1px solid rgba(255,255,255,.02);color:var(--text2);transition:color .2s}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(255,255,255,.02);color:var(--text)}}
.reason-cell{{font-size:11px;max-width:180px;color:var(--text3);line-height:1.6}}
.bet-cell{{font-size:11px;white-space:nowrap}}

/* Charts */
.ch{{width:100%;height:300px}}
.ch-sm{{width:100%;height:220px}}

/* Analysis Grid */
.ag{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}}
.ac{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--rs);padding:16px;transition:all .25s cubic-bezier(.16,1,.3,1)}}
.ac:hover{{border-color:var(--border-hover);transform:translateY(-2px)}}
.ac-n{{font-weight:600;color:var(--text);font-size:14px;margin-bottom:8px}}
.ac-v{{font-size:11px;color:var(--text3);line-height:1.8}}
.bc{{height:6px;background:var(--border);border-radius:4px;margin:6px 0;overflow:hidden}}
.bc .b{{height:100%;border-radius:4px;transition:width .6s cubic-bezier(.16,1,.3,1)}}

/* Tags */
.th{{color:var(--home)}} .td{{color:var(--draw)}} .ta{{color:var(--away)}}

/* Layout helpers */
.l2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}

/* Footer */
.ft{{text-align:center;font-size:10px;color:var(--text3);padding:24px 0 4px;border-top:1px solid var(--border);margin-top:16px;line-height:2}}

/* Responsive */
@media(max-width:768px){{body{{padding:16px}}.st{{grid-template-columns:repeat(2,1fr);gap:8px}}.pg{{grid-template-columns:1fr}}.l2{{grid-template-columns:1fr;gap:16px}}.hd{{flex-direction:column;align-items:flex-start;gap:10px}}.st-v{{font-size:22px}}.cd{{padding:16px}}.tw{{border:none}}}}

/* Animations */
@keyframes fi{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes gl{{0%,100%{{opacity:.3}}50%{{opacity:.7}}}}
.cd,.st-c,.pc,.ac{{animation:fi .5s cubic-bezier(.16,1,.3,1) both}}
.pc:nth-child(2){{animation-delay:.06s}}
.pc:nth-child(3){{animation-delay:.1s}}
.pc:nth-child(4){{animation-delay:.14s}}
.st-c:nth-child(2){{animation-delay:.06s}}
.st-c:nth-child(3){{animation-delay:.1s}}
.st-c:nth-child(4){{animation-delay:.14s}}

/* Agent PK */
.ap{{margin-top:6px}}
.am{{background:var(--bg2);border:1px solid var(--border);border-radius:var(--rs);padding:16px;margin-bottom:12px;transition:border-color .2s}}
.am:hover{{border-color:var(--border-hover)}}
.am-h{{font-weight:600;font-size:15px;margin-bottom:10px;color:var(--text);text-wrap:balance}}
.am-cb{{display:flex;height:8px;border-radius:4px;overflow:hidden;margin-bottom:12px}}
.am-cb-s{{display:flex;align-items:center;justify-content:center;font-size:7px;color:#fff;font-weight:600;transition:width .4s cubic-bezier(.16,1,.3,1)}}
.am-agents{{display:flex;flex-direction:column;gap:8px}}
.ar{{padding:10px 12px;border-left:3px solid;background:rgba(255,255,255,.015);border-radius:0 6px 6px 0;display:flex;flex-direction:column;gap:4px;transition:background .2s}}
.ar:hover{{background:rgba(255,255,255,.03)}}
.ar-h{{display:flex;align-items:center;gap:10px}}
.ar-n{{font-weight:600;font-size:12px;min-width:48px}}
.ar-r{{font-weight:700;font-size:14px;min-width:36px}}
.ar-s{{font-size:13px;color:var(--text2);font-weight:600;min-width:36px;font-variant-numeric:tabular-nums}}
.ar-c{{font-size:10px;color:var(--text3);min-width:36px}}
.ar-cons{{font-size:8px;padding:2px 6px;border-radius:4px;background:rgba(34,197,94,.15);color:#22c55e;margin-left:6px;font-weight:600;letter-spacing:.03em}}
.ar-reason{{font-size:10px;color:var(--text3);line-height:1.6}}
/* Leaderboard */
.lb-w{{overflow-x:auto}}
.lb-w table{{font-size:11px}}
.lb-w th{{font-size:8px;padding:8px 10px}}
.lb-w td{{padding:6px 10px}}

/* Empty state */
.empty-state{{text-align:center;padding:32px 16px;color:var(--text3);font-size:13px;line-height:2}}
.empty-state::before{{content:'📊';display:block;font-size:28px;margin-bottom:8px;opacity:.5}}
</style>
</head>
<body>

<!-- Header -->
<header class="hd">
  <div class="hd-l">
    <div class="hd-icon">⚽</div>
    <div class="hd-t">世界杯2026预测系统</div>
  </div>
  <div class="hd-r">
    <strong id="liveClock"></strong><br>
    48支球队 · 104场比赛 · FIFA排名+ML模型
  </div>
</header>

<!-- Match Review -->
{finished_review_section}

<!-- Latest Matchday -->
<div class="cd cd-md">
  <div class="cd-h">
    <span class="cd-h-dot"></span>
    <h2>最新比赛日 · {matchday_date_str}</h2>
    <span class="cd-h-b">MATCHDAY</span>
  </div>
  <div class="tw">
  <table>
    <tr><th>时间</th><th>主队</th><th>客队</th><th>预测</th><th>比分</th><th>概率</th><th>赔率(主/平/客)</th><th>推荐理由</th><th>投注建议</th></tr>
    {matchday_html_rows}
  </table>
  </div>
</div>

<!-- Agent PK Debate + Leaderboard -->
<div class="l2">
  <div class="cd">
    <div class="cd-h">
      <span class="cd-h-dot"></span>
      <h2>智能体PK辩论 · {matchday_date_str}</h2>
      <span class="cd-h-b">5 AGENTS</span>
    </div>
    {agent_pk_html if agent_pk_html else '<div class="empty-state">暂无智能体预测数据<br>请先运行 agents.py</div>'}
  </div>
  <div class="cd">
    <div class="cd-h">
      <span class="cd-h-dot"></span>
      <h2>智能体业绩排行榜</h2>
      <span class="cd-h-b">LEADERBOARD</span>
    </div>
    <div class="lb-w">
      <table>
        <tr><th>排名</th><th>智能体</th><th>结果分布</th><th>平均置信度</th><th>真实准确率</th><th>权重</th><th>总分</th></tr>
        {agent_leaderboard_rows if agent_leaderboard_rows else '<tr><td colspan="7"><div class="empty-state" style="padding:24px 16px">暂无排行榜数据</div></td></tr>'}
      </table>
    </div>
  </div>
</div>

<!-- Weight History Chart -->
<div class="cd">
  <div class="cd-h">
    <span class="cd-h-dot"></span>
    <h2>智能体权重历史</h2>
    <span class="cd-h-b">WEIGHT TRACKING</span>
  </div>
  <div id="weightChart" class="ch"></div>
</div>

<!-- Stats -->
<div class="st">
  <div class="st-c">
    <div class="st-v" style="color:var(--accent2)">{avg_acc:.1%}</div>
    <div class="st-l">回测准确率</div>
  </div>
  <div class="st-c">
    <div class="st-v" style="color:var(--away)">{n_matches}</div>
    <div class="st-l">历史训练场次</div>
  </div>
  <div class="st-c">
    <div class="st-v" style="color:var(--green)">{len(high_conf_picks)}</div>
    <div class="st-l">待预测比赛</div>
  </div>
  <div class="st-c">
    <div class="st-v" style="font-size:22px;color:var(--accent)">2026.06.11</div>
    <div class="st-l">开幕日</div>
  </div>
</div>

<!-- High Confidence Predictions -->
<div class="cd">
  <div class="cd-h">
    <span class="cd-h-dot"></span>
    <h2>高置信度预测</h2>
    <span class="cd-h-b">{len(high_conf_picks)} MATCHES</span>
  </div>
  <div id="picksContainer" class="pg"></div>
</div>

<!-- Backtest Chart -->
<div class="cd">
  <div class="cd-h">
    <span class="cd-h-dot"></span>
    <h2>模型回测收益曲线</h2>
  </div>
  <div id="backtestChart" class="ch"></div>
</div>

<!-- Media + Social -->
<div class="l2">
  <div class="cd">
    <div class="cd-h">
      <span class="cd-h-dot"></span>
      <h2>媒体情感分析</h2>
    </div>
    <div id="mediaChart" class="ch-sm"></div>
  </div>
  <div class="cd">
    <div class="cd-h">
      <span class="cd-h-dot"></span>
      <h2>社交媒体热度</h2>
    </div>
    <div id="socialChart" class="ch-sm"></div>
  </div>
</div>

<!-- Odds Comparison -->
<div class="cd">
  <div class="cd-h">
    <span class="cd-h-dot"></span>
    <h2>赔率对比分析</h2>
    <span class="cd-h-b">实力悬殊最大15场</span>
  </div>
  <div id="oddsChart" class="ch"></div>
</div>

<!-- Team Analysis -->
<div class="cd">
  <div class="cd-h">
    <span class="cd-h-dot"></span>
    <h2>球队综合分析</h2>
    <span class="cd-h-b">媒体情感 + 社交热度</span>
  </div>
  <div id="teamAnalysis" class="ag"></div>
</div>

<!-- All Predictions -->
<div class="cd">
  <div class="cd-h">
    <span class="cd-h-dot"></span>
    <h2>全部预测详情</h2>
    <span class="cd-h-b">{len(high_conf_picks)} MATCHES</span>
  </div>
  <div class="tw" id="allPredictionsTable" style="max-height:480px;overflow-y:auto"></div>
</div>

<!-- Footer -->
<div class="ft">
  预测仅供参考 · 数据来源: football-data.org + FIFA Rankings · 模型: RandomForest + GradientBoosting + ExtraTrees + Logistic Ensemble
</div>

<script>
// ===== Live Clock =====
function updateClock() {{
  var now = new Date();
  var s = now.getFullYear() + '-' +
    String(now.getMonth()+1).padStart(2,'0') + '-' +
    String(now.getDate()).padStart(2,'0') + ' ' +
    String(now.getHours()).padStart(2,'0') + ':' +
    String(now.getMinutes()).padStart(2,'0') + ':' +
    String(now.getSeconds()).padStart(2,'0');
  document.getElementById('liveClock').textContent = s;
}}
updateClock();
setInterval(updateClock, 1000);

// ===== Prediction Cards =====
var picks = {json.dumps(high_conf_picks, ensure_ascii=False)};
var ph = document.getElementById('picksContainer');
if (picks.length === 0) {{
  ph.innerHTML = '<div class="empty-state">暂无预测数据<br>请先运行主流程</div>';
}} else {{
  picks.forEach(function(p){{
    var c = p.pred_r === 'H' ? 'var(--home)' : p.pred_r === 'D' ? 'var(--draw)' : 'var(--away)';
    var d = document.createElement('div'); d.className = 'pc';
    var pb = '<div class="pc-prb">' +
      '<div class="pc-prb-b"><div class="p" '+(parseFloat(p.pH)>40?'style="color:var(--home);font-weight:600"':'')+'>'+p.pH+'%</div><div class="l">主胜</div><div class="o">@'+p.oH+'</div></div>' +
      '<div class="pc-prb-b"><div class="p" '+(parseFloat(p.pD)>35?'style="color:var(--draw);font-weight:600"':'')+'>'+p.pD+'%</div><div class="l">平局</div><div class="o">@'+p.oD+'</div></div>' +
      '<div class="pc-prb-b"><div class="p" '+(parseFloat(p.pA)>40?'style="color:var(--away);font-weight:600"':'')+'>'+p.pA+'%</div><div class="l">客胜</div><div class="o">@'+p.oA+'</div></div>' +
      '</div>';
    var confPct = parseFloat(p.conf);
    var confColor = confPct >= 50 ? 'var(--green)' : confPct >= 40 ? 'var(--draw)' : 'var(--text3)';
    var scoreStr = p.hs !== undefined ? '<span style="font-size:13px;color:var(--text2);font-weight:600;margin-left:6px">'+p.hs+'-'+p.as+'</span>' : '';
    d.innerHTML = '<span class="pc-d">'+p.date+'</span>' +
      '<div class="pc-t">'+p.home+'<span style="color:var(--text3)"> vs </span>'+p.away+scoreStr+'</div>' +
      '<div class="pc-pr" style="color:'+c+'">'+p.pred+' <span>'+p.conf+'% 置信度</span></div>' +
      pb +
      '<div class="pc-cb"><div class="f" style="width:'+p.conf+'%;background:'+confColor+'"></div></div>';
    ph.appendChild(d);
  }});
}}

// ===== Backtest Curve =====
var cumRet = {cum_ret_str};
var bc = echarts.init(document.getElementById('backtestChart'));
bc.setOption({{
  backgroundColor:'transparent',
  tooltip:{{trigger:'axis',formatter:function(p){{return '<b>#'+p[0].axisValue+'</b><br/>'+p[0].marker+' 累计收益: ¥'+p[0].value.toFixed(0);}}}},
  grid:{{left:'8%',right:'4%',top:'8%',bottom:'10%'}},
  xAxis:{{type:'category',data:Array.from({{length:cumRet.length}},(_,i)=>'#'+(i+1)),axisLabel:{{fontSize:9,color:'#4a5270',show:false}},axisLine:{{lineStyle:{{color:'#1a1d2e'}}}},axisTick:{{show:false}}}},
  yAxis:{{type:'value',name:'累计盈亏 (¥)',nameTextStyle:{{color:'#4a5270',fontSize:10}},splitLine:{{lineStyle:{{color:'#1a1d2e'}}}},axisLabel:{{color:'#4a5270',fontSize:10,formatter:'¥{{value}}'}}}},
  series:[{{
    name:'策略收益',type:'line',data:cumRet,smooth:true,
    lineStyle:{{width:2,color:'#f59e0b'}},
    areaStyle:{{color:{{type:'linear',x:0,y:0,x2:0,y2:1,colorStops:[{{offset:0,color:'rgba(245,158,11,.25)'}},{{offset:1,color:'rgba(245,158,11,0)'}}]}}}},
    markLine:{{silent:true,data:[{{yAxis:0,lineStyle:{{color:'#2a2d42',type:'dashed',width:1}},label:{{formatter:'盈亏平衡',color:'#4a5270',fontSize:10}}}}]}}
  }}]
}});

// ===== Weight History Chart =====
var weightData = {weight_history_json};
var wc = echarts.init(document.getElementById('weightChart'));
if (weightData.length > 0 && weightData[0].rounds.length > 0) {{
  wc.setOption({{
    backgroundColor:'transparent',
    tooltip:{{trigger:'axis',formatter:function(ps){{var s='<b>'+ps[0].axisValue+'</b>';ps.forEach(function(p){{s+='<br/>'+p.marker+' '+p.seriesName+': '+p.value.toFixed(3)}});return s;}}}},
    legend:{{data:weightData.map(function(d){{return d.name}}),textStyle:{{color:'#94a3b8',fontSize:10}},top:0}},
    grid:{{left:'6%',right:'4%',top:'18%',bottom:'10%'}},
    xAxis:{{type:'category',data:weightData[0].rounds,axisLabel:{{color:'#4a5270',fontSize:10}},axisLine:{{lineStyle:{{color:'#1a1d2e'}}}},axisTick:{{show:false}}}},
    yAxis:{{type:'value',name:'权重',min:0,nameTextStyle:{{color:'#4a5270',fontSize:10}},splitLine:{{lineStyle:{{color:'#1a1d2e'}}}},axisLabel:{{color:'#4a5270',fontSize:10}}}},
    series:weightData.map(function(d){{return {{
      name:d.name,type:'line',data:d.values,smooth:true,
      lineStyle:{{width:2,color:d.color}},
      itemStyle:{{color:d.color}},
      symbol:'circle',symbolSize:6,
      markLine:{{silent:true,data:[{{yAxis:1,lineStyle:{{color:'#2a2d42',type:'dashed',width:1}},label:{{formatter:'基准1.0',color:'#4a5270',fontSize:9}}}}]}}
    }}}})
  }});
}}
window.addEventListener('resize', function(){{ wc.resize() }});

// ===== Media Sentiment =====
var mediaData = {media_str};
var topMedia = mediaData.slice().sort(function(a,b){{return b.sentiment_score - a.sentiment_score}}).slice(0,15);
var mc = echarts.init(document.getElementById('mediaChart'));
mc.setOption({{
  backgroundColor:'transparent',
  tooltip:{{trigger:'axis',formatter:function(p){{return p[0].name+'<br/>'+p[0].marker+' 情感得分: '+(p[0].value*100).toFixed(0)+'%'}}}},
  grid:{{left:'22%',right:'6%',top:'6%',bottom:'10%'}},
  xAxis:{{type:'value',max:0.9,axisLabel:{{color:'#4a5270',fontSize:9,formatter:'{{value}}'}},splitLine:{{lineStyle:{{color:'#1a1d2e'}}}}}},
  yAxis:{{type:'category',data:topMedia.map(function(d){{return d.team}}),axisLabel:{{color:'#94a3b8',fontSize:9}},axisLine:{{lineStyle:{{color:'#1a1d2e'}}}},axisTick:{{show:false}}}},
  series:[{{
    type:'bar',data:topMedia.map(function(d){{return d.sentiment_score}}),
    barWidth:'55%',
    itemStyle:{{color:function(p){{return p.value>0.5?'#22c55e':'#ef4444'}},borderRadius:[0,3,3,0]}},
    label:{{show:true,position:'right',color:'#94a3b8',fontSize:9,formatter:function(p){{return (p.value*100).toFixed(0)+'%'}}}}
  }}]
}});

// ===== Social Heat =====
var socialData = {social_str};
var sc = echarts.init(document.getElementById('socialChart'));
sc.setOption({{
  backgroundColor:'transparent',
  tooltip:{{trigger:'axis'}},
  grid:{{left:'22%',right:'6%',top:'6%',bottom:'10%'}},
  xAxis:{{type:'value',max:1.0,axisLabel:{{color:'#4a5270',fontSize:9,formatter:'{{value}}'}},splitLine:{{lineStyle:{{color:'#1a1d2e'}}}}}},
  yAxis:{{type:'category',data:socialData.map(function(d){{return d.team}}),axisLabel:{{color:'#94a3b8',fontSize:9}},axisLine:{{lineStyle:{{color:'#1a1d2e'}}}},axisTick:{{show:false}}}},
  series:[{{
    type:'bar',data:socialData.map(function(d){{return d.heat_score}}),
    barWidth:'45%',
    itemStyle:{{color:{{type:'linear',x:0,y:0,x2:1,y2:0,colorStops:[{{offset:0,color:'#f97316'}},{{offset:1,color:'#f59e0b'}}]}},borderRadius:[0,3,3,0]}},
    label:{{show:true,position:'right',color:'#94a3b8',fontSize:9,formatter:function(p){{return (p.value*100).toFixed(0)+'%'}}}}
  }}]
}});

// ===== Odds Comparison =====
var oddsData = {odds_str};
if (oddsData.length > 0) {{
  var oc = echarts.init(document.getElementById('oddsChart'));
  oc.setOption({{
    backgroundColor:'transparent',
    tooltip:{{trigger:'axis'}},
    legend:{{data:['主胜赔率','平局赔率','客胜赔率'],textStyle:{{color:'#94a3b8',fontSize:10}},top:0}},
    grid:{{left:'8%',right:'4%',top:'18%',bottom:'14%'}},
    xAxis:{{type:'category',data:oddsData.map(function(d){{return d.home+' vs '+d.away}}),axisLabel:{{fontSize:9,color:'#94a3b8',rotate:30}},axisLine:{{lineStyle:{{color:'#1a1d2e'}}}}}},
    yAxis:{{type:'value',name:'赔率',nameTextStyle:{{color:'#4a5270',fontSize:10}},splitLine:{{lineStyle:{{color:'#1a1d2e'}}}},axisLabel:{{color:'#4a5270',fontSize:9}}}},
    series:[
      {{name:'主胜赔率',type:'bar',data:oddsData.map(function(d){{return d.oH}}),barWidth:'20%',itemStyle:{{color:'#ef4444',borderRadius:[3,3,0,0]}},label:{{show:true,position:'top',fontSize:9,color:'#ef4444',formatter:'{{c}}'}}}},
      {{name:'平局赔率',type:'bar',data:oddsData.map(function(d){{return d.oD}}),barWidth:'20%',itemStyle:{{color:'#f59e0b',borderRadius:[3,3,0,0]}},label:{{show:true,position:'top',fontSize:9,color:'#f59e0b',formatter:'{{c}}'}}}},
      {{name:'客胜赔率',type:'bar',data:oddsData.map(function(d){{return d.oA}}),barWidth:'20%',itemStyle:{{color:'#3b82f6',borderRadius:[3,3,0,0]}},label:{{show:true,position:'top',fontSize:9,color:'#3b82f6',formatter:'{{c}}'}}}}
    ]
  }});
}}

// ===== Team Analysis =====
var td = mediaData.concat(socialData).reduce(function(a, i) {{
  var t = i.team;
  if (!a[t]) a[t] = {{team:t, sentiment:0.5, heat:0.5}};
  if (i.sentiment_score !== undefined) a[t].sentiment = i.sentiment_score;
  if (i.heat_score !== undefined) a[t].heat = i.heat_score;
  return a;
}}, {{}});
var tarr = Object.values(td).sort(function(a,b){{return (b.sentiment+b.heat) - (a.sentiment+a.heat)}}).slice(0,20);
var tc = document.getElementById('teamAnalysis');
if (tarr.length > 0) {{
  tarr.forEach(function(t){{
    var score = (t.sentiment + t.heat) / 2 * 100;
    var d = document.createElement('div'); d.className = 'ac';
    d.innerHTML = '<div class="ac-n">'+t.team+'</div>' +
      '<div class="ac-v">媒体情感: '+(t.sentiment*100).toFixed(0)+'%</div>' +
      '<div class="bc"><div class="b" style="width:'+(t.sentiment*100)+'%;background:'+(t.sentiment>0.5?'#22c55e':'#ef4444')+'"></div></div>' +
      '<div class="ac-v">社交热度: '+(t.heat*100).toFixed(0)+'%</div>' +
      '<div class="bc"><div class="b" style="width:'+(t.heat*100)+'%;background:linear-gradient(90deg,#f97316,#f59e0b)"></div></div>' +
      '<div class="ac-v" style="color:var(--text2);font-weight:500;margin-top:2px">综合评分: '+score.toFixed(0)+'/100</div>';
    tc.appendChild(d);
  }});
}} else {{
  tc.innerHTML = '<div class="empty-state">暂无分析数据</div>';
}}

// ===== All Predictions Table =====
var predData = [{pred_rows}];
var at = document.getElementById('allPredictionsTable');
if (predData.length > 0) {{
  var h = '<table><tr><th>日期</th><th>主队</th><th>客队</th><th>预测</th><th>比分</th><th>主胜</th><th>平局</th><th>客胜</th><th>置信度</th><th>主胜赔</th><th>平赔</th><th>客胜赔</th></tr>';
  predData.forEach(function(r){{
    var pc = r.pred_r === 'H' ? 'var(--home)' : r.pred_r === 'D' ? 'var(--draw)' : 'var(--away)';
    h += '<tr><td>'+r.date+'</td><td>'+r.home+'</td><td>'+r.away+'</td>' +
      '<td style="color:'+pc+';font-weight:600">'+r.pred+'</td>' +
      '<td style="font-weight:600;color:var(--text2)">'+r.hs+'-'+r.as+'</td>' +
      '<td>'+r.ph+'%</td><td>'+r.pd+'%</td><td>'+r.pa+'%</td>' +
      '<td>'+r.conf+'%</td><td>'+r.oh+'</td><td>'+r.od+'</td><td>'+r.oa+'</td></tr>';
  }});
  h += '</table>';
  at.innerHTML = h;
}} else {{
  at.innerHTML = '<div class="empty-state">暂无预测数据</div>';
}}

window.addEventListener('resize', function(){{ bc.resize(); if(typeof mc !== 'undefined'){{mc.resize();}} if(typeof sc !== 'undefined'){{sc.resize();}} }});
</script>
</body>
</html>'''

    path = os.path.join(OUTPUT_DIR, f"worldcup_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    latest_path = os.path.join(OUTPUT_DIR, "worldcup_dashboard_latest.html")
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"仪表盘已生成: {path}")
    return path


if __name__ == "__main__":
    data = load_all_data()
    if data:
        generate_dashboard(data)
    else:
        log.error("无数据，请先运行 main.py")
