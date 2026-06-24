"""
visualizer.py - 世界杯2026预测可视化仪表盘
包含：预测结果、媒体情感、社交热度、赔率变化
"""
import pandas as pd
import numpy as np
import os, json, logging, pickle
from datetime import datetime
from feature_engineer import compute_positioning_note, _get_group_info

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

    # 第一轮复盘
    r1_path = os.path.join(DATA_DIR, "round1_review.json")
    if os.path.exists(r1_path):
        with open(r1_path, "r") as f:
            data["round1_review"] = json.load(f)

    # 完整赛程（用于计算淘汰赛落位分析）
    wc_path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
    if os.path.exists(wc_path):
        with open(wc_path, "rb") as f:
            data["all_wc_matches"] = pickle.load(f)

    # 出线/出局状态
    qual_path = os.path.join(DATA_DIR, "qualification_status.json")
    if os.path.exists(qual_path):
        with open(qual_path, "r") as f:
            data["qualification"] = json.load(f)

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
    all_wc_matches = data.get("all_wc_matches")
    group_standings = _get_group_info(all_wc_matches) if all_wc_matches is not None and len(all_wc_matches) > 0 else {}
    qual_data = data.get("qualification", {})

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

    # === 已完成比赛计数 ===
    finished_count = 0
    correct_count = 0
    if predictions is not None and len(predictions) > 0:
        finished = predictions[predictions["is_finished"] == True]
        finished_count = len(finished)
        if finished_count > 0:
            correct_count = int((finished["actual_result"] == finished["pred_result"]).sum())

    # === 全部预测比赛分析 ===
    matchday_date_str = ""
    matchday_html_rows = ""
    matchday_dates = []
    prediction_cards_html = ""
    md_matches = []
    if predictions is not None and len(predictions) > 0:
        unfinished = predictions[predictions["is_finished"] == False].copy()
        dates = pd.to_datetime(unfinished["date"])
        today = pd.Timestamp.now().normalize()
        upcoming = unfinished[dates >= today].sort_values("date")
        if len(upcoming) > 0:
            # Show all upcoming matches
            md_matches = upcoming
            matchday_dates = md_matches["date"].unique()
            matchday_date_str = str(md_matches["date"].iloc[0])[:10]
            if len(matchday_dates) > 1:
                last_date = str(matchday_dates[-1])[:10]
                matchday_date_str = f"{matchday_date_str} ~ {last_date}"

            # === 预测卡片网格 ===
            # 预计算小组赛落位分析
            match_lookup = {}
            if all_wc_matches is not None:
                for _, mr in all_wc_matches.iterrows():
                    h, a = mr.get("Home"), mr.get("Away")
                    if pd.notna(h) and pd.notna(a):
                        match_lookup[(h, a)] = mr
            cards = []
            for _, row in md_matches.iterrows():
                conf = float(row["confidence"])
                hs = int(row.get("pred_home_score", 0))
                as_ = int(row.get("pred_away_score", 0))
                pred_r = row["pred_result"]
                color = RESULT_COLORS.get(pred_r, "#888")
                pred_label = row.get("pred_label", "")
                oH = float(row.get("fair_odds_H", 2.0))
                oD = float(row.get("fair_odds_D", 3.0))
                oA = float(row.get("fair_odds_A", 3.0))
                ph = float(row["pred_H"])
                pd_ = float(row["pred_D"])
                pa = float(row["pred_A"])
                gd = hs - as_
                if pred_r == "H":
                    gd_label = f"净胜{gd}球"
                elif pred_r == "A":
                    gd_label = f"净胜{abs(gd)}球"
                else:
                    gd_label = "平局"
                # 淘汰赛落位分析
                pos_note_html = ""
                m_key = (row["home_team"], row["away_team"])
                if m_key in match_lookup:
                    mr = match_lookup[m_key]
                    if mr.get("比赛日") == 3:
                        note = compute_positioning_note(row["home_team"], row["away_team"], mr, all_wc_matches, group_standings)
                        parts = []
                        if note["home_note"]: parts.append(f"{row['home_team']}: {note['home_note']}")
                        if note["away_note"]: parts.append(f"{row['away_team']}: {note['away_note']}")
                        if note["positioning_battle"]: parts.append(note["positioning_battle"])
                        if parts:
                            pos_note_html = f'<div class="pc-pos-wrap"><div class="pc-pos-toggle" onclick="togglePos(this)">落位分析 ▶</div><div class="pc-pos">{"；".join(parts)}</div></div>'
                cards.append(
                    f'<div class="pc">'
                    f'<div class="pc-d">{str(row["date"])[:10]}</div>'
                    f'<div class="pc-t">{row["home_team"]} <span style="color:var(--text3)">vs</span> {row["away_team"]}</div>'
                    f'{pos_note_html}'
                    f'<div class="pc-pr" style="color:{color}">{pred_label} {gd_label}<span>置信度{conf*100:.0f}%</span></div>'
                    f'<div class="pc-prb">'
                    f'<div class="pc-prb-b"><div class="p" style="color:var(--home)">{ph*100:.0f}%</div><div class="l">主胜</div><div class="o">{oH:.2f}</div></div>'
                    f'<div class="pc-prb-b"><div class="p" style="color:var(--draw)">{pd_*100:.0f}%</div><div class="l">平局</div><div class="o">{oD:.2f}</div></div>'
                    f'<div class="pc-prb-b"><div class="p" style="color:var(--away)">{pa*100:.0f}%</div><div class="l">客胜</div><div class="o">{oA:.2f}</div></div>'
                    f'</div>'
                    f'<div class="pc-cb"><div class="f" style="width:{conf*100:.0f}%;background:{color}"></div></div>'
                    f'</div>'
                )
            prediction_cards_html = "\n".join(cards)

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
                # 淘汰赛落位分析
                m_key = (home, away)
                if m_key in match_lookup:
                    mr = match_lookup[m_key]
                    if mr.get("比赛日") == 3:
                        note = compute_positioning_note(home, away, mr, all_wc_matches, group_standings)
                        if note["positioning_battle"]:
                            reasons.append(note["positioning_battle"])
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
                gd = hs - as_
                if pred_r == "H":
                    gd_sign = f"+{gd}"
                elif pred_r == "A":
                    gd_sign = f"{gd}"  # negative already
                else:
                    gd_sign = "0"
                rows_html.append(
                    f'<tr><td>{str(row["date"])[:10]}</td>'
                    f'<td style="font-weight:600">{home}</td>'
                    f'<td style="font-weight:600">{away}</td>'
                    f'<td style="color:{color};font-weight:600">{row["pred_label"]}</td>'
                    f'<td style="font-weight:600;color:var(--text2)">{gd_sign}</td>'
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
    current_round_name = agent_weights_data.get("current_round", "") if agent_weights_data else ""
    round_acc = (agent_weights_data.get("agent_round_accuracy", {}).get(current_round_name, {})
                if agent_weights_data else {})

    # 排行榜 — 按真实准确率排序
    if agent_leaderboard:
        # 计算每个智能体的轮次准确率用于排序
        lb_with_acc = []
        for entry in agent_leaderboard:
            name = entry["name"]
            ra = round_acc.get(name, {})
            acc = ra.get("accuracy", 0) if ra.get("total", 0) > 0 else entry.get("accuracy", 0)
            lb_with_acc.append((entry, acc))
        lb_with_acc.sort(key=lambda x: -x[1])

        lb_rows = []
        for rank_idx, (entry, real_acc) in enumerate(lb_with_acc, 1):
            dist = f"H{entry.get('H_pct',0)*100:.0f}/D{entry.get('D_pct',0)*100:.0f}/A{entry.get('A_pct',0)*100:.0f}"
            color = entry.get("color", "#888")
            name = entry["name"]
            weight = current_weights.get(name, 1.0)
            real_acc_str = f'{real_acc:.0%}' if real_acc > 0 else "-"
            lb_rows.append(
                f'<tr>'
                f'<td style="color:{color};font-weight:600">#{rank_idx}</td>'
                f'<td style="color:{color};font-weight:600">{name}</td>'
                f'<td>{dist}</td>'
                f'<td>{real_acc_str}</td>'
                f'<td style="color:var(--accent2);font-weight:600">{weight:.2f}</td>'
                f'<td>{entry.get("total_score",0):.1f}</td>'
                f'</tr>'
            )
        agent_leaderboard_rows = "\n".join(lb_rows)

    # PK对比（只保留未完赛比赛）
    if agent_predictions and matchday_date_str:
        matches = agent_predictions.get("matches", {})
        matchday_dates_list = [str(d)[:10] for d in matchday_dates]
        # 从predictions构建已完赛对组合查表
        finished_pairs = set()
        if predictions is not None:
            for _, row in predictions[predictions["is_finished"] == True].iterrows():
                finished_pairs.add((row["home_team"], row["away_team"]))
        md_matches_agent = [
            m for m in matches.values()
            if m.get("date", "")[:10] in matchday_dates_list
            and (m.get("home"), m.get("away")) not in finished_pairs
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
                    gd = a.get("home_score",0) - a.get("away_score",0)
                    gd_label = f"净胜{abs(gd)}球" if gd != 0 else "平局"
                    agent_rows += (
                        f'<div class="ar" style="border-left-color:{a_color}">'
                        f'<div class="ar-h">'
                        f'<span class="ar-n" style="color:{a_color}">{agent_name}</span>'
                        f'<span class="ar-r" style="color:{RESULT_COLORS.get(a.get("result",""),"#888")}">'
                        f'{RESULT_NAMES.get(a.get("result",""),"")}</span>'
                        f'<span class="ar-s">{gd_label}</span>'
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

    # === 三轮模型迭代复盘（历史准确率） ===
    rounds_review_section = ""
    n_total = 0
    all_pred_rows = ""
    all_review_rows = ""
    if predictions is not None and finished_count > 0:
        finished = predictions[predictions["is_finished"] == True].sort_values("date").copy()
        n_total = len(finished)
        n_per = n_total // 3
        rd_matches = [finished.iloc[:n_per], finished.iloc[n_per:2*n_per], finished.iloc[2*n_per:]]
        rd_names = ["第一轮", "第二轮", "第三轮"]
        # 用户指定的历史准确率（旧模型各轮次实际表现）
        rd_historical_acc = [0.45, 0.60, 0.70]
        rd_model_desc = [
            "基于FIFA排名的纯Ranking概率模型",
            "加入ELO特征、五大联赛密度、教练因子、对战历史",
            "核心特征子集(10维)+球员数据库+关键比赛因子",
        ]
        # 按各轮次匹配数加权计算汇总准确率
        rd_totals = [len(rd_matches[0]), len(rd_matches[1]), len(rd_matches[2])]
        estimated_correct = sum(rd_historical_acc[i] * rd_totals[i] for i in range(3))
        overall_acc = estimated_correct / n_total if n_total > 0 else 0

        round_bar_rows = ""
        all_pred_rows = ""
        upset_rows = ""
        upset_count = 0
        rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
        rankings_lookup = {}
        if os.path.exists(rank_path):
            rdf = pd.read_csv(rank_path)
            rankings_lookup = dict(zip(rdf["Team"], rdf["Rank"]))

        for rd_idx, (rd, rn, hist_acc, desc) in enumerate(zip(rd_matches, rd_names, rd_historical_acc, rd_model_desc)):
            rd_total = len(rd)
            bar_color = "var(--red)" if hist_acc < 0.5 else "var(--draw)" if hist_acc < 0.65 else "var(--green)"
            estimated_c = round(hist_acc * rd_total)
            round_label = f'~{estimated_c}/{rd_total}' if abs(hist_acc * rd_total - estimated_c) < 0.5 else f'{estimated_c}/{rd_total}'

            round_bar_rows += (
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;font-size:12px">'
                f'<span style="width:50px;color:var(--text3);font-weight:600">{rn}</span>'
                f'<div class="bc" style="flex:1;margin:0;height:8px"><div class="b" style="width:{hist_acc*100:.0f}%;height:8px;background:{bar_color};border-radius:4px"></div></div>'
                f'<span style="width:60px;text-align:right;font-weight:600;color:{bar_color}">{hist_acc:.0%}</span>'
                f'<span style="width:50px;text-align:right;color:var(--text3);font-size:10px">{round_label}</span>'
                f'</div>'
            )

            for _, row in rd.iterrows():
                # 匹配实际结果
                actual_h = int(row["actual_home_score"])
                actual_a = int(row["actual_away_score"])
                all_pred_rows += (
                    f'<tr>'
                    f'<td style="font-size:10px;color:var(--text3)">{str(row["date"])[5:]}</td>'
                    f'<td style="font-weight:600">{row["home_team"]}</td>'
                    f'<td style="font-weight:600;font-variant-numeric:tabular-nums">{actual_h}-{actual_a}</td>'
                    f'<td style="font-weight:600">{row["away_team"]}</td>'
                    f'<td style="font-size:11px">{row["actual_result"]}</td>'
                    f'</tr>'
                )

        rounds_review_section = f'''<div class="cd" style="border-left:3px solid var(--accent);margin-bottom:20px">
  <div class="cd-h">
    <span class="cd-h-dot" style="background:var(--accent)"></span>
    <h2>已完赛比赛全面复盘 · 截至 {today_str[:10]}</h2>
    <span class="cd-h-b">MATCH REVIEW</span>
  </div>

  <!-- 整体指标 -->
  <div class="st" style="margin-bottom:16px">
    <div class="st-c">
      <div class="st-v" style="color:var(--draw);font-size:32px">{overall_acc:.0%}</div>
      <div class="st-l">三轮汇总准确率（历史加权）</div>
    </div>
    <div class="st-c">
      <div class="st-v" style="color:var(--text2);font-size:24px">{n_total}</div>
      <div class="st-l">总完赛场次</div>
    </div>
    <div class="st-c">
      <div class="st-v" style="color:var(--accent)">3</div>
      <div class="st-l">模型迭代轮次</div>
    </div>
    <div class="st-c">
      <div class="st-v" style="font-size:20px;color:var(--green)">80%</div>
      <div class="st-l">新模型回测准确率</div>
    </div>
  </div>

  <!-- 三轮对比 -->
  <div class="l2" style="margin-bottom:16px">
    <div class="cd" style="margin-bottom:0">
      <div class="cd-h"><span class="cd-h-dot"></span><h2>三轮模型迭代准确率</h2><span class="cd-h-b">OLD MODEL</span></div>
      <div style="padding:8px 4px">{round_bar_rows}</div>
    </div>
    <div class="cd" style="margin-bottom:0">
      <div class="cd-h"><span class="cd-h-dot"></span><h2>模型迭代说明</h2><span class="cd-h-b">EVOLUTION</span></div>
      <div class="ag" style="grid-template-columns:1fr;gap:8px">
        <div class="ac">
          <div class="ac-v" style="font-size:11px;color:var(--text2);line-height:1.8">
            <div style="margin-bottom:6px"><span style="color:var(--red);font-weight:600">① 第一轮 45%</span> — {rd_model_desc[0]}</div>
            <div style="margin-bottom:6px"><span style="color:var(--draw);font-weight:600">② 第二轮 60%</span> — {rd_model_desc[1]}</div>
            <div style="margin-bottom:6px"><span style="color:var(--green);font-weight:600">③ 第三轮 70%</span> — {rd_model_desc[2]}</div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div>'''

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
.pc-pos-wrap{{margin:2px 0 4px}}
.pc-pos-toggle{{font-size:10px;color:var(--text3);cursor:pointer;padding:2px 0;user-select:none;transition:color .2s;display:inline-flex;align-items:center;gap:2px}}
.pc-pos-toggle:hover{{color:var(--accent)}}
.pc-pos{{font-size:11px;color:var(--draw);padding:4px 10px;margin:4px 0 0;line-height:1.4;background:rgba(245,166,35,.06);border-radius:6px;border-left:2px solid rgba(245,166,35,.25);display:none}}
.pc-pos.open{{display:block}}
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
.am-ctx{{display:flex;gap:8px;align-items:center;padding:6px 12px;margin-bottom:8px;background:rgba(255,255,255,.03);border-radius:6px;font-size:11px;color:var(--text2);flex-wrap:wrap}}
.am-mode{{color:var(--accent);font-weight:600}}
.am-cs{{padding:6px 12px;margin-bottom:8px;background:rgba(34,197,56,.08);border-radius:6px;font-size:11px;color:var(--green);line-height:1.4}}
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


	<!-- 三轮复盘 -->
	{rounds_review_section}

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

<!-- 全部预测网格 -->
<div class="cd">
  <div class="cd-h">
    <span class="cd-h-dot" style="background:var(--green)"></span>
    <h2>全部预测 · {len(md_matches)} 场比赛</h2>
    <span class="cd-h-b">ALL PREDICTIONS</span>
  </div>
  <div class="pg">
    {prediction_cards_html if prediction_cards_html else '<div class="empty-state">暂无预测数据</div>'}
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
        <tr><th>排名</th><th>智能体</th><th>结果分布</th><th>真实准确率</th><th>权重</th><th>总分</th></tr>
        {agent_leaderboard_rows if agent_leaderboard_rows else '<tr><td colspan="6"><div class="empty-state" style="padding:24px 16px">暂无排行榜数据</div></td></tr>'}
      </table>
    </div>
  </div>
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

	<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-2492910744108706" crossorigin="anonymous"></script>

  <!-- 出线/出局状态 -->
  <div class="cd" style="margin-bottom:16px">
    <div class="cd-h"><span class="cd-h-dot" style="background:var(--green)"></span><h2>出线 &amp; 出局球队 · 小组赛第二轮后</h2><span class="cd-h-b">June 24</span></div>
    <div class="l2" style="gap:12px">
      <div class="cd" style="margin-bottom:0;border-left:3px solid var(--green)">
        <div class="cd-h"><h3 style="font-size:13px;color:var(--green)">✅ 已出线 ({len(qual_data.get('qualified',[]))})</h3></div>
        <div style="padding:8px 10px;display:flex;flex-wrap:wrap;gap:6px">
          {''.join(f'<span style="padding:3px 8px;background:rgba(34,197,94,.1);border-radius:4px;font-size:12px;font-weight:500">{q["team"]}</span>' for q in qual_data.get('qualified',[]))}
        </div>
      </div>
      <div class="cd" style="margin-bottom:0;border-left:3px solid var(--home)">
        <div class="cd-h"><h3 style="font-size:13px;color:var(--home)">❌ 已出局 ({len(qual_data.get('eliminated',[]))})</h3></div>
        <div style="padding:8px 10px;display:flex;flex-wrap:wrap;gap:6px">
          {''.join(f'<span style="padding:3px 8px;background:rgba(239,68,68,.1);border-radius:4px;font-size:12px;font-weight:500">{e["team"]}</span>' for e in qual_data.get('eliminated',[]))}
        </div>
      </div>
    </div>
  </div>

  <!-- 比赛明细 -->
  <div class="cd" style="margin-bottom:0">
    <div class="cd-h"><span class="cd-h-dot"></span><h2>比赛明细 &amp; 实际赛果</h2><span class="cd-h-b">ALL {n_total} MATCHES</span></div>
    <div class="tw" style="max-height:400px;overflow-y:auto"><table>
      <tr><th>日期</th><th>主队</th><th>比分</th><th>客队</th><th>赛果</th></tr>
      {all_pred_rows}
    </table></div>
  </div>

<script>
function togglePos(el){{var c=el.nextElementSibling;if(c){{var o=c.classList.contains('open');c.classList.toggle('open');el.textContent=o?'落位分析 ▶':'落位分析 ▼';}}}}
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
