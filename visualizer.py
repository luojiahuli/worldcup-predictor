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

RESULT_COLORS = {"H": "#ff4d4f", "D": "#faad14", "A": "#1890ff"}
RESULT_NAMES = {"H": "主胜", "D": "平局", "A": "客胜"}


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
                    "color": RESULT_COLORS.get(row.get("pred_result", ""), "#888"),
                })

    # === 最新比赛日分析 ===
    matchday_date_str = ""
    matchday_html_rows = ""
    if predictions is not None and len(predictions) > 0:
        unfinished = predictions[predictions["is_finished"] == False].copy()
        dates = pd.to_datetime(unfinished["date"])
        today = pd.Timestamp.now().normalize()
        upcoming = unfinished[dates >= today].sort_values("date")
        if len(upcoming) > 0:
            md_date = upcoming["date"].iloc[0]
            md_matches = upcoming[upcoming["date"] == md_date]
            matchday_date_str = str(md_date)[:10]
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
                bet_color = "#52c41a" if kelly > 0.08 else "#faad14" if kelly > 0 else "#888"
                rows_html.append(
                    f'<tr><td>{str(row["date"])[:10]}</td>'
                    f'<td style="font-weight:bold">{home}</td>'
                    f'<td style="font-weight:bold">{away}</td>'
                    f'<td style="color:{color};font-weight:bold">{row["pred_label"]}</td>'
                    f'<td>{conf*100:.0f}%</td>'
                    f'<td style="font-size:10px">{oH:.2f}/{oD:.2f}/{oA:.2f}</td>'
                    f'<td style="font-size:10px;max-width:180px">{"；".join(reasons)}</td>'
                    f'<td style="font-size:11px;color:{bet_color}">{bet}</td></tr>'
                )
            matchday_html_rows = "\n".join(rows_html)

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

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>🏆 世界杯2026预测系统 | {today_str}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0e1a;color:#e0e0e0;font-family:'PingFang SC','Microsoft YaHei',sans-serif;padding:16px;max-width:1300px;margin:0 auto}}
h1{{background:linear-gradient(135deg,#ffd700,#ff6b35);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:28px;text-align:center;margin-bottom:2px;padding-top:8px}}
.sub{{color:#888;text-align:center;font-size:12px;margin-bottom:16px}}
.wc-badge{{text-align:center;font-size:11px;color:#ffd700;margin-bottom:16px;letter-spacing:2px}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}}
.stat-card{{background:linear-gradient(135deg,#1a1a3e,#0f1a2e);border-radius:10px;padding:14px;text-align:center;border:1px solid #2a2a5e}}
.stat-card .v{{font-size:22px;font-weight:bold;margin-bottom:2px}}
.stat-card .l{{font-size:11px;color:#888}}
.card{{background:linear-gradient(135deg,#1a1a3e,#0f1a2e);border-radius:10px;padding:14px;border:1px solid #2a2a5e;margin-bottom:14px}}
.card h2{{color:#ffd700;font-size:14px;margin-bottom:8px;display:flex;align-items:center;gap:6px}}
.chart{{width:100%;height:300px}}
.chart-sm{{width:100%;height:220px}}
.pick-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px}}
.pick-card{{background:linear-gradient(135deg,#1a1a4e,#12122a);border-radius:8px;padding:10px;border:1px solid #2a2a5e;position:relative}}
.pick-card .date-tag{{position:absolute;top:6px;right:8px;font-size:10px;padding:1px 5px;border-radius:3px;background:#2a2a5e;color:#888}}
.pick-card .teams{{font-size:14px;font-weight:bold;color:#ffd700;margin-bottom:4px}}
.pick-card .prediction{{font-size:18px;font-weight:bold;margin:6px 0}}
.pick-card .probs{{display:flex;gap:6px;margin:4px 0}}
.pick-card .prob{{flex:1;text-align:center;font-size:11px;padding:3px;border-radius:4px;background:#0a0e1a}}
.pick-card .prob .pct{{font-size:13px;font-weight:bold}}
.pick-card .prob .lbl{{color:#888}}
.pick-card .odds{{font-size:11px;color:#888;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th{{background:#2a2a5e;color:#ffd700;padding:5px 6px;text-align:left}}
td{{padding:4px 6px;border-bottom:1px solid #1a1a3e}}
.tag-H{{color:#ff4d4f;font-weight:bold}}
.tag-D{{color:#faad14;font-weight:bold}}
.tag-A{{color:#1890ff;font-weight:bold}}
.analysis-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}}
.analysis-card{{background:#1a1a4e;border-radius:8px;padding:10px;border:1px solid #2a2a5e}}
.analysis-card .aname{{font-weight:bold;color:#ffd700;font-size:13px;margin-bottom:4px}}
.analysis-card .aval{{font-size:11px;color:#aaa;line-height:1.6}}
.bar-container{{height:6px;background:#0a0e1a;border-radius:3px;margin:2px 0;overflow:hidden}}
.bar{{height:100%;border-radius:3px;transition:width 0.5s}}
@media(max-width:600px){{.grid4{{grid-template-columns:repeat(2,1fr)}}}}
.highlight-row{{animation:fadeIn 0.5s}}
@keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
</style>
</head>
<body>

<h1>🏆 世界杯2026预测系统</h1>
<p class="wc-badge">⚽ 2026年6月11日 · 美国/加拿大/墨西哥 · 48支球队 · 104场比赛</p>
<p class="sub">{today_str} | 基于FIFA排名+历史世界杯数据+集成学习模型 | 媒体情感·社交热度·赔率分析</p>

<!-- 最新比赛日 -->
<div class="card" style="border-color:#ffd700;margin-bottom:16px">
  <h2>⚡ 最新比赛日 · {matchday_date_str}</h2>
  <div style="overflow-x:auto">
  <table>
    <tr><th>时间</th><th>主队</th><th>客队</th><th>预测</th><th>概率</th><th>赔率(主/平/客)</th><th>推荐理由</th><th>投注建议</th></tr>
    {matchday_html_rows}
  </table>
  </div>
</div>

<div class="grid4">
  <div class="stat-card"><div class="v" style="color:#ffd700">{avg_acc:.1%}</div><div class="l">模型回测准确率</div></div>
  <div class="stat-card"><div class="v" style="color:#1890ff">{n_matches}</div><div class="l">历史训练场次</div></div>
  <div class="stat-card"><div class="v" style="color:#52c41a">{len(high_conf_picks)}</div><div class="l">待预测比赛</div></div>
  <div class="stat-card"><div class="v" style="color:#ff6b35">2026.06.11</div><div class="l">开幕倒计时</div></div>
</div>

<!-- 预测结果 -->
<div class="card">
  <h2>🔮 比赛预测 (高置信度)</h2>
  <div id="picksContainer" class="pick-grid"></div>
</div>

<div class="card">
  <h2>📊 模型回测收益曲线</h2>
  <div id="backtestChart" class="chart"></div>
</div>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px">
  <div class="card">
    <h2>📰 媒体情感分析</h2>
    <div id="mediaChart" class="chart-sm"></div>
  </div>
  <div class="card">
    <h2>🔥 社交媒体热度</h2>
    <div id="socialChart" class="chart-sm"></div>
  </div>
</div>

<div class="card">
  <h2>💰 赔率对比分析 (实力悬殊最大的15场)</h2>
  <div id="oddsChart" class="chart"></div>
</div>

<div class="card">
  <h2>🏟️ 球队综合分析</h2>
  <div id="teamAnalysis" class="analysis-grid"></div>
</div>

<div class="card">
  <h2>📋 全部预测详情</h2>
  <div id="allPredictionsTable" style="max-height:500px;overflow-y:auto"></div>
</div>

<p class="sub" style="margin-top:20px;font-size:10px">
⚠️ 预测仅供参考 | 数据来源: football-data.org + FIFA Rankings | 模型: RandomForest+GradientBoosting+ExtraTrees+Logistic Ensemble
</p>

<script>
// ===== 预测卡片 =====
var picks = {json.dumps(high_conf_picks, ensure_ascii=False)};
var ph = document.getElementById('picksContainer');
if (picks.length === 0) {{
  ph.innerHTML = '<div style="text-align:center;padding:20px;color:#888;font-size:13px">暂无预测数据，请先运行主流程</div>';
}} else {{
  picks.forEach(function(p){{
    var predColor = p.pred_r === 'H' ? '#ff4d4f' : p.pred_r === 'D' ? '#faad14' : '#1890ff';
    var div = document.createElement('div');
    div.className = 'pick-card';
    var probsHtml = '<div class="probs">' +
      '<div class="prob"><div class="pct" '+(parseFloat(p.pH)>40?'style="color:#ff4d4f;font-weight:bold"':'')+'>'+p.pH+'%</div><div class="lbl">主胜</div><div class="odds">@'+p.oH+'</div></div>' +
      '<div class="prob"><div class="pct" '+(parseFloat(p.pD)>35?'style="color:#faad14;font-weight:bold"':'')+'>'+p.pD+'%</div><div class="lbl">平局</div><div class="odds">@'+p.oD+'</div></div>' +
      '<div class="prob"><div class="pct" '+(parseFloat(p.pA)>40?'style="color:#1890ff;font-weight:bold"':'')+'>'+p.pA+'%</div><div class="lbl">客胜</div><div class="odds">@'+p.oA+'</div></div>' +
      '</div>';
    div.innerHTML = '<span class="date-tag">'+p.date+'</span>' +
      '<div class="teams">'+p.home+' vs '+p.away+'</div>' +
      '<div class="prediction" style="color:'+predColor+'">🏆 '+p.pred+'</div>' +
      probsHtml +
      '<div style="font-size:10px;color:#888;margin-top:2px">置信度: '+p.conf+'%</div>';
    ph.appendChild(div);
  }});
}}

// ===== 回测收益曲线 =====
var cumRet = {cum_ret_str};
var bc = echarts.init(document.getElementById('backtestChart'));
bc.setOption({{
  backgroundColor:'transparent',
  tooltip:{{trigger:'axis',formatter:function(p){{return '<b>投注 #'+p[0].axisValue+'</b><br/>'+p[0].marker+' 累计收益: ¥'+p[0].value.toFixed(0);}}}},
  grid:{{left:'10%',right:'5%',top:'10%',bottom:'12%'}},
  xAxis:{{type:'category',data:Array.from({{length:cumRet.length}},(_,i)=>'#'+(i+1)),axisLabel:{{fontSize:9,color:'#ccc',show:false}},axisLine:{{lineStyle:{{color:'#333'}}}}}},
  yAxis:{{type:'value',name:'累计盈亏 (¥)',nameTextStyle:{{color:'#888'}},splitLine:{{lineStyle:{{color:'#1a1a3e'}}}},axisLabel:{{color:'#888',formatter:'¥{{value}}'}}}},
  series:[{{
    name:'策略收益',type:'line',data:cumRet,smooth:true,
    lineStyle:{{width:2.5,color:'#ffd700'}},
    areaStyle:{{color:{{type:'linear',x:0,y:0,x2:0,y2:1,colorStops:[{{offset:0,color:'#ffd70044'}},{{offset:1,color:'#ffd70000'}}]}}}},
    markLine:{{data:[{{yAxis:0,lineStyle:{{color:'#555',type:'dashed'}},label:{{formatter:'盈亏平衡',color:'#888'}}}}]}}
  }}]
}});

// ===== 媒体情感 =====
var mediaData = {media_str};
var topMedia = mediaData.slice().sort(function(a,b){{return b.sentiment_score - a.sentiment_score}}).slice(0,15);
var mc = echarts.init(document.getElementById('mediaChart'));
mc.setOption({{
  backgroundColor:'transparent',
  tooltip:{{trigger:'axis',formatter:function(p){{return p[0].name+'<br/>'+p[0].marker+' 情感得分: '+(p[0].value*100).toFixed(0)+'%'}}}},
  grid:{{left:'20%',right:'5%',top:'5%',bottom:'15%'}},
  xAxis:{{type:'value',max:0.9,axisLabel:{{color:'#888',formatter:'{{value}}'}},splitLine:{{lineStyle:{{color:'#1a1a3e'}}}}}},
  yAxis:{{type:'category',data:topMedia.map(function(d){{return d.team}}),axisLabel:{{color:'#ccc',fontSize:10}},axisLine:{{lineStyle:{{color:'#333'}}}}}},
  series:[{{
    type:'bar',data:topMedia.map(function(d){{return d.sentiment_score}}),
    barWidth:'50%',
    itemStyle:{{color:function(p){{return p.value>0.5?'#52c41a':'#ff4d4f'}}}},
    label:{{show:true,position:'right',color:'#ffd700',fontSize:10,formatter:function(p){{return (p.value*100).toFixed(0)+'%'}}}}
  }}]
}});

// ===== 社交热度 =====
var socialData = {social_str};
var sc = echarts.init(document.getElementById('socialChart'));
sc.setOption({{
  backgroundColor:'transparent',
  tooltip:{{trigger:'axis'}},
  grid:{{left:'20%',right:'5%',top:'5%',bottom:'15%'}},
  xAxis:{{type:'value',max:1.0,axisLabel:{{color:'#888',formatter:'{{value}}'}},splitLine:{{lineStyle:{{color:'#1a1a3e'}}}}}},
  yAxis:{{type:'category',data:socialData.map(function(d){{return d.team}}),axisLabel:{{color:'#ccc',fontSize:10}},axisLine:{{lineStyle:{{color:'#333'}}}}}},
  series:[{{
    type:'bar',data:socialData.map(function(d){{return d.heat_score}}),
    barWidth:'40%',
    itemStyle:{{color:{{type:'linear',x:0,y:0,x2:1,y2:0,colorStops:[{{offset:0,color:'#ff6b35'}},{{offset:1,color:'#ffd700'}}]}}}},
    label:{{show:true,position:'right',color:'#fff',fontSize:10,formatter:function(p){{return (p.value*100).toFixed(0)+'%'}}}}
  }}]
}});

// ===== 赔率对比 =====
var oddsData = {odds_str};
if (oddsData.length > 0) {{
  var oc = echarts.init(document.getElementById('oddsChart'));
  oc.setOption({{
    backgroundColor:'transparent',
    tooltip:{{trigger:'axis'}},
    legend:{{data:['主胜赔率','平局赔率','客胜赔率'],textStyle:{{color:'#ccc'}},top:0}},
    grid:{{left:'8%',right:'5%',top:'15%',bottom:'15%'}},
    xAxis:{{type:'category',data:oddsData.map(function(d){{return d.home+' vs '+d.away}}),axisLabel:{{fontSize:9,color:'#ccc',rotate:30}},axisLine:{{lineStyle:{{color:'#333'}}}}}},
    yAxis:{{type:'value',name:'赔率',nameTextStyle:{{color:'#888'}},splitLine:{{lineStyle:{{color:'#1a1a3e'}}}},axisLabel:{{color:'#888'}}}},
    series:[
      {{name:'主胜赔率',type:'bar',data:oddsData.map(function(d){{return d.oH}}),barWidth:'20%',itemStyle:{{color:'#ff4d4f'}},label:{{show:true,position:'top',fontSize:9,color:'#ff4d4f',formatter:'{{c}}'}}}},
      {{name:'平局赔率',type:'bar',data:oddsData.map(function(d){{return d.oD}}),barWidth:'20%',itemStyle:{{color:'#faad14'}},label:{{show:true,position:'top',fontSize:9,color:'#faad14',formatter:'{{c}}'}}}},
      {{name:'客胜赔率',type:'bar',data:oddsData.map(function(d){{return d.oA}}),barWidth:'20%',itemStyle:{{color:'#1890ff'}},label:{{show:true,position:'top',fontSize:9,color:'#1890ff',formatter:'{{c}}'}}}}
    ]
  }});
}}

// ===== 球队综合分析 =====
var teamData = mediaData.concat(socialData).reduce(function(acc, item) {{
  var t = item.team || item.team;
  if (!acc[t]) {{
    acc[t] = {{team:t, sentiment:0.5, heat:0.5, volume:0}};
  }}
  if (item.sentiment_score !== undefined) acc[t].sentiment = item.sentiment_score;
  if (item.heat_score !== undefined) acc[t].heat = item.heat_score;
  if (item.news_volume !== undefined) acc[t].volume = item.news_volume;
  if (item.active_users_24h !== undefined) acc[t].volume = item.active_users_24h;
  return acc;
}}, {{}});
var teams = Object.values(teamData).sort(function(a,b){{return (b.sentiment+b.heat) - (a.sentiment+a.heat)}}).slice(0,20);
var tc = document.getElementById('teamAnalysis');
if (teams.length > 0) {{
  teams.forEach(function(t){{
    var score = (t.sentiment + t.heat) / 2 * 100;
    var div = document.createElement('div');
    div.className = 'analysis-card';
    div.innerHTML = '<div class="aname">'+t.team+'</div>' +
      '<div class="aval">媒体情感: '+(t.sentiment*100).toFixed(0)+'%</div>' +
      '<div class="bar-container"><div class="bar" style="width:'+(t.sentiment*100)+'%;background:'+(t.sentiment>0.5?'#52c41a':'#ff4d4f')+'"></div></div>' +
      '<div class="aval">社交热度: '+(t.heat*100).toFixed(0)+'%</div>' +
      '<div class="bar-container"><div class="bar" style="width:'+(t.heat*100)+'%;background:linear-gradient(90deg,#ff6b35,#ffd700)"></div></div>' +
      '<div class="aval">综合评分: '+score.toFixed(0)+'/100</div>';
    tc.appendChild(div);
  }});
}} else {{
  tc.innerHTML = '<div style="text-align:center;padding:20px;color:#888;font-size:13px">暂无分析数据</div>';
}}

// ===== 全部预测详情 =====
var predData = [{pred_rows}];
var at = document.getElementById('allPredictionsTable');
if (predData.length > 0) {{
  var html = '<table><tr><th>日期</th><th>主队</th><th>客队</th><th>预测</th><th>主胜</th><th>平局</th><th>客胜</th><th>置信度</th><th>主胜赔</th><th>平赔</th><th>客胜赔</th></tr>';
  predData.forEach(function(r){{
    var predColor = r.pred_r === 'H' ? '#ff4d4f' : r.pred_r === 'D' ? '#faad14' : '#1890ff';
    html += '<tr><td>'+r.date+'</td><td>'+r.home+'</td><td>'+r.away+'</td>' +
      '<td style="color:'+predColor+';font-weight:bold">'+r.pred+'</td>' +
      '<td>'+r.ph+'%</td><td>'+r.pd+'%</td><td>'+r.pa+'%</td>' +
      '<td>'+r.conf+'%</td><td>'+r.oh+'</td><td>'+r.od+'</td><td>'+r.oa+'</td></tr>';
  }});
  html += '</table>';
  at.innerHTML = html;
}} else {{
  at.innerHTML = '<div style="text-align:center;padding:20px;color:#888">暂无预测数据</div>';
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

    log.info(f"✅ 仪表盘已生成: {path}")
    return path


if __name__ == "__main__":
    data = load_all_data()
    if data:
        generate_dashboard(data)
    else:
        log.error("无数据，请先运行 main.py")
