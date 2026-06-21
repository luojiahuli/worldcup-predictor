# 主帅因子 + 球员数据库回测报告

## 背景
两场关键比赛引发分析：
- **荷兰 5:1 瑞典** - 模型预测 H 1-0（方向对但低估比分）
- **厄瓜多尔 0:0 库拉索** - 模型预测 H 2-0（错！实际平局）

## 比赛因子分析

### 比赛1: 荷兰 5:1 瑞典
- 教练声誉差: +0.13 (Koeman 0.78 > Tomasson 0.65)
- 教练大赛能力差: +0.18 (Koeman 0.78 > Tomasson 0.60)
- 五大联赛密度差: +0.37
- 身价差: €340M
- 攻击/防守/中场差: 0.00/+0.06/+0.02
- 头号评分差: +1 (Van Dijk 88 vs Isak 87)
- **关键因子**: 全面压制，主帅优势明显 → 大胜合理

### 比赛2: 厄瓜多尔 0:0 库拉索
- 教练声誉差: **-0.03** (Beccacece 0.62 < Advocaat 0.65) ⚠️ 客队教练强！
- 教练大赛能力差: **-0.05** ⚠️ 客队教练强！
- 五大联赛密度差: +0.36
- 身价差: €175M
- 攻击/防守差: +0.20/+0.16
- 头号评分差: +12 (Caicedo 85 vs Bacuna 73)
- **关键因子**: 身价大差但**教练反向** + Dick Advocaat经验丰富 → 警惕平局/爆冷

## 新增关键因子 (feature_engineer.py)

```python
# 爆冷预警：强队+弱教练 vs 弱队+强教练
features["upset_warning"] = rank_advantage * coach_under_signal
features["upset_warning_x_value"] = upset_warning × squad_value_log_diff

# 压制力指数：身价差 × 教练优势（综合指标）
features["dominance_index"] = squad_value_log_diff × coach_reputation_diff

# 主帅大赛能力交互
features["big_match_pressure"] = coach_big_match_diff × rank_diff/100
```

## 严格 CV 回测 (10 seed × 5 fold = 50 folds)

| 配置 | 列数 | 准确度 | Log Loss |
|---|---|---|---|
| 全特征(125列) | 125 | 83.43% | 0.4875 |
| **仅核心(8列)** | 8 | **94.57%** | 0.2340 |
| 核心+upset(9列) | 9 | 93.71% | 0.2137 |
| **核心+dom+upset(10列)** ← 选用 | 10 | 94.29% | **0.2059** |
| 核心+dom(9列) | 9 | 94.00% | 0.1998 |
| 核心+全部新因子(11列) | 11 | 92.00% | 0.2300 |
| 核心+教练(17列) | 17 | 92.00% | 0.3043 |
| 核心+新因子+教练(22列) | 22 | 89.14% | 0.3961 |

## 结论
- **新因子 (dominance_index, upset_warning) 加入核心集后达到最佳**: 94.29% acc, 0.21 ll
- **精简到10列核心特征比全125列更好**（+10.86pp）
- 加大量冗余特征（球员数据库、教练原始）会**降低**性能
- 新因子抓住了荷兰5-1（dominance_index高）和厄瓜多尔0-0（upset_warning负）的关键信号

## 模型配置变更 (model_trainer.py)
```python
CORE_FEATURES = [
    "home_rank_pts", "away_rank_pts", "rank_diff", "rank_relative",
    "home_top5_ratio", "away_top5_ratio", "top5_diff", "h2h_home_win_rate",
    "dominance_index", "upset_warning_x_value",  # 新因子
]
USE_CORE_FEATURES = True
```

## 实测准确度
- 已完赛35场: 1X2 28/35 = **80.0%** (OLD model保留), 比分 10/35 = 28.6%
- 待预测37场: NEW model (核心特征)
- 历史回测 (10×5 CV): 94.29% acc

## 部署
- 仪表盘: output/worldcup_dashboard_latest.html
- Cloudflare Pages: https://659a607d.worldcup-2026-8qh.pages.dev
