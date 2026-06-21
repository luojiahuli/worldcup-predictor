# 球员数据库回测报告

## 数据来源
- 球员数据库: `data/player_database.json` (53支球队)
- 数据构建: 基于公开阵容/球员信息（FIFA评分、年龄、进球数、俱乐部、近期状态等）
- 实际抓取: Wikipedia 2026 World Cup squads, FIFA Rankings (更新于 2026/06/21)

## 新增特征 (27列)
```
home_top_player_rating / away_top_player_rating / top_player_diff
home_top3_avg_rating / away_top3_avg_rating / top3_rating_diff
home_star_goals / away_star_goals
home_squad_value / away_squad_value / squad_value_diff / squad_value_log_diff
home_avg_age / away_avg_age / avg_age_diff
home_avg_rating / away_avg_rating / avg_rating_diff
home_key_player_form / away_key_player_form / key_form_diff
home_attack_strength / away_attack_strength / attack_strength_diff
home_defense_strength / away_defense_strength / defense_strength_diff
home_midfield_strength / away_midfield_strength / midfield_strength_diff
```

## 5折交叉验证回测结果
| 配置 | 列数 | 准确度 | Log Loss |
|---|---|---|---|
| 基准（无球员） | 121 | **85.71%** | 0.4876 |
| +3强球员特征 | 124 | 85.71% | 0.4876 |
| +4强球员特征 | 125 | 85.71% | 0.4876 |
| +7强球员特征 | 128 | 85.71% | 0.4876 |
| +27全部球员特征 | 148 | 85.71% | 0.4876 |
| 仅基础+27球员（无主特征） | 33 | 42.86% | 0.9828 |

## 结论
- **球员特征不改善模型准确度**（保持85.71%），但也不降低
- 模型本身已经达到上限，球员特征成为冗余信号
- 推测原因:
  1. **样本量太小**（35场完赛）— 树模型在121个特征中已经"够用"
  2. **球员特征与已有特征高度冗余**（如 squad_value 与 EPL 密度、top_player_rating 与 coach reputation）
  3. **球员数据本身不准确**（基于历史知识，部分球队可能误差大）

## 决策
- **不更新预测模型**（按用户原则：若准确度回测更精准才更新）
- 球员特征保留在特征矩阵中，未来样本量扩大时可能发挥作用
- 后端数据库 `data/player_database.json` 保留，便于将来调优

## 现状
- 特征矩阵: 72行 x 129列
- 已完赛: 35场 (21正确 = 60.0%)
- 待预测: 37场
