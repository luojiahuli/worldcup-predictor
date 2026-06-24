# 世界杯2026预测系统 — 回测总结报告

生成时间: 2026-06-24 20:19

## 总体表现

| 指标 | 数值 |
|---|---|
| 总比赛 | 72 |
| 已完赛 | 36 |
| 待预测 | 36 |
| 方向准确率 | 27/36 = 75.0% |
| 比分准确率 | 9/36 = 25.0% |

## 分结果准确率

| 结果 | 正确/总数 | 准确率 |
|---|---|---|
| H | 17/19 | 89.5% |
| D | 4/11 | 36.4% |
| A | 6/6 | 100.0% |

## 高置信度预测 (>=50%)
- 19场已完赛高置信度, 准确率: 100.0%

## 智能体排行榜

| 排名 | 智能体 | 总分 | 准确率 | 收益率 |
|---|---|---|---|---|
| 1 | 爆冷派 | 62.5 | 26.0% | -7.0% |
| 2 | 稳健派 | 61.1 | 58.9% | 25.9% |
| 3 | 防守派 | 58.6 | 48.5% | 15.5% |
| 4 | 激进派 | 57.9 | 53.0% | 20.0% |
| 5 | 价值派 | 54.3 | 37.9% | 4.9% |
| 6 | 数据派 | 50.3 | 37.3% | 4.3% |

## 最近完赛比赛

| 日期 | 主队 | 客队 | 预测 | 实际 | 正确 |
|---|---|---|---|---|---|
| 2026-06-22 | New Zealand | Egypt | 客胜 | A | ✅ |
| 2026-06-21 | Ecuador | Curaçao | 主胜 | D | ❌ |
| 2026-06-20 | Brazil | Haiti | 主胜 | H | ✅ |
| 2026-06-20 | Turkey | Paraguay | 客胜 | A | ✅ |
| 2026-06-20 | Netherlands | Sweden | 主胜 | H | ✅ |
| 2026-06-20 | Germany | Ivory Coast | 主胜 | H | ✅ |
| 2026-06-19 | Mexico | South Korea | 主胜 | H | ✅ |
| 2026-06-19 | United States | Australia | 客胜 | H | ❌ |
| 2026-06-19 | Scotland | Morocco | 客胜 | A | ✅ |
| 2026-06-18 | Uzbekistan | Colombia | 客胜 | A | ✅ |

## 模型信息

- 特征数: 13核心特征 (含dominance_index, upset_warning_x_value)
- 模型: RF + GB + ET + LR + MLP 集成加权投票
- 混合模型: Elo + Top5密度 (权重动态优化)
- 轮次: group_3
- 智能体权重:
  - 稳健派: 1.200
  - 激进派: 1.200
  - 价值派: 1.200
  - 防守派: 0.923
  - 数据派: 1.200
  - 爆冷派: 0.929

## 高置信度待预测比赛

- Jordan vs Argentina: 客胜 (H:13% D:16% A:72%, conf:72%)
- France vs Iraq: 主胜 (H:70% D:20% A:10%, conf:70%)
- Portugal vs Uzbekistan: 主胜 (H:66% D:22% A:12%, conf:66%)
- Argentina vs Austria: 主胜 (H:62% D:25% A:13%, conf:62%)
- New Zealand vs Belgium: 客胜 (H:15% D:23% A:62%, conf:62%)
- England vs Ghana: 主胜 (H:60% D:26% A:14%, conf:60%)
- Cape Verde Islands vs Saudi Arabia: 客胜 (H:18% D:22% A:60%, conf:60%)
- Norway vs France: 客胜 (H:18% D:23% A:59%, conf:59%)
- Uruguay vs Cape Verde Islands: 主胜 (H:59% D:26% A:15%, conf:59%)
- Jordan vs Algeria: 客胜 (H:19% D:23% A:59%, conf:59%)
- Senegal vs Iraq: 主胜 (H:58% D:26% A:15%, conf:58%)
- Morocco vs Haiti: 主胜 (H:58% D:26% A:16%, conf:58%)
- Curaçao vs Ivory Coast: 客胜 (H:19% D:24% A:57%, conf:57%)
- Scotland vs Brazil: 客胜 (H:19% D:24% A:57%, conf:57%)
- Belgium vs Iran: 主胜 (H:56% D:25% A:18%, conf:56%)
- Czechia vs Mexico: 客胜 (H:19% D:25% A:56%, conf:56%)
- Spain vs Saudi Arabia: 主胜 (H:56% D:24% A:20%, conf:56%)
- Croatia vs Ghana: 主胜 (H:55% D:26% A:19%, conf:55%)
- Colombia vs Congo DR: 主胜 (H:53% D:26% A:20%, conf:53%)
- Panama vs England: 客胜 (H:18% D:29% A:53%, conf:53%)
- Congo DR vs Uzbekistan: 主胜 (H:52% D:27% A:20%, conf:52%)
- Switzerland vs Canada: 主胜 (H:52% D:24% A:24%, conf:52%)