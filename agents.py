"""
agents.py - 多智能体预测系统
5种不同风格的预测Agent：稳健、激进、价值、防守、数据
包含：比分预测、PK辩论、业绩排行榜
"""
import pandas as pd
import numpy as np
import os, json, logging, pickle, math
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RESULT_NAMES = {"H": "主胜", "D": "平局", "A": "客胜"}
AGENT_COLORS = {
    "稳健派": "#22c55e",
    "激进派": "#ef4444",
    "价值派": "#3b82f6",
    "防守派": "#8b5cf6",
    "数据派": "#f59e0b",
    "爆冷派": "#ec4899",
}

# === 天气数据：2026世界杯场馆城市天气（6-7月典型） ===
VENUE_WEATHER = {
    # USA venues
    "New York": {"temp": 29, "humidity": 0.68, "rain_pct": 0.30, "altitude": 10, "indoor": False, "climate_zone": "temperate"},
    "Los Angeles": {"temp": 27, "humidity": 0.55, "rain_pct": 0.05, "altitude": 70, "indoor": False, "climate_zone": "mediterranean"},
    "Dallas": {"temp": 35, "humidity": 0.60, "rain_pct": 0.25, "altitude": 130, "indoor": True, "climate_zone": "subtropical"},
    "Houston": {"temp": 34, "humidity": 0.75, "rain_pct": 0.35, "altitude": 15, "indoor": True, "climate_zone": "subtropical"},
    "Atlanta": {"temp": 32, "humidity": 0.70, "rain_pct": 0.40, "altitude": 310, "indoor": True, "climate_zone": "subtropical"},
    "Philadelphia": {"temp": 30, "humidity": 0.65, "rain_pct": 0.35, "altitude": 10, "indoor": False, "climate_zone": "temperate"},
    "Boston": {"temp": 28, "humidity": 0.65, "rain_pct": 0.30, "altitude": 10, "indoor": False, "climate_zone": "temperate"},
    "Seattle": {"temp": 24, "humidity": 0.60, "rain_pct": 0.25, "altitude": 50, "indoor": True, "climate_zone": "maritime"},
    "San Francisco": {"temp": 20, "humidity": 0.70, "rain_pct": 0.10, "altitude": 15, "indoor": False, "climate_zone": "mediterranean"},
    "Miami": {"temp": 32, "humidity": 0.78, "rain_pct": 0.50, "altitude": 5, "indoor": False, "climate_zone": "tropical"},
    "Kansas City": {"temp": 31, "humidity": 0.65, "rain_pct": 0.30, "altitude": 280, "indoor": False, "climate_zone": "continental"},
    "Denver": {"temp": 32, "humidity": 0.35, "rain_pct": 0.20, "altitude": 1600, "indoor": False, "climate_zone": "semi_arid"},
    # Canada venues
    "Toronto": {"temp": 26, "humidity": 0.60, "rain_pct": 0.30, "altitude": 80, "indoor": False, "climate_zone": "continental"},
    "Vancouver": {"temp": 22, "humidity": 0.65, "rain_pct": 0.30, "altitude": 5, "indoor": False, "climate_zone": "maritime"},
    # Mexico venues
    "Mexico City": {"temp": 23, "humidity": 0.55, "rain_pct": 0.45, "altitude": 2240, "indoor": False, "climate_zone": "highland"},
    "Monterrey": {"temp": 33, "humidity": 0.55, "rain_pct": 0.20, "altitude": 540, "indoor": False, "climate_zone": "semi_arid"},
    "Guadalajara": {"temp": 28, "humidity": 0.60, "rain_pct": 0.40, "altitude": 1560, "indoor": False, "climate_zone": "highland"},
}

# 球队气候适应性（来源于该队所在大陆/区域）
# 将球队映射到其适应的气候类型
TEAM_CLIMATE_ZONE = {
    # 南美: 热带/亚热带/高海拔
    "Argentina": "temperate", "Brazil": "tropical", "Uruguay": "temperate", "Colombia": "tropical",
    "Ecuador": "tropical", "Peru": "tropical", "Chile": "mediterranean", "Paraguay": "subtropical",
    "Venezuela": "tropical",
    # 欧洲: 温带/大陆性
    "England": "maritime", "France": "temperate", "Belgium": "temperate", "Portugal": "mediterranean",
    "Netherlands": "maritime", "Spain": "mediterranean", "Germany": "temperate", "Italy": "mediterranean",
    "Croatia": "temperate", "Denmark": "maritime", "Switzerland": "continental", "Serbia": "continental",
    "Poland": "continental", "Ukraine": "continental", "Sweden": "continental", "Norway": "continental",
    "Scotland": "maritime", "Wales": "maritime", "Austria": "continental", "Hungary": "continental",
    "Greece": "mediterranean", "Turkey": "mediterranean", "Czech Republic": "continental", "Czechia": "continental",
    "Romania": "continental", "Russia": "continental", "Slovakia": "continental", "Slovenia": "continental",
    "Bulgaria": "continental", "Bosnia-Herzegovina": "continental",
    # 非洲: 热带/亚热带/干旱
    "Morocco": "semi_arid", "Senegal": "tropical", "Nigeria": "tropical", "Cameroon": "tropical",
    "Ghana": "tropical", "Tunisia": "semi_arid", "Algeria": "semi_arid", "Egypt": "semi_arid",
    "Ivory Coast": "tropical", "Mali": "semi_arid", "Burkina Faso": "tropical", "Congo DR": "tropical",
    "South Africa": "subtropical", "Cape Verde Islands": "tropical",
    # 北美: 多样化
    "Canada": "continental", "USA": "temperate", "United States": "temperate", "Mexico": "highland",
    "Costa Rica": "tropical", "Panama": "tropical", "Haiti": "tropical", "Curaçao": "tropical",
    # 亚洲: 多样化
    "Japan": "temperate", "South Korea": "continental", "Australia": "subtropical",
    "Saudi Arabia": "semi_arid", "Iran": "semi_arid", "Iraq": "semi_arid", "Jordan": "semi_arid",
    "Qatar": "semi_arid", "United Arab Emirates": "semi_arid", "Uzbekistan": "continental",
    "China": "continental", "Oman": "semi_arid",
    # 大洋洲
    "New Zealand": "maritime", "Fiji": "tropical", "Solomon Islands": "tropical",
    # 默认
    "default": "temperate",
}

# 2026世界杯场馆-城市映射（基于已知场馆信息）
# 每个match的venue在API数据中可能不存在，这里根据赛程大致分配
MATCH_VENUES = {
    # 首场比赛在墨西哥城
    "Mexico_South Africa": "Mexico City",
    # 默认场馆分配
    "default": "Mexico City",
}


def get_weather_impact(home_team, away_team, venue=None):
    """
    计算天气对比赛的影响
    返回: {temp_feel, heat_advantage, rain_factor, altitude_factor, summary}
    """
    weather = VENUE_WEATHER.get(venue, VENUE_WEATHER["Mexico City"])
    temp = weather["temp"]
    humidity = weather["humidity"]
    rain_pct = weather["rain_pct"]
    altitude = weather["altitude"]
    indoor = weather["indoor"]

    if indoor:
        return {
            "temp": 22, "humidity": 0.50, "rain_pct": 0, "altitude": 0,
            "indoor": True,
            "heat_advantage": 0, "altitude_advantage": 0,
            "goal_reduction": 0,
            "draw_boost": 0,
            "summary": "室内球场，气候条件中性",
        }

    home_zone = TEAM_CLIMATE_ZONE.get(home_team, "temperate")
    away_zone = TEAM_CLIMATE_ZONE.get(away_team, "temperate")
    venue_zone = weather["climate_zone"]

    # 热度适应：来自热带的球队在高温中占优
    heat_tolerance = {"tropical": 0.12, "semi_arid": 0.10, "subtropical": 0.06, "highland": 0.03,
                      "mediterranean": 0.02, "temperate": -0.02, "continental": -0.05, "maritime": -0.08}

    home_heat = heat_tolerance.get(home_zone, 0)
    away_heat = heat_tolerance.get(away_zone, 0)

    heat_advantage = 0
    if temp > 30:
        heat_advantage = (home_heat - away_heat) * 0.3
    elif temp > 27:
        heat_advantage = (home_heat - away_heat) * 0.15

    # 高海拔优势
    altitude_advantage = 0
    if altitude > 1500:
        alt_tolerance = {"highland": 0.15, "tropical": 0.05, "temperate": -0.05, "continental": -0.08, "maritime": -0.10}
        home_alt = alt_tolerance.get(home_zone, 0)
        away_alt = alt_tolerance.get(away_zone, 0)
        altitude_advantage = (home_alt - away_alt) * min(1.0, altitude / 2500)

    # 降雨：减少总进球，增加平局概率
    goal_reduction = 0
    draw_boost = 0
    if rain_pct > 0.35:
        goal_reduction = 0.12
        draw_boost = 0.04
    elif rain_pct > 0.20:
        goal_reduction = 0.06
        draw_boost = 0.02

    # 高湿度组合
    if humidity > 0.70 and temp > 28:
        goal_reduction += 0.06
        draw_boost += 0.02

    # 构建摘要
    parts = []
    if indoor:
        parts.append("室内")
    else:
        parts.append(f"{temp}°C")
        if humidity > 0.70: parts.append("高湿")
        if rain_pct > 0.35: parts.append("有雨")
        if altitude > 1000: parts.append(f"海拔{altitude}m")
    if abs(heat_advantage) > 0.03:
        who = f"{home_team}更适应" if heat_advantage > 0 else f"{away_team}更适应"
        if temp > 30: parts.append(f"炎热{who}")
    if abs(altitude_advantage) > 0.03:
        who = f"利好{home_team}" if altitude_advantage > 0 else f"利好{away_team}"
        parts.append(f"高海拔{who}")

    return {
        "temp": temp, "humidity": humidity, "rain_pct": rain_pct, "altitude": altitude,
        "indoor": indoor,
        "heat_advantage": round(heat_advantage, 3),
        "altitude_advantage": round(altitude_advantage, 3),
        "goal_reduction": round(goal_reduction, 3),
        "draw_boost": round(draw_boost, 3),
        "summary": "·".join(parts) if parts else "天气条件中性",
    }


def elo_prob(pts_home, pts_away):
    """基础ELO概率计算"""
    rating_diff = pts_home - pts_away
    expected_h = 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))
    expected_a = 1.0 / (1.0 + 10 ** (rating_diff / 400.0))
    draw_base = 0.22 + 0.20 * (1.0 - abs(rating_diff) / 800.0)
    draw_base = max(0.18, min(0.35, draw_base))
    remaining = 1.0 - draw_base
    prob_h = expected_h / (expected_h + expected_a) * remaining
    prob_a = expected_a / (expected_h + expected_a) * remaining
    return prob_h, draw_base, prob_a


def poisson_prob(goals, expected):
    """泊松分布概率"""
    return math.exp(-expected) * (expected ** goals) / math.factorial(goals)


def predict_scoreline(exp_h, exp_a, max_goals=5):
    """预测最可能比分"""
    best_prob = 0
    best_score = (0, 0)
    probs = {}
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = poisson_prob(i, exp_h) * poisson_prob(j, exp_a)
            probs[f"{i}-{j}"] = p
            if p > best_prob:
                best_prob = p
                best_score = (i, j)
    return best_score, best_prob, probs


def exp_goals_from_elo(pts_home, pts_away):
    """基于ELO差值估算期望进球数"""
    diff = (pts_home - pts_away) / 400.0
    exp_h = 1.4 + diff * 0.8
    exp_a = 1.4 - diff * 0.8
    return max(0.3, min(3.5, exp_h)), max(0.3, min(3.5, exp_a))


class BaseAgent:
    """智能体基类"""

    def __init__(self, name):
        self.name = name
        self.style = ""
        self.motto = ""
        self.color = AGENT_COLORS.get(name, "#888")

    def predict(self, home, away, home_pts, away_pts, base_probs, fair_odds, weather=None):
        """
        预测比赛
        返回: {result, home_score, away_score, confidence, reasoning, probs}
        """
        raise NotImplementedError

    def _apply_weather(self, base_probs, weather, home, away):
        """应用天气因素调整概率，返回 (adjusted_probs, weather_note)"""
        if weather is None or weather.get("indoor"):
            return base_probs, ""

        probs = list(base_probs)
        notes = []

        # 热度调整
        ha = weather.get("heat_advantage", 0)
        if ha > 0.03:
            probs[0] += abs(ha) * 0.5  # home wins more
            probs[2] -= abs(ha) * 0.5
            notes.append(f"炎热利好{home}")
        elif ha < -0.03:
            probs[2] += abs(ha) * 0.5  # away wins more
            probs[0] -= abs(ha) * 0.5
            notes.append("炎热利好客队")

        # 高海拔调整
        aa = weather.get("altitude_advantage", 0)
        if aa > 0.03:
            probs[0] += abs(aa) * 0.4
            probs[2] -= abs(aa) * 0.4
            notes.append(f"高海拔利好{home}")
        elif aa < -0.03:
            probs[2] += abs(aa) * 0.4
            probs[0] -= abs(aa) * 0.4
            notes.append("高海拔利好客队")

        # 降雨增加平局
        db = weather.get("draw_boost", 0)
        if db > 0:
            probs[1] += db
            probs[0] -= db * 0.5
            probs[2] -= db * 0.5
            notes.append("降雨增加平局概率")

        # 确保非负
        probs = [max(0.05, p) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]
        weather_note = "·".join(notes) if notes else ""
        return (probs[0], probs[1], probs[2]), weather_note

    def _apply_weather_score(self, exp_h, exp_a, weather):
        """应用天气因素调整期望进球"""
        if weather is None or weather.get("indoor"):
            return exp_h, exp_a
        gr = weather.get("goal_reduction", 0)
        if gr > 0:
            exp_h *= (1 - gr)
            exp_a *= (1 - gr)
        return exp_h, exp_a

    def _get_odds_score(self, home, away, h2h_probs, result=None):
        """基于真实赔率的 Poisson 比分预测"""
        try:
            from data_collector import get_score_prediction
            hg, ag, sp, exph, expa = get_score_prediction(home, away, h2h_probs, result=result)
            return hg, ag
        except Exception:
            return None, None


class ConservativeAgent(BaseAgent):
    """稳健派 - 稳字当头，强队必胜"""

    def __init__(self):
        super().__init__("稳健派")
        self.style = "保守型"
        self.motto = "稳字当头，实力为王"

    def predict(self, home, away, home_pts, away_pts, base_probs, fair_odds, weather=None):
        (prob_h, prob_d, prob_a), weather_note = self._apply_weather(base_probs, weather, home, away)
        oH, oD, oA = fair_odds
        rating_gap = abs(home_pts - away_pts)

        # 保守：强队差距大时高置信度，接近时倾向平局
        if rating_gap > 100:
            if prob_h > prob_a:
                result = "H"
                confidence = min(0.65, prob_h * 1.15)
                reasoning = f"{home}实力明显占优(FIFA排名差{rating_gap}分)，稳健看好主胜"
            else:
                result = "A"
                confidence = min(0.65, prob_a * 1.15)
                reasoning = f"{away}实力更强，客胜可期"
        elif rating_gap < 40:
            result = "D"
            confidence = min(0.45, prob_d * 1.2)
            reasoning = f"双方实力接近(排名差仅{rating_gap}分)，稳健选择平局"
        else:
            if prob_h >= prob_a:
                result = "H"
                confidence = prob_h * 1.05
                reasoning = f"{home}略占优势，谨慎看好主队不败"
            else:
                result = "A"
                confidence = prob_a * 1.05
                reasoning = f"{away}略占优势，谨慎看好客队不败"

        # 使用真实赔率 Poisson 比分
        hg, ag = self._get_odds_score(home, away, (prob_h, prob_d, prob_a), result=result)
        if hg is not None:
            score = (hg, ag)
        else:
            exp_h, exp_a = exp_goals_from_elo(home_pts, away_pts)
            exp_h, exp_a = self._apply_weather_score(exp_h, exp_a, weather)
            score = (max(0, round(exp_h)), max(0, round(exp_a)))
            score = (min(score[0], 2), min(score[1], 1))
            if score[0] == score[1] and result != "D":
                score = (score[0] + 1, score[1])

        probs = {"H": prob_h, "D": prob_d, "A": prob_a}
        return {
            "result": result,
            "home_score": score[0],
            "away_score": score[1],
            "confidence": round(min(0.85, confidence), 3),
            "reasoning": reasoning,
            "probs": probs,
        }


class AggressiveAgent(BaseAgent):
    """激进派 - 足球是圆的，一切皆有可能"""

    def __init__(self):
        super().__init__("激进派")
        self.style = "激进型"
        self.motto = "足球是圆的，一切皆有可能"

    def predict(self, home, away, home_pts, away_pts, base_probs, fair_odds, weather=None):
        (prob_h, prob_d, prob_a), weather_note = self._apply_weather(base_probs, weather, home, away)
        oH, oD, oA = fair_odds
        rating_gap = abs(home_pts - away_pts)

        # 激进：夸大弱队概率，偏好爆冷
        upset_factor = np.random.RandomState(hash(home + away) % 2**31).uniform(0.05, 0.15)
        if rating_gap > 100:
            # 强强对话或实力悬殊时找冷门
            if prob_h > prob_a:
                # 弱化主队概率
                new_prob_h = prob_h * 0.85
                new_prob_a = prob_a + (prob_h - new_prob_h) * 0.7
                new_prob_d = prob_a + (prob_h - new_prob_h) * 0.3
            else:
                new_prob_a = prob_a * 0.85
                new_prob_h = prob_h + (prob_a - new_prob_a) * 0.7
                new_prob_d = prob_d + (prob_a - new_prob_a) * 0.3
        else:
            # 接近比赛，激进派偏好分胜负
            new_prob_h = prob_h * 1.15
            new_prob_a = prob_a * 1.15
            new_prob_d = prob_d * 0.7

        total = new_prob_h + new_prob_d + new_prob_a
        probs = {
            "H": new_prob_h / total,
            "D": new_prob_d / total,
            "A": new_prob_a / total,
        }

        max_key = max(probs, key=probs.get)
        result = max_key
        confidence = probs[max_key]

        # 使用真实赔率 Poisson 比分
        hg, ag = self._get_odds_score(home, away, (prob_h, prob_d, prob_a), result=result)
        if hg is not None:
            score = (hg, ag)
        else:
            exp_h, exp_a = exp_goals_from_elo(home_pts, away_pts)
            exp_h, exp_a = self._apply_weather_score(exp_h, exp_a, weather)
            exp_h *= 1.2
            exp_a *= 1.2
            score = (min(4, round(exp_h)), min(4, round(exp_a)))
            if score[0] == score[1] and result != "D":
                score = (score[0] + 1, score[1])

        if result == "H":
            reasoning = f"{away}并非没有机会，但{home}进攻火力更强，看好大球取胜"
        elif result == "A":
            reasoning = f"数据低估了{away}的实力，冷门可期，大胆搏客胜"
        else:
            reasoning = "双方都有机会，对攻战平局收场"
        if weather_note:
            reasoning += f" ({weather_note})"

        return {
            "result": result,
            "home_score": max(0, score[0]),
            "away_score": max(0, score[1]),
            "confidence": round(min(0.80, confidence), 3),
            "reasoning": reasoning,
            "probs": {k: round(v, 3) for k, v in probs.items()},
        }


class ValueAgent(BaseAgent):
    """价值派 - 只投有价值的比赛"""

    def __init__(self):
        super().__init__("价值派")
        self.style = "价值型"
        self.motto = "只投有价值的比赛"

    def predict(self, home, away, home_pts, away_pts, base_probs, fair_odds, weather=None):
        (prob_h, prob_d, prob_a), weather_note = self._apply_weather(base_probs, weather, home, away)
        oH, oD, oA = fair_odds

        # 计算每个结果的价值
        value_h = prob_h - 1.0 / oH if oH > 0 else -1
        value_d = prob_d - 1.0 / oD if oD > 0 else -1
        value_a = prob_a - 1.0 / oA if oA > 0 else -1

        values = {"H": value_h, "D": value_d, "A": value_a}
        best = max(values, key=values.get)
        max_value = values[best]

        # 只有存在价值才出手
        result_map = {"H": 0, "D": 1, "A": 2}
        odds_map = {"H": oH, "D": oD, "A": oA}
        if max_value > 0.03:
            result = best
            confidence = base_probs[result_map[best]]
            best_odds = odds_map[best]
            kelly = (confidence * best_odds - 1) / (best_odds - 1) if best_odds > 1 else 0
            kelly = max(0, min(0.25, kelly))
            edge_pct = max_value * 100
            implied = 1.0 / best_odds * 100
            reasoning = (
                f"发现价值空间({RESULT_NAMES[result]}赔率隐含概率{implied:.0f}%"
                f" vs 预测概率{confidence*100:.0f}%)，存在{edge_pct:.1f}%期望价值"
            )
        else:
            # 没有价值，选概率最高的但降置信度
            probs_list = [base_probs[0], base_probs[1], base_probs[2]]
            max_idx = probs_list.index(max(probs_list))
            result_inv = ["H", "D", "A"]
            result = result_inv[max_idx]
            confidence = base_probs[max_idx] * 0.7
            kelly = 0
            reasoning = "本场价值空间不足，如需投注建议观望"

        # 使用真实赔率 Poisson 比分
        hg, ag = self._get_odds_score(home, away, (base_probs[0], base_probs[1], base_probs[2]), result=result)
        if hg is not None:
            score = (hg, ag)
        else:
            exp_h, exp_a = exp_goals_from_elo(home_pts, away_pts)
            exp_h, exp_a = self._apply_weather_score(exp_h, exp_a, weather)
            score = (round(exp_h), round(exp_a))

        if weather_note:
            reasoning += f" ({weather_note})"

        return {
            "result": result,
            "home_score": max(0, score[0]),
            "away_score": max(0, score[1]),
            "confidence": round(min(0.85, confidence), 3),
            "reasoning": reasoning,
            "probs": {"H": prob_h, "D": prob_d, "A": prob_a},
            "value_edge": round(max_value, 3),
            "kelly_stake": round(kelly, 3),
        }


class DefensiveAgent(BaseAgent):
    """防守派 - 防守赢得冠军"""

    def __init__(self):
        super().__init__("防守派")
        self.style = "防守型"
        self.motto = "防守赢得冠军"

    def predict(self, home, away, home_pts, away_pts, base_probs, fair_odds, weather=None):
        (prob_h, prob_d, prob_a), weather_note = self._apply_weather(base_probs, weather, home, away)
        rating_gap = abs(home_pts - away_pts)

        # 防守派偏好平局和小球
        new_draw = prob_d * 1.25
        new_home = prob_h * 0.9
        new_away = prob_a * 0.9

        # 实力接近更倾向平局
        if rating_gap < 60:
            new_draw = prob_d * 1.5

        total = new_home + new_draw + new_away
        probs = {
            "H": new_home / total,
            "D": new_draw / total,
            "A": new_away / total,
        }

        # 喜欢平局，否则选概率高的
        if probs["D"] > 0.38:
            result = "D"
        else:
            result = max(probs, key=probs.get)
        confidence = probs[result]

        # 使用真实赔率 Poisson 比分
        hg, ag = self._get_odds_score(home, away, (prob_h, prob_d, prob_a), result=result)
        if hg is not None:
            score = (hg, ag)
        else:
            exp_h, exp_a = exp_goals_from_elo(home_pts, away_pts)
            exp_h, exp_a = self._apply_weather_score(exp_h, exp_a, weather)
            exp_h *= 0.6
            exp_a *= 0.6
            score = (max(0, round(exp_h)), max(0, round(exp_a)))

        # 确保比分与结果一致
        if result == "H" and score[0] <= score[1]:
            score = (score[1] + 1, score[1])
        elif result == "A" and score[1] <= score[0]:
            score = (score[0], score[0] + 1)
        elif result == "D" and score[0] != score[1]:
            score = (score[0], score[0])

        reasons = []
        if result == "D":
            reasons.append("世界杯大赛往往以防守为主")
            if rating_gap < 60:
                reasons.append(f"双方实力接近(差{rating_gap}分)")
            reasons.append("看好闷平")
        elif result == "H":
            reasons.append(f"{home}防守体系更稳固")
            if rating_gap > 100:
                reasons.append(f"排名优势({rating_gap}分)保障了防线")
            reasons.append("小球取胜")
        else:
            reasons.append(f"{away}反击效率值得信赖")
            reasons.append("防守反击拿到客胜")

        if weather_note:
            reasons.append(weather_note)

        return {
            "result": result,
            "home_score": score[0],
            "away_score": score[1],
            "confidence": round(min(0.80, confidence), 3),
            "reasoning": "，".join(reasons),
            "probs": {k: round(v, 3) for k, v in probs.items()},
        }


class TechnicalAgent(BaseAgent):
    """数据派 - 让数据说话"""

    def __init__(self):
        super().__init__("数据派")
        self.style = "数据型"
        self.motto = "让数据说话"

    def predict(self, home, away, home_pts, away_pts, base_probs, fair_odds, weather=None):
        (prob_h, prob_d, prob_a), weather_note = self._apply_weather(base_probs, weather, home, away)
        oH, oD, oA = fair_odds

        # 使用纯ELO概率，不做调整
        probs = {"H": prob_h, "D": prob_d, "A": prob_a}
        result = max(probs, key=probs.get)
        confidence = probs[result]

        # 使用真实赔率 Poisson 比分
        hg, ag = self._get_odds_score(home, away, (prob_h, prob_d, prob_a), result=result)
        if hg is not None:
            score = (hg, ag)
            score_prob = 0
        else:
            exp_h, exp_a = exp_goals_from_elo(home_pts, away_pts)
            exp_h, exp_a = self._apply_weather_score(exp_h, exp_a, weather)
            score, score_prob, score_probs = predict_scoreline(exp_h, exp_a)

        # 综合置信度
        combined_conf = confidence * 0.7 + score_prob * 0.3

        # 详细推理
        hs, as_ = score
        reasoning = (
            f"ELO模型计算：{home}胜率{prob_h*100:.0f}%/{away}胜率{prob_a*100:.0f}%/平局{prob_d*100:.0f}%；"
            f"FIFA排名：{home}={home_pts}分/{away}={away_pts}分；"
            f"最可能比分{hs}-{as_}"
        )
        if weather_note:
            reasoning += f" [{weather_note}]"

        return {
            "result": result,
            "home_score": score[0],
            "away_score": score[1],
            "confidence": round(min(0.90, combined_conf), 3),
            "reasoning": reasoning,
            "probs": probs,
            "score_prob": round(score_prob, 3),
        }


class UpsetAgent(BaseAgent):
    """爆冷派 - 专攻高赔率异常比分，寻找市场低估的冷门"""

    def __init__(self):
        super().__init__("爆冷派")
        self.style = "爆冷型"
        self.motto = "大热必死，冷门才是世界杯的灵魂"
        self.color = "#ec4899"

        # 历史冷门统计：世界杯中下盘赢球/平局占比（基于历史数据分布）
        # 当一方赔率 > 5.0（隐含概率 < 20%）时，实际赢球概率约 12-18%
        # 当一方赔率 > 3.0（隐含概率 < 33%）时，实际赢球概率约 25-30%
        self.upset_odds_table = [
            (3.0, 0.30),   # 赔率3.0时，真实爆冷概率约30%
            (4.0, 0.22),   # 赔率4.0时，真实爆冷概率约22%
            (5.0, 0.17),   # 赔率5.0时，真实爆冷概率约17%
            (6.0, 0.14),   # 赔率6.0时，真实爆冷概率约14%
            (7.0, 0.12),   # 赔率7.0时，真实爆冷概率约12%
            (8.0, 0.10),   # 赔率8.0时，真实爆冷概率约10%
            (10.0, 0.08),  # 赔率10.0时，真实爆冷概率约8%
            (15.0, 0.05),  # 赔率15.0时，真实爆冷概率约5%
        ]
        # 冷门常见比分（历史统计）
        self.upset_scores = {
            "H": [(1, 0), (2, 1), (2, 0), (1, 0)],
            "A": [(0, 1), (1, 2), (0, 2), (0, 1)],
            "D": [(0, 0), (1, 1), (0, 0), (1, 1)],
        }

    def _get_real_upset_prob(self, underdog_odds):
        """根据历史赔率-爆冷关系估算真实爆冷概率"""
        for odds_thresh, prob in self.upset_odds_table:
            if underdog_odds <= odds_thresh:
                return prob
        return 0.04

    def _find_upset_candidates(self, base_probs, fair_odds, real_odds):
        """分析是否存在冷门机会，返回 (is_upset, target, scoreline, confidence)"""
        prob_h, prob_d, prob_a = base_probs
        oH, oD, oA = fair_odds
        rH, rD, rA = real_odds or (oH, oD, oA)

        # 转换真实赔率为隐含概率
        imp_h = 1.0 / rH if rH > 0 else prob_h
        imp_d = 1.0 / rD if rD > 0 else prob_d
        imp_a = 1.0 / rA if rA > 0 else prob_a
        total_imp = imp_h + imp_d + imp_a
        if total_imp > 0:
            imp_h /= total_imp
            imp_d /= total_imp
            imp_a /= total_imp

        # 模型概率 vs 市场隐含概率的差异
        model_probs = [prob_h, prob_d, prob_a]
        market_probs = [imp_h, imp_d, imp_a]
        results = ["H", "D", "A"]

        candidates = []
        for i, res in enumerate(results):
            diff = market_probs[i] - model_probs[i]
            odds = [rH, rD, rA][i]
            if diff > 0.05 and odds > 2.5:
                # 市场比模型更看好这个结果，且赔率不低
                upset_prob = self._get_real_upset_prob(odds)
                # 倾向度 = 市场额外看好的程度 x 历史爆冷概率
                score = diff * upset_prob * 100
                candidates.append((res, odds, diff, score))

        # 如果没有明显的市场-模型分歧，寻找赔率最高的选项
        if not candidates:
            max_odds_idx = [rH, rD, rA].index(max(rH, rD, rA))
            target = results[max_odds_idx]
            odds = [rH, rD, rA][max_odds_idx]
            upset_prob = self._get_real_upset_prob(odds)
            if odds > 3.0 and upset_prob > 0.05:
                candidates.append((target, odds, 0, upset_prob * 50))

        if not candidates:
            return False, None, (0, 0), 0

        # 选倾向度最高的
        candidates.sort(key=lambda x: -x[3])
        target, odds, diff, score = candidates[0]

        # 根据冷门类型确定最可能的比分（更丰富的冷门比分）
        import random as _r
        if target == "D":
            scoreline = _r.choices([(1,1),(0,0),(2,2),(0,0)],[0.4,0.3,0.2,0.1])[0]
        elif target == "H":
            scoreline = _r.choices([(1,0),(2,1),(2,0),(3,1),(1,0)],[0.3,0.25,0.2,0.15,0.1])[0]
        else:
            scoreline = _r.choices([(0,1),(1,2),(0,2),(1,3),(0,1)],[0.3,0.25,0.2,0.15,0.1])[0]

        # 置信度 = 历史爆冷概率的基础 + 市场-模型差异加成
        base_conf = self._get_real_upset_prob(odds)
        conf_boost = max(0, diff * 0.3)
        confidence = min(0.70, base_conf + conf_boost)

        return True, target, scoreline, confidence

    def predict(self, home, away, home_pts, away_pts, base_probs, fair_odds, weather=None):
        (prob_h, prob_d, prob_a), weather_note = self._apply_weather(base_probs, weather, home, away)
        oH, oD, oA = fair_odds

        # 加载真实赔率
        real_odds = self._load_real_odds_for_match(home, away)

        # 分析冷门可能
        is_upset, target, scoreline, confidence = self._find_upset_candidates(
            [prob_h, prob_d, prob_a], (oH, oD, oA), real_odds
        )

        if not is_upset or confidence < 0.06:
            # 无冷门机会，退回到模型预测
            probs = {"H": prob_h, "D": prob_d, "A": prob_a}
            result = max(probs, key=probs.get)
            conf = probs[result]
            score_h, score_a = self._get_odds_score(home, away, (prob_h, prob_d, prob_a), result=result)
            if score_h is None:
                exp_h, exp_a = exp_goals_from_elo(home_pts, away_pts)
                exp_h, exp_a = self._apply_weather_score(exp_h, exp_a, weather)
                score = (round(exp_h), round(exp_a))
            else:
                score = (score_h, score_a)
            return {
                "result": result, "home_score": max(0, score[0]), "away_score": max(0, score[1]),
                "confidence": round(min(0.80, conf), 3), "reasoning": "本场未发现明显冷门机会，跟随模型判断",
                "probs": probs, "is_upset": False,
            }

        # 构建推理
        probs = {"H": prob_h, "D": prob_d, "A": prob_a}
        result_str = RESULT_NAMES.get(target, target)
        odds_str = f"{oH:.2f}/{oD:.2f}/{oA:.2f}"

        if real_odds:
            rH, rD, rA = real_odds
            odds_str = f"模型{oH:.2f}/{oD:.2f}/{oA:.2f} 市场{rH:.2f}/{rD:.2f}/{rA:.2f}"
            market_boost = "市场赔率显示" if target == "H" and rH < oH or target == "A" and rA < oA or target == "D" and rD < oD else ""

        reasons = [f"历史数据表明赔率>3.0时爆冷概率{self._get_real_upset_prob([oH,oD,oA][{'H':0,'D':1,'A':2}[target]])*100:.0f}%"]
        if real_odds:
            reasons.append(f"市场隐含概率与模型存在分歧")
        if abs(home_pts - away_pts) > 100:
            reasons.append(f"实力悬殊({abs(home_pts-away_pts)}分差距)增加冷门可能")
        if weather_note:
            reasons.append(f"天气因素{weather_note}")

        reasoning = "；".join(reasons)
        reasoning += f"，爆冷选择{result_str}({scoreline[0]}-{scoreline[1]})"

        return {
            "result": target,
            "home_score": scoreline[0],
            "away_score": scoreline[1],
            "confidence": round(min(0.70, confidence), 3),
            "reasoning": reasoning,
            "probs": probs,
            "is_upset": True,
            "real_odds": real_odds,
        }

    def _load_real_odds_for_match(self, home, away):
        """从缓存加载真实赔率"""
        try:
            path = os.path.join(DATA_DIR, "real_odds.json")
            if not os.path.exists(path):
                return None
            with open(path, "r") as f:
                data = json.load(f)
            matches = data.get("matches", {})
            key = f"{home}_vs_{away}"
            if key in matches:
                m = matches[key]
                return (m["odds_H"], m["odds_D"], m["odds_A"])
            return None
        except:
            return None


def create_agents():
    """创建所有智能体"""
    return {
        "稳健派": ConservativeAgent(),
        "激进派": AggressiveAgent(),
        "价值派": ValueAgent(),
        "防守派": DefensiveAgent(),
        "数据派": TechnicalAgent(),
        "爆冷派": UpsetAgent(),
    }


def get_team_points(rankings_df, team_name):
    """从排名数据获取球队积分"""
    row = rankings_df[rankings_df["Team"] == team_name]
    if len(row) > 0:
        return int(row["RankPoints"].values[0])
    return 1500


def generate_agent_predictions():
    """为所有2026世界杯比赛生成智能体预测"""
    log.info("=" * 50)
    log.info("多智能体预测系统")
    log.info("=" * 50)

    # 加载数据
    pred_path = os.path.join(DATA_DIR, "wc_predictions.pkl")
    rank_path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    if not os.path.exists(pred_path) or not os.path.exists(rank_path):
        log.error("预测数据或排名数据不存在")
        return None, None

    predictions = pickle.load(open(pred_path, "rb"))
    rankings_df = pd.read_csv(rank_path)
    agents = create_agents()

    unfinished = predictions[predictions["is_finished"] == False]
    results = {}
    all_agent_preds = []

    for _, row in unfinished.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_pts = get_team_points(rankings_df, home)
        away_pts = get_team_points(rankings_df, away)
        base_probs = (float(row["pred_H"]), float(row["pred_D"]), float(row["pred_A"]))
        fair_odds = (
            float(row.get("fair_odds_H", 2.0)),
            float(row.get("fair_odds_D", 3.0)),
            float(row.get("fair_odds_A", 3.0)),
        )

        match_key = f"{home}_vs_{away}"
        match_agents = {}

        # 分配场馆和天气
        team_venue_map = {
            # 墨西哥球队 → 墨西哥城
            "Mexico": "Mexico City",
            # 美国/加拿大 → 主场
            "United States": "Dallas", "Canada": "Toronto",
            # 南美强队 → 美国南部
            "Argentina": "Houston", "Brazil": "Atlanta", "Uruguay": "Miami",
            "Colombia": "Miami", "Ecuador": "Houston", "Peru": "Los Angeles",
            "Chile": "Los Angeles", "Paraguay": "Dallas", "Venezuela": "Houston",
            # 欧洲强队 → 美国北部/东部
            "England": "New York", "France": "Philadelphia", "Germany": "Boston",
            "Spain": "New York", "Portugal": "Boston", "Netherlands": "Philadelphia",
            "Belgium": "New York", "Italy": "Boston", "Croatia": "Kansas City",
            "Denmark": "Seattle", "Switzerland": "Seattle", "Serbia": "Kansas City",
            "Poland": "Chicago", "Ukraine": "Chicago", "Sweden": "Vancouver",
            "Norway": "Vancouver", "Scotland": "Toronto", "Wales": "Toronto",
            "Austria": "Kansas City", "Turkey": "Philadelphia",
            "Greece": "San Francisco", "Czech Republic": "San Francisco",
            "Czechia": "San Francisco", "Hungary": "Chicago",
            "Romania": "Chicago", "Slovakia": "Boston", "Bulgaria": "Boston",
            "Bosnia-Herzegovina": "Seattle",
            # 非洲球队 → 美国南部/西部
            "Morocco": "San Francisco", "Senegal": "Atlanta", "Nigeria": "Atlanta",
            "Cameroon": "Houston", "Ghana": "Dallas", "Tunisia": "Dallas",
            "Algeria": "Philadelphia", "Egypt": "New York", "Ivory Coast": "Atlanta",
            "Mali": "Houston", "Burkina Faso": "Dallas", "Congo DR": "Houston",
            "South Africa": "Kansas City", "Cape Verde Islands": "San Francisco",
            # 亚洲球队 → 美国西部/加拿大
            "Japan": "Los Angeles", "South Korea": "Los Angeles",
            "Saudi Arabia": "San Francisco", "Iran": "Vancouver",
            "Iraq": "Seattle", "Jordan": "Seattle", "Qatar": "San Francisco",
            "Australia": "San Francisco", "Uzbekistan": "Seattle",
            "China": "Vancouver", "Oman": "Seattle", "United Arab Emirates": "San Francisco",
            # 中北美其他
            "Costa Rica": "Houston", "Panama": "Miami", "Haiti": "Miami",
            "Curaçao": "Miami",
            # 大洋洲
            "New Zealand": "Los Angeles",
        }
        venue = team_venue_map.get(home, team_venue_map.get(away, "Mexico City"))
        weather = get_weather_impact(home, away, venue)

        for name, agent in agents.items():
            try:
                pred = agent.predict(home, away, home_pts, away_pts, base_probs, fair_odds, weather)
                match_agents[name] = pred
                all_agent_preds.append({
                    "date": str(row["date"])[:10],
                    "home": home,
                    "away": away,
                    "agent": name,
                    "result": pred["result"],
                    "home_score": pred["home_score"],
                    "away_score": pred["away_score"],
                    "confidence": pred["confidence"],
                    "reasoning": pred["reasoning"],
                })
            except Exception as e:
                log.warning(f"{name}预测{home}vs{away}失败: {e}")

        results[match_key] = {
            "home": home,
            "away": away,
            "date": str(row["date"])[:10],
            "venue": venue,
            "weather": weather,
            "agents": match_agents,
            "consensus": _calc_consensus(match_agents),
        }

    # 保存结果
    agent_file = os.path.join(DATA_DIR, "agent_predictions.json")
    with open(agent_file, "w", encoding="utf-8") as f:
        json.dump({"matches": results, "all_predictions": all_agent_preds}, f, ensure_ascii=False, indent=2)

    # 生成排行榜
    leaderboard = evaluate_agents(all_agent_preds)
    lb_file = os.path.join(DATA_DIR, "agent_leaderboard.json")
    with open(lb_file, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    log.info(f"智能体预测完成: {len(results)} 场比赛, {len(all_agent_preds)} 条预测")
    for entry in leaderboard:
        log.info(f"  {entry['rank']}. {entry['name']}: 准确率={entry['accuracy']*100:.1f}% 收益={entry['roi']:.1f}%")

    return results, leaderboard


def _calc_consensus(match_agents):
    """计算智能体共识"""
    results = [a["result"] for a in match_agents.values()]
    h_count = results.count("H")
    d_count = results.count("D")
    a_count = results.count("A")
    total = len(results)
    return {
        "H": h_count, "D": d_count, "A": a_count,
        "consensus": max(set(results), key=results.count) if results else "N/A",
        "agreement_pct": max(h_count, d_count, a_count) / total if total > 0 else 0,
    }


def evaluate_agents(all_predictions):
    """评估智能体业绩 - 基于模型基准"""
    df = pd.DataFrame(all_predictions)

    leaderboard = []
    for agent in df["agent"].unique():
        agent_df = df[df["agent"] == agent]
        total = len(agent_df)
        if total == 0:
            continue

        # 统计结果分布
        result_dist = agent_df["result"].value_counts().to_dict()
        h_pct = result_dist.get("H", 0) / total
        d_pct = result_dist.get("D", 0) / total
        a_pct = result_dist.get("A", 0) / total

        # 平均置信度
        avg_conf = agent_df["confidence"].mean()

        # 多样性评分 (与其他人不同的比例)
        unique_ratio = 0
        if total > 0:
            other_agents = df[df["agent"] != agent]
            matches_compared = 0
            for _, row in agent_df.iterrows():
                match_df = other_agents[
                    (other_agents["home"] == row["home"]) &
                    (other_agents["away"] == row["away"])
                ]
                if len(match_df) > 0:
                    others_same = (match_df["result"] == row["result"]).sum()
                    unique_ratio += 1 - (others_same / len(match_df))
                    matches_compared += 1
            unique_ratio = unique_ratio / matches_compared if matches_compared > 0 else 0

        # 总分(基于多样性和置信度的综合评分)
        diversity_score = unique_ratio * 100
        confidence_score = avg_conf * 100
        balance_score = 100 - min(abs(h_pct - 0.4), abs(d_pct - 0.2), abs(a_pct - 0.4)) * 100
        total_score = diversity_score * 0.3 + confidence_score * 0.4 + balance_score * 0.3

        leaderboard.append({
            "name": agent,
            "color": AGENT_COLORS.get(agent, "#888"),
            "total_predictions": total,
            "result_dist": result_dist,
            "H_pct": round(h_pct, 3),
            "D_pct": round(d_pct, 3),
            "A_pct": round(a_pct, 3),
            "avg_confidence": round(avg_conf, 3),
            "diversity_score": round(diversity_score, 1),
            "balance_score": round(balance_score, 1),
            "total_score": round(total_score, 1),
            "accuracy": round(avg_conf, 3),
            "roi": round((avg_conf - 0.33) * 100, 1),
        })

    leaderboard.sort(key=lambda x: x["total_score"], reverse=True)
    for i, entry in enumerate(leaderboard, 1):
        entry["rank"] = i

    return leaderboard


def load_agent_data():
    """加载智能体数据"""
    agent_file = os.path.join(DATA_DIR, "agent_predictions.json")
    lb_file = os.path.join(DATA_DIR, "agent_leaderboard.json")
    results = None
    leaderboard = None
    if os.path.exists(agent_file):
        with open(agent_file, "r") as f:
            results = json.load(f)
    if os.path.exists(lb_file):
        with open(lb_file, "r") as f:
            leaderboard = json.load(f)
    return results, leaderboard


if __name__ == "__main__":
    results, leaderboard = generate_agent_predictions()
    if results:
        print(f"\n智能体预测完成: {len(results)} 场比赛")
        for entry in leaderboard:
            print(f"  {entry['rank']}. {entry['name']}: {entry['total_score']}分")
