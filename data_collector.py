"""
data_collector.py - 世界杯2026数据采集器
从 football-data.org API 获取世界杯赛程和历史数据
"""
import pandas as pd
import numpy as np
import os, pickle, time, logging, requests

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

API_TOKEN = "fc2bdc810c27460ca24fd3dcbfcbbb81"
API_HEADERS = {"X-Auth-Token": API_TOKEN}
API_BASE = "https://api.football-data.org/v4"

# 已知的世界杯正赛参赛队（2026年48队扩容）
KNOWN_TEAMS = [
    "Argentina", "France", "Brazil", "England", "Belgium", "Portugal",
    "Netherlands", "Spain", "Germany", "Italy", "Croatia", "Denmark",
    "Switzerland", "Uruguay", "Colombia", "Mexico", "USA", "Japan",
    "South Korea", "Australia", "Saudi Arabia", "Iran", "Senegal",
    "Morocco", "Nigeria", "Cameroon", "Ghana", "Tunisia", "Algeria",
    "Egypt", "Canada", "Costa Rica", "Panama", "Ecuador", "Peru",
    "Chile", "Paraguay", "Venezuela", "Qatar", "United Arab Emirates",
    "Iraq", "Oman", "Uzbekistan", "Jordan", "China", "Mali",
    "Burkina Faso", "Congo DR", "South Africa", "Ivory Coast",
    "Serbia", "Poland", "Ukraine", "Sweden", "Norway", "Scotland",
    "Wales", "Austria", "Hungary", "Greece", "Turkey", "Czech Republic",
    "Romania", "Russia", "Slovakia", "Slovenia", "Bulgaria",
    "New Zealand", "Fiji", "Tahiti", "Solomon Islands",
]

# 初始FIFA排名（基于2025-2026年实际情况）
FIFA_RANKINGS = {
    "Argentina": 1858, "France": 1845, "Brazil": 1834, "England": 1812,
    "Belgium": 1795, "Portugal": 1788, "Netherlands": 1765, "Spain": 1758,
    "Germany": 1745, "Italy": 1738, "Croatia": 1728, "USA": 1720,
    "United States": 1720, "Mexico": 1715, "Uruguay": 1712, "Colombia": 1708,
    "Denmark": 1705, "Switzerland": 1698, "Japan": 1695, "Morocco": 1692,
    "Senegal": 1688, "Iran": 1685, "South Korea": 1682, "Australia": 1678,
    "Saudi Arabia": 1675, "Nigeria": 1672, "Canada": 1668, "Ecuador": 1665,
    "Peru": 1662, "Chile": 1658, "Paraguay": 1655, "Costa Rica": 1652,
    "Panama": 1648, "Venezuela": 1645, "Cameroon": 1642, "Ghana": 1638,
    "Tunisia": 1635, "Algeria": 1632, "Egypt": 1628, "Ivory Coast": 1625,
    "Mali": 1622, "Burkina Faso": 1618, "Congo DR": 1615, "South Africa": 1612,
    "Serbia": 1640, "Poland": 1635, "Ukraine": 1628, "Sweden": 1625,
    "Norway": 1622, "Scotland": 1618, "Wales": 1615, "Austria": 1612,
    "Turkey": 1608, "Czech Republic": 1605, "Czechia": 1605, "Hungary": 1602,
    "New Zealand": 1580, "Qatar": 1575,
    # Teams in 2026 WC data
    "Bosnia-Herzegovina": 1598, "Cape Verde Islands": 1560, "Curaçao": 1500,
    "Haiti": 1490, "Iraq": 1550, "Jordan": 1540, "Uzbekistan": 1545,
}


def fetch_wc_matches(season=2026):
    """获取指定赛季的世界杯比赛"""
    url = f"{API_BASE}/competitions/WC/matches?season={season}"
    try:
        resp = requests.get(url, headers=API_HEADERS, timeout=15)
        if resp.status_code == 429:
            log.warning("限流，等待60秒...")
            time.sleep(60)
            resp = requests.get(url, headers=API_HEADERS, timeout=15)
        if resp.status_code != 200:
            log.warning(f"世界杯 {season}: HTTP {resp.status_code}")
            return []
        data = resp.json()
        matches = data.get("matches", [])
        log.info(f"世界杯 {season}: {len(matches)} 场比赛")
        result = []
        for m in matches:
            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")
            status = m.get("status", "")
            date_str = m.get("utcDate", "")
            matchday = m.get("matchday", 0)
            stage = m.get("stage", "")
            group = m.get("group", "")
            score = m.get("score", {}).get("fullTime", {})
            hs = score.get("home", -1)
            aws = score.get("away", -1)
            if status != "FINISHED":
                hs, aws = -1, -1
            # 获取赔率（如果存在）
            odds_data = {}
            odds = m.get("odds", {})
            if odds:
                odds_data = {
                    "odds_home": odds.get("homeWin", None),
                    "odds_draw": odds.get("draw", None),
                    "odds_away": odds.get("awayWin", None),
                }
            row = {
                "Date": date_str,
                "Home": home,
                "Away": away,
                "HomeGoals": hs,
                "AwayGoals": aws,
                "赛季": f"{season}",
                "比赛日": matchday,
                "stage": stage,
                "group": group,
                "status": status,
            }
            row.update(odds_data)
            result.append(row)
        time.sleep(0.6)
        return result
    except Exception as e:
        log.warning(f"世界杯 {season}: {e}")
        return []


def fetch_team_matches(team_name, limit=20):
    """获取特定球队近期比赛（用于构建训练数据）"""
    url = f"{API_BASE}/teams"
    try:
        resp = requests.get(url, headers=API_HEADERS, timeout=15, params={"name": team_name})
        if resp.status_code == 200:
            data = resp.json()
            teams = data.get("teams", [])
            if teams:
                team_id = teams[0].get("id")
                if team_id:
                    time.sleep(0.5)
                    m_url = f"{API_BASE}/teams/{team_id}/matches?limit={limit}"
                    m_resp = requests.get(m_url, headers=API_HEADERS, timeout=15)
                    if m_resp.status_code == 200:
                        m_data = m_resp.json()
                        matches = []
                        for m in m_data.get("matches", []):
                            home = m.get("homeTeam", {}).get("name", "")
                            away = m.get("awayTeam", {}).get("name", "")
                            comp = m.get("competition", {}).get("name", "")
                            status = m.get("status", "")
                            date_str = m.get("utcDate", "")
                            score = m.get("score", {}).get("fullTime", {})
                            hs = score.get("home", -1)
                            aws = score.get("away", -1)
                            if status != "FINISHED":
                                hs, aws = -1, -1
                            matches.append({
                                "Date": date_str,
                                "Home": home,
                                "Away": away,
                                "HomeGoals": hs,
                                "AwayGoals": aws,
                                "competition": comp,
                                "status": status,
                            })
                        return matches
    except Exception as e:
        log.warning(f"球队 {team_name}: {e}")
    return []


def fetch_all_world_cups():
    """采集所有世界杯历史数据 + 2026赛程"""
    log.info("=" * 50)
    log.info("采集世界杯数据...")
    log.info("=" * 50)

    all_matches = []
    for season in [2014, 2018, 2022, 2026]:
        matches = fetch_wc_matches(season)
        if matches:
            df = pd.DataFrame(matches)
            df = df.drop_duplicates(subset=["Date", "Home", "Away"])
            finished = df[df["HomeGoals"] >= 0]
            log.info(f"  => {len(df)} 场 (已完赛{len(finished)})")
            path = os.path.join(DATA_DIR, f"wc_{season}.pkl")
            with open(path, "wb") as f:
                pickle.dump(df, f)
            all_matches.append(df)
        time.sleep(1)

    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        log.info(f"\n总计: {len(combined)} 场比赛")
        log.info(f"  已完赛: {len(combined[combined['HomeGoals']>=0])}")
        log.info(f"  待赛: {len(combined[combined['HomeGoals']<0])}")

        path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
        with open(path, "wb") as f:
            pickle.dump(combined, f)
        return combined
    return None


def fetch_fifa_rankings():
    """尝试从football-data.org获取排名，否则使用预置数据"""
    try:
        url = f"{API_BASE}/teams"
        resp = requests.get(url, headers=API_HEADERS, timeout=15, params={"limit": 100})
        if resp.status_code == 200:
            log.info("获取排名数据成功")
    except:
        pass
    # 使用预置排名
    rankings_df = pd.DataFrame([
        {"Team": team, "Rank": rank, "RankPoints": pts}
        for rank, (team, pts) in enumerate(sorted(FIFA_RANKINGS.items(), key=lambda x: -x[1]), 1)
    ])
    path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    rankings_df.to_csv(path, index=False)
    log.info(f"FIFA排名已保存: {len(rankings_df)} 支球队")
    return rankings_df


def collect_all():
    """采集所有数据"""
    log.info("=" * 50)
    log.info("世界杯2026预测 - 数据采集")
    log.info("=" * 50)

    # 1. 世界杯比赛数据
    wc_data = fetch_all_world_cups()

    # 2. FIFA排名
    rankings = fetch_fifa_rankings()

    # 3. 从实际数据中提取参赛队
    actual_teams = set()
    if wc_data is not None:
        valid = wc_data[wc_data["Home"].notna() & wc_data["Away"].notna()]
        for _, row in valid.iterrows():
            if str(row.get("Home", "")).strip():
                actual_teams.add(str(row["Home"]).strip())
            if str(row.get("Away", "")).strip():
                actual_teams.add(str(row["Away"]).strip())

    teams_list = sorted(actual_teams) if actual_teams else KNOWN_TEAMS
    teams_df = pd.DataFrame({"Team": teams_list})
    teams_path = os.path.join(DATA_DIR, "worldcup_teams.csv")
    teams_df.to_csv(teams_path, index=False)
    log.info(f"参赛球队: {len(teams_df)} 支 (来自API数据)")

    log.info(f"\n✅ 数据采集完成!")
    return wc_data, rankings


def load_data():
    """加载缓存数据"""
    path = os.path.join(DATA_DIR, "all_wc_matches.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_rankings():
    """加载排名数据"""
    path = os.path.join(DATA_DIR, "fifa_rankings.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def get_wc_2026_matches():
    """获取2026世界杯比赛"""
    path = os.path.join(DATA_DIR, "wc_2026.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            df = pickle.load(f)
            return df[df["status"] != "FINISHED"] if "status" in df.columns else df
    return None


if __name__ == "__main__":
    collect_all()
