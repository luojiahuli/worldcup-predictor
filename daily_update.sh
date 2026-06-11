#!/bin/bash
set -e
cd "$(dirname "$0")"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "===== 世界杯2026 每日更新 ====="

ODDS_API_KEY="${ODDS_API_KEY:-99459d111ea685ab723b0385811713be}"
CLOUDFLARE_API_TOKEN="${CLOUDFLARE_API_TOKEN:-cfut_P166Xqky47mpuPPVIHAwgXhnheQ1wqEygZdYaxgn27fa94bc}"

export ODDS_API_KEY
export CLOUDFLARE_API_TOKEN

# 1. 刷新实时赔率
log "获取最新赔率数据..."
python3 -c "from data_collector import fetch_real_odds; fetch_real_odds()" 2>&1 || log "赔率获取失败（可能是API配额不足）"

# 2. Run pipeline (data collection + training + prediction)

python3 main.py 2>&1 || log "main.py 失败"
python3 agents.py 2>&1 || log "agents.py 失败"

# 3. Generate dashboard
python3 -c "
from visualizer import load_all_data, generate_dashboard
data = load_all_data()
if data:
    generate_dashboard(data)
    print('仪表盘生成成功')
else:
    print('加载数据失败')
" 2>&1

# 4. Deploy to Cloudflare
mkdir -p /tmp/wc-deploy
cp output/worldcup_dashboard_latest.html /tmp/wc-deploy/index.html 2>/dev/null || log "无最新仪表盘文件"
if [ -n "$CLOUDFLARE_API_TOKEN" ]; then
  npx wrangler pages deploy /tmp/wc-deploy --project-name worldcup-2026 --branch main 2>&1
else
  log "CLOUDFLARE_API_TOKEN 未设置，跳过部署"
fi

# 5. Commit and push to GitHub
git add -A 2>/dev/null
git commit -m "daily update $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || log "无变更需要提交"
git push origin main 2>&1 || log "Git push 失败"

log "===== 每日更新完成: $(date '+%Y-%m-%d %H:%M') ====="
