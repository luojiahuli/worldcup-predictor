#!/bin/bash
set -e
cd "$(dirname "$0")"

# 1. Run pipeline (data collection + training + prediction)
python3 main.py 2>&1 || echo "main.py failed (may need API keys)"

# 2. Run multi-agent predictions
python3 agents.py 2>&1

# 3. Generate dashboard
python3 -c "from visualizer import load_all_data, generate_dashboard; data = load_all_data(); generate_dashboard(data)" 2>&1

# 4. Deploy to Cloudflare
cp output/worldcup_dashboard_latest.html /tmp/wc-deploy/index.html
if [ -n "$CLOUDFLARE_API_TOKEN" ]; then
  npx wrangler pages deploy /tmp/wc-deploy --project-name worldcup-2026 --branch main 2>&1
else
  echo "CLOUDFLARE_API_TOKEN not set, skipping deploy"
fi

# 5. Commit and push to GitHub
git add -A 2>/dev/null
git commit -m "daily update $(date '+%Y-%m-%d %H:%M')" 2>/dev/null || echo "No changes to commit"
git push origin main 2>&1 || echo "Git push failed"

echo "Daily update complete: $(date '+%Y-%m-%d %H:%M')"
