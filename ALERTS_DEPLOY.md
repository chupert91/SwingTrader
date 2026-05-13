# Deploying alerts to Vercel

End-state: daily cron at 21:30 UTC (4:30 PM EST / 5:30 PM EDT, after market close) runs `/api/scan`, evaluates every enabled `AlertRule` against fresh yfinance data, sends Resend emails for new signals, and writes them to Vercel KV.

## One-time setup

### 1. Push the repo to GitHub
```powershell
git add .
git commit -m "Vercel alerts"
git push -u origin main
```

### 2. Create the Vercel project
1. https://vercel.com/new
2. Import the GitHub repo
3. Framework preset: **Other** (Vercel auto-detects Python from `requirements.txt` and `api/`)
4. Build command: leave blank
5. Output directory: leave default
6. Click **Deploy** — first deploy will set up the function infrastructure

### 3. Create the Vercel KV store
1. In the project dashboard → **Storage** tab → **Create Database** → **KV**
2. Name it `trisigma-kv` (or anything)
3. Click **Connect** → this auto-injects `KV_REST_API_URL`, `KV_REST_API_TOKEN`, `KV_URL`, `KV_REST_API_READ_ONLY_TOKEN` into the project's env vars
4. **Redeploy** the project so the functions pick up the new env vars

### 4. Set up Resend
1. https://resend.com → sign up (free: 3000 emails/month)
2. **Domains** → add a domain you own and follow DNS verification. For testing-only you can skip this and use `onboarding@resend.dev` as the from address.
3. **API Keys** → create a key (starts with `re_...`)
4. In Vercel project **Settings → Environment Variables**, add:
   - `RESEND_API_KEY` = `re_...`
   - `ALERT_FROM` = `alerts@yourdomain.com` (or `onboarding@resend.dev` for testing)

### 5. Secure the cron endpoint
1. Generate a random secret:
   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. In Vercel env vars: add `CRON_SECRET` = that value
3. Vercel Cron automatically sends `Authorization: Bearer <CRON_SECRET>` — `api/scan.py` checks this and 401s otherwise.

### 6. Verify the cron is scheduled
1. Push any commit so Vercel re-reads `vercel.json`
2. In project dashboard → **Crons** tab → you should see `/api/scan` scheduled `30 21 * * 1-5`

## Testing after deploy

### Manual scan
```powershell
curl -X GET "https://<your-app>.vercel.app/api/scan" `
  -H "Authorization: Bearer <CRON_SECRET>"
```
Expected response:
```json
{ "ok": true, "scanned_tickers": 4, "fired": [...], "rules_active": 1, ... }
```

### List rules
```powershell
curl "https://<your-app>.vercel.app/api/rules"
```

### Create a rule
```powershell
curl -X POST "https://<your-app>.vercel.app/api/rules" `
  -H "Content-Type: application/json" `
  -d '{"id":"","name":"Optimized","tickers":["TSLA","NVDA","PLTR","MP"],"side":"long","entry_sigma":-2.0,"require_trend":true,"min_trend_pct":30,"exit_target_pct":20,"exit_stop_pct":10,"leverage":5,"enabled":true,"notify_email":"you@gmail.com"}'
```

### View history
```powershell
curl "https://<your-app>.vercel.app/api/history?limit=20"
```

### Open the UI
Visit `https://<your-app>.vercel.app/` — sidebar shows the **Alert Rules** section. First load triggers `/api/rules` and shows the seeded default (created on the first scan).

## Local dev parity

The same code path works locally with `python run.py`:
- KV adapter falls back to `state/kv.json` (no Vercel KV needed)
- Resend adapter falls back to SMTP if `RESEND_API_KEY` is unset (uses Gmail vars from `.env`)
- Trigger a scan: `curl -X POST http://localhost:8000/api/scan`

## Notes & limits

- **Function size**: `pandas + numpy + yfinance` is ~150-200 MB on disk; safely under Vercel's 250 MB Hobby limit. If a future dep busts it, split `api/rules.py` and `api/history.py` into their own subdirectories with minimal `requirements.txt` files (they don't need pandas).
- **Cold start**: first cron run each day will be ~3-5s slower than warm calls. Doesn't matter for a daily cron.
- **Cron limits**: Hobby tier currently allows 2 crons at any cadence (was "daily only" before Oct 2024).
- **yfinance reliability**: `fetch_bars` includes a 60s in-memory cache but Vercel functions are stateless, so each cron invocation is a cold yfinance call. If yfinance becomes flaky, consider switching to Polygon or Alpaca (the adapter is `backend/data.py:fetch_bars`).
- **State backup**: Vercel KV doesn't auto-snapshot. The `swt:history` key is the only persistent record of fired alerts — call `/api/history` periodically and save the response if you need an audit trail.

## Rollback

The local FastAPI app is unchanged — `python run.py` works exactly as before whether or not Vercel is set up. To remove the Vercel deploy, delete the project from the Vercel dashboard; the repo continues to work locally.
