# AlphaRank backend refresh setup

This backend refresh pack updates `results.csv` automatically.

## Best architecture
Use a scheduled GitHub Actions workflow to run `update_results.py`, write a fresh `results.csv`, commit it back to your repo, and let Render auto-redeploy your app when the repo changes. GitHub Actions can run on a schedule using POSIX cron syntax, can also be run manually with `workflow_dispatch`, and the shortest supported interval is every 5 minutes. Render can automatically redeploy your linked service whenever you push changes to the connected branch. citeturn617997search8turn617997search1turn617997search3

## Files in this pack
- `update_results.py` — builds a fresh `results.csv`
- `tickers.csv` — the list of tickers to refresh
- `requirements-backend.txt` — Python packages for the refresh script
- `.github/workflows/refresh-results.yml` — scheduled workflow
- `README_backend_refresh.md` — this guide

## What to do
1. Put these files into the same GitHub repo as your hosted app.
2. Place:
   - `update_results.py` in the repo root
   - `tickers.csv` in the repo root
   - `requirements-backend.txt` in the repo root
   - `.github/workflows/refresh-results.yml` in `.github/workflows/`
3. Commit and push to your default branch.
4. In Render, make sure your web service is linked to that same branch and that auto-deploy is enabled. Render can redeploy automatically on each push to the linked branch. citeturn617997search1turn617997search3
5. In GitHub, go to **Actions** and manually run **Refresh AlphaRank Results** once to test.
6. After the workflow updates `results.csv` and pushes it, Render should redeploy and show the new data.

## Schedule
This workflow is set to run daily at `06:00` UTC:
- `0 6 * * *`

GitHub scheduled workflows run on the latest commit on the default branch, use POSIX cron syntax, and by default run in UTC. GitHub also now supports an optional timezone field if you want timezone-aware scheduling. citeturn617997search8turn617997search6

## How it works
- `tickers.csv` supplies the universe
- `update_results.py` downloads about 2 years of Yahoo Finance data per ticker
- it builds features
- it trains a lighter XGBoost classifier per ticker
- it writes a fresh `results.csv`
- the GitHub Action commits that file back to the repo
- Render sees the new commit and redeploys automatically. citeturn617997search1turn617997search3

## Notes
- This version is intentionally lighter than your earlier app pipeline:
  - 2 years of history
  - fewer features
  - fixed XGBoost parameters
  - no grid search
- That keeps refresh times and hosting load down.
- Your hosted app should read `results.csv` only; it should not retrain live.

## Optional next step
If you want, the next upgrade is to split your ticker universe into:
- `core_watchlist.csv`
- `ftse100.csv`
- `sp500.csv`

and generate multiple result files.
