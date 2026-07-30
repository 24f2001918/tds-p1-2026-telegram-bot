# Data-Analyst Telegram Bot

An LLM agent that answers data-analysis questions sent to it on Telegram.
Built for IIT Madras Tools in Data Science, Project 1, Q5.

## What it does

Message the bot a data-analysis question (inline data, or a pointer to a public
dataset such as MOSPI / data.gov.in). The agent works out the answer — fetching
data and running pandas/numpy code in a sandboxed `run_python` tool when needed —
and replies with exactly one JSON object:

```json
{"answer": {"state": "Assam"}, "log_url": "https://<host>/run.jsonl"}
```

- `answer` is shaped exactly as the question asks (keys, nesting, types).
- `log_url` is a public, wget-able JSONL log of every agent step (questions,
  tool calls, tool outputs, final answers) — one JSON object per line, served
  directly by the app at `/run.jsonl`.

Multi-turn conversations are supported: per-chat history is kept, the agent
always replies to every message received (even setup-only ones), and answers
the latest message in context.

## Architecture

Everything lives in `bot.py`:

- FastAPI app serving `/health` (keep-alive) and `/run.jsonl` (the public log)
- A background thread long-polling the Telegram Bot API (`getUpdates`) — no
  webhook/HTTPS cert needed
- An agentic loop over an OpenAI-compatible chat API (via aipipe) with a
  `run_python` tool (pandas, numpy, requests, BeautifulSoup, openpyxl
  available; network on) — capped at 10 steps and a ~210s wall-clock budget
  per question, leaving margin under the grader's ~300s timeout
- A keep-warm thread that self-pings `/health` every 10 min so a free host
  doesn't idle out

## Local setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real values, then export them, e.g.:
export BOT_TOKEN=...          # from @BotFather
export AIPIPE_TOKEN=...       # aipipe.org token
export MODEL=gpt-4o
export BASE_URL=http://localhost:8000
uvicorn bot:app --host 0.0.0.0 --port 8000
```

Then message your bot on Telegram and confirm you get back one clean JSON
object, and `wget http://localhost:8000/run.jsonl` shows the run log.

## Deploy on Render

1. Push this repo to GitHub (public).
2. Render dashboard → New → Web Service → connect this repo.
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
   - Or just use the included `render.yaml` (New → Blueprint).
3. Set environment variables in the Render dashboard (never commit these):
   - `BOT_TOKEN` — from @BotFather
   - `AIPIPE_TOKEN` — your aipipe token
   - `MODEL` — `gpt-4o`
   - `MODEL_BASE_URL` — `https://aipipe.org/openai/v1`
   - `BASE_URL` — `https://<your-service>.onrender.com` (set this **after**
     Render assigns the service URL, then trigger a manual deploy — changing
     env vars alone does not restart the service)
4. Verify:
   ```bash
   curl https://<your-service>.onrender.com/health
   wget https://<your-service>.onrender.com/run.jsonl
   ```
5. Message the bot from a real Telegram account and confirm a clean JSON reply.

## Notes on grading requirements

- The bot replies to **every** message in a conversation, not just the last
  one — multi-turn grading waits for a reply after each message.
- Replies are **exactly one JSON object**, no prose, no markdown fences.
- `log_url` always points at the live `/run.jsonl` route, which is public and
  `wget`-able.
- Model is pinned to `gpt-4o` (not a mini model) — smaller models were found
  to get real published statistics wrong.
