"""Data-analyst Telegram bot — TDS Project 1, Q5.

An LLM agent that answers data-analysis questions sent over Telegram.
Replies to EVERY message with exactly one JSON object:
    {"answer": <shaped as the question asks>, "log_url": "<public JSONL log>"}

Architecture:
  - FastAPI app serves /health and /run.jsonl (the public agent log, app-served).
  - A background thread long-polls Telegram getUpdates.
  - Each incoming message runs an agentic loop (OpenAI-compatible chat API,
    via aipipe, with run_python + wikipedia_search tools) until the model
    produces the final JSON answer.
  - A keep-warm thread pings our own public URL so a free host doesn't idle out.

Design note: no web_search (DuckDuckGo scraping) tool. That path was fragile
(bot detection) and the paid OpenRouter :online alternative needs a balance
we don't have. Instead the agent fetches data directly via run_python +
requests/pandas/bs4 (given a short list of known dataset root domains in the
system prompt), uses wikipedia_search for well-known facts/figures, and
falls back honestly to trained knowledge only after genuine fetch attempts
fail — never silently guessing a URL.
"""

import io
import json
import os
import re
import threading
import time
import traceback
import contextlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse

# ---------------------------------------------------------------- config
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")
MODEL = os.environ.get("MODEL", "gpt-4o")
MODEL_BASE_URL = os.environ.get("MODEL_BASE_URL", "https://aipipe.org/openai/v1")
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
LOG_PATH = os.environ.get("LOG_PATH", "/tmp/run.jsonl")
LOG_URL = f"{BASE_URL}/run.jsonl"
TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

MAX_AGENT_STEPS = 14
PY_TIMEOUT = 60          # seconds for one run_python call
ANSWER_BUDGET = 210      # wall-clock seconds before we force a final answer
                          # (grader timeout is ~300s per question — leave margin)

_log_lock = threading.Lock()
_histories: dict[int, list[dict]] = {}   # chat_id -> chat-completion messages
_hist_lock = threading.Lock()


# ---------------------------------------------------------------- logging
def log_event(**fields):
    fields["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(fields, ensure_ascii=False, default=str)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------------------------------------------------------------- tools
def run_python(code: str, env: dict | None = None) -> str:
    """Execute Python code, return captured stdout (or the error).

    `env` is the exec namespace. Pass the SAME dict back in on subsequent
    calls (e.g. one per question in solve()) so imports, variables, and
    functions defined in one run_python call persist into the next —
    otherwise every call starts from scratch and the model has to
    re-import/re-fetch everything each step, wasting steps and time.
    """
    out = io.StringIO()
    result: dict = {}
    if env is None:
        env = {"__name__": "__main__"}

    def target():
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(code, env)
            result["ok"] = True
        except Exception:
            result["ok"] = False
            out.write("\n" + traceback.format_exc(limit=4))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(PY_TIMEOUT)
    if t.is_alive():
        return "ERROR: code timed out after %ss" % PY_TIMEOUT
    text = out.getvalue()
    return text[-8000:] if text else "(no output — use print())"


_SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def wikipedia_search(query: str) -> str:
    """Look up a topic on Wikipedia via its official public API (no key,
    not scraping — a sanctioned endpoint, so it doesn't hit bot detection
    the way DuckDuckGo scraping does). Returns page titles + snippets, and
    the full extract of the top match."""
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 5,
            },
            headers=_SEARCH_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        hits = r.json().get("query", {}).get("search", [])
        if not hits:
            return "(no Wikipedia results found)"
        lines = []
        for h in hits:
            snippet = re.sub("<[^<]+?>", "", h.get("snippet", ""))
            lines.append(f"- {h['title']}: {snippet}")
        # also fetch a fuller extract of the top match
        top_title = hits[0]["title"]
        r2 = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "titles": top_title,
                "format": "json",
            },
            headers=_SEARCH_HEADERS,
            timeout=20,
        )
        r2.raise_for_status()
        pages = r2.json().get("query", {}).get("pages", {})
        extract = next(iter(pages.values()), {}).get("extract", "")
        if extract:
            lines.append(f"\nFull extract of '{top_title}':\n{extract[:3000]}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: wikipedia_search failed: {e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wikipedia_search",
            "description": (
                "Look up a topic on Wikipedia via its official API (reliable, "
                "no key, not scraping). Good for confirming well-known facts, "
                "statistics, rankings, and figures (e.g. state-wise records, "
                "demographic data, historical figures) when you don't have a "
                "direct dataset URL to fetch, or as a cross-check on a computed "
                "result."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Topic or question to look up"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Run Python code on the server and get its printed output. "
                "pandas, numpy, requests, bs4, openpyxl are installed and the "
                "network is available. This is your PRIMARY way to get data: "
                "construct a direct URL to a known dataset root (see system "
                "prompt for known domains: data.gov.in, MOSPI, PIB, RBI DBIE, "
                "NFHS) and fetch it with requests, or download/parse CSV/XLSX/"
                "HTML tables the question points at directly. Variables, "
                "imports, and functions PERSIST across calls within this "
                "conversation — you do not need to re-import or re-fetch "
                "something you already loaded in an earlier call. Always "
                "print() what you need to see — nothing else is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source to execute"}},
                "required": ["code"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are an expert data-analyst agent answering questions sent to a Telegram bot.

Rules:
1. Work out the answer to the user's LATEST message. Earlier messages in the chat are context for multi-turn tasks. Even if the latest message is only setup/context ("I will send data next."), you must still reply with a small JSON ack, e.g. {"answer": "ok", "log_url": "LOG_URL"} — never stay silent.
2. The message may embed data inline, or reference a public dataset (MOSPI, data.gov.in, PIB, RBI, NFHS, etc.). You do NOT have a general web-search tool. Instead:
   a. If the message gives you a URL, fetch it directly with run_python + requests.
   b. If it names a dataset/portal but no URL, try a direct fetch of that portal's known root first, e.g.:
      - data.gov.in — https://api.data.gov.in/resource/... or https://data.gov.in/catalogs
      - MOSPI — https://www.mospi.gov.in (publications, PLFS, NAS, NSS reports)
      - PIB — https://pib.gov.in (press releases with official figures)
      - RBI — https://dbie.rbi.org.in (Database on Indian Economy)
      - NFHS — https://main.mohfw.gov.in or rchiips.org/nfhs
      Inspect the raw response (print status code, headers, first N chars of text) before assuming its structure.
   c. Use wikipedia_search to confirm well-known published statistics, rankings, or figures when a direct fetch isn't feasible or as a cross-check.
   d. Never invent a specific numeric result from memory when you could compute it — always prefer fetching + computing over guessing.
3. If a fetch or parse fails (wrong URL, unexpected format, parser error, etc.), do NOT give up after one attempt. Try again: a different URL structure, a different known domain, or adjust your parsing approach. Make at least 2-3 real attempts with different URLs/approaches before considering the data unreachable.
4. Only if you have made genuine repeated attempts and still cannot fetch the data, fall back honestly to well-established general knowledge for well-known published statistics (e.g. widely reported MOSPI/SRS/NFHS figures). Even then you must still produce a real, specific value in the correct shape — never a prose apology, never an error message, never "unable to fetch" as the answer content. Give your best-informed concrete answer (e.g. an actual state name, an actual number) even under uncertainty, and never pretend a knowledge-based answer came from live retrieval.
5. The message usually spells out the exact JSON shape it wants, e.g. Reply with ONLY {"answer": {"state": "<state>"}, "log_url": "..."}. Match that shape EXACTLY in every case, including fallback/failure cases: same keys, same nesting, correct types (numbers as JSON numbers unless a string is explicitly requested), and round numbers exactly as instructed (if unspecified, give reasonable precision, e.g. 2 decimal places).
6. When ready, reply with ONLY that JSON object — no prose before or after, no markdown code fences, nothing else in the message. Use the literal placeholder string "LOG_URL" for the log_url value; the harness substitutes the real URL automatically.
7. If the message does not specify a shape at all, reply {"answer": <your concise answer>, "log_url": "LOG_URL"}.
8. Never add keys inside "answer" that were not asked for. Never wrap the answer in extra explanation text, and never put an error message or apology where a concrete answer value is expected.
9. You have a limited time budget. If you are told time is up, immediately output your best-guess final JSON in the exact requested shape — a late perfect answer scores zero, but a wrong-but-present, correctly-shaped answer beats a missing reply or a prose error message every time.
"""


# ---------------------------------------------------------------- llm
def chat_completion(messages, use_tools=True):
    body = {"model": MODEL, "messages": messages, "temperature": 0}
    if use_tools:
        body["tools"] = TOOLS
    r = requests.post(
        f"{MODEL_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {AIPIPE_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (data-analyst-bot)",
        },
        json=body,
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def extract_json(text: str):
    """Pull the first balanced JSON object out of model text."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def solve(chat_id: int, question: str) -> str:
    """Run the agent loop; return the final JSON reply text."""
    with _hist_lock:
        history = _histories.setdefault(chat_id, [])
        history.append({"role": "user", "content": question})
        del history[:-20]  # keep the last 20 turns
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(history)

    log_event(event="question", chat_id=chat_id, text=question)

    py_env: dict = {"__name__": "__main__"}  # persists across run_python calls for this question
    final_text = None
    deadline = time.time() + ANSWER_BUDGET
    for step in range(MAX_AGENT_STEPS):
        out_of_time = time.time() > deadline
        if out_of_time:
            messages.append(
                {"role": "user", "content": "Time is up. Reply NOW with only your best final JSON object."}
            )
        try:
            msg = chat_completion(messages, use_tools=not out_of_time)
        except Exception as e:
            log_event(event="llm_error", chat_id=chat_id, error=str(e))
            time.sleep(2)
            try:
                msg = chat_completion(messages, use_tools=True)
            except Exception as e2:
                log_event(event="llm_error_final", chat_id=chat_id, error=str(e2))
                break

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                if fn_name == "wikipedia_search":
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}
                    query = args.get("query", "")
                    log_event(event="tool_call", chat_id=chat_id, step=step, tool="wikipedia_search", query=query)
                    output = wikipedia_search(query)
                    log_event(event="tool_result", chat_id=chat_id, step=step, tool="wikipedia_search", output=output[:4000])
                elif fn_name == "run_python":
                    try:
                        code = json.loads(tc["function"]["arguments"]).get("code", "")
                    except json.JSONDecodeError:
                        code = tc["function"]["arguments"]
                    log_event(event="tool_call", chat_id=chat_id, step=step, tool="run_python", code=code[:4000])
                    output = run_python(code, env=py_env)
                    log_event(event="tool_result", chat_id=chat_id, step=step, tool="run_python", output=output[:4000])
                else:
                    output = f"ERROR: unknown tool '{fn_name}'"
                    log_event(event="tool_error", chat_id=chat_id, step=step, tool=fn_name, error=output)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": output})
            continue

        final_text = msg.get("content") or ""
        break

    obj = extract_json(final_text) if final_text else None
    if obj is None:
        obj = {"answer": (final_text or "unable to determine").strip()[:1000]}
    if "answer" not in obj:
        obj = {"answer": obj}
    obj["log_url"] = LOG_URL
    reply = json.dumps(obj, ensure_ascii=False)

    with _hist_lock:
        _histories.setdefault(chat_id, []).append({"role": "assistant", "content": reply})
    log_event(event="answer", chat_id=chat_id, reply=reply)
    return reply


# ---------------------------------------------------------------- telegram
def tg(method, **params):
    r = requests.post(f"{TG_API}/{method}", json=params, timeout=65)
    return r.json()


def handle_update(upd):
    msg = upd.get("message") or upd.get("edited_message")
    if not msg:
        return
    text = msg.get("text") or msg.get("caption") or ""
    chat_id = msg["chat"]["id"]
    if not text:
        return
    try:
        reply = solve(chat_id, text)
    except Exception:
        log_event(event="agent_crash", chat_id=chat_id, error=traceback.format_exc())
        reply = json.dumps({"answer": "internal error", "log_url": LOG_URL})
    tg("sendMessage", chat_id=chat_id, text=reply)


def poll_loop():
    log_event(event="startup", base_url=BASE_URL, model=MODEL)
    offset = 0
    pool = ThreadPoolExecutor(max_workers=6)
    while True:
        try:
            resp = requests.get(
                f"{TG_API}/getUpdates",
                params={"offset": offset, "timeout": 50},
                timeout=65,
            ).json()
            for upd in resp.get("result", []):
                offset = upd["update_id"] + 1
                pool.submit(handle_update, upd)
        except Exception as e:
            log_event(event="poll_error", error=str(e))
            time.sleep(5)


def keepwarm_loop():
    """Ping our own public URL so a free host never spins down."""
    while True:
        time.sleep(600)
        try:
            requests.get(f"{BASE_URL}/health", timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------- web app
app = FastAPI()


@app.on_event("startup")
def _start():
    if not BOT_TOKEN:
        log_event(event="config_error", error="BOT_TOKEN not set")
    if not AIPIPE_TOKEN:
        log_event(event="config_error", error="AIPIPE_TOKEN not set")
    if not os.path.exists(LOG_PATH):
        log_event(event="log_created")
    threading.Thread(target=poll_loop, daemon=True).start()
    threading.Thread(target=keepwarm_loop, daemon=True).start()


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"ok": True, "model": MODEL, "log_url": LOG_URL}


@app.get("/run.jsonl")
def run_log():
    if os.path.exists(LOG_PATH):
        return FileResponse(LOG_PATH, media_type="application/jsonl; charset=utf-8", filename="run.jsonl")
    return PlainTextResponse("", media_type="application/jsonl")


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"service": "data-analyst-telegram-bot", "log_url": LOG_URL}
