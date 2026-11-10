import os
import sys
import subprocess
import time
import json
import re
import uuid
import sqlite3
import asyncio
import urllib.request
import threading
from threading import Thread

print("=== Starting DeepSeek Server with Server-Side Chat & State ===", flush=True)

# 1. Install dependencies
print("Installing FastAPI, Uvicorn, transformers, bitsandbytes...", flush=True)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "fastapi", "uvicorn", "transformers", "accelerate", "bitsandbytes"
])

# 2. Download Cloudflare Tunnel binary
CLOUDFLARED_BIN = "/tmp/cloudflared"
if not os.path.exists(CLOUDFLARED_BIN):
    print("Downloading Cloudflare tunnel binary...", flush=True)
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        CLOUDFLARED_BIN
    )
    os.chmod(CLOUDFLARED_BIN, 0o755)

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

class EventStreamingResponse(StreamingResponse):
    def __init__(self, content, status_code=200, headers=None, media_type="text/event-stream", background=None):
        default_headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
        if headers:
            default_headers.update(headers)
        super().__init__(content, status_code=status_code, headers=default_headers, media_type=media_type, background=background)
        self.raw_headers = [
            (b"content-type", b"text/event-stream") if k.lower() == b"content-type" else (k, v)
            for k, v in self.raw_headers
        ]
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer,
    BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
)

MODEL_ID = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"
print(f"Loading {MODEL_ID} in 4-bit NF4...", flush=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()
print("Model loaded successfully onto GPU!", flush=True)

STOP_TOKENS = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "<s>", "</s>"]

# 3. SQLite Database for Server-Side Persistence
DB_PATH = "/tmp/deepseek_chats.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute('''CREATE TABLE IF NOT EXISTS chats (
        id TEXT PRIMARY KEY,
        title TEXT,
        created INTEGER
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        role TEXT,
        content TEXT,
        created INTEGER,
        FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
    )''')
    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# 4. In-Memory Tracking for Live Background Generations
# active_sessions = { chat_id: { "stop_event": Event, "current_text": "", "is_generating": bool, "subscribers": [...] } }
active_sessions = {}
sessions_lock = threading.Lock()

class StopOnEventCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        super().__init__()
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

def clean_thought_from_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    raw = raw_text
    for st in STOP_TOKENS:
        raw = raw.replace(st, "")

    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    if "<think>" in raw:
        raw = raw.replace("<think>", "").strip()

    lower = raw.strip().lower()
    monologue_keywords = [
        "thinking process:", "analyze the input:", "we need to",
        "the user is", "the user said", "the user wants",
        "the assistant should", "okay, let's"
    ]
    if any(lower.startswith(k) for k in monologue_keywords):
        parts = raw.split("\n\n")
        if len(parts) > 1:
            raw = "\n\n".join(parts[1:]).strip()
        else:
            raw = ""

    return raw.strip()

def background_generation_worker(chat_id: str, stop_event: threading.Event):
    """
    Runs completely on the server independent of client browser connections.
    Even if user switches chats, closes browser, or disconnects, this completes and saves.
    """
    print(f"[{chat_id}] Generation worker started on server...", flush=True)
    full_output = ""
    clean_final = ""
    buffered = ""
    started_streaming = False
    in_thinking = False

    try:
        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
        rows = c.fetchall()
        conn.close()

        raw_messages = [{"role": r["role"], "content": r["content"]} for r in rows]

        anti_thinking_note = " Answer directly, accurately and concisely in the user's language without any internal monologue or <think> tags."
        has_system = any(m["role"] == "system" for m in raw_messages)
        messages = []
        if not has_system:
            messages.append({
                "role": "system",
                "content": "You are a helpful AI assistant." + anti_thinking_note
            })

        for item in raw_messages:
            role = item["role"]
            content = item["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            for st in STOP_TOKENS:
                content = content.replace(st, "")
            if role == "system":
                content += anti_thinking_note
            if content:
                messages.append({"role": role, "content": content})

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except Exception:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        if prompt.endswith("<think>\n"):
            prompt = prompt[:-len("<think>\n")] + "<think>\n\n</think>\n\n"
        elif prompt.endswith("<think>"):
            prompt = prompt[:-len("<think>")] + "<think>\n\n</think>\n\n"

        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, timeout=120.0, skip_prompt=True, skip_special_tokens=False)

        eos_token_ids = [tokenizer.eos_token_id]
        for st in STOP_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(st)
            if isinstance(token_id, int) and token_id not in eos_token_ids:
                eos_token_ids.append(token_id)

        stopping_criteria = StoppingCriteriaList([StopOnEventCriteria(stop_event)])

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            use_cache=True,
        )

        gen_thread = Thread(target=model.generate, kwargs=generation_kwargs)
        gen_thread.start()

        for new_token in streamer:
            if stop_event.is_set():
                print(f"[{chat_id}] Stop requested. Terminating early...", flush=True)
                break

            if any(st in new_token for st in STOP_TOKENS):
                for st in STOP_TOKENS:
                    new_token = new_token.replace(st, "")
                if not started_streaming:
                    final_text = clean_thought_from_text(buffered + new_token)
                    if final_text:
                        full_output += final_text
                        broadcast_token(chat_id, final_text)
                        started_streaming = True
                elif new_token:
                    full_output += new_token
                    broadcast_token(chat_id, new_token)
                break

            if not started_streaming:
                buffered += new_token
                lower = buffered.strip().lower()

                if "<think>" in buffered:
                    in_thinking = True
                if in_thinking:
                    if "</think>" in buffered:
                        in_thinking = False
                        buffered = buffered.split("</think>")[-1].lstrip("\n")
                        if buffered:
                            full_output += buffered
                            broadcast_token(chat_id, buffered)
                            started_streaming = True
                            buffered = ""
                    continue

                monologues = ["thinking process:", "thinking process", "analyze the input:", "we need to", "okay, let"]
                is_prefix = any(k.startswith(lower) for k in monologues) if lower else False
                if is_prefix and len(lower) < 25:
                    continue

                if any(lower.startswith(k) for k in monologues):
                    if "\n\n" in buffered:
                        parts = buffered.split("\n\n", 1)
                        buffered = parts[1].lstrip("\n")
                        if buffered:
                            full_output += buffered
                            broadcast_token(chat_id, buffered)
                            started_streaming = True
                            buffered = ""
                    continue

                if buffered:
                    full_output += buffered
                    broadcast_token(chat_id, buffered)
                    started_streaming = True
                    buffered = ""
            else:
                for st in STOP_TOKENS:
                    new_token = new_token.replace(st, "")
                if new_token:
                    full_output += new_token
                    broadcast_token(chat_id, new_token)

    except Exception as e:
        print(f"[{chat_id}] Generation worker exception: {e}", flush=True)
    finally:
        if not started_streaming and buffered:
            clean_b = clean_thought_from_text(buffered)
            if clean_b:
                full_output += clean_b
                broadcast_token(chat_id, clean_b)

        # Save assistant message to SQLite
        clean_final = full_output.strip()
        if clean_final:
            try:
                conn = get_db()
                c = conn.cursor()
                c.execute("INSERT INTO messages (chat_id, role, content, created) VALUES (?, ?, ?, ?)",
                          (chat_id, "assistant", clean_final, int(time.time())))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"[{chat_id}] Error saving message to DB: {e}", flush=True)

        broadcast_done(chat_id)

        with sessions_lock:
            if chat_id in active_sessions:
                active_sessions[chat_id]["is_generating"] = False
                active_sessions[chat_id]["current_text"] = clean_final

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{chat_id}] Generation completed on server ({len(clean_final)} chars).", flush=True)

def broadcast_token(chat_id: str, token: str):
    with sessions_lock:
        session = active_sessions.get(chat_id)
        if not session:
            return
        session["current_text"] += token

def broadcast_done(chat_id: str):
    pass

# 5. FastAPI App with CORS
app = FastAPI(title="DeepSeek V4 Server-Side Managed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Server-Side Web UI API Endpoints ---

@app.get("/api/chats")
def get_all_chats():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, title, created FROM chats ORDER BY created DESC")
    rows = c.fetchall()
    conn.close()

    result = []
    with sessions_lock:
        for r in rows:
            cid = r["id"]
            is_gen = active_sessions.get(cid, {}).get("is_generating", False)
            result.append({
                "id": cid,
                "title": r["title"],
                "created": r["created"],
                "is_generating": is_gen
            })
    return result

@app.post("/api/chats")
def create_new_chat():
    new_id = "chat_" + uuid.uuid4().hex[:10]
    now = int(time.time())
    title = "گفتگوی جدید"
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO chats (id, title, created) VALUES (?, ?, ?)", (new_id, title, now))
    conn.commit()
    conn.close()
    return {"id": new_id, "title": title, "created": now, "messages": [], "is_generating": False}

@app.get("/api/chats/{chat_id}")
def get_chat_details(chat_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, title, created FROM chats WHERE id = ?", (chat_id,))
    chat_row = c.fetchone()
    if not chat_row:
        conn.close()
        return JSONResponse({"error": "Not found"}, status_code=404)

    c.execute("SELECT role, content, created FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    msg_rows = c.fetchall()
    conn.close()

    messages = [{"role": m["role"], "content": m["content"], "created": m["created"]} for m in msg_rows]

    with sessions_lock:
        session = active_sessions.get(chat_id, {})
        is_gen = session.get("is_generating", False)
        current_text = session.get("current_text", "")

    return {
        "id": chat_id,
        "title": chat_row["title"],
        "created": chat_row["created"],
        "messages": messages,
        "is_generating": is_gen,
        "current_text": current_text if is_gen else ""
    }

@app.delete("/api/chats/{chat_id}")
def delete_chat_by_id(chat_id: str):
    # Stop if currently generating
    with sessions_lock:
        if chat_id in active_sessions:
            active_sessions[chat_id]["stop_event"].set()
            del active_sessions[chat_id]

    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}

async def create_chat_event_generator(chat_id: str):
    # Initial keep-alive comment and Cloudflare buffer flush padding
    yield ": " + (" " * 4096) + "\n\n"
    yield ": ping\n\n"

    # Graceful wait up to 2.5s in case request arrives slightly before background worker registers session
    for _ in range(50):
        with sessions_lock:
            if chat_id in active_sessions:
                break
        await asyncio.sleep(0.05)

    last_len = 0
    ping_ticks = 0

    while True:
        with sessions_lock:
            session = active_sessions.get(chat_id)
            if not session:
                yield "data: [DONE]\n\n"
                break
            current_text = session.get("current_text", "")
            is_gen = session.get("is_generating", False)

        cur_len = len(current_text)
        if cur_len > last_len:
            new_slice = current_text[last_len:cur_len]
            last_len = cur_len
            payload = json.dumps({"token": new_slice, "full_text": current_text, "is_generating": is_gen}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

        if not is_gen:
            with sessions_lock:
                session = active_sessions.get(chat_id)
                if session:
                    final_text = session.get("current_text", "")
                    if len(final_text) > last_len:
                        payload = json.dumps({"token": final_text[last_len:], "full_text": final_text, "is_generating": False}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
            break

        ping_ticks += 1
        if ping_ticks >= 25:  # ~1 second keep-alive flush
            yield ": ping\n\n"
            ping_ticks = 0

        await asyncio.sleep(0.03)

@app.post("/api/chats/{chat_id}/send")
async def send_chat_message(chat_id: str, request: Request):
    body = await request.json()
    user_text = body.get("content", "").strip()
    stream_requested = body.get("stream", False)
    if not user_text:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    conn = get_db()
    c = conn.cursor()
    # Check if chat exists, if not create
    c.execute("SELECT title FROM chats WHERE id = ?", (chat_id,))
    chat_row = c.fetchone()
    now = int(time.time())
    if not chat_row:
        c.execute("INSERT INTO chats (id, title, created) VALUES (?, ?, ?)", (chat_id, "گفتگوی جدید", now))
        chat_title = "گفتگوی جدید"
    else:
        chat_title = chat_row["title"]

    # Insert user message
    c.execute("INSERT INTO messages (chat_id, role, content, created) VALUES (?, ?, ?, ?)",
              (chat_id, "user", user_text, now))

    # Auto-title on first user message
    if chat_title == "گفتگوی جدید":
        new_title = user_text[:28] + "..." if len(user_text) > 28 else user_text
        c.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, chat_id))

    conn.commit()
    conn.close()

    # Launch background worker on server
    stop_event = threading.Event()
    with sessions_lock:
        active_sessions[chat_id] = {
            "stop_event": stop_event,
            "current_text": "",
            "is_generating": True
        }

    thread = Thread(target=background_generation_worker, args=(chat_id, stop_event), daemon=True)
    thread.start()

    if stream_requested:
        return EventStreamingResponse(
            create_chat_event_generator(chat_id),
            headers={
                "Cache-Control": "no-cache, no-transform, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )

    return {"status": "started", "chat_id": chat_id}

@app.get("/api/chats/{chat_id}/stream")
@app.post("/api/chats/{chat_id}/stream")
async def stream_live_chat(chat_id: str, request: Request = None):
    return EventStreamingResponse(
        create_chat_event_generator(chat_id),
        headers={
            "Cache-Control": "no-cache, no-transform, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Type": "text/event-stream"
        }
    )

@app.post("/api/chats/{chat_id}/stop")
def stop_chat_generation(chat_id: str):
    with sessions_lock:
        session = active_sessions.get(chat_id)
        if session and session.get("is_generating", False):
            session["stop_event"].set()
            return {"status": "stopping"}
    return {"status": "not_generating"}

# --- OpenAI-Compatible Standard Endpoints (for 9router and other clients) ---

MODEL_DATA = [
    {
        "id": "deepseek-v4",
        "object": "model",
        "created": 1700000000,
        "owned_by": "deepseek",
        "permission": [],
        "root": "deepseek-v4",
        "parent": None
    },
    {
        "id": MODEL_ID,
        "object": "model",
        "created": 1700000000,
        "owned_by": "xtools",
        "permission": [],
        "root": MODEL_ID,
        "parent": None
    },
    {
        "id": "gpt-4o",
        "object": "model",
        "created": 1700000000,
        "owned_by": "openai",
        "permission": [],
        "root": "gpt-4o",
        "parent": None
    },
    {
        "id": "gpt-3.5-turbo",
        "object": "model",
        "created": 1700000000,
        "owned_by": "openai",
        "permission": [],
        "root": "gpt-3.5-turbo",
        "parent": None
    }
]

@app.get("/health")
@app.get("/v1/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "gpu": torch.cuda.is_available()}

@app.get("/v1")
@app.get("/v1/")
def v1_root():
    return {"status": "ok", "message": "DeepSeek V4 OpenAI-compatible API is running.", "models_url": "/v1/models"}

@app.get("/v1/models")
@app.get("/v1/models/")
@app.get("/models")
@app.get("/models/")
@app.get("/v1/v1/models")
@app.get("/v1/v1/models/")
@app.get("/api/v1/models")
@app.get("/api/v1/models/")
def list_models():
    return {"object": "list", "data": MODEL_DATA}

@app.post("/v1/chat/completions")
@app.post("/v1/chat/completions/")
@app.post("/chat/completions")
@app.post("/chat/completions/")
@app.post("/v1/v1/chat/completions")
@app.post("/v1/v1/chat/completions/")
@app.post("/api/v1/chat/completions")
@app.post("/api/v1/chat/completions/")
async def chat_completions(request: Request):
    body = await request.json()
    raw_messages = body.get("messages", [])
    stream = body.get("stream", False)
    req_model = body.get("model", "deepseek-v4")
    temperature = max(0.1, float(body.get("temperature", 0.7)))
    max_tokens = int(body.get("max_tokens", 2048))

    has_system = any(m.get("role") == "system" for m in raw_messages)
    anti_thinking_note = " Answer directly, accurately and concisely in the user's language without any internal monologue or <think> tags."

    messages = []
    if not has_system:
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant." + anti_thinking_note
        })

    for item in raw_messages:
        role = item.get("role", "user")
        content = str(item.get("content", ""))
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        for st in STOP_TOKENS:
            content = content.replace(st, "")
        if role == "system":
            content += anti_thinking_note
        if content:
            messages.append({"role": role, "content": content})

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    if prompt.endswith("<think>\n"):
        prompt = prompt[:-len("<think>\n")] + "<think>\n\n</think>\n\n"
    elif prompt.endswith("<think>"):
        prompt = prompt[:-len("<think>")] + "<think>\n\n</think>\n\n"

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, timeout=90.0, skip_prompt=True, skip_special_tokens=False)

    eos_token_ids = [tokenizer.eos_token_id]
    for st in STOP_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(st)
        if isinstance(token_id, int) and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        eos_token_id=eos_token_ids,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    if stream:
        def sse_generator():
            try:
                # Initial 4096-byte padding SSE comment to defeat reverse proxy / Cloudflare buffering
                yield f": {' ' * 4096}\n\n"
                first_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(first_chunk)}\n\n"

                buffered = ""
                started_streaming = False
                in_thinking = False

                for new_token in streamer:
                    if any(st in new_token for st in STOP_TOKENS):
                        for st in STOP_TOKENS:
                            new_token = new_token.replace(st, "")
                        if not started_streaming:
                            final_text = clean_thought_from_text(buffered + new_token)
                            if final_text:
                                chunk = {
                                    "id": chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_time,
                                    "model": req_model,
                                    "choices": [{"index": 0, "delta": {"content": final_text}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                started_streaming = True
                        elif new_token:
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": {"content": new_token}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        break

                    if not started_streaming:
                        buffered += new_token
                        lower = buffered.strip().lower()

                        if "<think>" in buffered:
                            in_thinking = True
                        if in_thinking:
                            if "</think>" in buffered:
                                in_thinking = False
                                buffered = buffered.split("</think>")[-1].lstrip("\n")
                                if buffered:
                                    chunk = {
                                        "id": chat_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": req_model,
                                        "choices": [{"index": 0, "delta": {"content": buffered}, "finish_reason": None}]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    started_streaming = True
                                    buffered = ""
                            continue

                        monologues = ["thinking process:", "thinking process", "analyze the input:", "we need to", "okay, let"]
                        is_prefix = any(k.startswith(lower) for k in monologues) if lower else False
                        if is_prefix and len(lower) < 25:
                            continue

                        if any(lower.startswith(k) for k in monologues):
                            if "\n\n" in buffered:
                                parts = buffered.split("\n\n", 1)
                                buffered = parts[1].lstrip("\n")
                                if buffered:
                                    chunk = {
                                        "id": chat_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": req_model,
                                        "choices": [{"index": 0, "delta": {"content": buffered}, "finish_reason": None}]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                                    started_streaming = True
                                    buffered = ""
                            continue

                        if buffered:
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": {"content": buffered}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            started_streaming = True
                            buffered = ""
                    else:
                        for st in STOP_TOKENS:
                            new_token = new_token.replace(st, "")
                        if new_token:
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": req_model,
                                "choices": [{"index": 0, "delta": {"content": new_token}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                final_chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": req_model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return EventStreamingResponse(
            sse_generator(),
            headers={
                "Cache-Control": "no-cache, no-transform, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Content-Type": "text/event-stream"
            }
        )

    else:
        full_output = ""
        for token in streamer:
            full_output += token
            if any(st in full_output for st in STOP_TOKENS):
                break

        clean_content = clean_thought_from_text(full_output)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created_time,
            "model": req_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": clean_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(inputs.input_ids[0]),
                "completion_tokens": len(full_output.split()),
                "total_tokens": len(inputs.input_ids[0]) + len(full_output.split())
            }
        }

# --- ChatGPT Web UI (Connected to Server State & Stop Button) ---

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek V4 Uncensored (Server-Side Persistence)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        body { background-color: #212121; color: #ececec; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .sidebar { background-color: #171717; }
        .user-msg { background-color: #2f2f2f; }
        .bot-msg { background-color: transparent; }
        
        /* ChatGPT-style Code Block */
        .code-block-container {
            border: 1px solid #383838;
            border-radius: 0.75rem;
            overflow: hidden;
            margin: 1rem 0;
            background-color: #181818;
            direction: ltr;
            text-align: left;
        }
        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.45rem 1rem;
            background-color: #262626;
            border-bottom: 1px solid #383838;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.75rem;
            color: #b3b3b3;
            user-select: none;
        }
        .copy-code-btn {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.75rem;
            padding: 0.2rem 0.55rem;
            border-radius: 0.375rem;
            transition: all 0.2s;
            background: transparent;
            border: 1px solid transparent;
            cursor: pointer;
            color: #d1d5db;
        }
        .copy-code-btn:hover {
            background-color: #383838;
            color: #ffffff;
            border-color: #4b5563;
        }
        .code-block-container pre {
            margin: 0 !important;
            padding: 1rem !important;
            overflow-x: auto;
            background-color: #181818 !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 0.85rem;
            line-height: 1.55;
        }
        .code-block-container code {
            background: transparent !important;
            padding: 0 !important;
            font-family: inherit;
        }

        /* Markdown Typography */
        .bot-markdown p { margin-bottom: 0.75rem; }
        .bot-markdown p:last-child { margin-bottom: 0; }
        .bot-markdown ul { list-style-type: disc; margin-right: 1.5rem; margin-bottom: 0.75rem; }
        .bot-markdown ol { list-style-type: decimal; margin-right: 1.5rem; margin-bottom: 0.75rem; }
        .bot-markdown li { margin-bottom: 0.25rem; }
        .bot-markdown strong { color: #ffffff; font-weight: 600; }
        .bot-markdown h1, .bot-markdown h2, .bot-markdown h3 { color: #f3f4f6; font-weight: 700; margin: 1rem 0 0.5rem; }
        .bot-markdown blockquote { border-right: 3px solid #10b981; padding-right: 0.75rem; color: #9ca3af; margin: 0.75rem 0; }
        .bot-markdown code:not(.hljs) {
            background-color: #2d2d2d;
            color: #fca5a5;
            padding: 0.15rem 0.4rem;
            border-radius: 0.375rem;
            font-family: monospace;
            font-size: 0.875em;
            direction: ltr;
            display: inline-block;
        }
        .bot-markdown table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.875rem; }
        .bot-markdown th, .bot-markdown td { border: 1px solid #383838; padding: 0.5rem 0.75rem; text-align: right; }
        .bot-markdown th { background-color: #282828; }

        .tps-badge { font-size: 0.75rem; color: #10b981; background: #064e3b; padding: 2px 8px; border-radius: 9999px; display: inline-block; margin-top: 6px; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-thumb { background: #424242; border-radius: 3px; }
        .chat-item:hover .delete-btn { opacity: 1; }
        .chat-active { background-color: #262626; border-color: #404040; }
    </style>
</head>
<body class="h-screen flex overflow-hidden">
    <!-- Sidebar -->
    <div class="sidebar w-64 md:w-72 flex flex-col h-full border-l border-[#2e2e2e]">
        <div class="p-3">
            <button onclick="createNewChat()" class="w-full flex items-center justify-between px-3.5 py-2.5 text-sm text-white font-medium rounded-xl border border-[#3e3e3e] hover:bg-[#262626] transition shadow-sm">
                <span class="flex items-center gap-2">
                    <span class="text-emerald-400 text-lg">➕</span>
                    <span>گفتگوی جدید</span>
                </span>
                <span class="text-xs bg-[#333] text-gray-400 px-1.5 py-0.5 rounded">Ctrl+K</span>
            </button>
        </div>

        <!-- Chat List from Server DB -->
        <div class="flex-1 overflow-y-auto px-2 space-y-1" id="chatList"></div>

        <div class="p-3 border-t border-[#2e2e2e] flex flex-col gap-1.5 text-xs text-gray-400 text-center">
            <div class="flex items-center justify-between px-1">
                <span class="text-emerald-400 font-medium">Server-Side Worker: فعال</span>
                <span class="text-gray-500">Dual T4 GPU</span>
            </div>
            <div class="text-[11px] text-gray-400">تمام پیام‌ها روی سرور پایدار ذخیره می‌شوند</div>
        </div>
    </div>

    <!-- Main Chat Area -->
    <div class="flex-1 flex flex-col h-full relative">
        <!-- Header -->
        <div class="h-14 border-b border-[#2e2e2e] flex items-center justify-between px-4 bg-[#212121]">
            <div class="flex items-center gap-2.5">
                <span class="w-2.5 h-2.5 bg-emerald-500 rounded-full animate-pulse"></span>
                <span class="font-semibold text-sm text-gray-200" id="currentChatHeaderTitle">گفتگو</span>
                <button onclick="copyBaseUrl()" class="text-xs bg-emerald-950/80 hover:bg-emerald-900 text-emerald-300 border border-emerald-700/50 px-2.5 py-1 rounded-full transition flex items-center gap-1.5">
                    <span>📋 کپی Base URL</span>
                </button>
            </div>
            <div class="flex items-center gap-3">
                <button onclick="deleteCurrentChat()" title="حذف این چت" class="text-xs text-gray-400 hover:text-red-400 transition flex items-center gap-1">
                    <span>🗑️ حذف چت</span>
                </button>
            </div>
        </div>

        <!-- Messages Container -->
        <div id="messagesContainer" class="flex-1 overflow-y-auto p-4 md:p-8 space-y-6 max-w-4xl w-full mx-auto"></div>

        <!-- Input Box -->
        <div class="p-4 border-t border-[#2e2e2e] bg-[#212121]">
            <div class="max-w-4xl mx-auto relative flex items-center">
                <textarea id="userInput" rows="1" placeholder="پیام خود را بنویسید..." 
                    class="w-full bg-[#2f2f2f] text-white rounded-2xl py-3.5 pr-4 pl-24 resize-none focus:outline-none focus:ring-1 focus:ring-gray-400 text-sm leading-relaxed"
                    onkeydown="handleKeyDown(event)" oninput="autoGrow(this)"></textarea>
                
                <div class="absolute left-2.5 flex items-center gap-1.5">
                    <!-- Stop Button (Visible only when generating) -->
                    <button id="stopBtn" onclick="stopCurrentGeneration()" style="display: none;"
                        class="px-3 py-1.5 rounded-xl bg-red-600 hover:bg-red-700 text-white text-xs font-semibold flex items-center gap-1.5 transition shadow">
                        <span class="w-2 h-2 bg-white rounded-sm"></span>
                        <span>توقف</span>
                    </button>

                    <!-- Send Button -->
                    <button id="sendBtn" onclick="sendMessage()" 
                        class="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center hover:bg-gray-200 transition disabled:opacity-40">
                        ⬆️
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        function escapeHtml(str) {
            return String(str == null ? "" : str)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#39;");
        }

        // Setup Marked with ChatGPT-style Code Block Renderer
        const customRenderer = new marked.Renderer();
        customRenderer.code = function(codeOrToken, language) {
            try {
                let codeStr = "";
                let lang = "";

                // Marked v12+ passes an object: { text, lang, escaped, raw, ... }
                // Older Marked passes: (codeString, langString)
                if (codeOrToken && typeof codeOrToken === 'object') {
                    if (typeof codeOrToken.text === 'string') {
                        codeStr = codeOrToken.text;
                    } else if (typeof codeOrToken.raw === 'string') {
                        codeStr = codeOrToken.raw;
                    } else if (codeOrToken.text != null) {
                        codeStr = String(codeOrToken.text);
                    } else if (codeOrToken.raw != null) {
                        codeStr = String(codeOrToken.raw);
                    } else {
                        codeStr = "";
                    }
                    lang = typeof codeOrToken.lang === 'string' ? codeOrToken.lang : (typeof language === 'string' ? language : "");
                } else if (typeof codeOrToken === 'string') {
                    codeStr = codeOrToken;
                    lang = typeof language === 'string' ? language : "";
                } else if (codeOrToken != null) {
                    codeStr = String(codeOrToken);
                    lang = typeof language === 'string' ? language : "";
                } else {
                    codeStr = "";
                    lang = typeof language === 'string' ? language : "";
                }

                lang = (lang || "").trim().toLowerCase();
                if (lang.includes(" ")) {
                    lang = lang.split(/\s+/)[0];
                }
                const displayLang = lang || "code";

                const validLang = (typeof hljs !== 'undefined' && typeof hljs.getLanguage === 'function' && lang && hljs.getLanguage(lang)) ? lang : "";

                let highlighted = "";
                try {
                    if (validLang && typeof hljs !== 'undefined' && typeof hljs.highlight === 'function') {
                        highlighted = hljs.highlight(codeStr, { language: validLang }).value;
                    } else if (typeof hljs !== 'undefined' && typeof hljs.highlightAuto === 'function' && codeStr.trim().length > 0) {
                        highlighted = hljs.highlightAuto(codeStr).value;
                    } else {
                        highlighted = escapeHtml(codeStr);
                    }
                } catch (hlErr) {
                    highlighted = escapeHtml(codeStr);
                }

                return `
                    <div class="code-block-container my-3 rounded-xl overflow-hidden border border-[#383838] bg-[#181818]" dir="ltr">
                        <div class="code-header flex items-center justify-between px-3.5 py-1.5 bg-[#262626] border-b border-[#383838] text-xs text-gray-300 font-mono">
                            <span class="text-gray-400 font-medium select-none">${escapeHtml(displayLang)}</span>
                            <button onclick="copyCodeBlock(this)" class="copy-code-btn flex items-center gap-1.5 px-2 py-0.5 rounded text-gray-300 hover:text-white hover:bg-[#383838] transition text-xs">
                                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                                </svg>
                                <span>کپی کد</span>
                            </button>
                        </div>
                        <pre class="p-4 overflow-x-auto text-xs md:text-sm font-mono leading-relaxed"><code class="hljs ${validLang ? 'language-' + validLang : ''}">${highlighted}</code></pre>
                    </div>
                `;
            } catch (err) {
                console.error("Renderer code fallback error:", err);
                let fallbackText = "";
                if (codeOrToken && typeof codeOrToken === 'object') {
                    fallbackText = typeof codeOrToken.text === 'string' ? codeOrToken.text : (typeof codeOrToken.raw === 'string' ? codeOrToken.raw : "");
                } else if (codeOrToken != null) {
                    fallbackText = String(codeOrToken);
                }
                return `<div class="code-block-container my-3 rounded-xl overflow-hidden border border-[#383838] bg-[#181818]" dir="ltr"><pre class="p-4 overflow-x-auto text-xs font-mono"><code>${escapeHtml(fallbackText)}</code></pre></div>`;
            }
        };

        try {
            marked.use({ renderer: customRenderer, breaks: true, gfm: true });
        } catch (e) {
            marked.setOptions({ renderer: customRenderer, breaks: true, gfm: true });
        }

        function safeMarkdownParse(text) {
            if (!text) return "";
            try {
                if (typeof marked !== 'undefined') {
                    if (typeof marked.parse === 'function') {
                        return marked.parse(text);
                    } else if (typeof marked === 'function') {
                        return marked(text);
                    }
                }
            } catch (err) {
                console.error("safeMarkdownParse error:", err);
            }
            return `<pre class="whitespace-pre-wrap font-sans">${escapeHtml(text)}</pre>`;
        }

        // Copy Code Functionality
        function copyCodeBlock(button) {
            const container = button.closest('.code-block-container');
            if (!container) return;
            const codeEl = container.querySelector('pre code');
            if (!codeEl) return;
            const text = codeEl.innerText || codeEl.textContent || "";
            
            const showSuccess = () => {
                const originalHTML = button.innerHTML;
                button.innerHTML = `
                    <svg class="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    <span class="text-emerald-400 font-medium">کپی شد!</span>
                `;
                setTimeout(() => {
                    button.innerHTML = originalHTML;
                }, 2000);
            };

            const fallbackCopy = () => {
                try {
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    textarea.style.left = '-999999px';
                    textarea.style.top = '-999999px';
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                    showSuccess();
                } catch (err) {
                    console.error("Copy failed:", err);
                }
            };

            if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                navigator.clipboard.writeText(text).then(showSuccess).catch(fallbackCopy);
            } else {
                fallbackCopy();
            }
        }

        let chats = [];
        let activeChatId = null;
        let activeEventSource = null;
        let activeAbortController = null;
        let liveSyncInterval = null;
        let isCurrentlyGenerating = false;

        window.onload = async function() {
            await fetchChats();
            if (!chats.length) {
                await createNewChat();
            } else {
                const savedActive = localStorage.getItem('last_active_chat_id');
                const target = chats.find(c => c.id === savedActive) || chats[0];
                await switchChat(target.id);
            }
        };

        document.addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
                e.preventDefault();
                createNewChat();
            }
        });

        async function fetchChats() {
            try {
                const res = await fetch('/api/chats');
                chats = await res.json();
                renderSidebar();
            } catch (e) {
                console.error("Failed to load chats", e);
            }
        }

        function renderSidebar() {
            const listEl = document.getElementById('chatList');
            if (!listEl) return;

            let html = '';
            chats.forEach(chat => {
                const isActive = chat.id === activeChatId;
                const safeTitle = (chat.title || "گفتگو").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                html += `
                    <div onclick="switchChat('${chat.id}')" 
                        class="chat-item group flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer text-sm transition border ${isActive ? 'chat-active border-[#3f3f3f] text-white font-medium' : 'border-transparent text-gray-300 hover:bg-[#212121] hover:text-white'}">
                        <div class="flex items-center gap-2.5 truncate flex-1">
                            <span class="text-base ${chat.is_generating ? 'animate-pulse text-emerald-400' : 'text-gray-400'}">💬</span>
                            <span class="truncate text-xs md:text-sm">${safeTitle}</span>
                        </div>
                        <button onclick="deleteChat('${chat.id}', event)" class="delete-btn opacity-0 group-hover:opacity-100 p-1 text-gray-400 hover:text-red-400 transition" title="حذف">
                            🗑️
                        </button>
                    </div>
                `;
            });
            listEl.innerHTML = html;
        }

        async function createNewChat() {
            stopLiveSync();

            try {
                const res = await fetch('/api/chats', { method: 'POST' });
                const newChat = await res.json();
                chats.unshift(newChat);
                renderSidebar();
                await switchChat(newChat.id);
                document.getElementById('userInput')?.focus();
            } catch (e) {
                console.error("Failed to create chat", e);
            }
        }

        async function switchChat(chatId) {
            if (activeChatId === chatId && document.getElementById('messagesContainer').children.length > 0) {
                return;
            }

            stopLiveSync();
            activeChatId = chatId;
            localStorage.setItem('last_active_chat_id', activeChatId);
            renderSidebar();

            const container = document.getElementById('messagesContainer');
            container.innerHTML = '<div class="text-center py-10 text-gray-500 text-sm">در حال بارگذاری گفتگو از سرور...</div>';

            try {
                const res = await fetch('/api/chats/' + chatId);
                if (!res.ok) throw new Error("Chat not found");
                const data = await res.json();

                document.getElementById('currentChatHeaderTitle').innerText = data.title;
                container.innerHTML = '';

                if (data.messages.length === 0 && !data.is_generating) {
                    const baseUrl = window.location.origin + '/v1';
                    container.innerHTML = `
                        <div class="text-center my-8 text-gray-400">
                            <div class="text-4xl mb-3">⚡</div>
                            <h2 class="text-xl font-bold text-gray-200">DeepSeek Uncensored آماده است!</h2>
                            <p class="text-sm mt-1 text-gray-400 mb-4">گفتگوی جدید را شروع کنید. تمام پیام‌ها روی سرور پایدار باقی می‌مانند.</p>
                            <div class="inline-block bg-[#2f2f2f] border border-[#424242] rounded-xl p-3 text-right text-xs max-w-md w-full dir-ltr text-gray-300">
                                <div><b class="text-emerald-400">Base URL:</b> <span>${baseUrl}</span></div>
                                <div><b class="text-emerald-400">Model:</b> deepseek-v4</div>
                                <div><b class="text-emerald-400">API Key:</b> هر مقداری (مثلاً sk-dummy)</div>
                            </div>
                        </div>
                    `;
                } else {
                    data.messages.forEach(msg => {
                        appendMessageUI(msg.role, msg.content, false);
                    });
                }

                if (data.is_generating) {
                    setGeneratingState(true);
                    const botMsgDiv = appendMessageUI("assistant", data.current_text, true);
                    const contentDiv = botMsgDiv.querySelector('.bot-content');
                    listenToLiveStream(chatId, contentDiv, data.current_text);
                } else {
                    setGeneratingState(false);
                }

                container.scrollTop = container.scrollHeight;

            } catch (e) {
                container.innerHTML = `<div class="text-center py-10 text-red-400 text-sm">خطا در دریافت گفتگو: ${e.message}</div>`;
                setGeneratingState(false);
            }
        }

        async function deleteChat(chatId, event) {
            if (event) event.stopPropagation();
            try {
                await fetch('/api/chats/' + chatId, { method: 'DELETE' });
                chats = chats.filter(c => c.id !== chatId);
                if (!chats.length) {
                    await createNewChat();
                } else if (activeChatId === chatId) {
                    await switchChat(chats[0].id);
                } else {
                    renderSidebar();
                }
            } catch (e) {
                console.error("Failed to delete", e);
            }
        }

        async function deleteCurrentChat() {
            if (!activeChatId) return;
            if (confirm("آیا از حذف این گفتگو اطمینان دارید؟")) {
                await deleteChat(activeChatId, null);
            }
        }

        function setGeneratingState(isGen) {
            isCurrentlyGenerating = isGen;
            const sendBtn = document.getElementById('sendBtn');
            const stopBtn = document.getElementById('stopBtn');
            if (isGen) {
                sendBtn.style.display = 'none';
                stopBtn.style.display = 'flex';
            } else {
                sendBtn.style.display = 'flex';
                sendBtn.disabled = false;
                stopBtn.style.display = 'none';
            }
        }

        async function stopCurrentGeneration() {
            if (!activeChatId) return;
            try {
                await fetch('/api/chats/' + activeChatId + '/stop', { method: 'POST' });
            } catch (e) {}
            stopLiveSync();
            setGeneratingState(false);
            await fetchChats();
        }

        function stopLiveSync() {
            if (activeAbortController) {
                try { activeAbortController.abort(); } catch (e) {}
                activeAbortController = null;
            }
            if (liveSyncInterval) {
                clearInterval(liveSyncInterval);
                liveSyncInterval = null;
            }
            if (activeEventSource) {
                try { activeEventSource.close(); } catch (e) {}
                activeEventSource = null;
            }
        }

        async function sendMessage() {
            if (isCurrentlyGenerating) return;
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if (!text || !activeChatId) return;

            input.value = '';
            input.style.height = 'auto';

            const container = document.getElementById('messagesContainer');
            if (container.querySelector('.text-4xl')) {
                container.innerHTML = '';
            }

            appendMessageUI("user", text, true);
            setGeneratingState(true);

            const botMsgDiv = appendMessageUI("assistant", "", true);
            const contentDiv = botMsgDiv.querySelector('.bot-content');
            renderBotContent(contentDiv, "", false);

            await startStreamForMessage(activeChatId, text, contentDiv);
        }

        async function startStreamForMessage(chatId, text, contentDiv) {
            stopLiveSync();
            let accumulated = "";
            let startTime = Date.now();
            let tokenCount = 0;
            activeAbortController = new AbortController();

            // 1. Companion safeguard poller (every 600ms)
            liveSyncInterval = setInterval(async () => {
                if (activeChatId !== chatId) {
                    stopLiveSync();
                    return;
                }
                try {
                    const res = await fetch('/api/chats/' + chatId);
                    if (!res.ok) return;
                    const data = await res.json();
                    if (data.current_text && data.current_text.length > accumulated.length) {
                        accumulated = data.current_text;
                        tokenCount = accumulated.split(/\s+/).length;
                        renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                    }
                    if (!data.is_generating) {
                        stopLiveSync();
                        setGeneratingState(false);
                        const lastMsg = (data.messages && data.messages.length) ? data.messages[data.messages.length - 1] : null;
                        const finalText = (lastMsg && lastMsg.role === 'assistant') ? lastMsg.content : (data.current_text || accumulated);
                        renderBotContent(contentDiv, finalText.trim(), true, tokenCount, startTime);
                        fetchChats();
                    }
                } catch (e) {}
            }, 600);

            // 2. High-speed Direct POST Stream (Streams live every single token through Cloudflare!)
            try {
                const response = await fetch('/api/chats/' + chatId + '/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: text, stream: true }),
                    signal: activeAbortController.signal
                });

                if (!response.ok) throw new Error("Server returned HTTP " + response.status);

                // Update sidebar title in background
                fetchChats();

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || trimmed.startsWith(':')) continue;

                        if (trimmed === 'data: [DONE]') {
                            stopLiveSync();
                            setGeneratingState(false);
                            renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                            fetchChats();
                            return;
                        }

                        if (trimmed.startsWith('data: ')) {
                            try {
                                const parsed = JSON.parse(trimmed.slice(6));
                                if (parsed.full_text && parsed.full_text.length >= accumulated.length) {
                                    accumulated = parsed.full_text;
                                    tokenCount = accumulated.split(/\s+/).length;
                                    renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                                } else if (parsed.token) {
                                    accumulated += parsed.token;
                                    tokenCount++;
                                    renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                                }

                                if (parsed.is_generating === false) {
                                    stopLiveSync();
                                    setGeneratingState(false);
                                    renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                                    fetchChats();
                                    return;
                                }
                            } catch (e) {}
                        }
                    }
                }

                // Stream ended normally
                stopLiveSync();
                setGeneratingState(false);
                renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                fetchChats();

            } catch (err) {
                if (err.name === 'AbortError') return;
                console.warn("Direct POST stream error, falling back to listenToLiveStream:", err);
                listenToLiveStream(chatId, contentDiv, accumulated);
            }
        }

        async function listenToLiveStream(chatId, contentDiv, initialText) {
            stopLiveSync();
            let accumulated = initialText || "";
            let startTime = Date.now();
            let tokenCount = accumulated ? accumulated.split(/\s+/).length : 0;
            activeAbortController = new AbortController();

            // 1. Companion poller (every 600ms)
            liveSyncInterval = setInterval(async () => {
                if (activeChatId !== chatId) {
                    stopLiveSync();
                    return;
                }
                try {
                    const res = await fetch('/api/chats/' + chatId);
                    if (!res.ok) return;
                    const data = await res.json();
                    if (data.current_text && data.current_text.length > accumulated.length) {
                        accumulated = data.current_text;
                        tokenCount = accumulated.split(/\s+/).length;
                        renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                    }
                    if (!data.is_generating) {
                        stopLiveSync();
                        setGeneratingState(false);
                        const lastMsg = (data.messages && data.messages.length) ? data.messages[data.messages.length - 1] : null;
                        const finalText = (lastMsg && lastMsg.role === 'assistant') ? lastMsg.content : (data.current_text || accumulated);
                        renderBotContent(contentDiv, finalText.trim(), true, tokenCount, startTime);
                        fetchChats();
                    }
                } catch (e) {}
            }, 600);

            // 2. High-speed POST stream (Streams live through Cloudflare without buffering!)
            try {
                const response = await fetch('/api/chats/' + chatId + '/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    signal: activeAbortController.signal
                });

                if (!response.ok) throw new Error("HTTP error " + response.status);

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || trimmed.startsWith(':')) continue;

                        if (trimmed === 'data: [DONE]') {
                            stopLiveSync();
                            setGeneratingState(false);
                            renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                            fetchChats();
                            return;
                        }

                        if (trimmed.startsWith('data: ')) {
                            try {
                                const parsed = JSON.parse(trimmed.slice(6));
                                if (parsed.full_text && parsed.full_text.length >= accumulated.length) {
                                    accumulated = parsed.full_text;
                                    tokenCount = accumulated.split(/\s+/).length;
                                    renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                                } else if (parsed.token) {
                                    accumulated += parsed.token;
                                    tokenCount++;
                                    renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                                }

                                if (parsed.is_generating === false) {
                                    stopLiveSync();
                                    setGeneratingState(false);
                                    renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                                    fetchChats();
                                    return;
                                }
                            } catch (e) {}
                        }
                    }
                }

                stopLiveSync();
                setGeneratingState(false);
                renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                fetchChats();

            } catch (err) {
                if (err.name === 'AbortError') return;
                console.warn("POST stream failed, falling back to EventSource:", err);
                try {
                    activeEventSource = new EventSource('/api/chats/' + chatId + '/stream');
                    activeEventSource.onmessage = function(event) {
                        if (event.data === '[DONE]') {
                            stopLiveSync();
                            setGeneratingState(false);
                            renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                            fetchChats();
                            return;
                        }
                        try {
                            const parsed = JSON.parse(event.data);
                            if (parsed.full_text && parsed.full_text.length >= accumulated.length) {
                                accumulated = parsed.full_text;
                                tokenCount = accumulated.split(/\s+/).length;
                                renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                            } else if (parsed.token) {
                                accumulated += parsed.token;
                                tokenCount++;
                                renderBotContent(contentDiv, accumulated, false, tokenCount, startTime);
                            }
                            if (parsed.is_generating === false) {
                                stopLiveSync();
                                setGeneratingState(false);
                                renderBotContent(contentDiv, accumulated.trim(), true, tokenCount, startTime);
                                fetchChats();
                            }
                        } catch (e) {}
                    };
                    activeEventSource.onerror = function() {
                        if (activeEventSource) {
                            activeEventSource.close();
                            activeEventSource = null;
                        }
                    };
                } catch (e) {}
            }
        }

        function renderBotContent(container, text, isFinal, tokenCount, startTime) {
            let html = "";
            if (!text && !isFinal) {
                html = `
                    <div class="flex items-center gap-2.5 text-gray-400 text-sm font-medium py-1 animate-pulse">
                        <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
                        <span class="italic text-gray-300">در حال نوشتن پاسخ...</span>
                    </div>
                `;
            } else if (text) {
                html = `<div class="leading-relaxed bot-markdown">${safeMarkdownParse(text)}</div>`;
                if (!isFinal) {
                    html += `<span class="inline-block w-1.5 h-4 bg-emerald-400 animate-pulse align-middle ml-1"></span>`;
                }
            }

            if (isFinal && tokenCount > 0) {
                const elapsed = (Date.now() - startTime) / 1000;
                const tps = (tokenCount / Math.max(0.1, elapsed)).toFixed(1);
                html += `<div class="mt-2.5"><span class="tps-badge">⚡ ${tps} توکن/ثانیه | ${elapsed.toFixed(1)}s</span></div>`;
            }

            container.innerHTML = html;

            const scrollContainer = document.getElementById('messagesContainer');
            if (scrollContainer) {
                scrollContainer.scrollTop = scrollContainer.scrollHeight;
            }
        }

        function appendMessageUI(role, text, shouldScroll = true) {
            const container = document.getElementById('messagesContainer');
            const wrapper = document.createElement('div');
            wrapper.className = `flex gap-4 ${role === 'user' ? 'justify-end' : 'justify-start'}`;

            if (role === 'user') {
                wrapper.innerHTML = `
                    <div class="user-msg max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed text-gray-100 whitespace-pre-wrap">
                        ${escapeHtml(text)}
                    </div>
                `;
            } else {
                const contentHtml = text
                    ? `<div class="leading-relaxed bot-markdown">${safeMarkdownParse(text)}</div>`
                    : '<div class="flex items-center gap-2.5 text-gray-400 text-sm font-medium py-1 animate-pulse"><span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span><span class="italic text-gray-300">در حال نوشتن پاسخ...</span></div>';

                wrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-emerald-600 flex items-center justify-center text-xs font-bold shrink-0 text-white shadow">AI</div>
                    <div class="bot-msg flex-1 max-w-[90%] text-sm leading-relaxed text-gray-200 bot-content">
                        ${contentHtml}
                    </div>
                `;
            }

            container.appendChild(wrapper);
            if (shouldScroll) {
                container.scrollTop = container.scrollHeight;
            }
            return wrapper;
        }

        function copyBaseUrl() {
            const baseUrl = window.location.origin + '/v1';
            const fallback = () => {
                try {
                    const textarea = document.createElement('textarea');
                    textarea.value = baseUrl;
                    textarea.style.position = 'fixed';
                    textarea.style.left = '-999999px';
                    textarea.style.top = '-999999px';
                    document.body.appendChild(textarea);
                    textarea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textarea);
                    alert('✅ Base URL کپی شد:\\n' + baseUrl);
                } catch (e) {
                    prompt('Base URL:', baseUrl);
                }
            };

            if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                navigator.clipboard.writeText(baseUrl).then(() => {
                    alert('✅ Base URL کپی شد:\\n' + baseUrl);
                }).catch(fallback);
            } else {
                fallback();
            }
        }

        function autoGrow(element) {
            element.style.height = "5px";
            element.style.height = (Math.min(element.scrollHeight, 180)) + "px";
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)

# 6. Start Cloudflare Tunnel in background
def run_tunnel():
    print("Starting Cloudflare Tunnel on port 8000...", flush=True)
    proc = subprocess.Popen(
        [CLOUDFLARED_BIN, "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
        if match:
            url = match.group(0)
            print("\n" + "=" * 65, flush=True)
            print(f"🔥 BASE URL (OPENAI COMPATIBLE): {url}/v1", flush=True)
            print(f"🔥 CHATGPT WEB UI: {url}", flush=True)
            print("=" * 65 + "\n", flush=True)
            try:
                with open("/tmp/public_url.txt", "w") as f:
                    f.write(url)
            except:
                pass

# Clean up any stale tunnel or port 8000 processes before launching
try:
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
    subprocess.run(["fuser", "-k", "8000/tcp"], check=False)
    if os.path.exists("/tmp/public_url.txt"):
        os.remove("/tmp/public_url.txt")
except Exception:
    pass
time.sleep(1)

Thread(target=run_tunnel, daemon=True).start()

# 7. Start Uvicorn Server in dedicated thread (avoids Jupyter/Colab event loop conflicts)
if __name__ == "__main__":
    import uvicorn
    import time
    
    def start_server():
        config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        server.run()

    server_thread = Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait for the tunnel URL to be generated and verified
    print("Waiting for Cloudflare Tunnel URL...", flush=True)
    tunnel_url = None
    for _ in range(60):
        if os.path.exists("/tmp/public_url.txt"):
            try:
                with open("/tmp/public_url.txt", "r") as f:
                    u = f.read().strip()
                    if u.startswith("https://"):
                        tunnel_url = u
                        break
            except Exception:
                pass
        time.sleep(1)

    if tunnel_url:
        print(f"\n🚀 SERVER SUCCESSFULLY INITIALIZED & LIVE: {tunnel_url}", flush=True)
    else:
        print("⚠️ Tunnel URL file not ready within 60s, continuing...", flush=True)

    # In standalone CLI python (non-Jupyter), keep main thread alive
    if "ipykernel" not in sys.modules:
        try:
            while True:
                server_thread.join(timeout=1.0)
                if not server_thread.is_alive():
                    break
        except (KeyboardInterrupt, SystemExit):
            pass

