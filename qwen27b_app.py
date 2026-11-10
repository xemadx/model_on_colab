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

print("=== Starting Qwen 27B Q5_K (Abliterated) Server with llama-cpp-python ===", flush=True)

# 1. Setup CUDA dynamic linker
print("Configuring CUDA libraries...", flush=True)
import glob
nvidia_dirs = glob.glob("/usr/local/lib/python3.*/dist-packages/nvidia/*/lib")
if nvidia_dirs:
    with open("/etc/ld.so.conf.d/nvidia-python.conf", "w") as f:
        for d in nvidia_dirs:
            f.write(d + "\n")
    subprocess.run(["ldconfig"])

# 2. Install dependencies if needed
try:
    import llama_cpp
except ImportError:
    print("Installing llama-cpp-python (CUDA 12.4), fastapi, uvicorn, huggingface_hub...", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python",
        "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"
    ])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "huggingface_hub"
])

# 3. Re-check CUDA libraries after pip installs
nvidia_dirs = glob.glob("/usr/local/lib/python3.*/dist-packages/nvidia/*/lib")
if nvidia_dirs:
    with open("/etc/ld.so.conf.d/nvidia-python.conf", "w") as f:
        for d in nvidia_dirs:
            f.write(d + "\n")
    subprocess.run(["ldconfig"])

# 4. Download Cloudflare Tunnel binary
CLOUDFLARED_BIN = "/tmp/cloudflared"
if not os.path.exists(CLOUDFLARED_BIN):
    print("Downloading cloudflared...", flush=True)
    subprocess.run([
        "curl", "-sL", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        "-o", CLOUDFLARED_BIN
    ], check=True)
    os.chmod(CLOUDFLARED_BIN, 0o755)

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse as EventStreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Download Model: Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf
MODEL_DIR = "/content/models"
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf")

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 18 * 1024 * 1024 * 1024:
    print(f"Downloading Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf (~19.5 GB)...", flush=True)
    try:
        hf_hub_download(
            repo_id="huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF",
            filename="Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf",
            local_dir=MODEL_DIR
        )
        print("Download finished! Saved to:", MODEL_PATH, flush=True)
    except Exception as e:
        print(f"hf_hub_download failed ({e}), trying aria2c...", flush=True)
        url = "https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf"
        subprocess.run(["apt-get", "install", "-y", "-q", "aria2"])
        subprocess.run([
            "aria2c", "-c", "-x", "16", "-s", "16", "-k", "1M",
            "-d", MODEL_DIR, "-o", "Huihui-Qwen3.8-27B-abliterated-Q5_K.gguf", url
        ], check=True)
        if not os.path.exists(MODEL_PATH):
            subprocess.run([
                "curl", "-L", "-C", "-", "-o", MODEL_PATH, url
            ], check=True)
        print("Download finished! Saved to:", MODEL_PATH, flush=True)

print(f"Loading {MODEL_PATH} on Nvidia L4 with full GPU offloading...", flush=True)
try:
    print("Initializing Qwen 27B with 32,768 (32K) context window and Q4_0 KV cache...", flush=True)
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,      # Offload ALL layers to Nvidia L4 VRAM
        n_ctx=32768,          # 32,768 tokens (Full 32K Context Window)
        flash_attn=True,      # Flash attention for maximum speed & minimal memory
        type_k=2,             # GGML_TYPE_Q4_0 for KV cache (~2.1 GB VRAM for 32k tokens)
        type_v=2,             # GGML_TYPE_Q4_0 for KV cache
        n_batch=512,
        chat_format="chatml", # ChatML format
        verbose=False
    )
    print("Initialized successfully with 32K context (32,768 tokens)!", flush=True)
except Exception as e32:
    print(f"32K initialization fallback ({e32}), trying 16,384 tokens with Q8_0 KV cache...", flush=True)
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=16384,          # 16,384 tokens context window
        flash_attn=True,
        type_k=8,             # GGML_TYPE_Q8_0
        type_v=8,             # GGML_TYPE_Q8_0
        n_batch=512,
        chat_format="chatml",
        verbose=False
    )
    print("Initialized successfully with 16K context (16,384 tokens)!", flush=True)

# 5. Anti-hallucination & God-tier Coding System Prompts
DEFAULT_SYSTEM_PROMPT = """You are an elite, world-class programmer and software architect.
CRITICAL INSTRUCTIONS:
1. Always wrap ALL internal reasoning, analysis, edge cases, and architectural design strictly inside <think> and </think> tags.
   Example format:
   <think>
   - Step 1: Analyze code bugs...
   - Step 2: Implement robust solution...
   </think>
2. After </think>, provide:
   - Concise Persian (فارسی) explanations of the root causes and architecture.
   - Flawless, complete, production-ready Python code in standard markdown code blocks (```python ... ```).
3. Strictly ZERO hallucination: never invent non-existent APIs or arguments. Ensure syntactically and logically perfect code.
4. Never output English planning or monologue outside <think> tags."""

FAST_SYSTEM_PROMPT = """You are an elite, world-class programmer and software architect.
CRITICAL INSTRUCTIONS:
1. DO NOT output any internal monologue, reasoning, or English planning. Do NOT use <think> tags.
2. Directly provide:
   - Concise Persian (فارسی) explanations.
   - Complete, flawless, production-ready Python code inside standard markdown code blocks (```python ... ```).
3. Strictly ZERO hallucination: use only verified libraries (e.g. aiohttp, asyncio), robust error handling, and optimal concurrency."""

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 3)

def trim_messages(messages: list, max_input_tokens: int = 24000) -> list:
    if not messages:
        return messages
    sys_msg = None
    chat_msgs = []
    for m in messages:
        if m.get("role") == "system" and sys_msg is None:
            sys_msg = m
        else:
            chat_msgs.append(m)
    if not chat_msgs:
        return [sys_msg] if sys_msg else []

    latest_msg = chat_msgs[-1]
    middle_msgs = chat_msgs[:-1]

    sys_tokens = estimate_tokens(sys_msg.get("content", "")) if sys_msg else 0
    latest_tokens = estimate_tokens(latest_msg.get("content", ""))
    available_budget = max(2000, max_input_tokens - sys_tokens - latest_tokens)

    selected_middle = []
    used_tokens = 0
    for m in reversed(middle_msgs):
        cost = estimate_tokens(m.get("content", ""))
        if used_tokens + cost <= available_budget:
            selected_middle.insert(0, m)
            used_tokens += cost
        else:
            break

    result = []
    if sys_msg:
        result.append(sys_msg)
    result.extend(selected_middle)
    result.append(latest_msg)
    return result

def clean_thinking(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip("\n ")
    elif "<think>" in text:
        return ""
    return text.lstrip("\n ")

# 6. SQLite Database for Server-Side Persistence
DB_PATH = "/content/qwen27b_chats.db"

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

# Active generation states
active_sessions = {}
sessions_lock = threading.Lock()
llm_lock = threading.Lock()

app = FastAPI(title="Qwen 27B Q5_K Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background generation worker
def generate_chat_worker(chat_id: str, prompt_messages: list, think_mode: bool = False):
    if chat_id not in active_sessions:
        return
    session = active_sessions[chat_id]
    stop_event = session["stop_event"]
    session["is_generating"] = True
    session["current_text"] = ""

    # Insert system prompt if not present
    msgs = []
    has_system = any(m.get("role") == "system" for m in prompt_messages)
    if not has_system:
        chosen_sys = DEFAULT_SYSTEM_PROMPT if think_mode else FAST_SYSTEM_PROMPT
        msgs.append({"role": "system", "content": chosen_sys})
    msgs.extend(prompt_messages)

    # Context window guard: smart trim to ensure inputs never overflow
    trimmed_msgs = trim_messages(msgs, max_input_tokens=24000)

    raw_response = ""
    token_count = 0

    try:
        with llm_lock:
            # Low temperature 0.2 for strictly logical code without hallucination
            # max_tokens=8192 for complete, un-truncated scripts
            response_stream = llm.create_chat_completion(
                messages=trimmed_msgs,
                stream=True,
                temperature=0.2,
                top_p=0.9,
                repeat_penalty=1.1,
                max_tokens=8192
            )

            for chunk in response_stream:
                if stop_event.is_set():
                    break
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        raw_response += content
                        token_count += 1
                        session["current_text"] = raw_response
    except Exception as e:
        err_msg = str(e)
        if "exceed context window" in err_msg:
            raw_response += "\n\n⚠️ **هشدار حافظه:** حجم ورودی و تاریخچه فراتر از ظرفیت مجاز رفت. گفتگو به صورت خودکار بهینه‌سازی خواهد شد."
        else:
            raw_response += f"\n[خطا در پردازش مدل: {err_msg}]"
    finally:
        session["is_generating"] = False
        session["current_text"] = raw_response.strip()

        # Save to database
        if raw_response.strip():
            conn = get_db()
            c = conn.cursor()
            c.execute(
                "INSERT INTO messages (chat_id, role, content, created) VALUES (?, ?, ?, ?)",
                (chat_id, "assistant", raw_response.strip(), int(time.time()))
            )
            c.execute("SELECT COUNT(*) as cnt FROM messages WHERE chat_id = ? AND role = 'user'", (chat_id,))
            cnt = c.fetchone()["cnt"]
            if cnt <= 1:
                answer_part = raw_response.split("</think>", 1)[1] if "</think>" in raw_response else raw_response
                first_line = answer_part.strip().split("\n")[0][:45]
                c.execute("UPDATE chats SET title = ? WHERE id = ? AND title = 'گفتگوی جدید'", (first_line, chat_id))
            conn.commit()
            conn.close()

# REST Endpoints for Chats
@app.get("/api/chats")
def list_chats():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, title, created FROM chats ORDER BY created DESC")
    chats = [dict(row) for row in c.fetchall()]
    conn.close()
    return chats

@app.post("/api/chats")
def create_chat():
    cid = f"chat_{uuid.uuid4().hex[:10]}"
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO chats (id, title, created) VALUES (?, ?, ?)", (cid, "گفتگوی جدید", int(time.time())))
    conn.commit()
    conn.close()
    return {"id": cid, "title": "گفتگوی جدید"}

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, title, created FROM chats WHERE id = ?", (chat_id,))
    chat = c.fetchone()
    if not chat:
        conn.close()
        return JSONResponse({"error": "Chat not found"}, status_code=404)
    c.execute("SELECT role, content, created FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    messages = [dict(row) for row in c.fetchall()]
    conn.close()

    session = active_sessions.get(chat_id, {})
    is_gen = session.get("is_generating", False)
    curr_text = session.get("current_text", "")

    return {
        "id": chat["id"],
        "title": chat["title"],
        "messages": messages,
        "is_generating": is_gen,
        "current_text": curr_text
    }

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str):
    with sessions_lock:
        if chat_id in active_sessions:
            active_sessions[chat_id]["stop_event"].set()
            active_sessions.pop(chat_id, None)
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}

@app.post("/api/chats/{chat_id}/send")
async def send_chat_message(chat_id: str, req: Request):
    data = await req.json()
    user_content = data.get("content", "").strip()
    think_mode = data.get("think_mode", False)

    if not user_content:
        return JSONResponse({"error": "Message is empty"}, status_code=400)

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM chats WHERE id = ?", (chat_id,))
    if not c.fetchone():
        c.execute("INSERT INTO chats (id, title, created) VALUES (?, ?, ?)", (chat_id, "گفتگوی جدید", int(time.time())))
        conn.commit()

    # Save user message
    c.execute(
        "INSERT INTO messages (chat_id, role, content, created) VALUES (?, ?, ?, ?)",
        (chat_id, "user", user_content, int(time.time()))
    )
    conn.commit()

    # Fetch full history
    c.execute("SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    history = [{"role": row["role"], "content": row["content"]} for row in c.fetchall()]
    conn.close()

    # Initialize session
    with sessions_lock:
        if chat_id in active_sessions and active_sessions[chat_id].get("is_generating", False):
            active_sessions[chat_id]["stop_event"].set()
            time.sleep(0.1)

        active_sessions[chat_id] = {
            "is_generating": True,
            "current_text": "",
            "stop_event": threading.Event()
        }

    # Start generation thread
    t = Thread(target=generate_chat_worker, args=(chat_id, history, think_mode), daemon=True)
    t.start()

    return {"status": "generating", "chat_id": chat_id}

# SSE Streaming endpoint for real-time live tokens
async def event_generator(chat_id: str):
    last_len = 0
    while True:
        session = active_sessions.get(chat_id)
        if not session:
            break
        current_text = session.get("current_text", "")
        is_gen = session.get("is_generating", False)

        if len(current_text) > last_len:
            new_chunk = current_text[last_len:]
            last_len = len(current_text)
            payload = json.dumps({"token": new_chunk, "full_text": current_text, "is_generating": is_gen})
            yield f"data: {payload}\n\n"

        if not is_gen:
            payload = json.dumps({"token": "", "full_text": current_text, "is_generating": False})
            yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"
            break

        await asyncio.sleep(0.05)

@app.get("/api/chats/{chat_id}/stream")
def stream_get(chat_id: str):
    return EventStreamingResponse(event_generator(chat_id))

@app.post("/api/chats/{chat_id}/stream")
def stream_post(chat_id: str):
    return EventStreamingResponse(event_generator(chat_id))

@app.post("/api/chats/{chat_id}/stop")
def stop_generation(chat_id: str):
    with sessions_lock:
        if chat_id in active_sessions:
            active_sessions[chat_id]["stop_event"].set()
            active_sessions[chat_id]["is_generating"] = False
    return {"status": "stopped"}

# OpenAI Compatible Endpoints
@app.get("/health")
@app.get("/v1/health")
def health():
    return {"status": "ok", "model": "huihui-ai/Huihui-Qwen3.8-27B-abliterated-Q5_K", "gpu": True}

@app.get("/v1/models")
def get_models():
    return {
        "object": "list",
        "data": [
            {"id": "huihui-ai/Huihui-Qwen3.8-27B-abliterated-Q5_K", "object": "model", "owned_by": "huihui-ai"},
            {"id": "qwen-27b", "object": "model", "owned_by": "huihui-ai"},
            {"id": "gpt-4o", "object": "model", "owned_by": "huihui-ai"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    data = await req.json()
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    temperature = data.get("temperature", 0.2)
    req_max = data.get("max_tokens")
    max_tokens = min(req_max, 8192) if req_max else 8192

    think_mode = data.get("think_mode", False)

    # Prepend coding system prompt if none
    if not any(m.get("role") == "system" for m in messages):
        chosen_sys = DEFAULT_SYSTEM_PROMPT if think_mode else FAST_SYSTEM_PROMPT
        messages = [{"role": "system", "content": chosen_sys}] + messages

    # Guard context window
    messages = trim_messages(messages, max_input_tokens=24000)

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if stream:
        def sse_stream():
            yield f": {' ' * 2048}\n\n"
            initial_chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": "qwen-27b", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(initial_chunk)}\n\n"

            with llm_lock:
                for chunk in llm.create_chat_completion(
                    messages=messages, stream=True, temperature=temperature, top_p=0.9, repeat_penalty=1.1, max_tokens=max_tokens
                ):
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            payload = {
                                "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                                "model": "qwen-27b", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]
                            }
                            yield f"data: {json.dumps(payload)}\n\n"

            final_chunk = {
                "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": "qwen-27b", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return EventStreamingResponse(sse_stream(), media_type="text/event-stream")
    else:
        with llm_lock:
            resp = llm.create_chat_completion(
                messages=messages, stream=False, temperature=temperature, top_p=0.9, repeat_penalty=1.1, max_tokens=max_tokens
            )
        return resp

# Web UI HTML
HTML_CONTENT = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen 3.8 27B Q5_K (Elite Coding & Abliterated)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        body { background-color: #212121; color: #ececec; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
        .sidebar { background-color: #171717; }
        .user-msg { background-color: #2f2f2f; }
        .bot-msg { background-color: transparent; }
        
        /* Force scrollbar on the right for messages scroll wrapper */
        #messagesScrollWrapper {
            direction: ltr !important;
            overflow-y: auto;
            height: 100%;
        }
        #messagesContainer {
            direction: rtl !important;
            min-height: 100%;
        }

        .code-block-container {
            border: 1px solid #383838;
            border-radius: 0.75rem;
            overflow: hidden;
            margin: 1rem 0;
            background-color: #181818;
            direction: ltr !important;
            text-align: left !important;
            unicode-bidi: isolate;
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
            direction: ltr !important;
        }
        .copy-code-btn {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            background: transparent;
            border: none;
            color: #b3b3b3;
            cursor: pointer;
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 0.375rem;
            transition: all 0.15s ease;
            direction: rtl;
        }
        .copy-code-btn:hover { background-color: #383838; color: #ffffff; }
        .copy-code-btn.copied { color: #10a37f; }
        .code-block-container pre { margin: 0; padding: 1rem; overflow-x: auto; font-size: 0.875rem; line-height: 1.5; direction: ltr !important; text-align: left !important; }
        .code-block-container code { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; direction: ltr !important; text-align: left !important; }
        
        .markdown-body { font-size: 0.95rem; line-height: 1.8; word-break: break-word; }
        .markdown-body p, .markdown-body li { direction: auto; text-align: start; unicode-bidi: plaintext; }
        .markdown-body p { margin-bottom: 0.85rem; }
        .markdown-body ul, .markdown-body ol { margin-right: 1.5rem; margin-bottom: 0.85rem; }
        .markdown-body ul { list-style-type: disc; }
        .markdown-body ol { list-style-type: decimal; }
        .markdown-body h1, .markdown-body h2, .markdown-body h3 { font-weight: 700; margin-top: 1.25rem; margin-bottom: 0.6rem; direction: auto; }
        .markdown-body code:not(pre code) { background-color: #2d2d2d; padding: 0.2rem 0.4rem; border-radius: 0.375rem; font-size: 0.85em; font-family: ui-monospace, SFMono-Regular, monospace; direction: ltr !important; display: inline-block; unicode-bidi: isolate; }
        
        /* Thinking box */
        .think-container {
            direction: ltr !important;
        }
        .think-content {
            direction: auto !important;
            text-align: start !important;
            unicode-bidi: plaintext !important;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }
        
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #424242; border-radius: 3px; }
    </style>
</head>
<body class="h-screen flex overflow-hidden">

    <!-- Sidebar -->
    <div class="sidebar w-64 flex flex-col flex-shrink-0 border-l border-zinc-800 select-none z-20">
        <div class="p-3">
            <button onclick="createNewChat()" class="w-full flex items-center justify-between px-3 py-2.5 rounded-lg border border-zinc-700 hover:bg-zinc-800 transition text-sm font-medium">
                <span class="flex items-center gap-2">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
                    گفتگوی جدید
                </span>
                <span class="text-xs bg-emerald-600/20 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded">27B Q5</span>
            </button>
        </div>

        <div class="flex-1 overflow-y-auto px-2 space-y-1" id="chatList"></div>

        <div class="p-3 border-t border-zinc-800 text-xs text-zinc-400 space-y-2">
            <div class="flex items-center justify-between">
                <span>سرور مستقل:</span>
                <span class="text-emerald-400 font-bold">Nvidia L4 (24GB)</span>
            </div>
            <div class="flex items-center justify-between">
                <span>مدل هوش مصنوعی:</span>
                <span class="text-zinc-300">Qwen 3.8 27B</span>
            </div>
            <div class="flex items-center justify-between">
                <span>کوانتیزاسیون:</span>
                <span class="text-amber-400 font-mono">Q5_K (High Prec)</span>
            </div>
            <div class="flex items-center justify-between">
                <span>پنجره کانتکست:</span>
                <span class="text-emerald-400 font-mono">32,768 (32K)</span>
            </div>
            <div class="flex items-center justify-between">
                <span>حداکثر طول پاسخ:</span>
                <span class="text-blue-400 font-mono">8,192 Tokens</span>
            </div>
            <button onclick="copyBaseUrl()" class="w-full mt-2 py-1.5 px-2 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded text-center transition flex items-center justify-center gap-1">
                <span>📋 کپی Base URL برای API</span>
            </button>
        </div>
    </div>

    <!-- Main Chat Window -->
    <div class="flex-1 flex flex-col h-full bg-[#212121] relative overflow-hidden">
        <!-- Top bar -->
        <div class="h-14 border-b border-zinc-800 flex items-center justify-between px-6 bg-[#212121]/90 backdrop-blur z-10">
            <div class="flex items-center gap-3">
                <div class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></div>
                <h1 class="font-bold text-sm tracking-wide flex items-center gap-2">
                    <span>Qwen 3.8 27B Abliterated</span>
                    <span class="text-xs px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 border border-zinc-700">Q5_K • Elite Coder</span>
                </h1>
            </div>
            <div class="flex items-center gap-4">
                <label class="flex items-center gap-1.5 cursor-pointer text-xs text-zinc-300 hover:text-white select-none bg-zinc-800/80 hover:bg-zinc-700/80 px-2.5 py-1.5 rounded-lg border border-zinc-700 transition">
                    <input type="checkbox" id="thinkModeToggle" class="rounded bg-zinc-700 border-zinc-600 text-emerald-500 focus:ring-0 cursor-pointer">
                    <span class="font-medium">🧠 تفکر عمیق (Deep Reasoning)</span>
                </label>
                <div class="text-xs text-zinc-400 flex items-center gap-2">
                    <span class="inline-block w-2 h-2 rounded-full bg-green-500"></span>
                    <span>Zero Hallucination</span>
                </div>
            </div>
        </div>

        <!-- Messages Scroll Wrapper (dir=ltr ensures scrollbar is on the RIGHT!) -->
        <div id="messagesScrollWrapper" class="flex-1 p-4 md:px-24 md:py-6 relative">
            <div id="messagesContainer" class="space-y-6 pb-28">
                <div class="h-full flex flex-col items-center justify-center text-center text-zinc-500 space-y-3 pt-24">
                    <div class="w-16 h-16 rounded-2xl bg-zinc-800/80 border border-zinc-700 flex items-center justify-center text-2xl text-emerald-400">
                        ⚡
                    </div>
                    <h2 class="text-xl font-bold text-zinc-200">کدنویسی فوق‌پیشرفته و بدون توهم با Qwen 27B</h2>
                    <p class="max-w-md text-sm text-zinc-400">یک مسئله، پروژه، الگوریتم یا باگ پیچیده را بپرسید تا با نهایت دقت و سرعت برایتان حل کند.</p>
                </div>
            </div>
        </div>

        <!-- Floating Scroll Down Button -->
        <button id="scrollBottomBtn" onclick="scrollToBottomSmooth()" 
            class="hidden absolute bottom-24 left-1/2 transform -translate-x-1/2 z-30 bg-zinc-800/95 hover:bg-zinc-700 text-zinc-200 border border-zinc-600 px-4 py-2 rounded-full text-xs shadow-2xl transition flex items-center gap-2 backdrop-blur cursor-pointer select-none">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3"></path></svg>
            <span>مشاهده جدیدترین پیام</span>
        </button>

        <!-- Input Bar -->
        <div class="p-4 md:px-24 bg-[#212121] border-t border-zinc-800/80 z-20">
            <div class="max-w-4xl mx-auto relative flex items-center">
                <textarea id="userInput" rows="1" placeholder="پیام خود را به Qwen 27B بنویسید..." 
                    class="w-full bg-[#2f2f2f] text-zinc-100 placeholder-zinc-500 rounded-2xl py-3.5 pr-4 pl-24 focus:outline-none focus:ring-1 focus:ring-zinc-500 resize-none max-h-48 text-sm"
                    onkeydown="handleKeyDown(event)"></textarea>

                <div class="absolute left-3 flex items-center gap-1.5">
                    <button id="stopBtn" onclick="stopCurrentGeneration()" class="hidden bg-zinc-700 hover:bg-zinc-600 text-white px-2.5 py-1.5 rounded-xl text-xs font-medium transition flex items-center gap-1">
                        <span class="w-2 h-2 bg-red-400 rounded-sm"></span>
                        <span>توقف</span>
                    </button>
                    <button id="sendBtn" onclick="sendMessage()" class="bg-zinc-100 hover:bg-white text-black p-2 rounded-xl transition disabled:opacity-30">
                        <svg class="w-4 h-4 transform rotate-180" fill="currentColor" viewBox="0 0 20 20"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"></path></svg>
                    </button>
                </div>
            </div>
            <div class="text-center text-[11px] text-zinc-500 mt-2">
                سرور مستقل Qwen 27B Q5_K • پاسخ‌ها و تاریخچه به صورت دائمی در سرور ذخیره می‌شوند.
            </div>
        </div>
    </div>

    <script>
        let activeChatId = null;
        let isCurrentlyGenerating = false;
        let activeAbortController = null;
        let userScrolledUp = false;

        // Custom Marked Renderer for ChatGPT-like code blocks
        const customRenderer = {
            code(codeOrToken, language) {
                let code = "";
                let lang = "";
                if (typeof codeOrToken === 'object' && codeOrToken !== null) {
                    code = codeOrToken.text || codeOrToken.raw || "";
                    lang = codeOrToken.lang || "";
                } else {
                    code = String(codeOrToken || "");
                    lang = String(language || "");
                }

                lang = lang.trim().toLowerCase() || 'code';
                let highlighted = code;
                if (window.hljs && hljs.getLanguage(lang)) {
                    try { highlighted = hljs.highlight(code, { language: lang }).value; } catch (e) {}
                }

                const uniqueId = 'code_' + Math.random().toString(36).substring(2, 9);
                window[uniqueId] = code;

                return `
                <div class="code-block-container" dir="ltr">
                    <div class="code-header" dir="ltr">
                        <span>${lang}</span>
                        <button class="copy-code-btn" onclick="copyCodeBlock('${uniqueId}', this)">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                            <span>کپی کد</span>
                        </button>
                    </div>
                    <pre dir="ltr"><code class="hljs language-${lang}" dir="ltr">${highlighted}</code></pre>
                </div>`;
            }
        };

        marked.use({ renderer: customRenderer, breaks: true, gfm: true });

        function escapeHtml(str) {
            return (str || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
        }

        function formatUserText(text) {
            if (!text) return "";
            if (text.includes("```")) {
                return marked.parse(text);
            }
            const lines = text.split("\\n");
            let formatted = "";
            let inCodeGroup = false;
            let codeGroup = [];

            const isCodeLine = (l) => {
                const tr = l.trim();
                return tr.startsWith("def ") || tr.startsWith("class ") || tr.startsWith("import ") || 
                       tr.startsWith("from ") || tr.startsWith("async ") || tr.startsWith("if ") || 
                       tr.startsWith("for ") || tr.startsWith("while ") || tr.startsWith("try:") || 
                       tr.startsWith("except") || tr.startsWith("return ") || tr.startsWith("print(") ||
                       tr.endsWith(":") || tr.startsWith("@") || /^[a-zA-Z_][a-zA-Z0-9_]*\\s*=/.test(tr);
            };

            for (const line of lines) {
                if (isCodeLine(line)) {
                    inCodeGroup = true;
                    codeGroup.push(line);
                } else {
                    if (inCodeGroup && codeGroup.length > 0) {
                        formatted += `<div class="font-mono text-xs bg-zinc-900/90 p-3 rounded-lg my-2 border border-zinc-700/60 overflow-x-auto" dir="ltr" style="text-align: left; unicode-bidi: isolate;">${escapeHtml(codeGroup.join("\\n"))}</div>`;
                        codeGroup = [];
                        inCodeGroup = false;
                    }
                    if (line.trim()) {
                        formatted += `<div style="direction: auto; unicode-bidi: plaintext; text-align: start;">${escapeHtml(line)}</div>`;
                    } else {
                        formatted += `<div class="h-2"></div>`;
                    }
                }
            }
            if (inCodeGroup && codeGroup.length > 0) {
                formatted += `<div class="font-mono text-xs bg-zinc-900/90 p-3 rounded-lg my-2 border border-zinc-700/60 overflow-x-auto" dir="ltr" style="text-align: left; unicode-bidi: isolate;">${escapeHtml(codeGroup.join("\\n"))}</div>`;
            }
            return formatted || escapeHtml(text);
        }

        function safeMarkdownParse(content) {
            try {
                let rawText = content || "";
                let text = rawText.trim();
                const thinkModeToggle = document.getElementById('thinkModeToggle');
                const isThinkModeOn = thinkModeToggle ? thinkModeToggle.checked : false;

                // 1. Tag-based Thinking extraction (<think>...</think>)
                if (text.includes('</think>')) {
                    const parts = text.split('</think>');
                    const thinkPart = parts[0].replace('<think>', '').trim();
                    const answerPart = parts[1].trim();
                    let html = '';
                    if (thinkPart) {
                        html += `
                            <details class="think-container mb-4 rounded-xl bg-zinc-800/40 border border-zinc-700/60 p-3 text-xs" dir="ltr" ${isThinkModeOn ? 'open' : ''}>
                                <summary class="cursor-pointer text-zinc-400 font-semibold hover:text-emerald-400 select-none flex items-center justify-between" dir="rtl">
                                    <span class="flex items-center gap-2">
                                        <span>💭</span> <span>تحلیل و فرآیند اندیشیدن مدل (کلیک برای باز/بستن)</span>
                                    </span>
                                    <span class="text-[10px] bg-zinc-700/60 text-zinc-400 px-2 py-0.5 rounded">${isThinkModeOn ? 'تفکر عمیق' : 'خلاصه'}</span>
                                </summary>
                                <div class="think-content mt-2 text-zinc-300 whitespace-pre-wrap leading-relaxed border-r-2 border-emerald-500/50 pr-3 text-[11px] max-h-72 overflow-y-auto" style="direction: auto; text-align: start; unicode-bidi: plaintext;">${escapeHtml(thinkPart)}</div>
                            </details>
                        `;
                    }
                    html += marked.parse(answerPart);
                    return html;
                } else if (text.startsWith('<think>')) {
                    const raw = text.replace('<think>', '').trim();
                    return `
                        <div class="think-container mb-4 rounded-xl bg-zinc-800/40 border border-emerald-500/30 p-3 text-xs" dir="ltr">
                            <div class="text-emerald-400 font-semibold flex items-center justify-between mb-1.5 animate-pulse" dir="rtl">
                                <span class="flex items-center gap-2"><span>🧠</span> <span>در حال اندیشیدن و طراحی دقیق الگوریتم...</span></span>
                                <span class="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded">Deep Analysis</span>
                            </div>
                            <div class="think-content text-zinc-300 whitespace-pre-wrap leading-relaxed border-r-2 border-emerald-500/30 pr-3 text-[11px] max-h-72 overflow-y-auto" style="direction: auto; text-align: start; unicode-bidi: plaintext;">${escapeHtml(raw)}</div>
                        </div>
                    `;
                }

                // 2. Monologue detection (when model reasons in English without tags)
                const monologueRegex = /^(?:Thinking Process:|Analysis:|We need|Need to|Need debug|Let's |First, let's|I need to|To solve this|The user wants|The user is asking)[\\s\\S]*?(?=\\n\\n(?:[^\\x00-\\x7F]|```|###|##|#|با توجه|برای|در ادامه|کد|پاسخ))/i;
                const match = text.match(monologueRegex);
                if (match) {
                    const thinkPart = match[0].trim();
                    const answerPart = text.slice(match[0].length).trim();
                    let html = `
                        <details class="think-container mb-4 rounded-xl bg-zinc-800/40 border border-zinc-700/60 p-3 text-xs" dir="ltr" ${isThinkModeOn ? 'open' : ''}>
                            <summary class="cursor-pointer text-zinc-400 font-semibold hover:text-emerald-400 select-none flex items-center justify-between" dir="rtl">
                                <span class="flex items-center gap-2">
                                    <span>💭</span> <span>تحلیل اولیه و فرآیند اندیشیدن</span>
                                </span>
                                <span class="text-[10px] bg-zinc-700/60 text-zinc-400 px-2 py-0.5 rounded">Deep Reasoning</span>
                            </summary>
                            <div class="think-content mt-2 text-zinc-300 whitespace-pre-wrap leading-relaxed border-r-2 border-emerald-500/50 pr-3 text-[11px] max-h-72 overflow-y-auto" style="direction: ltr; text-align: left; unicode-bidi: isolate;">${escapeHtml(thinkPart)}</div>
                        </details>
                    `;
                    if (answerPart) {
                        html += marked.parse(answerPart);
                    }
                    return html;
                }

                return marked.parse(text);
            }
            catch (e) {
                const escaped = (content || "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
                return "<pre class='p-3 bg-zinc-900 rounded overflow-x-auto text-sm' dir='ltr'>" + escaped + "</pre>";
            }
        }

        function copyCodeBlock(codeId, btn) {
            const rawCode = window[codeId] || "";
            if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                navigator.clipboard.writeText(rawCode).then(() => showCopied(btn)).catch(() => fallbackCopy(rawCode, btn));
            } else { fallbackCopy(rawCode, btn); }
        }

        function fallbackCopy(text, btn) {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.left = '-999999px';
            document.body.appendChild(textarea);
            textarea.select();
            try { document.execCommand('copy'); showCopied(btn); } catch (e) {}
            document.body.removeChild(textarea);
        }

        function showCopied(btn) {
            const originalHTML = btn.innerHTML;
            btn.classList.add('copied');
            btn.innerHTML = `<span>✓ کپی شد</span>`;
            setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = originalHTML; }, 2000);
        }

        function copyBaseUrl() {
            const url = window.location.origin + '/v1';
            if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
                navigator.clipboard.writeText(url).then(() => alert("Base URL کپی شد:\\n" + url));
            } else { prompt("Base URL را کپی کنید:", url); }
        }

        function initScrollListener() {
            const container = document.getElementById('messagesScrollWrapper');
            if (!container) return;
            container.addEventListener('scroll', () => {
                const distanceFromBottom = container.scrollHeight - container.scrollTop - container.clientHeight;
                userScrolledUp = distanceFromBottom > 100;
                const btn = document.getElementById('scrollBottomBtn');
                if (btn) {
                    btn.classList.toggle('hidden', !userScrolledUp);
                }
            });
        }

        function scrollToBottomSmooth() {
            userScrolledUp = false;
            const container = document.getElementById('messagesScrollWrapper');
            if (container) {
                container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
            }
            const btn = document.getElementById('scrollBottomBtn');
            if (btn) btn.classList.add('hidden');
        }

        async function init() {
            initScrollListener();
            const thinkToggle = document.getElementById('thinkModeToggle');
            if (thinkToggle) {
                thinkToggle.addEventListener('change', (e) => {
                    const isOpen = e.target.checked;
                    document.querySelectorAll('#messagesContainer details').forEach(d => {
                        d.open = isOpen;
                    });
                });
            }
            await fetchChats();
            const res = await fetch('/api/chats');
            const chats = await res.json();
            if (chats.length > 0) {
                switchChat(chats[0].id);
            } else {
                createNewChat();
            }
        }

        async function fetchChats() {
            try {
                const res = await fetch('/api/chats');
                const chats = await res.json();
                const listEl = document.getElementById('chatList');
                listEl.innerHTML = '';
                chats.forEach(chat => {
                    const btn = document.createElement('div');
                    btn.className = `group flex items-center justify-between p-2 rounded-lg cursor-pointer text-xs font-medium transition ${chat.id === activeChatId ? 'bg-zinc-800 text-white' : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'}`;
                    btn.onclick = () => switchChat(chat.id);
                    btn.innerHTML = `
                        <div class="flex items-center gap-2 truncate flex-1">
                            <span>💬</span>
                            <span class="truncate">${escapeHtml(chat.title || 'گفتگوی جدید')}</span>
                        </div>
                        <button onclick="event.stopPropagation(); deleteChat('${chat.id}')" class="opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                        </button>
                    `;
                    listEl.appendChild(btn);
                });
            } catch (e) {}
        }

        async function createNewChat() {
            try {
                const res = await fetch('/api/chats', { method: 'POST' });
                const chat = await res.json();
                await fetchChats();
                switchChat(chat.id);
            } catch (e) {}
        }

        async function switchChat(chatId) {
            if (activeAbortController) {
                activeAbortController.abort();
                activeAbortController = null;
            }
            activeChatId = chatId;
            await fetchChats();
            const container = document.getElementById('messagesContainer');
            container.innerHTML = '';

            try {
                const res = await fetch('/api/chats/' + chatId);
                const data = await res.json();

                if (!data.messages || data.messages.length === 0) {
                    container.innerHTML = `
                        <div class="h-full flex flex-col items-center justify-center text-center text-zinc-500 space-y-3 pt-24">
                            <div class="w-16 h-16 rounded-2xl bg-zinc-800/80 border border-zinc-700 flex items-center justify-center text-2xl text-emerald-400">⚡</div>
                            <h2 class="text-xl font-bold text-zinc-200">کدنویسی بدون توهم با Qwen 27B</h2>
                            <p class="max-w-md text-sm text-zinc-400">یک مسئله، پروژه، الگوریتم یا باگ پیچیده را بپرسید تا با نهایت دقت و سرعت برایتان حل کند.</p>
                        </div>`;
                } else {
                    data.messages.forEach(m => appendMessageUI(m.role, m.content, false));
                }

                if (data.is_generating) {
                    setGeneratingState(true);
                    const botMsgDiv = appendMessageUI("assistant", data.current_text || "", true);
                    const contentDiv = botMsgDiv.querySelector('.bot-content');
                    listenToLiveStream(chatId, contentDiv, data.current_text || "");
                } else {
                    setGeneratingState(false);
                }
            } catch (e) {}
        }

        async function deleteChat(chatId) {
            try {
                await fetch('/api/chats/' + chatId, { method: 'DELETE' });
                await fetchChats();
                const res = await fetch('/api/chats');
                const chats = await res.json();
                if (chats.length > 0) {
                    switchChat(chats[0].id);
                } else {
                    createNewChat();
                }
            } catch (e) {}
        }

        function setGeneratingState(isGen) {
            isCurrentlyGenerating = isGen;
            document.getElementById('sendBtn').classList.toggle('hidden', isGen);
            document.getElementById('stopBtn').classList.toggle('hidden', !isGen);
        }

        function appendMessageUI(role, content, isLive) {
            const container = document.getElementById('messagesContainer');
            if (container.querySelector('.text-xl')) container.innerHTML = '';

            const isUser = role === 'user';
            const msgDiv = document.createElement('div');
            msgDiv.className = `flex gap-4 p-4 rounded-xl ${isUser ? 'user-msg max-w-3xl mr-auto ml-4' : 'bot-msg max-w-4xl w-full'}`;

            if (isUser) {
                const formattedUser = formatUserText(content);
                msgDiv.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center flex-shrink-0 text-xs select-none">👤</div>
                    <div class="flex-1 text-sm pt-1 leading-relaxed user-content">${formattedUser}</div>
                `;
            } else {
                msgDiv.innerHTML = `
                    <div class="w-8 h-8 rounded-full bg-emerald-600/30 text-emerald-400 border border-emerald-500/30 flex items-center justify-center flex-shrink-0 text-sm font-bold select-none">Q</div>
                    <div class="flex-1 overflow-hidden pt-0.5">
                        <div class="bot-content markdown-body text-sm leading-relaxed">${safeMarkdownParse(content)}</div>
                    </div>
                `;
            }

            container.appendChild(msgDiv);
            if (!userScrolledUp) {
                const scrollWrapper = document.getElementById('messagesScrollWrapper');
                if (scrollWrapper) scrollWrapper.scrollTop = scrollWrapper.scrollHeight;
            }
            return msgDiv;
        }

        async function sendMessage() {
            if (isCurrentlyGenerating) return;
            const input = document.getElementById('userInput');
            const text = input.value.trim();
            if (!text || !activeChatId) return;

            input.value = '';
            input.style.height = 'auto';

            appendMessageUI("user", text, true);
            setGeneratingState(true);

            const botMsgDiv = appendMessageUI("assistant", "", true);
            const contentDiv = botMsgDiv.querySelector('.bot-content');
            contentDiv.innerHTML = '<div class="text-zinc-500 text-xs animate-pulse">در حال اندیشیدن و تولید کد دقیق...</div>';

            const thinkMode = document.getElementById('thinkModeToggle') ? document.getElementById('thinkModeToggle').checked : false;

            try {
                const res = await fetch('/api/chats/' + activeChatId + '/send', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content: text, think_mode: thinkMode })
                });
                if (!res.ok) throw new Error("Server error");
                listenToLiveStream(activeChatId, contentDiv, "");
            } catch (e) {
                contentDiv.innerHTML = `<span class="text-red-400">خطا: ${e.message}</span>`;
                setGeneratingState(false);
            }
        }

        async function listenToLiveStream(chatId, contentDiv, initialText) {
            if (activeAbortController) {
                activeAbortController.abort();
            }
            activeAbortController = new AbortController();
            let accumulated = initialText || "";

            try {
                const response = await fetch('/api/chats/' + chatId + '/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    signal: activeAbortController.signal
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (!trimmed || trimmed.startsWith(':')) continue;

                        if (trimmed === 'data: [DONE]') {
                            setGeneratingState(false);
                            contentDiv.innerHTML = safeMarkdownParse(accumulated.trim());
                            fetchChats();
                            return;
                        }

                        if (trimmed.startsWith('data: ')) {
                            try {
                                const parsed = JSON.parse(trimmed.slice(6));
                                if (parsed.full_text && parsed.full_text.length >= accumulated.length) {
                                    accumulated = parsed.full_text;
                                    
                                    // Preserve thinking box scroll position if user is reading inside it
                                    const oldThink = contentDiv.querySelector('.think-content');
                                    const oldThinkScroll = oldThink ? oldThink.scrollTop : null;
                                    const wasThinkAtBottom = oldThink ? (oldThink.scrollHeight - oldThink.scrollTop - oldThink.clientHeight < 40) : true;

                                    contentDiv.innerHTML = safeMarkdownParse(accumulated);

                                    const newThink = contentDiv.querySelector('.think-content');
                                    if (newThink && oldThinkScroll !== null) {
                                        newThink.scrollTop = wasThinkAtBottom ? newThink.scrollHeight : oldThinkScroll;
                                    }

                                    // Auto-scroll main window ONLY if user hasn't scrolled up
                                    if (!userScrolledUp) {
                                        const scrollWrapper = document.getElementById('messagesScrollWrapper');
                                        if (scrollWrapper) scrollWrapper.scrollTop = scrollWrapper.scrollHeight;
                                    }
                                }
                                if (parsed.is_generating === false) {
                                    setGeneratingState(false);
                                    contentDiv.innerHTML = safeMarkdownParse(accumulated.trim());
                                    fetchChats();
                                    return;
                                }
                            } catch (e) {}
                        }
                    }
                }
            } catch (e) {
                if (e.name !== 'AbortError') {
                    console.error("Stream reader error:", e);
                }
            } finally {
                setGeneratingState(false);
                if (accumulated) {
                    contentDiv.innerHTML = safeMarkdownParse(accumulated.trim());
                }
            }
        }

        async function stopCurrentGeneration() {
            if (!activeChatId) return;
            try {
                await fetch('/api/chats/' + activeChatId + '/stop', { method: 'POST' });
            } catch (e) {}
            if (activeAbortController) {
                activeAbortController.abort();
                activeAbortController = null;
            }
            setGeneratingState(false);
        }

        function handleKeyDown(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        }

        window.onload = init;
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

# 7. Start Background Web Server and Cloudflare Tunnel
def run_uvicorn():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

def start_server():
    server_thread = Thread(target=run_uvicorn, daemon=True)
    server_thread.start()
    time.sleep(2)

    print("\nStarting Cloudflare Tunnel on port 8000...", flush=True)
    tunnel_cmd = [
        CLOUDFLARED_BIN, "tunnel",
        "--url", "http://127.0.0.1:8000",
        "--metrics", "127.0.0.1:45678",
        "--no-autoupdate"
    ]
    tunnel_proc = subprocess.Popen(
        tunnel_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    public_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    for _ in range(60):
        line = tunnel_proc.stdout.readline()
        if not line:
            time.sleep(0.5)
            continue
        match = url_pattern.search(line)
        if match:
            public_url = match.group(0)
            break

    if public_url:
        print("\n" + "=" * 60, flush=True)
        print(f"🚀 QWEN 27B Q5_K SERVER LIVE AT: {public_url}", flush=True)
        print(f"📖 Web UI: {public_url}", flush=True)
        print(f"⚡ OpenAI Compatible API: {public_url}/v1", flush=True)
        print("=" * 60 + "\n", flush=True)
        with open("/tmp/my_qwen_url.txt", "w") as f:
            f.write(public_url)
    else:
        print("Could not retrieve Cloudflare Tunnel URL automatically. Check cloudflared logs.")

    while True:
        time.sleep(3600)

if __name__ == "__main__":
    start_server()
