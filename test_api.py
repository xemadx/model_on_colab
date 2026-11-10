import requests
import json
import time
import http.client
from urllib.parse import urlparse

def run_tests(base_url):
    print("=" * 60)
    print(f"RUNNING VERIFICATION AGAINST: {base_url}")
    print("=" * 60)

    # 1. Health check
    print("\n[1] Testing /health endpoint...")
    r = requests.get(f"{base_url}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    health_data = r.json()
    print("Health response:", health_data)
    assert health_data.get("gpu") is True, "GPU is not active!"

    # 2. Web UI HTML check
    print("\n[2] Testing Web UI HTML / endpoint...")
    r = requests.get(base_url, timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    html = r.text
    assert "startStreamForMessage" in html, "HTML missing startStreamForMessage"
    assert "getReader" in html, "HTML missing ReadableStream getReader"
    print("Web UI HTML successfully verified (contains modern ReadableStream POST streaming).")

    # 3. OpenAI Models
    print("\n[3] Testing /v1/models endpoint...")
    r = requests.get(f"{base_url}/v1/models", timeout=10)
    assert r.status_code == 200
    models_data = r.json()
    model_ids = [m["id"] for m in models_data.get("data", [])]
    print("Available models:", model_ids)
    assert "deepseek-v4" in model_ids

    # 4. Chat Creation
    print("\n[4] Testing /api/chats creation...")
    r = requests.post(f"{base_url}/api/chats", timeout=10)
    assert r.status_code == 200
    chat = r.json()
    chat_id = chat["id"]
    print(f"Created chat: {chat_id}")

    # 5. Live Direct POST Streaming Test
    print("\n[5] Testing Direct POST Streaming (/api/chats/{chat_id}/send with stream: true)...")
    prompt = "سلام! لطفا در سه جمله کوتاه توضیح بده هوش مصنوعی چیست؟"
    parsed = urlparse(base_url)
    conn = http.client.HTTPSConnection(parsed.hostname, 443)
    post_body = json.dumps({"content": prompt, "stream": True}).encode("utf-8")
    
    t_start = time.time()
    conn.request(
        "POST",
        f"/api/chats/{chat_id}/send",
        body=post_body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"}
    )
    resp = conn.getresponse()
    assert resp.status == 200, f"Expected 200, got {resp.status}"

    chunks_received = 0
    first_chunk_latency = None
    accumulated_text = ""
    timestamps = []

    while True:
        line = resp.readline()
        if not line:
            break
        decoded = line.decode("utf-8", errors="replace")
        if not decoded.strip():
            continue
        
        elapsed = time.time() - t_start
        if first_chunk_latency is None:
            first_chunk_latency = elapsed
            print(f"--> FIRST CHUNK ARRIVED AT: {first_chunk_latency:.3f}s (No Cloudflare buffering!)")

        if decoded.startswith(":"):
            continue
        if decoded.strip() == "data: [DONE]":
            print(f"--> [DONE] received at {elapsed:.3f}s")
            break
        if decoded.startswith("data: "):
            chunks_received += 1
            timestamps.append(elapsed)
            payload = json.loads(decoded[6:])
            token = payload.get("token", "")
            accumulated_text += token
            if chunks_received <= 5 or chunks_received % 10 == 0:
                print(f"[{elapsed:.2f}s] Chunk {chunks_received:3d}: {token!r}")

    total_time = time.time() - t_start
    print(f"\nStreaming Results:")
    print(f"- Total chunks: {chunks_received}")
    print(f"- First chunk latency: {first_chunk_latency:.3f}s")
    print(f"- Total duration: {total_time:.2f}s")
    print(f"- Final accumulated text ({len(accumulated_text)} chars):\n{accumulated_text}\n")

    assert first_chunk_latency is not None and first_chunk_latency < 3.0, f"First chunk took too long: {first_chunk_latency}s"
    assert chunks_received >= 5, f"Expected >= 5 streaming chunks, got {chunks_received}"
    assert len(accumulated_text) > 20, "Accumulated text is too short!"

    # 6. Verify SQLite persistence
    print("\n[6] Verifying SQLite persistence for completed chat...")
    r = requests.get(f"{base_url}/api/chats/{chat_id}", timeout=10)
    assert r.status_code == 200
    chat_details = r.json()
    assert len(chat_details["messages"]) == 2, f"Expected 2 messages (user+assistant), got {len(chat_details['messages'])}"
    assert chat_details["messages"][0]["role"] == "user"
    assert chat_details["messages"][1]["role"] == "assistant"
    assert chat_details["is_generating"] is False
    print("Chat correctly persisted in SQLite and status is is_generating=False.")

    # 7. OpenAI chat/completions streaming
    print("\n[7] Testing /v1/chat/completions (OpenAI Compatible Streaming)...")
    t0 = time.time()
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": "deepseek-v4",
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "stream": True
        },
        stream=True,
        timeout=30
    )
    assert r.status_code == 200
    openai_chunks = 0
    openai_first = None
    for chunk in r.iter_lines():
        if chunk:
            decoded = chunk.decode("utf-8")
            if decoded.startswith("data: ") and decoded != "data: [DONE]":
                openai_chunks += 1
                if openai_first is None:
                    openai_first = time.time() - t0
    print(f"OpenAI Stream: {openai_chunks} chunks, first at {openai_first:.3f}s")
    assert openai_chunks >= 3

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 60)

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else None
    if not url:
        with open("./remote_public_url.txt") as f:
            url = f.read().strip()
    run_tests(url)
