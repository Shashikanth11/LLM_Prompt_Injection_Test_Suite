import requests
import time
from config import CHAT_URL

def chat_ollama(model: str, system: str, user: str, temperature: float = 0.2, timeout: int = 120):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"temperature": temperature},
    }
    start = time.time()
    r = requests.post(CHAT_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    content = data.get("message", {}).get("content", "")
    return {"text": content, "elapsed_ms": int((time.time()-start)*1000), "raw": data}
