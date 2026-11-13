"""АРКАНУМ — локальный сервер с Yandex AI Studio.

Нужны переменные окружения YANDEX_API_KEY и YANDEX_FOLDER_ID.
Рядом с файлом должен лежать index.html.
"""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import BoundedSemaphore, Lock
from datetime import date
from pathlib import Path
from urllib import request, error

BASE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
HOST = os.environ.get("HOST", "0.0.0.0")
DAILY_LIMIT = int(os.environ.get("DAILY_LIMIT", "100"))
quota_lock = Lock()
quota_day = date.today()
quota_used = 0
slots = BoundedSemaphore(2)
MODEL_NAME = os.environ.get("YANDEX_MODEL", "yandexgpt/latest")
URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
TOPICS = {"Любовь и отношения", "Работа и финансы", "Судьба и жизненный путь", "Свой вопрос"}
FIELDS = ("overview", "connection", "advice")

SYSTEM = (
    "Ты — таролог приложения «АРКАНУМ». Пиши по-русски, выразительно, атмосферно, "
    "но конкретно и без канцелярита. Интерпретируй три карты как ЕДИНЫЙ ответ, "
    "не назначая им позиции прошлого, настоящего и будущего. Учитывай вопрос, "
    "тему, конкретные карты, масти, прямое или перевёрнутое положение и их взаимодействие. "
    "Сначала назови возможный характер ситуации или события, затем объясни связь карт. "
    "Не придумывай биографию пользователя и не подгоняй ответ под неуказанный контекст. "
    "Не выдавай мысли третьих лиц за факты, не ставь диагнозы, не давай финансовых "
    "или медицинских рекомендаций. Никаких запугиваний и гарантированных предсказаний. "
    "Верни ТОЛЬКО корректный JSON-объект с тремя строковыми полями: "
    "overview (общее толкование, примерно 100–150 слов), "
    "connection (как взаимодействуют три карты, примерно 130–200 слов), "
    "advice (практический итог, примерно 60–100 слов). "
    "Никаких markdown-блоков или текста вне JSON."
)


def parse_reading(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    try:
        content = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object in AI response")
        content = json.loads(raw[start:end + 1])
    if not isinstance(content, dict) or not all(isinstance(content.get(k), str) for k in FIELDS):
        raise ValueError("AI response is missing expected fields")
    return {k: content[k] for k in FIELDS}


class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, kind="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content if isinstance(content, bytes) else content.encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            return self.send(200, '{"status":"ok"}')
        if self.path in ("/", "/index.html"):
            page = BASE / "index.html"
            if not page.is_file():
                return self.send(404, '{"error":"index.html not found next to server"}')
            return self.send(200, page.read_bytes(), "text/html; charset=utf-8")
        return self.send(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path != "/api/reading":
            return self.send(404, '{"error":"not found"}')
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 4096:
                raise ValueError("Invalid request length")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Invalid request")
            topic, question, cards = data["topic"], data["question"], data["cards"]
            if topic not in TOPICS or not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
                raise ValueError("Invalid topic or question")
            if not isinstance(cards, list) or len(cards) != 3:
                raise ValueError("Exactly three cards required")
            if any(not isinstance(c, dict) or not isinstance(c.get("name"), str)
                   or not 1 <= len(c["name"].strip()) <= 40
                   or type(c.get("reversed")) is not bool for c in cards):
                raise ValueError("Invalid cards")
            if len({c["name"] for c in cards}) != 3:
                raise ValueError("Cards must be unique")
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return self.send(400, json.dumps({"error": str(exc)}, ensure_ascii=False))

        global quota_day, quota_used
        with quota_lock:
            today = date.today()
            if today != quota_day:
                quota_day, quota_used = today, 0
            if quota_used >= DAILY_LIMIT:
                return self.send(429, '{"error":"Daily AI request limit reached"}')
            quota_used += 1  # Reserve a request even if the upstream call fails.
        if not slots.acquire(blocking=False):
            return self.send(429, '{"error":"Server busy, try again shortly"}')
        try:
            return self.call_ai(data)
        finally:
            slots.release()

    def call_ai(self, data):
        key = os.environ.get("YANDEX_API_KEY")
        folder_id = os.environ.get("YANDEX_FOLDER_ID")
        if not key or not folder_id:
            return self.send(503, json.dumps({"error": "Set YANDEX_API_KEY and YANDEX_FOLDER_ID"}))

        model_uri = MODEL_NAME if MODEL_NAME.startswith("gpt://") else f"gpt://{folder_id}/{MODEL_NAME}"
        payload = {
            "modelUri": model_uri,
            "completionOptions": {"stream": False, "temperature": 0.7, "maxTokens": "1800"},
            "messages": [
                {"role": "system", "text": SYSTEM},
                {"role": "user", "text": json.dumps(data, ensure_ascii=False)},
            ],
        }
        req = request.Request(
            URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Api-Key {key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=90) as resp:
                answer = json.load(resp)
            raw = answer["result"]["alternatives"][0]["message"]["text"]
            content = parse_reading(raw)
            return self.send(200, json.dumps(content, ensure_ascii=False))
        except error.HTTPError as exc:
            print("Yandex AI HTTP error:", exc.code)
            print("Yandex AI details:", exc.read().decode("utf-8", errors="replace"))
            return self.send(502, '{"error":"Yandex AI service request failed"}')
        except (error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            print("Yandex AI response error:", type(exc).__name__, str(exc)[:300])
            return self.send(502, '{"error":"AI request failed"}')


if __name__ == "__main__":
    print(f"АРКАНУМ: http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
