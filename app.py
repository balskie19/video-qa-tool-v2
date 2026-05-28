import os
import json
import asyncio
import time
import hmac
import hashlib
import urllib.request
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RUNS_LOG = LOGS_DIR / "runs.jsonl"


def write_run_log(source: str, report: dict, duration_s: float, error: str = None):
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "duration_s": round(duration_s, 1),
        "issue_count": report.get("issue_count", 0) if report else 0,
        "summary": report.get("summary", "") if report else "",
        "issues": report.get("issues", []) if report else [],
        "error": error,
    }
    with open(RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def _send_webhook(payload: dict) -> None:
    """Fire n8n webhook — non-blocking, never raises. Failure is logged, not surfaced."""
    url = os.getenv("N8N_WEBHOOK_URL", "").strip()
    if not url:
        return
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=10)
        )
    except Exception as exc:
        print(f"[webhook] send failed: {exc}", flush=True)

load_dotenv()

# ── Auth ──────────────────────────────────────────────────────────────────────
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
_SECRET = os.getenv("SECRET_KEY") or hashlib.sha256(
    (APP_PASSWORD + "video-qa-cab-2026").encode()
).hexdigest()
_COOKIE = "qa_session"
_COOKIE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days


def _make_token() -> str:
    return hmac.new(_SECRET.encode(), b"authenticated", hashlib.sha256).hexdigest()


def _is_authenticated(request: Request) -> bool:
    if not APP_PASSWORD:
        return True  # no password configured → open access
    token = request.cookies.get(_COOKIE)
    if not token:
        return False
    return hmac.compare_digest(token, _make_token())


# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Video QA Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/login")
async def login_page():
    return FileResponse("static/login.html")


@app.post("/login")
async def login_submit(password: str = Form(...)):
    if APP_PASSWORD and password == APP_PASSWORD:
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(
            _COOKIE,
            _make_token(),
            max_age=_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
        )
        return resp
    return RedirectResponse(url="/login?error=1", status_code=303)


@app.get("/logout")
async def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie(_COOKIE)
    return resp


@app.get("/")
async def root(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/analyze")
async def analyze_video(
    request: Request,
    url: str = Form(None),
    file: UploadFile = File(None),
    context: str = Form(None),
):
    if not _is_authenticated(request):
        return Response(status_code=401, content="Unauthorized")

    source = (file.filename if file else None) or url or "unknown"

    async def generate():
        start = time.monotonic()
        last_report = None
        last_error = None
        try:
            from qa_engine import run_qa_analysis
            async for update in run_qa_analysis(url=url, file=file, context=context):
                if update.get("type") == "complete":
                    last_report = update.get("report")
                elif update.get("type") == "error":
                    last_error = update.get("message")
                yield f"data: {json.dumps(update)}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            import traceback
            last_error = str(e)
            tb = traceback.format_exc()
            print(f"[ERROR] {last_error}\n{tb}", flush=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'details': tb})}\n\n"
        finally:
            duration_s = round(time.monotonic() - start, 1)
            write_run_log(source, last_report, duration_s, last_error)
            # Fire n8n webhook — non-blocking
            asyncio.create_task(_send_webhook({
                "filename":   last_report.get("filename", source) if last_report else source,
                "source":     source,
                "timestamp":  datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "duration_s": duration_s,
                "issue_count": last_report.get("issue_count", 0) if last_report else 0,
                "summary":    last_report.get("summary", "") if last_report else "",
                "issues":     last_report.get("issues", []) if last_report else [],
                "had_script": bool(context and context.strip()),
                "status":     "error" if last_error else "complete",
                "error":      last_error or None,
            }))
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
