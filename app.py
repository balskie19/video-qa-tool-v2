import os
import json
import asyncio
import time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
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

load_dotenv()

app = FastAPI(title="Video QA Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.post("/analyze")
async def analyze_video(
    url: str = Form(None),
    file: UploadFile = File(None),
    context: str = Form(None),
):
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
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'details': traceback.format_exc()})}\n\n"
        finally:
            write_run_log(source, last_report, time.monotonic() - start, last_error)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
