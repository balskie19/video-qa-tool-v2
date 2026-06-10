import os
import re
import json
import base64
import asyncio
import subprocess
import tempfile
import shutil
from typing import AsyncGenerator, Optional

import whisper
import yt_dlp
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Windows: set Tesseract path if not in system PATH
try:
    import pytesseract as _pt
    import shutil as _shutil
    if not _shutil.which("tesseract"):
        _win_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(_win_path):
            _pt.pytesseract.tesseract_cmd = _win_path
except Exception:
    pass

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "anthropic/claude-3.5-sonnet")
CAPTION_READ_MODEL = os.getenv("CAPTION_READ_MODEL", "anthropic/claude-3-haiku")  # fast model for OCR caption reading
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
) if GROQ_API_KEY else None

_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


# ─────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────

async def run_qa_analysis(url: Optional[str] = None, file=None, context: Optional[str] = None) -> AsyncGenerator:
    tmp_dir = tempfile.mkdtemp(prefix="qa_")
    try:
        loop = asyncio.get_event_loop()

        # ── Step 1: Download / receive video ──
        yield {"type": "progress", "step": "Preparing video...", "percent": 5}
        video_path, video_filename = await loop.run_in_executor(None, lambda: get_video(url, file, tmp_dir))

        yield {"type": "progress", "step": "Reading video metadata...", "percent": 12}
        video_info = get_video_info(video_path)

        # ── Step 2: Frames + audio extraction in parallel ──
        yield {"type": "progress", "step": "Extracting frames and audio...", "percent": 18}
        (frames, fps_val), audio_path = await asyncio.gather(
            loop.run_in_executor(None, lambda: extract_frames(video_path, tmp_dir, video_info["duration"])),
            loop.run_in_executor(None, lambda: extract_audio(video_path, tmp_dir)),
        )

        # ── Step 3: Audio analysis + transcription + caption extraction + endcard audio + scene cuts all in parallel ──
        yield {"type": "progress", "step": "Analyzing audio, transcribing, reading captions...", "percent": 30}
        _dur = video_info["duration"]
        results = await asyncio.gather(
            loop.run_in_executor(None, lambda: analyze_audio(audio_path)),
            loop.run_in_executor(None, lambda: transcribe_audio(audio_path)),
            loop.run_in_executor(None, lambda: extract_captions(video_path, tmp_dir)),
            loop.run_in_executor(None, lambda: analyze_audio_levels(audio_path, _dur)),
            loop.run_in_executor(None, lambda: detect_scene_cuts(video_path)),
            return_exceptions=True,
        )
        audio_data   = results[0] if not isinstance(results[0], Exception) else {"dead_air": [], "music_gaps": [], "crackling_flag": False, "inconsistency_flag": False}
        transcript   = results[1] if not isinstance(results[1], Exception) else {"full_text": "", "segments": []}
        captions     = results[2] if not isinstance(results[2], Exception) else {"found": False, "entries": []}
        audio_levels = results[3] if not isinstance(results[3], Exception) else None
        scene_cuts   = results[4] if not isinstance(results[4], Exception) else []

        # ── Step 4: Caption split detection ──
        yield {"type": "progress", "step": "Checking captions...", "percent": 58}
        split_errors = await loop.run_in_executor(None, lambda: detect_caption_split_errors(captions))

        # Detect sign-based UGC ads early — no VO + no embedded SRT
        # Sign-based videos get the OCR pass skipped: the bottom strips would contain sign text,
        # not real captions, so running OCR would pollute split_errors with false data.
        transcript_words = len((transcript.get("full_text") or "").split())
        is_sign_based = transcript_words <= 5 and not captions.get("found")

        if not captions.get("found") and not is_sign_based:
            # No embedded SRT — run OCR-based burned-in caption reading
            caption_strips = await loop.run_in_executor(
                None, lambda: extract_caption_strips(tmp_dir, video_info["duration"], fps_val)
            )
            if caption_strips:
                yield {"type": "progress", "step": "Reading burned-in captions...", "percent": 65}
                ocr_captions = await read_captions_from_frames_ai(caption_strips)
                ocr_splits = detect_splits_from_caption_list(ocr_captions)
                if ocr_splits:
                    split_errors = ocr_splits

        # ── Step 5: AI analysis ──
        yield {"type": "progress", "step": "Running AI analysis...", "percent": 74}

        # Inject sign-based context note so the visual agent focuses on sign readability + branding
        if is_sign_based:
            sign_note = (
                "NOTE: This appears to be a SIGN-BASED UGC AD — the talent holds physical signs with handwritten text. "
                "Almost no spoken VO was detected. Your visual analysis MUST specifically check: "
                "(1) Is the text on every sign fully readable in every frame it appears? "
                "If a dark overlay, animation, or transition covers any part of the sign text — flag it. "
                "(2) Is there a client logo, branded graphic, or branded endcard screen anywhere in the video? "
                "Handwritten text on a physical prop does NOT count as branding. If no brand identifier exists — flag it."
            )
            auto_context = sign_note if not context else f"{context}\n\n{sign_note}"
        else:
            auto_context = context

        report = await analyze_with_ai(frames, audio_data, transcript, captions, video_info, split_errors, context=auto_context, audio_levels=audio_levels, scene_cuts=scene_cuts)
        report["filename"] = video_filename
        report["duration"] = round(video_info["duration"])

        yield {"type": "complete", "report": report}

    except Exception as e:
        import traceback
        yield {"type": "error", "message": str(e), "details": traceback.format_exc()}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────
# VIDEO ACQUISITION
# ─────────────────────────────────────────────────────────

def get_video(url: Optional[str], file, tmp_dir: str) -> tuple:
    """Returns (video_path, display_filename)."""
    if file is not None:
        out = os.path.join(tmp_dir, "upload.mp4")
        with open(out, "wb") as f:
            f.write(file.file.read())
        name = (file.filename or "").strip() or "upload.mp4"
        return out, name
    if url:
        path, name = download_video(url.strip(), tmp_dir)
        return path, name
    raise ValueError("No video source provided.")


def download_video(url: str, tmp_dir: str) -> tuple:
    """Returns (video_path, display_filename)."""
    out_tmpl = os.path.join(tmp_dir, "video.%(ext)s")

    # Standard Dropbox share link
    if "dropbox.com" in url and "replay.dropbox.com" not in url:
        # Extract filename from URL path
        path_part = url.split("?")[0].rstrip("/")
        name = path_part.split("/")[-1] or "dropbox_video.mp4"
        if "." not in name:
            name += ".mp4"
        direct = re.sub(r"[?&]dl=\d", "", url)
        sep = "&" if "?" in direct else "?"
        direct = direct.replace("www.dropbox.com", "dl.dropboxusercontent.com") + sep + "dl=1"
        return _download_direct(direct, tmp_dir), name

    # Dropbox Replay link — try yt-dlp first (fast), fall back to Playwright
    if "replay.dropbox.com" in url:
        try:
            path = _download_yt_dlp(url, out_tmpl, tmp_dir)
            name = os.path.basename(path)
            return path, name
        except Exception:
            pass
        path, name = _download_replay_playwright(url, tmp_dir)
        return path, name

    # Any other URL (YouTube, Vimeo, direct mp4, etc.)
    path = _download_yt_dlp(url, out_tmpl, tmp_dir)
    return path, os.path.basename(path)


def _download_yt_dlp(url: str, out_tmpl: str, tmp_dir: str) -> str:
    opts = {
        "outtmpl": out_tmpl,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    for f in os.listdir(tmp_dir):
        if f.startswith("video") and not f.endswith(".part"):
            return os.path.join(tmp_dir, f)
    raise FileNotFoundError("yt-dlp: download finished but file not found.")


def _download_direct(url: str, tmp_dir: str) -> str:
    import requests
    r = requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    out = os.path.join(tmp_dir, "video_dl.mp4")
    with open(out, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    return out


def _download_replay_playwright(url: str, tmp_dir: str) -> tuple:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError(
            "Dropbox Replay links require Playwright.\n"
            "Run: pip install playwright && python -m playwright install chromium"
        )

    share_id = url.rstrip("/").split("/")[-1][:16]
    captured: list[tuple[str, str]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 720},
            locale="en-US",
            timezone_id="America/New_York",
        )
        # Hide webdriver flag
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = context.new_page()

        def _on_response(response):
            ru = response.url
            ct = response.headers.get("content-type", "").lower()
            if ("video" in ct or "mpegurl" in ct or ru.endswith(".m3u8") or "dropboxusercontent.com" in ru) \
                    and ru not in [c[0] for c in captured]:
                captured.append((ru, ct))

        page.on("response", _on_response)
        try:
            page.goto(url, wait_until="load", timeout=30_000)
            page.wait_for_timeout(1_500)
            # Try clicking play to trigger video URL load
            try:
                page.click("video", timeout=1_500)
            except Exception:
                try:
                    page.click("[aria-label*='play' i]", timeout=1_000)
                except Exception:
                    pass
            page.wait_for_timeout(2_000)
        except Exception:
            pass

        try:
            # Wait up to 8s for the title to contain the actual video name (not just "Dropbox Replay")
            try:
                page.wait_for_function(
                    "document.title && document.title.length > 20 && !document.title.startsWith('Dropbox')",
                    timeout=8_000
                )
            except Exception:
                pass
            title = page.title()
            # Strip "- Dropbox Replay" / "| Replay" suffix and " MP4" suffix from Dropbox titles
            clean = re.sub(r'\s*[-|]\s*(Dropbox\s*Replay|Replay).*$', '', title, flags=re.IGNORECASE).strip()
            clean = re.sub(r'\s+MP4\s*$', '', clean, flags=re.IGNORECASE).strip()
            # Treat "Dropbox Replay" or very short cleaned titles as failures
            if clean and len(clean) > 5 and "dropbox" not in clean.lower():
                display_name = clean + ".mp4"
            else:
                display_name = f"replay_{share_id}.mp4"
        except Exception as e:
            display_name = f"replay_{share_id}.mp4"

        browser.close()

    if not captured:
        raise RuntimeError(
            "Headless browser found no video URLs on the Replay page. "
            "The link may be private or expired. Try the standard dropbox.com/s/... link instead."
        )

    def _priority(item: tuple[str, str]) -> int:
        u, ct = item
        if any(x in ct for x in ["image/", "text/vtt", "video/mp2t"]):
            return 99
        if "hls_master_playlist" in u: return 0
        if "hls_playlist" in u: return 1
        if "mpegurl" in ct: return 2
        if "video" in ct: return 3
        return 10

    for video_url, ct in sorted(captured, key=_priority):
        if _priority((video_url, ct)) == 99:
            continue
        try:
            if "mpegurl" in ct or video_url.endswith(".m3u8"):
                out = os.path.join(tmp_dir, display_name)
                subprocess.run(
                    ["ffmpeg", "-i", video_url, "-c", "copy", out, "-y", "-loglevel", "quiet"],
                    check=True,
                )
                return out, display_name
            else:
                return _download_direct(video_url, tmp_dir), display_name
        except Exception:
            continue

    raise RuntimeError("Could not download a valid video from Dropbox Replay.")


# ─────────────────────────────────────────────────────────
# REPLAY COMMENT SCRAPING
# ─────────────────────────────────────────────────────────

def fetch_replay_comments(url: str) -> str:
    """
    Navigate to a Dropbox Replay URL with Playwright and return raw comment text
    scraped from the page. Caller passes this to AI for structuring.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise RuntimeError("Playwright not installed. Run: pip install playwright && python -m playwright install chromium")

    raw_text = ""

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        page = ctx.new_page()

        try:
            page.goto(url, wait_until="load", timeout=30_000)
            # Wait for comments panel to populate
            page.wait_for_timeout(5_000)

            # Try to get just the comments sidebar text — Dropbox Replay renders a right panel
            # Try common containers; fall back to full body text
            sidebar_text = None
            for selector in [
                "[class*='Comment']",
                "[class*='comment']",
                "[data-testid*='comment']",
                "[class*='Sidebar']",
                "[class*='sidebar']",
                "[class*='Panel']",
            ]:
                try:
                    els = page.query_selector_all(selector)
                    if els:
                        sidebar_text = "\n".join(el.inner_text() for el in els if el.inner_text().strip())
                        if sidebar_text.strip():
                            break
                except Exception:
                    continue

            raw_text = sidebar_text if sidebar_text and sidebar_text.strip() else page.inner_text("body")
        except Exception as e:
            raw_text = f"[PAGE LOAD ERROR: {e}]"
        finally:
            browser.close()

    return raw_text


def summarize_replay_comments(raw_text: str) -> dict:
    """
    Pass scraped Replay page text to AI. Returns {comments: [...], summary: str, action_items: [...]}.
    """
    ai_client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

    system = (
        "You are a video production assistant. You will receive raw text scraped from a Dropbox Replay video review page. "
        "Your job is to:\n"
        "1. Extract every comment left by reviewers — each comment has a timestamp (format like 0:01.802 or 0:30), "
        "   an author name, and comment text. Some comments are replies to others.\n"
        "2. Produce a clean numbered action-item list of what the editor needs to fix or decide on.\n\n"
        "Output a JSON object with these fields:\n"
        "  comments: array of {timestamp, author, text} — all comments found, in order\n"
        "  action_items: array of strings — numbered list of editor actions derived from the comments\n"
        "  summary: one sentence summarizing what kind of feedback was left\n\n"
        "If no comments are found, return comments=[], action_items=[], summary='No comments found.'\n"
        "Return only valid JSON."
    )

    try:
        resp = ai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"RAW PAGE TEXT:\n\n{raw_text[:8000]}"},
            ],
            max_tokens=1024,
            temperature=0,
        )
        content = resp.choices[0].message.content.strip()
        # Strip markdown fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception as e:
        return {"comments": [], "action_items": [], "summary": f"AI parse error: {e}"}


# ─────────────────────────────────────────────────────────
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    data = json.loads(r.stdout)
    duration = float(data["format"].get("duration", 0))
    w = h = 0
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            w, h = s.get("width", 0), s.get("height", 0)
            break
    return {"duration": duration, "width": w, "height": h}


def extract_frames(video_path: str, tmp_dir: str, duration: float) -> tuple:
    frames_dir = os.path.join(tmp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Frame rate based on duration — denser for short videos to catch caption changes
    if duration <= 60:
        fps_val = 2.0    # every 0.5s — catches fast caption transitions
    elif duration <= 120:
        fps_val = 1.0
    else:
        fps_val = 0.5

    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps_val}",
        "-q:v", "3",
        os.path.join(frames_dir, "frame_%05d.jpg"),
        "-y", "-loglevel", "quiet",
    ], check=True)

    # Ensure last frame is captured
    subprocess.run([
        "ffmpeg", "-i", video_path, "-sseof", "-0.5",
        "-vframes", "1",
        os.path.join(frames_dir, "frame_last.jpg"),
        "-y", "-loglevel", "quiet",
    ])

    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

    # Cap at 40 frames, evenly distributed
    if len(files) > 40:
        indices = [int(i * (len(files) - 1) / 39) for i in range(40)]
        files = [files[i] for i in indices]

    result = []
    for fname in files:
        fpath = os.path.join(frames_dir, fname)
        with open(fpath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        if fname == "frame_last.jpg":
            ts = duration
        else:
            try:
                num = int(re.search(r"frame_(\d+)", fname).group(1))
                ts = num / fps_val
            except Exception:
                ts = 0.0
        result.append({"ts": ts, "b64": b64})

    return result, fps_val


def extract_audio(video_path: str, tmp_dir: str) -> str:
    out = os.path.join(tmp_dir, "audio.wav")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        out, "-y", "-loglevel", "quiet",
    ], check=True)
    return out


def analyze_audio(audio_path: str) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    def _run(args):
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")

    # Run all 3 core passes in parallel
    with ThreadPoolExecutor(max_workers=3) as ex:
        vol_f   = ex.submit(_run, ["ffmpeg", "-i", audio_path, "-af", "volumedetect", "-f", "null", "-", "-loglevel", "info"])
        sil35_f = ex.submit(_run, ["ffmpeg", "-i", audio_path, "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-", "-loglevel", "info"])
        sil60_f = ex.submit(_run, ["ffmpeg", "-i", audio_path, "-af", "silencedetect=noise=-60dB:d=0.3", "-f", "null", "-", "-loglevel", "info"])
        vol_r  = vol_f.result()
        sil_r  = sil35_f.result()
        sil_r2 = sil60_f.result()

    mean_m = re.search(r"mean_volume: ([-\d.]+) dB", vol_r.stderr)
    max_m  = re.search(r"max_volume: ([-\d.]+) dB",  vol_r.stderr)
    mean_db = float(mean_m.group(1)) if mean_m else None
    max_db  = float(max_m.group(1))  if max_m  else None

    # Voice silence gaps
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", sil_r.stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)",   sil_r.stderr)]
    voice_silent = []
    for i, s in enumerate(starts):
        if i < len(ends):
            dur = round(ends[i] - s, 2)
            if dur >= 0.4:
                voice_silent.append({"start": round(s, 2), "end": round(ends[i], 2), "duration": dur})

    # Total silence gaps
    starts2 = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", sil_r2.stderr)]
    ends2   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)",   sil_r2.stderr)]
    totally_silent = []
    for i, s in enumerate(starts2):
        if i < len(ends2):
            dur = round(ends2[i] - s, 2)
            if dur >= 0.4:
                totally_silent.append({"start": round(s, 2), "end": round(ends2[i], 2), "duration": dur})

    def _overlaps(a, b, tol=0.5):
        return a["start"] <= b["end"] + tol and b["start"] <= a["end"] + tol

    dead_air, music_gap = [], []
    for gap in voice_silent:
        (dead_air if any(_overlaps(gap, ts) for ts in totally_silent) else music_gap).append(gap)

    # Loudnorm — only run if max_db suggests hot audio (avoids slow pass on clean files)
    lra = tp = None
    if max_db is not None and max_db > -3.0:
        ln_r = subprocess.run(
            ["ffmpeg", "-i", audio_path, "-af", "loudnorm=print_format=json",
             "-f", "null", "-", "-loglevel", "info"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        jm = re.search(r"\{[\s\S]+?\}", ln_r.stderr)
        if jm:
            try:
                d = json.loads(jm.group())
                lra = float(d.get("input_lra", 0))
                tp  = float(d.get("input_tp", -99))
            except Exception:
                pass

    # Crackling: use loudnorm tp if available, else approximate from max_db
    crackling_flag = False
    if tp is not None:
        crackling_flag = tp > -0.3 or (lra is not None and lra > 22)
    elif max_db is not None:
        crackling_flag = max_db > -1.0  # fallback heuristic

    # Per-section inconsistency — only for videos longer than 60s (skip for short ads)
    inconsistency_flag = False
    try:
        dur_m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", vol_r.stderr)
        if dur_m:
            total_sec = int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))
            if total_sec >= 60:
                half = total_sec / 2
                def _section_mean(start, duration):
                    r = subprocess.run(
                        ["ffmpeg", "-ss", str(start), "-t", str(duration), "-i", audio_path,
                         "-af", "volumedetect", "-f", "null", "-", "-loglevel", "info"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                    )
                    m = re.search(r"mean_volume: ([-\d.]+) dB", r.stderr)
                    return float(m.group(1)) if m else None
                with ThreadPoolExecutor(max_workers=2) as ex:
                    f1 = ex.submit(_section_mean, 0, half)
                    f2 = ex.submit(_section_mean, half, half)
                    mean_first, mean_second = f1.result(), f2.result()
                if mean_first is not None and mean_second is not None:
                    if abs(mean_first - mean_second) >= 14:
                        inconsistency_flag = True
    except Exception:
        pass

    return {
        "mean_db":            mean_db,
        "max_db":             max_db,
        "dead_air":           dead_air,
        "music_gaps":         music_gap,
        "silence_periods":    dead_air,
        "lra":                lra,
        "true_peak":          tp,
        "crackling_flag":     crackling_flag,
        "inconsistency_flag": inconsistency_flag,
    }


def analyze_audio_levels(audio_path: str, video_duration: float) -> dict:
    """
    Run ffmpeg volumedetect on the full audio and on the endcard section (last 5s).
    Returns has_audio, endcard_has_audio, mean_db_overall, mean_db_endcard.
    Mean volume above -50 dB = audio present.
    """
    SILENCE_THRESHOLD_DB = -50.0

    def _vol(path: str, start: float = None, duration: float = None) -> float:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "info"]
        if start is not None:
            cmd += ["-ss", str(start)]
        if duration is not None:
            cmd += ["-t", str(duration)]
        cmd += ["-i", path, "-af", "volumedetect", "-f", "null", "-"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
            m = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", r.stderr)
            return float(m.group(1)) if m else -99.0
        except Exception:
            return -99.0

    endcard_start = max(0.0, video_duration - 5.0)
    endcard_dur   = min(5.0, video_duration)

    mean_overall  = _vol(str(audio_path))
    mean_endcard  = _vol(str(audio_path), start=endcard_start, duration=endcard_dur)

    return {
        "has_audio":        mean_overall > SILENCE_THRESHOLD_DB,
        "endcard_has_audio": mean_endcard > SILENCE_THRESHOLD_DB,
        "mean_db_overall":  round(mean_overall, 1),
        "mean_db_endcard":  round(mean_endcard, 1),
    }


def detect_scene_cuts(video_path: str) -> list:
    """
    Use PySceneDetect to find all hard cuts in the video.
    Returns list of {timecode, time_s, score} dicts sorted by time.
    Falls back to empty list on any error.
    """
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(video_path)
        manager = SceneManager()
        manager.add_detector(ContentDetector(threshold=27.0))
        manager.detect_scenes(video, show_progress=False)
        scene_list = manager.get_scene_list()

        cuts = []
        for i, (start, end) in enumerate(scene_list):
            if i == 0:
                continue  # skip very first scene boundary at t=0
            ts = start.get_seconds()
            m, s = int(ts // 60), int(ts % 60)
            cuts.append({
                "timecode": f"{m}:{s:02d}",
                "time_s": round(ts, 2),
                "score": 0.0,  # score not available in v0.7 without stats file
            })
        return cuts
    except Exception as e:
        print(f"[scene_cuts] failed: {e}", flush=True)
        return []


def transcribe_audio(audio_path: str) -> dict:
    # ── Groq Whisper API (fast, preferred when key is set) ──
    if groq_client:
        try:
            with open(audio_path, "rb") as f:
                resp = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )
            segs = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in (resp.segments or [])
            ]
            return {"full_text": resp.text, "segments": segs}
        except Exception:
            pass  # fall through to local Whisper

    # ── Local Whisper fallback ──
    model = get_whisper()
    result = None
    for use_word_ts in (True, False):
        try:
            result = model.transcribe(audio_path, word_timestamps=use_word_ts)
            break
        except Exception:
            continue
    if result is None:
        return {"full_text": "", "segments": []}
    segs = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    return {"full_text": result["text"], "segments": segs}


def extract_captions(video_path: str, tmp_dir: str) -> dict:
    out = os.path.join(tmp_dir, "captions.srt")
    r = subprocess.run(
        ["ffmpeg", "-i", video_path, "-map", "0:s:0", out, "-y", "-loglevel", "quiet"],
        capture_output=True,
    )
    if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
        with open(out, encoding="utf-8", errors="ignore") as f:
            raw = f.read()
        return {"found": True, "raw": raw, "entries": _parse_srt(raw)}
    return {"found": False, "entries": []}


def _parse_srt(raw: str) -> list:
    entries = []
    for block in re.split(r"\n\n+", raw.strip()):
        lines = block.strip().splitlines()
        if len(lines) >= 3:
            m = re.match(r"(\d{2}:\d{2}:\d{2}[,\.]\d+) --> (\d{2}:\d{2}:\d{2}[,\.]\d+)", lines[1])
            if m:
                entries.append({
                    "start": m.group(1),
                    "end": m.group(2),
                    "text": " ".join(lines[2:]).strip(),
                })
    return entries


_SPLIT_STOP_WORDS = {
    "and","the","but","for","at","in","on","is","it","or","to","a","an",
    "that","this","with","from","they","then","when","what","here","have",
    "also","just","like","some","into","over","only","even","back","other",
    "your","our","its","very","each","been","same","does","will","make",
    "more","most","much","away","well","look","good","down","now","out",
    "up","all","no","so","do","go","there","their","these","those",
    "ingredients","product","formula","skin","face","body","hair","cream",
    "serum","oil","water","love","help","know","think","want","need","feel",
    "see","get","got","new","first","last","every","always","never","after",
    "before","which","where","about","still","really","actually","use",
    "using","works","working","feel","feeling","been","without","within",
    "something","everything","anything","nothing","someone","everyone",
}


def detect_caption_split_errors(captions: dict) -> list:
    """
    Detects bad caption splits where a word appears at the end of one line
    and again at the start of the next line.
    Example:  "...other anti"  followed by  "anti-redness products"
    Returns a list of confirmed split errors with timestamps.
    """
    entries = captions.get("entries", [])
    if len(entries) < 2:
        return []

    errors = []
    for i in range(len(entries) - 1):
        curr_text = entries[i]["text"].strip().rstrip(".,!?")
        next_text = entries[i + 1]["text"].strip()

        if not curr_text or not next_text:
            continue

        curr_words = curr_text.lower().split()
        next_words = next_text.lower().split()

        if not curr_words or not next_words:
            continue

        last_word  = curr_words[-1].strip(".,!?-")
        first_word = next_words[0].strip(".,!?-")

        # Hyphenated prefix check BEFORE stop words
        if '-' in first_word:
            prefix = first_word.split('-')[0]
            if len(prefix) >= 3:
                ocr_corrected = last_word[:-1] + "ti" if last_word.endswith("d") and len(last_word) >= 2 else None
                if last_word == prefix or (ocr_corrected and ocr_corrected == prefix):
                    errors.append({
                        "line_a": entries[i]["text"],
                        "line_b": entries[i + 1]["text"],
                        "timestamp": entries[i]["start"],
                        "duplicate_word": prefix,
                    })
                    continue

        if not last_word or last_word in _SPLIT_STOP_WORDS:
            continue

        # Match: last word of line N == first word (or prefix) of line N+1
        if last_word and first_word and len(last_word) >= 4:
            if first_word.startswith(last_word) or last_word.startswith(first_word):
                errors.append({
                    "line_a": entries[i]["text"],
                    "line_b": entries[i + 1]["text"],
                    "timestamp": entries[i]["start"],
                    "duplicate_word": last_word,
                })

    return errors


def extract_caption_strips(tmp_dir: str, duration: float, fps_val: float = 2.0) -> list:
    """
    Crop the bottom 22% of every frame (where captions live) and resize small.
    Returns list of {ts, b64} for the AI caption-reading pass.
    fps_val must be the original extraction fps — never derive it from capped frame count.
    """
    try:
        from PIL import Image
        import io
    except ImportError:
        return []

    frames_dir = os.path.join(tmp_dir, "frames")
    if not os.path.exists(frames_dir):
        return []

    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") and f != "frame_last.jpg"])
    if not files:
        return []

    # Cap at 40 strips — enough to catch all caption transitions without excess API calls
    MAX_STRIPS = 40
    if len(files) > MAX_STRIPS:
        indices = [int(i * (len(files) - 1) / (MAX_STRIPS - 1)) for i in range(MAX_STRIPS)]
        files = [files[i] for i in indices]

    strips = []
    for fname in files:
        try:
            fpath = os.path.join(frames_dir, fname)
            img = Image.open(fpath)
            w, h = img.size
            # Bottom 22% strip — where captions typically live
            crop = img.crop((0, int(h * 0.78), w, h))
            # Resize to 480px wide — small enough to be token-efficient
            new_w = 480
            new_h = max(1, int(crop.height * new_w / crop.width))
            crop = crop.resize((new_w, new_h), Image.LANCZOS)

            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode()

            num = int(re.search(r"frame_(\d+)", fname).group(1))
            ts = round((num - 1) / fps_val, 2)  # frame numbers start at 1

            strips.append({"ts": ts, "b64": b64})
        except Exception:
            continue

    return strips


def find_caption_transitions(strips: list, threshold: float = 12.0) -> list:
    """
    Compare consecutive caption strips pixel-by-pixel.
    Return only strips where the caption visually changed — i.e. caption transitions.
    This reduces 50+ strips to ~15-20 meaningful transition frames.
    """
    try:
        from PIL import Image
        import io as _io
    except ImportError:
        return strips

    transitions = []
    prev_data = None

    for strip in strips:
        try:
            img_bytes = base64.b64decode(strip["b64"])
            img = Image.open(_io.BytesIO(img_bytes)).convert("L").resize((120, 30))
            curr_data = list(img.getdata())

            if prev_data is None:
                transitions.append(strip)  # always include first
            else:
                diff = sum(abs(a - b) for a, b in zip(curr_data, prev_data)) / len(curr_data)
                if diff > threshold:
                    transitions.append(strip)

            prev_data = curr_data
        except Exception:
            transitions.append(strip)

    return transitions


async def read_captions_from_frames_ai(strips: list) -> list:
    """
    Dedicated AI pass: read burned-in captions from frame strips.
    Sends in batches of 10 with explicit frame numbering.
    Returns list of {ts, text}.
    """
    if not strips:
        return []

    # Filter to transition frames only
    transition_strips = find_caption_transitions(strips)

    all_captions = []
    BATCH = 15  # balanced batch size — fewer API calls, still reliable

    for batch_start in range(0, len(transition_strips), BATCH):
        batch = transition_strips[batch_start: batch_start + BATCH]

        # Build frame reference list for prompt
        frame_refs = "\n".join(
            f"Frame {i+1} → timestamp {strip['ts']:.1f}s"
            for i, strip in enumerate(batch)
        )

        content = [{"type": "text", "text": f"""Read the caption text burned into the bottom of each video frame strip below.

Frame timestamps:
{frame_refs}

Return ONLY a JSON array with exactly {len(batch)} entries — one per frame, in order:
[{{"ts": <timestamp>, "text": "<exact caption text or empty string>"}}, ...]

IMPORTANT:
- Copy exact spelling including typos
- Use the timestamp values listed above
- Return empty string if no caption is visible
- No preamble, no markdown"""}]

        for i, strip in enumerate(batch):
            content.append({"type": "text", "text": f"Frame {i+1} ({strip['ts']:.1f}s):"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{strip['b64']}"}})

        resp = ai_client.chat.completions.create(
            model=CAPTION_READ_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
        )

        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        try:
            batch_caps = json.loads(raw)
            all_captions.extend(batch_caps)
        except Exception:
            # If JSON parse fails, still continue with next batch
            pass

    return all_captions


def detect_duplicate_captions(frames_dir: str, fps_val: float) -> list:
    """
    Scans the bottom 35% of each frame with pytesseract looking for two distinct
    caption text bands within that zone — indicating two simultaneous caption layers.
    Ignores upper-half overlays (ingredient callouts, product names, etc.) entirely.
    Returns list of {timestamp} dicts.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return []

    if not os.path.exists(frames_dir):
        return []

    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg") and f != "frame_last.jpg"])
    duplicates = []
    seen_windows = []

    for fname in files:
        try:
            fpath = os.path.join(frames_dir, fname)
            img = Image.open(fpath)
            w, h = img.size

            # Only scan bottom 35% — where captions live. Ignore upper overlays.
            crop_top = int(h * 0.65)
            caption_zone = img.crop((0, crop_top, w, h))
            zone_h = caption_zone.height

            data = pytesseract.image_to_data(
                caption_zone,
                output_type=pytesseract.Output.DICT,
                config="--psm 11",
            )

            # Collect Y-center positions of confident text words within the zone
            text_y = []
            for i, conf in enumerate(data["conf"]):
                try:
                    conf_val = int(conf)
                except (ValueError, TypeError):
                    continue
                if conf_val > 50 and len(data["text"][i].strip()) >= 2:
                    y_center = data["top"][i] + data["height"][i] / 2
                    text_y.append(y_center)

            if len(text_y) < 4:  # need enough words to confirm two real lines
                continue

            # Cluster text into bands by Y position
            # Sort and look for a gap > 20% of zone height between clusters
            text_y_sorted = sorted(text_y)
            zone_gap_threshold = zone_h * 0.20

            band_break = None
            for idx in range(len(text_y_sorted) - 1):
                gap = text_y_sorted[idx + 1] - text_y_sorted[idx]
                if gap > zone_gap_threshold:
                    band_break = idx
                    break

            if band_break is None:
                continue  # all text in one band — normal single caption

            band_a = text_y_sorted[:band_break + 1]
            band_b = text_y_sorted[band_break + 1:]

            # Both bands must have at least 2 words to confirm real duplicate caption lines
            if len(band_a) >= 2 and len(band_b) >= 2:
                num = int(re.search(r"frame_(\d+)", fname).group(1))
                ts = round((num - 1) / fps_val, 2)
                m, s = int(ts // 60), int(ts % 60)
                ts_str = f"{m}:{s:02d}"

                # Deduplicate by 4-second window to avoid flooding
                window = int(ts) // 4
                if window not in seen_windows:
                    seen_windows.append(window)
                    duplicates.append({"timestamp": ts_str, "ts": ts})

        except Exception:
            continue

    return duplicates


def _srt_to_sec(ts: str) -> float:
    """Convert SRT timestamp '00:00:21,000' to seconds float."""
    try:
        ts = ts.replace(",", ".")
        parts = ts.split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except Exception:
        return 0.0


def _align_captions_transcript(captions: dict, transcript: dict) -> list:
    """
    Align every embedded SRT caption line with the Whisper transcript text
    for the same time window.  Returns a list of dicts:
      { timestamp, caption_text, transcript_text, has_diff }
    has_diff = True when words (>3 chars) differ between the two texts.
    Used to inject a side-by-side diff into the AI prompt so mismatches
    are flagged even when a frame wasn't sampled at that moment.
    """
    if not captions.get("found") or not captions.get("entries"):
        return []
    segments = transcript.get("segments", [])
    if not segments:
        return []

    aligned = []
    for entry in captions["entries"]:
        cap_start = _srt_to_sec(entry["start"])
        cap_end   = _srt_to_sec(entry["end"])
        cap_text  = entry["text"].strip()
        if not cap_text:
            continue

        # Collect transcript segments that overlap this caption window (+/- 1.5s tolerance)
        overlap = []
        for seg in segments:
            if seg["start"] < cap_end + 1.5 and seg["end"] > cap_start - 1.5:
                overlap.append(seg["text"].strip())
        tx_text = " ".join(overlap).strip() if overlap else ""

        # Simple word-level diff — ignore short/stop words, detect meaningful differences
        cap_words = set(
            w.lower().strip(".,!?-'\"")
            for w in re.findall(r"\b\w+\b", cap_text)
            if len(w) > 3
        )
        tx_words = set(
            w.lower().strip(".,!?-'\"")
            for w in re.findall(r"\b\w+\b", tx_text)
            if len(w) > 3
        )
        has_diff = bool(tx_text) and bool(cap_words.symmetric_difference(tx_words))

        m_min, m_sec = int(cap_start // 60), int(cap_start % 60)
        aligned.append({
            "timestamp":       f"{m_min}:{m_sec:02d}",
            "caption_text":    cap_text,
            "transcript_text": tx_text,
            "has_diff":        has_diff,
        })

    return aligned


def detect_splits_from_caption_list(caption_list: list) -> list:
    """
    Takes AI-read caption list [{ts, text}, ...] and detects split errors
    where a word appears at end of line N and start of line N+1.
    """
    if len(caption_list) < 2:
        return []

    # Deduplicate consecutive identical captions first
    deduped = []
    for cap in caption_list:
        if not deduped or cap["text"].strip().lower() != deduped[-1]["text"].strip().lower():
            deduped.append(cap)

    errors = []
    for i in range(len(deduped) - 1):
        curr = deduped[i]["text"].strip().rstrip(".,!?")
        nxt  = deduped[i + 1]["text"].strip()

        if not curr or not nxt:
            continue

        curr_words = curr.lower().split()
        next_words = nxt.lower().split()

        if not curr_words or not next_words:
            continue

        last  = curr_words[-1].strip(".,!?-")
        first = next_words[0].strip(".,!?-")

        # Hyphenated prefix check runs BEFORE stop words — catches OCR misreads like
        # "anti" → "and" where "and" would otherwise be filtered as a stop word.
        if '-' in first:
            prefix = first.split('-')[0]
            if len(prefix) >= 3:
                ocr_corrected = last[:-1] + "ti" if last.endswith("d") and len(last) >= 2 else None
                if last == prefix or (ocr_corrected and ocr_corrected == prefix):
                    ts = deduped[i]["ts"]
                    m, s = int(ts // 60), int(ts % 60)
                    errors.append({
                        "line_a": deduped[i]["text"],
                        "line_b": deduped[i + 1]["text"],
                        "timestamp": f"{m}:{s:02d}",
                        "duplicate_word": prefix,
                    })
                    continue

        if not last or last in _SPLIT_STOP_WORDS:
            continue

        if len(last) >= 4 and (first.startswith(last) or last.startswith(first)):
            ts = deduped[i]["ts"]
            m, s = int(ts // 60), int(ts % 60)
            errors.append({
                "line_a":         deduped[i]["text"],
                "line_b":         deduped[i + 1]["text"],
                "timestamp":      f"{m}:{s:02d}",
                "duplicate_word": last,
            })

    return errors


# ─────────────────────────────────────────────────────────
# CHECKLIST LOADER
# ─────────────────────────────────────────────────────────

CHECKLISTS_DIR = os.path.join(os.path.dirname(__file__), "checklists")


def _load_checklist(name: str) -> dict:
    path = os.path.join(CHECKLISTS_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _format_checklist(data: dict) -> str:
    checks = "\n".join(f"- {c}" for c in data["checks"])
    do_not = "\n".join(f"- {c}" for c in data.get("do_not_flag", []))
    out = f"CHECK FOR:\n{checks}"
    if do_not:
        out += f"\n\nDO NOT FLAG:\n{do_not}"
    return out


# ─────────────────────────────────────────────────────────
# AI ANALYSIS — TWO PARALLEL AGENTS
# ─────────────────────────────────────────────────────────

async def _run_agents(frames, audio_ctx, video_info, caption_cl, visual_cl, context):
    """Run both agents in parallel and return (captions_result, visual_result)."""
    captions_task = asyncio.create_task(
        _captions_agent(frames, audio_ctx, video_info, caption_cl, context=context)
    )
    visual_task = asyncio.create_task(
        _visual_agent(frames, audio_ctx, video_info, visual_cl, context=context)
    )
    captions_result, visual_result = await asyncio.gather(captions_task, visual_task, return_exceptions=True)
    if isinstance(captions_result, Exception):
        print(f"[ERROR] captions_agent failed: {captions_result}", flush=True)
        captions_result = {"issues": [], "summary": "", "issue_count": 0}
    if isinstance(visual_result, Exception):
        print(f"[ERROR] visual_agent failed: {visual_result}", flush=True)
        visual_result = {"issues": [], "summary": "", "issue_count": 0}
    return captions_result, visual_result


async def analyze_with_ai(frames, audio_data, transcript, captions, video_info, split_errors=None, dup_captions=None, context: Optional[str] = None, audio_levels: Optional[dict] = None, scene_cuts: Optional[list] = None) -> dict:
    audio_ctx = _build_audio_context(audio_data, transcript, captions, split_errors or [], dup_captions or [], audio_levels=audio_levels, video_duration=video_info.get("duration", 0), scene_cuts=scene_cuts or [])
    caption_cl = _load_checklist("captions")
    visual_cl  = _load_checklist("visual")

    captions_result, visual_result = await _run_agents(frames, audio_ctx, video_info, caption_cl, visual_cl, context)

    # If both agents returned 0 issues on a non-trivial video, run once more.
    # Gemini 2.0 Flash is non-deterministic — a single 0-result pass may be a miss.
    # Retry only when the video has real content (has frames and some audio/transcript signal).
    has_content = len(frames) > 5 and (
        audio_data.get("mean_db") is not None or
        len(transcript.get("segments", [])) > 0 or
        audio_levels is not None
    )
    if (not captions_result.get("issues") and not visual_result.get("issues")) and has_content:
        print("[analyze_with_ai] 0 issues on non-trivial video — retrying once", flush=True)
        captions_result2, visual_result2 = await _run_agents(frames, audio_ctx, video_info, caption_cl, visual_cl, context)
        # Take whichever pass found more issues
        if len(captions_result2.get("issues", [])) + len(visual_result2.get("issues", [])) > 0:
            captions_result, visual_result = captions_result2, visual_result2

    all_issues = captions_result.get("issues", []) + visual_result.get("issues", [])
    all_issues.sort(key=lambda x: {"Critical": 0, "Major": 1, "Minor": 2}.get(x.get("severity", "Minor"), 2))

    # Deduplicate: both agents sometimes flag the same underlying problem independently.
    # Fingerprint each issue by its meaningful title keywords + timestamp bucket.
    # Keep only the first (highest-severity) occurrence of each fingerprint.
    _GENERIC = {"audio", "video", "caption", "overlay", "screen", "frame",
                "missing", "absent", "present", "issue", "error", "check"}

    def _fp(issue: dict) -> tuple:
        # Use 5-char stems so "brand" and "branding" both hash to "brand"
        words = frozenset(
            w.strip(".,!?-").lower()[:5] for w in issue.get("issue", "").split()
            if len(w.strip(".,!?-")) > 3 and w.strip(".,!?-").lower() not in _GENERIC
        )
        ts = issue.get("timestamp", "").lower()
        # "end" and "throughout" are both global scope — same dedup bucket
        bucket = "global" if ("throughout" in ts or ts.strip() == "end") else ts[:4]
        return words, bucket

    seen_fps: list = []
    deduped: list = []
    for issue in all_issues:
        fp_words, fp_ts = _fp(issue)
        is_dup = any(
            fp_ts == s_ts and len(fp_words) > 0 and len(fp_words & s_words) >= 1
            for s_words, s_ts in seen_fps
        )
        if not is_dup:
            seen_fps.append((fp_words, fp_ts))
            deduped.append(issue)
    all_issues = deduped

    total = len(all_issues)
    if total == 0:
        summary = "No issues found. Video looks clean."
    else:
        summaries = [s for s in [captions_result.get("summary"), visual_result.get("summary")] if s]
        summary = " | ".join(summaries) if summaries else f"{total} issue(s) found."

    return {"issues": all_issues, "summary": summary, "issue_count": total}


async def _captions_agent(frames, audio_ctx: str, video_info: dict, checklist: dict, context: Optional[str] = None) -> dict:
    duration = video_info["duration"]
    checklist_text = _format_checklist(checklist)

    context_section = ""
    if context and context.strip():
        context_section = f"""
── ADDITIONAL CONTEXT FROM QA TEAM ──
{context.strip()}

Use this context to guide your analysis. It may contain the video script, client brief, brand guidelines, known issues from previous rounds, or any other relevant information. If a script is included, use it for sentence splitting and VO mismatch checks — split sentences at every period, exclamation mark, or question mark and flag any caption that merges two script sentences or breaks mid-phrase.
"""

    content = [{"type": "text", "text": f"""You are a caption QA specialist for a creative ad agency. Find only real, confirmed caption issues.

VIDEO INFO: {duration:.1f}s | {video_info['width']}x{video_info['height']}

{audio_ctx}{context_section}

FRAMES: {len(frames)} frames (~{duration / max(len(frames), 1):.1f}s apart). Read every caption on screen carefully.

{checklist_text}

PLAIN LANGUAGE RULE: Write every issue description for a video editor, not a technician. No dB values, no LUFS, no True Peak, no LRA, no codec terms. Say what the editor sees and what they need to fix.

TIMESTAMP FORMAT: Always use MM:SS format (e.g. 0:05, 0:17, 1:02). Transcript segment times are given in raw seconds — convert them: 17.38s = 0:17, 35.0s = 0:35, 90.5s = 1:30. Never write a timestamp like "17:38" or "35:00" for a short video.

RETURN VALID JSON ONLY — no preamble, no markdown fences:
{{
  "issues": [
    {{
      "category": "Captions",
      "severity": "Critical|Major|Minor",
      "timestamp": "0:05 or 0:10-0:14 or Throughout",
      "issue": "Short issue title",
      "description": "What the editor sees and what needs fixing. Max 2 sentences."
    }}
  ],
  "summary": "One-sentence caption verdict.",
  "issue_count": 0
}}"""}]

    for fr in frames:
        m, s = int(fr["ts"] // 60), int(fr["ts"] % 60)
        content.append({"type": "text", "text": f"[Frame {m}:{s:02d}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fr['b64']}"}})

    return await _call_ai(content)


async def _visual_agent(frames, audio_ctx: str, video_info: dict, checklist: dict, context: Optional[str] = None) -> dict:
    duration = video_info["duration"]
    checklist_text = _format_checklist(checklist)

    context_section = ""
    if context and context.strip():
        context_section = f"\n── ADDITIONAL CONTEXT FROM QA TEAM ──\n{context.strip()}\n\nUse this as background when checking branding, endcard, and visual requirements.\n"

    content = [{"type": "text", "text": f"""You are a visual QA specialist for a creative ad agency. Find only real, confirmed visual and audio issues.

VIDEO INFO: {duration:.1f}s | {video_info['width']}x{video_info['height']}

{audio_ctx}{context_section}
FRAMES: {len(frames)} frames (~{duration / max(len(frames), 1):.1f}s apart). Inspect every frame carefully.

{checklist_text}

── MANDATORY PRE-CHECK BEFORE WRITING ISSUES ──
Before filling in the issues array, answer these two questions internally:

1. SIGN CHECK: Does any frame show a person holding a physical sign, card, or paper with text?
   If YES — go through every such frame and ask: "Can I read ALL the text on this sign right now?"
   Any frame where the answer is NO (text covered, cut off, or obscured by a dark/colored overlay or animation) MUST appear as an issue.
   Do not skip this because the overlay looks like a transition or animation style. If text is unreadable = flag it, no exceptions.

2. BRANDING CHECK: Does a client logo, brand name (as a graphic or styled text — not handwritten on a prop), or branded endcard screen appear anywhere in the video?
   If NO brand identifier appears at any point = flag as "Branding absent."

Only after completing both pre-checks, write your issues array.

PLAIN LANGUAGE RULE: Write every issue description for a video editor, not a technician. Never include dB values, LUFS, True Peak, LRA, codec terms, or any raw audio measurements. Translate technical signals into what the editor hears or sees and what they need to fix. Example: instead of "True Peak=-0.82 dB indicates clipping", write "The audio is too loud and will distort on most devices — reduce the volume."

TIMESTAMP FORMAT: Always use MM:SS format (e.g. 0:05, 0:17, 1:02). Transcript segment times are given in raw seconds — convert them: 17.38s = 0:17, 35.0s = 0:35, 90.5s = 1:30. Never write a timestamp like "17:38" or "35:00" for a short video.

RETURN VALID JSON ONLY — no preamble, no markdown fences:
{{
  "issues": [
    {{
      "category": "Audio|Video|Overlays",
      "severity": "Critical|Major|Minor",
      "timestamp": "0:05 or 0:10-0:14 or Throughout or End",
      "issue": "Short issue title",
      "description": "What the editor sees and what needs fixing. Max 2 sentences."
    }}
  ],
  "summary": "One-sentence visual verdict.",
  "issue_count": 0
}}"""}]

    for fr in frames:
        m, s = int(fr["ts"] // 60), int(fr["ts"] % 60)
        content.append({"type": "text", "text": f"[Frame {m}:{s:02d}]"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fr['b64']}"}})

    return await _call_ai(content)


async def _call_ai(content: list) -> dict:
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: ai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0,
        ),
    )
    if not resp or not resp.choices:
        return {"issues": [], "summary": "AI returned no response.", "issue_count": 0}
    raw = resp.choices[0].message.content
    if not raw:
        return {"issues": [], "summary": "AI returned empty content.", "issue_count": 0}
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"issues": [], "summary": "JSON parse error.", "raw": raw, "issue_count": 0}


def _build_audio_context(audio_data: dict, transcript: dict, captions: dict, split_errors: list = None, dup_captions: list = None, audio_levels: Optional[dict] = None, video_duration: float = 0, scene_cuts: list = None) -> str:
    lines = ["── AUDIO DATA ──"]

    # Inject endcard audio metadata so AI doesn't false-flag endcard music absence
    if audio_levels is not None:
        lines.append(
            f"[AUDIO METADATA] Overall audio present: {'YES' if audio_levels['has_audio'] else 'NO'} "
            f"(mean {audio_levels['mean_db_overall']} dB)  |  "
            f"Endcard audio present: {'YES' if audio_levels['endcard_has_audio'] else 'NO'} "
            f"(mean {audio_levels['mean_db_endcard']} dB)"
        )
        lines.append(
            "Note: 'audio present' = any signal (VO + music + ambient). "
            "If Endcard audio present = YES, background music is running during the endcard — "
            "do NOT flag the endcard for missing audio."
        )

    if audio_data.get("mean_db") is not None:
        lines.append(f"Mean volume: {audio_data['mean_db']} dB  |  Max: {audio_data['max_db']} dB")
    if audio_data.get("crackling_flag"):
        lines.append(
            "[AUDIO FLAG — CLIPPING] Audio levels are too hot — possible clipping or over-compression. "
            "Flag once as: audio is too loud and may distort on playback."
        )
    if audio_data.get("inconsistency_flag"):
        lines.append(
            "[AUDIO FLAG — INCONSISTENCY] Volume level changes significantly between the first and second half of the video. "
            "Flag as: audio is inconsistent — one section sounds noticeably louder or quieter than another."
        )

    dead_air = audio_data.get("dead_air", [])
    music_gaps = audio_data.get("music_gaps", [])

    if dead_air:
        # Filter out end-of-video dead air (last 3s) — natural pacing, not an error
        eov_cutoff = max(0, video_duration - 3.0) if video_duration > 6 else 0
        dead_air_filtered = [g for g in dead_air if eov_cutoff == 0 or g["start"] < eov_cutoff]

        # Merge consecutive dead air gaps within 2 seconds of each other into one entry
        merged_dead_air = []
        for gap in sorted(dead_air_filtered, key=lambda x: x["start"]):
            if merged_dead_air and gap["start"] - merged_dead_air[-1]["end"] <= 2.0:
                merged_dead_air[-1]["end"] = max(merged_dead_air[-1]["end"], gap["end"])
                merged_dead_air[-1]["duration"] = round(merged_dead_air[-1]["end"] - merged_dead_air[-1]["start"], 2)
            else:
                merged_dead_air.append(dict(gap))
        if merged_dead_air:
            lines.append(f"DEAD AIR (truly silent — no voice, no music) [{len(merged_dead_air)} gap(s) — report each as ONE issue]:")
            for s in merged_dead_air[:20]:
                lines.append(f"  {s['start']}s - {s['end']}s  ({s['duration']}s)  <-- FLAG THIS as ONE issue")

    if music_gaps:
        long_gaps  = [s for s in music_gaps if s["duration"] >= 2.0]
        short_gaps = [s for s in music_gaps if s["duration"] < 2.0]
        if long_gaps:
            lines.append(f"INTER-SENTENCE PAUSES (voice stops, music present, >=2.0s — FLAG EVERY ONE) [{len(long_gaps)} gap(s)]:")
            for s in long_gaps[:20]:
                lines.append(f"  {s['start']}s - {s['end']}s  ({s['duration']}s)  <-- FLAG THIS")
        if short_gaps:
            lines.append(f"SHORT PAUSES (<2.0s — do NOT flag, natural speech pacing) [{len(short_gaps)} gap(s)]:")
            for s in short_gaps[:10]:
                lines.append(f"  {s['start']}s - {s['end']}s  ({s['duration']}s)")

    lines.append("\n── TRANSCRIPT (what talent actually says) ──")
    for seg in transcript.get("segments", [])[:80]:
        lines.append(f"  [{seg['start']}s-{seg['end']}s]  {seg['text']}")

    if captions.get("found"):
        aligned = _align_captions_transcript(captions, transcript)
        if aligned:
            diff_count = sum(1 for a in aligned if a["has_diff"])
            lines.append(f"\n── CAPTION vs TRANSCRIPT ALIGNMENT ({len(aligned)} lines, {diff_count} with word differences) ──")
            lines.append("Caption text shown alongside the matching transcript text for that moment.")
            lines.append("Lines marked ← DIFF have word differences — review those for real mismatches.")
            lines.append("RULES: Skip brand names, product names, technical/specialized terms — Whisper mishears those. Only flag common everyday words where meaning is clearly wrong.")
            for item in aligned[:80]:
                diff_marker = "  ← DIFF" if item["has_diff"] else ""
                lines.append(
                    f"  [{item['timestamp']}] "
                    f"Caption: \"{item['caption_text']}\"  |  "
                    f"Transcript: \"{item['transcript_text']}\"{diff_marker}"
                )
        else:
            lines.append("\n── EMBEDDED CAPTIONS (what appears on screen) ──")
            for e in captions["entries"][:80]:
                lines.append(f"  [{e['start']}]  {e['text']}")
    else:
        lines.append(
            "\n── CAPTIONS: None embedded — "
            "captions may be burned into the video, read them from frames visually ──"
        )

    if split_errors:
        lines.append(f"\n── CAPTION SPLIT ERRORS DETECTED ({len(split_errors)}) — FLAG ALL OF THESE ──")
        for e in split_errors:
            lines.append(
                f"  [{e['timestamp']}]  '{e['line_a']}'  -->  '{e['line_b']}'  "
                f"(word '{e['duplicate_word']}' duplicated across split)"
            )

    if scene_cuts:
        lines.append(f"\n── SCENE CUTS DETECTED BY SCENEDETECT ({len(scene_cuts)} cuts) ──")
        lines.append("These are the ONLY confirmed hard cuts in this video. Use this list when evaluating jump cuts.")
        lines.append("A cut is only worth flagging if it is jarring, disorienting, or clearly unintentional — not just because it exists.")
        for c in scene_cuts[:30]:
            lines.append(f"  {c['timecode']}")

    return "\n".join(lines)
