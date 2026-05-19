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

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL = os.getenv("MODEL", "anthropic/claude-3.5-sonnet")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

ai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

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

        yield {"type": "progress", "step": "Preparing video...", "percent": 5}
        video_path, video_filename = await loop.run_in_executor(None, lambda: get_video(url, file, tmp_dir))

        yield {"type": "progress", "step": "Reading video metadata...", "percent": 12}
        video_info = get_video_info(video_path)

        yield {"type": "progress", "step": "Extracting frames...", "percent": 20}
        frames, fps_val = await loop.run_in_executor(
            None, lambda: extract_frames(video_path, tmp_dir, video_info["duration"])
        )

        yield {"type": "progress", "step": "Extracting audio...", "percent": 30}
        audio_path = await loop.run_in_executor(None, lambda: extract_audio(video_path, tmp_dir))

        yield {"type": "progress", "step": "Analyzing audio levels...", "percent": 38}
        audio_data = await loop.run_in_executor(None, lambda: analyze_audio(audio_path))

        yield {"type": "progress", "step": "Transcribing audio (this may take a moment)...", "percent": 45}
        transcript = await loop.run_in_executor(None, lambda: transcribe_audio(audio_path))

        yield {"type": "progress", "step": "Extracting captions...", "percent": 60}
        captions = await loop.run_in_executor(None, lambda: extract_captions(video_path, tmp_dir))

        # Caption split detection — embedded SRT first, then OCR pass for burned-in captions
        yield {"type": "progress", "step": "Reading captions from frames...", "percent": 64}
        split_errors = await loop.run_in_executor(None, lambda: detect_caption_split_errors(captions))
        if not split_errors:
            # No embedded captions or no errors found — run OCR-based caption reading
            caption_strips = await loop.run_in_executor(
                None, lambda: extract_caption_strips(tmp_dir, video_info["duration"], fps_val)
            )
            if caption_strips:
                yield {"type": "progress", "step": "AI reading burned-in captions...", "percent": 68}
                ocr_captions = await read_captions_from_frames_ai(caption_strips)
                split_errors = detect_splits_from_caption_list(ocr_captions)

        yield {"type": "progress", "step": "Running AI analysis...", "percent": 74}
        report = await analyze_with_ai(frames, audio_data, transcript, captions, video_info, split_errors, context=context)
        report["filename"] = video_filename
        print(f"[DEBUG] filename in report: {report.get('filename')!r}")

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

    # Dropbox Replay link
    if "replay.dropbox.com" in url:
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
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        def _on_response(response):
            ru = response.url
            ct = response.headers.get("content-type", "").lower()
            if ("video" in ct or "mpegurl" in ct or ru.endswith(".m3u8") or "dropboxusercontent.com" in ru) \
                    and ru not in [c[0] for c in captured]:
                captured.append((ru, ct))

        page.on("response", _on_response)
        try:
            page.goto(url, wait_until="networkidle", timeout=45_000)
            page.wait_for_timeout(5_000)
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
            print(f"[DEBUG] Replay page title: {title!r}")
            # Strip "- Dropbox Replay" / "| Replay" suffix and " MP4" suffix from Dropbox titles
            clean = re.sub(r'\s*[-|]\s*(Dropbox\s*Replay|Replay).*$', '', title, flags=re.IGNORECASE).strip()
            clean = re.sub(r'\s+MP4\s*$', '', clean, flags=re.IGNORECASE).strip()
            # Treat "Dropbox Replay" or very short cleaned titles as failures
            if clean and len(clean) > 5 and "dropbox" not in clean.lower():
                display_name = clean + ".mp4"
            else:
                display_name = f"replay_{share_id}.mp4"
        except Exception as e:
            print(f"[DEBUG] Replay title extraction failed: {e}")
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
# VIDEO PROCESSING
# ─────────────────────────────────────────────────────────

def get_video_info(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
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

    # Cap at 60 frames, evenly distributed
    if len(files) > 60:
        indices = [int(i * (len(files) - 1) / 59) for i in range(60)]
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
    # Volume levels
    vol_r = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-af", "volumedetect", "-f", "null", "-", "-loglevel", "info"],
        capture_output=True, text=True,
    )
    mean_m = re.search(r"mean_volume: ([-\d.]+) dB", vol_r.stderr)
    max_m = re.search(r"max_volume: ([-\d.]+) dB", vol_r.stderr)

    # Pass 1 — voice silence (no voice at -35dB)
    sil_r = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-af", "silencedetect=noise=-35dB:d=0.3",
         "-f", "null", "-", "-loglevel", "info"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", sil_r.stderr)]
    ends   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)",   sil_r.stderr)]

    voice_silent = []
    for i, s in enumerate(starts):
        if i < len(ends):
            dur = round(ends[i] - s, 2)
            if dur >= 0.4:
                voice_silent.append({"start": round(s, 2), "end": round(ends[i], 2), "duration": dur})

    # Pass 2 — total silence (nothing at all at -60dB, catches music/ambient)
    sil_r2 = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-af", "silencedetect=noise=-60dB:d=0.3",
         "-f", "null", "-", "-loglevel", "info"],
        capture_output=True, text=True,
    )
    starts2 = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", sil_r2.stderr)]
    ends2   = [float(x) for x in re.findall(r"silence_end: ([\d.]+)",   sil_r2.stderr)]

    totally_silent = []
    for i, s in enumerate(starts2):
        if i < len(ends2):
            dur = round(ends2[i] - s, 2)
            if dur >= 0.4:
                totally_silent.append({"start": round(s, 2), "end": round(ends2[i], 2), "duration": dur})

    # Classify: dead_air = silent at BOTH thresholds (no music underneath)
    #           music_gap = silent at -35dB but NOT at -60dB (background music present)
    def _overlaps(a, b, tol=0.5):
        return a["start"] <= b["end"] + tol and b["start"] <= a["end"] + tol

    dead_air  = []
    music_gap = []
    for gap in voice_silent:
        has_total_silence = any(_overlaps(gap, ts) for ts in totally_silent)
        if has_total_silence:
            dead_air.append(gap)
        else:
            music_gap.append(gap)

    # Loudnorm — LRA + true peak for crackling detection
    ln_r = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-af", "loudnorm=print_format=json",
         "-f", "null", "-", "-loglevel", "info"],
        capture_output=True, text=True,
    )
    lra = tp = None
    jm = re.search(r"\{[\s\S]+?\}", ln_r.stderr)
    if jm:
        try:
            d = json.loads(jm.group())
            lra = float(d.get("input_lra", 0))
            tp  = float(d.get("input_tp", -99))
        except Exception:
            pass

    # Per-section volume check — detect inconsistency between first and second half
    inconsistency_flag = False
    try:
        dur_m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", vol_r.stderr)
        if dur_m:
            total_sec = int(dur_m.group(1)) * 3600 + int(dur_m.group(2)) * 60 + float(dur_m.group(3))
            half = total_sec / 2
            def _section_mean(start, duration):
                r = subprocess.run(
                    ["ffmpeg", "-ss", str(start), "-t", str(duration), "-i", audio_path,
                     "-af", "volumedetect", "-f", "null", "-", "-loglevel", "info"],
                    capture_output=True, text=True,
                )
                m = re.search(r"mean_volume: ([-\d.]+) dB", r.stderr)
                return float(m.group(1)) if m else None
            mean_first  = _section_mean(0, half)
            mean_second = _section_mean(half, half)
            if mean_first is not None and mean_second is not None:
                if abs(mean_first - mean_second) >= 8:
                    inconsistency_flag = True
    except Exception:
        pass

    return {
        "mean_db":           float(mean_m.group(1)) if mean_m else None,
        "max_db":            float(max_m.group(1))  if max_m  else None,
        "dead_air":          dead_air,
        "music_gaps":        music_gap,
        "silence_periods":   dead_air,
        "lra":               lra,
        "true_peak":         tp,
        "crackling_flag":    tp is not None and (tp > -1.0 or (lra is not None and lra > 20)),
        "inconsistency_flag": inconsistency_flag,
    }


def transcribe_audio(audio_path: str) -> dict:
    model = get_whisper()
    try:
        result = model.transcribe(audio_path, word_timestamps=True)
    except Exception:
        # word_timestamps can fail on certain audio (reshape tensor bug) — retry without
        result = model.transcribe(audio_path, word_timestamps=False)
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

    # Cap at 80 strips — higher than frame cap so caption transitions aren't missed
    MAX_STRIPS = 80
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
    BATCH = 10  # small batches for reliable reading

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
            model=MODEL,
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

async def analyze_with_ai(frames, audio_data, transcript, captions, video_info, split_errors=None, context: Optional[str] = None) -> dict:
    audio_ctx = _build_audio_context(audio_data, transcript, captions, split_errors or [])
    caption_cl = _load_checklist("captions")
    visual_cl  = _load_checklist("visual")

    captions_task = asyncio.create_task(
        _captions_agent(frames, audio_ctx, video_info, caption_cl, context=context)
    )
    visual_task = asyncio.create_task(
        _visual_agent(frames, audio_ctx, video_info, visual_cl, context=context)
    )

    captions_result, visual_result = await asyncio.gather(captions_task, visual_task)

    all_issues = captions_result.get("issues", []) + visual_result.get("issues", [])
    all_issues.sort(key=lambda x: {"Critical": 0, "Major": 1, "Minor": 2}.get(x.get("severity", "Minor"), 2))

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

PLAIN LANGUAGE RULE: Write every issue description for a video editor, not a technician. Never include dB values, LUFS, True Peak, LRA, codec terms, or any raw audio measurements. Translate technical signals into what the editor hears or sees and what they need to fix. Example: instead of "True Peak=-0.82 dB indicates clipping", write "The audio is too loud and will distort on most devices — reduce the volume."

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
        ),
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return {"issues": [], "summary": "JSON parse error.", "raw": raw, "issue_count": 0}


def _build_audio_context(audio_data: dict, transcript: dict, captions: dict, split_errors: list = None) -> str:
    lines = ["── AUDIO DATA ──"]

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
        lines.append(f"DEAD AIR (truly silent — no voice, no music) [{len(dead_air)} gap(s)]:")
        for s in dead_air[:20]:
            lines.append(f"  {s['start']}s - {s['end']}s  ({s['duration']}s)  <-- FLAG THIS")

    if music_gaps:
        long_gaps = [s for s in music_gaps if s["duration"] >= 1.0]
        short_gaps = [s for s in music_gaps if s["duration"] < 1.0]
        if long_gaps:
            lines.append(f"INTER-SENTENCE PAUSES (voice stops, music present, >=1.0s — FLAG THESE) [{len(long_gaps)} gap(s)]:")
            for s in long_gaps[:20]:
                lines.append(f"  {s['start']}s - {s['end']}s  ({s['duration']}s)  <-- FLAG: pause between sentences, should be trimmed")
        if short_gaps:
            lines.append(f"SHORT MUSIC GAPS (<1.0s — do NOT flag) [{len(short_gaps)} gap(s)]:")
            for s in short_gaps[:10]:
                lines.append(f"  {s['start']}s - {s['end']}s  ({s['duration']}s)")

    lines.append("\n── TRANSCRIPT (what talent actually says) ──")
    for seg in transcript.get("segments", [])[:80]:
        lines.append(f"  [{seg['start']}s-{seg['end']}s]  {seg['text']}")

    if captions.get("found"):
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

    return "\n".join(lines)
