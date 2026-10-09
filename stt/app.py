"""
app.py — faster-whisper ASR server (Malayalam / English + 90+ languages).

A drop-in replacement for the previous whisper.cpp server. It exposes the same
`/inference` multipart endpoint (so the bundled web UI and existing curl clients
keep working) and keeps the model resident on the GPU for fast transcription.

Environment variables:
    ASR_MODEL         model name or local path   (default: /models/faster-whisper-medium-ml)
    ASR_DEVICE        cuda | cpu                 (default: cuda)
    ASR_COMPUTE_TYPE  int8_float16 | float16 ... (default: int8_float16, ~1.2GB VRAM)
    ASR_LANGUAGE      force a language, e.g. ml  (default: auto-detect)
    ASR_BEAM_SIZE     beam size                  (default: 5)

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8080
"""

import os
import tempfile
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from faster_whisper import WhisperModel

MODEL_NAME = os.environ.get("ASR_MODEL", "/models/faster-whisper-medium-ml")
DEVICE = os.environ.get("ASR_DEVICE", "cuda")
COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "int8_float16")
DEFAULT_LANGUAGE = os.environ.get("ASR_LANGUAGE") or None
DEFAULT_BEAM = int(os.environ.get("ASR_BEAM_SIZE", "5"))

print(f"⏳ Loading model '{MODEL_NAME}' ({DEVICE}/{COMPUTE_TYPE}) ...", flush=True)
_started = time.time()
try:
    model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
except Exception as exc:  # pragma: no cover - startup guard
    if DEVICE == "cuda":
        print(f"⚠️  CUDA failed ({exc}); falling back to CPU int8.", flush=True)
        DEVICE, COMPUTE_TYPE = "cpu", "int8"
        model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
    else:
        raise
print(f"✅ Model ready in {time.time() - _started:.1f}s.", flush=True)

app = FastAPI(title="Malayalam Voice ASR", version="2.0")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine": "faster-whisper",
        "model": MODEL_NAME,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
    }


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    language: str = Form(""),
    temperature: float = Form(0.0),
    beam_size: int = Form(0),
):
    """Transcribe an uploaded audio file. Returns JSON with the full transcript."""
    lang = language.strip() or DEFAULT_LANGUAGE
    beam = beam_size or DEFAULT_BEAM

    suffix = os.path.splitext(file.filename or "audio")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        started = time.time()
        segments, info = model.transcribe(
            tmp_path,
            language=lang,
            beam_size=beam,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
            condition_on_previous_text=False,
        )
        segments = list(segments)
    except Exception as exc:  # pragma: no cover - request guard
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    text = "".join(segment.text for segment in segments).strip()
    return {
        "text": text,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "elapsed": round(time.time() - started, 3),
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text,
            }
            for s in segments
        ],
    }


# Serve the bundled web UI at the root (must be registered last).
PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")
if os.path.isdir(PUBLIC_DIR):
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))
