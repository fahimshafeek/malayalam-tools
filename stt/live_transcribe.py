#!/usr/bin/env python3
"""
live_transcribe.py — near-live transcription, cut on pauses.

Records continuously (indefinitely) until Ctrl+C. pydub is used to slice the
incoming audio on short pauses between words/phrases; each finished slice is
sent to the whisper.cpp server and printed as soon as it comes back:

    🎙️  Listening… (Ctrl+C to stop)   [pause ≥ 400ms, threshold -34.0 dBFS]
    📝 ഹലോ ഞാൻ പറയുന്നത് മനസ്സിലാവുന്നുണ്ടോ
    📝 എന്റെ പേര് ഹരി എന്നാണ്

At startup the background level is measured and the silence threshold is set
just above it (override with ``--silence-thresh``). Cutting only happens on
pauses at least ``--min-silence`` ms long, so words aren't split mid-phrase.

Options
-------
    .venv/bin/python live_transcribe.py
    .venv/bin/python live_transcribe.py --min-silence 300
    .venv/bin/python live_transcribe.py --silence-thresh -38
    .venv/bin/python live_transcribe.py --language ml
    .venv/bin/python live_transcribe.py --list-devices

Requires: ``arecord`` (alsa-utils), ``numpy``, ``pydub``, and a running
whisper.cpp server (default http://localhost:8081/inference).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

SAMPLE_RATE = 16_000
DEFAULT_SERVER = "http://localhost:8081/inference"


# --------------------------------------------------------------------------- #
# Audio capture                                                                #
# --------------------------------------------------------------------------- #
class MicCapture:
    """Stream raw 16-bit mono PCM from `arecord` into a queue of float32 arrays."""

    def __init__(self, device: str, sample_rate: int):
        self.device = device or "default"
        self.sample_rate = sample_rate
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if not shutil.which("arecord"):
            raise RuntimeError(
                "`arecord` was not found. Install ALSA tools, e.g. "
                "`sudo apt install alsa-utils`."
            )
        cmd = [
            "arecord", "-q", "-D", self.device,
            "-f", "S16_LE", "-r", str(self.sample_rate), "-c", "1", "-t", "raw",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        time.sleep(0.3)
        if self._proc.poll() is not None:
            err = (self._proc.stderr.read() or b"").decode(errors="replace").strip()
            raise RuntimeError(err or f"arecord exited immediately (device {self.device!r})")
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(4096)  # 2048 int16 samples ≈ 128 ms
            if not chunk:
                break
            samples = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
            self._queue.put(samples)

    def drain(self) -> np.ndarray | None:
        parts: list[np.ndarray] = []
        while True:
            try:
                parts.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else np.concatenate(parts)

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=1)


# --------------------------------------------------------------------------- #
# Server communication                                                         #
# --------------------------------------------------------------------------- #
def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def transcribe(server: str, audio: np.ndarray, sample_rate: int, language: str,
               timeout: float = 60.0) -> str:
    """POST a float32 clip to the whisper.cpp server and return its text."""
    boundary = "----livestt" + uuid.uuid4().hex
    wav = _encode_wav(audio, sample_rate)
    body = io.BytesIO()

    def field(name: str, value: str) -> None:
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(value.encode())
        body.write(b"\r\n")

    def file_field(name: str, filename: str, content: bytes, ctype: str) -> None:
        body.write(f"--{boundary}\r\n".encode())
        body.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(content)
        body.write(b"\r\n")

    file_field("file", "audio.wav", wav, "audio/wav")
    field("language", language)
    field("temperature", "0.0")
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        server,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return (payload.get("text") or "").strip()


def check_server(server: str) -> None:
    base = server.rsplit("/", 1)[0] or server
    try:
        urllib.request.urlopen(base, timeout=5)
    except urllib.error.HTTPError:
        return
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach the whisper.cpp server at {server!r} ({exc}).\n"
            "   Start it first, e.g.:  cd stt && docker compose up -d"
        )


def db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 1e-7 else -140.0


# --------------------------------------------------------------------------- #
# Near-live segmenter                                                          #
# --------------------------------------------------------------------------- #
class LiveTranscriber:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sr = args.sample_rate
        self.pending = np.zeros(0, dtype=np.float32)
        self.ambient_dbfs = -60.0
        self.thresh = args.silence_thresh if args.silence_thresh is not None else -40.0

        self.capture = MicCapture(args.device, self.sr)
        self.tx_queue: "queue.Queue[np.ndarray | None]" = queue.Queue()
        self._quit = threading.Event()
        self._worker = threading.Thread(target=self._tx_loop, daemon=True)
        self._last_process = 0.0

    # -- transcription worker ---------------------------------------------- #
    def _tx_loop(self) -> None:
        while True:
            audio = self.tx_queue.get()
            if audio is None:
                return
            seconds = audio.shape[0] / self.sr
            try:
                text = transcribe(self.args.server, audio, self.sr, self.args.language,
                                  timeout=max(30.0, seconds * 3 + 15))
            except Exception as exc:
                print(f"⚠️  transcription failed: {exc}", flush=True)
                continue
            text = text.strip()
            if text:
                print(f"📝 {text}", flush=True)

    # -- segmentation ------------------------------------------------------- #
    def _emit(self, start_ms: float, end_ms: float) -> None:
        a = max(0, int(start_ms * self.sr / 1000))
        b = min(self.pending.shape[0], int(end_ms * self.sr / 1000))
        if b - a < int(self.args.min_segment * self.sr / 1000):
            return
        self.tx_queue.put(self.pending[a:b].copy())

    def _crop(self, keep_from_ms: float) -> None:
        start = max(0, min(self.pending.shape[0], int(keep_from_ms * self.sr / 1000)))
        self.pending = self.pending[start:]

    def _process(self) -> None:
        if self.pending.shape[0] < int(0.05 * self.sr):
            return
        pcm = (np.clip(self.pending, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        seg = AudioSegment(data=pcm, sample_width=2, frame_rate=self.sr, channels=1)
        total = len(seg)  # ms

        ranges = detect_nonsilent(
            seg,
            min_silence_len=self.args.min_silence,
            silence_thresh=self.thresh,
            seek_step=self.args.seek_step,
        )

        if not ranges:
            # All silence: drop it, but keep a short tail to preserve a word start.
            keep = min(total, self.args.keep_silence)
            self._crop(total - keep)
            return

        last_start, last_end = ranges[-1]
        trailing = total - last_end

        if trailing >= self.args.min_silence:
            # The last utterance is also followed by a long enough pause: commit all.
            completed = ranges
            keep_ms = total
        else:
            # The last utterance is still in progress: commit the earlier ones.
            completed = ranges[:-1]
            keep_ms = max(0, last_start - self.args.keep_silence)

        for start, end in completed:
            pad_a = max(0, start - self.args.keep_silence)
            pad_b = min(total, end + self.args.keep_silence)
            self._emit(pad_a, pad_b)

        # If a single utterance drags on with no pause, force-flush it in chunks.
        if trailing < self.args.min_silence:
            in_progress = total - last_start
            if in_progress >= self.args.max_segment * 1000:
                cut = last_start + self.args.max_segment * 1000
                self._emit(last_start, cut)
                keep_ms = cut

        self._crop(keep_ms)

    # -- calibrate + run ---------------------------------------------------- #
    def _calibrate(self) -> None:
        print("🎚️  measuring background noise — stay quiet…", flush=True)
        deadline = time.time() + self.args.calibrate
        parts: list[np.ndarray] = []
        while time.time() < deadline:
            chunk = self.capture.drain()
            if chunk is None:
                time.sleep(0.02)
                continue
            parts.append(chunk)
        if parts:
            audio = np.concatenate(parts)
            self.ambient_dbfs = db(float(np.sqrt(np.mean(audio * audio))))
        else:
            self.ambient_dbfs = -60.0
        if self.args.silence_thresh is None:
            self.thresh = max(-60.0, min(-3.0, self.ambient_dbfs + self.args.thresh_margin))

    def run(self) -> int:
        try:
            check_server(self.args.server)
        except RuntimeError as exc:
            print(f"❌ {exc}")
            return 1
        try:
            self.capture.start()
        except Exception as exc:
            print(f"❌ Could not open the microphone ({self.args.device!r}): {exc}")
            print("   Try `--list-devices` and pass `--device <name>`.")
            return 1

        self._calibrate()
        self._worker.start()
        print(
            f"🎙️  Listening… (Ctrl+C to stop)   "
            f"[pause ≥ {self.args.min_silence}ms, "
            f"threshold {self.thresh:.1f} dBFS "
            f"(ambient {self.ambient_dbfs:.1f})]\n",
            flush=True,
        )

        try:
            while True:
                chunk = self.capture.drain()
                if chunk is not None and chunk.size:
                    self.pending = (
                        chunk if self.pending.size == 0
                        else np.concatenate((self.pending, chunk))
                    )
                now = time.time()
                if now - self._last_process >= self.args.process_interval:
                    self._last_process = now
                    self._process()
                time.sleep(0.02)
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.tx_queue.put(None)
            self._worker.join(timeout=5)
            print("\n👋 stopped.", flush=True)
        finally:
            self.capture.stop()
        return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def list_devices() -> None:
    print("=== ALSA capture devices (`arecord -l`) ===")
    if shutil.which("arecord"):
        subprocess.run(["arecord", "-l"], check=False)
    else:
        print("  arecord not found.")
    print('\nPass a device with `--device <name>` (e.g. "default", "plughw:2,0").', flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Near-live transcription that cuts on pauses (Malayalam/English).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--server", default=DEFAULT_SERVER, help="whisper.cpp /inference endpoint")
    p.add_argument("--language", default="ml",
                   help="language code passed to the server ('ml', 'en', or 'auto')")
    p.add_argument("--device", default="default", help="ALSA capture device for arecord")
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="capture sample rate (Hz)")

    g = p.add_argument_group("segmentation (pydub)")
    g.add_argument("--min-silence", type=int, default=400,
                   help="pause length (ms) that cuts an utterance")
    g.add_argument("--keep-silence", type=int, default=200,
                   help="silence (ms) kept around each slice")
    g.add_argument("--silence-thresh", type=float, default=None,
                   help="silence threshold in dBFS (default: auto from background)")
    g.add_argument("--thresh-margin", type=float, default=4.0,
                   help="auto threshold = background + this many dB")
    g.add_argument("--calibrate", type=float, default=1.0,
                   help="seconds of background measured at startup")
    g.add_argument("--seek-step", type=int, default=20,
                   help="pydub detection step (ms); larger = cheaper")
    g.add_argument("--max-segment", type=float, default=12.0,
                   help="force-flush an utterance after this long (s)")
    g.add_argument("--min-segment", type=float, default=0.25,
                   help="ignore slices shorter than this (s)")
    g.add_argument("--process-interval", type=float, default=0.15,
                   help="how often to look for pauses (s)")
    p.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        list_devices()
        return 0
    return LiveTranscriber(args).run()


if __name__ == "__main__":
    sys.exit(main())
