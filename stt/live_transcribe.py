#!/usr/bin/env python3
"""
live_transcribe.py — record a short clip and transcribe it.

Simple test mode: capture a fixed number of seconds of microphone audio, send
it to the whisper.cpp server, and print the transcript. No live streaming, no
voice gate.

    $ .venv/bin/python live_transcribe.py
    🎙️  Recording 5.0s — speak now
    ⏹️  captured 5.00s  rms 0.0432 (-27.3 dBFS)  peak 0.3100
    📝 ഹലോ ...

Options
-------
    .venv/bin/python live_transcribe.py --seconds 5
    .venv/bin/python live_transcribe.py --loop          # repeat until Ctrl+C
    .venv/bin/python live_transcribe.py --language ml
    .venv/bin/python live_transcribe.py --list-devices

Requires: ``arecord`` (alsa-utils), ``numpy``, and a running whisper.cpp server
(default http://localhost:8081/inference).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np

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

    def get(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

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


# --------------------------------------------------------------------------- #
# Record + transcribe                                                          #
# --------------------------------------------------------------------------- #
def db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 1e-7 else -140.0


def record(capture: MicCapture, seconds: float, sample_rate: int) -> np.ndarray:
    """Collect exactly `seconds` of audio from the capture queue."""
    target = int(seconds * sample_rate)
    frames: list[np.ndarray] = []
    got = 0
    while got < target:
        chunk = capture.get(timeout=0.5)
        if chunk is None:
            continue
        frames.append(chunk)
        got += chunk.shape[0]
    if not frames:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames)[:target]


def run_once(args: argparse.Namespace, capture: MicCapture) -> None:
    print(f"🎙️  Recording {args.seconds:.1f}s — speak now", flush=True)
    audio = record(capture, args.seconds, args.sample_rate)

    if audio.size == 0:
        print("⚠️  No audio captured.")
        return

    rms = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))
    print(
        f"⏹️  captured {audio.shape[0] / args.sample_rate:.2f}s  "
        f"rms {rms:.4f} ({db(rms):5.1f} dBFS)  peak {peak:.4f}",
        flush=True,
    )

    started = time.time()
    try:
        text = transcribe(args.server, audio, args.sample_rate, args.language)
    except Exception as exc:
        print(f"⚠️  transcription failed: {exc}", flush=True)
        return
    elapsed = time.time() - started
    rtf = (audio.shape[0] / args.sample_rate) / elapsed if elapsed else 0.0
    print(f"📝 {text if text else '(nothing recognised)'}", flush=True)
    print(f"   transcribed in {elapsed:.2f}s ({rtf:.1f}x realtime)\n", flush=True)


def list_devices() -> None:
    print("=== ALSA capture devices (`arecord -l`) ===")
    if shutil.which("arecord"):
        subprocess.run(["arecord", "-l"], check=False)
    else:
        print("  arecord not found.")
    print('\nPass a device with `--device <name>` (e.g. "default", "plughw:2,0").', flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record a short clip and transcribe it (Malayalam/English).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seconds", type=float, default=5.0, help="clip length to record")
    p.add_argument("--loop", action="store_true", help="record + transcribe repeatedly until Ctrl+C")
    p.add_argument("--server", default=DEFAULT_SERVER, help="whisper.cpp /inference endpoint")
    p.add_argument("--language", default="ml",
                   help="language code passed to the server ('ml', 'en', or 'auto')")
    p.add_argument("--device", default="default", help="ALSA capture device for arecord")
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="capture sample rate (Hz)")
    p.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        list_devices()
        return 0

    try:
        check_server(args.server)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1

    capture = MicCapture(args.device, args.sample_rate)
    try:
        capture.start()
    except Exception as exc:
        print(f"❌ Could not open the microphone ({args.device!r}): {exc}")
        print("   Try `--list-devices` and pass `--device <name>`.")
        return 1

    try:
        while True:
            run_once(args, capture)
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n👋 stopped.")
    finally:
        capture.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
