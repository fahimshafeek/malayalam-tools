#!/usr/bin/env python3
"""
live_transcribe.py — continuous, live transcription right in your terminal.

Speak into your microphone and watch the transcript appear live. It keeps
running until you press Ctrl+C.

How it works
------------
* ``arecord`` streams raw 16 kHz mono PCM from the microphone on a background
  thread, so no audio is lost while transcription is happening.
* At startup the background noise level is measured once, and a voice-activity
  detector (VAD) threshold is derived from it. The threshold keeps adapting to
  slow changes in the noise floor, so a fan / hum doesn't get transcribed.
* The VAD splits the stream into utterances: recording starts a few frames
  after you begin speaking (the pre-roll keeps the first word intact) and the
  utterance ends after a short pause.
* While you speak, the growing utterance is re-transcribed every
  ``--partial-interval`` seconds and shown on the current line (live preview).
* When you pause, the utterance is transcribed once more and printed as a
  final, committed line.
* Transcription runs on a worker thread, so Ctrl+C is always responsive, and
  long, uninterrupted speech is force-flushed every ``--max-utterance`` seconds
  to keep latency bounded.

This script does not load a model itself. It sends audio to a running
whisper.cpp server (the one started by ``docker compose up``) and prints the
text it returns, which keeps it a single, dependency-light file.

Usage
-----
    .venv/bin/python live_transcribe.py
    .venv/bin/python live_transcribe.py --server http://localhost:8081/inference
    .venv/bin/python live_transcribe.py --language ml --device default
    .venv/bin/python live_transcribe.py --list-devices

If it triggers on background noise when you are silent, raise
``--noise-multiplier`` (e.g. 4) or set a fixed floor with
``--energy-threshold``. If it misses quiet speech, lower them.

Requires: ``arecord`` (package ``alsa-utils``), ``numpy``, and a running
whisper.cpp server (default http://localhost:8081/inference).
"""

from __future__ import annotations

import argparse
import io
import json
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
        # Give arecord a moment to fail fast on a bad device.
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
        """Return all queued audio since the last call, or None if empty."""
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
    """Fail fast with a clear message if the server is unreachable."""
    base = server.rsplit("/", 1)[0] or server
    try:
        urllib.request.urlopen(base, timeout=5)
    except urllib.error.HTTPError:
        return  # reachable, just returned an error status (e.g. 404 on "/")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach the whisper.cpp server at {server!r} ({exc}).\n"
            "   Start it first, e.g.:  cd stt && docker compose up -d"
        )


# --------------------------------------------------------------------------- #
# Live transcription                                                           #
# --------------------------------------------------------------------------- #
class LiveTranscriber:
    FRAME_SECONDS = 0.02  # VAD frame size (20 ms)

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rate = args.sample_rate
        self.frame = int(self.FRAME_SECONDS * self.rate)
        self.preroll_len = int(0.3 * self.rate)   # 300 ms kept before speech
        self.attack_frames = max(1, int(args.attack / self.FRAME_SECONDS))

        # VAD state
        self.preroll = np.zeros(0, dtype=np.float32)
        self.utter: list[np.ndarray] = []
        self.speech_samples = 0
        self.silence_run = 0.0
        self.hot_run = 0
        self.in_speech = False
        self.noise_floor = 0.01

        # Transcript bookkeeping
        self.pending = np.zeros(0, dtype=np.float32)
        self.last_partial = 0.0
        self.partial_len = 0

        # Transcription worker: a single coalescing mailbox keeps exactly one
        # request (the newest, and a pending "final" always wins).
        self._cv = threading.Condition()
        self._mailbox: tuple[np.ndarray, bool] | None = None
        self._quit = threading.Event()
        self._out_lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        self.capture = MicCapture(args.device, self.rate)

    # -- terminal rendering ------------------------------------------------- #
    def _render_partial_locked(self, text: str) -> None:
        line = "  " + text
        sys.stdout.write("\r\x1b[K" + line)
        sys.stdout.flush()
        self.partial_len = len(line)

    def _clear_partial_locked(self) -> None:
        if self.partial_len:
            sys.stdout.write("\r\x1b[K")
            sys.stdout.flush()
            self.partial_len = 0

    # -- transcription worker ----------------------------------------------- #
    def _submit(self, audio: np.ndarray, is_final: bool) -> None:
        with self._cv:
            if self._mailbox is None:
                self._mailbox = (audio, is_final)
            else:
                _old_audio, old_final = self._mailbox
                self._mailbox = (audio, old_final or is_final)
            self._cv.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while self._mailbox is None and not self._quit.is_set():
                    self._cv.wait(0.2)
                if self._mailbox is None and self._quit.is_set():
                    return
                audio, is_final = self._mailbox
                self._mailbox = None

            seconds = audio.shape[0] / self.rate
            timeout = max(30.0, seconds * 3.0 + 15.0)
            try:
                text = transcribe(
                    self.args.server, audio, self.rate, self.args.language, timeout=timeout
                )
            except Exception as exc:
                if is_final:
                    with self._out_lock:
                        self._clear_partial_locked()
                        sys.stderr.write(f"\n⚠️  transcription failed: {exc}\n")
                        sys.stderr.flush()
                continue

            with self._out_lock:
                if is_final:
                    self._clear_partial_locked()
                    if text:
                        print(text, flush=True)
                elif text:
                    self._render_partial_locked(text)

    # -- VAD ----------------------------------------------------------------- #
    def _threshold(self) -> float:
        return max(self.args.energy_threshold,
                   self.noise_floor * self.args.noise_multiplier)

    def _audio_seconds(self) -> float:
        return self.speech_samples / self.rate

    def _process_frame(self, frame: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(frame * frame)))
        threshold = self._threshold()

        if not self.in_speech:
            # Track the noise floor only on clearly-quiet frames.
            if rms < threshold:
                self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms

            self.preroll = (
                frame if self.preroll.size == 0
                else np.concatenate((self.preroll, frame))
            )
            if self.preroll.shape[0] > self.preroll_len:
                self.preroll = self.preroll[-self.preroll_len:]

            self.hot_run = self.hot_run + 1 if rms > threshold else 0
            if self.hot_run >= self.attack_frames:
                # Speech starts: seed the utterance with the pre-roll buffer.
                self.in_speech = True
                self.utter = [self.preroll.copy()]
                self.speech_samples = 0
                self.silence_run = 0.0
                self.hot_run = 0
                self.last_partial = 0.0
        else:
            self.utter.append(frame)
            self.speech_samples += frame.shape[0]
            if rms > threshold:
                self.silence_run = 0.0
            else:
                self.silence_run += self.FRAME_SECONDS

    def _finalize(self) -> None:
        """Hand the current utterance to the worker and reset for the next one."""
        if self.utter:
            audio = np.concatenate(self.utter)
            if audio.shape[0] >= int(self.args.min_speech * self.rate):
                self._submit(audio, is_final=True)
        self.utter = []
        self.speech_samples = 0
        self.silence_run = 0.0
        self.in_speech = False
        self.hot_run = 0
        self.last_partial = 0.0

    # -- event loop ---------------------------------------------------------- #
    def _pump(self) -> None:
        chunk = self.capture.drain()
        if chunk is not None and chunk.size:
            self.pending = (
                chunk if self.pending.size == 0
                else np.concatenate((self.pending, chunk))
            )

        while self.pending.shape[0] >= self.frame:
            frame = self.pending[: self.frame]
            self.pending = self.pending[self.frame:]
            self._process_frame(frame)

        if self.in_speech:
            now = time.time()
            if self.silence_run >= self.args.end_silence or \
                    self._audio_seconds() >= self.args.max_utterance:
                self._finalize()
            elif now - self.last_partial >= self.args.partial_interval and \
                    self._audio_seconds() >= self.args.min_speech:
                self.last_partial = now
                self._submit(np.concatenate(self.utter), is_final=False)
        else:
            time.sleep(0.02)

    def _calibrate(self) -> None:
        """Measure the ambient noise floor before we start listening."""
        frames: list[float] = []
        deadline = time.time() + self.args.calibrate
        while time.time() < deadline:
            chunk = self.capture.drain()
            if chunk is None or not chunk.size:
                time.sleep(0.02)
                continue
            n = chunk.shape[0] // self.frame
            for i in range(n):
                f = chunk[i * self.frame:(i + 1) * self.frame]
                frames.append(float(np.sqrt(np.mean(f * f))))
        if frames:
            self.noise_floor = float(np.median(frames))

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
            f"Speak now: (Ctrl+C to stop)   "
            f"[noise floor {self.noise_floor:.4f}, "
            f"speech threshold {self._threshold():.4f}]\n",
            flush=True,
        )

        try:
            while True:
                self._pump()
        except KeyboardInterrupt:
            # Ignore any further Ctrl+C so shutdown can't be interrupted.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self._finalize()
            self._quit.set()
            with self._cv:
                self._cv.notify()
            self._worker.join(timeout=2)
            with self._out_lock:
                self._clear_partial_locked()
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
        description="Live, continuous speech-to-text in the terminal (Malayalam/English).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--server", default=DEFAULT_SERVER,
                   help="whisper.cpp /inference endpoint")
    p.add_argument("--language", default="ml",
                   help="language code passed to the server ('ml', 'en', or 'auto')")
    p.add_argument("--device", default="default",
                   help="ALSA capture device for arecord")
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                   help="capture sample rate in Hz")
    p.add_argument("--partial-interval", type=float, default=1.0,
                   help="how often to refresh the live preview while speaking (s)")
    p.add_argument("--end-silence", type=float, default=0.6,
                   help="silence duration that ends an utterance (s)")
    p.add_argument("--min-speech", type=float, default=0.25,
                   help="minimum speech length to transcribe (s)")
    p.add_argument("--max-utterance", type=float, default=20.0,
                   help="force-flush an utterance after this long (s)")
    p.add_argument("--attack", type=float, default=0.12,
                   help="how long energy must persist to start an utterance (s)")
    p.add_argument("--calibrate", type=float, default=0.7,
                   help="seconds of ambient audio measured at startup (s)")
    p.add_argument("--energy-threshold", type=float, default=0.01,
                   help="absolute minimum VAD RMS (lower bound for the adaptive threshold)")
    p.add_argument("--noise-multiplier", type=float, default=2.5,
                   help="threshold = noise_floor * this (raise if noise triggers it)")
    p.add_argument("--list-devices", action="store_true",
                   help="list microphones and exit")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.list_devices:
        list_devices()
        return 0
    return LiveTranscriber(args).run()


if __name__ == "__main__":
    sys.exit(main())
