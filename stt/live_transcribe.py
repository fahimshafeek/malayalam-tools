#!/usr/bin/env python3
"""
live_transcribe.py — live transcription with a voice gate you can tune.

Only speech is sent to the model: a lightweight energy gate drops silence and
low-level background so nearby chatter / noise doesn't get transcribed. The
terminal shows a live meter and lets you move the gate while you watch:

    ഹലോ ഞാൻ പറയുന്നത് …                       <- live preview of the utterance
    [██████░░░░░░░░░░░░░░] rms 0.0421 (-27.5 dBFS)  peak 0.2100  thr 0.0800  ● speech
    +/- adjust gate    r recalibrate    Ctrl+C stop

Press ``+`` / ``-`` to raise / lower the speech gate until the meter only lights
up when *you* talk. ``r`` re-measures the background. The gate is remembered in
place, so you can dial it in live.

How it works
------------
* ``arecord`` streams raw 16 kHz mono PCM on a background thread.
* At startup the background is measured once to set the gate automatically.
* A frame is "speech" when its RMS is above the gate (with hysteresis and a
  short attack so clicks don't trigger it). Speech is collected into an
  utterance; a short pause ends it, and the utterance is transcribed and
  printed. While you speak, the growing utterance is previewed live.
* Transcription runs on a worker thread, so the meter and keys stay responsive.

Usage
-----
    .venv/bin/python live_transcribe.py
    .venv/bin/python live_transcribe.py --threshold-rms 0.05   # fix the gate
    .venv/bin/python live_transcribe.py --noise-multiplier 4   # more aggressive
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
import select
import shutil
import signal
import subprocess
import sys
import termios
import textwrap
import threading
import time
import tty
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
# Terminal live view                                                           #
# --------------------------------------------------------------------------- #
class LiveDisplay:
    def __init__(self) -> None:
        self.n = 0

    def render(self, lines: list[str]) -> None:
        out = []
        if self.n:
            out.append(f"\x1b[{self.n}A")
        for line in lines:
            out.append("\r\x1b[2K" + line + "\n")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self.n = len(lines)

    def clear(self) -> None:
        if not self.n:
            return
        out = [f"\x1b[{self.n}A"]
        for _ in range(self.n):
            out.append("\r\x1b[2K\n")
        out.append(f"\x1b[{self.n}A")
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self.n = 0


# --------------------------------------------------------------------------- #
# Live transcription with a tunable speech gate                                #
# --------------------------------------------------------------------------- #
class LiveTranscriber:
    FRAME_SECONDS = 0.02  # 20 ms

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rate = args.sample_rate
        self.frame = int(self.FRAME_SECONDS * self.rate)
        self.preroll_len = int(0.4 * self.rate)   # keep 400 ms before speech
        self.attack_frames = max(1, int(args.attack / self.FRAME_SECONDS))

        self.preroll = np.zeros(0, dtype=np.float32)
        self.utter: list[np.ndarray] = []
        self.speech_samples = 0
        self.silence_run = 0.0
        self.hot_run = 0
        self.in_speech = False

        self.noise_floor = 0.01
        self.threshold = args.threshold_rms if args.threshold_rms else 0.02
        self.pending = np.zeros(0, dtype=np.float32)
        self.rms = 0.0
        self.peak = 0.0
        self.text = ""
        self.gen = 0

        self.capture = MicCapture(args.device, self.rate)
        self.display = LiveDisplay()
        self._out_lock = threading.Lock()

        self._cv = threading.Condition()
        self._mailbox: tuple[np.ndarray, bool, int] | None = None
        self._busy = False
        self._result: tuple[str, bool, int] | None = None
        self._quit = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._last_partial = 0.0
        self._old_term = None

    # -- worker ------------------------------------------------------------- #
    def _submit(self, audio: np.ndarray, is_final: bool, gen: int) -> None:
        with self._cv:
            if self._busy:
                return
            self._busy = True
            self._mailbox = (audio, is_final, gen)
            self._cv.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while self._mailbox is None and not self._quit.is_set():
                    self._cv.wait(0.2)
                if self._mailbox is None and self._quit.is_set():
                    return
                audio, is_final, gen = self._mailbox
                self._mailbox = None
            seconds = audio.shape[0] / self.rate
            try:
                text = transcribe(self.args.server, audio, self.rate,
                                  self.args.language, timeout=max(30.0, seconds * 3 + 15))
            except Exception:
                text = ""
            with self._cv:
                self._result = (text, is_final, gen)
                self._busy = False

    # -- keyboard ----------------------------------------------------------- #
    def _read_key(self) -> str | None:
        if not sys.stdin.isatty():
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if not r:
            return None
        return sys.stdin.read(1)

    def _handle_key(self, key: str) -> None:
        if key in ("+", "="):
            self.threshold = min(0.9, self.threshold * 1.15)
        elif key in ("-", "_"):
            self.threshold = max(0.001, self.threshold / 1.15)
        elif key in ("r", "R"):
            self._calibrate()
        elif key in ("q", "Q"):
            raise KeyboardInterrupt

    # -- rendering ---------------------------------------------------------- #
    def _db(self, value: float) -> float:
        return 20.0 * math.log10(value) if value > 1e-7 else -140.0

    def _status_line(self) -> str:
        width = 20
        frac = max(0.0, min(1.0, (self._db(self.rms) + 60.0) / 60.0))
        filled = int(frac * width)
        bar = "█" * filled + "░" * (width - filled)
        state = "● speech" if self.in_speech else "○ quiet"
        return (
            f"[{bar}] rms {self.rms:.4f} ({self._db(self.rms):5.1f} dBFS)  "
            f"peak {self.peak:.4f}  thr {self.threshold:.4f} "
            f"({self._db(self.threshold):5.1f} dBFS)  {state}"
        )

    def _draw(self) -> None:
        cols, rows = shutil.get_terminal_size((100, 24))
        max_lines = max(1, rows - 4)
        wrapped: list[str] = []
        for para in (self.text or "").splitlines() or [""]:
            wrapped.extend(textwrap.wrap(para, width=max(20, cols - 4)) or [""])
        shown = wrapped[-max_lines:] if wrapped and any(wrapped) else [""]
        lines = (
            ["  " + s for s in shown]
            + ["  " + self._status_line()]
            + ["  +/- adjust gate   r recalibrate   Ctrl+C stop"]
        )
        with self._out_lock:
            self.display.render(lines)

    # -- VAD ---------------------------------------------------------------- #
    def _process_frame(self, frame: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(frame * frame)))
        start_thr = self.threshold
        end_thr = start_thr * 0.6  # hysteresis: easier to keep talking than to start

        if not self.in_speech:
            if rms < start_thr:
                self.noise_floor = 0.98 * self.noise_floor + 0.02 * rms
            self.preroll = (
                frame if self.preroll.size == 0
                else np.concatenate((self.preroll, frame))
            )
            if self.preroll.shape[0] > self.preroll_len:
                self.preroll = self.preroll[-self.preroll_len:]
            self.hot_run = self.hot_run + 1 if rms > start_thr else 0
            if self.hot_run >= self.attack_frames:
                self.in_speech = True
                self.utter = [self.preroll.copy()]
                self.speech_samples = 0
                self.silence_run = 0.0
                self.hot_run = 0
                self._last_partial = 0.0
        else:
            self.utter.append(frame)
            self.speech_samples += frame.shape[0]
            self.silence_run = 0.0 if rms > end_thr else self.silence_run + self.FRAME_SECONDS

    def _utterance_audio(self) -> np.ndarray:
        return np.concatenate(self.utter) if self.utter else np.zeros(0, dtype=np.float32)

    def _end_utterance(self) -> None:
        audio = self._utterance_audio()
        if audio.shape[0] >= int(self.args.min_speech * self.rate):
            self._submit(audio, is_final=True, gen=self.gen)
        self.gen += 1
        self.utter = []
        self.speech_samples = 0
        self.silence_run = 0.0
        self.in_speech = False
        self.text = ""

    def _calibrate(self) -> None:
        """Measure the background and set the gate above it."""
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
        if self.args.threshold_rms is None:
            self.threshold = max(self.args.energy_threshold,
                                 self.noise_floor * self.args.noise_multiplier)

    # -- main loop ---------------------------------------------------------- #
    def _pump(self) -> None:
        # keys (live gate tuning)
        while True:
            key = self._read_key()
            if key is None:
                break
            self._handle_key(key)

        chunk = self.capture.drain()
        if chunk is not None and chunk.size:
            self.pending = chunk if self.pending.size == 0 else np.concatenate((self.pending, chunk))
            self.rms = float(np.sqrt(np.mean(chunk * chunk)))
            self.peak = float(np.max(np.abs(chunk)))

        while self.pending.shape[0] >= self.frame:
            frame = self.pending[: self.frame]
            self.pending = self.pending[self.frame:]
            self._process_frame(frame)

        if self.in_speech:
            now = time.time()
            if self.silence_run >= self.args.end_silence or \
                    self.speech_samples / self.rate >= self.args.max_utterance:
                self._end_utterance()
            elif now - self._last_partial >= self.args.partial_interval and \
                    self.speech_samples >= int(self.args.min_speech * self.rate):
                self._last_partial = now
                self._submit(self._utterance_audio(), is_final=False, gen=self.gen)

        # worker results
        result = None
        with self._cv:
            if self._result is not None:
                result = self._result
                self._result = None
        if result is not None:
            text, is_final, gen = result
            if is_final:
                with self._out_lock:
                    self.display.clear()
                    if text.strip():
                        print(text.strip(), flush=True)
            elif gen == self.gen:
                self.text = text

        self._draw()
        time.sleep(0.03)

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

        self._worker.start()
        self._calibrate()

        if sys.stdin.isatty():
            self._old_term = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

        print(f"Speak now: (Ctrl+C to stop)   "
              f"[gate {self.threshold:.4f} / {self._db(self.threshold):.1f} dBFS]\n",
              flush=True)

        try:
            while True:
                self._pump()
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            self._quit.set()
            with self._cv:
                self._cv.notify()
            self._worker.join(timeout=2)
            with self._out_lock:
                self.display.clear()
                if self.text.strip():
                    print(self.text.strip(), flush=True)
                print("\n👋 stopped.", flush=True)
        finally:
            if self._old_term is not None:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_term)
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
        description="Live speech-to-text with a tunable voice gate (Malayalam/English).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--server", default=DEFAULT_SERVER, help="whisper.cpp /inference endpoint")
    p.add_argument("--language", default="ml",
                   help="language code passed to the server ('ml', 'en', or 'auto')")
    p.add_argument("--device", default="default", help="ALSA capture device for arecord")
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="capture sample rate (Hz)")

    g = p.add_argument_group("voice gate")
    g.add_argument("--threshold-rms", type=float, default=None,
                   help="fix the speech gate to this RMS (skips auto-calibration)")
    g.add_argument("--noise-multiplier", type=float, default=3.0,
                   help="auto gate = measured background * this")
    g.add_argument("--energy-threshold", type=float, default=0.008,
                   help="absolute minimum gate")
    g.add_argument("--calibrate", type=float, default=1.2,
                   help="seconds of background measured at startup")
    g.add_argument("--attack", type=float, default=0.12,
                   help="how long energy must persist to start an utterance (s)")
    g.add_argument("--end-silence", type=float, default=0.6,
                   help="silence that ends an utterance (s)")
    g.add_argument("--min-speech", type=float, default=0.3,
                   help="minimum speech length to transcribe (s)")
    g.add_argument("--max-utterance", type=float, default=20.0,
                   help="force-flush an utterance after this long (s)")
    g.add_argument("--partial-interval", type=float, default=0.8,
                   help="how often to preview the utterance (s)")
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
