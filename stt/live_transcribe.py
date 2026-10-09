#!/usr/bin/env python3
"""
live_transcribe.py — continuous, live transcription in the terminal.

No voice-activity threshold is used: audio is transcribed continuously, so
nothing gets clipped. The terminal shows a live, self-updating view:

    <recent transcript>                <- last lines of the current segment
    [████████░░░░░░░░░░░░] rms 0.0512 (-25.8 dBFS)  peak 0.8701  buf 6.3s

The bottom line is the raw, real-time level readout — use it to see the actual
numbers while you speak vs. stay quiet.

How it works
------------
* ``arecord`` streams raw 16 kHz mono PCM from the microphone on a background
  thread.
* Audio accumulates into a segment. Every ``--step`` seconds the whole segment
  is re-transcribed and the view is refreshed, so the transcript grows live.
* When the segment reaches ``--window`` seconds it is "committed" (printed as a
  normal line above the live view) and a new segment starts. This bounds
  latency and keeps the transcript from growing forever.
* Transcription runs on a worker thread, so the meter and Ctrl+C stay
  responsive.

There are deliberately **no thresholds** here. Once the numbers are known, a
proper segmenter can be added back.

Usage
-----
    .venv/bin/python live_transcribe.py
    .venv/bin/python live_transcribe.py --window 12 --step 1.0
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
import signal
import subprocess
import sys
import textwrap
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
# Terminal live view (redraws a block of lines in place)                       #
# --------------------------------------------------------------------------- #
class LiveDisplay:
    """A block of lines at the bottom of the terminal that can be redrawn."""

    def __init__(self) -> None:
        self.n = 0  # number of lines currently drawn

    def render(self, lines: list[str]) -> None:
        out = []
        if self.n:
            out.append(f"\x1b[{self.n}A")  # move cursor up to the block start
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
# Live transcription (no thresholds)                                           #
# --------------------------------------------------------------------------- #
class LiveTranscriber:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rate = args.sample_rate

        self.buffer = np.zeros(0, dtype=np.float32)
        self.rms = 0.0
        self.peak = 0.0
        self.text = ""
        self.gen = 0  # bumped on every commit so stale results are ignored

        self.capture = MicCapture(args.device, self.rate)
        self.display = LiveDisplay()
        self._out_lock = threading.Lock()

        self._cv = threading.Condition()
        self._mailbox: tuple[np.ndarray, int] | None = None
        self._quit = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)

        self._busy = False          # a transcription request is in flight
        self._committing = False    # current request is the "commit" snapshot
        self._submitted_len = 0     # buffer length of the in-flight snapshot
        self._last_submit = 0.0

    # -- worker ------------------------------------------------------------- #
    def _submit(self, audio: np.ndarray, gen: int) -> None:
        with self._cv:
            if self._busy:
                return  # one transcription at a time; newer audio stays buffered
            self._busy = True
            self._mailbox = (audio, gen)
            self._cv.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while self._mailbox is None and not self._quit.is_set():
                    self._cv.wait(0.2)
                if self._mailbox is None and self._quit.is_set():
                    return
                audio, gen = self._mailbox
                self._mailbox = None

            seconds = audio.shape[0] / self.rate
            timeout = max(30.0, seconds * 3.0 + 15.0)
            try:
                text = transcribe(
                    self.args.server, audio, self.rate, self.args.language, timeout=timeout
                )
            except Exception:
                text = self.text

            with self._cv:
                if gen == self.gen:  # drop results for an already-committed segment
                    self.text = text
                self._busy = False

    # -- rendering ---------------------------------------------------------- #
    def _status_line(self) -> str:
        rms = self.rms
        db = 20.0 * math.log10(rms) if rms > 1e-7 else -140.0
        width = 20
        frac = max(0.0, min(1.0, (db + 60.0) / 60.0))
        filled = int(frac * width)
        bar = "█" * filled + "░" * (width - filled)
        return (
            f"[{bar}] rms {rms:.4f} ({db:5.1f} dBFS)  "
            f"peak {self.peak:.4f}  buf {self.buffer.shape[0] / self.rate:4.1f}s"
        )

    def _draw(self) -> None:
        cols, rows = shutil.get_terminal_size((90, 24))
        max_lines = max(1, rows - 3)
        wrapped: list[str] = []
        for para in (self.text or "").splitlines() or [""]:
            wrapped.extend(textwrap.wrap(para, width=max(20, cols - 4)) or [""])
        shown = wrapped[-max_lines:] if wrapped and any(wrapped) else ["(listening…)"]
        lines = ["  " + s for s in shown] + ["  " + self._status_line()]
        with self._out_lock:
            self.display.render(lines)

    # -- commit ------------------------------------------------------------- #
    def _flush(self) -> None:
        """Print the current segment as a normal line and start a new one."""
        with self._cv:
            text = self.text.strip()
            self.gen += 1
            self.text = ""
        # Keep any audio captured after the committed snapshot for the next segment.
        keep = min(self._submitted_len, self.buffer.shape[0])
        self.buffer = self.buffer[keep:]
        self._submitted_len = 0
        self._last_submit = time.time()
        self._committing = False
        with self._out_lock:
            self.display.clear()
            if text:
                print(text, flush=True)

    # -- main loop ---------------------------------------------------------- #
    def _pump(self) -> None:
        chunk = self.capture.drain()
        if chunk is not None and chunk.size:
            self.buffer = np.concatenate((self.buffer, chunk))
            self.rms = float(np.sqrt(np.mean(chunk * chunk)))
            self.peak = float(np.max(np.abs(chunk)))

        now = time.time()
        if not self._busy:
            secs = self.buffer.shape[0] / self.rate
            if self._committing:
                # The full-segment snapshot is ready: commit it and start over.
                self._flush()
            elif secs >= self.args.window:
                # Grab a final snapshot of the whole segment, then commit when done.
                self._submitted_len = self.buffer.shape[0]
                self._submit(self.buffer.copy(), self.gen)
                self._last_submit = now
                self._committing = True
            elif now - self._last_submit >= self.args.step and self.buffer.shape[0] > 0:
                # Live preview of the growing segment.
                self._submitted_len = self.buffer.shape[0]
                self._submit(self.buffer.copy(), self.gen)
                self._last_submit = now

        self._draw()
        time.sleep(0.05)

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
        print("Speak now: (Ctrl+C to stop)    [no thresholds — continuous stream]\n",
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
                final = self.text.strip()
                if final:
                    print(final, flush=True)
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
        description="Continuous live speech-to-text in the terminal (no thresholds).",
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
    p.add_argument("--window", type=float, default=8.0,
                   help="commit the segment (and start a new line) after this many seconds")
    p.add_argument("--step", type=float, default=1.0,
                   help="re-transcribe the current segment this often (s)")
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
