#!/usr/bin/env python3
"""
record_transcribe.py — press-to-talk Malayalam / English speech-to-text.

Flow:
    1. A microphone stream opens automatically and starts capturing.
    2. You speak in Malayalam or English (language is auto-detected).
    3. Press ENTER to stop recording.
    4. The whole voice note is transcribed locally with faster-whisper on the GPU
       and the transcript is printed in the terminal.

Run:
    .venv/bin/python record_transcribe.py                       # Malayalam fine-tune (default)
    .venv/bin/python record_transcribe.py --model openai/whisper-medium   # generic multilingual
    .venv/bin/python record_transcribe.py --language ml         # force Malayalam
    .venv/bin/python record_transcribe.py --list-devices        # list microphones

Default model: the fine-tuned Malayalam model `thennal/whisper-medium-ml`
converted to CTranslate2 at models/faster-whisper-medium-ml (build it once with
./download_model.sh). At int8_float16 it uses ~1.2 GB of VRAM, comfortable on a
6 GB GPU (RTX 4050 Laptop).

Microphone capture uses ALSA `arecord` when available (works with PulseAudio /
PipeWire, no extra system packages), and falls back to `sounddevice`/PortAudio.
"""

import os
import sys
import site


# --------------------------------------------------------------------------- #
# CUDA runtime bootstrap                                                       #
# --------------------------------------------------------------------------- #
# CTranslate2 (the engine behind faster-whisper) needs cuBLAS + cuDNN at runtime.
# When installed via pip (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`) those live
# inside the virtualenv, so we add them to LD_LIBRARY_PATH and re-exec once.
# This lets you run the script directly without exporting anything yourself.
def _ensure_cuda_libs() -> None:
    try:
        bases = list(site.getsitepackages())
    except Exception:
        bases = []
    try:
        bases.append(site.getusersitepackages())
    except Exception:
        pass

    lib_dirs = []
    for base in bases:
        for sub in ("nvidia/cublas/lib", "nvidia/cudnn/lib", "nvidia/cuda_nvrtc/lib"):
            d = os.path.join(base, sub)
            if os.path.isdir(d):
                lib_dirs.append(os.path.abspath(d))

    if not lib_dirs:
        return

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p]
    if all(d in current for d in lib_dirs):
        return

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs + current)
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])


_ensure_cuda_libs()


import argparse
import shutil
import subprocess
import threading
import time

import numpy as np
from faster_whisper import WhisperModel

try:  # optional: only used if `arecord` is unavailable
    import sounddevice as sd
except Exception:  # PortAudio not installed, etc.
    sd = None


SAMPLE_RATE = 16_000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, "models", "faster-whisper-medium-ml")


# --------------------------------------------------------------------------- #
# Audio helpers                                                                #
# --------------------------------------------------------------------------- #
def list_devices() -> None:
    print("=== ALSA capture devices (arecord -l) ===")
    if shutil.which("arecord"):
        subprocess.run(["arecord", "-l"], check=False)
    else:
        print("  arecord not found.")
    if sd is not None:
        print("\n=== PortAudio devices (sounddevice) ===")
        print(sd.query_devices())
        print("Default input device:", sd.default.device[0])
    print('\nUse --mic <name> (e.g. "default", "plughw:2,0") with arecord,')
    print("or --engine sounddevice --mic <index> for a PortAudio device index.")


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    n_dst = int(round(audio.shape[0] * dst_rate / src_rate))
    x_src = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
    x_dst = np.linspace(0.0, 1.0, n_dst, endpoint=False)
    return np.interp(x_dst, x_src, audio).astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int, margin_ms: int = 150) -> np.ndarray:
    """Drop leading/trailing near-silence so the model gets a tight clip."""
    if audio.size == 0:
        return audio
    frame = max(1, int(sr * 0.02))
    n_frames = audio.shape[0] // frame
    if n_frames < 3:
        return audio
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float32) ** 2, axis=1))
    peak = float(rms.max())
    if peak <= 1e-6:
        return audio
    threshold = max(peak * 0.06, 0.0035)
    voiced = np.where(rms > threshold)[0]
    if voiced.size == 0:
        return audio
    margin = int(margin_ms / 20)  # frames
    start = max(0, int(voiced[0]) - margin) * frame
    end = min(n_frames, int(voiced[-1]) + margin + 1) * frame
    return audio[start:end]


# --------------------------------------------------------------------------- #
# Recorders                                                                    #
# --------------------------------------------------------------------------- #
class _BaseRecorder:
    """Capture mono float32 audio until stop() is called."""

    samplerate: int

    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._level = 0.0
        self._lock = threading.Lock()

    def _push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        with self._lock:
            self._frames.append(samples)
            self._level = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    @property
    def level(self) -> float:
        return self._level

    def _collect(self) -> np.ndarray:
        with self._lock:
            if not self._frames:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._frames, axis=0).reshape(-1).astype(np.float32)

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> np.ndarray:
        raise NotImplementedError


class ArecordRecorder(_BaseRecorder):
    """Microphone capture via ALSA `arecord`, streaming raw PCM on stdout."""

    def __init__(self, device: str, samplerate: int):
        super().__init__()
        self.device = device or "default"
        self.samplerate = samplerate
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        cmd = [
            "arecord", "-q", "-D", self.device,
            "-f", "S16_LE", "-r", str(self.samplerate), "-c", "1", "-t", "raw",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        # Give arecord a moment to fail fast on a bad device.
        time.sleep(0.25)
        if self._proc.poll() is not None:
            err = (self._proc.stderr.read() or b"").decode(errors="replace").strip()
            raise RuntimeError(err or "arecord exited immediately")
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(4096)  # 2048 int16 samples
            if not chunk:
                break
            self._push(np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0)

    def stop(self) -> np.ndarray:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=1)
        return self._collect()


class SoundDeviceRecorder(_BaseRecorder):
    """Microphone capture via PortAudio (used only if `arecord` is missing)."""

    def __init__(self, device, samplerate: int):
        super().__init__()
        self.device = device
        self.samplerate = samplerate
        self._stream = None

    def start(self) -> None:
        def callback(indata, frames, time_info, status):
            self._push(indata.copy().reshape(-1))

        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.samplerate,
            dtype="float32",
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self._collect()


def make_recorder(engine: str, mic: str | None):
    if engine == "sounddevice" or (engine == "auto" and not shutil.which("arecord")):
        if sd is None:
            raise RuntimeError(
                "sounddevice/PortAudio is not available and `arecord` was not found. "
                "Install one of them (e.g. `sudo apt install alsa-utils` or `libportaudio2`)."
            )
        device = int(mic) if (mic or "").isdigit() else mic
        try:
            info = sd.query_devices(device, "input")
            samplerate = int(info["default_samplerate"])
        except Exception:
            samplerate = SAMPLE_RATE
        return SoundDeviceRecorder(device, samplerate)
    return ArecordRecorder(mic or "default", SAMPLE_RATE)


def run_level_meter(recorder: _BaseRecorder, stop_event: threading.Event, quiet: bool) -> None:
    if quiet or not sys.stdout.isatty():
        return
    width = 24
    while not stop_event.is_set():
        level = min(1.0, recorder.level * 8.0)
        filled = int(level * width)
        bar = "█" * filled + "░" * (width - filled)
        sys.stdout.write(f"\r  🎙️  [{bar}]")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
def load_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    print(f"⏳ Loading model '{model_name}' ({device}/{compute_type}) ...", flush=True)
    started = time.time()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as exc:
        if device == "cuda":
            print(f"⚠️  CUDA failed ({exc}); falling back to CPU int8.", flush=True)
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
        else:
            raise
    print(f"✅ Model ready in {time.time() - started:.1f}s.\n", flush=True)
    return model


def transcribe(model, audio, language, beam_size, temperature):
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        temperature=temperature,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    text = "".join(segment.text for segment in segments).strip()
    return text, info


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Press-to-talk Malayalam/English STT (faster-whisper).")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="faster-whisper model: local CT2 path or a name like medium/large-v3 "
                        "(default: the fine-tuned Malayalam model in models/faster-whisper-medium-ml)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="inference device (default: cuda)")
    p.add_argument("--compute-type", default="int8_float16",
                   help="float16 / int8_float16 / int8 (default: int8_float16, ~2GB VRAM for large-v3)")
    p.add_argument("--language", default=None,
                   help="force a language code (ml/en) or omit for auto-detect")
    p.add_argument("--beam-size", type=int, default=5, help="beam size; 1 = fastest, 5 = most accurate (default: 5)")
    p.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (default: 0.0)")
    p.add_argument("--engine", default="auto", choices=["auto", "arecord", "sounddevice"],
                   help="microphone backend (default: auto)")
    p.add_argument("--mic", default=None,
                   help="input device: ALSA name for arecord (default: default) or index for sounddevice")
    p.add_argument("--save-audio", metavar="FILE", default=None, help="also save the recorded clip as WAV")
    p.add_argument("--list-devices", action="store_true", help="list microphones and exit")
    p.add_argument("--quiet-meter", action="store_true", help="disable the live input level meter")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.list_devices:
        list_devices()
        return 0

    if args.model == DEFAULT_MODEL and not os.path.isdir(DEFAULT_MODEL):
        print("❌ Fine-tuned Malayalam model not found:")
        print(f"   {DEFAULT_MODEL}")
        print("   Build it once with:  ./download_model.sh")
        print("   (or pass --model medium / --model large-v3 to use a generic model)")
        return 1

    print("=" * 60)
    print("  🗣️  Speak something (Malayalam or English)")
    print("  Press ENTER to stop recording and transcribe.")
    print("=" * 60)

    try:
        recorder = make_recorder(args.engine, args.mic)
        recorder.start()
    except Exception as exc:
        print(f"\n❌ Could not open the microphone: {exc}")
        print("   Try `--list-devices` and pass `--mic <device>`.")
        return 1

    print("🔴 Recording ... speak now.\n")

    stop_event = threading.Event()
    meter = threading.Thread(
        target=run_level_meter, args=(recorder, stop_event, args.quiet_meter), daemon=True
    )
    meter.start()

    try:
        input()  # blocks until the user presses ENTER
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        stop_event.set()
        meter.join(timeout=0.5)
        audio = recorder.stop()

    src_rate = getattr(recorder, "samplerate", SAMPLE_RATE)
    duration = audio.shape[0] / src_rate if src_rate else 0.0
    print(f"⏹️  Stopped. Captured {duration:.1f}s of audio.")

    if audio.size == 0 or duration < 0.2:
        print("⚠️  No audio captured — check your microphone and try again.")
        return 1

    if src_rate != SAMPLE_RATE:
        audio = resample(audio, src_rate, SAMPLE_RATE)
    audio = trim_silence(audio, SAMPLE_RATE)

    if args.save_audio:
        try:
            import soundfile as sf

            sf.write(args.save_audio, audio, SAMPLE_RATE)
            print(f"💾 Saved clip to {args.save_audio}")
        except Exception as exc:
            print(f"⚠️  Could not save audio: {exc}")

    model = load_model(args.model, args.device, args.compute_type)

    print("🧠 Transcribing ...", flush=True)
    started = time.time()
    text, info = transcribe(model, audio, args.language, args.beam_size, args.temperature)
    elapsed = time.time() - started
    rtf = (duration / elapsed) if elapsed > 0 else 0.0

    print("\n" + "─" * 60)
    print(f"📝 {text}" if text else "📝 (nothing recognised)")
    print("─" * 60)
    print(
        f"   language: {info.language} ({info.language_probability:.0%}) | "
        f"transcribed in {elapsed:.2f}s"
        + (f" | {rtf:.1f}x realtime" if rtf else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
