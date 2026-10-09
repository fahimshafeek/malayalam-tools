# Malayalam Voice ASR

An easy-to-deploy, Dockerized solution for offline Malayalam Automatic Speech Recognition (ASR). This project makes it simple for anyone to spin up a local speech-to-text server using the power of `whisper.cpp` and a fine-tuned Malayalam model.

## 🚀 Features

*   **Easy Setup:** Get running in minutes with Docker and a simple helper script.
*   **Lightweight & Fast:** Uses the optimized `whisper.cpp` backend.
*   **CUDA Accelerated:** `whisper.cpp` is compiled from source with the CUDA backend enabled and runs on the GPU.
*   **Web Interface:** Includes a clean, simple HTML interface for uploading and transcribing audio files.
*   **Optimized Model:** Pre-configured to use a quantized (q4_0) medium model for a good balance of speed and accuracy on consumer hardware.

## 🛠️ Quick Start

### Prerequisites

*   Docker & Docker Compose
*   `curl` (or `aria2c` for faster downloads)

### Running the Project

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hxri-nxrxyxn/malayalam-voice-asr.git
    cd malayalam-voice-asr
    ```

2.  **Download the Model:**
    Run the included script to fetch the fine-tuned Malayalam model (~430MB).
    ```bash
    ./download_model.sh
    ```

3.  **Start the Server:**
    ```bash
    docker compose up -d --build
    ```

4.  **Access the GUI:**
    Open your browser and navigate to [http://localhost:8081](http://localhost:8081) (the host port maps to the server's `8080`).

### GPU / CUDA Build

The Docker image compiles `whisper.cpp` from source with the CUDA backend (`GGML_CUDA=ON`) — see `Dockerfile`. The build args in `docker-compose.yml` control it:

| Build arg            | Default  | Description                                                                 |
| -------------------- | -------- | --------------------------------------------------------------------------- |
| `CUDA_ARCHITECTURES` | `89`     | GPU compute capability (89 = Ada / RTX 40xx). Use e.g. `89;86` for multiple. |
| `WHISPER_CPP_REF`    | `master` | whisper.cpp git ref; pin a tag (e.g. `v1.7.4`) for reproducible builds.      |

The container requests the GPU via `gpus: all`, so the CUDA backend is used automatically. Confirm it at startup:

```bash
docker compose logs whisper | grep -E "use gpu|CUDA"
# whisper_init_with_params_no_state: use gpu    = 1
# ggml_cuda_init: found 1 CUDA devices:
```

> Requires an NVIDIA GPU with a recent driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

## 🖥️ API Usage (Terminal)

You can also send transcription requests directly via the terminal using `curl`:

```bash
curl http://localhost:8081/inference \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio.mp3" \
  -F "temperature=0.0" \
  -F "language=ml"
```

## 🎙️ Live Terminal Transcription

`live_transcribe.py` captures your microphone and streams a live transcript to
the terminal until you press Ctrl+C. It talks to the same server, so start the
container first.

```bash
.venv/bin/python live_transcribe.py                       # Malayalam (default)
.venv/bin/python live_transcribe.py --device plughw:2,0   # pick a mic
.venv/bin/python live_transcribe.py --list-devices
.venv/bin/python live_transcribe.py --language auto       # auto-detect
```

Requirements: `arecord` (package `alsa-utils`) and `numpy`. The script measures
your background noise at startup and adapts its speech threshold; tune it with
`--noise-multiplier` / `--energy-threshold` if it triggers on noise or misses
quiet speech.

## ❤️ Credits & Acknowledgements

This project stands on the shoulders of giants. A special thanks to:

*   **[sujithatz](https://huggingface.co/sujithatz)**: For providing the [ggml-whisper-medium-ml](https://huggingface.co/sujithatz/ggml-whisper-medium-ml) model on Hugging Face, which makes this specific implementation possible.
*   **[Thennal](https://thennal.com/)**: For their incredible work in the Malayalam computing space and for making projects like this a reality through their resources and community efforts.
*   **[ggerganov](https://github.com/ggerganov)**: For the amazing [whisper.cpp](https://github.com/ggerganov/whisper.cpp) project.
