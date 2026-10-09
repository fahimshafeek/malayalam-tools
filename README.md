# Malayalam Tools

A collection of Dockerized, offline speech tools for Malayalam — speech-to-text
and text-to-speech — built to run locally on consumer hardware.

| Tool | Description | Docs |
| ---- | ----------- | ---- |
| [`stt/`](stt/) | Malayalam **Speech-to-Text**. `whisper.cpp` + a fine-tuned Whisper Medium model, compiled with CUDA. Ships an HTTP API and a web UI. | [stt/README.md](stt/README.md) |
| [`tts/`](tts/) | Malayalam **Text-to-Speech**. Piper voices (`arjun`, `meera`) served through an OpenAI-compatible HTTP API. | [tts/README.md](tts/README.md) |

## Quick start

Each tool is self-contained and documented in its own README:

```bash
# Speech-to-text (http://localhost:8081)
cd stt
./download_model.sh
docker compose up -d --build

# Text-to-speech (http://localhost:8001)
cd ../tts
./get_voices.sh
docker compose up -d
```

## Model weights

Model binaries are **not** stored in this repository (they are large). Download
them with the helper scripts:

- `stt/download_model.sh` — fine-tuned Malayalam Whisper Medium (GGML)
- `tts/get_voices.sh` — Piper Malayalam voices

## License

See the individual tool directories. Model credits belong to their respective
authors (see each README).
