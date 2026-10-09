#!/usr/bin/env python3
"""
convert_model.py — convert a Hugging Face Whisper checkpoint to CTranslate2,
the format used by faster-whisper.

Default source is `thennal/whisper-medium-ml`, which is the exact fine-tuned
Malayalam model that the old `sujithatz/ggml-whisper-medium-ml` ggml file was
converted from. Same weights, same Malayalam accuracy — just in the format
faster-whisper needs.

The converted int8_float16 model uses ~1.2 GB of VRAM (comfortable on a 6 GB GPU).

Usage:
    python convert_model.py                       # thennal/whisper-medium-ml -> models/faster-whisper-medium-ml
    python convert_model.py --quantization float16
    python convert_model.py --model openai/whisper-medium --output models/faster-whisper-medium-openai

Requires: ctranslate2, transformers, torch (CPU is fine).
"""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a HF Whisper model to CTranslate2.")
    parser.add_argument("--model", default="thennal/whisper-medium-ml",
                        help="HF repo id or local path (default: thennal/whisper-medium-ml)")
    parser.add_argument("--output", default=os.path.join("models", "faster-whisper-medium-ml"),
                        help="output directory (default: models/faster-whisper-medium-ml)")
    parser.add_argument("--quantization", default="int8_float16",
                        choices=["float32", "float16", "bfloat16", "int8", "int8_float16", "int8_float32"],
                        help="weight quantization (default: int8_float16)")
    parser.add_argument("--force", action="store_true", help="reconvert even if output exists")
    args = parser.parse_args()

    output = os.path.abspath(args.output)
    if os.path.isfile(os.path.join(output, "model.bin")) and not args.force:
        print(f"✅ Converted model already present at: {output}")
        print("   (use --force to convert again)")
        return 0

    try:
        from ctranslate2.converters import TransformersConverter
        from transformers import WhisperTokenizer
    except ImportError as exc:
        print(f"❌ Missing dependency: {exc}")
        print("   Install the conversion tools: .venv/bin/pip install -r convert-requirements.txt")
        return 1

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"🔧 Converting '{args.model}' -> '{output}' ({args.quantization}) ...", flush=True)
    print("   (first run downloads the source model, ~3 GB)", flush=True)
    converter = TransformersConverter(
        args.model,
        copy_files=["preprocessor_config.json", "generation_config.json"],
        low_cpu_mem_usage=True,
    )
    converter.convert(output, quantization=args.quantization, force=True)

    # faster-whisper loads `tokenizer.json`; make sure it exists in the output.
    tokenizer_file = os.path.join(output, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        print("🔤 Writing tokenizer.json ...", flush=True)
        WhisperTokenizer.from_pretrained(args.model).save_pretrained(output)

    print(f"✅ Done. faster-whisper model ready at: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
