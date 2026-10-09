#!/bin/bash
set -e

# Target file setup
MODEL_DIR="models"
MODEL_FILE="${MODEL_DIR}/ggml-model-q4_0.bin"
MIN_FILE_SIZE_BYTES=10000000 # 10MB threshold

mkdir -p "$MODEL_DIR"

# Clean up corrupted/truncated file if it exists and is too small
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -lt "$MIN_FILE_SIZE_BYTES" ]; then
        echo "⚠️  Found existing file '$MODEL_FILE' of size ${FILE_SIZE} bytes (corrupted/error page). Removing..."
        rm -f "$MODEL_FILE"
    else
        echo "✅ Valid model already exists: $MODEL_FILE (${FILE_SIZE} bytes)"
        exit 0
    fi
fi

# Download sources
PRIMARY_URL="https://huggingface.co/sujithatz/ggml-whisper-medium-ml/resolve/main/ggml-model-q4_0.bin"
FALLBACK_URL="https://huggingface.co/ukta-app/indic-whisper-ggml/resolve/main/ggml-ml-small.bin"

download_file() {
    local url="$1"
    local output="$2"
    
    echo "⬇️  Attempting download from: $url"
    if command -v aria2c &> /dev/null; then
        if [ -n "$HF_TOKEN" ]; then
            aria2c --header="Authorization: Bearer $HF_TOKEN" -x 4 -s 4 -o "$(basename "$output")" -d "$(dirname "$output")" "$url" || return 1
        else
            aria2c -x 4 -s 4 -o "$(basename "$output")" -d "$(dirname "$output")" "$url" || return 1
        fi
    else
        echo "⚠️  aria2c not found, falling back to curl..."
        if [ -n "$HF_TOKEN" ]; then
            curl --fail -H "Authorization: Bearer $HF_TOKEN" -L "$url" -o "$output" || return 1
        else
            curl --fail -L "$url" -o "$output" || return 1
        fi
    fi
}

# Try primary model first
if download_file "$PRIMARY_URL" "$MODEL_FILE"; then
    echo "Successfully downloaded primary model."
else
    echo "⚠️  Primary model (sujithatz/ggml-whisper-medium-ml) failed or is gated/restricted."
    echo "🔄 Falling back to public Malayalam GGML model (ukta-app/indic-whisper-ggml)..."
    download_file "$FALLBACK_URL" "$MODEL_FILE"
fi

# Final size check
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_FILE" 2>/dev/null || stat -f%z "$MODEL_FILE" 2>/dev/null || echo 0)
    if [ "$FILE_SIZE" -ge "$MIN_FILE_SIZE_BYTES" ]; then
        echo "✅ Download complete! Model saved to $MODEL_FILE (${FILE_SIZE} bytes)."
        exit 0
    fi
fi

echo "❌ Error: Model download failed or produced invalid file size."
rm -f "$MODEL_FILE"
exit 1
