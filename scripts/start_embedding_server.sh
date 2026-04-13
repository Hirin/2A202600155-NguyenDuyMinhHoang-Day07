#!/bin/bash
# Start llama-server for VietLegal-Harrier embedding.
#
# Prerequisites:
#   - llama.cpp built with llama-server binary
#   - or: brew install llama.cpp (macOS)
#   - or: pip install llama-cpp-python[server]
#
# Model will auto-download from HuggingFace on first run (~650MB for Q8_0).
#
# Usage:
#   bash scripts/start_embedding_server.sh
#   bash scripts/start_embedding_server.sh --port 9090

set -euo pipefail

PORT="${1:-8086}"

echo "🚀 Starting VietLegal-Harrier embedding server on port $PORT..."
echo "   Model: mradermacher/vietlegal-harrier-0.6b-GGUF:Q8_0"
echo "   Endpoint: http://localhost:$PORT/v1/embeddings"
echo ""

llama-server \
    -hf mradermacher/vietlegal-harrier-0.6b-GGUF:Q8_0 \
    --embedding \
    --port "$PORT" \
    --ctx-size 512 \
    --threads "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
