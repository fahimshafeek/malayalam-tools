# syntax=docker/dockerfile:1
#
# whisper.cpp server with CUDA support.
#
# Stage 1 compiles whisper.cpp (server example) from source with GGML_CUDA=ON
# against the NVIDIA CUDA devel image. Stage 2 is a slim CUDA *runtime* image
# that only carries the binary + bundled web UI, so the final image stays small.

###############################################################################
# Stage 1 — build whisper.cpp with CUDA
###############################################################################
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

# whisper.cpp source. Pin WHISPER_CPP_REF to a release tag (e.g. v1.7.4) for
# reproducible images; leave "master" to always build the latest.
ARG WHISPER_CPP_REPO=https://github.com/ggml-org/whisper.cpp.git
ARG WHISPER_CPP_REF=master

# CUDA compute capabilities to compile kernels for (semicolon-separated list):
#   89 = Ada (RTX 40xx) | 86/80 = Ampere | 75 = Turing | 61 = Pascal
# Building for the exact GPU(s) you deploy on keeps the binary and build time
# small. Default targets the RTX 4050 Laptop (sm_89).
ARG CUDA_ARCHITECTURES=89

# Ubuntu 22.04 ships CMake 3.22, which predates CUDA 12.6 support. Pin a modern
# CMake from the official static tarball instead of the distro package.
ARG CMAKE_VERSION=3.31.6

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
        -o /tmp/cmake.tar.gz \
    && tar -xzf /tmp/cmake.tar.gz -C /opt \
    && rm /tmp/cmake.tar.gz \
    && ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cmake" /usr/local/bin/cmake \
    && ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/ctest" /usr/local/bin/ctest \
    && ln -sf "/opt/cmake-${CMAKE_VERSION}-linux-x86_64/bin/cpack" /usr/local/bin/cpack \
    && cmake --version

WORKDIR /opt
RUN git clone --depth 1 --branch "${WHISPER_CPP_REF}" "${WHISPER_CPP_REPO}" whisper.cpp

WORKDIR /opt/whisper.cpp
# CMake options:
#   BUILD_SHARED_LIBS=OFF        static libs -> self-contained server binary
#   GGML_CUDA=ON                 compile the CUDA backend
#   GGML_CUDA_NCCL=OFF           single-GPU inference needs no NCCL
#   GGML_CUDA_COMPRESSION_MODE   link compression needs CUDA >= 12.8; off here
#   GGML_NATIVE=OFF              don't bake host-only CPU instructions
#   CMAKE_CUDA_ARCHITECTURES     GPU(s) to compile kernels for (see ARG above)
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_NCCL=OFF \
        -DGGML_CUDA_COMPRESSION_MODE=none \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DWHISPER_BUILD_TESTS=OFF \
        -DWHISPER_BUILD_EXAMPLES=ON \
        -DWHISPER_BUILD_SERVER=ON \
    && cmake --build build --config Release --target whisper-server -j"$(nproc)"

###############################################################################
# Stage 2 — slim CUDA runtime image
###############################################################################
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# libgomp1: ggml OpenMP runtime. cuBLAS/cudart already ship in the CUDA image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The CUDA-enabled server binary (statically linked against whisper/ggml).
COPY --from=builder /opt/whisper.cpp/build/bin/whisper-server /app/whisper-server

# Bundled web UI served at "/" by the server.
COPY public /app/public

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/whisper-server /app/entrypoint.sh

EXPOSE 8080

# entrypoint.sh fixes up LD_LIBRARY_PATH for modern NVIDIA host drivers, then
# execs the server with the arguments below.
ENTRYPOINT ["/app/entrypoint.sh"]

# Defaults: Malayalam fine-tuned model, GPU enabled, listen on all interfaces.
# Override at runtime, e.g.:
#   docker run ... stt-whisper -m /models/ggml-model-q5_0.bin -l ml -fa
CMD ["-m", "/models/ggml-model-q4_0.bin", "--host", "0.0.0.0", "--port", "8080", "--public", "/app/public", "-t", "6", "-l", "ml"]
