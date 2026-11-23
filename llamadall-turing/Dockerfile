# llamADALL (straightdot/llamADALL v1-27, a llama.cpp b11243 fork with MTP drafting) for Linux, CUDA sm_75.
# The fork's GitHub repo only holds a README; the source ships as the release asset below, verified by sha256.
#
# Portable build: no -march=native (GGML_NATIVE=OFF) and no web UI download (LLAMA_USE_PREBUILT_UI=OFF), so the only
# network input is the sha256-pinned source zip. Base images are pinned by digest.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance,
# an SBOM, and a GitHub build attestation on the digest.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266 AS build
ARG LLAMADALL_URL=https://github.com/straightdot/llamADALL/releases/download/v1-27/llamaADALL-v1-27-win-cuda13.3-x64-source.zip
ARG LLAMADALL_SHA256=7a3c1e863afef4e184e23360186da55bae3990eaa36845ebd42033b4328a9305
ARG CUDA_ARCH=75-real
ARG JOBS=8
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build curl unzip ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN curl -fsSL -o src.zip "$LLAMADALL_URL" \
    && echo "$LLAMADALL_SHA256  src.zip" | sha256sum -c - \
    && unzip -q src.zip && rm src.zip && mv llamaADALL-* llamadall
WORKDIR /src/llamadall
# The fork's BUILD.md flags (Windows), minus the MSVC-only bits, for sm_75.
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_CUDA_COMPRESSION_MODE=speed \
        -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=ON -DGGML_OPENMP=ON -DGGML_CPU_REPACK=ON \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    && cmake --build build -j"${JOBS}" --target llama-server llama-bench \
    && mkdir -p /out && cp -a build/bin/. /out/

FROM nvidia/cuda:13.0.3-runtime-ubuntu24.04@sha256:76f46f3e649824ebd3685d3d86c42b0ea62af04f59242f419f56cd5804a72efb
# curl, sha256sum and python3 for the release workflow's `llama` smoke mode; same /opt/llama layout as llamacpp-prism.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama PATH=/opt/llama:$PATH
WORKDIR /opt/llama
EXPOSE 8080
ENTRYPOINT ["/opt/llama/llama-server"]
