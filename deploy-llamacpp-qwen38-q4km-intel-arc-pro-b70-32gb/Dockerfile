# ghcr.io/sybil-solutions/deploy-llamacpp-qwen38-q4km-intel-arc-pro-b70-32gb:
# Qwen3.8-27B (Q4_K_M GGUF) on Intel Arc Pro B70 (Xe2 / Battlemage) through
# llama.cpp built for SYCL. One image serves both registry recipes:
#   - qwen38-q4km-arcb70-llamacpp-tp1  (GPU_COUNT=1, single card, 131072 ctx)
#   - qwen38-q4km-arcb70-llamacpp-tp2  (GPU_COUNT=2, tensor-parallel, 262144 ctx)
# The tensor-split / device selection is chosen at launch from GPU_COUNT, so the
# same digest covers tp1 and tp2; everything else (model, KV, fused-kernel knobs)
# is identical. This is the canonical replacement for the archived qwen38-b70.
#
# Serving code: mndodd/llama.cpp @ 4302fb59969a5d8cf9f8e5f55fdd4506d0ed2126 plus the
# steveseguin/b70-optimization-lab TP2 stack. Both patches are vendored in this
# directory and SHA-256-verified against the lab's published digests:
#   patches/tp2-full-stack.patch  sha256 f21e9b557c3d024527ac98d5f189cf7ea72fa8c38a5faf2a22ee339fd1988998
#   patches/q4k-increment.patch   sha256 0a27858525f6a402cf9c92d1b93daee0a80e2ffaef9137bf7bce784a549b58b6
#
# Base = intel/oneapi-basekit 2025.3.2-0-devel-ubuntu24.04 by digest (linux/amd64;
# the default platform the release-image workflow builds). It carries the oneAPI
# DPC++ compiler (icpx) and Level Zero that the SYCL build and runtime need.
#
# SYCL JIT build (AOT bmg_g31 needs oneAPI 2026.1.x, whose UR level-zero adapter
# does not yet enumerate devices on any publicly available GPU userspace; JIT is
# the verified-working path and reaches decode parity with the lab record,
# 51 vs 49.7 tok/s). Device code (spir64) is finalised on first launch on the B70.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance,
# an SBOM, and a GitHub build attestation on the digest.
FROM intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04@sha256:79a5e333ff6f773793d124b78047001c51cbcd53035e5100313abf2f771af95a

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.source.repo="sybil-solutions/qwen-serve"
LABEL ai.omarchy.source.commit="e21b0229727fed42e4d5f12ead5d9687b7ff9545"
LABEL ai.omarchy.engine="llamacpp"
LABEL ai.omarchy.engine.version="mndodd/llama.cpp@4302fb59969a5d8cf9f8e5f55fdd4506d0ed2126+b70-lab-tp2"
LABEL ai.omarchy.quant="q4_k_m"
LABEL ai.omarchy.hardware="intel-arc-pro-b70-32gb"
LABEL ai.omarchy.model.repo="ggml-org/Qwen3.8-27B-GGUF"
LABEL ai.omarchy.model.revision="0669b98607d47046c7c2b3f801011d54a08cfccf"
LABEL ai.omarchy.model.local_name="Qwen3.8-27B-Q4_K_M.gguf"
LABEL ai.omarchy.model.mount="/models"
LABEL ai.omarchy.model.quant="q4_k_m"
LABEL ai.omarchy.target_context_tokens="262144"
LABEL ai.omarchy.pinned_startup="llama-server --device SYCL0,SYCL1 --split-mode tensor --tensor-split 1,1 --gpu-layers 99 --flash-attn on --ctx-size 262144 --port 8010 (tp1: --device SYCL0 --ctx-size 131072)"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates python3-pip python3-venv ffmpeg \
    && pip3 install --break-system-packages "huggingface_hub[cli]" \
    && rm -rf /var/lib/apt/lists/*

# Patched source at the pinned commit; fail closed if the checkout is anything else.
ARG LLAMACPP_REPO=https://github.com/mndodd/llama.cpp
ARG LLAMACPP_COMMIT=4302fb59969a5d8cf9f8e5f55fdd4506d0ed2126
RUN git init -q /build/llama.cpp \
 && cd /build/llama.cpp \
 && git fetch -q --depth 1 "${LLAMACPP_REPO}" "${LLAMACPP_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${LLAMACPP_COMMIT}"

# Full TP2 stack + Q4K increment (vendored, digest-verified above).
COPY patches/ /patches/
RUN cd /build/llama.cpp \
 && git apply --check /patches/tp2-full-stack.patch \
 && git apply /patches/tp2-full-stack.patch \
 && git apply --check /patches/q4k-increment.patch \
 && git apply /patches/q4k-increment.patch \
 && git diff --check

# SYCL JIT build of llama-server (see header for why JIT, not AOT bmg_g31).
SHELL ["/bin/bash", "-c"]
RUN cd /build/llama.cpp \
 && . /opt/intel/oneapi/setvars.sh --force \
 && cmake -G "Unix Makefiles" -S . -B build-sycl \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=/usr/bin/cc \
        -DCMAKE_CXX_COMPILER="$(command -v icpx)" \
        -DBUILD_SHARED_LIBS=ON \
        -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL \
        -DGGML_SYCL_F16=ON -DGGML_SYCL_GRAPH=OFF -DGGML_SYCL_DNN=OFF \
        -DGGML_SYCL_HOST_MEM_FALLBACK=OFF \
        -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
 && cmake --build build-sycl --target llama-server -j"$(nproc)" \
 && test -x /build/llama.cpp/build-sycl/bin/llama-server

COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh

# Model pins (SHA-256 verified against the b70-optimization-lab repro digest; the
# entrypoint downloads @ MODEL_REV into /models and re-checks MODEL_SHA256 at
# every start).
ENV MODELS_DIR=/models \
    MODEL_REPO=ggml-org/Qwen3.8-27B-GGUF \
    MODEL_REV=0669b98607d47046c7c2b3f801011d54a08cfccf \
    MODEL_FILE=Qwen3.8-27B-Q4_K_M.gguf \
    MODEL_SHA256=31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34 \
    MTP_FILE=mtp-Qwen3.8-27B-Q4_0.gguf \
    MMPROJ_FILE=mmproj-Qwen3.8-27B-Q8_0.gguf

EXPOSE 8010
ENTRYPOINT ["/entrypoint.sh"]
