# Inkling-2-Small EXL3 server for the local-ai registry: github.com/0xSero/inkling2-exl3 (exllamav3 1.6.6 overlay,
# b12x fused Trellis MoE, token-sharded prefill, DSpark speculative decoding with adaptive verify, 16 x 380K).
# Same steps as that repo's Dockerfile; the engine source comes from a pinned commit instead of the build context.
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance and a GitHub build attestation.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python
RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip ninja-build ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.10.2

# Engine source: github.com/0xSero/inkling2-exl3 at a pinned commit (overlay, patches, server, scripts, locks)
ARG INKLING2_COMMIT=8ba45137011b5df7660d190e909560e05b07a2a9
RUN git init -q /src && cd /src \
    && git fetch -q --depth 1 https://github.com/0xSero/inkling2-exl3.git ${INKLING2_COMMIT} \
    && git checkout -q FETCH_HEAD && test "$(git rev-parse HEAD)" = "${INKLING2_COMMIT}" \
    && mkdir -p /opt/inkling2 && cp -r docker patches overlay bench server scripts /opt/inkling2/ \
    && echo ${INKLING2_COMMIT} > /opt/inkling2/SOURCE_COMMIT && rm -rf /src

# Python 3.12.10 venv pinned to the validated b12x serving env:
# torch 2.10.0+cu128, nvidia-cutlass-dsl[cu13] 4.8.1, cuda-python 12.9.10, apache-tvm-ffi 0.1.16 (locks in docker/)
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN uv venv /opt/venv --python 3.12.10 --seed
RUN uv pip install --no-cache -r /opt/inkling2/docker/requirements-engine.lock \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
RUN uv pip install --no-cache --no-deps -r /opt/inkling2/docker/requirements-b12x.lock

# b12x fused Trellis MoE (Local Inference Lab, Apache-2.0), from FlashInfer flashinfer/experimental/b12x at a pinned commit.
# Used from source (no FlashInfer build), imported as `b12x` through its _compat shim. The deps of FlashInfer's `b12x`
# extra are in the lock (cutlass-dsl[cu13], rich, safetensors), except its torch>=2.12 floor: the kernels are
# validated on torch 2.10 here.
ARG FLASHINFER_COMMIT=9c50996794564e1e3b2a4b0683e9aa2bcc528df8
RUN git init -q /opt/flashinfer && cd /opt/flashinfer \
    && git fetch -q --depth 1 https://github.com/flashinfer-ai/flashinfer.git ${FLASHINFER_COMMIT} \
    && git checkout -q FETCH_HEAD && test "$(git rev-parse HEAD)" = "${FLASHINFER_COMMIT}" \
    && echo ${FLASHINFER_COMMIT} > /opt/flashinfer/COMMIT && rm -rf .git

# Pinned upstream exllamav3 (turboderp-org, MIT) + Inkling-2 overlay
ARG EXLLAMAV3_REF=v1.6.6
RUN git clone -q https://github.com/turboderp-org/exllamav3 /opt/exllamav3 && git -C /opt/exllamav3 checkout -q ${EXLLAMAV3_REF}
RUN git -C /opt/exllamav3 apply /opt/inkling2/patches/upstream-modifications.patch \
    && cp -r /opt/inkling2/overlay/exllamav3/. /opt/exllamav3/exllamav3/

# Build the CUDA extension for Blackwell (sm_120) against torch 2.10.0+cu128 with the CUDA 12.8 toolkit; no GPU needed
ENV CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4
RUN cd /opt/exllamav3 && uv pip install --no-cache --no-build-isolation --no-deps .

# Serving defaults = the engine's launch_server.sh: b12x MoE on, EP2xTP2 expert layout, FP8 weight-only draft projections,
# EXL3_LOAD_ARENA=0 (needed by the b12x repack), I2_MOE_GROUP=64 for prefill-sized coop MoE calls. KV format, 380K context,
# 16 slots, per-bucket draft length (--spec-k) and the fused/skinny kernels are the code's own defaults.
ENV MODEL_DIR=/models/Inkling-2-Small-EXL3-4bpw RESULTS_DIR=/models/results I2_MOE_GROUP=64 NCCL_CUMEM_ENABLE=0 \
    I2_B12X=1 I2_EP_LAYOUT=ep2tp2 I2_DRAFT_FP8=1 I2_PC_SHARD=1 I2_B12X_PATH=/opt/flashinfer EXL3_LOAD_ARENA=0 \
    PYTHONPATH=/opt/flashinfer:/opt/flashinfer/flashinfer/experimental/b12x/_compat
EXPOSE 8000
WORKDIR /opt/inkling2
ENTRYPOINT ["/opt/inkling2/scripts/entrypoint.sh"]
CMD ["sweep"]
