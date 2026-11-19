# syntax=docker/dockerfile:1
# Basalt for the local-ai registry: github.com/jesdga95/basalt (MIT), a single-process engine and OpenAI/Anthropic
# server for Qwen3.8-Flash-Next, Qwen3.8-27B and DeepSeek-V4.1-Flash on NVIDIA Blackwell (sm_120 only), serving its
# own `.basalt` packs (jesdga/*-Basalt on Hugging Face).
# The same steps as that repository's Dockerfile at the pinned commit; the source comes from that commit instead of
# the build context, and both CUDA bases are pinned by digest.
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance and a GitHub build attestation.
# Building needs no GPU: the engine compiles for sm_120 with no device present, and nothing runs CUDA during the build.
#
# Mounts at run time:
#   /data    basalt_config.json (config version 2). HOME is /data, so the server keeps its per-model folder
#            (engine log, optional conversation disk tier) under /data/.cache/basalt
#   /models  the .basalt model files (read-only)
# The host needs a driver of the R580 branch or newer, and the container `--ulimit memlock=-1:-1` (pinned host buffers).

# ------------------------------------------------------------------ build stage
FROM nvidia/cuda:13.4.3-devel-ubuntu24.04@sha256:5626a7d0ecafe3e7c115a452f6b05f20f214f8509346196a7835b205dc5b7fd9 AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Engine source: github.com/jesdga95/basalt at a pinned commit (0.4.1 plus fixes, 2026-11-18)
ARG BASALT_COMMIT=899e820af777705cd29b35b1deb8612f24c42c5e
RUN git init -q /opt/basalt && cd /opt/basalt \
    && git fetch -q --depth 1 https://github.com/jesdga95/basalt.git ${BASALT_COMMIT} \
    && git checkout -q FETCH_HEAD && test "$(git rev-parse HEAD)" = "${BASALT_COMMIT}" \
    && echo ${BASALT_COMMIT} > /opt/basalt/SOURCE_COMMIT && rm -rf .git

WORKDIR /opt/basalt
# requirements.txt pins every server dependency by version
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)" \
    && python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------- final stage
FROM nvidia/cuda:13.4.3-runtime-ubuntu24.04@sha256:f58bd7a0701fae3a2160f22e1d451b77b349bd9ef83dad76f5529aa413c149a8

# The base carries the CUDA runtime and cuBLAS, the only CUDA libraries the engine links. python3 is what the venv's
# python points at.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/basalt

# The runtime tree of a release archive: the binaries, the server and the tools it imports, the data the engine reads
# (expert profiles), the venv, and the licences. The venv's scripts name /opt/basalt absolutely, so both stages use it.
COPY --from=builder /opt/basalt/.venv            /opt/basalt/.venv
COPY --from=builder /opt/basalt/bin              /opt/basalt/bin
COPY --from=builder /opt/basalt/serve            /opt/basalt/serve
COPY --from=builder /opt/basalt/tools            /opt/basalt/tools
COPY --from=builder /opt/basalt/data             /opt/basalt/data
COPY --from=builder /opt/basalt/requirements.txt /opt/basalt/requirements.txt
COPY --from=builder /opt/basalt/SOURCE_COMMIT    /opt/basalt/SOURCE_COMMIT
COPY --from=builder /opt/basalt/LICENSE /opt/basalt/THIRD_PARTY_LICENSES /opt/basalt/

ENV HOME=/data \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/basalt/bin:$PATH

# The served port is server.port in the config (the registry launch uses 8000). Arguments after the image name
# follow the entrypoint.
EXPOSE 8000

ENTRYPOINT ["basalt", "serve", "--config", "/data/basalt_config.json", "--host", "0.0.0.0"]
