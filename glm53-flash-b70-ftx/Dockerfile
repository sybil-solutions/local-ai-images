# ghcr.io/sybil-solutions/glm53-flash-b70-ftx: GLM-5.3-Flash EXL3 (3.05bpw, 125 GB) on ONE Intel Arc Pro B70 32 GB with
# the routed experts in host RAM and the FTX CPU expert tier, OpenAI-compatible API (SGLang 0.5.20-xpu + exl3xpu).
# Serving code: github.com/sybil-solutions/glm-flash-lite b70/ at the pinned commit below (GLM tier plugin, post-import
# hooks, FTX tier, sglang GLM5-Next/DSA file replacements, the GLM MoE kernel with the FTX id -1 skip).
#   Measured (run G262-I0s800c1k, omarchy, 2026-10-17; kit sweep.py v1, 262,144 context, 270,336-token fp8 KV, 800 VRAM
#   expert slots, FTX on CPUs 4-23, --memory 260g): prefill 8k 189 / 16k 174 / 32k 147 / 64k 112 tok/s; decode C1 17.5,
#   C2 21.5 agg, C4 24.5 agg tok/s. Host RAM: cgroup peak 144.3 GB plus ~114.6 GB of experts in USM host memory that
#   the xe driver does not charge to the cgroup -> plan ~260 GB free.
# Device: one render node (--device /dev/dri/renderDNNN, ZE_AFFINITY_MASK=0); --ulimit memlock=-1, --shm-size 8g; no
# --ipc host, seccomp or ptrace changes. Weights mounted read-only at /models/turboderp-GLM-5.3-Flash-exl3-3.05bpw;
# nothing else is mounted (the entrypoint adds the kpool_aux tensors to the index through a symlink view in /tmp).
#
# Base = the exact image the campaign ran (with --entrypoint python3 and bind mounts, now baked in below):
# qwen38-flash-next-b70-offload (qwen38-flash-next-b70-offload 2f108fb, trellis-serve 72fd2d4, lmsysorg/sglang
# v0.5.20-xpu, torch 2.13 XPU, oneAPI 2026.0). Its ENV is inherited unchanged, as in the campaign container.
# This layer builds _moe_glm_ftx.so, _ftx.so and the FTX host extension CPU-only with the image's icpx; the three are
# byte-identical to the measured binaries when built in this base (checked 2026-10-17).
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a build attestation.
FROM ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload@sha256:d30df691d6ab890b95f7c6179e4b677a0d759018e39b20e93bf0efdb8780450c

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20-xpu"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="intel-arc-pro-b70-32gb"
LABEL ai.omarchy.model.repo="turboderp/GLM-5.3-Flash-exl3"
LABEL ai.omarchy.model.revision="332ab457b709b7ba30dd9a448be5de03b80a7ac9"

ARG GLM_REPO=https://github.com/sybil-solutions/glm-flash-lite
ARG GLM_COMMIT=d0f956580e50efa43e6cdd84115ee868ef284c17
LABEL ai.omarchy.glm-flash-lite.commit="${GLM_COMMIT}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN git init -q /tmp/glm-flash-lite \
 && cd /tmp/glm-flash-lite \
 && git fetch -q --depth 1 "${GLM_REPO}" "${GLM_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${GLM_COMMIT}" \
 && mv /tmp/glm-flash-lite/b70 /opt/glm-b70-src \
 && echo "${GLM_COMMIT}" > /opt/glm-b70-src/COMMIT \
 && rm -rf /tmp/glm-flash-lite

# Overlay, plugin files and sglang replacements at the paths the campaign bind-mounted them; kernels + FTX host
# extension built here (CPU only): /moe/_moe_n128.so, /opt/glmopt/ftx/_ftx.so, /opt/glmopt-build/ct/ft_tier_host.so.
RUN bash /opt/glm-b70-src/install.sh \
 && test -s /moe/_moe_n128.so \
 && test -s /opt/glmopt/ftx/_ftx.so \
 && test -s /opt/glmopt-build/ct/ft_tier_host.so \
 && cp /opt/glm-b70-src/COMMIT /opt/glm-b70/COMMIT \
 && sha256sum /moe/_moe_n128.so /opt/glmopt/ftx/_ftx.so /opt/glmopt-build/ct/ft_tier_host.so | tee /opt/glm-b70/BINARIES.sha256

# The run's environment (serve2.sh -e flags + arm.sh EXTRA_ENV2). EXL3_NVTIER_PRIOR is set empty on purpose (the base
# points it at the Qwen prior). FTX_CPUS=auto: one FTX worker per physical core of the cpuset (4-23 on the measured host).
ENV EXL3_MODEL_PATH=/models/turboderp-GLM-5.3-Flash-exl3-3.05bpw \
    EXL3_MOE_LIB=/moe/_moe_n128.so EXL3_MOE_GU=8 EXL3_MOE_DN=4 \
    EXL3_MOE_SLOTS=800 EXL3_MOE_CACHE=1 \
    EXL3_TIER_H=4096 EXL3_TIER_I=2048 EXL3_TIER_E=288 EXL3_TIER_DENSE=3 EXL3_TIER_L=42 \
    EXL3_NVTIER=0 EXL3_NVTIER_PRIOR= EXL3_QSA_XPU=0 EXL3_NGRAM_HOST=0 EXL3_HC_XPU=0 \
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0 \
    EXL3_TORCH_MEM_FRACTION=0.95 TORCHINDUCTOR_COMPILE_THREADS=1 TRITON_CACHE_DIR=/root/.triton-cache \
    HF_HUB_OFFLINE=1 NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=1 EnableRecoverablePageFaults=1 \
    GLM_MOE_PREFILL_MIN_M=256 EXL3_MOE_STAGE_MIN_M=128 FTX=1 FTX_CPUS=auto

# CPU-only check: MoE + FTX libraries load next to torch, the FTX host extension is reused (not rebuilt), the
# glm_opt hooks / exl3xpu plugin / glm5_next import.
RUN python3 /opt/glm-b70/serve/selfcheck.py

EXPOSE 30260
ENTRYPOINT ["/opt/glm-b70/serve/entrypoint.sh"]
