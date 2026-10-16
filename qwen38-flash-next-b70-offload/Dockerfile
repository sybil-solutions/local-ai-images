# ghcr.io/sybil-solutions/qwen38-flash-next-b70-offload: Qwen3.8-Flash-Next EXL3 (3.05bpw_h5_ng5) on ONE Intel Arc Pro
# B70 32 GB with the routed experts on NVMe, OpenAI-compatible API (SGLang 0.5.20-xpu + exl3xpu). Serving code:
# github.com/sybil-solutions/qwen38-flash-next-b70-offload at the pinned commit below (tier plugin, nvtier.py with the
# N130 admission-controlled decode tier, Triton hyper-connection kernels, SYCL sparse attention, _moe_a8 kernels).
#   QWEN_B70_MODE=nvme16 (default) --memory 16g: RAM tier 2 GB, 8 decode-fill queues, admission control (run N130 f16):
#       8k C1 29.2 / C2 38.7 / C4 52.6 agg, 32k C1 30.6 / C2 41.3 tok/s, prefill 1,139-1,332 tok/s; decode approximate
#       (masked expert picks; paired decode-vs-prefill KL 0.055 vs 0.021 at nvme32)
#   QWEN_B70_MODE=nvme32 --memory 32g: RAM tier 7 GB, srv23 tier path (same-session control): 25.6 / 30.5 / 32.4,
#       27.3 / 31.3 tok/s, prefill 1,135-1,321 tok/s
#   `IMAGE pack-store` / `IMAGE verify-store`: build / re-check the 45.8 GB packed expert store in /nvx (CPU only,
#   checkpoint at /models). The server probes the store and O_DIRECT on /nvx before it starts.
# Device: one render node (--device /dev/dri/renderDNNN, ZE_AFFINITY_MASK=0); no --ipc host, seccomp or ptrace changes.
#
# Base = the exact image the numbers were measured on (ghcr.io/0xsero/sglang-exl3-xpu-flashnext, trellis-serve 72fd2d4,
# lmsysorg/sglang v0.5.20-xpu, torch 2.13 XPU, oneAPI 2026.0). This layer builds _moe_a8.so (spir64, the campaign's
# flags) and qsa_row_sycl.so (AOT intel_gpu_bmg_g31 + spir64) from the repo sources and installs the tier plugin files
# where the campaign bind-mounted them.
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a build attestation.
FROM ghcr.io/0xsero/sglang-exl3-xpu-flashnext@sha256:24c872759256d4b0d3b9d0419ebdf6f3547e0fdbd3891fbb9f90e32ad255c94f

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.20-xpu"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="intel-arc-pro-b70-32gb"
LABEL ai.omarchy.model.repo="turboderp/Qwen3.8-Flash-Next-exl3"
LABEL ai.omarchy.model.revision="69e33439ae950f17bcbe95c98f117d80f759ab6d"

# Serving code now lives in sybil-solutions/qwen-serve (qwen3.8-flash-next/b70-offload); the old repository is archived.
# d163a2f: Qwen4-Exp MTP draft shim + MTP MoE layer pinned in the RAM tier (NEXTN MTP works in both tier modes).
ARG QWEN_B70_REPO=https://github.com/sybil-solutions/qwen-serve
ARG QWEN_B70_COMMIT=d163a2f0dc47ab41c299eb8dca06520ef077b098
ARG QWEN_B70_DIR=qwen3.8-flash-next/b70-offload
LABEL ai.omarchy.qwen38-flash-next-b70-offload.commit="${QWEN_B70_COMMIT}"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN git init -q /tmp/qwen-serve \
 && cd /tmp/qwen-serve \
 && git fetch -q --depth 1 "${QWEN_B70_REPO}" "${QWEN_B70_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${QWEN_B70_COMMIT}" \
 && mv "/tmp/qwen-serve/${QWEN_B70_DIR}" /opt/qwen-b70 \
 && echo "${QWEN_B70_COMMIT}" > /opt/qwen-b70/COMMIT \
 && rm -rf /tmp/qwen-serve

# Kernels: _moe_a8.so (NVMe-tier MoE kernels) and the SYCL sparse attention + selection library.
RUN bash /opt/qwen-b70/kernels/moe-a8/build.sh \
 && test -s /opt/qwen-b70/kernels/moe-a8/_moe_a8.so \
 && bash /opt/qwen-b70/kernels/n107-qsa/csrc/build_sycl.sh \
 && test -s /opt/qwen-b70/kernels/n107-qsa/build/qsa_row_sycl.so

# Tier plugin files at the paths the campaign bind-mounted them over (trellis-serve is installed editable).
RUN P=/opt/trellis-serve/xpu/exl3xpu \
 && cp /opt/qwen-b70/plugin/sglang_plugin.tier.py $P/sglang_plugin.py \
 && cp /opt/qwen-b70/plugin/exl3xpu/nvtier.py $P/nvtier.py \
 && cp $P/qsa_xpu.py $P/qsa_xpu_stock.py \
 && cp /opt/qwen-b70/kernels/n107-qsa/qsa_xpu_base.py $P/qsa_xpu_base.py \
 && cp /opt/qwen-b70/kernels/n107-qsa/qsa_xpu_overlay.py $P/qsa_xpu.py \
 && chmod +x /opt/qwen-b70/serve/entrypoint.sh

ENV EXL3_MOE_LIB=/opt/qwen-b70/kernels/moe-a8/_moe_a8.so \
    EXL3_MODEL_PATH=/models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 \
    HF_HUB_OFFLINE=1 \
    NEOReadDebugKeys=1 EnableSharedSystemUsmSupport=1 EnableRecoverablePageFaults=1 \
    EXL3_MOE_SLOTS=8000 EXL3_MOE_CACHE=1 \
    EXL3_NVTIER=1 EXL3_NVTIER_STORE=/nvx/qwen_experts.bin EXL3_NVTIER_LAYERS=48 \
    EXL3_NVTIER_PRIOR=/opt/qwen-b70/serve/qwen_expert_prior.json \
    EXL3_NVTIER_SVM_PREFETCH=1 EXL3_NVTIER_STAGE=1 EXL3_NVTIER_NO_VICTIM=0 EXL3_NVTIER_ASYNC=1 \
    EXL3_NVTIER_STAGE_PROF=0 EXL3_NVTIER_PREFILL_M=5 EXL3_NVTIER_KERNEL_PREFILL_M=5 EXL3_NVTIER_PROF_EVERY=48 \
    EXL3_NVTIER_FILL_THREADS=32 \
    EXL3_HC_DIR=/opt/qwen-b70/kernels/n107-hc EXL3_HC_XPU=1 EXL3_HC_MIX=epilogue EXL3_HC_COMBINE=torch \
    EXL3_N107_QSA_DIR=/opt/qwen-b70/kernels/n107-qsa EXL3_QSA_IMPL=sycl_row EXL3_QSA_SELECT=sycl \
    EXL3_QSA_TRITON=0 EXL3_QSA_DENSE_BYTES=134217728 EXL3_QSA_ROWS=64 \
    EXL3_NGRAM_TIER=nvme SGLANG_EXL3_NGRAM_RAM_GB=1 SGLANG_EXL3_KV_BITS=5 \
    EXL3_TORCH_MEM_FRACTION=0.95 TORCHINDUCTOR_COMPILE_THREADS=1 \
    QWEN_B70_MODE=nvme16

# Every library loads next to torch the way the server loads it (CPU only), and the plugin modules import.
RUN python3 -c "import os, torch; torch.ops.load_library(os.environ['EXL3_MOE_LIB']); \
torch.ops.load_library('/opt/qwen-b70/kernels/n107-qsa/build/qsa_row_sycl.so'); \
import exl3xpu.nvtier as n; assert hasattr(n.NvTierAsync, 'step') and n.NvTierAsync.step.__name__ == '_async_step_admit'; \
print('qwen-b70 libraries load, nvtier N130 installed')"

ENTRYPOINT ["/opt/qwen-b70/serve/entrypoint.sh"]
