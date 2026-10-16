# ghcr.io/0xsero/glm53-flash-offload: GLM-5.3-Flash EXL3 (3.05bpw, turboderp/GLM-5.3-Flash-exl3) on ONE 24 GB
# RTX 3090 (sm_86) + host DDR4, OpenAI-compatible API on :30000. Serving code: github.com/sybil-solutions/glm-flash-lite
# at the pinned commit below (monkeypatches over stock exllamav3 1.5.1: pinned zero-copy home copy of all routed
# experts, elastic CLOCK GPU expert cache + prefill staging, AVX2 CPU tier for cold decode misses).
#   GLM53_MODE=fast  (default) campaign run G067: prefill 8k 710 / 32k 951 tok/s, decode C1 28.15 / C2 31.57 /
#                    C4 33.98 agg, KV 131,072; prefill exact (panel 1.0000 / KL 0), decode NOT exact (CPU tier: paired
#                    KL 0.0053, top-1 0.977; fix in progress); host RAM ~218 GiB
#   GLM53_MODE=exact run G066a: prefill 697 / 945, decode ~12.6, bit-exact with exllamav3; host RAM ~111 GiB
#   GLM53_MODE=nvme  55 GB NVMe mode (N119 S3b2): routed experts from a packed 117.3 GB NVMe store mounted at /nvx
#                    (O_DIRECT reads, pthread pool, no libaio) into a RAM tier sized from the container cap
#                    (--memory 55g --memory-swap 55g), GPU expert cache, AVX2 CPU lane; prefill 583 / 812, decode
#                    C1 14.85 / C2 15.62 / C4 16.27 agg, C1 at 32k 13.46; prefill exact, decode not exact (paired KL
#                    0.0047, top-1 0.986)
#   GLM53_MODE=nvme-exact  the same without the CPU lane (N119 S2a): prefill 563 / 805, decode ~8.3, bit-exact
#   `IMAGE pack-store` / `IMAGE verify-store` build / re-check the store (CPU only; checkpoint at /models, store at /nvx).
#   NVMe modes pin their threads to the measured CPU layout when the container's CPU set holds it, else derive it from
#   the CPU set; pack-store / verify-store / the server probe O_DIRECT on /nvx first (glm-flash-lite PR #3).
#   NVMe modes size from the cgroup cap: below 24 GiB a small-host budget applies (-rcs 1, margin 1.5 GiB, 128-slot
#   prefill ring, 48 readers x 1 MiB), so --memory 16g runs (969-expert RAM tier, 14.4 GiB peak); refused only under
#   15 GiB (glm-flash-lite PR #5).
#   AVX2 CPU kernels clamp the 16-byte trellis window to the 96-byte tile (out-of-bounds read -> SIGSEGV at a mapping end;
#   bit-identical; glm-flash-lite PR #7). usage.completion_tokens counts across max_rq_tokens requeues (PR #8).
#   GLM53_MODE=nvme defaults on: N135 CPU-lane forward (PR #10) + N136 decode lookahead / fast nv_pub / side-stream
#   prefetch guess / MLA block-table trim (PR #11), each settable to 0 (PR #12).
#   Decode lookahead no longer stalls after a max_rq_tokens requeue (glm-flash-lite PR #18), so GLM53_LA=0 is not needed.
#   GLM53_B70=1 (default off, nvme mode only): optional B70 expert tier; needs the companion image
#   glm53-b70-expert-server on one Arc Pro B70 and a shared tmpfs dir at /run/local-ai/shared (PR #9, campaign N137:
#   55 GiB decode C1 17.5 -> 26.5, C4 19.1 -> 33.5, decode KL 0.0034).
#
# Base = the stack the numbers were measured on: lmsysorg/sglang v0.5.20 (CUDA 13.0, torch, Triton, nvcc; same digest
# as ./sglang-exl3) + exllamav3 v1.5.1 (tag commit 958ec93) built from source for sm_86 (binaries also run on sm_89).
# Baked in: the repo's routing stats, exllamav3 MoE tune cache, the exact Triton autotune pins (empty Triton cache, no
# benchmarking), and pre-built torch extensions (expert cache, CPU tier host -mavx2 -mfma -mf16c + GPU split/combine,
# fast split, fused hyper-connection kernels, NVMe tier engines nv_tier + nv2).
# The model is NOT in the image: mount the checkpoint at /models, or mount an empty writable dir there and the
# entrypoint downloads turboderp/GLM-5.3-Flash-exl3 @ 332ab457 (branch 3.05bpw, 125.3 GB).
#
# A CPU-tier fix ships as a new digest by bumping GLM53_COMMIT (the exllamav3 layer stays cached).
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a build attestation.
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="exllamav3"
LABEL ai.omarchy.engine.version="1.5.1"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-3090-24gb"
LABEL ai.omarchy.cuda.archs="8.6"
LABEL ai.omarchy.model.repo="turboderp/GLM-5.3-Flash-exl3"
LABEL ai.omarchy.model.revision="332ab457b709b7ba30dd9a448be5de03b80a7ac9"
LABEL ai.omarchy.model.mount="/models"

ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ninja-build \
 && rm -rf /var/lib/apt/lists/*

# ExLlamaV3 v1.5.1 (MIT, turboderp), exactly the tag the measurements used. pip's exit status is lost in the pipe,
# so `pip show` is the check.
ARG EXLLAMAV3_COMMIT=958ec933361b24eb8426ec7222e5b0062a679dcd
RUN git clone https://github.com/turboderp-org/exllamav3 /opt/exllamav3-src \
 && cd /opt/exllamav3-src && git checkout -q ${EXLLAMAV3_COMMIT} \
 && test "$(git rev-parse HEAD)" = "${EXLLAMAV3_COMMIT}" \
 && MAX_JOBS=4 pip install --no-build-isolation --no-deps -v . 2>&1 | tail -3 \
 && pip install --no-deps marisa-trie \
 && pip show exllamav3 >/dev/null \
 && rm -rf /opt/exllamav3-src

# Serving code at the pinned commit; fail closed if the checkout is anything else.
ARG GLM53_REPO=https://github.com/sybil-solutions/glm-flash-lite
ARG GLM53_COMMIT=0bd10f371abdcdb8c3648453f9da834c6dd2c980
LABEL ai.omarchy.glm53-flash-offload.commit="${GLM53_COMMIT}"
RUN git init -q /opt/glm53 \
 && cd /opt/glm53 \
 && git fetch -q --depth 1 "${GLM53_REPO}" "${GLM53_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${GLM53_COMMIT}" \
 && echo "${GLM53_COMMIT}" > /opt/glm53/COMMIT \
 && rm -rf .git \
 && chmod 0755 docker/entrypoint.sh scripts/install.sh \
 && MAX_JOBS=4 scripts/install.sh /opt/glm53

ENV GLM53_MODE=fast \
    GLM53_ZC_STATS=/opt/glm53/data/stats_own_dec.json GLM53_EC_WARM=/opt/glm53/data/stats_own_dec.json \
    GLM53_TRITON_PIN=/opt/glm53/data/triton_pin_exact.json \
    EXLLAMAV3_TUNE_CACHE=/opt/glm53/data/coop_autotune_1gpu.bin TRITON_CACHE_DIR=/opt/glm53/triton_cache

EXPOSE 30000
ENTRYPOINT ["/opt/glm53/docker/entrypoint.sh"]
