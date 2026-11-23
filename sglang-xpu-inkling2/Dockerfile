# ghcr.io/sybil-solutions/sglang-xpu-inkling2: Inkling-2-Mini (thinkingmachines/Inkling-2-Mini, NVFP4/MXFP8 MoE,
# vision, DSpark block drafter in mtp.safetensors) on Intel Arc Pro B70 (Xe2), 1 card or TP2/EP2 on 2 cards.
#
# Serving code: upstream SGLang at f0bf8ee5 (sgl-project/sglang, "[Fix] inkling2: Document the loop flags ...",
# the Inkling-2-Mini support commits) plus inkling2-xpu.patch from this directory: the XPU port (Triton attention
# + LSE, gather-then-attend prefill, SYCL split-KV decode and spec-verify attention, expert-grouped NVFP4 GEMV,
# XPU Triton DSpark sampler, fused verify prologue, oneCCL/vocab-gather fixes for TP2/EP2).
#
# Base: lmsysorg/sglang:v0.5.23-xpu by digest: the exact image the B70 campaign container (ink-dev) ran (torch
# 2.14.0+xpu, oneAPI icpx 2026.1, oneCCL, Level Zero). The base's own sglang is replaced by the patched tree,
# installed editable with --no-deps exactly as the campaign did; the campaign added one package (tml-renderers,
# the Inkling chat renderer from PyPI), pinned here by hash.
# Build: .github/workflows/release-image.yml (BuildKit provenance, SBOM, GitHub build attestation on the digest).
FROM lmsysorg/sglang@sha256:f5627c5ee6b094ae16e2bf38694b2c375b609aa8b5a3f4fab6af47ea461b20e9

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.engine.version="0.5.23-xpu+inkling2"
LABEL ai.omarchy.quant="nvfp4"
LABEL ai.omarchy.hardware="intel-arc-pro-b70-32gb"
LABEL ai.omarchy.model.repo="thinkingmachines/Inkling-2-Mini"
LABEL ai.omarchy.model.revision="ed40e7da625f594fc9c4990f5f66be24873c22d7"

ARG SGLANG_REPO=https://github.com/sgl-project/sglang
ARG SGLANG_COMMIT=f0bf8ee5aa6e19f408aaacd8eb79886ff3821fe1
LABEL ai.omarchy.sglang.commit="f0bf8ee5aa6e19f408aaacd8eb79886ff3821fe1"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# The base's WORKDIR /sgl-workspace holds its own sglang checkout, which a process started there imports as a
# namespace package ahead of the editable install: run from a directory without one.
WORKDIR /opt/inkling2-xpu

COPY inkling2-xpu.patch /opt/inkling2-xpu/inkling2-xpu.patch
# Upstream at the pinned commit (fail closed on anything else), then the XPU patch (fails if it does not apply).
RUN git init -q /opt/sglang && cd /opt/sglang \
 && git fetch -q --depth 1 "${SGLANG_REPO}" "${SGLANG_COMMIT}" \
 && git checkout -q --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}" \
 && git apply /opt/inkling2-xpu/inkling2-xpu.patch \
 && echo "${SGLANG_COMMIT}+inkling2-xpu.patch $(sha256sum /opt/inkling2-xpu/inkling2-xpu.patch | cut -c1-64)" > /opt/sglang/COMMIT \
 && rm -rf .git

RUN printf 'tml-renderers==0.4.2 --hash=sha256:da7cff5cca1447e2793066c9dffd127bfe518d3c6c923065be0016c677faa28a\n' > /tmp/req.txt \
 && pip install -q --no-deps --require-hashes -r /tmp/req.txt && rm /tmp/req.txt \
 && pip install -q --no-deps --no-build-isolation -e /opt/sglang/python \
 && python3 -c "import sglang; p = list(sglang.__path__); assert p and all(x.startswith('/opt/sglang/') for x in p), p; print(p)" \
 && python3 -c "import tml_renderers, sglang.kernels.ops.attention.inkling_verify; print('imports ok')"
# (the model modules import only with an XPU present: without a device, platform detection takes the CUDA path)

# SYCL kernels (decode, GEMV, grouped GEMV, verify, attn-res, router) are JIT-built by torch.utils.cpp_extension
# (icpx, AOT for bmg) on first use. Build them here so a fresh container does not compile at startup; ocloc AOT
# needs no device.
ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions TORCH_XPU_ARCH_LIST=bmg
RUN python3 - <<'PY'
import sglang.kernels.ops.attention.xpu_sycl as x
x.ink_decode_ext(); x.ink_gemv_ext(); x.ink_ggemv_ext(); x.ink_dense_ext()
for name, kw in (("ink_attn_res", {}), ("ink_router", {}), ("ink_verify", {"fast_math": False}),
                 ("ink_verify", {"fast_math": False, "defs": ("INK_GRF256",)})):
    x.load_ext(name, **kw)
print("SYCL extensions built")
PY
RUN test -n "$(ls -A /opt/torch_extensions)" && chmod -R a+rX /opt/torch_extensions

COPY entrypoint.sh /opt/inkling2-xpu/entrypoint.sh
RUN chmod 0755 /opt/inkling2-xpu/entrypoint.sh
EXPOSE 30000
ENTRYPOINT ["/opt/inkling2-xpu/entrypoint.sh"]
