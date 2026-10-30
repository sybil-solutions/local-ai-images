# ghcr.io/<owner>/glm53-flash-offload-multiarch: GLM-5.3-Flash EXL3 (3.05bpw, turboderp/GLM-5.3-Flash-exl3) with the
# glm53-flash-offload serving code and modes (fast, exact, nvme, nvme-exact, pack-store / verify-store; see that image's
# Dockerfile and README row), compiled for sm_86 AND sm_120, so the same image serves RTX 3090-class cards and consumer /
# workstation Blackwell (RTX 50-series, RTX PRO 6000 Blackwell). glm53-flash-offload (sm_86 only) stays published and
# unchanged for the 3090 recipes.
#
# The only build change is TORCH_CUDA_ARCH_LIST="8.6;12.0": exllamav3 v1.5.1 and glm-flash-lite's pre-built torch
# extensions (scripts/install.sh) are both compiled from source in this Dockerfile, so unlike sglang-exl3-flashnext no
# multiarch base is needed. The build fails unless every compiled extension carries SASS for sm_86 and sm_120 (the C++
# only *_host modules have none). sm_89 is not in the list: not measured.
#
# GLM53_TRITON_PIN on sm_120: the baked pin (data/triton_pin_exact.json) holds Triton autotune picks recorded on the
# 3090, and on sm_120 the FLA kernel's config list does not contain them, so the server exits at load:
#   AssertionError: triton_pin: no config of chunk_gated_delta_rule_fwd_kernel_h_blockdim64 matches
#   {'kwargs': {'BV': 32}, 'num_warps': 4, 'num_stages': 1}
# An empty GLM53_TRITON_PIN= turns the pin off (the entrypoint unsets an empty value) and Triton autotunes on the card.
# The default stays the pin, so a 3090 behaves exactly as with glm53-flash-offload; sm_120 recipes pass
# GLM53_TRITON_PIN= explicitly. Measured unpinned on one RTX PRO 6000 Blackwell (2026-10-09, this change on
# glm-flash-lite 0fbc0c5, GLM53_MODE=nvme-exact, 204,800 context): teacher-forced panel top-1 0.98700 / KL 0.003283
# against the pinned reference (1.0000 / 0; the pin's docstring gives 0.989 / 0.0034 for another pick set), decode C1
# 67.22 tok/s, needle 194,257 tokens pass.
#
# Build: pushed by .github/workflows/release-image.yml with BuildKit provenance, an SBOM, and a build attestation.
FROM lmsysorg/sglang@sha256:06e4f2ed21afde4ff513cda65070124e727ba23ccaeff7712b8c40e1097d611f

LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.engine="exllamav3"
LABEL ai.omarchy.engine.version="1.5.1"
LABEL ai.omarchy.quant="exl3"
LABEL ai.omarchy.hardware="rtx-3090-24gb;rtx-pro-6000-blackwell-96gb"
LABEL ai.omarchy.cuda.archs="8.6;12.0"
LABEL ai.omarchy.model.repo="turboderp/GLM-5.3-Flash-exl3"
LABEL ai.omarchy.model.revision="332ab457b709b7ba30dd9a448be5de03b80a7ac9"
LABEL ai.omarchy.model.mount="/models"

ARG TORCH_CUDA_ARCH_LIST="8.6;12.0"
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

# Fail closed unless exllamav3_ext and every pre-built glm-flash-lite extension carries SASS for sm_86 and sm_120
# (at 0fbc0c5: glm53_ec, glm53_k101, ft_split_fast, glm53_nv, glm53_nv2, ft_tier_cu; ft_tier_host is C++ only).
RUN set -e; n=0; \
 for f in $(find /opt/glm53/build -name '*.so' ! -name '*_host.so') \
          $(python3 -c "import torch, exllamav3_ext; print(exllamav3_ext.__file__)"); do \
   archs=$(cuobjdump --list-elf "$f" 2>/dev/null | grep -oE 'sm_[0-9]+a?' | sort -u | tr '\n' ' '); \
   echo "$(basename "$f"): $archs"; \
   for sm in sm_86 sm_120; do case " $archs" in *" $sm "*) ;; *) echo "missing $sm in $f"; exit 1;; esac; done; \
   n=$((n+1)); \
 done; test "$n" -ge 7

ENV GLM53_MODE=fast \
    GLM53_ZC_STATS=/opt/glm53/data/stats_own_dec.json GLM53_EC_WARM=/opt/glm53/data/stats_own_dec.json \
    GLM53_TRITON_PIN=/opt/glm53/data/triton_pin_exact.json \
    EXLLAMAV3_TUNE_CACHE=/opt/glm53/data/coop_autotune_1gpu.bin TRITON_CACHE_DIR=/opt/glm53/triton_cache

EXPOSE 30000
ENTRYPOINT ["/opt/glm53/docker/entrypoint.sh"]
