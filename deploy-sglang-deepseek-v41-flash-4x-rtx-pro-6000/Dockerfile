# ghcr.io/sybil-solutions/deploy-sglang-deepseek-v41-flash-4x-rtx-pro-6000:v1
# Owned, self-contained copy of the measured DeepSeek-V4.1-Flash launcher for
# 4 x 96 GB RTX PRO 6000 Blackwell + 128 GB DDR5 + local NVMe (single machine,
# DSpark enabled). Source: sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000
# at f27f4c398ab637f6eac214ff84ebd85283019351 (SGLang base digest-pinned; the
# adapter, SM120 runtime file, boot.py launcher, CPU adapter test and benchmark
# harness are vendored verbatim so this directory is a complete build context).
# Weights are NOT baked: boot.py downloads deepseek-ai/DeepSeek-V4.1-Flash at the
# pinned revision and verifies every SHA256/Git-blob hash before serving.
# Compute weights + model-native compressed KV stay on GPU; FP8 Engram rows use a
# bounded DDR5 cache with exact NVMe reads on misses. Build-status only here:
# TP4/EP4 GPU launch, model acceptance and quality remain separate host gates.
FROM lmsysorg/sglang@sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5
LABEL org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
LABEL ai.omarchy.source.repo="sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000"
LABEL ai.omarchy.source.commit="f27f4c398ab637f6eac214ff84ebd85283019351"
LABEL ai.omarchy.engine="sglang"
LABEL ai.omarchy.model.repo="deepseek-ai/DeepSeek-V4.1-Flash"
LABEL ai.omarchy.model.revision="fb2764a5cf321eaa5070ca8f9e892818f477c16d"
LABEL ai.omarchy.model.local_name="DeepSeek-V4.1-Flash"
LABEL ai.omarchy.model.served_name="deepseek-v4.1-flash"
LABEL ai.omarchy.model.mount="/models/DeepSeek-V4.1-Flash"
LABEL ai.omarchy.model.quant="fp4-experts+fp8-engram+bf16-components"
LABEL ai.omarchy.hardware="rtx-pro-6000"
LABEL ai.omarchy.hardware.count="4"
LABEL ai.omarchy.hardware.vram_gb="96"
LABEL ai.omarchy.hardware.arch="blackwell-sm120"
LABEL ai.omarchy.hardware.host_ram_gb="128"
LABEL ai.omarchy.hardware.storage="local-nvme"
LABEL ai.omarchy.target_context_tokens="409600"
LABEL ai.omarchy.engine_base_cuda_floor="13.0"
LABEL ai.omarchy.offload_mode="nvme"
LABEL ai.omarchy.capabilities.tools="deepseekv41"
LABEL ai.omarchy.capabilities.reasoning="deepseek-v41"
LABEL ai.omarchy.capabilities.vision="single-image"
LABEL ai.omarchy.pinned_startup="python3 -m sglang.launch_server --model-path /models/DeepSeek-V4.1-Flash --served-model-name deepseek-v4.1-flash --trust-remote-code --load-format safetensors --tp 4 --ep-size 4 --attention-backend dsv4 --moe-runner-backend flashinfer_mxfp4 --mem-fraction-static 0.85 --chunked-prefill-size 2048 --context-length 409600 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --min-free-slots-delay 1 --random-seed 0 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --enable-decoder-swa-bounded-replay --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41 --host 0.0.0.0 --port 8010"
WORKDIR /opt/dsv41
COPY adapter /opt/dsv41/adapter
RUN g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
    adapter/row_store.cpp -o adapter/librow_store.so
COPY runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
COPY boot.py /opt/dsv41/boot.py
COPY tests /opt/dsv41/tests
COPY benchmarks /opt/dsv41/benchmarks
COPY LICENSE NOTICE LICENSE.sglang /opt/dsv41/
ENV PYTHONPATH=/opt/dsv41/adapter \
    MODEL_PATH=/models/DeepSeek-V4.1-Flash \
    STATE_PATH=/state OFFLOAD_MODE=nvme DSV41_CACHE_GIB=64
EXPOSE 8010
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
    CMD ["python3", "/opt/dsv41/boot.py", "health"]
ENTRYPOINT ["python3", "-u", "/opt/dsv41/boot.py"]
CMD ["run"]
