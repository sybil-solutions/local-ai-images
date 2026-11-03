# ghcr.io/sybil-solutions/vllm-dsv41-h200: DeepSeek-V4.1-Flash-1017 (official MXFP4/FP8 checkpoint) on 4x H200 (sm_90), TP4.
#
# vLLM main @ 51d2a855 (wheel 0.32.1rc1.dev537, byte-identical to the one the numbers were measured on) on the v0.30.0
# image's userland, plus:
#   1. vllm#64893 at its head 382b4f475f (draft, mgoin): the "latency_specialized" decode executor for DeepSeek-V4.1-Flash
#      on Hopper TP4 (its 49 changed files under vllm/, copied over the wheel).
#   2. dsv41-h200.patch (24 files): the changes this image adds on top, all lossless (README.md has the list):
#      SSD Engram reader hooks, vllm#56697 MXFP4 Marlin padding backport, vllm#59418 small-head sparse decode backport,
#      a P2P-store fallback for the executor's NVLink all-reduce on nodes without NVSwitch (H200 NVL bridges),
#      BF16 operands for the executor's dense kernels, hybrid adaptive verification (VLLM_DSPARK_ADAPTIVE_MIN_REQS,
#      VLLM_DSPARK_ADAPTIVE_FULL_REQS),
#      the post-forward graph limit and forget_confidences without a host sync.
#   3. /opt/deepseek/ssd/disk_engram.py: Engram tables read from the checkpoint on NVMe at decode (MIT, from Tech2wild).
# Dependencies are pinned by constraints.txt (all but the two packages installed --no-deps after the wheel: humming-kernels,
# whose 0.1.19 replaces the wheel's 0.1.18 pin, and flashinfer-jit-cache); the build fails unless `pip freeze` equals
# freeze.expected.
# No GPU needed to build. Serve: README.md.

FROM vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90

ADD --checksum=sha256:5afb7e5e24863b98c910cfe11aa64b6c01876c05ba6ac19d959f8779d14b5c77 https://wheels.vllm.ai/51d2a855942907366bc15ba72d76130d0fc4346b/vllm-0.32.1rc1.dev537%2Bg51d2a8559-cp38-abi3-manylinux_2_28_x86_64.whl /tmp/whl/vllm-0.32.1rc1.dev537+g51d2a8559-cp38-abi3-manylinux_2_28_x86_64.whl
COPY constraints.txt freeze.expected /opt/dsv41-h200/
# flashinfer-cubin 0.7.x is not published (PyPI stops at 0.6.13) and deep_ep is built for torch 2.13 (unused at TP4
# without EP): both go. flashinfer-jit-cache 0.7.2.post1+cu130 comes from flashinfer.ai; humming-kernels 0.1.19 is
# installed over the wheel's 0.1.18 pin (vllm#64893 needs it).
RUN pip uninstall -y vllm flashinfer-jit-cache flashinfer-cubin deep_ep >/dev/null 2>&1; \
    pip install --no-cache-dir -c /opt/dsv41-h200/constraints.txt --extra-index-url https://download.pytorch.org/whl/cu130 \
      /tmp/whl/vllm-0.32.1rc1.dev537+g51d2a8559-cp38-abi3-manylinux_2_28_x86_64.whl \
      torch==2.14.0+cu130 torchvision==0.29.0+cu130 torchaudio==2.11.0+cu130 \
    && pip install --no-cache-dir --no-deps --extra-index-url https://flashinfer.ai/whl/cu130 flashinfer-jit-cache==0.7.2.post1 \
    && pip install --no-cache-dir --no-deps humming-kernels==0.1.19 \
    && rm -rf /tmp/whl \
    && pip freeze 2>/dev/null | grep -v '^vllm @ ' | LC_ALL=C sort > /tmp/freeze.txt \
    && LC_ALL=C sort /opt/dsv41-h200/freeze.expected | diff - /tmp/freeze.txt \
    && rm /tmp/freeze.txt

# 1. vllm#64893 head files.
ADD --checksum=sha256:33767a6ce3c8cbf6916c501746446cc2a58587365494bb067e2b3b486b94314a https://github.com/vllm-project/vllm/archive/382b4f475f2d8d7ad0adddae0f7a6e4f3313ed80.tar.gz /tmp/pr.tgz
COPY pr64893-files.txt dsv41-h200.patch /opt/dsv41-h200/
RUN set -e; D=/usr/local/lib/python3.12/dist-packages; mkdir -p /tmp/pr; tar -xzf /tmp/pr.tgz -C /tmp/pr --strip-components=1; \
    while read -r f; do mkdir -p "$D/$(dirname "$f")"; cp "/tmp/pr/$f" "$D/$f"; done < /opt/dsv41-h200/pr64893-files.txt; \
    rm -rf /tmp/pr /tmp/pr.tgz; \
    cd $D && patch -p1 --forward --batch --no-backup-if-mismatch < /opt/dsv41-h200/dsv41-h200.patch; \
    ! find $D/vllm -name '*.rej' | grep -q .; \
    grep -q 'self._confidence_probs.index_fill_(0, input_batch.idx_mapping\[:num_reqs\], 1.0)' $D/vllm/v1/worker/gpu/spec_decode/adaptive_verification.py; \
    grep -q 'VLLM_DSPARK_ADAPTIVE_MIN_REQS' $D/vllm/envs.py; \
    grep -q 'VLLM_DSPARK_ADAPTIVE_FULL_REQS' $D/vllm/envs.py

# 3. SSD Engram reader.
COPY ssd/ /opt/deepseek/ssd/

ENV HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:128,pinned_max_cached_size_mb:512 \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/opt/deepseek/ssd \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    NCCL_P2P_DISABLE=0 \
    VLLM_ALLREDUCE_USE_FLASHINFER=0 \
    VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0 \
    VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    VLLM_USE_NCCL_SYMM_MEM=0 \
    DSV41_ENGRAM_DISK=1 \
    DSV41_ENGRAM_GPU_DEQUANT=1 \
    DSV41_ENGRAM_VERIFY=40 \
    DSV41_ENGRAM_MMAP=1 \
    DSV41_ENGRAM_MMAP_SPLIT=32 \
    DSV41_ENGRAM_GPU_CACHE_ROWS=2000000 \
    DSV41_ENGRAM_WILLNEED_MAX=512 \
    DSV41_MARLIN_IALIGN=64 \
    DSV41_SMALL_HEAD_DECODE=1 \
    DSV41_SD_DENSE_BF16=1 \
    VLLM_DSPARK_ADAPTIVE_MIN_REQS=16 \
    VLLM_DSPARK_ADAPTIVE_FULL_REQS=160

# CPU import check (no GPU or driver in the build): every module the launch touches imports.
RUN CUDA_VISIBLE_DEVICES= python3 -c "import importlib; [importlib.import_module(m) for m in ['disk_engram', 'vllm.models.deepseek_v41.common.engram', 'vllm.models.deepseek_v41.nvidia.model_state', 'vllm.models.deepseek_v41.nvidia.flashmla', 'vllm.models.deepseek_v41.nvidia.ops.small_head_sparse_decode', 'vllm.models.deepseek_v41.nvidia.specialized_decode', 'vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.decode', 'vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.moe', 'vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.dense', 'vllm.models.deepseek_v41.nvidia.specialized_decode.hopper.steps', 'vllm.model_executor.layers.fused_moe.oracle.mxfp4', 'vllm.v1.worker.gpu.spec_decode.adaptive_verification']]; import torch, vllm, flashinfer, humming; print('IMPORT_OK', torch.__version__, vllm.__version__, flashinfer.__version__)"

LABEL org.opencontainers.image.title="vllm-dsv41-h200" \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash-1017 on 4x H200 TP4: vLLM main@51d2a855 + vllm#64893 latency_specialized executor + lossless Hopper patches + SSD Engram reader" \
      org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images"
