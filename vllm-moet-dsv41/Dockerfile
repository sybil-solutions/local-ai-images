# ghcr.io/sybil-solutions/vllm-moet-dsv41: DeepSeek-V4.1-Flash(-1017) on 4x RTX PRO 6000 Blackwell (sm_120), TP4.
#
# kacper-daftcode/vLLM-Moet @ 5d25cff (Apache-2.0), Dockerfile.sm120-dsv41-nightly = the "-r13" image: vLLM nightly
# df3bd094 + Dockerfile steps 1-28 (MXFP8 GEMV, FP4 KV record, MXFP4 indexer, b12x MoE at TP4 with the fused decode
# kernel, sparse-MLA gather loads, MXFP8 lm_heads, PCIe P2P all-reduce in NCCL's order, DSpark k=7 support).
# Below: that file from its first stage on, unchanged except that (a) the vLLM-Moet source comes from the pinned
# GitHub archive instead of the build context (the two COPY lines read --from=src) and (b) FROM names the base digest
# directly (release-image.yml requires a literal @sha256 pin). Then one layer of ours, engram_disk (MIT): with
# VLLM_MOET_ENGRAM_DISK=1 the 183 GiB Engram tables are read row by row from the checkpoint on NVMe at each step
# instead of pinned host RAM (bit-exact; upstream needs ~330 GB of host shm, this runs on a 128 GB host).
#
# No GPU needed to build (every extension sets its CUDA arch). Serve: see README.md.

FROM vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f@sha256:b91d90a63b75d00e780287e4236123a281ee3d29122c864a7044a19a680fdf9d AS src
ADD --checksum=sha256:4fcc0b374e040370ddbcb7870899c6ac4a8c047b9098775c4e57bce0e5927f94 https://github.com/kacper-daftcode/vLLM-Moet/archive/5d25cff77495e9a260420c3ec6ee35a96bb2056d.tar.gz /tmp/moet.tgz
RUN mkdir -p /src && tar -xzf /tmp/moet.tgz -C /src --strip-components=1 && rm /tmp/moet.tgz

# ---------------------------------------------------------------- deep_gemm
FROM vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f@sha256:b91d90a63b75d00e780287e4236123a281ee3d29122c864a7044a19a680fdf9d AS deepgemm-builder
# vLLM's pin (cmake/external_projects/deepgemm.cmake / tools/install_deepgemm.sh): the vllm-project fork of
# DeepGEMM, 727f8470 (one commit after a7a92773 of vllm#61744 - upstream 2.8.1 + the SM120 port, on torch's stable
# ABI - writing FP8 einsum output scale factors in the 128x4 tiled layout); needs the cutlass and deep_jit submodules.
ARG DEEPGEMM_REPO=https://github.com/vllm-project/DeepGEMM.git
ARG DEEPGEMM_REF=727f847082d2e7946f70dc02b94949db206310d3
# libdw-dev: deep_jit's exception.hpp includes elfutils/libdwfl.h (compile-time only; the
# vendored _C_extension links neither libdw nor libelf).
RUN apt-get update && apt-get install -y --no-install-recommends git libdw-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages pybind11 wheel
RUN git clone "$DEEPGEMM_REPO" /deepgemm \
    && cd /deepgemm && git checkout "$DEEPGEMM_REF" \
    && git submodule update --init --depth 1 third-party/cutlass third-party/deep_jit
COPY --from=src /src/tools/dsv41_sm120/patch_deepgemm.py /tmp/patch_deepgemm.py
RUN python3 /tmp/patch_deepgemm.py /deepgemm
RUN mkdir -p /tmp/linklibs \
    && ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libnvrtc.so.13 /tmp/linklibs/libnvrtc.so \
    && ln -sf /usr/local/cuda/lib64/libcudart.so.13 /tmp/linklibs/libcudart.so \
    && cd /deepgemm \
    && CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include \
       LIBRARY_PATH=/tmp/linklibs \
       python3 setup.py build_ext --inplace \
    && ls -la /deepgemm/deep_gemm/_C_extension.abi3.so

# ---------------------------------------------------------------- final image
FROM vllm/vllm-openai:nightly-df3bd094aeea829f27baa0a5282830464f601d9f@sha256:b91d90a63b75d00e780287e4236123a281ee3d29122c864a7044a19a680fdf9d

# 1. DeepGEMM _C_extension with the SM120 128-row paged MQA path and the SM120 per-group BLOCK_M (the vendored
#    Python and JIT device sources are the same commit, so they stay as shipped).
COPY --from=deepgemm-builder /deepgemm/deep_gemm/_C_extension.abi3.so \
     /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/_C_extension.abi3.so

# 2. The base's FlashInfer must carry flashinfer#5197 (the DSv4.1 dual cache) and the sm120f AOT modules; python,
#    cubin, the jit-cache shim and the providers share one version (FlashInfer refuses mixed versions).
RUN python3 -c "import importlib.metadata as md, inspect, flashinfer.mla as m; from flashinfer.jit import env; from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4 as f; assert hasattr(m, 'dsv41_fp4_quantize_pack_sparse_mla_cache'); assert 'fp8_dsv41_fp4_ca' in str(inspect.signature(f).parameters['kv_cache_format'].annotation); p = {x.provider_id: x for x in env.FLASHINFER_AOT_PROVIDERS}; assert {'sparse_mla_sm120', 'mxfp8_gemm_cutlass_sm120'} <= p['sm120f'].modules, p; v = {md.version(d).split('+')[0] for d in ('flashinfer-python', 'flashinfer-cubin', 'flashinfer-jit-cache', 'flashinfer-jit-cache-sm120f')}; assert len(v) == 1, v; print('flashinfer', *v, 'aot', sorted(p), len(p['sm120f'].modules), 'sm120f modules')" \
    && python3 -c "import torch, vllm.third_party.deep_gemm as dg; s = str(torch.ops.deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout.default._schema); assert 'num_groups' in s, s; print('deep_gemm', dg.__version__, s)"

# 3. The vLLM-side patches (see header). patch_vllm_indexer_sm120.py (varlen indexer / adaptive
#    verification) stays NOT applied: measured as a net loss on 2026-09-18 and 2026-09-22.
COPY --from=src /src/tools/dsv41_sm120/ /opt/vllm-moet/dsv41_sm120/
ENV TORCH_EXTENSIONS_DIR=/opt/vllm-moet/torch_extensions
RUN python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_mxfp8_gemv.py \
    && python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_wo_a_sm120.py \
    && python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_moe_glue_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/sm120_gemv'); import mxfp8_gemv_sm120 as g; e = g._ext(); assert hasattr(e, 'mxfp8_gemv_grouped'); print('sm120 gemv extension built')" \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_mxfp8_gemv_sm120/*.so

# 4. Reasoning-effort tiers (header).
ARG VLLM_MOET_REASONING_EFFORT=1
RUN if [ "$VLLM_MOET_REASONING_EFFORT" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_reasoning_effort.py \
    && python3 /opt/vllm-moet/dsv41_sm120/test_reasoning_effort_encoding.py; fi

# 5. nvfp4_ds_mla on sm_120 through FlashInfer's DSv4.1 dual cache + the MXFP4 indexer cache gate (header). On a
#    vLLM that writes the V4.1 records interleaved (vllm#61371) FlashInfer first learns kv_scale_placement and its
#    sparse_mla_sm120 module is rebuilt into data/aot (flashinfer_interleaved/, ~1 min).
#    VLLM_MOET_V41_KV_RECORDS names the base's placement for the image label the launcher reads (the KV offload's disk
#    tier keeps the two placements apart); the build fails if it is not what this vLLM writes.
ARG VLLM_MOET_NVFP4_KV=1
ARG VLLM_MOET_V41_KV_RECORDS=interleaved
RUN records=footer; grep -q "The V4.1 records are interleaved" /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/common/ops/cache_utils.py && records=interleaved; \
    [ "$records" = "$VLLM_MOET_V41_KV_RECORDS" ] || { echo "VLLM_MOET_V41_KV_RECORDS=$VLLM_MOET_V41_KV_RECORDS but this vLLM writes $records V4.1 records" >&2; exit 1; }; \
    if [ "$VLLM_MOET_NVFP4_KV" = "1" ]; then \
      if [ "$records" = "interleaved" ]; then \
        python3 /opt/vllm-moet/dsv41_sm120/flashinfer_interleaved/patch_flashinfer_dsv41_interleaved.py --build \
        && python3 -c "import inspect; from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4 as f; from flashinfer.jit.env import get_aot_artifacts; assert 'kv_scale_placement' in inspect.signature(f).parameters; a = get_aot_artifacts('sparse_mla_sm120'); assert a and a[0].provider_id == 'flashinfer-python', a; print('flashinfer reads interleaved V4.1 rows:', a[0].path)" \
        || exit 1; \
      fi; \
      python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_indexer_fp4_sm120.py \
      && python3 /opt/vllm-moet/dsv41_sm120/nvfp4_kv/patch_vllm_nvfp4_sm120_upstream.py; fi

# 6. Single-copy KV offload of the TP-replicated V4.1 KV (header).
ARG VLLM_MOET_OFFLOAD_DEDUP=1
RUN if [ "$VLLM_MOET_OFFLOAD_DEDUP" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_offload_replicated_mla.py \
    && python3 /opt/vllm-moet/dsv41_sm120/test_offload_replicated_mla.py; fi

# 8. MoE input quantization + DeepGEMM grouped-layout permutation in one kernel at decode shapes (header);
#    the extension is compiled here so nothing JIT-compiles at boot. GPU tests: tools/dsv41_sm120/README.md.
RUN python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_moe_quant_scatter_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/moe_quant_scatter'); import moe_quant_scatter_sm120 as q; e = q._ext(); assert hasattr(e, 'moe_quant_scatter'); print('sm120 moe quant+scatter extension built')" \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_moe_quant_scatter_sm120/*.so

# 10. The mHC boundary off the decode critical path (header); the extension is compiled here so nothing
#     JIT-compiles at boot. GPU tests: tools/dsv41_sm120/README.md. VLLM_MOET_DSPARK_MHC_STREAM=0 leaves the
#     drafter's layers without the stream (the served wiring; an image for the KV-capacity A/B of window 9).
ARG VLLM_MOET_MHC_OVERLAP=1
ARG VLLM_MOET_DSPARK_MHC_STREAM=1
RUN if [ "$VLLM_MOET_MHC_OVERLAP" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_mhc_overlap_sm120.py \
    && python3 -c "import vllm.models.deepseek_v41.nvidia.ops.mhc as h; assert hasattr(h, '_moet_mhc_shifted_post_pre_overlap'); print('mhc overlap patched')" \
    && if [ "$VLLM_MOET_DSPARK_MHC_STREAM" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_dspark_mhc_stream_sm120.py \
       && python3 -c "import vllm.models.deepseek_v41.nvidia.dspark as d; import inspect; s = inspect.getsource(d.DSparkDeepseekV4Model); assert 'mhc_stream=mhc_stream' in s and 'tensor_model_parallel_all_reduce(hidden_states)' in s; print('drafter mhc stream patched')"; fi \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/mhc_overlap'); import mhc_post_norm_sm120 as m; e = m._ext(); assert hasattr(e, 'mhc_post_norm') and hasattr(e, 'mhc_proj'); print('sm120 mhc overlap extension built')" \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_mhc_post_norm_sm120/*.so; fi

# 11. The small-M BF16 GEMMs with cuBLAS's bits (header); the extension is compiled here so nothing JIT-compiles at
#     boot. GPU tests: tools/dsv41_sm120/README.md.
ARG VLLM_MOET_BF16_GEMM=1
RUN if [ "$VLLM_MOET_BF16_GEMM" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_bf16_gemm_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/bf16_gemm'); import bf16_replica_gemm_sm120 as b; e = b._ext(); assert hasattr(e, 'chain_out') and hasattr(e, 'spread_out'); print('sm120 bf16 gemm extension built')" \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_bf16_replica_gemm_sm120/*.so; fi

# 12. Engram's wkv split by output columns (header); no extension. GPU test: tools/dsv41_sm120/README.md.
ARG VLLM_MOET_ENGRAM_WKV_TP=1
RUN if [ "$VLLM_MOET_ENGRAM_WKV_TP" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_engram_wkv_tp.py \
    && python3 -c "import vllm.models.deepseek_v41.common.engram as e; assert hasattr(e, '_moet_engram_wkv'); print('engram wkv split patched')"; fi

# 13. NCCL's per-collective serialEvent record nodes reduced to the last one in every captured graph (header); no extension.
ARG VLLM_MOET_NCCL_GRAPH_RECORDS=1
RUN if [ "$VLLM_MOET_NCCL_GRAPH_RECORDS" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/nccl_graph/patch_vllm_nccl_graph_records.py \
    && python3 -c "import vllm.v1.worker.gpu.cudagraph_utils as c, vllm.compilation.breakable_cudagraph as b, vllm.compilation.cuda_graph as w; assert all(hasattr(m, '_moet_nccl') for m in (c, b, w)); print('nccl graph records patched')"; fi

# 14. FC2's padding rows out of L2 at decode token counts (header); the extension is compiled here so nothing
#     JIT-compiles at boot. GPU tests: tools/dsv41_sm120/README.md.
ARG VLLM_MOET_MOE_FC2_DISCARD=1
RUN if [ "$VLLM_MOET_MOE_FC2_DISCARD" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_moe_fc2_discard_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/moe_l2'); import moe_l2_discard_sm120 as l; e = l._ext(); assert hasattr(e, 'discard_padding_rows'); print('sm120 moe l2 discard extension built')" \
    && python3 -c "import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as d; import inspect; assert hasattr(d, '_moet_drop_padding_rows') and hasattr(d, '_moet_l2_side_stream') and 'wait_stream(_moet_l2_side)' in inspect.getsource(d.DeepGemmFP4Experts.apply); print('moe fc2 discard patched')" \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_moe_l2_discard_sm120/*.so; fi

# 15. Dual-batch overlap of the prefill's TP all-reduces (header); no extension, off unless `vllm serve` gets
#     --enable-dbo (docker/sm120/run-dsv41.sh PREFILL_DBO=1). Needs step 10's fused all-reduce. GPU test:
#     tools/dsv41_sm120/README.md (prefill_dbo/test_prefill_dbo_tp4.py). Not ported to vLLM 3f1bcfd82b / 49cb2bbf yet (header).
ARG VLLM_MOET_PREFILL_DBO=0
RUN if [ "$VLLM_MOET_PREFILL_DBO" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_prefill_dbo_sm120.py \
    && python3 -c "import vllm.v1.worker.moet_prefill_dbo as d, vllm.v1.worker.gpu.ubatch_utils as u, vllm.v1.worker.gpu.dp_utils as p, vllm.models.deepseek_v41.nvidia.model_state as s; import inspect; assert hasattr(u, '_moet_dbo') and 'check_ubatch_thresholds(parallel_config, num_tokens, uniform_decode)' in inspect.getsource(p.dispatch_cg_and_sync_dp) and '_moet_num_ubatches' in inspect.getsource(s.DeepseekV41ModelState); print('prefill dbo patched')"; fi

# 16. (backport) vllm#61647: the indexer's TP row-sharding threshold as AttentionConfig.indexer_tp_shard_min_rows
#     (header); default unchanged, docker/sm120/run-dsv41.sh INDEXER_TP_SHARD_MIN_ROWS sets it.
ARG VLLM_MOET_INDEXER_SHARD_ROWS=1
RUN if [ "$VLLM_MOET_INDEXER_SHARD_ROWS" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_indexer_shard_rows.py \
    && python3 -c "from vllm.config.attention import AttentionConfig as A; import inspect, vllm.v1.attention.backends.mla.indexer as i; assert A().indexer_tp_shard_min_rows == 65536 and 'min_rows' in inspect.signature(i.balanced_prefill_row_shard).parameters; print('indexer shard rows configurable')"; fi

# 17. b12x W4A8 MoE through `--moe-backend b12x` at 576 intermediate rows per rank (TP4): FlashInfer's TPMoEScratchPlan.bind
#     attaches the compact N64 launchers the canonical preparation path attaches (patch_flashinfer_b12x_compact_bind.py);
#     without it the first decode-sized call stops with "compact W4A8 execution requires its prepared native launchers".
#     The default MoE backend (DeepGEMM, steps 3/8/14) is unchanged.
ARG VLLM_MOET_B12X_COMPACT_BIND=1
RUN if [ "$VLLM_MOET_B12X_COMPACT_BIND" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_flashinfer_b12x_compact_bind.py \
    && grep -q "_moet_compact_launches(self)" /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/fused_moe/_impl.py \
    && echo "b12x compact bind patched"; fi

# 18. The DSpark drafter's lm_head at decode row counts on the MXFP8 GEMV (header); needs step 3's patched linear kernel.
ARG VLLM_MOET_DRAFTER_LM_HEAD_MXFP8=1
RUN if [ "$VLLM_MOET_DRAFTER_LM_HEAD_MXFP8" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_dspark_lm_head_mxfp8_sm120.py \
    && grep -q "_moet_mxfp8_lm_head(self, normed)" /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py \
    && echo "drafter lm_head mxfp8 patched"; fi

# 19. The MoE router's top-6 bit-identical to the Triton dsv4_topk in a plain-launched CUDA kernel (header); the extension
#     is compiled here so nothing JIT-compiles at boot. GPU test: tools/dsv41_sm120/dsv4_topk/test_dsv4_topk_sm120.py.
ARG VLLM_MOET_DSV4_TOPK=1
RUN if [ "$VLLM_MOET_DSV4_TOPK" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_dsv4_topk_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/dsv4_topk'); import dsv4_topk_sm120 as t; e = t._ext(); assert hasattr(e, 'dsv4_topk_out'); print('sm120 dsv4 topk extension built')" \
    && grep -q "_moet_topk(gating_output)" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_dsv4_topk_sm120/*.so; fi

# 20. DSpark's Markov sampling in one launch per draft step on sm_120 (header); Triton, compiled at the first eager call.
#     GPU test: tools/dsv41_sm120/fused_markov/test_fused_markov_sm120.py.
ARG VLLM_MOET_FUSED_MARKOV=1
RUN if [ "$VLLM_MOET_FUSED_MARKOV" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_dspark_fused_markov_sm120.py \
    && python3 -c "import vllm.models.deepseek_v41.nvidia.dspark as d, vllm.models.deepseek_v41.amd.ops.fused_markov as f; assert hasattr(d.DSparkDeepseekV4ForCausalLM, 'sample_sequential') and hasattr(f, 'sample_sequential'); print('dspark fused markov patched')"; fi

# 21. The target's lm_head as an MXFP8 GEMV at decode row counts (runtime switch VLLM_MOET_TARGET_LM_HEAD_MXFP8=1, default off).
ARG VLLM_MOET_TARGET_LM_HEAD_MXFP8_PATCH=1
RUN if [ "$VLLM_MOET_TARGET_LM_HEAD_MXFP8_PATCH" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_target_lm_head_mxfp8_sm120.py \
    && python3 -c "import vllm.models.deepseek_v41.nvidia.model as m; assert hasattr(m, '_moet_target_mxfp8_lm_head'); print('target lm_head mxfp8 patched')"; fi

# 22. FC2 of b12x's compact W4A8 MoE path grouped by expert (header); the extension is compiled here.
#     GPU test: tools/dsv41_sm120/b12x_fc2/test_b12x_fc2_grouped_sm120.py.
ARG VLLM_MOET_B12X_FC2_GROUPED=1
RUN if [ "$VLLM_MOET_B12X_FC2_GROUPED" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_flashinfer_b12x_fc2_grouped_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/b12x_fc2'); import b12x_fc2_grouped_sm120 as f; e = f._ext(); assert hasattr(e, 'fc2_grouped'); print('sm120 b12x grouped FC2 extension built')" \
    && grep -q "_moet_fc2_grouped_on" /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_micro.py \
    && grep -q "_moet_fc2_grouped_on" /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/fused_moe/_preparation.py \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_b12x_fc2_grouped_sm120/*.so; fi

# 23. TP4 all-reduces of the decode step on a P2P push (runtime switch VLLM_MOET_P2P_AR=1, default off); the extension is
#     compiled here so the first serving call does not build it.
ARG VLLM_MOET_P2P_AR_PATCH=1
RUN if [ "$VLLM_MOET_P2P_AR_PATCH" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_p2p_allreduce_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/p2p_ar'); import p2p_allreduce_sm120 as m; e = m._build(); assert hasattr(e, 'all_reduce_twoshot') and hasattr(e, 'push'); print('sm120 P2P all-reduce extension built')" \
    && python3 -c "import vllm.models.deepseek_v41.nvidia.ops.mhc as h, vllm.models.deepseek_v41.nvidia.model as mm; assert hasattr(h, '_moet_p2p_all_reduce') and hasattr(h, '_moet_p2p_push_for_fused') and hasattr(h, '_moet_p2p_all_gather'); print('p2p all-reduce patched (23 + fused consumer 24 + logits gather 25)')"; fi

# 26. A deterministic DSA indexer top-k (runtime switch VLLM_MOET_DET_TOPK=1, default off); the extension is compiled here.
#     GPU test: tools/dsv41_sm120/det_topk/test_det_topk_sm120.py.
ARG VLLM_MOET_DET_TOPK_PATCH=1
RUN if [ "$VLLM_MOET_DET_TOPK_PATCH" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_vllm_det_topk_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/det_topk'); import det_topk_sm120 as t; e = t._ext(); assert hasattr(e, 'det_topk_prefill') and hasattr(e, 'det_topk_decode'); print('sm120 deterministic indexer top-k extension built')" \
    && python3 -c "import vllm.model_executor.layers.indexer_topk as a, vllm.model_executor.layers.sparse_attn_indexer as b; assert hasattr(a, '_moet_det_topk') and hasattr(b, '_moet_det_topk'); print('deterministic indexer top-k patched (decode + prefill)')" \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_det_topk_sm120/*.so; fi

# 27. The decode MoE of b12x's compact W4A8 path in one persistent kernel (runtime switch VLLM_MOET_B12X_MOE_FUSED=1, default
#     off); the extension is compiled here. GPU test: tools/dsv41_sm120/b12x_moe_fused/test_b12x_moe_fused_sm120.py.
ARG VLLM_MOET_B12X_MOE_FUSED_PATCH=1
RUN if [ "$VLLM_MOET_B12X_MOE_FUSED_PATCH" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_flashinfer_b12x_moe_fused_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/b12x_moe_fused'); import b12x_moe_fused_sm120 as f; e = f._ext(); assert hasattr(e, 'moe_fused'); print('sm120 b12x fused MoE extension built')" \
    && grep -q "_moet_moe_fused_on" /usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/moe/_shared/kernels/w4a8_compact_micro.py \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_b12x_moe_fused_sm120/*.so; fi

# 28. The sparse MLA decode's rows gathered by plain loads + the merge with its loads issued up front (runtime switch
#     VLLM_MOET_SPARSE_MLA_LDG=1, default off); the extension is compiled here against the installed FlashInfer's headers.
#     GPU test: tools/dsv41_sm120/sparse_mla_ldg/test_sparse_mla_ldg_sm120.py.
ARG VLLM_MOET_SPARSE_MLA_LDG_PATCH=1
RUN if [ "$VLLM_MOET_SPARSE_MLA_LDG_PATCH" = "1" ]; then python3 /opt/vllm-moet/dsv41_sm120/patch_flashinfer_sparse_mla_ldg_sm120.py \
    && python3 -c "import sys; sys.path.insert(0, '/opt/vllm-moet/dsv41_sm120/sparse_mla_ldg'); import sparse_mla_ldg_sm120 as m; e = m._ext(); assert hasattr(e, 'sparse_mla_decode_ldg'); print('sm120 sparse MLA load-based gather extension built')" \
    && python3 -c "import flashinfer.mla._sparse_mla_sm120._prepared as p; assert hasattr(p, '_moet_sparse_mla_ldg_on') and hasattr(p, '_moet_sparse_mla_ldg_execute_fn'); print('sparse MLA prepare/execute patched')" \
    && ls /opt/vllm-moet/torch_extensions/vllm_moet_sparse_mla_ldg_sm120/*.so; fi

# Marker the launcher reads through `docker image inspect` to pick the upstream KV plumbing.
LABEL org.opencontainers.image.title="vLLM-Moet DeepSeek-V4.1-Flash SM120 (vLLM main)" \
      org.opencontainers.image.description="vllm/vllm-openai:nightly df3bd094 (FlashInfer 0.7.2: DSv4.1 dual-cache sparse MLA, runtime dispatch, sm120f AOT modules; its reader taught the interleaved V4.1 records) + DeepGEMM 727f8470 with 128-row paged MQA logits and the per-group BLOCK_M on sm_120 + nvfp4_ds_mla on sm_120 through FlashInfer + MXFP8 decode GEMV + official reasoning-effort tiers + single-copy KV offload + fused MoE quant+scatter + mHC boundary off the critical path + cuBLAS-bit-identical small-M BF16 GEMMs + Engram wkv split over the TP ranks at decode + PDL in our decode kernels + NCCL graph event records reduced to one + FC2 padding rows out of L2 + the indexer row-sharding threshold (vllm#61647) + b12x compact W4A8 MoE launchers for --moe-backend b12x + the DSpark drafter's lm_head on the MXFP8 GEMV + the MoE router's top-6 in a plain-launched CUDA kernel bit-identical to Triton's + DSpark's Markov sampling in one launch per draft step + the target's lm_head on the MXFP8 GEMV (runtime switch) + b12x's MoE FC2 grouped by expert + a deterministic DSA indexer top-k (runtime switch) + b12x's decode MoE in one persistent kernel (runtime switch) + the sparse MLA decode's rows gathered by plain loads and its merge with hoisted loads (runtime switch)" \
      com.vllm-moet.kv-mode="upstream" \
      com.vllm-moet.flashinfer="base" \
      com.vllm-moet.decoder-replay="upstream" \
      com.vllm-moet.query-quant-gate="upstream" \
      com.vllm-moet.reasoning-effort="${VLLM_MOET_REASONING_EFFORT}" \
      com.vllm-moet.nvfp4-kv="${VLLM_MOET_NVFP4_KV}" \
      com.vllm-moet.offload-dedup="${VLLM_MOET_OFFLOAD_DEDUP}" \
      com.vllm-moet.mhc-overlap="${VLLM_MOET_MHC_OVERLAP}" \
      com.vllm-moet.dspark-mhc-stream="${VLLM_MOET_DSPARK_MHC_STREAM}" \
      com.vllm-moet.bf16-gemm="${VLLM_MOET_BF16_GEMM}" \
      com.vllm-moet.engram-wkv-tp="${VLLM_MOET_ENGRAM_WKV_TP}" \
      com.vllm-moet.nccl-graph-records="${VLLM_MOET_NCCL_GRAPH_RECORDS}" \
      com.vllm-moet.moe-fc2-discard="${VLLM_MOET_MOE_FC2_DISCARD}" \
      com.vllm-moet.prefill-dbo="${VLLM_MOET_PREFILL_DBO}" \
      com.vllm-moet.indexer-shard-rows="${VLLM_MOET_INDEXER_SHARD_ROWS}" \
      com.vllm-moet.b12x-compact-bind="${VLLM_MOET_B12X_COMPACT_BIND}" \
      com.vllm-moet.drafter-lm-head-mxfp8="${VLLM_MOET_DRAFTER_LM_HEAD_MXFP8}" \
      com.vllm-moet.dsv4-topk="${VLLM_MOET_DSV4_TOPK}" \
      com.vllm-moet.fused-markov="${VLLM_MOET_FUSED_MARKOV}" \
      com.vllm-moet.b12x-fc2-grouped="${VLLM_MOET_B12X_FC2_GROUPED}" \
      com.vllm-moet.det-topk="${VLLM_MOET_DET_TOPK_PATCH}" \
      com.vllm-moet.b12x-moe-fused="${VLLM_MOET_B12X_MOE_FUSED_PATCH}" \
      com.vllm-moet.sparse-mla-ldg="${VLLM_MOET_SPARSE_MLA_LDG_PATCH}" \
      com.vllm-moet.v41-kv-records="${VLLM_MOET_V41_KV_RECORDS}"

# ---------------------------------------------------------------- engram_disk (ours)
COPY --from=src /src/LICENSE /src/NOTICE /opt/vllm-moet/
COPY engram_disk.c patch_vllm_engram_disk.py test_engram_disk.py /opt/vllm-moet/engram_disk/
RUN gcc -O2 -fPIC -shared -pthread -o /opt/vllm-moet/engram_disk/libengram_disk.so /opt/vllm-moet/engram_disk/engram_disk.c \
 && python3 /opt/vllm-moet/engram_disk/patch_vllm_engram_disk.py "$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')" \
 && python3 -c "import vllm.models.deepseek_v41.common.engram as e; assert hasattr(e, '_MoetEngramDisk'); import vllm.model_executor.model_loader.weight_utils as w; assert hasattr(w, '_moet_should_skip_weight'); print('engram disk ok')"
ENV VLLM_MOET_ENGRAM_DISK=1
LABEL com.vllm-moet.engram-disk="1" \
      com.vllm-moet.source="https://github.com/kacper-daftcode/vLLM-Moet/tree/5d25cff77495e9a260420c3ec6ee35a96bb2056d" \
      org.opencontainers.image.source="https://github.com/sybil-solutions/local-ai-images" \
      org.opencontainers.image.description="vLLM-Moet r13 (DeepSeek-V4.1-Flash, sm_120, TP4) + Engram tables read from NVMe" \
      ai.omarchy.engine="vllm" \
      ai.omarchy.hardware="rtx-pro-6000-blackwell-96gb" \
      ai.omarchy.cuda.archs="12.0" \
      ai.omarchy.model.repo="deepseek-ai/DeepSeek-V4.1-Flash-1017" \
      ai.omarchy.model.mount="/models"
