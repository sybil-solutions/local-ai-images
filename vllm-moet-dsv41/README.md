# vllm-moet-dsv41

`ghcr.io/sybil-solutions/vllm-moet-dsv41`: DeepSeek-V4.1-Flash-1017 on **4x RTX PRO 6000 Blackwell** (sm_120,
96 GB each, PCIe peer-to-peer), tensor parallel 4, 512K context, DSpark speculative decoding (k = 7), vision,
reasoning and tool calls. OpenAI-compatible API from `vllm serve` (the image's entrypoint).

## Stack

| component | pin | licence |
|---|---|---|
| base `vllm/vllm-openai:nightly-df3bd094…` (vLLM main 2026-10-20, torch 2.14, FlashInfer 0.7.2) | `sha256:b91d90a6…fdf9d` | Apache-2.0 |
| [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet) `Dockerfile.sm120-dsv41-nightly` ("-r21", steps 1-36; step 37 not built, as upstream) | `b0dc737ae431d9a2ab3f62ed3fc30313b00d99c1` (archive sha256 `662f927c…0820`) | Apache-2.0 (LICENSE + NOTICE in `/opt/vllm-moet/`) |
| DeepGEMM (vllm-project fork, rebuilt with vLLM-Moet's SM120 patch) | `727f847082d2e7946f70dc02b94949db206310d3` | MIT |
| `engram_disk` (this directory) | this commit | MIT |
| model (not in the image) | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170` | MIT (DeepSeek) |

The Dockerfile is vLLM-Moet's file from its first stage on, unchanged except that its source comes from the pinned
GitHub archive (`COPY --from=src`) and `FROM` names the base digest directly. Built on the reference host from this
directory, every vLLM-Moet layer was a cache hit against the image built from the vLLM-Moet checkout itself, i.e. the
same inputs.

`engram_disk`: with `VLLM_MOET_ENGRAM_DISK=1` (set in the image) the 183 GiB Engram tables are not loaded. Each TP
rank keeps its E8M0 scales pinned (1.4 GiB) and reads the FP8 rows a step needs from the safetensors file with
parallel `pread` (`VLLM_MOET_ENGRAM_DISK_THREADS`, default 16), inside the captured decode graphs (a CUDA host
function between an async copy of the row ids and the unchanged lookup kernel). Bit-exact
(`test_engram_disk.py`). Upstream holds the table in ~330 GB of host shared memory; this runs on a 128 GB host.

## Serve

The recipe (launch, measured numbers, quality gate) lives in
[local-ai-registry](https://github.com/sybil-solutions/local-ai-registry):
`registry/launches/vllm-moet-deepseek-v4.1-flash-1017-512k-tp4-r21.json`. Speed switches as measured there (vLLM-Moet's
Fraser-bench configuration): `VLLM_MOET_B12X_MOE_FUSED=1 VLLM_MOET_SPARSE_MLA_LDG=1 VLLM_MOET_TARGET_LM_HEAD_MXFP8=1
VLLM_MOET_P2P_AR=1 VLLM_MOET_P2P_AR_FUSED_MAX_ROWS=8 VLLM_MOET_P2P_AG=0 VLLM_MOET_DFLASH_PREP_GROUPS=1
VLLM_MOET_CANDIDATE_DET_TOPK=1 VLLM_MOET_SWA_DECODE_BUILD=1 VLLM_MOET_MOE_TOPKSUM_ADD=1 VLLM_MOET_SWA_DRAFT_BUILD=1
NCCL_P2P_LEVEL=SYS NCCL_P2P_DISABLE=0`, `--moe-backend b12x`, DSpark k = 7 with `"moe_backend":"b12x"`. First start
~22 min (FlashInfer autotune + DeepGEMM JIT; mount a cache at `/root/.cache` and `/root/.deep_gemm`,
`--gpu-memory-utilization 0.92` that once), later ~7 min. The r13 image (`5d25cff`) stays published under its digest
for `vllm-moet-deepseek-v4.1-flash-1017-512k-tp4`.

## Build

`.github/workflows/release-image.yml` with `image=vllm-moet-dsv41` (linux/amd64). No GPU is needed: every
extension sets its CUDA arch (12.0).
