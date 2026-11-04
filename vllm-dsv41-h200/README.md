# vllm-dsv41-h200

DeepSeek-V4.1-Flash-1017 (the official checkpoint: FP8 dense, MXFP4 routed experts, Engram tables) on four H200
141 GB (sm_90), tensor parallel 4, 262,144-token context, DSpark speculative decoding (k=5).

Base: `vllm/vllm-openai:v0.30.0` by digest, with its vLLM replaced by vLLM main @ `51d2a855` (the published
`wheels.vllm.ai` wheel, sha256 `5afb7e5e…`). Dependencies are pinned by `constraints.txt`; the build fails unless
`pip freeze` equals `freeze.expected`.

## What is in it

| Layer | Files | What it does | Numerics |
|---|---|---|---|
| [vllm#64893](https://github.com/vllm-project/vllm/pull/64893) @ `382b4f475f` | 49 files under `vllm/` (`pr64893-files.txt`) | `--additional-config '{"decode_backend":"latency_specialized"}'`: a fused decode executor for DeepSeek-V4.1-Flash on Hopper TP4 (steps of up to 192 rows; larger steps run stock) | see below |
| SSD Engram reader | `common/engram.py`, `nvidia/model_state.py`, `model_loader/weight_utils.py`, `/opt/deepseek/ssd/disk_engram.py` | `DSV41_ENGRAM_DISK=1`: the 2 x 98 GB Engram tables stay in the checkpoint on NVMe; rows are read (mmap, 32 splits) and staged before each graph replay, with a 2M-row GPU cache. Host RAM stays small. | bit-exact (`DSV41_ENGRAM_VERIFY=40` checks 40 rows against the checkpoint at start) |
| vllm#56697 backport | `fused_moe/oracle/mxfp4.py` | `DSV41_MARLIN_IALIGN=64`: Marlin MXFP4 experts pad the TP4 intermediate to 576 instead of 640; 7.5 GB less weight per GPU, more KV | exact (padding only) |
| vllm#59418 backport | `nvidia/flashmla.py`, `nvidia/ops/small_head_sparse_decode.py` | `DSV41_SMALL_HEAD_DECODE=1`: split-KV sparse decode for 1-2 requests on SM90 | same attention; split-KV merge in the log2 domain, FP32 (passes the logprob check) |
| P2P all-reduce fallback | `ops/cute_dsl/all_reduce_mhc.py`, `ops/cute_dsl/primitives.py`, `ops/mhc.py`, `specialized_decode/__init__.py`, `boundary/all_reduce.py`, `boundary/publishing_gemm.py`, `hopper/dense_gemm_cute.py` | the executor's fused all-reduce needs NVLink multicast (NVSwitch). Without it (H200 NVL with NVLink bridges) each publisher stores to every peer's mailbox over NVLink P2P instead. `DSV41_MHC_P2P=0` turns it off. | same reduction as the multicast path; only the store instruction changes |
| Exact dense operands | `hopper/dense.py` | `DSV41_SD_DENSE_BF16=1`: the executor's dense kernels always take BF16 operands, never the f16 pre-scaled activations | weights and activations exact; FP32 accumulation |
| Hybrid adaptive verification | `envs.py`, `v1/worker/gpu/model_runner.py`, `cudagraph_utils.py`, `post_forward_cudagraph.py`, `spec_decode/dspark/speculator.py`, `spec_decode/rejection_sampler.py`, `specialized_decode/runtime/decode.py`, `runtime/metadata_graphs.py` | `VLLM_DSPARK_ADAPTIVE_MIN_REQS=16`: decode steps of up to 16 requests verify every draft (and run on the executor); steps of 17-160 requests use adaptive verification on the stock path; `VLLM_DSPARK_ADAPTIVE_FULL_REQS=160`: steps of more than 160 requests verify every scheduled draft again. Above 160 requests adaptive verification stalled the server (every draft rejected from the first finished request on) 5 times in ~34 bench cells at 224-256 concurrent requests; with the cap, 22 of 22 cells above 160 requests were clean and 256 requests decode 4.5 % faster (6,808 vs 6,516 tok/s, two A/B pairs) | lossless: the target model's rejection sampling decides every token in both modes |
| Host-sync removal | `spec_decode/adaptive_verification.py` | resets draft confidences with `index_fill_` instead of an indexed assign that copied a scalar from the host and synchronized every step | identical values |
| Offload scheduler | `kv_connector/v1/offloading/scheduler.py` | `dspark_mixed_target_groups` for the KV offload connector (not used by the launch below) | — |

The executor's own MoE kernel (W4A8, lossy) is not used: `--moe-backend marlin` keeps the routed experts on
Marlin W4A16, and `--linear-backend humming` serves the FP8 dense layers as W8A16 with exact BF16 activations.
Lossless check used for every change: greedy completions with per-token logprobs on 12 natural-text prompts,
compared against the image without the change and against a second run of the same image (noise floor).

## Serve

The model directory is the Hugging Face snapshot `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`
(510 GB) on local NVMe; Engram rows are read from it at every decode step.

```sh
docker run --rm --gpus '"device=0,1,2,3"' --shm-size 16g -p 8000:8000 \
  -v /path/to/DeepSeek-V4.1-Flash-1017:/models:ro \
  ghcr.io/sybil-solutions/vllm-dsv41-h200@sha256:<digest> \
  /models --served-model-name deepseek-v4.1-flash --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 4 --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"thinking":true}' --engram-config '{"cpu_offload":true}' \
  --max-model-len 262144 --max-num-seqs 256 --max-num-batched-tokens 16384 \
  --enable-prefix-caching --enable-chunked-prefill --enable-prompt-tokens-details \
  --gpu-memory-utilization 0.985 --kv-cache-memory-bytes 41725000000 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --safetensors-load-strategy lazy --no-disable-custom-all-reduce \
  --linear-backend humming --moe-backend marlin \
  --additional-config '{"decode_backend":"latency_specialized"}'
```

`--engram-config '{"cpu_offload":true}'` is overridden by `DSV41_ENGRAM_DISK=1` (image ENV): no pinned host copy is
made. `--kv-cache-memory-bytes 41725000000` gives 7.59M KV tokens per GPU set. The first start JIT-compiles
FlashInfer and CuTe kernels (several minutes; mount `/root/.cache` to keep them).

Measured results, the full launch and the replay command: the
[local-ai registry](https://github.com/sybil-solutions/local-ai-registry) recipe that pins this image.
