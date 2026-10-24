# vllm-moet-dsv41-engramdisk

DeepSeek-V4.1-Flash on 4x RTX PRO 6000 Blackwell (sm_120, TP4) with **[vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet)
by [kacper-daftcode](https://github.com/kacper-daftcode)**, plus one local-ai layer that keeps the Engram tables on NVMe.

All of the speed is vLLM-Moet's work: vLLM nightly `49cb2bbf` with FlashInfer 0.7.2's DSv4.1 sparse MLA on sm_120,
`nvfp4_ds_mla` KV, the MXFP4 indexer, DSpark (k=5) under FULL CUDA graphs, MXFP8 decode GEMV, fused MoE glue, PDL
decode kernels and the rest listed in its `Dockerfile.sm120-dsv41-nightly`. This directory builds that exact file
at `213d5fa9` (source tarball pinned by sha256) and changes nothing in it.

## The added layer: Engram tables from disk

Upstream keeps the 203 GB of Engram tables (FP8 e4m3 + E8M0 scales) in pinned host RAM (~330 GB of shm in
practice). A 128 GB host cannot do that. With `VLLM_MOET_ENGRAM_DISK=1` (set in the image):

- the loader skips `layers.*.engram.embed.weight`; only the per-row scales (6 GB total, split over TP) stay resident;
- each step, a CUDA host node (`cuLaunchHostFunc`) `pread`s this rank's rows straight from the checkpoint's
  safetensors file into pinned staging (`engram_disk.c`, up to 64 threads, `POSIX_FADV_RANDOM`); the OS page cache
  holds the hot rows;
- the upstream `_engram_lookup_kernel` then reads the staging buffer through UVA in SORTED mode with identity ids,
  so the output is bit-identical to the in-RAM lookup;
- host nodes are captured into FULL CUDA graphs, so DSpark keeps its graphs.

`test_engram_disk.py` (in the image at `/opt/vllm-moet/engram_disk/`) compares eager lookups at 1-4096 tokens and
three FULL-graph replays bit-for-bit against rows dequantized on the host:

    docker run --gpus device=0 -v <checkpoint>:/model:ro IMAGE python3 /opt/vllm-moet/engram_disk/test_engram_disk.py

Host needs: ~25 GB RAM for the engine processes, the rest of RAM as page cache for Engram rows, the checkpoint on
local NVMe, `memlock` unlimited.

## Weights and precision

`DeepSeek-V4.1-Flash` official checkpoint, unmodified: routed experts MXFP4 (296 GB), Engram FP8 (203 GB, on disk),
attention + shared experts FP8 e4m3 32x32 blocks with ue8m0 scales (7 GB), embeddings/head BF16.
