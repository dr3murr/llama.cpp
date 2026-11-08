# Qwen3.8-27B on a Tesla P100 + RTX 5080 (branch `qwen38-27b-p100`)

> Written with AI coding agents and measured on one machine (RTX 5080 + Tesla P100, Ryzen 9 9950X3D, 64 GB DDR5);
> not reviewed by a kernel/GPU engineer. Use at your own risk.

This branch is the August 2026 working tree of a private llama.cpp fork, committed as it was found on top of
upstream `c060ca974` ("model : support MTP in GLM-4.5-Air (#26534)"), plus this document and the files in
`qwen38-27b/`. It runs Qwen3.8-27B across both cards: 24 layers on the P100, 40 on the RTX 5080, the native
262,144-token context, an 8-bit KV cache, the model's built-in MTP head for speculative decoding, and its vision
projector. It is a memory-tight single-user "appliance" configuration, not a general-purpose build.

On the machine it was built for, it has since been replaced by Qwen3.8-Flash-Next on Basalt with the P100 as an
expert tier ([dr3murr/basalt `PASCAL.md`](https://github.com/dr3murr/basalt/blob/pascal-tier/PASCAL.md)). The newer
P100 decode kernels on branch [`p100-qr0n`](https://github.com/dr3murr/llama.cpp/blob/p100-qr0n/P100.md) are not
merged here.

## Requirements

- **Driver: the P2P-patched R580 setup.** Blackwell needs NVIDIA's open kernel modules and Pascal the proprietary
  ones, so the pair runs on the proprietary R580 driver with an undocumented module option, and this runtime was
  built and measured on 580.173.02 with the mixed-generation P2P patch set and `iommu=pt`. Both manifests set
  `"p2p": {"enabled": true}` (32-bit single-writer doorbells at the decode boundary) and the streamed-attention path
  copies K/V tiles peer to peer. Behaviour on a driver without P2P was never tested. Driver steps, the exact module
  options and the P2P patch:
  [dr3murr/open-gpu-kernel-modules `P2P_P100_RTX50.md`](https://github.com/dr3murr/open-gpu-kernel-modules/blob/p100-rtx50-p2p-580.173.02/P2P_P100_RTX50.md).
- **Linux only.** `llama-server` and `llama-perplexity` on this branch call `setenv`/`unsetenv` without a Windows
  guard; a Windows build was not attempted.
- **CUDA 12.9** (CUDA 13 cannot compile for the P100's `sm_60`), GCC 14, CMake, Ninja.
- **A SageAttention checkout beside the repository**, at build time only: `ggml/src/ggml-cuda/qwen38-sage-attention.cu`
  includes a header from `../SageAttention-reference` ([Build (Linux)](#build-linux)).
- **Device order:** the manifests set `CUDA_VISIBLE_DEVICES=1,0`, so logical `CUDA0` is the P100 and `CUDA1` the
  RTX 5080 on the machine it was built for. Check your own `nvidia-smi` order.

## What the branch adds (one commit, 55 modified and 17 new files)

- **KV cache:** signed `Q8_KV64` / `Q8_KV128` / `Q8_KV256` types (int8 values, one FP16 scale per 64/128/256 values)
  with CPU reference code, CUDA copy/set-rows/convert paths and flash-attention vec/MMA/tile kernels, including a
  direct Q8-to-MMA path on the RTX and a GQA-shared verification tile (each K/V row reused across two query positions
  and three GQA heads for Q=2..9).
- **Pascal (sm_60):** packed-half2 (HFMA2) KQ partials in the flash-attention tile, and transient FP16 expansion of
  quantized weights for bulk prefill with a fixed cuBLAS algorithm per projection class (6 for FFN gate/up/down,
  2 for GDN QKV, 3 for the attention query projections).
- **Streamed attention:** K/V tiles of P100-owned layers are copied peer to peer into a bounded RTX ring and merged as
  exact FP32 online-softmax partials (`ggml/src/ggml-cuda/qwen38-streamed-attn-runtime.cu`).
- **`llama-server`:** `-m <manifest>.json` (schema `qwen38-hybrid-manifest-v1`) selects the layer split, KV type, MTP,
  vision and the kernel policies above; staged layer-major prefill for cold prompts of 2,048+ positions (up to
  32,768 positions all 64 layers run on the RTX one layer at a time from a reusable weight slot; above that, layers
  0-23 do); a rolling draft context for the MTP head; layer-subset state restore and recurrent-state rollback fixes.
- **Web UI:** stops a generation after three exact repeats of a long (96-2,048 character) cycle.
- **Off unless enabled:** Sage-style attention, a chunked Gated DeltaNet kernel (`LLAMA_GDN_CHUNK_CUDA`), and
  single-token Pascal FFN/GDN-input ops that run from separately packed weight files (those files are not published).
- The sm_60 fp16 carve-out in `ggml-cuda/common.cuh` (sm_60 treated like sm_61 as slow-fp16) was added on
  2026-11-03, after the August work; it is the fix from ggml-org/llama.cpp#25593 (apollo-mg).

## Model files

| file | bytes | notes |
|---|---:|---|
| Unsloth `Qwen3.8-27B-UD-Q4_K_M.gguf` | 16,464,440,224 | mixed Q/IQ tensors, Q6_K output head; used by the 26/38 recovery manifest |
| `Qwen3.8-27B-UD-Q4_K_M-head-Q5_K.gguf` | 16,295,582,848 | same file with only `output.weight` re-quantized to Q5_K from the official BF16 `lm_head.weight`; SHA-256 `073a4541959394d56db7c66f5cfe70db1fd61076681e2d8b225fb34dd78626ed` |
| `mmproj-F16.gguf` | | the model's FP16 vision projector, kept on the P100 |

The Q5_K head is made with `qwen38-27b/qwen38_rehead.cpp` from the official shard `model-00018-of-00018.safetensors`
(SHA-256 `1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06`); every other tensor stays byte-identical:

```bash
make -f qwen38-27b/Makefile.rehead LLAMA=. BUILD=build CXX=g++-14     # after building the repo (below)
build/bin/qwen38_rehead Qwen3.8-27B-UD-Q4_K_M.gguf model-00018-of-00018.safetensors \
    Qwen3.8-27B-UD-Q4_K_M-head-Q5_K.gguf q5_k
```

Output-head quality, 32-chunk WikiText perplexity:

| head | head bytes | PPL |
|---|---:|---|
| Q6_K (original) | 1,042,944,000 | 5.6010 +/- 0.14688 |
| Q5_K (used) | 874,086,400 | 5.6124 +/- 0.14729 |
| Q4_K | 715,161,600 | 5.6280 +/- 0.14777 |

The Q5_K head frees the ~169 MB the 24/40 layout needs. An NVFP4 variant was tried and rejected by the quality gate.

## Build (Linux)

`ggml/src/ggml-cuda/qwen38-sage-attention.cu` includes
`../../../../SageAttention-reference/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh`: a SageAttention checkout named
`SageAttention-reference` in the directory that holds this repository, at upstream commit `d1a57a5` with the August
changes in [`qwen38-27b/sageattention-d1a57a5-qwen38.patch`](qwen38-27b/sageattention-d1a57a5-qwen38.patch) (builds
without PyTorch; adds the masks and strides Qwen3.8 needs; Apache-2.0). The Sage path is off at run time unless
enabled, but without the checkout the build stops at that file. From the directory that holds the repository
(`llama.cpp` is the clone's default name):

```bash
git clone https://github.com/thu-ml/SageAttention SageAttention-reference
git -C SageAttention-reference checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
git -C SageAttention-reference apply "$PWD/llama.cpp/qwen38-27b/sageattention-d1a57a5-qwen38.patch"
```

Then, in the repository:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES='60;120' \
    -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_NCCL=ON \
    -DGGML_CUDA_COMPRESSION_MODE=size -DGGML_NATIVE=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_OPENSSL=ON
cmake --build build --target llama-server test-backend-ops test-recurrent-state-rollback -j
```

These are the flags of the August build script. On 2026-11-08 a fresh clone of this branch (`062433e`) was built
with them in WSL2 (Ubuntu 24.04) using GCC 13.3 instead of 14 (CUDA 12.9 accepts both) and CUDA 12.9.86; NCCL was
not installed, which CMake reports and skips. Without the SageAttention checkout the build stopped at
`qwen38-sage-attention.cu` (`No such file or directory`); with it, the three targets above, the rest of the default
target and `qwen38_rehead` compiled and linked. The compilers print warnings for this branch's own files (missing
declarations, a switch without the `Q8_KV*` types in `ggml-cpu/ops.cpp`), none fatal. Nothing was run: no tests, no
GPU. The CUDA 12 libraries have to be in the linker's search path (the CUDA packages add them to
`/etc/ld.so.conf.d`), otherwise linking the programs fails on `libcudart.so.12` and `libcublas.so.12`.

## Run

Edit the two model paths in `qwen38-27b/qwen38-hybrid-production.json` (they read `/path/to/...`), then:

```bash
build/bin/llama-server -m qwen38-27b/qwen38-hybrid-production.json --host 127.0.0.1 --port 8080
```

| manifest | P100 / RTX layers | use |
|---|---|---|
| `qwen38-27b/qwen38-hybrid-production.json` | 24 / 40 | decode and production; staged layer-major cold prefill; Q5_K head; 4,096-position streamed-attention ring |
| `qwen38-27b/qwen38-hybrid-26x38-window-mtp-p100.json` | 26 / 38 | recovery profile with the original Q6_K head; no staged prefill |

Sampling defaults are Qwen's thinking-mode values (temperature 1.0, top-k 20, top-p 0.95, min-p 0). The server is
qualified for one slot only (`parallel = 1`).

## Results (August 2026, 24/40 production layout unless noted)

| workload | result |
|---|---|
| cold prompt, exact 10,000 positions (staged layer-major) | 787.08 tok/s |
| cold prompt, 10,000 positions, ordinary mixed-device path (before staged prefill) | 415.824 tok/s |
| 32,768-position needle retrieval | 940.46 tok/s, exact answer |
| cold prompt, exact 262,144 positions (full capacity) | 564.933 tok/s (464 s) |
| 261,632-position cold prompt + 512 generated | 561.624 tok/s prompt, 11.084 tok/s decode |
| decode, last 512 positions of the full context, shared-scan verifier | 11.103 tok/s (was 9.514, +16.7%) |
| vision, 1,115 uncached positions | 263.577 tok/s prompt, 32.809 tok/s decode |
| streamed attention at full context (older 26/38 layout) | 1.0704x throughput vs the same layout without it |
| short-prompt decode, older splits | 20.227 tok/s without speculation, 35.886 tok/s with MTP (width 2) |
| short-prompt decode, 24/41 split, `llama-bench` tg128, no MTP (2026-11-04) | 24.0 tok/s; 31.1 tok/s with the `p100-qr0n` kernels (different upstream base, quality gates not run) |

Peer-to-peer on this pair (PCIe 3.0 x4 behind the chipset for the P100): `cuMemcpyPeer` 64 MiB x 50, 3.078 GiB/s
P100 -> RTX 5080 and 1.800 GiB/s RTX 5080 -> P100; a 1 MiB copy takes 330 us one way and 545 us the other; no
native peer atomics (the P100 and the chipset ports don't support AtomicOps).

## Correctness and known issues

- The Q8_KV64/128/256 backend regression passes 12/12 cases on each GPU; greedy 256- and 512-token continuations
  reproduce fixed SHA-256 hashes across layouts.
- Single-slot only: an optional multi-sequence recurrent-state stress test changes greedy output (Q5 and Q6 heads
  alike).
- Memory is tight on purpose: at the peak of the late-context + decode + vision gate the RTX had 9 MiB free. When
  the streamed-attention arena can't keep its 48 MiB launch reserve, the request falls back to Pascal attention
  instead of failing.
- Two GPU faults found during development (an Xid 31 from a test-harness reproducer and an Xid 32 from peer reads of
  a transient VMM mapping) were fixed in August; the fault audits after those fixes were empty.

## Sources

The numbers above come from the August working notes of this runtime (not published): `QWEN38_CURRENT.md`
(2026-08-26: layouts, head table, "Qualified performance", "Correctness and fault status"), `QWEN38_FINAL.md`
(2026-08-24, superseded: streamed-attention 1.0704x, short-context MTP suite), `P2P_STATUS.md` and `P2P_LATENCY.md`
(2026-08-23). The 24.0 / 31.1 tok/s comparison is from the project's agent session log of 2026-11-04.

## Credits

- Qwen for the model and Unsloth for the GGUF.
- apollo-mg for the sm_60 fp16 fix ([ggml-org/llama.cpp#25593](https://github.com/ggml-org/llama.cpp/issues/25593)).
- The P2P driver patch builds on [tinygrad's](https://github.com/tinygrad/open-gpu-kernel-modules) BAR1 P2P method and
  [aikitoria's](https://github.com/aikitoria/open-gpu-kernel-modules) R580 port; see the NVIDIA fork's doc.
- The rest was written with AI coding agents for r3muxd.
