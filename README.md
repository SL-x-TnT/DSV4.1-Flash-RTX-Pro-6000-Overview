# DeepSeek V4.1 Flash on one 96 GB GPU: what we changed relative to DeepSeek's official inference code

Audience: an AI agent building or tuning its own inference engine for DeepSeek V4.1 Flash. You are assumed to have
DeepSeek's official inference code that ships with the checkpoint (`inference/model.py`, `kernel.py` (TileLang),
`engram.py`, `generate.py`, `convert.py`, `config.json`, `README.md`; here at
`/data/models/DeepSeek-V4.1-Flash-1017-FTW/inference/`, checkpoint revision "1017") and NOT our code. Every
technique below is described relative to that code, with enough detail to reimplement it. `model.py:N` and
`kernel.py:N` are line numbers in those files.

The official code is the baseline for everything: it defines the math ("correct" = what it computes) and it is the
starting design. Its README calls it "a readable reference implementation rather than a production serving engine".
There is no vLLM or SGLang speed baseline for this model on this host, and neither is used as a reference here.

Conventions:
- tok/s = DECODE-only throughput: tokens after the first, over the decode window. TTFT is always reported separately
  and never folded into tok/s, including concurrent aggregates.
- norm = acceptance-normalized speed = 3 x tok/s / A, where A = 1 + accepted/drafts. We verify 3 drafts per step,
  so A <= 4. It removes the run-to-run variation of speculative acceptance.
- C1 = one request. x8 = 8 concurrent requests (cached 32K prompt + unique tail), steady-state aggregate.
  mix8 = 6 uncached agent prompts (2.2-3.8K tokens) + 2 cached-32K requests together. uNk = an uncached N-thousand-token
  prompt. "Agent prompt" = a 2.2-3.8K-token coding-agent request.
- "Production: on" means the technique is enabled in our shipped configuration.

Hardware (all numbers): 1x RTX PRO 6000 Blackwell 96 GB (SM120, 99 KB shared memory per block, power capped at
400 W), AMD EPYC 9654 (96 Zen 4 cores, AVX-512 BF16, no AMX), 12-channel DDR5-4800 (460.8 GB/s theoretical, ~330 GB/s
achievable read), 377 GB RAM, RAID0 of 2x Samsung 990 PRO NVMe. Pinned host-to-device copies reach ~57.5 GB/s on
one copy engine.

## 1. Summary

The official code assumes every weight is resident across 8 GPUs (`convert.py --model-parallel 8`, README) and runs
plain one-token-at-a-time generation. On this host the weights do not fit: routed experts are ~296 GB
(40 x 384 + 3 x 128 experts x 18.8 MB in FP4), Engram tables ~189 GiB, on one 96 GB card. Our engine serves the
model anyway, with the official math:
- About 4.4K of the routed experts live in GPU slots. A 60-worker AVX-512 CPU executor computes the rest in place
  from host RAM, overlapped with the GPU's work. No expert is fetched over PCIe on the decode critical path.
- Engram tables are read from NVMe with O_DIRECT async I/O plus a small RAM cache.
- Prefill streams experts over PCIe, or splits them between the CPU executor and PCIe. Only the first 20 layers
  run over the whole prompt. Layers 20-39 run over an exact 2,541-row dependency cone at the end of it.
- DSpark (the checkpoint's draft head) proposes 3 tokens per step with stochastic sampling. An exact
  rejection-sampling verify of 4 rows shares each step's CPU expert cost over about 3 tokens.
- Almost everything is CUDA-graph captured at boot. KV is stored as packed FP4 (890 B/token, lossless relative to
  the official cache). The pool holds 2,097,152 tokens, up to 524,288 per request.

### Numbers

"Today" = the validation run of 2026-10-27 (shipped build), production C8 settings, 4 GiB host KV tier,
3 GiB prefix-state arena.

| Metric | Official code, this host | First V4.1 port (2026-09-10) | 1017 campaign start (~2026-10-20) | **Today** |
|---|---|---|---|---|
| C1 decode tok/s / norm | cannot serve (see below); offline study greedy ~2.7-2.9 tok/s | 9.73 eager; 31.9 graphed | 96 / 101 (29.7 ms/verify, 32K) | **144.4 / 147.5** (A 2.94, 20.3 ms/step) |
| x8 steady tok/s / norm | no serving path | no C8 | no C8 (4 slots) | **248.2 / 274.2** |
| mix8 steady; TTFT uncached / cached-32K | no serving path | n/a | n/a | **260.0 / 267.8 norm; 4.83 / 8.35 s** |
| Agent prompt TTFT, uncached 2.2-3.8K | not measured | not recorded | ~2.7-2.8 s | **1.27-1.50 s** |
| u8k TTFT (prompt tok/s) | not measured | not recorded | 3.10 s | **2.02 s (3,987)** |
| u32k TTFT (prompt tok/s) | not measured | not recorded | 5.76 s | **3.14 s (10,043)** |
| u128k TTFT (prompt tok/s) | not measured | not recorded | not recorded (Oct-21: 16.4 s) | **10.47 s (11,786)** |
| u500k TTFT (prompt tok/s) | not measured | not recorded | not recorded (Oct-21: 72.5 s) | **40.22 s (12,433)** |
| Cached 32K TTFT | no prefix cache (re-prefills every turn) | not recorded | 3.37 s | **0.237 s** |
| Startup (boot to ready) | not measured | not recorded | ~180 s | **87 s** (warm kernel caches) |
| Peak VRAM | needs ~8 GPUs' worth | not recorded | 95,334 MiB | **96,580 MiB** (incl. full-2M stress) |
| Context | `max_seq_len` construction arg: default 4,096, interactive 65,536 (`model.py:52`, `generate.py:126`) | 512K KV | 512K/request, 2,097,152 aggregate | **512K/request, 2,097,152 aggregate** (4 x 523K active verified) |

What is known about the official code's speed here (offline only, never a serving configuration):
- To run it at all for the per-layer exactness study (section 5), the official `model.py`/`kernel.py` were driven
  unmodified, with three harness changes. Engram tables stayed in host RAM, and the requested rows were gathered and
  then dequantized with the official arithmetic. Experts were loaded lazily into a 2,400-expert GPU LRU. `sparse_attn`
  ran with `num_stages=1` on 16-head groups, because the shipped launch config (64 heads, `kernel.py:322-339`) exceeds
  SM120's 99 KB shared memory. Per-head math is unchanged.
- Teacher-forced, layer-major (each layer's weights loaded once; 3 texts x 1,068 positions = 300-row prefill +
  768 one-token decode steps): 954 s. Free greedy continuation, token-major, batch 1, ~1.1K context: 128 tokens in
  44-47 s per text (~2.7-2.9 tok/s, dominated by LRU expert streaming). These are study timings, not serving speed.
- Structural limits for serving on one GPU: all weights must be resident. Prefill is one forward over the shortest
  prompt (`generate.py:67-74`). In a batch, prompt tokens past the shortest prompt are fed one token per forward
  (`generate.py:75`). The indexer materializes `[b, s, 32, s/ratio]` BF16 scores (`model.py:557`): about 137 GB
  for a 64K prompt at a ratio-2 layer. There is no chunked prefill (every `start_pos > 0` path assumes one token,
  e.g. `model.py:477-480`, `719`), no prefix cache (`generate.py:92-93`), no CUDA graphs, and no speculative loop
  (`model.py:129-130`).

Today's other measured rows:
- Decode in the first 16 tokens after a fresh uncached prefill: u8k 63.4, u32k 61.1, u128k 69.4 tok/s, while the
  post-prefill expert refill is still landing. Over 128 tokens: 131.9 (u8k) / 124.9 (u32k).
- Cached 500K follow-up turn: TTFT 0.21 s, decode 123.4 tok/s (140.8 norm). After an uncached u500k prefill: 115.9.
- 3 staggered requests: 191.2 / 209.5 norm. GPU-only weak-CPU mode (no CPU expert math, K2): C8 81.4 tok/s.
- Accuracy at validation: GPU reference mode bit-exact vs the stored reference; teacher-forced gate dNLL -0.52%,
  KL 0.0134, argmax 94.1%, 1 confident flip. That is placement spread; see 5.1.

### Milestones (decode tok/s / norm; TTFT separate)

| Date | State | C1 | Concurrency | Uncached TTFT 32K / 128K / 500K | Cached 32K | Startup |
|---|---|---|---|---|---|---|
| 09-10 | first V4.1 port | 9.73 eager, 31.9 graphed | - | - | - | - |
| 09-11 | native FP8 dense decode, fused projections | 47 -> 79.8 short ctx | - | - | - | - |
| 10-18 | first 1017 production | 107-114 at 32K, A ~3.0-3.15 | C2/C4 lifecycle only | - | - | - |
| ~10-20 | 1017 campaign start | 96 / 101 | - | 5.76 / - / - | 3.37 | ~180 |
| 10-21 | hybrid prefill, resident tail | 130.2 / 124.9 | C4 185.3 | 4.98 / 16.4 / 72.5 | 0.39 | ~180 |
| 10-22 | prefill kernels, two-micro-batch verify | 146.7 / 140.2 | C4 186.7 | 3.80 / 12.07 / 44.97 | 0.35 | 112-120 |
| 10-22 | C8 batched graphs, fair interleave, wave padding | 128-138 | C8 232-236 | - | - | 86-97 |
| 10-23 | swap unstall (residency keeps adapting) | 144.8 / 143.1 | x8 262.1 / 286.1 | - / 11.48 / - | 0.29-0.30 | 96 |
| 10-24 | 3 exactness fixes vs official | 143.1 / 140.8 | x8 244.5 / 259.6 | 3.60 / 11.48 / - | 0.37 | 110 |
| 10-27 | round 1 (refill gate, batch select, fold tiles) | 143.9 / 146.2 | x8 257.8 / 275.6, mix8 272 | 3.12 / 11.08 / - | 0.296 | 84 |
| 10-27 | exact decoder replay on | 148.8 / 151.3 | x8 249.3 / 264.0 | 4.45 / 12.11 / 41.67 | 0.233 | - |
| 10-27 | **shipped (GPU replay, fold, long-prefill kernels)** | **144.4 / 147.5** | **x8 248.2 / 274.2** | **3.14 / 10.47 / 40.22** | **0.237** | **87** |

## 2. The official model, as the techniques see it

| Fact (from `config.json` + `model.py`) | Where | Why it matters |
|---|---|---|
| 40 layers + 3 DSpark stages. `compress_ratios`: L0-1 = 0 (window only), L2-19 = 2, L20-39 = 1, DSpark = 0 | config; `Attention.__init__` `model.py:637,681-699` | Layer classes differ; the window-only layers use plain RoPE, the others YaRN |
| Compressed KV is produced only by `kv_source_layers` [2, 8, 14, 20] and shared down the stack: 3-7 read L2, 9-13 L8, 15-19 L14, 21-39 L20 | `_compress_kv` `model.py:740-764`; `SharedAttentionRuntime` `1169-1183` | Layers 21-39 add no KV of their own beyond a 128-token window. This makes the exact replay cone possible (4.8) |
| Index keys belong to the KV sources (`Indexer.owns_k`); `index_source_layers` [2,8,14,20,24,28,32,36] pick top-512; other layers reuse `topk_idxs` | `model.py:500,527-581,723-738` | Only 8 indexer runs per token |
| Two-level indexer: L20's scores select 2,048 blocks of 8 compressed positions (`select_candidate_blocks`); L24-36 score only inside them | `model.py:570-576,584-611` | Bounds deep-layer index cost |
| Attention is single-latent (MQA-like): one 512-d KV row per position serves as K and V for all 64 heads; sliding window of 128 raw rows + top-512 compressed rows in one `sparse_attn` call; per-head attention sink | `Attention.forward` `766-791`; `kernel.py:310-389` | Tiny KV per token |
| Window KV: FP8 quantize-dequantize of the post-RoPE row (`act_quant(..., inplace=True)`), kept in a 128-slot ring | `_window_kv` `701-721` | |
| Compressed KV: ratio-2 sources pool 2 tokens with an FP32 softmax gate (FP32 `wkv`/`wgate` weights); ratio 1 = BF16 projection + norm; RoPE at the group's first position; then FP4 quantize-dequantize, groups of 16 with E4M3 scales | `Compressor` `429-485`; `model.py:753-762` | Stored values are exactly FP4 x scale |
| Index keys/queries: FP4 quantize-dequantize, groups of 32, E8M0 scales; scores = `einsum` (BF16, cuBLAS) -> ReLU x `weights_proj` -> sum over 32 heads | `model.py:544-558` | FP4 products are exact, see 5.5 |
| Output: inverse RoPE on o, FP8 round trip of o, grouped `wo_a` (8 groups, BF16 weights, dequantized by `convert.py:157-173`), FP8 `wo_b` | `model.py:782-790` | Two of our exactness fixes |
| MoE: 384 routed FP4 experts, top-6, sqrt(softplus) scores in FP32, bias for selection only, renorm (+1e-20), x 1.5; 1 shared FP8 expert; SwiGLU clamp 10 | `Gate` `794-829`; `Expert` `832-853`; `MoE` `856-906` | Experts are 18.8 MB each with scales |
| Expert math: `linear()` = per-32 FP8 (E8M0 power-of-two scale) activation quant + `fp4_gemm` (FP4 upcast to FP8, FP8xFP8 MMA per K-block of 32, scales applied per block in FP32) -> BF16; gate/up upcast to FP32, clamp, SiLU*up, x route weight, -> BF16 -> w2 | `model.py:181-207,843-853`; `kernel.py:40-124,477-559` | Defines every rounding point we match |
| Combine: FP32 accumulator, routed outputs added in expert-id order, then the shared expert, then one cast | `MoE.forward` `891-906` | |
| mHC: residual carried as 4 copies; FP32 mixer `hc_mixes` -> `hc_split_sinkhorn` (20 Sinkhorn iterations); each sublayer uses the pre-mix computed by the previous one | `Block` `909-996`; `kernel.py:406-474` | The delay lets the mixer run off the critical path |
| Engram at L1 and L14, before the block: 2..4-gram hashes x 8 heads -> 24 rows of a ~384M-row FP8 table (per-32 E8M0 scales) -> FP8 `wkv` -> one key per hc copy + one value, sigmoid(signed-sqrt) gate | `model.py:296-365,1265-1266`; `engram.py:129-184` | 2 x ~98 GB tables: does not fit in VRAM |
| DSpark: block of 5 (input token + 4 noise tokens), non-causal inside the block, attends to the target's window KV; input = mean over hc copies of the inputs of layers 37-39; Markov head (rank 256) adds a per-position bias from the previous sampled token; 128 experts, top-3 | `model.py:1022-1159,1268-1269,1277-1285` | Proposal-only; no verify loop exists in the official code |
| LM head in FP32; sampling = Gumbel-max over softmax(logits/T) | `model.py:1007-1014,1288-1295` | |

## 3. Component comparison: official code vs ours

"Math" column: **bit-identical** = equal bits by construction or by test; **matched order** = same rounding points
and operand values, but a different summation order, so bits differ at ulp level; **different, verified** = a
deliberately different computation, checked numerically. Evidence is detailed in section 5.

| Component | Official (`file:line`) | Ours | Math | Speed effect (recorded) |
|---|---|---|---|---|
| Weight storage / placement | All weights on GPU, TP/EP over 8 ranks (`convert.py:127-146`) | Dense, attention, router, shared expert, heads and drafter non-expert weights in VRAM. Routed experts in pinned THP host banks (official FP4 layout, unchanged bytes). ~4.4K experts in GPU slots, in three tiers (4.1) | bit-identical bytes | the enabler |
| Routed expert compute, GPU | `Expert.forward` (`model.py:843`) via `linear()`/`act_quant`/`fp4_gemm_kernel` (`kernel.py:477`) | DS-FP4 tensor-core kernels on resident slots: decode GEMV per route, prefill grouped GEMM; same per-32 FP8 activation quant, FP4->FP8 operands, FP32 per-block scale accumulation, BF16 gate/up, weight before w2 | matched order | |
| Routed expert compute, CPU | n/a | Misses computed on the CPU: exact FP8-valued activations x exact E2M1->BF16 weights, BF16 pair dots, FP32 per-block scaling, parity accumulators (4.2) | matched rounding points, different order | removes PCIe from decode |
| Prefill expert GEMM, long chunks | `fp4_gemm_kernel` | W4A8 native MXFP8 x MXFP4 block-scaled MMA (b12x, forked epilogues) for streamed chunks >= 12,288 rows | different, verified (5.4) | u32k 4.49 -> 3.91 s |
| Shared expert | FP8 `Expert` (`model.py:889`) | FP8 GEMMs fused with routing; SwiGLU clamp fused | matched order | |
| Router / gate | FP32 `F.linear(x.float(), w.float())` (`model.py:813`), torch top-k | Prefill: same FP32 GEMM. Decode: fused BF16-input/FP32-accumulate GEMV + bit-identical sqrt(softplus) + one-CTA top-k (ties to the lower index) | prefill same op; decode <= 1 ulp | ~470 launches/step -> 2 per layer |
| MoE combine | FP32, expert-id order, + shared, one cast (`model.py:895-906`) | FP32, route order 0..5, + shared, one cast | matched rounding, order differs | none measurable |
| Sparse attention | TileLang `sparse_attn` (`kernel.py:310`), BF16 P | Decode: staged selected rows, split-K paged kernel, inverse RoPE + FP8 round trip of o in the merge. Prefill: one program per query (Gluon), in-kernel FP4/FP8 dequant. BF16 P | matched rounding, order differs | component 118.6 -> 20.9 us; prefill 1.38-1.45x/layer |
| Sliding window | 128-slot ring, FP8 round trip (`model.py:701-721`) | Request-local windows (43 per request, incl. DSpark) holding the same values | bit-identical values | |
| Compressed sparse attention + compressor | `Compressor` (`model.py:429-485`), FP32 pooling, carry in `kv_state` | Fused compressor (3 launches instead of 25), carry kept per request across chunks and snapshots | matched order | |
| Indexer (two-level) | `einsum` BF16 scores, ReLU x weights, head sum, top-k, candidate blocks (`model.py:527-611`) | Prefill: real FP4 tensor-core q.k with a per-tile exactness certificate and BF16 fallback; fused head reduction + top-512 select. Decode: scores straight from packed FP4 slabs, reproducing cuBLAS's MMA chains | bit-identical scores (5.5) | 500K chunk 641 -> 281 ms/layer; 500K decode +7.4% norm |
| `fp4_act_quant` | quantize-dequantize, stores BF16 (`kernel.py:184`) | Same quantizer (same amax floors, RN to E2M1, E4M3 or E8M0 scale); stores codes + scales | bit-identical values | 3.6x less KV |
| Engram | GPU table, sharded rows; hash on GPU (`model.py:296-365`; `engram.py`) | Tables on NVMe (O_DIRECT AIO), 4 GiB RAM row cache, scales in RAM; hash on the host, predicted and submitted early; fused dequant + `wkv` + gate | same values; gate reductions matched order | decode Engram mostly hidden |
| mHC (`hc_split_sinkhorn`) | FP32 mixer + Sinkhorn per sublayer (`model.py:950-996`) | Norm+contraction in one launch on the critical path; mixer + Sinkhorn as a sidecar of a later GEMM (PDL early trigger) | matched order | -8 us/layer decode |
| RoPE | complex multiply, YaRN for compressed layers (`model.py:368-406,681-699`) | Fused into projections / attention epilogues; same frequencies | matched | launches removed |
| KV cache layout / size | Dense BF16 per-slot buffers sized `max_batch_size x max_seq_len` (`model.py:664-680,521-525`) | Shared paged pool of FP4 codes + scales: 890 B/token, 2,097,152 tokens (1.87 GB), 512K/request; 4 GiB lossless host tier; 3 GiB snapshot arena | lossless | 512K request = 445 MiB |
| Prefill | One forward over the whole shortest prompt, all 40 layers (`generate.py:67-74`) | 64K-row chunks of layers 0-19 over all rows; layers 20-39 over the exact 2,541-row cone; tail folded into the last chunk; hybrid CPU/PCIe expert split; W4A8; multi-prompt batching | cone bit-exact vs full prefill (official model) | u500k 40.2 s |
| Decode | Eager, one token/forward, full model per step | CUDA graphs per lane; CPU expert doorbells inside the graph islands; residency swaps on the copy engine | non-expert kernels not batch-width invariant (5.3) | 9.7 -> 31.9 tok/s from graphs alone |
| Speculative decoding | DSpark forward only, nothing calls `forward_spec` (`model.py:129-130`) | Block drafted at the trained width 5, truncated to K=3; stochastic proposals with exact p/q rejection; 32K-token draft vocabulary with q = 0 outside | distribution-exact | spec 93.7-95.8 vs plain 63.5 tok/s (Sep); K3 stochastic +17-21.6% norm vs K5 greedy |
| Sampling | Gumbel-max, temperature only (`model.py:1288-1295`) | Temperature/top-p/top-k sampler with inverse-CDF draws; speculative rejection keeps the target distribution | same distribution, different sample path | |
| Batching / concurrency | Right-padded static batch, longer prompts teacher-forced one token per step (`generate.py:46-78`) | Up to 8 lanes, batched verify graphs C2-C8, two micro-batches at C2-C4, multi-prompt encoder prefill, fair prefill/decode interleave | per-request math unchanged | x8 248 tok/s |
| Startup / loading | `convert.py` reshards into one file per rank; `load_model` per rank (`generate.py:133`) | Load the HF checkpoint's FP4 experts unchanged into pinned THP banks (16 loader threads), dense weights in parallel; boot-time graph capture and kernel primes | bit-identical bytes | 87 s boot |

## 4. Techniques by area

Format: what the official code does -> what we do and how -> why it is faster -> measured effect -> pitfalls.
Effects are same-boot A/B unless noted. Everything listed is on in production unless marked.

### 4.1 Memory plan and expert placement

- **VRAM budget.** Resident: all non-expert weights in their checkpoint formats (FP8 dense with 32x32 E8M0
  blocks, BF16 `wo_a` as `convert.py` makes it, FP32-promoted compressor weights, BF16 heads streamed). The token
  embedding table is in pinned host memory (one row per token is gathered). The rest of the card is expert slots,
  graphs and the KV pool. Peak 96.58 GB with all 2M KV tokens occupied.
- **Expert slot tiers** (one slot = one expert's gate_up + down rows with scales, 18.8 MB; 4,350-4,480 resident in
  decode):
  - 2,688 fixed slots.
  - A context loan of up to 512 VMM-backed reclaimable slots. They are unmapped when long contexts need KV/scratch
    VRAM and returned afterwards (16-slot steps, mid-prefill reclaim, keep the loan for up to 8 live requests of
    <= 64K each). +4.3% norm when introduced; keeping it with 8 short concurrent requests gave mix8 +23-25%.
  - A 1,284-slot decode-only tail borrowed from the shared prefill workspace. Prefill scratch and these slots occupy
    the same VRAM. A slot is usable only if no byte of it was written by the last prefill (see the scratch-packing
    pitfall in 4.8).
- **DSpark experts** (128 per stage, top-3) sit in the same slot pool as the target's and always run on the GPU. A
  fixed-width (top-6) CPU submission does not fit their shape.
- **Engram decode staging**: 384 rows per table per lane instead of a full-size staging buffer. This frees 1.55 GiB,
  which became 64 more fixed slots.
- **Graphs**: C1 lane graphs 1.363 GiB, batched C2-C8 1.604 GiB. The last-verify acceptances run eagerly instead of
  as 12 captured graphs (~240 MiB back to the loan).

### 4.2 CPU expert executor (the decode workhorse for misses)

- **What it computes**: the official `Expert.forward` for routes whose expert is not GPU-resident, in place from the
  pinned host banks (no copy).
  - Activations are FP8-quantized like `act_quant` and held as BF16, which is exact. E2M1 weights are decoded to
    exact BF16 operands (a 3-instruction AVX-512 VBMI decode).
  - `vdpbf16ps` pair dots start from zero for each 32-element block. Each block result is scaled by both E8M0 scales
    in FP32 and added to alternating (parity) block accumulators, followed by one final horizontal reduce.
  - Gate/up are rounded to BF16, SwiGLU (with clamp) and the route weight follow the official order, and the result
    is rounded to BF16 before w2.
  - Inputs with BF16 exponents outside [9, 250] take an FP32 fallback, so the pair dot never sees a denormal or
    overflow.
- **Scale folding**: E8M0 scales are powers of two, so the scale can be folded into the decoded BF16 weights
  (one 64-byte decode LUT per scale byte), which drops the per-block multiply. This is exact when
  2 <= scale <= 127 and scale + the smallest nonzero activation exponent >= 136. It is checked per tile, with the
  unfolded kernel as fallback; real 1017 data always passes. 1.2-1.3x at 8-32 routes/expert; agent TTFT -4.6%.
  Used for groups of at least 3 rows.
- **Threads**: 60 workers in a 64-physical-core fence spread over all 12 L3 domains. Per-expert dedup and
  scheduling, spin waits. Results are pinned and read by the GPU without an H2D copy. Pure DRAM read saturates at
  ~3 cores per CCD; 72 workers or SMT gave +0-4% or less.
- **Speed**: at 1 route per expert it runs ~57 us/route (17.5K routes/s, ~330 GB/s, DRAM-bound). At 8-16 routes per
  expert it reaches 21-22 us/route (3.2-3.4 TFLOP/s, compute-bound at ~50-57% of the exact-dot ceiling). Direct
  BF16 dots instead of FP32 gave 1.13-1.36x pool speed. Fast prep plus masked output cut coordinator pre-dispatch
  19.4 -> 2.9 us (+4.5% norm). THP host banks without software prefetch: +1.6-1.8%.
- **Overlap**: per layer the GPU computes the resident routes, shared expert and attention while the CPU computes
  misses. The graph island is split at a doorbell. A GPU spin kernel waits for the CPU result (-0.3 ms/step at C1),
  and the combine reads the pinned results.
- **Why not stream misses to the GPU instead**: one 18.8 MB expert at 57.5 GB/s costs ~0.33 ms per layer on the
  critical path. Measured GPU-only demand streaming: 49.3 vs 90.5 tok/s, and 2.3-3.8x worse at reduced VRAM.

### 4.3 Residency, refill and the copy engine

- **Policy**: per-(layer, expert) routing histogram, halved per pass (recency-weighted LFU, not LRU, which thrashes
  under burst writes).
  - Admission requires a candidate to be hotter than its victim by a margin (25% and a minimum count): strict
    "hotter" makes pairs swap back and forth forever.
  - Swaps are decided every 2 steps and weighted by exposure: CPU time the layer cannot hide behind its GPU cover.
  - Copies run on the copy engine. Budget by concurrency C1..C8 = 12,16,24,24,48,48,48,48 experts per decision.
  - Drafter layers get weight 8. A request with no prefix lineage starts from a 16-step global route prior.
- **Post-prefill refill**: the prefill borrows slots, so the decode residency is snapshotted at the prefill handoff.
  Afterwards every displaced expert plus the hottest non-resident ones go back in one copy burst, hottest first.
  Empty slots at step 0 drop from ~1,500 to 48; +7.7% tok/s.
  - The burst is gated at 40 experts per decode step so it does not starve forward copies.
  - Empty ring slots after a hybrid prefill are filled from the global route prior. This costs ~132 ms of TTFT on
    short uncached prompts and roughly doubles their early decode.
- **Swap unstall**: a landed swap verdict that admitted nothing must release its pending state. Otherwise residency
  freezes after the first refill. C1 +6.1%, x8 +16% (norm +17.3%), mix8 +12%.
- **Swap gating behind Engram**: the H2D copy engine is one FIFO across streams. Swap DMA is started only after the
  forward's last small Engram H2D (layer 14). Per-expert swap cost 0.25 -> 0.01 ms, C1 +4.1%.
- **Pitfalls hit**:
  - A single-entry holder for gated swaps let a second swap overwrite the first. Under sustained C8 residency
    collapsed from 4,082 to 52 experts and decode fell to ~45 tok/s. Use a queue.
  - Refill-plan use-after-free across streams: the publish step read (slot, expert) pairs from a host buffer that
    was recycled before the copy-stream DMA consumed it. Publish from the pinned buffer the DMA itself reads.
  - A swap-release race caused an illegal address. Release only after the copy event.
  - A cancel must publish fill batches that already landed, not drop them.

### 4.4 Decode: graphs and kernels

Where a C1 step goes (older profile, 21.2 ms; today 20.3 ms): GPU non-expert 8.5 ms (FP8 dense 4.9, ~1,800 small
kernels 4.45, BF16 heads 1.55); GPU-resident experts 6.5 ms (hidden); exposed CPU-expert wait 2.3 ms; draft 2.3 ms.

- **Segmented CUDA graphs**: each lane's verify is captured as graph islands split at the CPU-expert doorbells,
  for every lane at boot. 9.73 -> 31.9 tok/s. Batched C2-C8 verify graphs; C2-C4 two-micro-batch families.
  Multi-layer graphs over the shared-KV layer runs [2,8), [8,14), [14,20), ... cut launches per verify 45 -> 17.
  Nothing is captured or compiled while serving: runtime-int kernel variants (see 6) and boot primes.
- **Native FP8 dense decode**: run the checkpoint's FP8 weights directly (no BF16 upcast): 47.3 -> 59.8 tok/s
  (+26.5%, the largest single kernel step). Then fused FP8 projections, shared-expert act-quant fusion, folded
  split reductions and a dense tile retune: 60.8 -> 71.9; 79.8 at short context.
- **Selected attention**: stage only the selected latent rows (128 window + top-512) and run the split-K kernel
  inside the graph. Component 118.6 -> 20.9 us; launches 190 -> 106 per verify. One kernel builds the
  reachability masks and ids: -4..-23%.
- **Verify-row fusions** (4-row verify): fused delayed mHC 22.3 -> 6.8 us, SwiGLU+FP8 8.8 -> 1.0 us, one-launch
  indexer top-k 40 -> 11 us, FP8 epilogues. -1.89 ms/verify, +7.9% norm.
- **mHC sidecar**: the official FFN/attention input uses the pre-mix of the previous sublayer (`model.py:984-992`),
  so the norm input never depends on this sublayer's mixer. Norm + contraction run in one critical-path launch. The
  mixer GEMV + Sinkhorn launch as a sidecar of a later GEMM via PDL early trigger, and inverse RoPE goes into the
  split-K merge. -8 us/layer, est. -0.4 ms/step. Bit-identical to the unfused chain.
- **Fused router**: 2 launches per layer instead of ~11 (decode: torch top-k alone was 0.30 ms/step).
- **PDL on every decode kernel, L2 prefetch** of the next layer's attention weights during the CPU wait (evict_last),
  wo_a/wq_b retile: est. -0.2 ms/step. Graph-input foreach copy: 21.7 -> 20.5 us. Lean verify host path: -1.6
  ms/verify (+5% norm).
- **Step-start idle**: run the residency pass right after launching verify + acceptance + proposal, not after the
  wait. Stage Engram rows on the side copy stream, and trust the early hash submission (re-checked after queueing).
  Bubble 0.55 -> 0.14 ms; C1 144.1 -> 150.9 tok/s (+4.7%), norm +2.4%.
- **Decode indexer from packed slabs**: K3-verify index scores read the FP4 slabs directly and reproduce cuBLAS's
  parity-dependent MMA chains (k16 for even n, k8 with zero-interleaved K for odd n). Only the 4-row shape is
  covered; other row counts make cuBLAS pick split-K kernels. At 500K: 23.7 -> 22.1 ms/verify, +7.4% norm.

### 4.5 Attention, indexer and KV storage

- **Packed FP4 KV**: per token, the compressed rows store 4-bit codes + scales only. A ratio-2 source holds 512
  FP4 + 32 E4M3 = 288 B per compressed row (144 B/token) x 3 sources, plus 288 B at L20. Index keys: 68 B per row
  (34 B/token) x 3, plus 68 B at L20. Total 890 B/token (official BF16 buffers: ~3.2 KB/token).
  - Lossless: an E2M1 value times an E4M3 or E8M0 scale is exactly representable in BF16, which is what the
    official code stores after `fp4_act_quant(inplace=True)`. Our encoder is the official quantizer (same amax
    floors 6x2^-9 / 6x2^-126, RN to E2M1).
  - The pool is paged and shared (2,097,152 tokens), 512K per request; windows are request-local.
- **Prefill sparse attention**: one program per query, FP4/FP8 dequant in-kernel, epilogue applies inverse RoPE and
  the `wo_a` FP8 round trip (`model.py:782-787`) before the store: 30.85 -> 27.52 ms/layer at 64K rows.
- **BF16 wo_a GEMM**: dequantize once per layer forward (64 MiB, 0.06 ms) and run a cp.async-pipelined BF16 GEMM.
  Bit-exactness against the old FP8-operand kernel required its MMA operand layout: both operands in kWidth=4, so
  each m16n8k16 sums the same k columns. Plain BF16 GEMMs (kWidth=2, cuBLAS) differ by ~1 ulp in ~3e-5 of outputs.
  22.1 -> 15.3 ms/layer. Output projection in 2,048-row groups: ~-1.1% TTFT.
- **FP4 prefill indexer** (owners L2/8/14): FP4 tensor-core q.k (`mma.m16n8k64 kind::mxf4nvf4`) with a per-tile
  certificate and BF16 fallback (5.5).
  - A query-stationary CUDA kernel keeps 16 query rows x 32 heads in shared memory and streams key tiles, each warp
    owning 64 keys. The epilogue per output pair is 4 instructions (`cvt.rn.bf16x2`, `fma.rn.relu.bf16x2`, 2x
    `fma.rn.f32.bf16`).
  - A compacted top-512 gather follows. 500K chunk 641 -> 281 ms/layer; the query-stationary kernel 1.70x on top.
  - Decoder index layers (20/24/28/32/36) inside the replay cone get the same kernels: 500K 142 -> 33 ms/layer.
- **Fused prefill indexer**: no per-head score table in HBM (the official `[b,s,32,t]` tensor). 128K 16.0 -> 14.2 s
  TTFT. An earlier FP32-reduction-scope fix in the indexer projections cut chunk-size divergence 0.267% -> 0.0145%
  RMS.

### 4.6 Engram (official: GPU-resident table)

- Tables (2 x ~384M rows x 256 B FP8 = 183 GiB) on the RAID0 NVMe; per-32 E8M0 scales (5.72 GiB) in RAM; 4 GiB RAM cache
  of hot rows. Dequant + `wkv` + gate are the official math (`model.py:312-365`).
- Reads: 512-B O_DIRECT libaio. One read costs ~7-8 us of kernel CPU, so one submitter caps at ~110K IOPS.
  - Prefill uses a 16-thread bulk pool (deep queues, per-owner dedup, L1 table first) and reaches ~1.5M IOPS.
    Workers are pinned to idle SMT siblings and avoid NVMe hardware-queue groups whose completion IRQs land on
    serving cores. Cold 32K TTFT 7.38 -> 4.93 s.
  - Decode uses 3 helper threads for misses.
- Hash: the official `NgramHashState` (`engram.py:159-184`) runs on the host (exact port).
  - Decode: the next step's ids are predicted from the proposal and submitted early; layer 0 is queued before the
    hash sync and every trusted id is re-checked afterwards. -2.9% ms/verify, 0 mismatches in serving.
  - Prefill: the next chunk's hash is computed while the GPU runs this chunk.
- RAM cache cut from 32 to 4 GiB: ~+0.1 ms/verify, 28 GiB of host RAM freed.

### 4.7 Speculative decoding (official: DSpark forward only)

- **Verify loop**: target verifies [last token + 3 drafts] in one 4-row forward. Proposals are sampled
  stochastically from the drafter's q, and acceptance is exact speculative sampling: accept with min(1, p/q); on
  reject, sample max(p - q, 0) renormalized; on full accept, a bonus token from p. Output distribution = target.
  K3 stochastic replaced K5 greedy: 81.5/92.2 -> 99.05 tok/s, +17-21.6% norm.
- **Draft at the trained width**: the DSpark block is non-causal (every block query sees all block positions,
  `get_dspark_topk_idxs` `model.py:1022-1031`). Drafting a narrower block shifts every position's distribution.
  Acceptance of positions 1-3 was 0.65/0.33/0.20 at width 3 vs 0.66/0.45/0.25 at width 5. Draft 5 wide, truncate to
  K=3. Verifying 5 rows costs +36% step for +30% tokens, which is not worth it here.
- **Draft vocabulary**: readout and Markov heads read only a frequency-ranked 32,768-token subset (plus the prompt's
  rare tokens, <= S/8). q = 0 outside, so the rejection step still yields p exactly. C1 +2.7% tok/s, +4.8% norm.
  q kept compact [K, 32768] with a token->column table: proposal tail 379 -> 350 us.
- **Graphs**: proposal and acceptance are captured; C2-C8 use one acceptance graph + one batched proposal graph
  after the batched verify. Markov projection 32.6 -> 20.8 us. Native MXFP4 drafter projections: tie, kept.
- **Pitfalls**:
  - Zero-mass probability rows came from the fast softmax/top-p path (~1 per few thousand batched steps), so the
    inverse-CDF draw returned token -1. Guard and recompute invalid rows.
  - A PDL race on the scalar temperature.
  - A multi-writer inverse-CDF left the sampled token unwritten in 37/300 near-one cases. Use a single-writer CDF.
  - Duplicate ids in the draft vocab caused over-acceptance.
  - Two lanes' proposals demand-fetching into the shared slot cache at once corrupted resident experts (NLL
    +12-24%). Serialize them.

### 4.8 Prefill (official: one forward, all 40 layers, all rows)

- **Encoder/decoder split with an exact cone** (the largest structural change):
  - Layers 21-39 read compressed KV and index keys produced from layer 20's input. Their only private state is the
    128-row window. So decode needs (a) layers 0-20's compressed state for all positions and (b) every layer's
    last-128-row window.
  - We run layers 0-19 over all rows (64K-row chunks), then layers 20-39 over the last 128 + 19 x 127 = 2,541 rows.
    Layer L's last 128 rows depend on 127 earlier rows of layer L-1 through the window.
  - The MoE of each replay layer runs only on the rows the next layer reads (the cone shrinks 127 per layer).
  - In the official model (torch-port kernels) the cone gives windows and logits bit-identical to a full prefill:
    128/128 rows exact at every layer, 65/65 logits rows.
  - The earlier bounded 128-row replay was an approximation: decode KL mean 0.004-0.0066, argmax 93.8% vs full
    prefill. It remains only as a rollback.
  - Cost of exactness: +1.3-1.4 s per uncached prompt when replayed on the CPU. Running the replay MoE on the
    hybrid CPU/PCIe split brought agent TTFT 2.3-2.6 -> 1.3-1.6 s.
- **Chunked, aligned prefill**: chunk ends are 128-aligned, so every boundary can be snapshotted for prefix reuse
  (4.9). The ratio-2 compressor's odd-token carry (the official `kv_state`/`score_state`) persists across chunks,
  and chunk splits are bit-identical to one long forward (tested).
  - The unaligned tail (<= 127 rows) is folded into the last chunk, with a snapshot taken mid-chunk at the aligned
    boundary: u8k 3.48 -> 2.00 s, u32k 4.56 -> 3.16 s.
  - Trade-off: decode right after an uncached prefill drops 9-13%, because the refill now runs during decode.
- **Expert path by chunk size**:
  - <= 256 rows: CPU executor, with GPU-resident experts computed on the slot bank and only misses sent to the CPU.
    Cached 32K turn 0.71 -> 0.39 s.
  - <= 12,160 rows: hybrid split. Routing is read back exactly, unrouted experts are skipped (7-32% of experts at
    <= 8K rows), the coldest misses go to the CPU and the rest stream over PCIe. 16K-row GEMM calls, copy-ordered
    GEMM groups (4), layer L+1 experts prefetched 12 ahead. TTFT 512 tokens 2.71 -> 1.00 s; agent 2.7-2.8 -> 1.5-1.8 s.
  - Larger chunks: every routed expert streams through a ring (one copy engine, 57.5 GB/s) into the W4A8 GEMM.
    Chunks up to ~8K rows are PCIe-bound; from ~16K they are GPU-compute bound.
- **W4A8 prefill MoE** (>= 12,288-row streamed chunks): b12x's materialized MXFP8 x MXFP4 phase kernels with forked
  epilogues in the official order (5.4). u32k 4.49 -> 3.91 s, u64k 8.60 -> 7.40 s.
  - The ring layers are repacked to b12x's layout in place and unpacked after (one launch, 13.7 -> 9.2 ms per
    7.2 GB layer each way), repack-ahead off the critical path.
  - Skipping all-padding M16 blocks of an expert's last M64 tile: MoE -3.3..-3.6%, exact.
  - Kernels are power-capped at 400 W (~1.55 GHz).
- **64K chunks + carried fills**: the next chunk's layer 0/1 expert copies start early. 128K 14.01 -> 12.77 s.
  - Carry only the ring layers released before the last one, because the copy-engine FIFO otherwise made each
    continuation chunk's input H2D wait ~87 ms. u500k 41.01 -> 40.12 s with hash-ahead.
- **Dense prefill kernels**: shape-aware FP8 tiles (7 shapes 74.9 -> 46.9 ms at 32K), fused SwiGLU + one-pass mHC
  mixer (-0.26 s at 16-32K), fused Engram/mHC prefill (-38/-67 ms per 32K chunk).
  - 64K-chunk layer profile: W4A8 MoE 89.5 ms (41%), dense FP8 41 (~425 TF; cuBLASLt/CUTLASS ceiling 440-475),
    wo_a 28 (before the BF16 GEMM), sparse attention 27, mHC/norms 16.
- **Multi-prompt encoder prefill**: queued prompts' encoder-only chunks run in one forward (<= 8 prompts, <= 16,384
  rows). Every non-expert op runs per sequence exactly as alone; the routed experts run once over all rows. mix8
  TTFT 6.06 -> 4.68 s, uncached prompt tok/s 1,739 -> 3,229.
- **Pitfalls**:
  - *Scratch-packing slot loss*: prefill temporaries live in the borrowed expert tail. A slot is lost if any of its
    rows (gate_up 11.25 MiB, down 5.625 MiB) is written. Best-fit allocation filled the smaller down region first:
    ~182 slots lost per GiB. Fix: for forwards <= 32,768 rows, place each new segment lowest-address in the region
    whose written-slot prefix grows least, so both banks' written rows overlap the same slots: ~59 slots/GiB.
    Decode after u8k/u32k 116 -> 132 / 108 -> 125 tok/s; mix8 +7%.
  - *Odd-history indexer fallback in the fold*: folding the tail can give the ratio-2 index owners an odd history
    length. The fused/FP4 score kernels need an even length (cuBLAS picks a different GEMM at odd n), so the whole
    chunk fell to the unfused chain: 744 vs 257 ms/layer, +1.2 s at u128k. Fix (exact): score rows before the
    in-chunk boundary against the boundary's (even) history and the tail rows against the full history, which is
    what two separate chunks compute.
  - *Copy-engine FIFO*: a 256 KB H2D queued behind a 6 GiB fill waits ~91 ms. Order big fills so small
    latency-critical copies never queue behind them.
  - The first big prefill after boot paid ~7 s of kernel compile. A boot warm-up (2,112 + 32,100 tokens, +18 s
    boot) cut the first 32K TTFT 12.9 -> 4.8 s.
  - NumPy arrays with `MADV_HUGEPAGE` under THP defrag=madvise stalled the first large hash for 0.5-1.9 s
    (compaction). Don't madvise them.
- **Fair interleave**: once prefill has held the scheduler 2 s while requests decode, decode gets 4 steps. Max decode
  gap behind a 250K prefill 21.0 -> 5.3 s (+1-3% TTFT).

### 4.9 Prefix reuse and multi-turn (official: re-prefills every turn)

- Radix prefix cache plus boundary snapshots at every 128-aligned position. A snapshot holds the windows, compressor
  carries, Engram hash history and the layer-20 input tail. That tail is what the exact cone replay needs; resumes
  are bit-exact vs a full prefill (tested).
  - Snapshots live in one prefaulted pinned 3 GiB arena; unbudgeted snapshots had leaked +10.6 MiB/turn.
  - 7.46 MiB per boundary.
- Snapshots are also taken during decode (side-stream D2H at the verify commit), so the next agent turn resumes after
  the previous reply instead of re-prefilling it. Turn TTFT 4K 1.22 -> 0.83 s, 128K 2.16 -> 1.50 s, C2 1.24 -> 0.82 s.
  - Requires the client to echo `reasoning_content` when tools are defined (the template keeps reasoning).
- Exact incremental tokenization (reuse the common prefix up to an added-token boundary): render+tokenize 32K
  90 -> 1-2 ms; cached 32K TTFT 0.43 -> 0.32 s.
- Lossless 4 GiB host KV tier for evicted pages (890 B/token = 1.2M tokens/GiB).

### 4.10 Concurrency and scheduling (official: static right-padded batch)

- Up to 8 lanes; batched verify graphs C2-C8 (one captured pass); batched index selection (x8 +4.5% norm, mix8
  +6.4%); captured batched history projection (x8 259.3 -> 272.9).
- Two micro-batches at C2-C4: one half's CPU experts overlap the other half's GPU work. C4 163.3 -> 183.6 (+14% norm),
  C3 +5.1%. Not used at C5-C8 (243 without vs 236 with at C8).
- **Wave padding**: a request within 2K+4 positions of its max_tokens fails the "room for the next proposal" check,
  which dropped the whole batch to round-robin verifies. Give each request 2K+4 positions of internal headroom and
  truncate at the client limit. C8 full window 112 -> 246 tok/s.
- Lane publish vs drafter slot-table overlap race: fixed by per-lane publish ordering.

### 4.11 Startup (official: `convert.py` reshard + per-rank `load_model`)

- Load the checkpoint's FP4 experts byte-for-byte into pinned THP banks with 16 threads (1.1 -> 2.4 GB/s).
  - Hint huge pages only up to the free order>=9 buddy blocks at fill start, so compaction never stalls mid-fill
    (bank load 94 -> 39 s; -18..-22 s more).
  - `MADV_NOHUGEPAGE` on the zero-filled snapshot arena: -33 s.
  - Dense load overlapped with the bank fill: 96 -> 84 s.
- Persistent Triton / JIT / CUDA compute caches across restarts. Boot-time graph capture and kernel primes add
  boot time but remove serving-time stalls. Today 87 s warm, ~108 s with a cold kernel cache.

## 5. Exactness: the official code is the math definition

### 5.1 Method

- **Per layer, not end to end.** Two correct implementations already differ end to end: the official TileLang
  kernels vs an exact torch port with the same rounding points and FP32 accumulation give KL20 0.0160, argmax
  94.75%, dNLL -0.40%, 0 confident flips over 2,304 teacher-forced positions. Their chained hidden-state difference
  reaches ~0.2 relative L2 by layer 20. End-to-end KL/argmax therefore cannot rank exactness below ~0.016.
- **Gold runner**: the official `model.py` unmodified with its own kernels (only the SM120 launch change). It runs
  the official teacher-forced semantics (one prefill, then one decode step per forced token), layer-major. Extra
  batch streams replace each layer's input with our dumped layer-L input, so each layer's output measures our layer
  alone given our own caches ("isolated" error). Shared attention state (top-k, candidates) is recorded per step and
  replayed to consumers.
- **Bit-exact GPU reference mode** (test only, regression check for "no math change"):
  - Every routed expert of every call is copied from the host banks into a private scratch bank (distinct experts
    sorted by id) and computed by the per-route DS-FP4 kernel, whose tile does not depend on row count.
  - Combine in FP32 in route order; eager islands; prefix reuse off; plain one-token greedy.
  - Placement-independent and bit-reproducible across boots. It never runs the CPU executor or W4A8.
- **Teacher-forced accuracy gate** (3 texts, 2,304 scored positions, vs a reference scored on the same texts):
  confident argmax flips <= 3, argmax agreement >= 93%, KL <= 0.02, |dNLL| <= 0.5%, top-1 within 1 pt.
  - Prefill changes are gated uncached; otherwise every arm reuses the first arm's cached prefix.
  - Placement alone (which experts are on CPU vs GPU) moves dNLL by up to ±0.5-0.57% with KL ~0.011-0.014. Read KL,
    argmax and flips first.

### 5.2 The three fixes found by the per-layer study

All three are on in production; together they cost nothing measurable (same boot, all on vs all off: C1 143.5 vs 140.0 tok/s, x8 244.0
vs 249.0, both within noise).
1. **FP8 round trip of the attention output before `wo_a`** (`model.py:787`, `act_quant(..., inplace=True)`).
   We had skipped it.
2. **BF16 P (and V) in P.V** (`kernel.py:377-380`, `acc_s_cast` BF16). We used FP16.
3. **FP32 MoE combine**: routed + shared summed in FP32 with one BF16 cast (`model.py:895-906`). We had rounded the
   routed sum to BF16, then added the shared expert in BF16.

After the fixes, the isolated MoE sublayer error vs gold went from 1.9-2.8e-3 to 1.1-3.6e-4 relative L2. At the
validation of the fixes, the production path vs gold gave KL20 0.0131, argmax 94.97%, 0 confident flips: inside the
two-correct-implementations floor.

### 5.3 What is NOT bit-identical to the official math, and how it is validated

| Item | Why it differs | Evidence |
|---|---|---|
| CPU-executed experts | Same rounding points (FP8 activations, BF16 gate/up, weight before w2, BF16 out), but BF16 pair dots + parity accumulators instead of FP8 MMA order | Gate placement spread above; gold KL20 0.0131 / 0 flips at fix validation |
| GPU DS-FP4 expert kernels | Same products and per-block FP32 scaling; tile/accumulation order differs from TileLang `fp4_gemm_kernel` | Isolated MoE error 1.1-3.6e-4 rel L2 per layer |
| MoE combine order | FP32 in route order; official adds in expert-id order (`model.py:897-902`) | Included in the MoE figure above |
| W4A8 prefill GEMM (>= 12,288-row chunks) | Native block-scaled MMA sums across K blocks inside the tensor core | 5.4 |
| Sparse attention (decode split-K, prefill per-query) | Different blocking of online softmax; BF16 P matched | Isolated attention-sublayer error 0.7-4e-2 rel L2 (torch-port floor at layer 0: 1.5e-3 whole layer) |
| Decode router GEMV | BF16 inputs (exact in FP32), FP32 accumulate in a Triton tree vs cuBLAS FP32 SGEMM | <= 1 ulp of the score, 1.5e-7 max rel; can flip only an exact near-tie |
| Batch-width variance | FP8 dense GEMMs and norms are not batch-width invariant: a token computed in a 4-row verify differs in bits from a 1-row decode (official is always 1 row) | |dlogp| up to 0.16 at a near-tie seen in reference runs; compare runs only at equal speculation settings |
| Sampling path | Speculative rejection + top-p sampler vs Gumbel-max | Distribution-exact by construction; not the same sample sequence |
| Engram gate, mHC, compressor, indexer projections | Same formulas, different reduction trees | Inside the per-layer isolated error (L1/L14 within the range of other layers) |
| Bounded replay (rollback only, off) | Truncated windows in layers 21-39 | Decode KL 0.004-0.0066, argmax 93.8% vs official full prefill |

Bit-identical (by construction or test): stored KV values; FP8/FP4 quantizers; FP4 index scores (5.5); exact cone vs
full prefill; chunk splits; prefix resumes; multi-prompt batching (non-expert ops); every exact kernel change
(GPU reference mode bit-exact at each step).

### 5.4 W4A8 vs the official `fp4_gemm_kernel`

- Official: FP4 weight upcast to FP8, an FP8 x FP8 MMA per 32-wide K block, the block's partial sum multiplied by
  both scales in FP32 and added to a separate accumulator (`kernel.py:535-554`). W4A8 (MXFP8 activation x MXFP4
  weight, scales applied inside the MMA) uses the same FP8 x FP4 products but sums across blocks in a different
  order.
- Stock b12x epilogues do not match the official activation order, in three ways:
  - They quantize the UNWEIGHTED FP32 SwiGLU to MXFP8 and apply the route weight to the FC2 output.
  - They skip the BF16 rounding of gate/up.
  - They use an approximate-reciprocal SiLU.
  Result: rel L2 0.0372 vs our exact path (0.0371 of it from the weighting point alone); uncached gate FAILED with
  dNLL +1.27..+1.42%, argmax 91.2%, KL 0.048.
- Forked epilogues:
  - Phase 1 rounds gate and up to BF16, computes SiLU as `g / (1 + exp(-g))` with IEEE division, multiplies by the
    route weight, then rounds to BF16 and MXFP8-quantizes with an amax floor of 1e-4 (the official order:
    `silu(gate)*up*w -> bf16 -> act_quant(.,32)`).
  - Phase 2 does not apply the weight again and rounds each route's FC2 output to BF16.
  - Append the fork's source digest to b12x's compile-cache key: it keys by geometry + package fingerprint only,
    so upstream and fork kernels collided.
- Result: fork vs exact path rel L2 3e-5 (2e-5 on a real capture). Error vs a float64 emulation: fork 1.4e-4,
  exact path 1.5e-4, so it is an equally accurate rounding realization. Uncached gate PASS: dNLL -0.30%, argmax
  95.53%, KL 0.0089, 1 confident flip. The GPU reference mode does not run W4A8; the uncached gate covers it.

### 5.5 FP4 indexer = real FP4 MMA, bit-exact vs BF16 scoring

- Official: `fp4_act_quant(inplace=True)` makes q and k FP4 x E8M0 values in BF16 (`model.py:546,553`), then the
  score is a BF16 cuBLAS einsum (`model.py:557`).
- Each product of two such values is exact. Over the 128-d dot, the sum is exact in FP32 for any order when the
  exponent spread fits: certificate max R_q + max R_k <= 24 - 2^-6 per 16 x 64-key tile. So a native FP4 MMA gives
  the same FP32 sum as cuBLAS.
- Tiles failing the certificate are re-scored on the BF16 path. The epilogue reproduces bf16(relu(bf16(s)) * w)
  plus the FP32 head sum. On real weights, top-k is identical with 0 fallback tiles.

### 5.6 Bugs that had changed outputs (all fixed)

Lane-proposal resident corruption; zero-mass sampling rows / token -1; unwritten inverse-CDF token; draft-vocab
duplicates; swap-release illegal address; CPU-prefill ring commit; W4A8 epilogue order; folded-chunk odd-history index;
reference requests seeding the prefix cache (exclude test requests from cache inserts); a stale hybrid-chunk flag in
the replay path.

## 6. Pitfalls worth knowing before you build

- The official `sparse_attn` launch config overflows SM120's 99 KB shared memory at 64 heads. Use `num_stages=1` and
  16-head groups; the arithmetic is unchanged.
- The H2D copy engine is a single FIFO across streams. Big fills delay every small copy behind them.
- cuBLAS kernel choice depends on shape (even vs odd n, row count). Reproducing its bits means reproducing its MMA
  chain per shape, and choosing chunk boundaries that keep the even path.
- Triton specializes integer arguments (==1, %16) and constexpr row counts. Every new chunk length or context length
  then compiled while serving: 0.4 s per new length in TTFT, 7.4 s at the first 500K decode. Pass runtime ints as
  `do_not_specialize` and prime launch classes at boot.
- THP: compaction stalls from `MADV_HUGEPAGE` on hash arrays and zero-filled arenas. Budget huge-page hints to the
  free high-order blocks.
- NVMe completion IRQs on serving cores starve the serving thread. Pin I/O workers away from those hardware queues.
- Sustained-C8 decode decay (~190 -> ~130 tok/s after ~3.5 min) was DIMM thermal throttling (-41% bandwidth at
  76-78 C), not code. Measure with short waves.
- Every bug listed in 4.3, 4.7, 4.8 and 5.6.

## 7. Tried and rejected

| Idea | Why rejected |
|---|---|
| K4 / K5 / adaptive K / confidence-head width | K5 117.5 tok/s did not reproduce (K3 won both confirmations); adaptive ~0 |
| Tree speculation, prompt lookup | analysis: -4..-5% / no gain |
| N-gram sidecar drafts | +10.6% on repeated JSON, -13.9% on the cached control; spec additions must be free on misses |
| Draft-token expert prediction | simulator ceiling < 1.3% |
| Draft-head fine-tuning | out of scope: inference-side only |
| Draft temperature 70/85/115%, proposal recycling | noise; recycling cut A 3.10 -> 2.65/2.14 |
| Async draft-miss fetch | -0.3% norm |
| MXFP8 draft head, draft misses on CPU | costs VRAM; tie |
| Next-layer expert prediction / lookahead, promotion staging | -2.3..-5.4% norm |
| Union-frequency placement, shared graph pools | -1.6% / -0.91% |
| Down-only GPU expert tier, scale-bit compression, LZ4/Zstd expert compression | no gain (compression 2.5-7%) |
| CPU L3 warm-up, K-chunked / parity tiles, fused gate+up, 72 workers, SMT | flat or slower; CPU path runs at ~95% of DRAM bandwidth |
| GPU reading CPU-resident experts over PCIe | ceiling 1.0-1.3%: steals DRAM bandwidth from the CPU workers |
| GPU-only expert streaming as default | ~2x slower (51.8 vs 96.4 tok/s); kept only as a weak-CPU mode |
| Layer-major MoE at full RAM | only useful in low-RAM configurations |
| Prefill chunks > 64K rows, W4A8 capacity > 16,384 | <= 0.6% TTFT for +1.7..4.6 GiB scratch |
| Background bank fill | dense load 6 -> 36 s; no net gain |
| AIMD refill budget | better at mix8, worse at C1 after uncached prefill |
| Deferring the full refill | TTFT -0.2..-0.36 s but decode -7..-14% |
| Early ring begin | no u32k gain |
| CUDA graphs for prefill tails, tail micro-batching | no TTFT gain for 0.76 GiB; +100 ms |
| q RoPE in the wq_b epilogue | bit-equal but slower (7.0 -> 9.6 ms per 32K layer-chunk) |
| wo_a as MXFP8 x MXFP8 MMA | -36% at 8K rows but not bit-exact (1e-5..1.5e-4 of outputs differ) |
| Router as a Triton FMA dot | not exact (cuBLAS SIMT SGEMM is not an in-order K chain), not faster |
| Dense FP8 via cuBLASLt | exact, only -1.3 ms/layer |
| W4A8 4 stages / M128 tiles / paired CTAs | exact but slower |
| Per-layer slot rebalance, C1 balance, promotion budget 2 / 8-16 at C1 | no gain / gate fail |
| Per-concurrency K or budget overrides at C3/C4 | lost to defaults |
| Two micro-batches at C5-C8 | noise, -32 slots |
| Bounded replay as the production path | an approximation; exact cone kept even at +1 s |
| 64-head sparse attention tile, Triton MXFP8 x MXFP4 `dot_scaled` (0.75-0.86x), persistent expert-kernel balancing | slower or already at bandwidth |
| MWAITX in spin waits | -22 W but -2.8% norm |
| Lossy ideas: low-weight expert skip, NVFP4 prefill activations, lossy index keys, router pruning, KV reduction | change output: excluded by principle |

## 8. Design principles

- Nothing that changes the model's output distribution. Exact changes must stay bit-exact in the reference mode.
  Placement changes (CPU vs GPU expert) and different-order kernels must pass the teacher-forced gate, and new
  numeric paths need a per-layer check against the official code.
- Keep the full context: 2,097,152-token pool, 512K per request, all of it usable at once within 96 GB.
- Decode throughput is decode-only; TTFT is separate; report raw and acceptance-normalized speed.
- A small decode cost is acceptable for a large TTFT cut; judge by per-request time.
- Measure at the existing power and clock settings (400 W cap): many kernels are power-bound, which changes which
  tiling wins.
- Remove features that are off or rejected but still cost time, VRAM or slots.

## 9. Map: official code -> techniques that replace or wrap it

| Official code | Techniques (section) |
|---|---|
| `Linear`, `linear()` `model.py:181-243`; `act_quant`, `fp8_gemm` `kernel.py:40-307` | native FP8 dense decode, tile retunes, fused act-quant (4.4, 4.8) |
| `fp4_gemm_kernel` `kernel.py:477-591`; `Expert.forward` `model.py:843` | DS-FP4 GPU kernels, CPU executor (4.2), W4A8 fork (5.4), reference mode (5.1) |
| `MoE.forward` `model.py:891-906` | slot tiers (4.1), residency (4.3), hybrid prefill split (4.8), FP32 combine (5.2) |
| `Gate.forward` `model.py:811-829` | fused decode router (4.4) |
| `Attention.forward`, `_window_kv`, `_compress_kv` `model.py:701-791` | selected attention, split-K merge epilogues, packed FP4 KV, request-local windows (4.4, 4.5) |
| `sparse_attn` `kernel.py:310-403` | decode split-K / prefill per-query kernels, BF16 P (4.5, 5.2) |
| `Indexer.forward`, `select_candidate_blocks` `model.py:527-611`; `fp4_act_quant` `kernel.py:184` | FP4 MMA indexer, fused selection, decode slab indexer (4.4, 4.5, 5.5) |
| `Compressor.forward` `model.py:458-485` | fused compressor, carry across chunks/snapshots (4.8, 4.9) |
| `Engram`, `ParallelEngramEmbedding` `model.py:296-365`; `NgramHashState` `engram.py:129` | NVMe row store, early hash, staging (4.6) |
| `Block.hc_mixes/hc_pre/hc_post/forward` `model.py:950-996`; `hc_split_sinkhorn` `kernel.py:406` | mHC sidecar, fused prefill mixer (4.4, 4.8) |
| `precompute_freqs_cis`, `apply_rotary_emb` `model.py:368-406` | RoPE fused into projections / epilogues |
| `DSpark*`, `forward_spec` `model.py:1022-1159,1277` | verify loop, trained-width drafting, draft vocab, graphs (4.7) |
| `Transformer.forward` `model.py:1244-1275` | encoder/decoder split + exact cone, chunking, fold (4.8) |
| `sample` `model.py:1288` | stochastic acceptance sampler, zero-mass guard (4.7) |
| `generate()` `generate.py:28-87` | chunked prefill, prefix reuse, snapshots, batching, fair interleave (4.8-4.10) |
| `convert.py` | direct load into pinned banks, wo_a BF16 as official (4.11) |
