# Qwen3.8-27B 4-bit on a single DGX Spark (GB10 / SM121)

A measured serving recipe for 4-bit Qwen3.8-27B on one DGX Spark, with a side-by-side
comparison of two 4-bit checkpoints and the FP8 baseline they are measured against.
Every number in [RESULTS.md](RESULTS.md) came from the harness in `bench/`.

**Headline: 75 tok/s single-stream (`k=14`), or 246 tok/s aggregate at 8-way concurrency
(`k=7`, default batch budget).** Those are two different configurations — draft depth
trades one against the other, and no single setting wins both. See [§3 of RESULTS](RESULTS.md#3-concurrency--and-why-k14-is-not-simply-better).

**This repository measures speed only.** No quality evaluation was performed here; every
accuracy figure below is cited from its publisher and unverified. Read
[LIMITATIONS.md](LIMITATIONS.md) before relying on any of it.

| | Fresh generation | Edit-heavy |
|---|---:|---:|
| FP8 + DSpark `k=7` | 20.00 | 45.0 – 47.1 |
| MixedInt4-AutoRound + DSpark `k=7` | 29.12 | 56.7 – 63.7 |
| NVFP4 (Unsloth) + DSpark `k=7` | 29.23 | 58.6 – 59.1 |
| **NVFP4 (Unsloth) + DSpark `k=14`** | **29.55** | **72.6 – 75.0** |
| *gain from quantization alone, `k=7`* | *1.46×* | *1.26 – 1.35×* |

The drafting configuration is identical across the first three rows, so that difference
is attributable to the weights alone. The last row adds draft depth, which requires
`--max-num-batched-tokens 16384`.

## Read this first: quantization only pays at low concurrency

Measured against FP8 on the same harness, DSpark `k=7`:

| Concurrency | FP8 | NVFP4 | 4-bit advantage |
|---|---:|---:|---:|
| c1 | 46.91 | 59.79 | **+27 %** |
| c4 | 134.21 | 161.36 | +20 % |
| c8 | 208.71 | 229.31 | +10 % |
| c16 | 256.08 | 256.47 | **+0.2 %** |

Single-stream decode is memory-bandwidth-bound, so fewer weight bytes is decisive. As the
batch grows, one weight read serves many sequences, the workload turns compute-bound, and
byte count stops mattering.

**If you serve a fleet at c16, this entire repository buys you nothing** — FP8 matches it
and carries no quantization quality question. Quantize for interactive latency and low
concurrency; do not quantize for fleet throughput.

4-bit also costs **7–9 % of prefill throughput** at 32K–100K prompts (§4), because Marlin
dequantization is compute-bound work that prefill pays for.

## Read this next if you are choosing a quantization

**GB10 has no native FP4 compute path.** SM121 lacks the tensor-core support the FP4
GEMM kernels need, so 4-bit weights are dequantized through Marlin and multiplied in
higher precision. Three consequences:

1. **`W4A4` buys nothing here.** You pay the accuracy cost of quantized activations and
   receive no compute benefit, because the activation path is not executed in FP4.
   **Prefer `W4A16`** — weight-only. The speedup on this device comes entirely from
   reading fewer bytes per token, not from faster math.
2. **CUTLASS FP4 kernels are reported to emit silent garbage on this architecture.**
   Not an error — wrong output. Keep them out of the path.
3. **`VLLM_MARLIN_USE_ATOMIC_ADD=1` is required.** There is a documented race condition
   in the Marlin kernel on SM121 that produces incorrect output without it. Since 4-bit
   weights go through Marlin here, this sits directly in your critical path — and a
   kernel race looks exactly like "this quantization is bad quality".

## Checkpoints compared

| Checkpoint | Size | Scheme | Published evaluation |
|---|---:|---|---|
| `Qwen/Qwen3.8-27B-FP8` | 28.5 GiB | FP8 (official) | — (reference) |
| `…-MixedInt4-AutoRound` | 20.8 GB | int4 W4A16 g128 on decoder Linears; `in_proj_a`/`in_proj_b` and vision tower BF16 | MMLU recovery **99.32 %** |
| `unsloth/Qwen3.8-27B-NVFP4` | 22.6 GB | NVFP4 with 8-bit groups for sensitive modules | none published |
| `…-NVFP4A16` (MLP-only) | 28.8 GiB | 4-bit MLP only; attention, SSM, vision, `lm_head` BF16 | ppl +3.0 %, lower divergence than official FP8 |
| ModelOpt / NVIDIA recipe | ~21 GB | 193 W4A16_NVFP4 (MLP + `lm_head`, g16) + 208 FP8 attention/GDN + 257 BF16 | GSM8K 81.25 %, **HumanEval 39/40**, IFEval 38/40 |

Two things worth noting from that table.

**The MLP-only NVFP4A16 build is a quality upgrade, not a speed one.** At 28.8 GiB it is
the same size as FP8, so decode throughput is unchanged. Sparing attention, SSM, vision
and `lm_head` is what keeps its divergence low — and what eliminates the size advantage.

**The ModelOpt recipe has the strongest published evidence**, and it is the only one of
these reporting a code benchmark. If you need to justify a quantization to someone,
HumanEval 39/40 is a better argument than an MMLU delta.

## Which layers must not be quantized

Qwen3.8-27B is hybrid: 48 GatedDeltaNet (linear-attention) layers and 16 full-attention
layers. Independent evaluation of the same-generation 27B found that NVFP4 builds which
quantize the linear-attention path **"consistently and significantly underperform"**,
while builds keeping it in 16-bit show only subtle degradation.

The specific fix: **linear attention can be quantized as long as `in_proj_a` and
`in_proj_b` stay in higher precision.** These are two tiny per-layer recurrence-control
projections. Miss them and the model degrades in ways benchmarks may not surface but
real work will.

If you quantize this model yourself, that is the single most important line in the recipe.

## Quickstart

```bash
./serve.sh                      # NVFP4 + DSpark k=7 by default
MODEL=<int4-repo> ./serve.sh    # or the AutoRound int4 build
python bench/edit_bench.py
python bench/conc_bench.py
```

## Where the speed actually comes from

Only a minority of the total speedup on this device is the quantization:

| Change | Edit-heavy | Cumulative |
|---|---:|---:|
| Stock FP8, no speculation, no prefix caching | 7.88 | 1.0× |
| + speculative decoding (DSpark `k=7`) | 47.1 | 6.0× |
| + 4-bit weights | 59.1 | 7.5× |
| + draft depth (`k=14`) | 75.0 | **9.5×** |

**Six of the 9.5× is decode strategy on unchanged FP8 weights** — speculative decoding is
output-preserving by construction. Establish that baseline before reaching for
quantization: it is free, reversible, and carries no accuracy question at all.

Quantization contributes the remaining 1.6×, and it is the only step here that changes
what the model knows. Treat it accordingly — see [Evaluation](#evaluation).

## Evaluation

Published benchmark scores are necessary but not sufficient. A same-family checkpoint in
this model line passed standard benchmarks and then failed a real code task badly enough
to justify a blanket "16-bit weights only" policy for months. Benchmarks did not predict
that; a concrete task did.

Before promoting any 4-bit build, run **your own representative task** — ideally a
multi-file change or an API-precise piece of work, since that is where quantization
damage shows up first, rather than one-shot generation which tends to look fine.

The speculative-decoding and prefix-caching configuration needs no such gate. It cannot
change output.

## License

MIT for the scripts and documentation in this repository. Model weights remain under
their upstream licenses.
