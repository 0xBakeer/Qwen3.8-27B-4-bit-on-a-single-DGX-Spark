# Measurements — Qwen3.8-27B 4-bit on one DGX Spark

Measured against a live vLLM `v0.27.1-aarch64` server on a DGX Spark (GB10 / SM121,
128 GB unified, 273 GB/s), sole occupant, `gpu-memory-utilization 0.85`,
`max-model-len 262144`, `temperature=0`, thinking disabled.

Where two numbers appear in a cell, they are two consecutive runs.

## Method

Two workloads at opposite ends of output predictability:

- **Fresh generation** — write a Python module from a one-line spec, 400 output tokens.
  Nothing in the prompt to copy; draft acceptance is low.
- **Edit-heavy** — a ~2,000 token source file plus "add a method to every class, output
  the complete file", 3,000 output tokens. Most of the output already exists in the
  prompt; acceptance approaches 99 %.

Reported: throughput, draft acceptance, and mean tokens per forward pass, the latter two
read from the engine's `spec_decode_*` counters.

## 1. Quantization gain, drafting held constant

All rows use DSpark `k=7`, so the difference is the weights alone.

| Weights | Size | Fresh gen | Edit-heavy | Accept (edit) | Mean tok/pass | KV tokens |
|---|---:|---:|---:|---:|---:|---:|
| FP8 (official) | 28.5 GiB | 20.00 | 45.02 / 47.10 | 98.6 % | 7.91 | 635,895 |
| MixedInt4-AutoRound | 20.8 GB | 29.12 | 56.69 / **63.67** | 98.7 % | 7.91 | 728,038 |
| NVFP4 (Unsloth) | 22.6 GB | **29.23** | 58.62 / 59.05 | 98.7 % | 7.91 | **1,323,090** |

Fresh-generation acceptance: FP8 31.7 %, MixedInt4 32.8 %, NVFP4 36.9 %.

**Quantization gain: 1.46× on fresh generation, 1.26–1.35× on edit-heavy.**

Fresh generation benefits more, which is consistent with the mechanism — at low
acceptance you execute more full forward passes, and those are bandwidth-bound, which is
precisely what fewer weight bytes fixes.

### The two 4-bit builds tie on single-stream decode — and nowhere else

On one request at a time they are inseparable: 29.12 vs 29.23 fresh, ~57–64 vs ~59
edit-heavy, all inside run-to-run variance.

**That equivalence does not survive contact with any other axis.** Measured at `k=7`,
batch budget 16384:

| | MixedInt4 | NVFP4 | NVFP4 advantage |
|---|---:|---:|---:|
| KV cache (tokens) | 716,736 | **1,323,090** | 1.85× |
| c4 aggregate | 145.26 | **161.36** | +11 % |
| c8 aggregate | 204.98 | **229.31** | +12 % |
| c16 aggregate | 223.29 | **256.47** | +15 % |
| Prefill ~8K | 930.4 | **1,527.3** | **+64 %** |
| Prefill ~32K | 866.7 | **1,225.5** | +41 % |
| Prefill ~100K | 713.3 | **861.1** | +21 % |

**NVFP4 is the better build**, and the concurrency margin *widens* with load — consistent
with the KV cache being what limits sequences in flight. The prefill gap is larger still.

The one argument on the other side is evidence, not performance: **MixedInt4 publishes
quality numbers (99.32 % MMLU recovery) and this NVFP4 build publishes none.** If you
cannot run your own evaluation, that asymmetry may matter more than 15 % of throughput.

An earlier revision of this document called these two "a statistical tie" on the strength
of the single-stream numbers alone. Concurrency and prefill data showed that was wrong.

## 2. Draft depth on 4-bit weights — `k=14` wins outright

NVFP4 (Unsloth), DSpark, `--max-num-batched-tokens 16384`.

| `k` | Fresh gen | Edit-heavy | Accept (edit) | Mean tok/pass |
|---|---:|---:|---:|---:|
| 7 | 29.23 | 58.62 / 59.05 | 98.7 % | 7.91 |
| **14** | **29.55** | **72.63 / 75.01** | 68.7 % | **10.62** |

`k=14` is faster on **both** workloads. Note that its acceptance *rate* is much lower
(68.7 % vs 98.7 %) while its tokens *per pass* are much higher (10.62 vs 7.91) — the rate
falls because later draft positions are speculative guesses, but the early positions are
nearly certain. Per-position acceptance observed live at `k=14`:

```
1.000  1.000  1.000  1.000  1.000  0.959  0.945
0.822  0.630  0.521  0.288  0.178  0.110  0.027
```

The first five positions are effectively free. Acceptance rate alone is therefore a poor
tuning signal — **mean tokens per forward pass is the metric that predicts throughput.**

`k=14` requires `--max-num-batched-tokens 16384`. At the default budget the engine
computes `max_num_scheduled_tokens = -1280` and refuses to start, because draft slots for
`k × max_num_seqs` exceed the default 2,048-token batch.

## 3. Concurrency — and why `k=14` is not simply "better"

Aggregate throughput across N simultaneous edit-heavy requests, 1,500 output tokens each,
distinct prompts.

| Config | Single stream | c4 aggregate | c8 aggregate |
|---|---:|---:|---:|
| DSpark `k=7` | 59.1 | **167.67** | **246.02** |
| DSpark `k=14` | **75.0** | 157.54 | 218.66 |

**The ranking inverts between single-stream and concurrent serving.** `k=14` is 27 %
faster for one request and 11 % *slower* at c8, because deeper drafting consumes
scheduling capacity that would otherwise serve additional sequences.

So there is no globally best `k`:

- **Interactive / one request at a time** → `k=14`
- **Fleet of parallel workers** → `k=7`

Quoting only the 75.0 figure would misrepresent this configuration for anyone running
concurrent agents.

> **Confound, and its resolution.** The two rows above differ in *two* variables: `k=7`
> ran at vLLM's default batch token budget, while `k=14` requires
> `--max-num-batched-tokens 16384` to start at all. A control run separates them — §3a.

### 3a. Control — `k=7` at the same batch budget

| Config | Batch budget | Single stream | c4 | c8 |
|---|---|---:|---:|---:|
| `k=7` | default | 58.62 / 59.05 | **167.67** | **246.02** |
| `k=7` | 16384 | 60.07 / **61.89** | 161.36 | 229.31 |
| `k=14` | 16384 | **72.63 / 75.01** | 157.54 | 218.66 |

The confound was real and material. Comparing like with like at 16384:

- `k=14` is **+21 % single-stream** (75.01 vs 61.89)
- `k=14` is **−2.4 % at c4** and **−4.6 % at c8**

So draft depth does cost concurrency, but roughly **half** of the 11 % penalty implied by
the uncontrolled comparison was the batch budget, not `k`. The honest cost of `k=14` is
about 5 % at c8, not 11 %.

**A second finding falls out of the control:** raising `--max-num-batched-tokens` from the
default to 16384 *hurt* concurrent throughput at identical `k` (246.02 → 229.31 at c8,
−6.8 %) while slightly helping single stream (59.05 → 61.89, +4.8 %). The flag is not a
free win — raise it only when a higher `k` requires it.

**Best measured configurations, therefore:**

| Goal | Configuration | Result |
|---|---|---:|
| Single-stream latency | `k=14`, batch budget 16384 | **75.01 tok/s** |
| Fleet throughput | `k=7`, default batch budget | **246.02 tok/s aggregate** |

### Full concurrency detail — NVFP4 + DSpark `k=7`, default batch budget

**Distinct prompts** (each stream a different source file — the honest configuration):

| Concurrency | Aggregate | Per stream |
|---|---:|---:|
| c1 | 59.79 | 59.79 |
| c4 | **167.67** | 41.92 |
| c8 | **246.02** | 30.75 |

**Identical prompts** (every stream sends the same prompt, so prefix caching serves most
of them — measured for contrast, not for quoting):

| Concurrency | Aggregate | Per stream |
|---|---:|---:|
| c1 | 59.79 | 59.79 |
| c4 | 183.27 | 45.82 |
| c8 | 277.37 | 34.67 |

Reusing one prompt across all streams overstates aggregate throughput by ~11 %
(277 → 246 at c8). Small, but it is the difference between a number that survives
scrutiny and one that does not.

Per-stream throughput degrades as concurrency rises (59.8 → 41.9 → 30.8) because
speculative decoding consumes batch slots: draft tokens occupy scheduling capacity that
would otherwise serve additional sequences. Aggregate still scales 1 → 2.8× → 4.1×.

**If you serve a fleet of parallel agents, c8 aggregate is the number that matters.
If you care about interactive latency, c1 is.** They differ by 4×.

## 3b. Concurrency, both checkpoints, full curve

DSpark `k=7`, batch budget 16384, distinct prompts, 1,500 output tokens per stream.

| Concurrency | MixedInt4 agg. | per stream | NVFP4 agg. | per stream |
|---|---:|---:|---:|---:|
| c1 | 64.35 | 64.35 | ~61.9 | ~61.9 |
| c4 | 145.26 | 36.32 | **161.36** | 40.34 |
| c8 | 204.98 | 25.62 | **229.31** | 28.66 |
| c16 | 223.29 | 13.96 | **256.47** | 16.03 |

Aggregate throughput is **still rising at c16** for both builds, so the peak lies beyond
what was tested. But per-stream throughput has roughly halved from c8 to c16 (28.66 →
16.03 on NVFP4) for a 12 % aggregate gain. Past c8 you are buying total throughput at a
steep latency cost — for a worker fleet where each task's turnaround matters, **c8 is the
better operating point despite c16 posting the larger number.**

## 4. Prefill throughput

Prompt sizes chosen to bracket realistic long-context use. Each measurement uses unique
content so prefix caching cannot serve it; `max_tokens=1` isolates prefill from decode.
Two independent cold runs per cell.

| Prompt | Tokens | MixedInt4 | NVFP4 |
|---|---:|---:|---:|
| ~8K | 10,260 | 930.4 | **1,527.3** |
| ~32K | 42,584 | 866.7 | **1,225.5** |
| ~100K | 136,624 | 713.3 | **861.1** |

Run-to-run agreement was within **0.05 %** on every cell — by a wide margin the most
reproducible measurement in this project, since prefill is compute-bound and does not
depend on drafter luck.

Both builds fade with prompt length (NVFP4 −44 % from 8K to 100K) as attention cost grows.
The two 4-bit builds differ far more than their decode numbers suggest, which is a second
reason the "tie" framing was wrong.

**Against FP8**, measured on the same harness:

| Prompt | FP8 | NVFP4 | FP8 advantage |
|---|---:|---:|---:|
| ~8K | 1,506.3 | **1,527.3** | −1.4 % |
| ~32K | **1,312.0** | 1,225.5 | +7.1 % |
| ~100K | **939.4** | 861.1 | +9.1 % |

The prediction held, with a caveat on its size: Marlin dequantization is compute-bound
work that decode hides and prefill pays for, so 4-bit does lose ground — but only **7–9 %
at long context**, and nothing at all at 8K. A real cost, and a smaller one than the
mechanism suggests.

## 4a. The quantization advantage disappears under concurrency

Same benchmark, FP8 versus NVFP4, both DSpark `k=7`:

| Concurrency | FP8 | NVFP4 | 4-bit advantage |
|---|---:|---:|---:|
| c1 | 46.91 | 59.79 | **+27 %** |
| c4 | 134.21 | 161.36 | +20 % |
| c8 | 208.71 | 229.31 | +10 % |
| c16 | 256.08 | 256.47 | **+0.2 %** |

A clean monotonic decay to zero — and the mechanism is the one that makes quantization
work at c1, running in reverse. Single-stream decode is memory-bandwidth-bound, so reading
fewer weight bytes is decisive. As the batch grows, one weight read serves many sequences,
the workload becomes compute-bound, and byte count stops mattering.

**If you serve a fleet at high concurrency, 4-bit buys you nothing on this hardware** —
FP8 matches it at c16, and carries no quantization quality question. Everything this
repository measures is worth having at c1 to c8, and worth nothing at c16.

This is the single most important result here, and it is not something any of the
published figures we compared against would have revealed, since they quote one operating
point.

## 5. Full context of the speedup

Everything below is edit-heavy, single stream.

| Configuration | tok/s | vs stock |
|---|---:|---:|
| FP8, no speculation, no prefix caching | 7.88 | 1.0× |
| FP8 + MTP `k=3` | 21.3 | 2.7× |
| FP8 + MTP `k=8` | 32.2 | 4.1× |
| FP8 + MTP `k=15` | 39.0 | 4.9× |
| FP8 + DSpark `k=7` | 47.1 | 6.0× |
| 4-bit + DSpark `k=7` | 59.1 | 7.5× |
| **4-bit + DSpark `k=14`** | **75.0** | **9.5×** |

The weights change only in the last two rows. **Six of the 9.5× is decode strategy on
unchanged FP8 weights**; quantization contributes the final 1.6×.

## 6. Required environment on SM121

Both variables were set for every 4-bit run in this document.

| Variable | Why |
|---|---|
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | Fixes a Marlin race condition on SM121 that yields **incorrect output**, not an error. 4-bit weights run through Marlin on this device. |
| `VLLM_USE_FLASHINFER_MOE_FP4=0` | Keeps the MoE path off CUTLASS FP4, which is reported to emit silent garbage. Not strictly required for this dense model; harmless and cheap insurance. |

Additionally reported by others for SM121, and **not** used in these runs — so treat as
untested here rather than recommended: `--attention-backend TRITON_ATTN` (FlashInfer is
reported to have accuracy bugs on this architecture), avoiding `--enforce-eager`
(reported ~55 % throughput cost), and staying on the 580.x driver branch.

## 7. Comparing against published figures

Numbers only compare when the conditions do. Two live examples:

- **38.28 tok/s** for this model on this device: single-stream, 4-bit, with speculative
  decoding. Our comparable figure is 59.1 at `k=7`, or 75.0 at `k=14`.
- **84.3 tok/s**: **8-way concurrent aggregate**, MTP `k=3`, with `--enforce-eager` and
  prefix caching disabled. Our comparable figure is the 246.02 above — the gap is
  explained by DSpark instead of MTP, no eager mode, and prefix caching left enabled.

Before comparing anything, establish: concurrency, output length, cold or warm cache,
which speculative method and `k`, and whether the quoted number is aggregate or
per-stream.
