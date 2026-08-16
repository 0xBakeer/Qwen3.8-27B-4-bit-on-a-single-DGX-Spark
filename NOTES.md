# Findings and gotchas — 4-bit on SM121

## 1. There is no native FP4 compute path on GB10

SM121 lacks the tensor-core support the FP4 GEMM kernels require. vLLM's actual path is:

```
NVFP4 weights  ->  Marlin dequant  ->  higher-precision GEMM
```

The speedup you get from 4-bit on this device comes **entirely from reading fewer bytes
per token**, not from faster arithmetic. Three practical consequences:

- **`W4A4` is strictly worse than `W4A16` here.** Quantized activations cost accuracy and
  return nothing, because that path is not executed in FP4. Choose weight-only.
- **CUTLASS FP4 kernels reportedly produce silent garbage** on this architecture — wrong
  output, not an exception.
- Because decode is bandwidth-bound rather than compute-bound, the Marlin dequant
  overhead is largely hidden at low concurrency. It is a worse trade for prefill, which
  is compute-bound.

## 2. `VLLM_MARLIN_USE_ATOMIC_ADD=1` is mandatory

A documented race condition in the Marlin kernel on SM121 yields **incorrect output**
without it. There is no error and no warning. Every 4-bit path on this device runs
through Marlin, so this variable sits in your critical path.

This one deserves emphasis because of how it fails: a kernel race and a bad quantization
produce the same symptom — a model that is subtly, unpredictably wrong. If you evaluate a
4-bit checkpoint without this set, you may reject a perfectly good checkpoint.

## 3. Which layers must stay in higher precision

Qwen3.8-27B is hybrid: 48 GatedDeltaNet (linear-attention) layers, 16 full-attention.

Independent evaluation of the same-generation 27B found NVFP4 builds that quantize the
linear-attention path **"consistently and significantly underperform"**, while builds
keeping it in 16-bit degrade only subtly. The specific finding: **linear attention can be
quantized as long as `in_proj_a` and `in_proj_b` remain in higher precision** — two small
per-layer recurrence-control projections.

A generic "quantize every Linear" recipe will hit them, and the resulting model can pass
aggregate benchmarks while failing real tasks.

Recipes observed in the wild that handle this correctly:

- **AutoRound int4 W4A16 g128** on decoder Linears, with `in_proj_a`/`in_proj_b` and the
  vision tower left in BF16.
- **ModelOpt / NVIDIA**: 193 W4A16_NVFP4 layers (MLP + `lm_head`, group 16), 208 FP8
  attention/GDN layers, 257 BF16 layers.
- **MLP-only NVFP4A16**: 4-bit on MLP projections only; attention, SSM, vision and
  `lm_head` all BF16. Lowest divergence of the group — and, at 28.8 GiB, no faster than
  FP8. Quality play, not a speed play.

If you quantize this model yourself, spare `in_proj_a` and `in_proj_b`, keep the MTP
tensors out of the quantized set (see §4), and calibrate on data resembling your workload
rather than generic web text.

## 4. Speculative-decoding heads must survive quantization

Checkpoints carrying an in-checkpoint MTP head must list those modules in
`quantization_config.ignore`. Where this is documented, the warning is blunt: quantize
the draft modules and acceptance falls to ~0 %, making generation **slower** than no
speculation at all.

Notably, the official FP8 checkpoint quantizes its own 22 `mtp.*` tensors to FP8 and
still drafts well — measured 82 % / 66 % / 48 % acceptance by draft position. So FP8 is
evidently gentle enough; 4-bit is where this becomes dangerous.

An external drafter (DSpark) sidesteps the issue entirely: it is a separate model and is
unaffected by how you quantize the target.

## 5. Quantization mostly does not change speculative behaviour

Acceptance was essentially identical across FP8 and both 4-bit builds on edit-heavy work
(98.6 % / 98.7 % / 98.7 %, mean 7.91 tokens per pass in all three). Fresh-generation
acceptance moved only slightly, 31.7 % → 32.8 % → 36.9 %.

This is convenient: **you can tune `k` on FP8 and carry the setting over to 4-bit.**
It also means the two levers are close to independent, so their gains multiply rather
than overlap.

## 6. KV cache capacity is not proportional to checkpoint size

| Checkpoint | Size on disk | KV tokens at `gmu 0.85` |
|---|---:|---:|
| MixedInt4-AutoRound | 20.8 GB | 728,038 |
| NVFP4 (Unsloth) | 22.6 GB | **1,323,090** |

The *larger* file leaves 1.8× more room for KV. Runtime footprint depends on how weights
are stored and unpacked in memory, not on download size. If concurrency or long context
matters to you, measure `GPU KV cache size` from the startup log rather than reasoning
from file sizes.

## 7. Benchmark honestly

Two failure modes that inflate numbers, both easy to commit by accident:

- **Reusing one prompt across concurrent streams.** Prefix caching then serves most of
  them. Measured here at ~11 % inflation (277 → 246 tok/s aggregate at c8).
- **Quoting aggregate throughput as if it were single-stream.** These differ by ~4× at
  c8 on this hardware. Published figures often omit which one they mean.

Also state: output length, cold or warm cache, speculative method and `k`, and whether
thinking was disabled. This model reasons at high effort by default, and an uncapped
request can emit tens of thousands of reasoning tokens — a benchmark that omits the
thinking flag is not measuring decode throughput.

## 8. Evaluate on a real task, not only on benchmarks

A same-family checkpoint in this model line passed standard benchmarks and then failed a
concrete code task badly enough to justify a blanket "16-bit only" policy. Aggregate
scores did not predict it.

Quantization damage tends to appear first in long dependency chains and precise
recall — multi-file changes, exact API signatures — and last in one-shot generation,
which usually looks fine. Choose your evaluation task accordingly.

By contrast, the speculative decoding and prefix caching configuration requires no such
gate: the target model verifies every drafted token, so output is preserved by
construction.
