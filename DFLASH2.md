# DFlash 2 on 4-bit Qwen3.8-27B

**The headline is a negative result with a workaround: DFlash 2 cannot be served on either
4-bit checkpoint in this repo under vLLM. It can be served on 4-bit weights under llama.cpp,
where it is the fastest single-stream configuration measured anywhere in this work —
37.58 tok/s generative and 60.89 tok/s edit-heavy.**

## Method for this section

These numbers come from a **different harness** than `bench/` — a separate suite written for
the DFlash2 evaluation. They are internally consistent but are **not** directly comparable,
cell for cell, with the DSpark/MTP tables elsewhere in this repo, which use different prompts
and output lengths. Where the two overlap they agree to within a few percent (this harness
measures stock FP8 at 7.94 tok/s against this repo's 7.88).

- 4 prompts — code generation, prose explanation, arithmetic word problem, refactor —
  3 repetitions each, `temperature=0`, thinking disabled, 512 output tokens.
- Reported single-stream figure is the **mean of per-prompt medians**, and excludes TTFT
  (`completion_tokens / (wall - ttft)`).
- Concurrency sweep uses 384 output tokens and reports **aggregate** throughput including
  TTFT, so its c=1 point reads slightly below the single-stream decode figure. Prompts are
  assigned round-robin, so **c=1 and c=2 use an unbalanced prompt mix**; c≥4 is balanced and
  is the comparable part of the curve.
- Acceptance length is `1 + accepted/drafts` from the engine's own `spec_decode_*` counters,
  deltaed across the run — the same definition Inco AI use.
- Engine flags held constant across every configuration: `--max-model-len 131072
  --gpu-memory-utilization 0.80 --max-num-batched-tokens 16384 --enable-prefix-caching`,
  image/video input disabled. Sole occupant of the device.


## 1. Why vLLM refuses

DFlash 2's selector reads the target model's top-16 candidates directly off the LM head, so
vLLM rejects any checkpoint whose head is quantized:

```
ValueError: DFlash2 requires an unquantized target LM head for candidate TopK.
```

```python
if not isinstance(self.lm_head.quant_method, UnquantizedEmbeddingMethod):
    raise ValueError(...)
```

| checkpoint | `lm_head` tensors | verdict |
|---|---|---|
| `unsloth/Qwen3.8-27B-NVFP4` | `lm_head.weight` + `lm_head.weight_scale` | head genuinely quantized — blocked |
| `Pilcothink/…-MixedInt4-AutoRound` | `lm_head.weight` only (bf16; `block_name_to_quantize` covers only `model.language_model.layers` and `mtp.layers`) | head is *stored* unquantized, but vLLM's auto-round path still assigns it a quantized method — blocked |
| `Qwen/Qwen3.8-27B-FP8` | `lm_head` in the `ignore` list | works |

The AutoRound case is the subtle one: inspecting the checkpoint suggests it should work, and
it does not. Confirm by launching, not by reading the config.

Pre-flight check:

```bash
python3 -c "import json;w=json.load(open('model.safetensors.index.json'))['weight_map'];
print([k for k in w if k.startswith('lm_head')])"
# ['lm_head.weight']                        -> necessary, not sufficient (see AutoRound above)
# ['lm_head.weight','lm_head.weight_scale'] -> certainly blocked
```

**This adds a constraint to the quantization decision this repo is about.** Choosing 4-bit
under vLLM today forecloses the best available drafter. That may change if a 4-bit build ships
with the head explicitly excluded, or if the upstream check is relaxed.

## 2. The workaround: llama.cpp

llama.cpp (PR #27342) imposes no such restriction, and Inco AI ship a pre-quantized GGUF
drafter. Unlike the vLLM PR this one touches C++ and must be compiled:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git fetch origin pull/27342/head:pr-27342 && git switch pr-27342
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_CURL=OFF
cmake --build build -j"$(nproc)" --target llama-server
```

GB10 resolves to `sm_121a`; the build takes ~2 minutes at `-j20`.

```bash
./build/bin/llama-server -m Qwen3.8-27B-Q4_K_M.gguf \
  -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -ngl 999 -c 65536 -np 16 -fa on --reasoning-budget 0
```

`--spec-type` also accepts `draft-mtp`, `draft-dspark`, `draft-eagle3` and ngram variants, so
llama.cpp can compare all three drafters on identical weights — which vLLM could not.
`--reasoning-budget 0` matters: Qwen3.8 defaults to thinking at `xhigh`.
`-np` must be set explicitly or the server serialises requests.

## 3. Results on Q4_K_M

| drafter | k | generative | edit-heavy | acceptance (gen) |
|---|---:|---:|---:|---:|
| none | — | 10.84 | 10.84 | — |
| MTP | 3 | 27.98 | 34.06 | 3.234 |
| MTP | 7 | 30.38 | — | 4.128 |
| DFlash2 | 5 | 33.29 | — | 3.99 |
| **DFlash2** | **7** | **37.58** | **60.89** | **4.471** |
| DFlash2 (Q8_0 drafter) | 7 | 36.18 | — | 4.481 |

**3.47x generative and 5.62x edit-heavy**, both over the same engine's own baseline.

Three findings:

**The Q4 drafter beats the Q8 drafter.** 37.58 vs 36.18 tok/s
for +0.01 acceptance (4.471 vs 4.481). Doubling the
drafter's size buys nothing and costs 4%. Use `Q4_K_M`.

**`k=7` is optimal, not merely maximal.** `k=5` gives 33.29 tok/s — a *higher*
accept rate (59.9% vs 49.7%) but a lower acceptance *length* (3.99 vs
4.471). Length is what pays. `k` cannot exceed 7 anyway: the drafter's
`dflash_config.block_size` is 8.

**At equal `k=7`, DFlash2 beats MTP by +8.3% on acceptance but
+23.7% on throughput.** If acceptance were the whole
mechanism those would match. The excess is the parallel draft: MTP builds a 7-token block with
7 sequential passes, DFlash2 with one. Expressed as a share of the ideal
(`delivered / (baseline x acceptance)`): DFlash2 78%,
MTP `k=7` 68%. Our acceptance gap
(+8.3%) closely reproduces Inco AI's published +12.1%.

## 4. The block-size ceiling

On edit-heavy work both drafters approach perfect acceptance, and the winner is then decided
entirely by block size:

| drafter | k | accept rate | acceptance | ceiling | tok/s |
|---|---:|---:|---:|---:|---:|
| MTP | 3 | **99.12%** | 3.974 | 4.00 | 34.06 |
| DFlash2 | 7 | 97.71% | 7.839 | 8.00 | **60.89** |

MTP accepts 99.1% of its guesses — 99.4% of everything its block can hold — and still loses by
79%, because its block holds four tokens and
DFlash2's holds eight.

> **When the drafter is often wrong, drafter *quality* decides. When it is nearly always right,
> quality is irrelevant and *block size* decides.**

### Reconciling this with §2 — `k` depends on how long the copied runs are

This repo's §2 measures DSpark `k=14` at **75.0** tok/s against `k=7` at **59.1** on 4-bit —
`k=14` winning by 27%. Our edit harness measures the opposite: `k=14` at **56.77** against
`k=7` at **59.07**, `k=14` losing by 4%.

Both are edit-heavy tests, and both are correct. The difference is **the length of
uninterrupted copied text**:

| | §2 harness | this harness |
|---|---|---|
| input file | ~2,000 tokens | ~400 tokens |
| output tokens | 3,000 | 1,200 |
| task | reproduce the complete modified file | reproduce the complete modified file |
| mean tokens/pass at `k=14` | 10.62 of 15 | 7.32 of 15 |
| accept rate at `k=14` | 68.7% | 45.1% |

§2's Fig. 2 shows draft positions 1-5 accepted *every single time* on its workload. A 3,000-token
reproduction of a 2,000-token file is an almost unbroken copy, so a 14-token draft stays on the
rails for most of its length. Our shorter edits break the copy more often, and the deep tail is
mostly discarded.

**So `k` should be tuned to the copy-run length of the real workload, not set globally.** Long
full-file rewrites reward `k=14`; short targeted edits reward `k=7`. Neither result generalises
to the other's workload — which is a good reason to run `bench/edit_bench.py` or
`bench/run_edit.py` against prompts that look like yours rather than adopting either number.

The underlying mechanism is the same in both cases: **raising `k` pays only while the acceptance
*rate* stays high**, because every drafted token is paid for whether or not it survives
verification. The two harnesses simply sit on opposite sides of that threshold.

## 5. Concurrency — llama.cpp does not scale

Aggregate tok/s:

| config | c1 | c2 | c4 | c8 | c12 | c16 |
|---|---:|---:|---:|---:|---:|---:|
| llama.cpp, no spec | 10.56 | 18.77 | 31.52 | 45.21 | 53.50 | 59.88 |
| **llama.cpp + DFlash2** | 39.19 | 38.61 | 63.47 | 71.71 | 75.08 | 77.25 |
| vLLM NVFP4 + MTP `k=3` | 28.85 | 37.79 | 77.60 | 126.72 | 166.72 | 219.82 |
| vLLM NVFP4 + DSpark `k=7` | 25.10 | 32.28 | 58.22 | 97.17 | 122.38 | 147.36 |

TTFT p50 (seconds):

| config | c1 | c2 | c4 | c8 | c12 | c16 |
|---|---:|---:|---:|---:|---:|---:|
| llama.cpp + DFlash2 | 0.452 | 0.559 | 1.335 | 2.260 | 2.941 | 4.013 |
| vLLM NVFP4 + MTP `k=3` | 0.308 | 0.477 | 0.461 | 0.606 | 0.770 | 1.101 |

**llama.cpp saturates at ~77 tok/s.** From c8 to c16 it gains
8% while the load doubles, and TTFT p50
reaches 4.0 s against vLLM's 1.1 s.
Acceptance holds at ~4.3 throughout, so this is the scheduler, not the drafter.

Crossover: llama.cpp wins c1 (39.19 vs 28.85), ties at
c2, and loses from c4 on — by 2.8x at c16.

**This sharpens the thesis this repo already argues.** §"quantization only pays at low
concurrency" says do not quantize for fleet throughput. The same now applies to the engine:
**do not run llama.cpp for a fleet.** DFlash 2's niche is the single interactive user, where it
is decisively the best option available.

## 6. And the drafter you are probably running is the wrong one

Measured on identical NVFP4 weights and identical flags, varying only the drafter:

| c | DSpark `k=7` | MTP `k=3` (in-checkpoint, free) | gain |
|---|---:|---:|---:|
| 1 | 25.10 | 28.85 | +15% |
| 2 | 32.28 | 37.79 | +17% |
| 4 | 58.22 | 77.60 | +33% |
| 8 | 97.17 | 126.72 | +30% |
| 12 | 122.38 | 166.72 | +36% |
| 16 | 147.36 | 219.82 | +49% |

MTP wins at every concurrency, and the gap widens with load. It also costs nothing: no 2.6 GiB
drafter to download, no extra weights resident, no KV displaced, one less config line.
DSpark's acceptance is 2.941 against MTP's 3.141 — a worse drafter that
also needs a second model loaded.

```
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

Caveat: this is the `Doopeworld` DSpark checkpoint. A stronger DSpark drafter
(e.g. `RadixArk`) may close the gap; it was not tested here.

## 7. Quality

Neither this repo nor the FP8 one previously carried a quality evaluation. Measured here,
greedy, thinking disabled:

| config | GSM8K (n=200) | MMLU (n=400) |
|---|---:|---:|
| FP8, no speculation | 93.5% | 84.0% |
| FP8 + DFlash2 | 92.5% | 84.0% |
| int4 AutoRound, no speculation | 94.0% | 82.0% |

All differences sit inside their confidence intervals (+/-3.3 pp on GSM8K, +/-3.7 pp on MMLU
at 95%), so none of them is established as a real difference. Per-item agreement is the
sharper test: DFlash2 vs no-speculation agrees on **99.8% of MMLU items**, while FP8 vs int4
agrees on 96.2%. **Speculation perturbs the output less than changing precision does.**

This does not resolve the 2 pp MMLU gap between FP8 and int4 — that needs roughly 2,000 items.
It does exclude a large quantization regression on these two tasks.

## 8. Caveats

- Different harness from `bench/` — see the method note above. Not cell-for-cell comparable
  with this repo's existing tables.
- Cross-engine comparisons (llama.cpp vs vLLM) compare very different schedulers as well as
  different quantization formats.
- c=1 and c=2 sweep points use an unbalanced prompt mix; c>=4 is balanced.
- No NVFP4 no-speculation concurrency curve was swept, so §5 gives absolute numbers for those
  rows rather than speedups.
- Both upstream PRs were open and unmerged at time of writing; behaviour may change.

See [SOURCES.md](SOURCES.md).
