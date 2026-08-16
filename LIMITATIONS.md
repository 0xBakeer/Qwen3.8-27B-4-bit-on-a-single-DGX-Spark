# Limitations and methodology caveats

Everything in [RESULTS.md](RESULTS.md) is a real measurement taken on real hardware. This
document records what those measurements do **not** establish, so nobody — including
future readers of this repository — over-reads them.

## 1. No quality evaluation was performed here

**This repository measures speed only.** Every quality figure quoted in the README
(MMLU recovery 99.32 %, HumanEval 39/40, perplexity deltas) is **cited from the
checkpoint publishers and was not independently verified.**

That distinction matters more than usual for this model family. A same-family checkpoint
passed standard aggregate benchmarks and then failed a concrete code task badly enough to
justify a months-long "16-bit weights only" policy. Published benchmark scores did not
predict that outcome.

**Do not treat the throughput numbers here as an endorsement of any 4-bit checkpoint's
output quality.** Run your own representative task before promoting one.

The speculative-decoding and prefix-caching results need no such caveat — the target model
verifies every drafted token, so output is preserved by construction, and identical
prompts produced byte-identical completions across configurations.

## 2. The stock baseline is not a controlled comparison

The `7.88 tok/s` figure that all the multipliers are computed against was measured with
the model **sharing the device with three other (idle) engines at
`gpu-memory-utilization 0.44`**. The 47–75 tok/s figures were measured with the model as
**sole occupant at `gpu-memory-utilization 0.85`**.

A stock, no-speculation baseline was never re-measured in the dedicated configuration.

Evidence suggests this does **not** inflate the multipliers: the same MTP `k=3` config
measured 19.00 shared vs 17.70 dedicated, i.e. dedicating the device was slightly
*slower*, not faster. So a dedicated stock baseline would likely be ≤ 7.88 and the true
multiplier would be equal or larger. But it was not measured, and "9.5×" therefore carries
a configuration change it does not advertise.

## 3. Sample sizes are small and there are no error bars

Each cell is **one or two runs**. No repetitions beyond that, no variance analysis, no
confidence intervals.

Observed run-to-run spread was ~1 % on stable configurations and up to ~12 % where a warm
prefix cache was involved (MixedInt4: 56.69 then 63.67 on identical consecutive prompts).
Treat differences smaller than ~5 % as noise. In particular, **the two 4-bit checkpoints
are a statistical tie**, and the repo says so rather than declaring a winner.

## 4. Derived quantities are derived, not measured

The cost-per-draft-token figures (MTP 0.153, DSpark 0.046) are **arithmetic on measured
throughput and mean tokens-per-pass**, assuming base decode of 7.88 forward passes/sec.
They were not measured by instrumenting the drafter directly.

Any statement about an "asymptotic ceiling" extrapolated from those coefficients is a
model, not data — and it was extrapolated from a single data point. The `k=14` result
later showed the underlying curve is more favourable than the first extrapolation
suggested, which is exactly the failure mode you would expect from one-point fits.

## 5. The edit-heavy workload is synthetic and probably optimistic

The benchmark asks the model to reproduce a source file of 45 near-identical dataclasses
with one method added. That is unusually repetitive, so drafter acceptance (98–99 % at
`k=7`) sits at the favourable end of what real work produces.

Real refactors — heterogeneous files, genuine logic changes, mixed prose and code — should
be expected to land **below** these numbers. The fresh-generation column is the pessimistic
bound and real workloads will fall between the two columns, not at the edit-heavy one.

## 6. Draft depth trades single-stream speed against concurrency

`k=14` is faster than `k=7` on a single stream and **slower under concurrency**:

| | Single stream | c4 aggregate | c8 aggregate |
|---|---:|---:|---:|
| DSpark `k=7` | 59.1 | **167.7** | **246.0** |
| DSpark `k=14` | **75.0** | 157.5 | 218.7 |

Deeper drafting consumes scheduling capacity that would otherwise serve additional
sequences. So there is **no single best `k`** — it depends on whether you optimise for
interactive latency or fleet throughput. A README that quoted only 75.0 would be
misleading for anyone running parallel agents.

## 7. Coverage gaps

Closed since the first revision: concurrency for both checkpoints (c1–c16), prefill curves
for both checkpoints, and the `k=7`/`k=14` batch-budget control.

Still open:

- **Concurrency was not tested beyond c16.** Aggregate throughput was still rising at c16
  on both builds (223.29 and 256.47), so the peak is above what was measured. Per-stream
  had roughly halved from c8, so the useful operating range is likely already behind us.
- **`k=14` was tested on 4-bit weights only** when this section was written; see the
  companion 8-bit repository for whether FP8 behaves the same way.
- **Fresh generation is the noisiest measurement here.** The same MixedInt4 weights
  produced 29.12 and 25.96 tok/s in two runs with acceptance of 32.8 % and 27.0 %. Low
  acceptance means high variance — treat the fresh-generation column as ±15 %, not ±1 %.
  The edit-heavy and prefill columns are far more reproducible (prefill agreed to 0.05 %).
- **Multimodal paths were not benchmarked**, despite the checkpoints shipping vision
  towers.
- **Prefix caching on hybrid models is documented upstream as "opt-in while the feature
  matures".** Our correctness check was one deterministic prompt comparison, not a proof.
- **Draft depth was tested at `k=7` and `k=14` only.** Nothing between or beyond, and
  `k=14` was chosen as 2× the drafter's native `block_size 7` rather than by search.

## 8. Single hardware sample, single software version

One DGX Spark, vLLM `v0.27.1-aarch64`, one driver branch. Nothing here has been
reproduced on a second device or a second engine version. Kernel selection on SM121 is
version-sensitive, and several of the gotchas in [NOTES.md](NOTES.md) are themselves
version-specific — some may be fixed, or newly broken, in other releases.
