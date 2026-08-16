#!/usr/bin/env bash
# Serve 4-bit Qwen3.8-27B on a single DGX Spark (GB10 / SM121) with vLLM.
#
# Defaults to the Unsloth NVFP4 checkpoint with DSpark k=7 speculative decoding and
# prefix caching. See RESULTS.md for the measurements behind these choices.
#
#   MODEL=<repo>            swap checkpoints (the AutoRound int4 build is the
#                           alternative measured here - same speed, 1.8x less KV cache,
#                           but it publishes quality numbers and Unsloth's does not)
#   SPEC=dspark (default)   external 5-layer drafter, block_size 7
#   SPEC=mtp                in-checkpoint head
#   SPEC=off                no speculative decoding
set -euo pipefail

MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
DRAFTER="${DRAFTER:-Doopeworld/Qwen3.8-27B-DSpark-vLLM}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.27.1-aarch64}"
NAME="${NAME:-qwen38-4bit}"
PORT="${PORT:-8002}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"

HF_CACHE="${HF_CACHE:-$HOME/models/hf}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/models/vllm-cache}"

GMU="${GMU:-0.85}"
MAX_LEN="${MAX_LEN:-262144}"

# Raise this when raising k: draft slots come out of the batch token budget, and
# k * max_num_seqs exceeding it makes max_num_scheduled_tokens negative at startup.
MAX_BATCHED="${MAX_BATCHED:-16384}"

SPEC="${SPEC:-dspark}"
K="${K:-7}"

case "$SPEC" in
  dspark) SPEC_CFG="{\"method\":\"dspark\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\"}" ;;
  mtp)    SPEC_CFG="{\"method\":\"mtp\",\"num_speculative_tokens\":$K}" ;;
  off)    SPEC_CFG="" ;;
  *)      echo "SPEC must be one of: dspark, mtp, off" >&2; exit 2 ;;
esac

mkdir -p "$HF_CACHE" "$VLLM_CACHE"
docker rm -f "$NAME" >/dev/null 2>&1 || true

ARGS=(
  serve "$MODEL"
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GMU"
  --max-num-batched-tokens "$MAX_BATCHED"
  --enable-prefix-caching
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --limit-mm-per-prompt.image 2
  --limit-mm-per-prompt.video 0
)
[ -n "$SPEC_CFG" ] && ARGS+=( --speculative-config "$SPEC_CFG" )

echo "starting $NAME :: $MODEL :: spec=$SPEC k=$K gmu=$GMU"

# VLLM_MARLIN_USE_ATOMIC_ADD is not optional on SM121. Without it a race in the Marlin
# kernel produces INCORRECT OUTPUT rather than an error - and 4-bit weights are
# dequantized through Marlin on this device, so it sits directly in the decode path.
# VLLM_USE_FLASHINFER_MOE_FP4=0 keeps the MoE path away from CUTLASS FP4, which is
# reported to emit silent garbage here. Unused by this dense model; cheap insurance.
docker run -d --name "$NAME" --gpus all --ipc host \
  -p "127.0.0.1:$PORT:$PORT" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$VLLM_CACHE":/root/.cache/vllm \
  --entrypoint vllm "$IMAGE" "${ARGS[@]}" >/dev/null

echo -n "waiting for readiness"
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo; echo "ready on http://127.0.0.1:$PORT/v1  (model: $SERVED_NAME)"
    docker logs "$NAME" 2>&1 | grep -o "GPU KV cache size: [0-9,]*" | tail -1
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo; echo "container exited. last errors:" >&2
    docker logs "$NAME" 2>&1 | grep -iE "ValueError|Value error|ImportError|Error:" | tail -5 >&2
    exit 1
  fi
  echo -n "."
  sleep 5
done

echo; echo "timed out waiting for readiness" >&2
exit 1
