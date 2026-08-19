#!/usr/bin/env bash
# Concurrency sweep 1..16 for the configs that matter. Same flags as the main matrix
# (gmu 0.80, 128K, prefix caching) so the c=8 point reproduces the earlier batch x8 run.
set -uo pipefail
cd /home/bakeer/dflash2-bench
FP8=Qwen/Qwen3.8-27B-FP8
NVFP4=unsloth/Qwen3.8-27B-NVFP4
DF2='{"method":"dflash","model":"incoai/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}'
MTP='{"method":"mtp","num_speculative_tokens":3}'
DSP='{"method":"dspark","model":"Doopeworld/Qwen3.8-27B-DSpark-vLLM","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
LEV=1,2,4,8,12,16

echo "=== tearing down production stack for the sweep ==="; date -Is
for c in nemotron35-nvfp4 vllm-embed qwen38fp8 vllm-rerank; do docker rm -f $c >/dev/null 2>&1 && echo "  removed $c"; done
sleep 10; free -g | sed -n 2p

vllm_sweep() { # tag model spec
  local tag="$1" model="$2" spec="$3"
  echo; echo "################ sweep $tag ################"; date -Is
  if ./serve_one.sh "$tag" "$model" "$spec"; then
    python3 sweep.py --tag "$tag" --levels $LEV --max-tokens 384 || echo "[$tag] SWEEP FAILED"
  else echo "[$tag] SERVE FAILED"; fi
}

vllm_sweep fp8-dflash2   "$FP8"   "$DF2"
vllm_sweep fp8-nospec    "$FP8"   none
vllm_sweep nvfp4-mtp3    "$NVFP4" "$MTP"
vllm_sweep nvfp4-dspark7 "$NVFP4" "$DSP"
docker rm -f qwenbench >/dev/null 2>&1; sleep 8

# llama.cpp needs explicit parallel slots (-np); default is 1 and would serialise.
LC=/home/bakeer/llama.cpp; HUB=/home/bakeer/models/hf/hub
TARGET=$(find $HUB -name "Qwen3.8-27B-Q4_K_M.gguf" | head -1)
DF2G=$(find $HUB -name "Qwen3.8-27B-DFlash2-Q4_K_M.gguf" | head -1)
lc_sweep() { # tag extra...
  local tag="$1"; shift
  pkill -f "llama-serv" 2>/dev/null; sleep 3
  echo; echo "################ sweep $tag ################"; date -Is
  nohup $LC/build/bin/llama-server -m "$TARGET" -ngl 999 -c 65536 -np 16 -fa on \
    --host 127.0.0.1 --port 8080 --metrics --reasoning-budget 0 "$@" > logs/sweep-$tag.serverlog 2>&1 &
  for i in $(seq 1 40); do
    curl -sf --max-time 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "[$tag] healthy"; break; }
    pgrep -f "llama-serv" >/dev/null || { echo "[$tag] DIED"; tail -20 logs/sweep-$tag.serverlog; return 1; }
    sleep 10
  done
  python3 sweep.py --tag "$tag" --base http://127.0.0.1:8080/v1 \
     --metrics http://127.0.0.1:8080/metrics --levels $LEV --max-tokens 384 || echo "[$tag] SWEEP FAILED"
}
lc_sweep gguf-q4km-dflash2 -md "$DF2G" --spec-type draft-dflash --spec-draft-n-max 7
lc_sweep gguf-q4km-nospec
pkill -f "llama-serv" 2>/dev/null

echo; echo "=== restoring production stack ==="; date -Is
PROFILE=b4 /home/bakeer/serve-stack2.sh
echo; echo "######## SWEEP LEG DONE ########"; date -Is
