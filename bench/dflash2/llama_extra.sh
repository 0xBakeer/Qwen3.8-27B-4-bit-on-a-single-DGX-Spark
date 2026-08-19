#!/usr/bin/env bash
# Complete the Q4_K_M row: MTP and DSpark drafters on the SAME 4-bit weights,
# which vLLM could not do because DFlash2 was blocked there.
set -uo pipefail
cd /home/bakeer/dflash2-bench
LC=/home/bakeer/llama.cpp
HUB=/home/bakeer/models/hf/hub
gguf() { find $HUB -name "$1" 2>/dev/null | head -1; }
TARGET=$(gguf "Qwen3.8-27B-Q4_K_M.gguf"); MTP_Q4=$(gguf "mtp-Qwen3.8-27B-Q4_0.gguf")
DF2=$(gguf "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")

serve() { local tag="$1"; shift
  pkill -f llama-server 2>/dev/null; sleep 3
  echo; echo "################ $tag ################"; date -Is
  nohup $LC/build/bin/llama-server -m "$TARGET" -ngl 999 -c 32768 -fa on \
    --host 127.0.0.1 --port 8080 --metrics --reasoning-budget 0 "$@" > logs/$tag.serverlog 2>&1 &
  for i in $(seq 1 40); do
    curl -sf --max-time 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "[$tag] healthy"; return 0; }
    pgrep -f llama-server >/dev/null || { echo "[$tag] SERVER DIED"; tail -25 logs/$tag.serverlog; return 1; }
    sleep 10; done
  echo "[$tag] TIMEOUT"; return 1; }

run() { local tag="$1" suite="$2"; shift 2
  if serve "$tag" "$@"; then
    local s=bench_dflash.py mt=512; [ "$suite" = edit ] && { s=run_edit.py; mt=1200; }
    python3 $s --tag "$tag" --base http://127.0.0.1:8080/v1 --metrics http://127.0.0.1:8080/metrics \
      --model qwen3.8-27b --reps 3 --max-tokens $mt --temp 0 --batch 8 || echo "[$tag] BENCH FAILED"
  fi; }

run gguf-q4km-mtp3    gen  -md "$MTP_Q4" --spec-type draft-mtp    --spec-draft-n-max 3
run gguf-q4km-mtp7    gen  -md "$MTP_Q4" --spec-type draft-mtp    --spec-draft-n-max 7
run gguf-q4km-mtp3-ed edit -md "$MTP_Q4" --spec-type draft-mtp    --spec-draft-n-max 3
run gguf-q4km-df2k5   gen  -md "$DF2"    --spec-type draft-dflash --spec-draft-n-max 5
pkill -f llama-server 2>/dev/null
echo; echo "######## LLAMA EXTRA DONE ########"; date -Is
