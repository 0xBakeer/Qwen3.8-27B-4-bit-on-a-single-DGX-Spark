#!/usr/bin/env bash
# llama.cpp leg: TRUE Q4_K_M target + DFlash2 Q4 drafter -- the 4-bit + DFlash2
# combination that vLLM blocks on the unquantized-LM-head constraint.
set -uo pipefail
cd /home/bakeer/dflash2-bench
LC=/home/bakeer/llama.cpp
HUB=/home/bakeer/models/hf/hub
export PATH=/usr/local/cuda/bin:$PATH

gguf() { find $HUB -name "$1" 2>/dev/null | head -1; }
TARGET=$(gguf "Qwen3.8-27B-Q4_K_M.gguf")
DF2_Q4=$(gguf "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")
DF2_Q8=$(gguf "Qwen3.8-27B-DFlash2-Q8_0.gguf")
MTP_Q4=$(gguf "mtp-Qwen3.8-27B-Q4_0.gguf")
echo "target : $TARGET"; echo "dflash2: $DF2_Q4"; echo "df2-q8 : $DF2_Q8"; echo "mtp    : $MTP_Q4"
[ -z "$TARGET" ] && { echo "TARGET GGUF MISSING"; exit 1; }

echo "=== building llama.cpp (PR 27342) ==="; date -Is
cmake --build $LC/build -j20 --target llama-server 2>&1 | tail -6
[ -x $LC/build/bin/llama-server ] || { echo "BUILD FAILED"; exit 1; }
echo "=== build done ==="; date -Is
echo "--- speculative flags exposed ---"
$LC/build/bin/llama-server --help 2>&1 | grep -iE "spec|draft" | sed 's/^/    /'

serve() {  # tag  extra-args...
  local tag="$1"; shift
  pkill -f "llama-server" 2>/dev/null; sleep 3
  echo; echo "################ $tag ################"; date -Is
  nohup $LC/build/bin/llama-server -m "$TARGET" -ngl 999 -c 32768 -fa on \
      --host 127.0.0.1 --port 8080 --metrics --reasoning-budget 0 \
      "$@" > logs/$tag.serverlog 2>&1 &
  for i in $(seq 1 60); do
    curl -sf --max-time 3 http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "[$tag] healthy (~$((i*10))s)"; return 0; }
    pgrep -f llama-server >/dev/null || { echo "[$tag] SERVER DIED"; tail -30 logs/$tag.serverlog; return 1; }
    sleep 10
  done
  echo "[$tag] TIMEOUT"; tail -30 logs/$tag.serverlog; return 1
}

run() {  # tag suite extra-args...
  local tag="$1" suite="$2"; shift 2
  if serve "$tag" "$@"; then
    local script=bench_dflash.py; [ "$suite" = edit ] && script=run_edit.py
    local mt=512;               [ "$suite" = edit ] && mt=1200
    python3 $script --tag "$tag" --base http://127.0.0.1:8080/v1 \
       --metrics http://127.0.0.1:8080/metrics --model qwen3.8-27b \
       --reps 3 --max-tokens $mt --temp 0 --batch 8 || echo "[$tag] BENCH FAILED"
    grep -iE "draft acceptance|n_draft|accept" logs/$tag.serverlog | tail -5
  fi
}

run gguf-q4km-nospec     gen
run gguf-q4km-dflash2    gen  -md "$DF2_Q4" --spec-type draft-dflash --spec-draft-n-max 7
run gguf-q4km-dflash2q8  gen  -md "$DF2_Q8" --spec-type draft-dflash --spec-draft-n-max 7
run gguf-q4km-dflash2ed  edit -md "$DF2_Q4" --spec-type draft-dflash --spec-draft-n-max 7
run gguf-q4km-nospec-ed  edit
pkill -f llama-server 2>/dev/null
echo; echo "######## LLAMA LEG DONE ########"; date -Is
