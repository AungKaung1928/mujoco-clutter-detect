#!/usr/bin/env bash
# Safety net: if the chat is closed before the chain finishes, this still lands
# the artifacts in git. Detached (setsid), so it does not die with the terminal.
# If the session is still open and the results were committed by hand first,
# this finds nothing to commit and exits quietly.
set -u
cd "$(dirname "$0")"
source ~/personal/ml/env.sh

# wait up to 4 h for the chain, then give up rather than hang forever
for _ in $(seq 1 960); do
  grep -q "all done" runs/log_chain.txt 2>/dev/null && break
  sleep 15
done

[ -f runs/det_hard_none.pt ] && \
  nice -n 10 python view_cnn.py --ckpt runs/det_hard_none.pt --regime hard \
      --out out/cnn_detections.png > runs/log_fig.txt 2>&1

{
  echo "generated $(date -Iseconds)"
  for f in runs/det_hard_none.json runs/det_easy_none.json runs/ablation.json runs/onnx.json; do
    [ -f "$f" ] && { echo; echo "--- $f ---"; cat "$f"; }
  done
} > runs/SUMMARY.txt 2>/dev/null

git add -A runs/*.json runs/SUMMARY.txt out/*.png 2>/dev/null
if ! git diff --cached --quiet; then
  git commit -q -m "block 2: step 3-5 result artifacts (metrics json, detection figure)

Committed by the unattended chain. README prose for these steps is written
separately -- these are the raw numbers so nothing is lost if the session ends."
  git push -q origin master && echo "pushed $(date -Iseconds)" >> runs/log_chain.txt
fi
