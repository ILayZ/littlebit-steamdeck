#!/usr/bin/env bash
set -euo pipefail
# Run in a configured CUDA environment with authorized Llama-2 weight access.
root=$(cd "$(dirname "$0")" && pwd)
upstream="$root/upstream"
rev=0ceadcfa842b2c26e14a6e00d98d47aed8e53dde
if [[ ! -e "$upstream" ]]; then
  git clone https://github.com/SamsungLabs/LittleBit.git "$upstream"
  git -C "$upstream" checkout --detach "$rev"
fi
[[ $(git -C "$upstream" rev-parse HEAD) == "$rev" ]]
cd "$upstream"
for mode in False True; do
  output="$root/outputs/llama2-7b-itq-$mode"
  python -m main --model_id meta-llama/Llama-2-7b-hf --save_dir "$output" \
    --dataset c4_wiki --num_train_epochs 5 --per_device_train_batch_size 4 \
    --lr 4e-5 --warmup_ratio 0.02 --report tensorboard --quant_func SmoothSign \
    --quant_mod LittleBitLinear --residual True --eff_bit 1.0 --kv_factor 1.0 \
    --min_split_dim 8 --l2l_loss_scale 10.0 --use_itq "$mode" --seed 42
  mapfile -t configs < <(find "$output" -name littlebit_config.json -not -path '*/checkpoint-*/*')
  [[ ${#configs[@]} == 1 ]]
  python eval.py --model_id "$(dirname "${configs[0]}")" --seqlen 2048 \
    --ppl_task wikitext2 --zeroshot_task hellaswag,arc_easy,arc_challenge,piqa,winogrande \
    2>&1 | tee "$output/eval.log"
done
