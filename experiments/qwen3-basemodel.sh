#! /bin/bash
model_name=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B")
dataset=("gsm8k_test" "gsm8k-platinum_test")
num_samples=(1 32 64 128)
temperature=(0 0.6 1.0 1.2)
max_tokens=(16384)

for model in ${model_name[@]}; do
    for dataset in ${dataset[@]}; do
        for num_samples in ${num_samples[@]}; do
            for temperature in ${temperature[@]}; do
                for max_tokens in ${max_tokens[@]}; do
                    python scripts/run_baseline.py \
                        --model_name_or_path $model \
                        --dataset $dataset \
                        --num_samples $num_samples \
                        --temperature $temperature \
                        --max_tokens $max_tokens \
                        --save_dir results/qwen3-basemodel/$model/$dataset/$num_samples/$temperature/$max_tokens
                done
            done
        done
    done
done