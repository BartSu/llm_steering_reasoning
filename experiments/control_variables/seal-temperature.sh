#!/bin/bash
experiment_name="deepseek-seal"
model_name=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
dataset=("gsm8k_test")
temperature=(0.7 1.0 1.2)
max_tokens=(8192)
num_samples=(1)
vector_scale=(1)

for model in ${model_name[@]}; do
    for ds in ${dataset[@]}; do
        for num_samples in ${num_samples[@]}; do
            for temperature in ${temperature[@]}; do
                for max_tokens in ${max_tokens[@]}; do
                    for scale in ${vector_scale[@]}; do
                        python scripts/run_seal.py \
                            --model_name_or_path $model \
                            --dataset $ds \
                            --temperature $temperature \
                            --max_tokens $max_tokens \
                            --num_samples $num_samples \
                            --vector_dir ./vectors/seal/$model \
                            --scale $scale \
                            --save_dir results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/$scale
                    done
                done
            done
        done
    done
done