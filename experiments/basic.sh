#!/bin/bash
experiment_name="deepseek-basic"
model_name=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
dataset=("aime25")
temperature=(0.1)
max_tokens=(10000)
num_samples=(100)
vector_scale=(1)
pair_number=(3)

for model in ${model_name[@]}; do
    for ds in ${dataset[@]}; do
        for num_samples in ${num_samples[@]}; do
            for temperature in ${temperature[@]}; do
                for max_tokens in ${max_tokens[@]}; do
                    for scale in ${vector_scale[@]}; do
                            for pair_number in ${pair_number[@]}; do
                            python scripts/run_basic.py \
                                --model_name_or_path $model \
                                --dataset $ds \
                                --temperature $temperature \
                                --max_tokens $max_tokens \
                                --num_samples $num_samples \
                                --vector_dir ./vectors/basic/$model/$pair_number \
                                --scale $scale \
                                --save_dir results/$experiment_name/reason-gguf-$pair_number/$model/$ds/$num_samples/$temperature/$max_tokens/$scale
                        done
                    done
                done
            done
        done
    done
done