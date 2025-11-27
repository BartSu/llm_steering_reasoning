#!/bin/bash
experiment_name="deepseek-seal"
model_name=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
dataset=("aime25" "aime24" "MATH500" "AMO-Bench" "minervamath" "olympiadbench")
temperature=(0.1 0.3)
max_tokens=(10000)
num_samples=(100)
vector_scale=(1)

for model in ${model_name[@]}; do
    for ds in ${dataset[@]}; do
        for num_samples in ${num_samples[@]}; do
            for temperature in ${temperature[@]}; do
                for max_tokens in ${max_tokens[@]}; do
                    for scale in ${vector_scale[@]}; do
                        if [ ! -f "results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/$scale/predictions.jsonl" ]; then
                            echo "Running seal for $model $ds $num_samples $temperature $max_tokens $scale"
                            python scripts/run_seal.py \
                                --model_name_or_path $model \
                                --dataset $ds \
                                --temperature $temperature \
                                --max_tokens $max_tokens \
                                --num_samples $num_samples \
                                --vector_dir ./vectors/seal/$model \
                                --scale $scale \
                                --save_dir results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/$scale
                            fi
                        done
                    done
                done
            done
        done
    done
done