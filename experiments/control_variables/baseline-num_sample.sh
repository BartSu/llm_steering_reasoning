#! /bin/bash
experiment_name="deepseek-baseline"
model_name=("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
dataset=("MATH500")
temperature=(0.7) # when temperature is 0, the num_sample has to be 1
max_tokens=(8192)
num_samples=(8 16)

for model in ${model_name[@]}; do
    for ds in ${dataset[@]}; do
        for num_samples in ${num_samples[@]}; do
            for temperature in ${temperature[@]}; do
                for max_tokens in ${max_tokens[@]}; do
                    python scripts/run_baseline.py \
                        --model_name_or_path $model \
                        --dataset $ds \
                        --num_samples $num_samples \
                        --temperature $temperature \
                        --max_tokens $max_tokens \
                        --save_dir results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens
                done
            done
        done
    done
done