#! /bin/bash
experiment_name="qwen3-baseline-with-emotion"
model_name=("Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B")
dataset=("gsm8k_test")
temperature=(0 0.1 0.3 0.5 0.7 0.9 1.0 1.2)
max_tokens=(10000)
num_samples=(1)
emotion=("happy" "sad" "angry" "fearful" "disgusted" "superised")

for model in ${model_name[@]}; do
    for ds in ${dataset[@]}; do
        for num_samples in ${num_samples[@]}; do
            for temperature in ${temperature[@]}; do
                for max_tokens in ${max_tokens[@]}; do
                    for emotion in ${emotion[@]}; do
                        # if the directory results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens exists, skip
                        if [ ! -f "results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/predictions.jsonl" ]; then
                            echo "Running baseline for $model $ds $num_samples $temperature $max_tokens"
                            python scripts/run_baseline_with_emotion.py \
                                --model_name_or_path $model \
                                --dataset $ds \
                                --num_samples $num_samples \
                                --temperature $temperature \
                                --max_tokens $max_tokens \
                                --save_dir results/$experiment_name/$model/$ds/$num_samples/$temperature/$max_tokens/$emotion \
                                --emotion $emotion 
                        fi
                    done
                done
            done
        done
    done
done