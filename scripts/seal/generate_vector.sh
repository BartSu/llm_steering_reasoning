python3 scripts/seal/eval_MATH_vllm.py \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --save_dir vectors/qwen3-seal/Qwen/Qwen3-0.6B \
    --max_tokens 10000 \
    --use_chat_format \
    --dataset "MATH_train" \
    --remove_bos

python3 scripts/seal/hidden_analysis.py \
    --model_path Qwen/Qwen3-0.6B \
    --data_path data/SEAL_MATH/train.jsonl \
    --data_dir vectors/qwen3-seal/Qwen/Qwen3-0.6B \
    --type incorrect \
    --start 0 \
    --sample 500 \

python3 scripts/seal/hidden_analysis.py \
    --model_path Qwen/Qwen3-0.6B \
    --data_path data/SEAL_MATH/train.jsonl \
    --data_dir vectors/qwen3-seal/Qwen/Qwen3-0.6B \
    --type correct \
    --start 0 \
    --sample 500 \

python3 scripts/seal/vector_generation.py \
    --data_dir vectors/qwen3-seal/Qwen/Qwen3-0.6B \
    --prefixs correct_0_500 incorrect_0_500 \
    --layers 20 \
    --save_prefix 500_500 