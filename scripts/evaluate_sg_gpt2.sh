python eval/sg_evaluator_gpt2.py \
    --model_type gpt \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --pretrain_dir save/gpt2_baseline_1 \
    --task_dir sg_processed \
    --output_path sg_result/gpt2_baseline_1.txt \
    --cuda_num 6 \
    --alltest
    # --task_path sg_processed/center_embed.json