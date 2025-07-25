python eval/sg_evaluator_0000.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --pretrain_dir save/0000_right_1/model_0000.bin \
    --task_dir sg_processed \
    --output_path sg_result/0000_right.txt \
    --alltest
    # --task_path sg_processed/center_embed.json