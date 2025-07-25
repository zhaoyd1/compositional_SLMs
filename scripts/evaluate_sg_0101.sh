python eval/sg_evaluator_0101.py \
    --model_type tg \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg/config.json \
    --vocab_dir data/gpt2-small-tg \
    --pretrain_dir save/0101_1/ \
    --task_dir sg_processed \
    --output_path sg_result/0101_1.txt \
    --cuda_num 2 \
    --tg_enabled 1 \
    --alltest
    # --task_path sg_processed/center_embed.json
