python eval/sg_evaluator_0111.py \
    --model_type tg \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg-pointer/config.json \
    --vocab_dir data/gpt2-small-tg-pointer \
    --pretrain_dir save/0111_1/ \
    --task_dir sg_processed_xx1x \
    --output_path sg_result/0111_4.txt \
    --cuda_num 1 \
    --alltest
    # --task_path sg_processed/center_embed.json