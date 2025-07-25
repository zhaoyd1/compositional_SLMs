python eval/sg_evaluator_tg_seq.py \
    --model_type tg \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg/config.json \
    --vocab_dir data/gpt2-small-tg \
    --pretrain_dir save/tg_seq_1/ \
    --task_dir sg_processed_xx1x \
    --output_path sg_result/tg_seq_1.txt \
    --cuda_num 6 \
    --alltest
    # --task_path sg_processed/center_embed.json