python eval/sg_evaluator_0001.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --pretrain_dir save/0001_1/model.bin \
    --task_dir sg_processed \
    --output_path sg_result/0001_1.txt \
    --alltest
    # --task_path sg_processed/center_embed.json