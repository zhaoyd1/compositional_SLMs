
python eval/sg_evaluator_1010.py \
    --model_type n-ary-gen \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg-pointer/config.json \
    --vocab_dir data/gpt2-small-tg-pointer \
    --pretrain_dir save/1010_1/model.bin \
    --task_dir sg_processed_xx1x \
    --output_path sg_result/1010_1.txt \
    --cuda_num 5 \
    --alltest
    # --task_path sg_processed/center_embed.json