# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=2 trainer/dial_trainer_tg_seq.py \
    --model_type tg \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg/config.json \
    --vocab_dir data/gpt2-small-tg \
    --summary_dir xsum_1111 \
    --output_dir save/dial_tg_seq_5 \
    --pretrain_dir save/tg_seq/model.bin \
    --log_step 20 \
    --save_step 1000000000 \
    --epochs 5 \
    --batch_size 16 \
    --lr 5e-5 \
    --parser_lr 1e-3 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --pool_size 4 \
    --word_sync \
    --gradient_checkpoint \
    --document_threshold 1600 \
    --summary_threshold 400 \
    --max_seq_len 2048 