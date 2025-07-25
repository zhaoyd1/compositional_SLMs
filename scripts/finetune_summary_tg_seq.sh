# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes=1 --nproc_per_node=3 trainer/xsum_trainer_tg_seq.py \
    --model_type tg \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg/config.json \
    --vocab_dir data/gpt2-small-tg \
    --summary_dir xsum_1111 \
    --output_dir save/xsum_tg_seq_15 \
    --pretrain_dir save/tg_seq_1/ \
    --log_step 20 \
    --save_step 1000000000 \
    --epochs 15 \
    --batch_size 16 \
    --lr 5e-5 \
    --parser_lr 1e-3 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --pool_size 4 \
    --word_sync \
    --gradient_checkpoint \
    --document_threshold 1800 \
    --summary_threshold 200 \
    --max_seq_len 2048 