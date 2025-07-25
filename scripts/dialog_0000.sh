# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=2 trainer/dial_trainer_0000.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --summary_dir 0000 \
    --output_dir save/dial_0000_5 \
    --pretrain_dir save/0000_1/model.bin \
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
    --document_threshold 600 \
    --summary_threshold 150