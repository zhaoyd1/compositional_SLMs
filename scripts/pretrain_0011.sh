export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=4 trainer/ddp_train_0011.py \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg-pointer/config.json \
    --vocab_dir data/gpt2-small-tg-pointer \
    --lr 5e-5 \
    --corpus_path corpus/0010_train \
    --dev_path corpus/bllip \
    --output_dir save/0011_1 \
    --batch_size 8 \
    --accumulation_steps 3 \
    --model_type n-ary-gen \
    --num_samples 2000000 \
    --gradient_checkpoint \
    --log_step 50 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --temperature_end 1.0 \
    --pool_size 1 \
    --save_step 10000 \
    --left 1 \
    --eval_step 1000 \
    --tg_enabled 1 \
    --max_seq_len 2048