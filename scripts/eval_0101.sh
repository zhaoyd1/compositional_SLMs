export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=1 --nproc_per_node=1 trainer/ddp_train_0101.py \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg/config.json \
    --vocab_dir data/gpt2-small-tg \
    --lr 5e-5 \
    --corpus_path corpus/gpst_seq_train \
    --output_dir save/0101_doc_1 \
    --batch_size 2 \
    --accumulation_steps 1 \
    --model_type tg \
    --num_samples 100 \
    --gradient_checkpoint \
    --log_step 5 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --temperature_end 1.0 \
    --pool_size 1 \
    --save_step 10000 \
    --eval_mode 2 \
    --left 1 \
    --tg_enabled 1 \
    --pretrain_dir save/gpst_0101_1/