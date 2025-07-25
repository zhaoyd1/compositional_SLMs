export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=4 trainer/ddp_trainer_nosync.py \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --lr 5e-5 \
    --corpus_path corpus/bllip_supar_train \
    --dev_path corpus/bllip \
    --output_dir save/supar_train_sent_1  \
    --batch_size 128 \
    --accumulation_steps 1 \
    --model_type r2d2-gen-fast \
    --epoch 20 \
    --gradient_checkpoint \
    --log_step 50 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --temperature_end 1.0 \
    --pool_size 1 \
    --save_step 10000 \
    --eval_step 1000 \
    --left 1 \
    --train_sent 1
