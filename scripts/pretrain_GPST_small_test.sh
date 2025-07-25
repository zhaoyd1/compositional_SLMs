export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc_per_node=1 trainer/ddp_trainer_nosync.py \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --lr 5e-5 \
    --corpus_path corpus/bllip_dev \
    --output_dir save/r2d2_gen_fast_supervised_wiki103_small_test \
    --batch_size 2 \
    --accumulation_steps 1 \
    --model_type r2d2-gen-fast \
    --num_samples 400 \
    --max_seq_len 50 \
    --gradient_checkpoint \
    --log_step 50 \
    --temperature_end 1.0 \
    --pool_size 1 \
    --save_step 10000