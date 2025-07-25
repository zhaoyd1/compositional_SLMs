# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

start=14
end=14
# Only checkpoints of the last 5 finetune epochs are saved. 
base_path=/HOME/paratera_xy/pxy367/models_all/xsum_model_0000.bin

for i in $(seq $start $end)
do
  pretrain_path="${base_path}"
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 trainer/xsum_trainer_eval_0000.py \
    --model_type r2d2-gen-fast \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --summary_dir 0000 \
    --output_dir save/xsum_0000_eval_1000 \
    --pretrain_dir "$pretrain_path" \
    --log_step 100 \
    --save_step 1000000000 \
    --epochs 1 \
    --batch_size 64 \
    --beam_size 2 \
    --eval_batch_size 32 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --coeff_proportion 1.0 \
    --pool_size 3 \
    --document_threshold 600 \
    --summary_threshold 70 \
    --seed 1000 \
    --word_sync \
    --eval_perepoch
done