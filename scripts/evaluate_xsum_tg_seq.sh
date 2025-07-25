# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

start=14
end=14
# Only checkpoints of the last 5 finetune epochs are saved. 
base_path=save/xsum_tg_seq_15

for i in $(seq $start $end)
do
  pretrain_path="${base_path}/model${i}.bin"
  CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nnodes=1 --nproc_per_node=1 trainer/xsum_trainer_eval_tg_seq.py \
    --model_type tg \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg/config.json \
    --vocab_dir data/gpt2-small-tg \
    --summary_dir xsum_1111 \
    --output_dir save/xsum_tg_seq_eval_succ5_beam2_final \
    --pretrain_dir "$pretrain_path" \
    --log_step 100 \
    --save_step 1000000000 \
    --epochs 1 \
    --batch_size 64 \
    --eval_batch_size 32 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --coeff_proportion 1.0 \
    --pool_size 1 \
    --beam_size 2 \
    --document_threshold 1800 \
    --summary_threshold 200 \
    --max_seq_len 2048 \
    --word_sync \
    --eval_perepoch
done