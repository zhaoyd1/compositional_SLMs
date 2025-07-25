# Take Gigaword as an example
export TOKENIZERS_PARALLELISM=false

start=0
end=4
# Only checkpoints of the last 5 finetune epochs are saved. 
base_path=save/dial_gpt2_5

for i in $(seq $start $end)
do
  pretrain_path="${base_path}/model${i}.bin"
  CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 trainer/dial_trainer_eval_gpt2.py \
    --model_type gpt \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small/config.json \
    --vocab_dir data/gpt2-small \
    --summary_dir xsum_gpt2 \
    --output_dir save/dial_gpt2_eval_5 \
    --pretrain_dir "$pretrain_path" \
    --log_step 100 \
    --save_step 1000000000 \
    --epochs 1 \
    --batch_size 64 \
    --eval_batch_size 16 \
    --coeff_start 1.0 \
    --coeff_end 1.0 \
    --coeff_proportion 1.0 \
    --pool_size 1 \
    --beam_size 2 \
    --document_threshold 600 \
    --summary_threshold 70 \
    --eval_perepoch
done