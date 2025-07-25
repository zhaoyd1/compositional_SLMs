
for i in 10 30 100 300
do
CUDA_VISIBLE_DEVICES=0 python eval/sg_evaluator_1111.py \
    --model_type tg \
    --r2d2_config_path data/en_config/r2d2_256_4_1_supervised.json \
    --gpt_config_path data/gpt2-small-tg/config.json \
    --vocab_dir data/gpt2-small-tg \
    --pretrain_dir /HOME/paratera_xy/pxy367/models_all/model_tg_seq.bin \
    --task_dir sg_processed \
    --beam_size $i \
    --output_path sg_result/tg_seq_eff.txt \
    --task_path sg_processed/fgd_hierarchy.json
    # --alltest
done