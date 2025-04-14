seed=${RANDOM}
CUDA_VISIBLE_DEVICES=1 nohup python train_rl.py \
--image_dir data/mimic_cxr/images/ \
--ann_path data/mimic_cxr/annotation.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 4 \
--epochs 35 \
--save_dir /public/home/huarong/yixiulong/RM/Train/RL-AAGT/results/MIMIC-CXR/LHR-RFL-VSMT4 \
--step_size 1 \
--d_vf 2048 \
--gamma 0.8 \
--early_stop 2 \
--num_layers 3 \
--seed ${seed} \
--resume /public/home/huarong/yixiulong/RM/R2Gen-AAGT/results/MIMIC-CXR/model_best.pth