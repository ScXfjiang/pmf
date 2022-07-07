#!/bin/bash -l
cd $SLURM_SUBMIT_DIR
python ../main.py                                                       \
    --dataset="/home/people/22200056/workspace/dataset/ml-10m"          \
    --train_batch_size=8096                                             \
    --test_batch_size=8096                                              \
    --shuffle=True                                                      \
    --num_epoch=500                                                     \
    --lr=0.1                                                            \
    --momentum=0.9                                                      \
    --weight_decay=1e-4                                                 \
    --latent_dim=30                                                     \
    --use_cuda=True > stdout_train_ml10m.txt 2> stderr_train_ml10m.txt
