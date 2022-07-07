#!/bin/bash -l
cd $SLURM_SUBMIT_DIR
python ../main.py                                                       \
    --dataset="/dataset/ml-1m"	                                        \
    --train_batch_size=1024                                             \
    --test_batch_size=1024                                              \
    --shuffle=True                                                      \
    --num_epoch=200                                                     \
    --lr=0.1                                                            \
    --momentum=0.9                                                      \
    --weight_decay=1e-4                                                 \
    --latent_dim=15                                                     \
    --use_cuda=True > stdout.txt 2> stderr.txt
