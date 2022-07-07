#!/bin/bash -l
cd $SLURM_SUBMIT_DIR
python ../main.py                                                       \
    --dataset="/dataset/ml-20m"	                                        \
    --train_batch_size=20000                                            \
    --test_batch_size=20000                                             \
    --shuffle=True                                                      \
    --num_epoch=20                                                      \
    --lr=0.1                                                            \
    --momentum=0.9                                                      \
    --weight_decay=1e-4                                                 \
    --latent_dim=30                                                     \
    --use_cuda=True > stdout_train_ml20m.txt 2> stderr_train_ml20m.txt
