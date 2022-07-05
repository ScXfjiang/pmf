python main.py                                                          \
    --dataset="/dataset/ml-100k"	                                    \
    --train_batch_size=1024                                             \
    --test_batch_size=1024                                              \
    --shuffle=True                                                      \
    --num_epoch=500                                                     \
    --lr=0.1                                                            \
    --momentum=0.9                                                      \
    --latent_dim=10                                                     \
    --use_cuda=True
