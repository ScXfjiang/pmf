python main.py                                                          \
    --dataset="/dataset/ml-100k"	    	                            \
    --train_batch_size=1024                                             \
    --test_batch_size=1024                                              \
    --shuffle=True                                                      \
    --num_epoch=500                                                     \
    --lr=1.0                                                            \
    --latent_dim=20                                                     \
    --use_cuda=True
