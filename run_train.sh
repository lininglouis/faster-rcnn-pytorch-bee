#!/bin/bash

WORKER_NUMBER=8
BATCH_SIZE=16
LEARNING_RATE=4e-3
DECAY_STEP=8
MAX_EPOCHS=20
python -u trainval_net.py \
                   --dataset colony --net res101  \
                   --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                   --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                   --epochs $MAX_EPOCHS \
                   --cuda --mGPUs
