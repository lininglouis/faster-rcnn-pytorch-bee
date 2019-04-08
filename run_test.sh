#!/bin/bash


SESSION=1
EPOCH=20
CHECKPOINT=796

python test_net.py --dataset colony --net res101\
    --checksession $SESSION --checkepoch $EPOCH --checkpoint $CHECKPOINT \
    --cuda
