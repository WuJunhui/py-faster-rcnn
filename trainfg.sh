#!/bin/bash

export PYTHONUNBUFFERED="True"
EXP_DIR_NAME=workdir_fg

LOG="${EXP_DIR_NAME}/logs/train${EXP_DIR_NAME}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 1 \
  --solver ${EXP_DIR_NAME}/solver.prototxt \
  --imdb vocfg_2018_trainval \
  --iters 200000 \
  --cfg ${EXP_DIR_NAME}/train.yml \
  --weights ${EXP_DIR_NAME}/ZF_faster_rcnn_final.caffemodel \
  --set EXP_DIR ${EXP_DIR_NAME}
