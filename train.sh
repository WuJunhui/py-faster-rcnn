#!/bin/bash

export PYTHONUNBUFFERED="True"
EXP_DIR_NAME=workdir_fpn

LOG="${EXP_DIR_NAME}/logs/train${EXP_DIR_NAME}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_net.py --gpu 1 \
  --solver ${EXP_DIR_NAME}/solver.prototxt \
  --imdb voc_2007_trainval \
  --iters 90000 \
  --cfg ${EXP_DIR_NAME}/train.yml \
  --weights ${EXP_DIR_NAME}/ResNet-50-model.caffemodel
  --set EXP_DIR ${EXP_DIR_NAME}
  #--weights ${EXP_DIR_NAME}/shufflenet_1x_g3.caffemodel \
