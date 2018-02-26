#!/bin/bash
export PYTHONUNBUFFERED="True"
EXP_DIR_NAME=workdir_shuf
LOG="${EXP_DIR_NAME}/logs/test${EXP_DIR_NAME}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/test_net.py \
    --gpu 2 \
    --def ${EXP_DIR_NAME}/test.prototxt \
    --net ${EXP_DIR_NAME}/output/voc_2007_trainval/shuf_faster_rcnn_iter_60000.caffemodel \
    --cfg ${EXP_DIR_NAME}/train.yml \
    --set EXP_DIR ${EXP_DIR_NAME} \
