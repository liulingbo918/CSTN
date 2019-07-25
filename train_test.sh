#!/bin/bash
export HDF5_DISABLE_VERSION_CHECK=1
python train_test.py --model CSTN --lr 0.001 --batch_size 32 --gpus 1
