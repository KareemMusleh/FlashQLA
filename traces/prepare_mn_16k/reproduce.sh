#!/bin/sh
set -eu
cd /home/kareemm/forks/FlashQLA
CUDA_VISIBLE_DEVICES=1 TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache TRITON_CACHE_DIR=/tmp/flashqla-triton-cache /opt/ydt/bin/python profile/bench_prepare_mn.py
