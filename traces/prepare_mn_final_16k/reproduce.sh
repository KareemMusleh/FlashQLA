#!/bin/sh
set -eu
cd /home/kareemm/forks/FlashQLA
CUDA_VISIBLE_DEVICES=3 TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache TRITON_CACHE_DIR=/tmp/flashqla-triton-cache /opt/ydt/bin/python profile/bench_prepare_mn.py --flashinfer /tmp/flashinfer-gdn-compare --output traces/prepare_mn_final_16k
