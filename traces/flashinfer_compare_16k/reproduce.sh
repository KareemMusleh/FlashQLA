cd /home/kareemm/forks/FlashQLA
env CUDA_VISIBLE_DEVICES=1 TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache TRITON_CACHE_DIR=/tmp/flashqla-triton-cache /opt/ydt/bin/python traces/flashinfer_compare_16k/benchmark.py
