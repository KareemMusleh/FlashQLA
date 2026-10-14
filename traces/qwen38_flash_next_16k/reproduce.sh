cd /home/kareemm/forks/FlashQLA
env CUDA_VISIBLE_DEVICES=4 TILELANG_CACHE_DIR=/tmp/flashqla-tilelang-cache TRITON_CACHE_DIR=/tmp/flashqla-triton-cache /opt/ydt/bin/python profile/trace_gdn.py --config traces/qwen38_flash_next_16k/gdn_config.json --output traces/qwen38_flash_next_16k
