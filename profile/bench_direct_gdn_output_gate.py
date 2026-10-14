"""Matched direct-buffer GDN training: eager versus fused output-gate forward.

Both variants use identical direct-write backward and default uncached GDR
backward; only forward RMSNorm/SiLU/multiply differs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch

from bench_direct_gdn_training import Layer as BaseLayer, compare
from flash_qla.ops.gdn_layer import direct_gdn_layer


class Layer(BaseLayer):
    def training_forward(self, mode):
        return direct_gdn_layer(
            self.x, self.weights, self.conv_weights, self.a_log, self.dt_bias,
            self.norm_weight, self.wo, key_heads=self.hk, value_heads=self.hv,
            eps=self.args.norm_eps, partitions=self.args.partitions,
            output_gate_mode="eager" if mode == "eager_gate" else "fused",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlens", nargs="+", type=int, default=[16384,131072])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("traces/gdn_direct_fused_forward/results.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_stream(torch.cuda.Stream())
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__), root/"profile/bench_direct_gdn_training.py",
               root/"flash_qla/ops/gdn_layer.py",
               root/"flash_qla/ops/gated_delta_rule/chunk/hopper/norm_gate.py",
               root/"flash_qla/ops/gated_delta_rule/chunk/__init__.py"]
    results = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                   causal_conv1d=importlib.metadata.version("causal-conv1d"),
                   visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   scope="Synthetic whole layer, hidden2560 Hk16 Hv48 D128 conv4; BF16 with FP32 decay; "
                         "dt_max=.01, full-history optimized forward and default uncached GDR backward; "
                         "direct projection-gradient writes in both variants. Only output-gate forward differs. "
                         "No optimizer, distributed CP, state cache, or activation-stat logging.",
                   method="21 paired CUDA graph samples with alternating execution order; native allocator event history for eager memory checks",
                   sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for length in args.seqlens:
        torch.manual_seed(42)
        config=argparse.Namespace(seqlen=length,hidden_size=2560,key_heads=16,value_heads=48,
                                  conv_size=4,norm_eps=1e-6,dt_max=.01,partitions=8,
                                  conv_backend="dao",projection_mode="separate")
        layer=Layer(config)
        print("LAYER",length,flush=True)
        results[str(length)]=compare(layer,args.output.parent if args.trace else None,
                                    modes=("eager_gate","fused_gate"))
        args.output.write_text(json.dumps(results,indent=2)+"\n")
        del layer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
