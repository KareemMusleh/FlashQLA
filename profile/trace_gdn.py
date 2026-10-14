"""Capture warmed FlashQLA GDN forward/backward CPU and CUDA Chrome events."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import tilelang
import flash_qla
from flash_qla.ops.gated_delta_rule.chunk import CHUNK_SIZE, _auto_intra_cp_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seqlen', type=int, default=16384)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())['text_config']
    hk, hv = config['linear_num_key_heads'], config['linear_num_value_heads']
    dk, dv = config['linear_key_head_dim'], config['linear_value_head_dim']
    torch.manual_seed(42)
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    shape = (1, args.seqlen)
    q, k = [torch.randn(*shape, hk, dk, device='cuda', dtype=torch.bfloat16,
                        requires_grad=True) for _ in range(2)]
    v = torch.randn(*shape, hv, dv, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    g = (torch.nn.functional.logsigmoid(torch.randn(*shape, hv, device='cuda')) / 16).requires_grad_()
    beta = torch.randn(*shape, hv, device='cuda').sigmoid().requires_grad_()
    do = torch.randn_like(v)
    inputs = (q, k, v, g, beta)
    cp = _auto_intra_cp_context(k, v, None, CHUNK_SIZE, True, True)

    def step():
        with torch.profiler.record_function('FlashQLA/GDN/forward'):
            output, _ = flash_qla.chunk_gated_delta_rule(
                *inputs, use_qk_l2norm_in_kernel=True, auto_cp=True,
                output_final_state=False, enable_fwd_cp_cache=True,
            )
        with torch.profiler.record_function('FlashQLA/GDN/backward'):
            grads = torch.autograd.grad(output, inputs, do)
        return output, grads

    print(f'Warmup: B=1 T={args.seqlen} Hk={hk} Hv={hv} K={dk} V={dv}; intra CP={cp.is_intra}', flush=True)
    for _ in range(args.warmup):
        output, grads = step()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all().item()
    assert all(torch.isfinite(grad).all().item() for grad in grads)
    del output, grads
    print('Capturing warmed CPU + CUDA events', flush=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(args.iterations):
            with torch.profiler.record_function(f'FlashQLA/GDN/iteration_{i}'):
                output, grads = step()
            torch.cuda.synchronize()
            del output, grads
    length_label = f'{args.seqlen // 1024}k' if args.seqlen % 1024 == 0 else str(args.seqlen)
    trace_path = args.output / f'qwen38_flash_next_gdn_{length_label}_trace.json'
    prof.export_chrome_trace(str(trace_path))
    events = json.loads(trace_path.read_text())['traceEvents']
    kernels = [e for e in events if e.get('cat') == 'kernel']
    assert kernels, 'Trace has no GPU kernels; check CUPTI availability'
    summary = {
        'scope': 'FlashQLA GDN operator forward and backward, including Q/K L2 normalization; synthetic inputs, no model weights or projections/convolution/output norm',
        'config_source': str(args.config.resolve()),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        'flash_qla_path': flash_qla.__file__,
        'torch': torch.__version__, 'tilelang': tilelang.__version__,
        'gpu': torch.cuda.get_device_name(), 'visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
        'batch': 1, 'seqlen': args.seqlen, 'key_heads': hk, 'value_heads': hv,
        'key_dim': dk, 'value_dim': dv, 'qkv_dtype': 'bfloat16', 'gate_beta_dtype': 'float32',
        'seed': 42, 'gate_distribution': 'logsigmoid(N(0,1))/16, all heads gated',
        'initial_state': None, 'output_final_state': False, 'chunk_size': CHUNK_SIZE,
        'auto_cp': True, 'intra_cp_selected': cp.is_intra,
        'cp_boundaries': cp.intra_cp_cu_seqlens.tolist() if cp.is_intra else None,
        'warmup': args.warmup, 'iterations': args.iterations,
        'finite_output_and_all_input_gradients': True,
        'cuda_kernel_events': len(kernels),
        'mean_profiled_kernel_sum_ms': sum(e['dur'] for e in kernels) / args.iterations / 1000,
        'trace': str(trace_path.resolve()),
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'kernel_times.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=40))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
